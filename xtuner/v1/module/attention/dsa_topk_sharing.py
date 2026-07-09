# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Callable, Protocol

import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils.activation_offload import OffloadManager, SwapTensor


class DSATopKSharingLayerProtocol(Protocol):
    layer_idx: int
    source_layer_idx: int
    training: bool
    dsa_topk_last_use: dict[int, int]
    dsa_topk_recompute_release: dict[int, int]

    def _is_skip_topk_layer(self) -> bool: ...


class TopKResidencyBase:
    reuse_source_topk_in_recompute = True

    def has_cache(self, seq_ctx: SequenceContext, source_layer_idx: int) -> bool:
        return source_layer_idx in seq_ctx.dsa_topk_indices

    def store_gpu(self, seq_ctx: SequenceContext, source_layer_idx: int, topk_indices: torch.Tensor) -> None:
        seq_ctx.dsa_topk_indices[source_layer_idx] = topk_indices

    def read(self, seq_ctx: SequenceContext, source_layer_idx: int) -> torch.Tensor:
        return seq_ctx.dsa_topk_indices[source_layer_idx]

    def after_original_forward_last_use(self, seq_ctx: SequenceContext, source_layer_idx: int) -> None:
        return

    def after_recompute_release(self, seq_ctx: SequenceContext, source_layer_idx: int) -> None:
        seq_ctx.dsa_topk_indices.pop(source_layer_idx, None)

    def _offload_key(self, seq_ctx: SequenceContext, source_layer_idx: int) -> str:
        return f"dsa_topk_{seq_ctx.dsa_topk_context_id}_{source_layer_idx}"


class GpuTopKResidency(TopKResidencyBase):
    pass


class ActivationOffloadedTopKResidency(TopKResidencyBase):
    def __init__(self) -> None:
        self._streams: dict[int, torch.cuda.Stream] = {}

    def has_cache(self, seq_ctx: SequenceContext, source_layer_idx: int) -> bool:
        return source_layer_idx in seq_ctx.dsa_topk_indices or source_layer_idx in seq_ctx.dsa_topk_offloaded

    def read(self, seq_ctx: SequenceContext, source_layer_idx: int) -> torch.Tensor:
        if source_layer_idx in seq_ctx.dsa_topk_indices:
            return seq_ctx.dsa_topk_indices[source_layer_idx]

        key = seq_ctx.dsa_topk_offloaded[source_layer_idx]
        swap_tensor = OffloadManager().get(key)
        stream = self._stream_for_device(swap_tensor.tensor.device)
        working_stream = torch.cuda.current_stream(swap_tensor.tensor.device)

        # DSA top-k cache is not captured by saved_tensors_hooks, so this mirrors
        # activation offload's explicit H2D choreography for manual cache state.
        stream.wait_stream(working_stream)
        with torch.cuda.stream(stream):
            swap_tensor.launch_h2d(stream, True, stream)
        working_stream.wait_stream(stream)

        seq_ctx.dsa_topk_indices[source_layer_idx] = swap_tensor.tensor
        return swap_tensor.tensor

    def after_original_forward_last_use(self, seq_ctx: SequenceContext, source_layer_idx: int) -> None:
        topk_indices = seq_ctx.dsa_topk_indices.pop(source_layer_idx)
        if not topk_indices.is_cuda:
            seq_ctx.dsa_topk_indices[source_layer_idx] = topk_indices
            return

        key = self._offload_key(seq_ctx, source_layer_idx)
        cpu_buffer = OffloadManager().get_or_create_pin_memory(key, topk_indices.shape, topk_indices.dtype)
        swap_tensor = SwapTensor(topk_indices, key, tensor_cpu=cpu_buffer)
        stream = self._stream_for_device(topk_indices.device)
        stream.wait_stream(torch.cuda.current_stream(topk_indices.device))
        swap_tensor.launch_d2h(stream)
        swap_tensor.wait_d2h_finished(stream, True)
        OffloadManager().put(key, swap_tensor)
        seq_ctx.dsa_topk_offloaded[source_layer_idx] = key

    def after_recompute_release(self, seq_ctx: SequenceContext, source_layer_idx: int) -> None:
        super().after_recompute_release(seq_ctx, source_layer_idx)
        key = seq_ctx.dsa_topk_offloaded.pop(source_layer_idx, None)
        if key is None:
            return

        stream = self._stream_for_current_device()
        OffloadManager().del_may_npu_tensor(key, stream)
        if OffloadManager().exist(key):
            OffloadManager().clear(key)

    def _stream_for_current_device(self) -> torch.cuda.Stream:
        return self._stream_for_device(torch.device("cuda", torch.cuda.current_device()))

    def _stream_for_device(self, device: torch.device) -> torch.cuda.Stream:
        device_idx = torch.cuda.current_device() if device.index is None else device.index
        if device_idx not in self._streams:
            self._streams[device_idx] = torch.cuda.Stream(device=device_idx)
        return self._streams[device_idx]


class CrossLayerTopKSharingRuntime:
    def __init__(self) -> None:
        self._gpu_residency = GpuTopKResidency()
        self._offloaded_residency = ActivationOffloadedTopKResidency()

    def get_or_compute(
        self,
        *,
        layer: DSATopKSharingLayerProtocol,
        seq_ctx: SequenceContext,
        compute_source_topk: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        residency = self._residency()
        source_layer_idx = layer.source_layer_idx

        if layer._is_skip_topk_layer():
            self._assert_source_present(layer, seq_ctx, residency)
            return residency.read(seq_ctx, source_layer_idx)

        if (
            self._is_checkpoint_recompute(seq_ctx)
            and layer.layer_idx not in seq_ctx.dsa_topk_released_sources
            and residency.has_cache(seq_ctx, source_layer_idx)
        ):
            return residency.read(seq_ctx, source_layer_idx)

        topk_indices = compute_source_topk()
        if layer.layer_idx not in seq_ctx.dsa_topk_released_sources:
            residency.store_gpu(seq_ctx, layer.layer_idx, topk_indices)
        return topk_indices

    def after_sparse_mla_use(self, *, layer: DSATopKSharingLayerProtocol, seq_ctx: SequenceContext) -> None:
        residency = self._residency()
        source_layer_idx = layer.source_layer_idx
        if self._is_checkpoint_original_forward(layer):
            if layer.dsa_topk_last_use.get(source_layer_idx) == layer.layer_idx:
                # Reentrant checkpoint original forward runs under no_grad, so
                # SparseMLA has no autograd ctx. Keep/offload source top-k for
                # backward recompute, then release after source replay consumes it.
                seq_ctx.dsa_topk_checkpoint_active = True
                residency.after_original_forward_last_use(seq_ctx, source_layer_idx)
            return

        if not self._is_checkpoint_recompute(seq_ctx):
            return

        if layer.dsa_topk_recompute_release.get(source_layer_idx) != layer.layer_idx:
            return

        residency.after_recompute_release(seq_ctx, source_layer_idx)
        seq_ctx.dsa_topk_released_sources.add(source_layer_idx)

    def _residency(self) -> TopKResidencyBase:
        if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1 and torch.cuda.is_available():
            return self._offloaded_residency
        return self._gpu_residency

    def _is_checkpoint_original_forward(self, layer: DSATopKSharingLayerProtocol) -> bool:
        return layer.training and not torch.is_grad_enabled()

    def _is_checkpoint_recompute(self, seq_ctx: SequenceContext) -> bool:
        return seq_ctx.dsa_topk_checkpoint_active and torch.is_grad_enabled()

    def _assert_source_present(
        self,
        layer: DSATopKSharingLayerProtocol,
        seq_ctx: SequenceContext,
        residency: TopKResidencyBase,
    ) -> None:
        if residency.has_cache(seq_ctx, layer.source_layer_idx):
            return
        raise AssertionError(
            "DSA index-share: skip layer "
            f"{layer.layer_idx} needs source layer {layer.source_layer_idx} top-k, "
            "but it is not present in this microbatch SequenceContext. "
            "Cross-pipeline top-k sharing is not supported."
        )


_DSA_TOPK_SHARING_RUNTIME = CrossLayerTopKSharingRuntime()


def get_dsa_topk_sharing_runtime() -> CrossLayerTopKSharingRuntime:
    return _DSA_TOPK_SHARING_RUNTIME

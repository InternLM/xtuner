# Copyright (c) OpenMMLab. All rights reserved.
import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils.activation_offload import OffloadManager, SwapTensor


class DSATopKSharingLayerProtocol(Protocol):
    layer_idx: int
    source_layer_idx: int
    training: bool
    indexer_types: list[str] | None
    index_skip_topk_offset: int
    index_topk_freq: int
    dsa_topk_last_use: dict[int, int]
    dsa_topk_recompute_release: dict[int, int]

    def _is_skip_topk_layer(self) -> bool: ...


@dataclass(frozen=True)
class DSATopKReleasePlan:
    forward_last_use: dict[int, int]
    recompute_release: dict[int, int]


def build_dsa_topk_release_plan(
    *,
    num_main_layers: int,
    num_mtp_layers: int,
    indexer_types: list[str] | None,
    index_skip_topk_offset: int,
    index_topk_freq: int,
) -> DSATopKReleasePlan:
    consumers: dict[int, list[int]] = {}
    for layer_idx in range(num_main_layers + num_mtp_layers):
        source_layer_idx = _source_layer_for_plan(
            layer_idx=layer_idx,
            indexer_types=indexer_types,
            index_skip_topk_offset=index_skip_topk_offset,
            index_topk_freq=index_topk_freq,
        )
        consumers.setdefault(source_layer_idx, []).append(layer_idx)

    return DSATopKReleasePlan(
        forward_last_use={
            source_layer_idx: max(consumer_layers) for source_layer_idx, consumer_layers in consumers.items()
        },
        recompute_release={
            source_layer_idx: min(consumer_layers) for source_layer_idx, consumer_layers in consumers.items()
        },
    )


def _source_layer_for_plan(
    *,
    layer_idx: int,
    indexer_types: list[str] | None,
    index_skip_topk_offset: int,
    index_topk_freq: int,
) -> int:
    if indexer_types is not None:
        if layer_idx < len(indexer_types) and indexer_types[layer_idx] == "full":
            return layer_idx
        start_idx = min(layer_idx, len(indexer_types) - 1)
        for source_layer_idx in range(start_idx, -1, -1):
            if indexer_types[source_layer_idx] == "full":
                return source_layer_idx
        raise ValueError(f"DSA layer {layer_idx} has no preceding full indexer layer.")

    if index_topk_freq <= 1:
        return layer_idx

    source_layer_idx = layer_idx
    while _is_skip_topk_layer_for_plan(source_layer_idx, index_skip_topk_offset, index_topk_freq):
        source_layer_idx -= 1
    return source_layer_idx


def _is_skip_topk_layer_for_plan(layer_idx: int, skip_topk_offset: int, topk_freq: int) -> bool:
    layer_number = layer_idx + 1
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


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
        return self._read_offloaded(seq_ctx, source_layer_idx)

    # Pinned CPU buffers and stream-side effects must stay outside Inductor graphs.
    @torch.compiler.disable
    def _read_offloaded(self, seq_ctx: SequenceContext, source_layer_idx: int) -> torch.Tensor:
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

    # Pinned CPU buffers and stream-side effects must stay outside Inductor graphs.
    @torch.compiler.disable
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

    # Pinned CPU buffers and stream-side effects must stay outside Inductor graphs.
    @torch.compiler.disable
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
            if source_layer_idx in seq_ctx.dsa_topk_indices:
                return seq_ctx.dsa_topk_indices[source_layer_idx]
            return residency.read(seq_ctx, source_layer_idx)

        if (
            self._is_checkpoint_recompute(seq_ctx)
            and layer.layer_idx not in seq_ctx.dsa_topk_released_sources
            and residency.has_cache(seq_ctx, source_layer_idx)
        ):
            if source_layer_idx in seq_ctx.dsa_topk_indices:
                return seq_ctx.dsa_topk_indices[source_layer_idx]
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
                if self._should_defer_transfer():
                    seq_ctx.dsa_topk_pending_offloads.add(source_layer_idx)
                else:
                    residency.after_original_forward_last_use(seq_ctx, source_layer_idx)
            return

        if not self._is_checkpoint_recompute(seq_ctx):
            return

        if layer.dsa_topk_recompute_release.get(source_layer_idx) != layer.layer_idx:
            return

        if self._should_defer_transfer():
            seq_ctx.dsa_topk_pending_releases.add(source_layer_idx)
        else:
            residency.after_recompute_release(seq_ctx, source_layer_idx)
            seq_ctx.dsa_topk_released_sources.add(source_layer_idx)

    def before_layer_forward(self, *, layer: DSATopKSharingLayerProtocol, seq_ctx: SequenceContext) -> None:
        if not isinstance(self._residency(), ActivationOffloadedTopKResidency):
            return
        source_layer_idx = layer.source_layer_idx
        if source_layer_idx not in seq_ctx.dsa_topk_offloaded:
            return
        self._offloaded_residency.read(seq_ctx, source_layer_idx)

    def flush_pending(self, seq_ctx: SequenceContext) -> None:
        if not isinstance(self._residency(), ActivationOffloadedTopKResidency):
            seq_ctx.dsa_topk_pending_offloads.clear()
            seq_ctx.dsa_topk_pending_releases.clear()
            return

        for source_layer_idx in tuple(seq_ctx.dsa_topk_pending_offloads):
            self._offloaded_residency.after_original_forward_last_use(seq_ctx, source_layer_idx)
            seq_ctx.dsa_topk_pending_offloads.remove(source_layer_idx)

        for source_layer_idx in tuple(seq_ctx.dsa_topk_pending_releases):
            self._offloaded_residency.after_recompute_release(seq_ctx, source_layer_idx)
            seq_ctx.dsa_topk_released_sources.add(source_layer_idx)
            seq_ctx.dsa_topk_pending_releases.remove(source_layer_idx)

    def _residency(self) -> TopKResidencyBase:
        if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1 and torch.cuda.is_available():
            return self._offloaded_residency
        return self._gpu_residency

    def _should_defer_transfer(self) -> bool:
        return torch.compiler.is_compiling() and isinstance(self._residency(), ActivationOffloadedTopKResidency)

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


@torch.compiler.disable
def before_dsa_topk_decoder_forward(attention: object, seq_ctx: SequenceContext | list[SequenceContext]) -> None:
    if not hasattr(attention, "dsa_topk_last_use"):
        return

    runtime = get_dsa_topk_sharing_runtime()
    for ctx in seq_ctx if isinstance(seq_ctx, list) else [seq_ctx]:
        runtime.before_layer_forward(layer=cast(DSATopKSharingLayerProtocol, attention), seq_ctx=ctx)


@torch.compiler.disable
def flush_dsa_topk_decoder_pending(seq_ctx: SequenceContext | list[SequenceContext]) -> None:
    runtime = get_dsa_topk_sharing_runtime()
    for ctx in seq_ctx if isinstance(seq_ctx, list) else [seq_ctx]:
        runtime.flush_pending(ctx)


def _get_seq_ctx_from_forward(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> SequenceContext | list[SequenceContext] | None:
    if "seq_ctx" in kwargs:
        return kwargs["seq_ctx"]
    if len(args) >= 3 and isinstance(args[2], SequenceContext | list):
        return args[2]
    return None


def _dsa_topk_decoder_lifecycle_pre_hook(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    seq_ctx = _get_seq_ctx_from_forward(args, kwargs)
    if seq_ctx is None:
        return
    before_dsa_topk_decoder_forward(getattr(module, "self_attn", None), seq_ctx)


def _dsa_topk_decoder_lifecycle_post_hook(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    _output: Any,
) -> None:
    seq_ctx = _get_seq_ctx_from_forward(args, kwargs)
    if seq_ctx is None:
        return
    flush_dsa_topk_decoder_pending(seq_ctx)


def register_dsa_topk_decoder_lifecycle_hooks(decoder_layer: torch.nn.Module) -> None:
    if getattr(decoder_layer, "_dsa_topk_decoder_lifecycle_hooks_registered", False):
        return
    self_attn = getattr(decoder_layer, "self_attn", None)
    if not hasattr(self_attn, "dsa_topk_last_use"):
        return

    # Hooks run around the decoder module call, so reentrant checkpoint replay
    # triggers them again while the compiled decoder/attention forward stays
    # free of offload stream and pinned-memory side effects.
    decoder_layer.register_forward_pre_hook(_dsa_topk_decoder_lifecycle_pre_hook, with_kwargs=True)
    decoder_layer.register_forward_hook(_dsa_topk_decoder_lifecycle_post_hook, with_kwargs=True)
    object.__setattr__(decoder_layer, "_dsa_topk_decoder_lifecycle_hooks_registered", True)

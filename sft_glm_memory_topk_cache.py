"""Pseudo code for GLM-5.2 Cross-Layer Top-K Sharing.

This is design pseudo code, not production code. It shows how the latest
compile-aware GLM-5.2 path, the stashed top-k cache lifetime optimization, and
the proposed activation-offloaded top-k cache fit together behind one deeper
Module.
"""

from __future__ import annotations

import itertools
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Release-map construction: this mirrors the stashed correctness baseline.
# ---------------------------------------------------------------------------


def is_dsa_skip_topk_layer(layer_idx: int, skip_topk_offset: int, topk_freq: int) -> bool:
    layer_number = layer_idx + 1
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def dsa_source_compute_layer(layer_idx: int, skip_topk_offset: int, topk_freq: int) -> int:
    source = layer_idx
    while is_dsa_skip_topk_layer(source, skip_topk_offset, topk_freq):
        source -= 1
    return source


@dataclass(frozen=True)
class DSATopKReleasePlan:
    """Release plan for one logical GLM-5.2 layer stack.

    `recompute_release_without_source_reuse` is an aggressive GPU-memory policy:
    shared consumers reuse the original cache, then the cache is released before
    the source layer recomputes its own top-k.

    `recompute_release_with_source_reuse` is the default policy when a cache
    already exists for shared consumers: source recompute also reads that cache,
    so the source itself is the last recompute consumer.
    """

    forward_last_use: dict[int, int]
    recompute_release_without_source_reuse: dict[int, int]
    recompute_release_with_source_reuse: dict[int, int]
    first_recompute_use: dict[int, int]


def build_dsa_topk_release_plan(
    *,
    num_main_layers: int,
    num_mtp_layers: int,
    indexer_types: list[str] | None,
    index_skip_topk_offset: int,
    index_topk_freq: int,
) -> DSATopKReleasePlan:
    """Return source -> release layer maps for forward and checkpoint recompute."""

    consumers: dict[int, list[int]] = {}
    for layer_idx in range(num_main_layers + num_mtp_layers):
        if indexer_types is not None:
            if layer_idx < len(indexer_types) and indexer_types[layer_idx] == "full":
                source_layer_idx = layer_idx
            else:
                start_idx = min(layer_idx, len(indexer_types) - 1)
                for source_layer_idx in range(start_idx, -1, -1):
                    if indexer_types[source_layer_idx] == "full":
                        break
                else:
                    raise ValueError(f"DSA layer {layer_idx} has no preceding full indexer layer.")
        elif index_topk_freq <= 1 or not is_dsa_skip_topk_layer(
            layer_idx, index_skip_topk_offset, index_topk_freq
        ):
            source_layer_idx = layer_idx
        else:
            source_layer_idx = dsa_source_compute_layer(layer_idx, index_skip_topk_offset, index_topk_freq)

        consumers.setdefault(source_layer_idx, []).append(layer_idx)

    forward_last_use: dict[int, int] = {}
    recompute_release_without_source_reuse: dict[int, int] = {}
    recompute_release_with_source_reuse: dict[int, int] = {}
    first_recompute_use: dict[int, int] = {}
    for source_layer_idx, consumer_layers in consumers.items():
        forward_last_use[source_layer_idx] = max(consumer_layers)
        first_recompute_use[source_layer_idx] = max(consumer_layers)
        shared_consumers = [layer_idx for layer_idx in consumer_layers if layer_idx != source_layer_idx]
        # Checkpoint recompute runs in reverse layer order. If source 2 is used
        # by layers 3 and 4, layer 3 is the final shared recompute consumer.
        # The aggressive policy releases there and lets source 2 recompute its
        # own top-k. The default policy keeps the cache through source 2, so
        # release including source is min([2, 3, 4]) == 2.
        recompute_release_without_source_reuse[source_layer_idx] = (
            min(shared_consumers) if shared_consumers else source_layer_idx
        )
        recompute_release_with_source_reuse[source_layer_idx] = min(consumer_layers)

    return DSATopKReleasePlan(
        forward_last_use=forward_last_use,
        recompute_release_without_source_reuse=recompute_release_without_source_reuse,
        recompute_release_with_source_reuse=recompute_release_with_source_reuse,
        first_recompute_use=first_recompute_use,
    )


# ---------------------------------------------------------------------------
# SequenceContext state: current code has dsa_topk_indices as an always-present
# dict. The extra fields below are the proposed extension.
# ---------------------------------------------------------------------------


_SEQ_CTX_ID = itertools.count()


@dataclass
class SequenceContextPseudo:
    input_ids: torch.Tensor | None = None
    inputs_embeds: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    dsa_topk_indices: dict[int, torch.Tensor] = field(default_factory=dict)
    dsa_topk_offloaded: dict[int, str] = field(default_factory=dict)
    dsa_topk_released_sources: set[int] = field(default_factory=set)
    dsa_topk_checkpoint_active: bool = False
    dsa_topk_context_id: int = field(default_factory=lambda: next(_SEQ_CTX_ID))

    @staticmethod
    def cat(seq_ctx_list: list["SequenceContextPseudo"]) -> "SequenceContextPseudo":
        """Current MoE dense-prefix path builds one concat context."""

        return SequenceContextPseudo()

    def copy(self, **overrides: Any) -> "SequenceContextPseudo":
        """Copies share released_sources because it is microbatch lifetime state."""

        return SequenceContextPseudo(
            input_ids=overrides.get("input_ids", self.input_ids),
            inputs_embeds=overrides.get("inputs_embeds", self.inputs_embeds),
            position_ids=overrides.get("position_ids", self.position_ids),
            dsa_topk_indices=overrides.get("dsa_topk_indices", self.dsa_topk_indices),
            dsa_topk_offloaded=overrides.get("dsa_topk_offloaded", self.dsa_topk_offloaded),
            dsa_topk_released_sources=overrides.get(
                "dsa_topk_released_sources", self.dsa_topk_released_sources
            ),
            dsa_topk_checkpoint_active=overrides.get(
                "dsa_topk_checkpoint_active", self.dsa_topk_checkpoint_active
            ),
            dsa_topk_context_id=overrides.get("dsa_topk_context_id", self.dsa_topk_context_id),
        )

    def split_dsa_topk_indices_to(self, sequence_context_list: list["SequenceContextPseudo"]) -> None:
        """Current latest-code behavior: split dense-prefix top-k back to microbatches."""

        if not self.dsa_topk_indices:
            return

        lengths = [ctx.seq_len for ctx in sequence_context_list]
        for source_layer_idx, topk_indices in self.dsa_topk_indices.items():
            assert sum(lengths) == topk_indices.shape[0]
            for seq_ctx, single_topk_indices in zip(sequence_context_list, topk_indices.split(lengths, dim=0)):
                seq_ctx.dsa_topk_indices[source_layer_idx] = single_topk_indices

    @property
    def seq_len(self) -> int:
        if self.input_ids is not None:
            return int(self.input_ids.shape[1])
        if self.inputs_embeds is not None:
            return int(self.inputs_embeds.shape[1])
        assert self.position_ids is not None
        return int(self.position_ids.shape[-1])


# ---------------------------------------------------------------------------
# Residency Adapter seam. This is real because there are two adapters:
# GPU-only and activation-offloaded.
# ---------------------------------------------------------------------------


class TopKResidencyBase(ABC):
    """Base Adapter for where a DSA top-k cache resides.

    The base class owns the common lifecycle shape: source top-k is stored in
    ``SequenceContext``, recompute should reuse it by default, and release
    removes both the GPU tensor and any residency-specific handle. Subclasses
    only implement the storage medium differences.
    """

    @property
    def reuse_source_topk_in_recompute(self) -> bool:
        # Default policy: all compute layers, including source-only layers,
        # reuse their checkpoint-original top-k during recompute. The aggressive
        # memory policy can override this to trade indexer recompute for memory.
        return True

    def store_gpu(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int, topk: torch.Tensor) -> None:
        seq_ctx.dsa_topk_indices[source_layer_idx] = topk

    @abstractmethod
    def read(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> torch.Tensor: ...

    def prefetch_for_recompute_layer(
        self, seq_ctx: SequenceContextPseudo, source_layer_idx: int, stream: torch.cuda.Stream
    ) -> None:
        return

    @abstractmethod
    def after_original_forward_last_use(
        self, seq_ctx: SequenceContextPseudo, source_layer_idx: int, stream: torch.cuda.Stream
    ) -> None: ...

    def after_recompute_release(
        self, seq_ctx: SequenceContextPseudo, source_layer_idx: int, stream: torch.cuda.Stream
    ) -> None:
        seq_ctx.dsa_topk_indices.pop(source_layer_idx, None)
        self._release_residency_handle(seq_ctx, source_layer_idx, stream)

    def _offload_key(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> str:
        return f"dsa_topk_{seq_ctx.dsa_topk_context_id}_{source_layer_idx}"

    def _release_residency_handle(
        self, seq_ctx: SequenceContextPseudo, source_layer_idx: int, stream: torch.cuda.Stream
    ) -> None:
        return


class GpuTopKResidency(TopKResidencyBase):
    """No activation offload: keep compute-layer caches on GPU until recompute."""

    def read(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> torch.Tensor:
        if source_layer_idx not in seq_ctx.dsa_topk_indices:
            raise AssertionError(f"missing GPU DSA top-k for source layer {source_layer_idx}")
        return seq_ctx.dsa_topk_indices[source_layer_idx]

    def after_original_forward_last_use(
        self, seq_ctx: SequenceContextPseudo, source_layer_idx: int, stream: torch.cuda.Stream
    ) -> None:
        # Reentrant checkpoint original forward runs under no_grad. Keep GPU
        # cache because backward recompute still needs to read it.
        return


class ActivationOffloadedTopKResidency(TopKResidencyBase):
    """Use OffloadManager/SwapTensor-style D2H/H2D for top-k cache.

    This is pseudo code. Production code should reuse the existing pinned memory
    cache and stream choreography from xtuner.v1.utils.activation_offload.
    """

    def read(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> torch.Tensor:
        if source_layer_idx in seq_ctx.dsa_topk_indices:
            return seq_ctx.dsa_topk_indices[source_layer_idx]

        if source_layer_idx not in seq_ctx.dsa_topk_offloaded:
            raise AssertionError(f"missing GPU/CPU DSA top-k for source layer {source_layer_idx}")

        key = seq_ctx.dsa_topk_offloaded[source_layer_idx]
        # Pseudo production path:
        #   handle = OffloadManager().get(key)
        #   handle.launch_h2d(h2d_stream, flag=True, working_stream=torch.cuda.current_stream())
        #   topk = handle.tensor
        topk = torch.empty(0, dtype=torch.int32, device="cuda")  # placeholder for handle.tensor
        seq_ctx.dsa_topk_indices[source_layer_idx] = topk
        return topk

    def prefetch_for_recompute_layer(
        self, seq_ctx: SequenceContextPseudo, source_layer_idx: int, stream: torch.cuda.Stream
    ) -> None:
        if source_layer_idx in seq_ctx.dsa_topk_indices:
            return
        key = seq_ctx.dsa_topk_offloaded.get(source_layer_idx)
        if key is None:
            return
        # Pseudo production path, modeled after OffloadManager.prefetch_get:
        #   handle = OffloadManager().get(key)
        #   stream.wait_stream(d2h_stream)
        #   handle.prefetch_launch_h2d(stream, flag=True)
        #   seq_ctx.dsa_topk_indices[source_layer_idx] = handle.tensor
        return

    def after_original_forward_last_use(
        self, seq_ctx: SequenceContextPseudo, source_layer_idx: int, stream: torch.cuda.Stream
    ) -> None:
        topk = seq_ctx.dsa_topk_indices.pop(source_layer_idx)
        key = self._offload_key(seq_ctx, source_layer_idx)
        assert topk.dtype == torch.int32
        # Pseudo production path:
        #   cpu = OffloadManager().get_or_create_pin_memory(key, topk.shape, topk.dtype)
        #   handle = SwapTensor(topk, key, tensor_cpu=cpu)
        #   stream.wait_stream(torch.cuda.current_stream())
        #   handle.launch_d2h(stream)
        #   OffloadManager().put(key, handle)
        # A later layer boundary can call wait_d2h_finished + resize_(0), just
        # like activation offload's delayed storage release. Launching D2H here
        # lets the copy overlap with the rest of this decoder layer and later
        # forward compute.
        seq_ctx.dsa_topk_offloaded[source_layer_idx] = key

    def _release_residency_handle(
        self, seq_ctx: SequenceContextPseudo, source_layer_idx: int, stream: torch.cuda.Stream
    ) -> None:
        key = seq_ctx.dsa_topk_offloaded.pop(source_layer_idx, None)
        if key is not None:
            # Pseudo production path:
            #   OffloadManager().del_may_npu_tensor(key, stream)
            #   if OffloadManager().exist(key): OffloadManager().clear(key)
            pass


def topk_residency_adapter_from_env() -> TopKResidencyBase:
    if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1:
        return ActivationOffloadedTopKResidency()
    return GpuTopKResidency()


# ---------------------------------------------------------------------------
# Deep Module: Cross-Layer Top-K Sharing Runtime.
# ---------------------------------------------------------------------------


@dataclass
class DSALayerView:
    """Small view of DSAMultiLatentAttention needed by the runtime."""

    layer_idx: int
    source_layer_idx: int
    is_skip_topk_layer: bool
    training: bool
    dsa_topk_last_use: dict[int, int]
    dsa_topk_recompute_release_without_source_reuse: dict[int, int]
    dsa_topk_recompute_release_with_source_reuse: dict[int, int]
    dsa_topk_first_recompute_use: dict[int, int]
    offload_stream: torch.cuda.Stream


class CrossLayerTopKSharingRuntime:
    """Deep Module hiding Cross-Layer Top-K Sharing lifecycle details."""

    def __init__(self, residency: TopKResidencyBase | None = None) -> None:
        self.residency = residency if residency is not None else topk_residency_adapter_from_env()

    def get_or_compute(
        self,
        *,
        layer: DSALayerView,
        seq_ctx: SequenceContextPseudo,
        compute_source_topk: Any,
    ) -> torch.Tensor:
        self._prefetch_if_needed_for_current_layer(layer=layer, seq_ctx=seq_ctx)

        if layer.is_skip_topk_layer:
            return self.residency.read(seq_ctx, layer.source_layer_idx)

        if self._should_reuse_source_topk_in_recompute(layer=layer, seq_ctx=seq_ctx):
            return self.residency.read(seq_ctx, layer.source_layer_idx)

        topk = compute_source_topk()
        assert topk.dtype == torch.int32
        # In the default policy, checkpoint recompute should have read the
        # checkpoint-original cache above. This compute path exists for the
        # aggressive memory policy, where a cache may be released before source
        # recompute and the result is only for the current SparseMLA call.
        should_store_for_future_consumers = not self._is_checkpoint_recompute(seq_ctx)
        if should_store_for_future_consumers and layer.layer_idx not in seq_ctx.dsa_topk_released_sources:
            self.residency.store_gpu(seq_ctx, layer.layer_idx, topk)
        return topk

    def after_sparse_mla_use(self, *, layer: DSALayerView, seq_ctx: SequenceContextPseudo) -> None:
        """Release/offload only after SparseMLA has safely consumed top-k."""

        source = layer.source_layer_idx

        if self._is_checkpoint_original_forward(layer):
            if layer.dsa_topk_last_use.get(source) == layer.layer_idx:
                # Marker means: this microbatch entered checkpoint mode. It
                # also switches later grad-enabled forward into recompute
                # release semantics.
                seq_ctx.dsa_topk_checkpoint_active = True
                # Source-only and shared-source compute layers follow the same
                # default lifecycle. Keeping/offloading source-only top-k costs
                # little in GLM-5.2 and avoids expensive indexer recompute.
                self.residency.after_original_forward_last_use(seq_ctx, source, layer.offload_stream)
            return

        release_layers = layer.dsa_topk_last_use
        if self._is_checkpoint_recompute(seq_ctx):
            release_layers = self._recompute_release_layers(layer)

        if release_layers.get(source) != layer.layer_idx:
            self._prefetch_for_next_recompute_layer(layer=layer, seq_ctx=seq_ctx)
            return

        self.residency.after_recompute_release(seq_ctx, source, layer.offload_stream)
        if self._is_checkpoint_recompute(seq_ctx):
            seq_ctx.dsa_topk_released_sources.add(source)
        self._prefetch_for_next_recompute_layer(layer=layer, seq_ctx=seq_ctx)

    def split_concat_context_to_microbatches(
        self, cat_seq_ctx: SequenceContextPseudo | None, seq_ctx_list: list[SequenceContextPseudo]
    ) -> None:
        if cat_seq_ctx is not None:
            cat_seq_ctx.split_dsa_topk_indices_to(seq_ctx_list)

    def _is_checkpoint_original_forward(self, layer: DSALayerView) -> bool:
        return layer.training and not torch.is_grad_enabled()

    def _is_checkpoint_recompute(self, seq_ctx: SequenceContextPseudo) -> bool:
        return seq_ctx.dsa_topk_checkpoint_active and torch.is_grad_enabled()

    def _recompute_release_layers(self, layer: DSALayerView) -> dict[int, int]:
        if self.residency.reuse_source_topk_in_recompute:
            return layer.dsa_topk_recompute_release_with_source_reuse
        return layer.dsa_topk_recompute_release_without_source_reuse

    def _should_reuse_source_topk_in_recompute(
        self, *, layer: DSALayerView, seq_ctx: SequenceContextPseudo
    ) -> bool:
        if layer.is_skip_topk_layer:
            return False
        if not self._is_checkpoint_recompute(seq_ctx):
            return False
        if not self.residency.reuse_source_topk_in_recompute:
            return False
        if layer.layer_idx in seq_ctx.dsa_topk_released_sources:
            return False
        return (
            layer.source_layer_idx in seq_ctx.dsa_topk_indices
            or layer.source_layer_idx in seq_ctx.dsa_topk_offloaded
        )

    def _prefetch_if_needed_for_current_layer(
        self, *, layer: DSALayerView, seq_ctx: SequenceContextPseudo
    ) -> None:
        if not self._is_checkpoint_recompute(seq_ctx):
            return
        source = layer.source_layer_idx
        if layer.dsa_topk_first_recompute_use.get(source) == layer.layer_idx:
            self.residency.prefetch_for_recompute_layer(seq_ctx, source, layer.offload_stream)

    def _prefetch_for_next_recompute_layer(
        self, *, layer: DSALayerView, seq_ctx: SequenceContextPseudo
    ) -> None:
        if not self._is_checkpoint_recompute(seq_ctx):
            return
        next_layer_idx = layer.layer_idx - 1
        for source, first_recompute_layer in layer.dsa_topk_first_recompute_use.items():
            if first_recompute_layer == next_layer_idx:
                self.residency.prefetch_for_recompute_layer(seq_ctx, source, layer.offload_stream)


# ---------------------------------------------------------------------------
# DSAMultiLatentAttention usage. Its forward remains fullgraph=False in GLM-5.2
# compile config, so Python state mutation is intentionally allowed here.
# ---------------------------------------------------------------------------


class DSAMultiLatentAttentionPseudo:
    def __init__(
        self,
        *,
        layer_idx: int,
        indexer_types: list[str] | None,
        dsa_topk_last_use: dict[int, int],
        dsa_topk_recompute_release_without_source_reuse: dict[int, int],
        dsa_topk_recompute_release_with_source_reuse: dict[int, int],
        dsa_topk_first_recompute_use: dict[int, int],
        topk_runtime: CrossLayerTopKSharingRuntime,
    ) -> None:
        self.layer_idx = layer_idx
        self.indexer_types = indexer_types
        self.dsa_topk_last_use = dsa_topk_last_use
        self.dsa_topk_recompute_release_without_source_reuse = (
            dsa_topk_recompute_release_without_source_reuse
        )
        self.dsa_topk_recompute_release_with_source_reuse = dsa_topk_recompute_release_with_source_reuse
        self.dsa_topk_first_recompute_use = dsa_topk_first_recompute_use
        self.topk_runtime = topk_runtime
        self.training = True
        self.offload_stream = torch.cuda.Stream()
        self.source_layer_idx = self._source_compute_layer() if self._is_skip_topk_layer() else self.layer_idx

    def _is_skip_topk_layer(self) -> bool:
        if self.indexer_types is None:
            return False
        if self.layer_idx >= len(self.indexer_types):
            return True
        return self.indexer_types[self.layer_idx] == "shared"

    def _source_compute_layer(self) -> int:
        assert self.indexer_types is not None
        for idx in range(min(self.layer_idx, len(self.indexer_types) - 1), -1, -1):
            if self.indexer_types[idx] == "full":
                return idx
        raise ValueError(f"DSA shared layer {self.layer_idx} has no preceding full indexer layer.")

    def _layer_view(self) -> DSALayerView:
        return DSALayerView(
            layer_idx=self.layer_idx,
            source_layer_idx=self.source_layer_idx,
            is_skip_topk_layer=self._is_skip_topk_layer(),
            training=self.training,
            dsa_topk_last_use=self.dsa_topk_last_use,
            dsa_topk_recompute_release_without_source_reuse=(
                self.dsa_topk_recompute_release_without_source_reuse
            ),
            dsa_topk_recompute_release_with_source_reuse=self.dsa_topk_recompute_release_with_source_reuse,
            dsa_topk_first_recompute_use=self.dsa_topk_first_recompute_use,
            offload_stream=self.offload_stream,
        )

    def _compute_source_topk(self, hidden_states: Any, q_resid: Any, position_embeddings: Any) -> torch.Tensor:
        # Real implementation calls DSAIndexer with hidden/q_resid detached.
        return torch.empty(0, dtype=torch.int32, device="cuda")

    def forward(self, hidden_states: Any, position_embeddings: Any, seq_ctx: SequenceContextPseudo) -> dict[str, Any]:
        q_resid = "q_a_layernorm(q_a_proj(hidden_states))"
        layer_view = self._layer_view()
        topk_indices = self.topk_runtime.get_or_compute(
            layer=layer_view,
            seq_ctx=seq_ctx,
            compute_source_topk=lambda: self._compute_source_topk(hidden_states, q_resid, position_embeddings),
        )
        sparse_mla_outputs = self.sparse_mla_func(
            q="query_states",
            kv="key_states",
            indices=topk_indices,
            scaling="softmax_scale",
            value_dim="kv_lora_rank",
        )
        self.topk_runtime.after_sparse_mla_use(layer=layer_view, seq_ctx=seq_ctx)
        return {"projected_output": "o_proj(raw_output)", **sparse_mla_outputs}

    def sparse_mla_func(self, q: Any, kv: Any, indices: torch.Tensor, scaling: Any, value_dim: Any) -> dict[str, Any]:
        # SparseMLA autograd ctx owns per-layer backward indices when grad is
        # enabled. That is separate from Cross-Layer Top-K Sharing state.
        assert indices.dtype == torch.int32
        return {"raw_output": "raw_output", "softmax_lse": "softmax_lse"}


# ---------------------------------------------------------------------------
# MoE / GLM-5.2 integration. This shows current module interfaces used by SFT
# clients and where compile/offload constraints sit.
# ---------------------------------------------------------------------------


@dataclass
class DSAMLAConfigPseudo:
    indexer_types: list[str]
    index_skip_topk_offset: int = 3
    index_topk_freq: int = 4
    dsa_topk_last_use: dict[int, int] = field(default_factory=dict)
    dsa_topk_recompute_release_without_source_reuse: dict[int, int] = field(default_factory=dict)
    dsa_topk_recompute_release_with_source_reuse: dict[int, int] = field(default_factory=dict)
    dsa_topk_first_recompute_use: dict[int, int] = field(default_factory=dict)


@dataclass
class MTPConfigPseudo:
    num_layers: int = 1
    share_weights: bool = False


@dataclass
class MoEConfigPseudo:
    num_hidden_layers: int
    first_k_dense_replace: int
    attention: DSAMLAConfigPseudo
    mtp_config: MTPConfigPseudo | None
    compile_cfg: dict[str, dict[str, bool]] | bool | None = None

    def build(self) -> "MoEPseudo":
        return MoEPseudo(self)


class Glm52CompileCfgPseudo:
    """Latest compile intent: keep stateful DSA path out of strict fullgraph."""

    NON_EP = {
        "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer.forward": {"fullgraph": False},
        "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._pre_moe_forward": {"fullgraph": False},
        "xtuner.v1.module.attention.dsa_mla.DSAMultiLatentAttention.forward": {"fullgraph": False},
        "xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": {"fullgraph": False},
        "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEBlock.forward": {"fullgraph": True},
        "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._shared_experts_forward": {
            "fullgraph": True
        },
        "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer._post_moe_forward": {
            "fullgraph": True
        },
    }


class MoEPseudo:
    def __init__(self, config: MoEConfigPseudo) -> None:
        self.config = config
        self.compile_cfg = config.compile_cfg if config.compile_cfg is not None else Glm52CompileCfgPseudo.NON_EP
        self.topk_runtime = CrossLayerTopKSharingRuntime()
        self._configure_dsa_topk_release_layers(config)
        self.layers = self._build_layers(config)
        self.mtp_block = self._build_mtp_block(config) if config.mtp_config is not None else None

    def _configure_dsa_topk_release_layers(self, config: MoEConfigPseudo) -> None:
        num_mtp_layers = 0
        if config.mtp_config is not None:
            num_mtp_layers = 1 if config.mtp_config.share_weights else config.mtp_config.num_layers

        release_plan = build_dsa_topk_release_plan(
            num_main_layers=config.num_hidden_layers,
            num_mtp_layers=num_mtp_layers,
            indexer_types=config.attention.indexer_types,
            index_skip_topk_offset=config.attention.index_skip_topk_offset,
            index_topk_freq=config.attention.index_topk_freq,
        )
        config.attention.dsa_topk_last_use = release_plan.forward_last_use
        config.attention.dsa_topk_recompute_release_without_source_reuse = (
            release_plan.recompute_release_without_source_reuse
        )
        config.attention.dsa_topk_recompute_release_with_source_reuse = (
            release_plan.recompute_release_with_source_reuse
        )
        config.attention.dsa_topk_first_recompute_use = release_plan.first_recompute_use

    def _build_layers(self, config: MoEConfigPseudo) -> list["DecoderLayerPseudo"]:
        return [
            DecoderLayerPseudo(
                layer_idx=layer_idx,
                dense_prefix=layer_idx < config.first_k_dense_replace,
                self_attn=DSAMultiLatentAttentionPseudo(
                    layer_idx=layer_idx,
                    indexer_types=config.attention.indexer_types,
                    dsa_topk_last_use=config.attention.dsa_topk_last_use,
                    dsa_topk_recompute_release_without_source_reuse=(
                        config.attention.dsa_topk_recompute_release_without_source_reuse
                    ),
                    dsa_topk_recompute_release_with_source_reuse=(
                        config.attention.dsa_topk_recompute_release_with_source_reuse
                    ),
                    dsa_topk_first_recompute_use=config.attention.dsa_topk_first_recompute_use,
                    topk_runtime=self.topk_runtime,
                ),
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

    def _build_mtp_block(self, config: MoEConfigPseudo) -> "MTPBlockPseudo":
        assert config.mtp_config is not None
        mtp_layer_idx = config.num_hidden_layers
        return MTPBlockPseudo(
            decoder_layer=DecoderLayerPseudo(
                layer_idx=mtp_layer_idx,
                dense_prefix=False,
                self_attn=DSAMultiLatentAttentionPseudo(
                    layer_idx=mtp_layer_idx,
                    indexer_types=config.attention.indexer_types,
                    dsa_topk_last_use=config.attention.dsa_topk_last_use,
                    dsa_topk_recompute_release_without_source_reuse=(
                        config.attention.dsa_topk_recompute_release_without_source_reuse
                    ),
                    dsa_topk_recompute_release_with_source_reuse=(
                        config.attention.dsa_topk_recompute_release_with_source_reuse
                    ),
                    dsa_topk_first_recompute_use=config.attention.dsa_topk_first_recompute_use,
                    topk_runtime=self.topk_runtime,
                ),
            )
        )

    def __call__(self, *, seq_ctx: SequenceContextPseudo, loss_ctx: Any | None) -> dict[str, Any]:
        return self._forward(seq_ctx=seq_ctx, loss_ctx=loss_ctx)

    def _forward(self, *, seq_ctx: SequenceContextPseudo, loss_ctx: Any | None) -> dict[str, Any]:
        hidden_states = "embed_tokens(input_ids)"
        position_embeddings = "rotary_emb(hidden_states, position_ids)"
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        if self.mtp_block is not None:
            hidden_states = self.mtp_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        return {"loss": "loss_ctx(hidden_states)", "hidden_states": hidden_states}

    def _forward_micro_batch(
        self, *, seq_ctx_list: list[SequenceContextPseudo], loss_ctx_list: list[Any] | None
    ) -> dict[str, Any]:
        cat_seq_ctx: SequenceContextPseudo | None = None
        cat_hidden_states = "cat(embed_tokens(input_ids_i))"
        hidden_states_list: list[Any] = []
        entered_sparse_layers = False

        for layer in self.layers:
            if layer.dense_prefix:
                if cat_seq_ctx is None:
                    cat_seq_ctx = SequenceContextPseudo.cat(seq_ctx_list)
                cat_hidden_states = layer(
                    cat_hidden_states,
                    position_embeddings="cat_position_embeddings",
                    seq_ctx=cat_seq_ctx,
                )
                continue

            if not entered_sparse_layers:
                self.topk_runtime.split_concat_context_to_microbatches(cat_seq_ctx, seq_ctx_list)
                hidden_states_list = ["chunk_0", "chunk_1"]
                entered_sparse_layers = True

            hidden_states_list = [
                layer(hidden_states, position_embeddings="position_embeddings_i", seq_ctx=seq_ctx)
                for hidden_states, seq_ctx in zip(hidden_states_list, seq_ctx_list)
            ]

        return {"loss": "microbatch_losses"}


class DecoderLayerPseudo:
    def __init__(self, *, layer_idx: int, dense_prefix: bool, self_attn: DSAMultiLatentAttentionPseudo) -> None:
        self.layer_idx = layer_idx
        self.dense_prefix = dense_prefix
        self.self_attn = self_attn

    def __call__(self, hidden_states: Any, *, position_embeddings: Any, seq_ctx: SequenceContextPseudo) -> Any:
        attn = self.self_attn.forward(hidden_states, position_embeddings, seq_ctx)
        return f"decoder_layer_{self.layer_idx}({attn['projected_output']})"


class MTPBlockPseudo:
    def __init__(self, *, decoder_layer: DecoderLayerPseudo) -> None:
        self.decoder_layer = decoder_layer

    def __call__(self, hidden_states: Any, *, position_embeddings: Any, seq_ctx: SequenceContextPseudo) -> Any:
        return self.decoder_layer(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)


# ---------------------------------------------------------------------------
# Client usage: current SFT/tests should still use existing model interfaces.
# ---------------------------------------------------------------------------


def client_usage_current_interfaces(
    *,
    get_model_config_from_hf: Any,
    SequenceContext: Any,
    model_path: str,
    input_ids: torch.Tensor,
    loss_ctx: Any,
) -> None:
    """The caller does not touch Cross-Layer Top-K Sharing internals."""

    os.environ["XTUNER_ACTIVATION_OFFLOAD"] = "1"

    config = get_model_config_from_hf(model_path)
    # The real Glm52MoE.default_compile_cfg keeps DSA mutation path
    # fullgraph=False while pure MoE tensor paths stay fullgraph=True.
    config.compile_cfg = None
    model = config.build()

    seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))
    outputs = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
    outputs["loss"].backward()


def tests_should_cover() -> list[str]:
    return [
        "int32 DSA top-k from torch and tilelang indexers",
        "forward last-use releases GPU top-k in non-checkpoint path",
        "checkpoint original forward keeps or offloads top-k until recompute",
        "GPU-only source-only and shared-source recompute read existing cache and release at source",
        "activation-offloaded source-only and shared-source recompute read cached/offloaded top-k",
        "activation-offloaded source-only original forward creates and releases a CPU handle",
        "aggressive memory policy can release before source and recompute top-k",
        "activation-offloaded checkpoint recompute releases at including-source release layer",
        "MTP participates in logical layer release maps",
        "dense-prefix concat SequenceContext splits top-k into microbatch contexts",
        "compile cfg keeps DSAMultiLatentAttention.forward fullgraph=False",
        "activation offload launches D2H only after SparseMLA consumes top-k",
        "activation offload H2D prefetch path clears GPU cache and CPU handle",
    ]

"""Pseudo code for GLM-5.2 Cross-Layer Top-K Sharing.

This file mirrors the current production shape after the DSA top-k lifecycle
refactor. It is design pseudo code, not executable production code.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch


# ---------------------------------------------------------------------------
# Release-map construction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DSATopKReleasePlan:
    """Lifecycle maps for one logical GLM-5.2 stack.

    ``recompute_release`` includes the source layer. During checkpoint
    recompute, source layers reuse the top-k cache kept for shared consumers
    instead of recomputing indexer output.
    """

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
        source_layer_idx = source_layer_for_plan(
            layer_idx=layer_idx,
            indexer_types=indexer_types,
            index_skip_topk_offset=index_skip_topk_offset,
            index_topk_freq=index_topk_freq,
        )
        consumers.setdefault(source_layer_idx, []).append(layer_idx)

    return DSATopKReleasePlan(
        forward_last_use={source: max(layers) for source, layers in consumers.items()},
        recompute_release={source: min(layers) for source, layers in consumers.items()},
    )


def source_layer_for_plan(
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
    while is_skip_topk_layer_for_plan(source_layer_idx, index_skip_topk_offset, index_topk_freq):
        source_layer_idx -= 1
    return source_layer_idx


def is_skip_topk_layer_for_plan(layer_idx: int, skip_topk_offset: int, topk_freq: int) -> bool:
    layer_number = layer_idx + 1
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


# ---------------------------------------------------------------------------
# SequenceContext top-k state.
# ---------------------------------------------------------------------------


_DSA_TOPK_CONTEXT_IDS = itertools.count()


@dataclass
class DSATopKCacheStatePseudo:
    """Mutable Cross-Layer Top-K Sharing state for one microbatch."""

    # Format: {source_layer_idx: [seq_len, kv_group, topk]}.
    # Invalid/padded sparse slots are -1.
    indices: dict[int, torch.Tensor] = field(default_factory=dict)
    offloaded: dict[int, str] = field(default_factory=dict)
    released_sources: set[int] = field(default_factory=set)
    pending_offloads: set[int] = field(default_factory=set)
    pending_releases: set[int] = field(default_factory=set)
    checkpoint_active: bool = False
    context_id: int = field(default_factory=lambda: next(_DSA_TOPK_CONTEXT_IDS))


@dataclass
class SequenceContextPseudo:
    input_ids: torch.Tensor | None = None
    inputs_embeds: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    dsa_topk_cache: DSATopKCacheStatePseudo = field(default_factory=DSATopKCacheStatePseudo)

    # Legacy property compatibility. New code should prefer ``dsa_topk_cache``.
    @property
    def dsa_topk_indices(self) -> dict[int, torch.Tensor]:
        return self.dsa_topk_cache.indices

    @dsa_topk_indices.setter
    def dsa_topk_indices(self, value: dict[int, torch.Tensor]) -> None:
        self.dsa_topk_cache.indices = value

    @property
    def dsa_topk_offloaded(self) -> dict[int, str]:
        return self.dsa_topk_cache.offloaded

    @property
    def dsa_topk_released_sources(self) -> set[int]:
        return self.dsa_topk_cache.released_sources

    @property
    def dsa_topk_context_id(self) -> int:
        return self.dsa_topk_cache.context_id

    @staticmethod
    def cat(seq_ctx_list: list["SequenceContextPseudo"]) -> "SequenceContextPseudo":
        return SequenceContextPseudo()

    def copy(self, **overrides: Any) -> "SequenceContextPseudo":
        dsa_topk_cache = overrides.get("dsa_topk_cache", self.dsa_topk_cache)
        dsa_override_keys = {
            "dsa_topk_indices",
            "dsa_topk_offloaded",
            "dsa_topk_released_sources",
            "dsa_topk_pending_offloads",
            "dsa_topk_pending_releases",
            "dsa_topk_checkpoint_active",
            "dsa_topk_context_id",
        }
        if "dsa_topk_cache" not in overrides and dsa_override_keys.intersection(overrides):
            dsa_topk_cache = DSATopKCacheStatePseudo(
                indices=overrides.get("dsa_topk_indices", self.dsa_topk_cache.indices),
                offloaded=overrides.get("dsa_topk_offloaded", self.dsa_topk_cache.offloaded),
                released_sources=overrides.get(
                    "dsa_topk_released_sources", self.dsa_topk_cache.released_sources
                ),
                pending_offloads=overrides.get(
                    "dsa_topk_pending_offloads", self.dsa_topk_cache.pending_offloads
                ),
                pending_releases=overrides.get(
                    "dsa_topk_pending_releases", self.dsa_topk_cache.pending_releases
                ),
                checkpoint_active=overrides.get(
                    "dsa_topk_checkpoint_active", self.dsa_topk_cache.checkpoint_active
                ),
                context_id=overrides.get("dsa_topk_context_id", self.dsa_topk_cache.context_id),
            )

        return SequenceContextPseudo(
            input_ids=overrides.get("input_ids", self.input_ids),
            inputs_embeds=overrides.get("inputs_embeds", self.inputs_embeds),
            position_ids=overrides.get("position_ids", self.position_ids),
            dsa_topk_cache=dsa_topk_cache,
        )

    def split_dsa_topk_indices_to(self, sequence_context_list: list["SequenceContextPseudo"]) -> None:
        if not self.dsa_topk_cache.indices:
            return

        lengths = [ctx.seq_len for ctx in sequence_context_list]
        for source_layer_idx, topk_indices in self.dsa_topk_cache.indices.items():
            assert sum(lengths) == topk_indices.shape[0]
            for seq_ctx, single_topk in zip(sequence_context_list, topk_indices.split(lengths, dim=0)):
                seq_ctx.dsa_topk_cache.indices[source_layer_idx] = single_topk

    @property
    def seq_len(self) -> int:
        if self.input_ids is not None:
            return int(self.input_ids.shape[1])
        if self.inputs_embeds is not None:
            return int(self.inputs_embeds.shape[1])
        assert self.position_ids is not None
        return int(self.position_ids.shape[-1])


# ---------------------------------------------------------------------------
# Runtime layer protocol.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Residency adapters.
# ---------------------------------------------------------------------------


class TopKResidencyBase:
    reuse_source_topk_in_recompute = True

    def has_cache(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> bool:
        return source_layer_idx in seq_ctx.dsa_topk_cache.indices

    def store_gpu(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int, topk: torch.Tensor) -> None:
        seq_ctx.dsa_topk_cache.indices[source_layer_idx] = topk

    def read(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> torch.Tensor:
        return seq_ctx.dsa_topk_cache.indices[source_layer_idx]

    def after_original_forward_last_use(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> None:
        return

    def after_recompute_release(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> None:
        seq_ctx.dsa_topk_cache.indices.pop(source_layer_idx, None)

    def _offload_key(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> str:
        return f"dsa_topk_{seq_ctx.dsa_topk_cache.context_id}_{source_layer_idx}"


class GpuTopKResidency(TopKResidencyBase):
    pass


class ActivationOffloadedTopKResidency(TopKResidencyBase):
    """Pseudo OffloadManager/SwapTensor-backed residency."""

    def __init__(self) -> None:
        self._prefetched: dict[tuple[int, int], object] = {}

    def has_cache(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> bool:
        cache = seq_ctx.dsa_topk_cache
        return source_layer_idx in cache.indices or source_layer_idx in cache.offloaded

    def read(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> torch.Tensor:
        cache = seq_ctx.dsa_topk_cache
        if source_layer_idx in cache.indices:
            self._wait_prefetched(seq_ctx, source_layer_idx)
            return cache.indices[source_layer_idx]

        key = cache.offloaded[source_layer_idx]
        # Production code:
        #   swap_tensor = OffloadManager().get(key)
        #   launch H2D on a side stream
        #   current_stream.wait_stream(side_stream)
        topk = torch.empty(0, dtype=torch.int64, device="cuda")
        cache.indices[source_layer_idx] = topk
        _ = key
        return topk

    def prefetch(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> None:
        cache = seq_ctx.dsa_topk_cache
        if source_layer_idx in cache.indices or source_layer_idx not in cache.offloaded:
            return

        key = cache.offloaded[source_layer_idx]
        # Production code:
        #   swap_tensor = OffloadManager().get(key)
        #   swap_tensor.prefetch_launch_h2d(side_stream, True)
        #   cache.indices[source_layer_idx] = swap_tensor.tensor
        #   self._prefetched[(id(cache), source_layer_idx)] = swap_tensor
        topk = torch.empty(0, dtype=torch.int64, device="cuda")
        cache.indices[source_layer_idx] = topk
        self._prefetched[(id(cache), source_layer_idx)] = object()
        _ = key

    def _wait_prefetched(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> None:
        token = self._prefetched.pop((id(seq_ctx.dsa_topk_cache), source_layer_idx), None)
        if token is None:
            return
        # Production code:
        #   swap_tensor.wait_h2d_finished()
        _ = token

    def after_original_forward_last_use(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> None:
        cache = seq_ctx.dsa_topk_cache
        topk = cache.indices.pop(source_layer_idx)
        if not topk.is_cuda:
            cache.indices[source_layer_idx] = topk
            return

        key = self._offload_key(seq_ctx, source_layer_idx)
        # Production code:
        #   cpu_buffer = OffloadManager().get_or_create_pin_memory(key, topk.shape, topk.dtype)
        #   swap_tensor = SwapTensor(topk, key, tensor_cpu=cpu_buffer)
        #   launch D2H and wait before dropping the GPU tensor
        #   OffloadManager().put(key, swap_tensor)
        cache.offloaded[source_layer_idx] = key

    def after_recompute_release(self, seq_ctx: SequenceContextPseudo, source_layer_idx: int) -> None:
        cache = seq_ctx.dsa_topk_cache
        self._wait_prefetched(seq_ctx, source_layer_idx)
        super().after_recompute_release(seq_ctx, source_layer_idx)
        key = cache.offloaded.pop(source_layer_idx, None)
        if key is not None:
            # Production code clears OffloadManager state for this key.
            pass


# ---------------------------------------------------------------------------
# Deep Module: Cross-Layer Top-K Sharing Runtime.
# ---------------------------------------------------------------------------


class CrossLayerTopKSharingRuntime:
    def __init__(self) -> None:
        self._gpu_residency = GpuTopKResidency()
        self._offloaded_residency = ActivationOffloadedTopKResidency()

    def get_or_compute(
        self,
        *,
        layer: DSATopKSharingLayerProtocol,
        seq_ctx: SequenceContextPseudo,
        compute_source_topk: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        residency = self._residency()
        cache = seq_ctx.dsa_topk_cache
        source = layer.source_layer_idx

        if layer._is_skip_topk_layer():
            self._assert_source_present(layer, seq_ctx, residency)
            if source in cache.indices:
                return cache.indices[source]
            return residency.read(seq_ctx, source)

        if (
            self._is_checkpoint_recompute(seq_ctx)
            and layer.layer_idx not in cache.released_sources
            and residency.has_cache(seq_ctx, source)
        ):
            if source in cache.indices:
                return cache.indices[source]
            return residency.read(seq_ctx, source)

        topk = compute_source_topk()
        if layer.layer_idx not in cache.released_sources:
            residency.store_gpu(seq_ctx, layer.layer_idx, topk)
        return topk

    def after_sparse_mla_use(self, *, layer: DSATopKSharingLayerProtocol, seq_ctx: SequenceContextPseudo) -> None:
        residency = self._residency()
        cache = seq_ctx.dsa_topk_cache
        source = layer.source_layer_idx

        if self._is_checkpoint_original_forward(layer):
            if layer.dsa_topk_last_use.get(source) == layer.layer_idx:
                cache.checkpoint_active = True
                if self._should_defer_transfer():
                    cache.pending_offloads.add(source)
                else:
                    residency.after_original_forward_last_use(seq_ctx, source)
            return

        if not self._is_checkpoint_recompute(seq_ctx):
            return

        if layer.dsa_topk_recompute_release.get(source) != layer.layer_idx:
            return

        if self._should_defer_transfer():
            cache.pending_releases.add(source)
        else:
            residency.after_recompute_release(seq_ctx, source)
            cache.released_sources.add(source)

    def before_layer_forward(self, *, layer: DSATopKSharingLayerProtocol, seq_ctx: SequenceContextPseudo) -> None:
        if not isinstance(self._residency(), ActivationOffloadedTopKResidency):
            return
        source = layer.source_layer_idx
        if source not in seq_ctx.dsa_topk_cache.offloaded:
            return
        self._offloaded_residency.prefetch(seq_ctx, source)

    def flush_pending(self, seq_ctx: SequenceContextPseudo) -> None:
        cache = seq_ctx.dsa_topk_cache
        if not isinstance(self._residency(), ActivationOffloadedTopKResidency):
            cache.pending_offloads.clear()
            cache.pending_releases.clear()
            return

        for source in tuple(cache.pending_offloads):
            self._offloaded_residency.after_original_forward_last_use(seq_ctx, source)
            cache.pending_offloads.remove(source)

        for source in tuple(cache.pending_releases):
            self._offloaded_residency.after_recompute_release(seq_ctx, source)
            cache.released_sources.add(source)
            cache.pending_releases.remove(source)

    def _residency(self) -> TopKResidencyBase:
        if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1 and torch.cuda.is_available():
            return self._offloaded_residency
        return self._gpu_residency

    def _should_defer_transfer(self) -> bool:
        return torch.compiler.is_compiling() and isinstance(self._residency(), ActivationOffloadedTopKResidency)

    def _is_checkpoint_original_forward(self, layer: DSATopKSharingLayerProtocol) -> bool:
        return layer.training and not torch.is_grad_enabled()

    def _is_checkpoint_recompute(self, seq_ctx: SequenceContextPseudo) -> bool:
        return seq_ctx.dsa_topk_cache.checkpoint_active and torch.is_grad_enabled()

    def _assert_source_present(
        self,
        layer: DSATopKSharingLayerProtocol,
        seq_ctx: SequenceContextPseudo,
        residency: TopKResidencyBase,
    ) -> None:
        if residency.has_cache(seq_ctx, layer.source_layer_idx):
            return
        raise AssertionError(
            "DSA index-share: skip layer "
            f"{layer.layer_idx} needs source layer {layer.source_layer_idx} top-k, "
            "but it is not present in this microbatch SequenceContext."
        )


_DSA_TOPK_RUNTIME = CrossLayerTopKSharingRuntime()


def get_dsa_topk_sharing_runtime() -> CrossLayerTopKSharingRuntime:
    return _DSA_TOPK_RUNTIME


# ---------------------------------------------------------------------------
# Decoder lifecycle hook seam.
# ---------------------------------------------------------------------------


def configure_dsa_topk_decoder_lifecycle(
    *,
    decoder_layer: Any,
    attention: DSATopKSharingLayerProtocol,
    release_plan: DSATopKReleasePlan,
) -> None:
    # Release maps and hooks are one lifecycle contract.
    attention.dsa_topk_last_use = release_plan.forward_last_use
    attention.dsa_topk_recompute_release = release_plan.recompute_release
    register_dsa_topk_decoder_lifecycle_hooks(decoder_layer)


def before_dsa_topk_decoder_forward(
    attention: DSATopKSharingLayerProtocol,
    seq_ctx: SequenceContextPseudo | list[SequenceContextPseudo],
) -> None:
    runtime = get_dsa_topk_sharing_runtime()
    for ctx in seq_ctx if isinstance(seq_ctx, list) else [seq_ctx]:
        runtime.before_layer_forward(layer=attention, seq_ctx=ctx)


def flush_dsa_topk_decoder_pending(seq_ctx: SequenceContextPseudo | list[SequenceContextPseudo]) -> None:
    runtime = get_dsa_topk_sharing_runtime()
    for ctx in seq_ctx if isinstance(seq_ctx, list) else [seq_ctx]:
        runtime.flush_pending(ctx)


def register_dsa_topk_decoder_lifecycle_hooks(decoder_layer: Any) -> None:
    # Production code uses torch forward pre/post hooks with with_kwargs=True.
    decoder_layer.register_forward_pre_hook("before_dsa_topk_decoder_forward")
    decoder_layer.register_forward_hook("flush_dsa_topk_decoder_pending")


# ---------------------------------------------------------------------------
# DSAMultiLatentAttention usage.
# ---------------------------------------------------------------------------


class DSAMultiLatentAttentionPseudo:
    def __init__(self, *, layer_idx: int, indexer_types: list[str] | None) -> None:
        self.layer_idx = layer_idx
        self.indexer_types = indexer_types
        self.index_skip_topk_offset = 3
        self.index_topk_freq = 4
        self.dsa_topk_last_use: dict[int, int] = {}
        self.dsa_topk_recompute_release: dict[int, int] = {}
        self.training = True
        self.source_layer_idx = source_layer_for_plan(
            layer_idx=layer_idx,
            indexer_types=indexer_types,
            index_skip_topk_offset=self.index_skip_topk_offset,
            index_topk_freq=self.index_topk_freq,
        )

    def _is_skip_topk_layer(self) -> bool:
        return self.source_layer_idx != self.layer_idx

    def _compute_source_topk(self, hidden_states: Any, q_resid: Any, position_embeddings: Any) -> torch.Tensor:
        # Real implementation calls DSAIndexer.
        return torch.empty(0, dtype=torch.int64, device="cuda")

    def forward(self, hidden_states: Any, position_embeddings: Any, seq_ctx: SequenceContextPseudo) -> dict[str, Any]:
        q_resid = "q_a_layernorm(q_a_proj(hidden_states))"
        topk_indices = get_dsa_topk_sharing_runtime().get_or_compute(
            layer=self,
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
        get_dsa_topk_sharing_runtime().after_sparse_mla_use(layer=self, seq_ctx=seq_ctx)
        return {"projected_output": "o_proj(raw_output)", **sparse_mla_outputs}

    def sparse_mla_func(self, q: Any, kv: Any, indices: torch.Tensor, scaling: Any, value_dim: Any) -> dict[str, Any]:
        return {"raw_output": "raw_output", "softmax_lse": "softmax_lse"}


# ---------------------------------------------------------------------------
# MoE / GLM-5.2 integration.
# ---------------------------------------------------------------------------


@dataclass
class MTPConfigPseudo:
    num_layers: int = 1
    share_weights: bool = False


@dataclass
class MoEConfigPseudo:
    num_hidden_layers: int
    indexer_types: list[str]
    mtp_config: MTPConfigPseudo | None = None


class DecoderLayerPseudo:
    def __init__(self, self_attn: DSAMultiLatentAttentionPseudo) -> None:
        self.self_attn = self_attn

    def register_forward_pre_hook(self, hook: Any, with_kwargs: bool = True) -> None:
        _ = hook, with_kwargs

    def register_forward_hook(self, hook: Any, with_kwargs: bool = True) -> None:
        _ = hook, with_kwargs

    def __call__(self, hidden_states: Any, *, position_embeddings: Any, seq_ctx: SequenceContextPseudo) -> Any:
        before_dsa_topk_decoder_forward(self.self_attn, seq_ctx)
        try:
            return self.self_attn.forward(hidden_states, position_embeddings, seq_ctx)["projected_output"]
        finally:
            flush_dsa_topk_decoder_pending(seq_ctx)


class BaseMoEPseudo:
    def __init__(self, config: MoEConfigPseudo) -> None:
        self.config = config
        self.layers = [
            DecoderLayerPseudo(DSAMultiLatentAttentionPseudo(layer_idx=i, indexer_types=config.indexer_types))
            for i in range(config.num_hidden_layers)
        ]
        self.mtp_layers = self._build_mtp_layers(config)
        self._configure_model_specific_layer_lifecycle()

    def _build_mtp_layers(self, config: MoEConfigPseudo) -> list[DecoderLayerPseudo]:
        if config.mtp_config is None:
            return []
        num_mtp_layers = 1 if config.mtp_config.share_weights else config.mtp_config.num_layers
        return [
            DecoderLayerPseudo(
                DSAMultiLatentAttentionPseudo(
                    layer_idx=config.num_hidden_layers + i,
                    indexer_types=config.indexer_types,
                )
            )
            for i in range(num_mtp_layers)
        ]

    def _configure_model_specific_layer_lifecycle(self) -> None:
        return

    def __call__(self, *, seq_ctx: SequenceContextPseudo, loss_ctx: Any | None) -> dict[str, Any]:
        hidden_states = "embed_tokens(input_ids)"
        position_embeddings = "rotary_emb(hidden_states, position_ids)"
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        for mtp_layer in self.mtp_layers:
            hidden_states = mtp_layer(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        return {"loss": "loss_ctx(hidden_states)", "hidden_states": hidden_states}

    def forward_micro_batch(self, seq_ctx_list: list[SequenceContextPseudo]) -> None:
        cat_seq_ctx = SequenceContextPseudo.cat(seq_ctx_list)
        dense_prefix_outputs = "dense prefix runs on cat_seq_ctx"
        _ = dense_prefix_outputs
        cat_seq_ctx.split_dsa_topk_indices_to(seq_ctx_list)
        for seq_ctx in seq_ctx_list:
            self(seq_ctx=seq_ctx, loss_ctx=None)


class Glm52MoEPseudo(BaseMoEPseudo):
    def _configure_model_specific_layer_lifecycle(self) -> None:
        num_mtp_layers = len(self.mtp_layers)
        release_plan = build_dsa_topk_release_plan(
            num_main_layers=self.config.num_hidden_layers,
            num_mtp_layers=num_mtp_layers,
            indexer_types=self.config.indexer_types,
            index_skip_topk_offset=3,
            index_topk_freq=4,
        )

        for decoder_layer in [*self.layers, *self.mtp_layers]:
            configure_dsa_topk_decoder_lifecycle(
                decoder_layer=decoder_layer,
                attention=decoder_layer.self_attn,
                release_plan=release_plan,
            )


# ---------------------------------------------------------------------------
# Client usage and remaining design work.
# ---------------------------------------------------------------------------


def client_usage_current_interface(input_ids: torch.Tensor) -> None:
    """SFT clients do not touch Cross-Layer Top-K Sharing internals."""

    os.environ["XTUNER_ACTIVATION_OFFLOAD"] = "1"
    model = Glm52MoEPseudo(MoEConfigPseudo(num_hidden_layers=3, indexer_types=["full", "shared", "shared"]))
    seq_ctx = SequenceContextPseudo(input_ids=input_ids)
    outputs = model(seq_ctx=seq_ctx, loss_ctx="lm")
    outputs["loss"]


def remaining_design_work() -> list[str]:
    return []

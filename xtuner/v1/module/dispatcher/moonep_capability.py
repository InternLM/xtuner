"""MoonEP capability checks at the two points where they become decidable.

``check_config`` runs during meta model build, when only ``MoEConfig`` is
visible; ``check_fsdp_policy`` runs just before ``fully_shard``, when
``FSDPConfig`` is visible. Neither creates a resource. Two things stay out of
this module by design: the workspace still owns the preconditions for safely
calling VMM (node-local, peer access, chunk alignment), and
``fsdp_vmm_landing`` still owns the torch-version / FSDP-ABI fail-fast.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def check_config(config: Any) -> None:
    """Point 1: meta model build. Only ``MoEConfig`` is visible.

    Adding a new MoonEP capability constraint means adding one check here, and
    it cannot be missed from a second call site.
    """
    if config.dispatcher != "moonep":
        return

    # Config-shape checks first, so they surface even where the optional
    # backend is not installed.
    float8_cfg = getattr(config, "float8_cfg", None)
    if float8_cfg is not None and float8_cfg.enable_float8:
        raise ValueError("MoonEP currently requires BF16 expert compute; FP8 is not supported")
    if config.ep_size <= 1:
        raise ValueError("MoonEP requires expert parallelism")
    if config.ep_size not in (2, 4, 8):
        raise ValueError("MoonEP requires ep_size in {2, 4, 8}")
    if config.n_routed_experts % config.ep_size:
        raise ValueError("MoonEP requires n_routed_experts divisible by ep_size")
    if config.moe_bias:
        raise ValueError("MoonEP does not support routed-expert linear bias")
    if config.expert_tp_size > 1:
        raise ValueError("MoonEP first version is TP1 only")
    if config.intra_layer_micro_batch < 1:
        raise ValueError("intra_layer_micro_batch must be positive")

    from .moonep import require_moonep_backend

    require_moonep_backend()
    _require_cutlass_grouped_gemm_when_selected()


def check_fsdp_policy(config: Any, fsdp_config: Any) -> None:
    """Point 2: before ``fully_shard``. ``FSDPConfig`` is now visible.

    The two points cannot merge: ``FSDPConfig`` does not exist yet at
    ``MoE.__init__``.
    """
    if config.dispatcher != "moonep":
        return

    if fsdp_config.param_dtype is not torch.bfloat16 or fsdp_config.reduce_dtype is not torch.bfloat16:
        raise ValueError("MoonEP requires BF16 FSDP param and reduce dtypes")
    if fsdp_config.cpu_offload:
        raise ValueError("MoonEP VMM weights cannot use FSDP CPU offload")
    if not fsdp_config.requires_grad:
        raise ValueError("MoonEP v1 requires trainable FSDP parameters")
    if not fsdp_config.reshard_after_forward:
        raise ValueError("MoonEP requires reshard_after_forward=True")

    # A blocking rule from the domain model that had no code before: a
    # checkpoint-wrapped MTP physical layer must use reentrant checkpointing.
    if _any_mtp_layer_checkpoint_wrapped(config, fsdp_config) and not fsdp_config.mtp_checkpoint_use_reentrant:
        raise ValueError(
            "a checkpoint-wrapped MTP physical layer requires FSDPConfig.mtp_checkpoint_use_reentrant=True with MoonEP"
        )


def _require_cutlass_grouped_gemm_when_selected() -> None:
    # MoonEP keeps token counts device-resident. Triton already satisfies that
    # contract; grouped_gemm does so only with its CUTLASS backend.
    from xtuner.v1.module.grouped_linear import moe_group_linear
    from xtuner.v1.ops.moe.cuda import cutlass_group_gemm

    if cutlass_group_gemm is not None and moe_group_linear.group_gemm is cutlass_group_gemm:
        from grouped_gemm import backend as grouped_gemm_backend

        if os.environ.get("GROUPED_GEMM_USE_CUTLASS") != "1" or not grouped_gemm_backend.use_cutlass:
            raise RuntimeError(
                "MoonEP with grouped_gemm requires GROUPED_GEMM_USE_CUTLASS=1 before importing grouped_gemm"
            )


def _any_mtp_layer_checkpoint_wrapped(config: Any, fsdp_config: Any) -> bool:
    mtp_config = getattr(config, "mtp_config", None)
    if mtp_config is None:
        return False
    if mtp_config.share_weights:
        return True
    # Non-shared: a non-terminal MTP layer selected by the recompute ratio is
    # checkpoint-wrapped. Mirror ``MoE._should_recompute``'s global-index rule.
    total_layers = config.num_hidden_layers + mtp_config.num_layers
    num_recompute = int(total_layers * (getattr(fsdp_config, "recompute_ratio", 0.0) or 0.0))
    return any((config.num_hidden_layers + mtp_idx) < num_recompute for mtp_idx in range(mtp_config.num_layers - 1))


__all__ = ["check_config", "check_fsdp_policy"]

# Copyright (c) OpenMMLab. All rights reserved.
"""Resolve KPool preparation and selection from one serialized backend."""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.nn import functional as F

from xtuner.v1.ops.fp32_linear import fp32_linear
from xtuner.v1.utils.dtensor import materialize_full

from .kpool import kpool_topk_indices, torch_kpool_topk_indices
from .protocol import KPoolIndexerBackend, KPoolTopKIndicesProtocol


if TYPE_CHECKING:
    from xtuner.v1.model.moe.glm53.nope_dsa_mla import KPoolIndexer


@dataclass(frozen=True)
class KPoolPlan:
    """Preparation and selection resolved together when the indexer is
    built."""

    prepare: Callable[["KPoolIndexer", Tensor], tuple[Tensor, Tensor]]
    topk_indices: KPoolTopKIndicesProtocol


def _prepare_native(indexer: "KPoolIndexer", hidden_states: Tensor) -> tuple[Tensor, Tensor]:
    return indexer.k_norm(indexer.wk(hidden_states)), indexer.weights_proj(hidden_states).float()


def _prepare_fp32(indexer: "KPoolIndexer", hidden_states: Tensor) -> tuple[Tensor, Tensor]:
    projected_k = indexer.wk(hidden_states)
    norm_weight = materialize_full(indexer.k_norm.weight, name="indexer.k_norm.weight")
    norm_bias = materialize_full(indexer.k_norm.bias, name="indexer.k_norm.bias")
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        keys = F.layer_norm(
            projected_k.float(),
            indexer.k_norm.normalized_shape,
            norm_weight.float(),
            norm_bias.float(),
            indexer.k_norm.eps,
        ).to(projected_k.dtype)
    head_weight = materialize_full(cast(Tensor, indexer.weights_proj.weight), name="indexer.weights_proj.weight")
    return keys, fp32_linear(hidden_states, head_weight)


def _prepare_lmdeploy(indexer: "KPoolIndexer", hidden_states: Tensor, *, precision: str) -> tuple[Tensor, Tensor]:
    from .lmdeploy_kpool import require_lmdeploy_kpool_device

    require_lmdeploy_kpool_device(hidden_states.device, precision)
    return _prepare_fp32(indexer, hidden_states)


def get_kpool_topk_indices(backend: KPoolIndexerBackend) -> KPoolTopKIndicesProtocol:
    if backend == "torch":
        return torch_kpool_topk_indices
    if backend == "tilelang":
        return kpool_topk_indices
    if backend == "tilelang_deepselect":
        return functools.partial(kpool_topk_indices, selector="deep_select")
    if backend == "tilelang_cooperative":
        return functools.partial(kpool_topk_indices, scorer="cooperative")
    if backend == "lmdeploy_fp8":
        from .lmdeploy_kpool import lmdeploy_fp8_kpool_topk_indices

        return lmdeploy_fp8_kpool_topk_indices
    if backend == "lmdeploy_bf16":
        from .lmdeploy_kpool import lmdeploy_bf16_kpool_topk_indices

        return lmdeploy_bf16_kpool_topk_indices
    raise ValueError(f"Unsupported KPool indexer backend: {backend}")


def get_kpool_plan(
    backend: KPoolIndexerBackend,
    *,
    index_head_dim: int,
    index_n_heads: int,
    index_kpool: int,
    index_topk: int,
    balance_sp: bool = False,
    fp32_prep: bool = False,
) -> KPoolPlan:
    prepare = _prepare_fp32 if fp32_prep else _prepare_native
    if backend in ("lmdeploy_fp8", "lmdeploy_bf16"):
        from .lmdeploy_kpool import validate_lmdeploy_kpool_config

        validate_lmdeploy_kpool_config(
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_kpool=index_kpool,
            index_topk=index_topk,
            balance_sp=balance_sp,
        )
        # FP32 preparation is part of both serving adapters' contract.
        prepare = functools.partial(_prepare_lmdeploy, precision=backend.removeprefix("lmdeploy_"))
    return KPoolPlan(prepare, get_kpool_topk_indices(backend))

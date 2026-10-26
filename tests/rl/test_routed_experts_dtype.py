from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import ray
import torch

from xtuner.v1.module.decoder_layer.moe_decoder_layer import _prepare_rollout_routed_experts_for_router
from xtuner.v1.rl.trainer.role_worker import (
    _as_rollout_routed_experts_tensor,
    _rollout_routed_experts_storage_dtype,
    resolve_routed_experts,
)


@pytest.mark.parametrize("n_routed_experts", [256, 257, 65536])
def test_rollout_routed_experts_use_uint16_storage(n_routed_experts: int):
    assert _rollout_routed_experts_storage_dtype(n_routed_experts) == torch.uint16


def test_rollout_routed_experts_fall_back_to_long_above_uint16_capacity():
    assert _rollout_routed_experts_storage_dtype(65537) == torch.long


@pytest.mark.parametrize("expert_ids", [np.array([-1], dtype=np.int64), np.array([65536], dtype=np.int64)])
def test_rollout_routed_experts_reject_uint16_overflow(expert_ids: np.ndarray):
    with pytest.raises(ValueError, match="cannot be represented as uint16"):
        _as_rollout_routed_experts_tensor(expert_ids, n_routed_experts=65536)


def _language_cfg(*, n_routed_experts: int):
    return SimpleNamespace(
        n_routed_experts=n_routed_experts,
        num_hidden_layers=2,
        num_experts_per_tok=2,
    )


def test_training_worker_keeps_ray_routes_and_padding_in_uint16():
    model_cfg = _language_cfg(n_routed_experts=65536)
    route_ref = ray.ObjectRef(bytes(28))
    routed_experts = np.array([[[0, 255], [256, 65535]]], dtype=np.uint16)
    padding_marker = torch.empty(1)
    seq_ctx = SimpleNamespace(
        input_ids=torch.zeros((1, 2), dtype=torch.long),
        rollout_routed_experts=[route_ref, padding_marker],
    )

    with patch("xtuner.v1.rl.trainer.role_worker.ray.get", return_value=routed_experts):
        resolve_routed_experts(seq_ctx, model_cfg=model_cfg, pack_max_length=2)

    assert seq_ctx.rollout_routed_experts.dtype == torch.uint16
    torch.testing.assert_close(
        seq_ctx.rollout_routed_experts[0].long(),
        torch.tensor([[0, 255], [256, 65535]], dtype=torch.long),
    )


def test_training_worker_uses_uint16_for_full_padding_batch():
    model_cfg = _language_cfg(n_routed_experts=257)
    seq_ctx = SimpleNamespace(
        input_ids=torch.zeros((1, 2), dtype=torch.long),
        rollout_routed_experts=torch.empty(0),
    )

    resolve_routed_experts(seq_ctx, model_cfg=model_cfg, pack_max_length=2)

    assert seq_ctx.rollout_routed_experts.dtype == torch.uint16
    assert seq_ctx.rollout_routed_experts.shape == (2, 2, 2)


def test_training_worker_preserves_long_fallback():
    model_cfg = _language_cfg(n_routed_experts=65537)
    route_ref = ray.ObjectRef(bytes(28))
    routed_experts = np.array([[[65536, 0], [1, 2]]], dtype=np.int64)
    seq_ctx = SimpleNamespace(
        input_ids=torch.zeros((1, 1), dtype=torch.long),
        rollout_routed_experts=[route_ref],
    )

    with patch("xtuner.v1.rl.trainer.role_worker.ray.get", return_value=routed_experts):
        resolve_routed_experts(seq_ctx, model_cfg=model_cfg, pack_max_length=1)

    assert seq_ctx.rollout_routed_experts.dtype == torch.long
    assert seq_ctx.rollout_routed_experts[0, 0, 0].item() == 65536


@pytest.mark.parametrize("offload_rollout_routed_experts", [False, True])
def test_uint16_layer_slice_is_converted_to_router_index_dtype(offload_rollout_routed_experts: bool):
    stored_routes = torch.tensor(
        [
            [[0, 255], [256, 65535]],
            [[1, 2], [3, 4]],
        ],
        dtype=torch.uint16,
    )
    layer_slice = stored_routes[:, 1, :]
    assert not layer_slice.is_contiguous()

    router_routes = _prepare_rollout_routed_experts_for_router(
        layer_slice,
        torch.zeros((2, 4)),
        offload_rollout_routed_experts=offload_rollout_routed_experts,
    )

    assert router_routes.dtype == torch.long
    torch.testing.assert_close(router_routes, torch.tensor([[256, 65535], [3, 4]], dtype=torch.long))
    routing_weights = torch.zeros((2, 65536))
    routing_weights.gather(dim=1, index=router_routes)

"""Regression tests for FP32-ignored param remesh inside ``BaseModel._fully_shard``.

Contract: parameters matching ``fp32_keys_pattern`` must end as
``DTensor(world_mesh, all Replicate)`` so grad reduce / norm stay consistent
across FSDP ranks. Remesh must be idempotent under MoE's layer-then-root
``_fully_shard`` calls.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Replicate

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.base import HFSaveCfg
from xtuner.v1.model.moe.moe import MoE
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig


def _tiny_moe_cfg() -> Qwen3MoEConfig:
    return Qwen3MoEConfig(
        vocab_size=32,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_hidden_layers=1,
        hidden_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=2, num_key_value_heads=1, head_dim=8, qk_norm=True),
        tie_word_embeddings=False,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        hidden_factor=1.0,
        moe_intermediate_size=8,
        router=GreedyRouterConfig(scoring_func="softmax", norm_topk_prob=True, router_scaling_factor=1.0),
        ep_size=2,
        expert_tp_size=1,
        dispatcher="all2all",
        compile_cfg=False,
        balancing_loss_cfg=None,
        z_loss_cfg=None,
        # Match router gate HF key ``model.layers.*.mlp.gate.weight``.
        hf_save_cfg=HFSaveCfg(fp32_keys_pattern=[r"gate\.weight"]),
    )


def _mp_policy() -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)


class TestFP32IgnoredRemesh(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 4

    def _build_moe(self) -> MoE:
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        model = _tiny_moe_cfg().build().cuda()
        model.init_weights()
        model._init_device_mesh(
            FSDPConfig(
                ep_size=2,
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                torch_compile=False,
            )
        )
        return model

    def _call_fully_shard(self, model: MoE, module: nn.Module | None) -> None:
        assert model.fsdp_mesh is not None
        model._fully_shard(
            mesh=model.fsdp_mesh,
            mp_policy=_mp_policy(),
            reshard_after_forward=True,
            offload_policy=None,
            module=module,
        )

    def test_remesh_ep_replicate_onto_world(self) -> None:
        """MoE ``_replicate_other_params`` then layer ``_fully_shard`` remeshes gate to world."""
        self.create_pg("cuda")
        model = self._build_moe()
        layer = model.layers["0"]

        model._replicate_other_params(model)
        model._init_load_spec()

        pre = layer.gate.weight
        assert isinstance(pre, DTensor)
        assert pre.device_mesh == model.ep_mesh
        assert all(isinstance(placement, Replicate) for placement in pre.placements)
        # World remesh uses distribute_tensor(..., all Replicate), which broadcasts
        # from global rank 0; EP-only replicas may still differ across FSDP rows.
        expected = pre.to_local().detach().clone()
        dist.broadcast(expected, src=0)

        self._call_fully_shard(model, module=layer)

        remeshed = layer.gate.weight
        assert isinstance(remeshed, DTensor)
        assert remeshed.device_mesh == model.world_mesh
        assert all(isinstance(placement, Replicate) for placement in remeshed.placements)
        torch.testing.assert_close(remeshed.to_local(), expected)

    def test_remesh_is_idempotent_across_layer_and_root(self) -> None:
        """Layer then root ``_fully_shard`` must not rebuild an already-correct replica."""
        self.create_pg("cuda")
        model = self._build_moe()
        layer = model.layers["0"]

        model._replicate_other_params(model)
        model._init_load_spec()

        self._call_fully_shard(model, module=layer)
        after_layer = layer.gate.weight
        assert isinstance(after_layer, DTensor)
        assert after_layer.device_mesh == model.world_mesh

        # MoE root pass traverses into the already-sharded layer again.
        self._call_fully_shard(model, module=None)
        assert layer.gate.weight is after_layer

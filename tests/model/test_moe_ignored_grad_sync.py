"""真实 FSDP ignored 参数的复制域梯度回归；不依赖 checkpoint 或 attention kernel。

TestMoEIgnoredGradSync
    test_ignored_replica_mean_and_two_updates  EP/SP 下梯度尺度及两步更新保持副本一致，保留分片语义
"""

import unittest

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router.greedy import GreedyRouterConfig
from xtuner.v1.utils.interleaved_shard import InterleavedShard


def _config(ep_size: int, sp_size: int) -> Qwen3MoEConfig:
    config = Qwen3MoEConfig(
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
        attention=MHAConfig(num_attention_heads=2, num_key_value_heads=1, head_dim=8),
        tie_word_embeddings=False,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        hidden_factor=1.0,
        moe_intermediate_size=8,
        router=GreedyRouterConfig(scoring_func="softmax", norm_topk_prob=True, router_scaling_factor=1.0),
        ep_size=ep_size,
        dispatcher="all2all" if ep_size > 1 else None,
        mesh_prefix=f"ignored_ep{ep_size}_sp{sp_size}",
        compile_cfg=False,
        balancing_loss_cfg=None,
        z_loss_cfg=None,
    )
    config.hf_save_cfg.fp32_keys_pattern = [r"^model\.layers\.0\.input_layernorm\.weight$"]
    return config


@pytest.mark.gpu
@unittest.skipIf(torch.cuda.device_count() < 4, "Requires four CUDA devices and NCCL")
class TestMoEIgnoredGradSync(DeterministicDDPTestCase):
    """真实 FSDP ignored 参数的梯度尺度、更新和受保护分片路径。"""

    @property
    def world_size(self) -> int:
        return 4

    def test_ignored_replica_mean_and_two_updates(self) -> None:
        # 使用真实 meta 构建、EP 分发和 FSDP ignore 路径，避免手工伪造待修复的参数布局。
        torch.cuda.set_device(self.rank)
        pg = self.create_pg("cuda")
        try:
            for ep_size, sp_size in ((1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (4, 2)):
                self._check_case(ep_size, sp_size)
        finally:
            dist.destroy_process_group(pg)

    def _check_case(self, ep_size: int, sp_size: int) -> None:
        with torch.device("meta"):
            model = _config(ep_size, sp_size).build()
        assert all(parameter.is_meta for parameter in model.parameters())
        model.fully_shard(
            FSDPConfig(ep_size=ep_size, param_dtype=torch.bfloat16, reduce_dtype=torch.float32, recompute_ratio=0)
        )
        parameter = model.layers["0"].input_layernorm.weight
        assert isinstance(parameter, DTensor)
        assert parameter.dtype == torch.float32 and not parameter.is_meta
        assert all(isinstance(placement, Replicate) for placement in parameter.placements)
        assert model.world_mesh is not None
        assert model.world_mesh.size() == self.world_size
        assert parameter.device_mesh.size() == (ep_size if ep_size > 1 else self.world_size)
        assert any(parameter is p for _, p in model.trainable_parameters())
        with torch.no_grad():
            parameter.to_local().fill_(1.0)
        optimizer = torch.optim.SGD([parameter], lr=0.01)

        # 保护普通 FSDP shard 和 expert 原有归约语义；它们不能误走完整复制组平均。
        regular = model.embed_tokens.weight
        expert = model.layers["0"].experts.fused_w2.weight
        assert isinstance(regular, DTensor) and isinstance(expert, DTensor)
        assert any(isinstance(placement, Shard) for placement in regular.placements)
        regular_mesh = regular.device_mesh
        replica_names = tuple(
            regular_mesh.mesh_dim_names[i] for i, p in enumerate(regular.placements) if isinstance(p, Replicate)
        )
        regular_mean = float(self.rank + 1)
        if replica_names:
            replica_mesh = regular_mesh[replica_names[0]] if len(replica_names) == 1 else regular_mesh[replica_names]
            regular_mean = float((replica_mesh.mesh.float() + 1).mean())

        # InterleavedShard 也不属于完整副本，即使只有一个 placement。
        flat_mesh = model.world_mesh._flatten()
        interleaved = torch.nn.Parameter(
            DTensor.from_local(
                torch.zeros(8, 2, device="cuda"),
                flat_mesh,
                [InterleavedShard(0, num_local_stripes=2)],
                run_check=False,
            )
        )
        model.register_parameter("interleaved_probe", interleaved)

        # SP 只改变样本/序列分配。此处检验梯度归约尺度，不声称覆盖 attention 的 SP kernel。
        data_mesh = init_device_mesh(
            "cuda", (self.world_size // sp_size, sp_size), mesh_dim_names=(f"data{ep_size}{sp_size}", "sp")
        )
        sample_rank, sequence_rank = data_mesh.get_coordinate()
        tokens = torch.arange(1, self.world_size * 4 + 1, device="cuda", dtype=torch.float32)
        local_tokens = tokens.reshape(self.world_size // sp_size, 4 * sp_size)[sample_rank].chunk(sp_size)[
            sequence_rank
        ]
        expected_parameter = 1.0
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            # 全局 token 均值目标：world/N 对齐后续一次 replica mean，无额外 SP 除数。
            coefficient = local_tokens.sum() * self.world_size / tokens.numel() * (step + 1)
            (parameter.to_local() * coefficient).sum().backward()
            for protected in (regular, expert, interleaved):
                protected.grad = DTensor.from_local(
                    torch.full_like(protected.to_local(), float(self.rank + 1)),
                    protected.device_mesh,
                    protected.placements,
                    run_check=False,
                )
            model.scale_and_reduce_grad()
            expected_gradient = float(tokens.mean()) * (step + 1)
            torch.testing.assert_close(
                parameter.grad.to_local(), torch.full_like(parameter.to_local(), expected_gradient), rtol=0, atol=0
            )
            torch.testing.assert_close(
                regular.grad.to_local(), torch.full_like(regular.to_local(), regular_mean), rtol=0, atol=0
            )
            torch.testing.assert_close(
                expert.grad.to_local(), torch.full_like(expert.to_local(), (self.rank + 1) / ep_size), rtol=0, atol=0
            )
            torch.testing.assert_close(
                interleaved.grad.to_local(), torch.full_like(interleaved.to_local(), self.rank + 1), rtol=0, atol=0
            )
            optimizer.step()
            expected_parameter -= 0.01 * expected_gradient
            torch.testing.assert_close(
                parameter.to_local(), torch.full_like(parameter.to_local(), expected_parameter), rtol=0, atol=1e-7
            )
            replicas = [torch.empty_like(parameter.to_local()) for _ in range(self.world_size)]
            dist.all_gather(replicas, parameter.to_local())
            for replica in replicas:
                torch.testing.assert_close(replica, replicas[0], rtol=0, atol=0)

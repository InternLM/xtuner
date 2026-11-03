"""PPO actor/critic loss parity between SP=1 and sequence-parallel training.

The PPO training engine consumes the loss builder's local context, runs the head
forward/backward and applies one optimizer update. This test fixes the full,
unsplit data and checks that the SP-split path (``build(data, sp_mesh)``) yields
the same loss, the same global head gradient and the same parameter after one SGD
step as the SP=1 reference computed from the same full data.

Covered behavior:
- actor policy loss (clipped PPO / GRPO objective) with rollout logprobs and
  advantages.
- critic clipped value loss with returns and frozen old values.
- odd sequence length (SP padding), cross-sample ignore regions and a packed
  boundary inside the shard split.
"""

import os
import unittest

import parametrize
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.loss.critic_loss import CriticLossConfig
from xtuner.v1.loss.utils import sp_split
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.utils.test_utils import init_data_mesh


HIDDEN_SIZE = 8
VOCAB_SIZE = 32
IGNORE_IDX = -100
CLIPRANGE = 0.2


def _broadcast(data: dict, device: str) -> dict:
    for key, value in data.items():
        value = value.to(device)
        dist.broadcast(value, src=0)
        data[key] = value
    return data


def _full_case(seq_len: int = 33) -> dict:
    torch.manual_seed(20261103)
    labels = torch.randint(0, VOCAB_SIZE, (1, seq_len))
    # Semantic holes and a packed boundary inside the split region.
    labels[:, 0:3] = IGNORE_IDX
    labels[:, seq_len // 2 : seq_len // 2 + 2] = IGNORE_IDX
    return {
        "shifted_labels": labels,
        "advantages": torch.randn(1, seq_len),
        "rollout_logprobs": -torch.rand(1, seq_len) * 2,
        "returns": torch.randn(1, seq_len),
        "old_values": torch.randn(1, seq_len),
        "hidden": torch.randn(1, seq_len, HIDDEN_SIZE) * 0.5,
        "head_weight": torch.randn(VOCAB_SIZE, HIDDEN_SIZE),
        "value_weight": torch.randn(1, HIDDEN_SIZE),
        "value_hidden": torch.randn(1, seq_len, HIDDEN_SIZE) * 0.5,
    }


def _sp_view(full: dict, mesh) -> dict:
    return {
        "shifted_labels": sp_split(full["shifted_labels"], sp_mesh=mesh, split_dim=1, padding_value=IGNORE_IDX),
        "advantages": sp_split(full["advantages"], sp_mesh=mesh, split_dim=1, padding_value=0.0),
        "rollout_logprobs": sp_split(full["rollout_logprobs"], sp_mesh=mesh, split_dim=1, padding_value=0.0),
        "returns": sp_split(full["returns"], sp_mesh=mesh, split_dim=1, padding_value=0.0),
        "old_values": sp_split(full["old_values"], sp_mesh=mesh, split_dim=1, padding_value=0.0),
        "hidden": sp_split(full["hidden"], sp_mesh=mesh, split_dim=1, padding_value=0.0),
        "value_hidden": sp_split(full["value_hidden"], sp_mesh=mesh, split_dim=1, padding_value=0.0),
    }


def _actor_step(full: dict, *, mesh=None, local: dict | None = None, lr: float = 0.1) -> tuple:
    cfg = GRPOLossConfig(
        policy_loss_cfg={"loss_type": "vanilla", "cliprange_low": CLIPRANGE, "cliprange_high": CLIPRANGE},
        loss_reduction="token",
    )
    head = full["head_weight"].detach().clone().requires_grad_(True)
    if mesh is None:
        data = {
            "shifted_labels": full["shifted_labels"],
            "advantages": full["advantages"],
            "rollout_logprobs": full["rollout_logprobs"],
        }
        ctx = cfg.build(data=data)
        ctx.loss_kwargs.old_logprobs = full["rollout_logprobs"].detach().clone()
        hidden = full["hidden"].detach().clone().requires_grad_(True)
    else:
        ctx = cfg.build(
            data={
                "shifted_labels": full["shifted_labels"],
                "advantages": full["advantages"],
                "rollout_logprobs": full["rollout_logprobs"],
            },
            sp_mesh=mesh,
        )
        ctx.loss_kwargs.old_logprobs = sp_split(full["rollout_logprobs"], sp_mesh=mesh, split_dim=1, padding_value=0.0)
        hidden = local["hidden"].detach().clone().requires_grad_(True)
    ctx = ctx.__class__.build_batches([ctx])[0]
    if mesh is None:
        # 参照路径在每个 rank 上都持有完整 batch；抵消 build_batches 的跨 rank 归约，
        # 得到与 SP 分片总和相同的全局分母。
        assert ctx.loss_kwargs.policy_loss_weight is not None
        ctx.loss_kwargs.policy_loss_weight = ctx.loss_kwargs.policy_loss_weight * dist.get_world_size()
    loss, _ = ctx.forward(hidden, head, None)
    if mesh is None:
        # 参照路径在每个 rank 上重复完整 batch；forward 的 SUM all_reduce 会按 world 放大，
        # 这里还原成单份全局 loss（梯度同样按 world 还原）。
        loss = loss / dist.get_world_size()
    loss.backward()
    head_grad = head.grad.detach().clone()
    if mesh is not None:
        dist.all_reduce(head_grad, op=dist.ReduceOp.AVG)
    with torch.no_grad():
        head -= lr * head_grad
    return loss.detach(), head_grad, head.detach()


def _critic_step(full: dict, *, mesh=None, local: dict | None = None, lr: float = 0.1) -> tuple:
    cfg = CriticLossConfig(cliprange_value=CLIPRANGE)
    head = full["value_weight"].detach().clone().requires_grad_(True)
    data = {
        "shifted_labels": full["shifted_labels"],
        "returns": full["returns"],
        "old_values": full["old_values"],
    }
    ctx = cfg.build(data=data, sp_mesh=mesh, device="cuda")
    ctx = ctx.__class__.build_batches([ctx])[0]
    if mesh is None:
        assert ctx.loss_kwargs.loss_weight is not None
        ctx.loss_kwargs.loss_weight = ctx.loss_kwargs.loss_weight * dist.get_world_size()
    hidden = (full["value_hidden"] if mesh is None else local["value_hidden"]).detach().clone().requires_grad_(True)
    loss, _ = ctx.forward(hidden, head, None)
    if mesh is None:
        # 参照路径在每个 rank 上重复完整 batch；forward 的 SUM all_reduce 会按 world 放大，
        # 这里还原成单份全局 loss（梯度同样按 world 还原）。
        loss = loss / dist.get_world_size()
    loss.backward()
    head_grad = head.grad.detach().clone()
    if mesh is not None:
        dist.all_reduce(head_grad, op=dist.ReduceOp.AVG)
    with torch.no_grad():
        head -= lr * head_grad
    return loss.detach(), head_grad, head.detach()


def _assert_close_all_ranks(actual: torch.Tensor, expected: torch.Tensor, name: str, tol: float = 1e-5) -> None:
    error = (actual.float() - expected.float()).abs().max()
    metrics = torch.tensor([error], device=actual.device)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    assert metrics[0].item() < tol, f"{name}: max rank error {metrics[0].item()}"


class TestPPOEngineSPParity(DistributedTestBase):
    @parametrize.parametrize("kind", ["actor", "critic"])
    def test_sp_loss_grad_and_update_parity(self, kind: str) -> None:
        device = "cuda"
        pg = self.create_pg(device)
        sp = self.world_size
        if sp < 2:
            raise unittest.SkipTest("SP parity tests need sp >= 2")
        mesh = init_data_mesh(device, sp)["sp"]

        full = _broadcast(_full_case(), device)
        local = _sp_view(full, mesh)
        step = _actor_step if kind == "actor" else _critic_step
        ref_loss, ref_grad, ref_head = step(full, mesh=None)
        sp_loss, sp_grad, sp_head = step(full, mesh=mesh, local=local)

        _assert_close_all_ranks(sp_loss, ref_loss, f"{kind} loss")
        _assert_close_all_ranks(sp_grad, ref_grad, f"{kind} head grad")
        _assert_close_all_ranks(sp_head, ref_head, f"{kind} updated head")
        dist.destroy_process_group(pg)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))

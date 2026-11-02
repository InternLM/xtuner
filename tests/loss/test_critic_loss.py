"""Critic loss 单测。

覆盖的行为:
- TestCriticLossShapes
    - test_mse_loss_shapes: MSE 分支在各 (B,S) 形状下的 loss 数值与 head 梯度
    - test_clipped_loss_shapes: clipped 分支在各 (B,S) 形状下的 loss 数值与 head 梯度
- TestCriticLossSP
    - test_sp_value_loss_parity: SP 分片与 SP=1 参照的 loss / head 梯度 / hidden 梯度对齐
      (奇数长度、跨样本 ignore、空 rank、多 microbatch 共用全局分母,MSE / clipped 两分支)
"""

import os
import unittest

import parametrize
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.loss.critic_loss import CriticLossConfig, CriticLossContext, CriticLossKwargs
from xtuner.v1.loss.utils import sp_split
from xtuner.v1.utils.test_utils import init_data_mesh

CLIPRANGE_VALUE = 0.2
HIDDEN_SIZE = 8
IGNORE_IDX = -100


class TestCriticLossShapes(unittest.TestCase):
    """监督字段形状对齐:回归 S==1 时 squeeze 吞掉序列维导致跨样本广播的问题(F2)。"""

    def _run_loss_fn(self, batch: int, seq_len: int, with_old_values: bool) -> None:
        torch.manual_seed(0)
        hidden_size = 4
        hidden = torch.randn(batch, seq_len, hidden_size)
        head_weight = torch.randn(1, hidden_size, requires_grad=True)
        shifted_labels = torch.zeros(batch, seq_len, dtype=torch.long)
        returns = torch.randn(batch, seq_len, 1)  # keep a trailing dim to exercise reshape_as
        loss_weight = torch.rand(batch, seq_len, 1)

        old_values = torch.randn(batch, seq_len, 1) if with_old_values else None
        loss_kwargs = CriticLossKwargs(
            shifted_labels=shifted_labels,
            returns=returns,
            old_values=old_values,
            loss_weight=loss_weight,
        )
        loss_ctx = CriticLossContext(CriticLossConfig(cliprange_value=CLIPRANGE_VALUE), loss_kwargs)

        out = loss_ctx.loss_fn(hidden, head_weight, None, loss_kwargs)

        v = torch.nn.functional.linear(hidden, head_weight.detach()).float().squeeze(-1)
        returns = returns.reshape_as(v)
        loss_weight = loss_weight.reshape_as(v)
        if not with_old_values:
            expected = (0.5 * (v - returns) ** 2 * loss_weight).sum()
        else:
            old = old_values.reshape_as(v)
            v_clip = old + (v - old).clamp(-CLIPRANGE_VALUE, CLIPRANGE_VALUE)
            expected = (0.5 * torch.maximum((v - returns) ** 2, (v_clip - returns) ** 2) * loss_weight).sum()

        self.assertEqual(out[0].shape, ())
        torch.testing.assert_close(out[0], expected)
        out[0].backward()
        self.assertIsNotNone(head_weight.grad)
        self.assertTrue(torch.isfinite(head_weight.grad).all())

    @parametrize.parametrize("batch, seq_len", [(1, 32), (2, 1), (2, 3)])
    def test_mse_loss_shapes(self, batch: int, seq_len: int) -> None:
        # MSE 分支:S==1 时监督字段不得被 squeeze 成 [B] 后与预测交叉广播
        self._run_loss_fn(batch, seq_len, with_old_values=False)

    @parametrize.parametrize("batch, seq_len", [(1, 32), (2, 1), (2, 3)])
    def test_clipped_loss_shapes(self, batch: int, seq_len: int) -> None:
        # clipped 分支:old_values 同样按 reshape_as 对齐,两分支形状行为一致
        self._run_loss_fn(batch, seq_len, with_old_values=True)


def _make_case_datas(case: str) -> list[dict]:
    """Build full (unsplit) tensors for a named SP parity case.

    Args:
        case (str): One of "odd_length", "cross_sample", "empty_rank", "microbatches".

    Returns:
        list[dict]: One dict per microbatch with full-sequence shifted_labels / returns /
        old_values / hidden / head_weight.
    """
    torch.manual_seed(1234)

    def _one(seq_len: int, ignore_slices: list[tuple[int, int]]) -> dict:
        labels = torch.randint(0, 100, (1, seq_len))
        for start, end in ignore_slices:
            labels[:, start:end] = IGNORE_IDX
        return {
            "shifted_labels": labels,
            "returns": torch.randn(1, seq_len),
            "old_values": torch.randn(1, seq_len),
            "hidden": torch.randn(1, seq_len, HIDDEN_SIZE),
            "head_weight": torch.randn(1, HIDDEN_SIZE),
        }

    if case == "odd_length":
        # 长度不被 sp 整除:最后一个 rank 分片带 padding
        return [_one(33, [(0, 5)])]
    if case == "cross_sample":
        # ignore 段落在分片边界区域,模拟 packed sample 边界
        return [_one(32, [(10, 14), (22, 25)])]
    if case == "empty_rank":
        # 后半全 ignore:sp=2 时某个 rank 的本地 token 全部无效
        return [_one(32, [(16, 32)])]
    if case == "microbatches":
        # 多个 microbatch 共用一个全局分母
        return [_one(33, [(0, 5)]), _one(32, [(10, 14), (22, 25)])]
    raise ValueError(case)


def _broadcast_datas(datas: list[dict], device: str) -> list[dict]:
    # 确保 world_size 张卡数据完全一致(NCCL 只支持 CUDA 张量,先搬设备再广播)
    for data in datas:
        for key, value in data.items():
            data[key] = value.to(device)
            dist.broadcast(data[key], src=0)
    return datas


def _reference(kind: str, datas: list[dict], device: str) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """SP=1 参照:完整序列上手工计算 loss 与梯度,不经过任何集合通信。

    Args:
        kind (str): "mse" or "clipped".
        datas (list[dict]): Full-sequence case data, one dict per microbatch.
        device (str): Compute device.

    Returns:
        tuple: (total loss, head weight grad, per-microbatch full hidden grads).
    """
    head_weight = datas[0]["head_weight"].detach().clone().requires_grad_(True)
    total = torch.zeros((), device=device)
    hidden_grads = []
    for data in datas:
        hidden = data["hidden"].detach().clone().requires_grad_(True)
        returns = data["returns"].to(device)
        old_values = data["old_values"].to(device)
        mask = (data["shifted_labels"] != IGNORE_IDX).to(device).float()
        v = F.linear(hidden, head_weight).float().squeeze(-1)
        if kind == "mse":
            per_token = 0.5 * (v - returns) ** 2
        else:
            v_clip = old_values + (v - old_values).clamp(-CLIPRANGE_VALUE, CLIPRANGE_VALUE)
            per_token = 0.5 * torch.maximum((v - returns) ** 2, (v_clip - returns) ** 2)
        # 全局 token 归一化:所有 microbatch 共用一个分母
        loss = (per_token * mask).sum() / sum(int((d["shifted_labels"] != IGNORE_IDX).sum()) for d in datas)
        total = total + loss
        grads = torch.autograd.grad(loss, hidden, retain_graph=True)[0]
        hidden_grads.append(grads)
    total.backward()
    return total.detach(), head_weight.grad.detach(), [g.detach() for g in hidden_grads]


def _run_sp(kind: str, datas: list[dict], device: str, mesh) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """SP 分片路径:走 CriticLossKwargs.sp_split → build_batches → forward 的公开协议。

    Args:
        kind (str): "mse" or "clipped".
        datas (list[dict]): Full-sequence case data, one dict per microbatch.
        device (str): Compute device.
        mesh: SP device mesh covering the whole world.

    Returns:
        tuple: (world-reduced loss, global head grad, per-microbatch local hidden grads).
    """
    loss_cfg = CriticLossConfig(cliprange_value=CLIPRANGE_VALUE)
    head_weight = datas[0]["head_weight"].detach().clone().requires_grad_(True)

    ctx_list = []
    hidden_shards = []
    for data in datas:
        loss_kwargs = CriticLossKwargs(
            shifted_labels=data["shifted_labels"].to(device),
            returns=data["returns"].to(device),
            old_values=data["old_values"].to(device) if kind == "clipped" else None,
        )
        loss_kwargs.sp_split(mesh)
        hidden_shard = sp_split(data["hidden"].to(device), sp_mesh=mesh, split_dim=1, padding_value=0.0)
        hidden_shard.requires_grad_(True)
        hidden_shards.append(hidden_shard)
        ctx_list.append(CriticLossContext(loss_cfg, loss_kwargs))

    ctx_list = CriticLossContext.build_batches(ctx_list)

    total = torch.zeros((), device=device)
    for ctx, hidden_shard in zip(ctx_list, hidden_shards):
        loss, _ = ctx.forward(hidden_shard, head_weight, None)
        total = total + loss
    total.backward()
    # autograd all_reduce 的 backward 会再规约一次梯度,各 rank 持有 world_size 倍本地贡献;
    # 真实训练由 FSDP 平均消掉该因子,这里用 AVG 等价还原全局梯度(与 test_oreal_loss 相同约定)
    head_grad = head_weight.grad.detach().clone()
    dist.all_reduce(head_grad, op=dist.ReduceOp.AVG)
    world_size = dist.get_world_size()
    return (
        total.detach(),
        head_grad,
        [shard.grad.detach() / world_size for shard in hidden_shards],
    )


def _assert_parity(actual: torch.Tensor, expected: torch.Tensor, valid_len: int | None, name: str) -> None:
    """所有 rank 一起断言,避免单 rank 失败后其他 rank 卡在集合通信上。

    Args:
        actual (torch.Tensor): SP-shard tensor.
        expected (torch.Tensor): Matching slice of the SP=1 reference.
        valid_len (int | None): Non-padded length of the shard (None for replicated tensors).
        name (str): Assertion label.
    """
    if valid_len is not None:
        actual = actual[:, :valid_len]
    error = (actual.float() - expected.float()).abs().max()
    metrics = torch.tensor([error], device=actual.device)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    assert metrics[0].item() < 1e-5, f"{name}: max rank error {metrics[0].item()}"


class TestCriticLossSP(DistributedTestBase):
    """SP parity:SP 分片与 SP=1 参照在 loss / head 梯度 / hidden 梯度上逐一对齐(F4)。"""

    @parametrize.parametrize(
        "kind, case",
        [
            ("mse", "odd_length"),
            ("mse", "cross_sample"),
            ("mse", "empty_rank"),
            ("mse", "microbatches"),
            ("clipped", "odd_length"),
            ("clipped", "cross_sample"),
            ("clipped", "empty_rank"),
            ("clipped", "microbatches"),
        ],
    )
    def test_sp_value_loss_parity(self, kind: str, case: str) -> None:
        # SP 分片后的全局 loss/梯度必须与 SP=1 完整序列的计算结果一致
        device = "cuda"
        pg = self.create_pg(device)
        sp = self.world_size
        if sp < 2:
            raise unittest.SkipTest("SP parity tests need sp >= 2")
        mesh = init_data_mesh(device, sp)["sp"]
        rank = dist.get_rank()

        datas = _broadcast_datas(_make_case_datas(case), device)
        ref_loss, ref_head_grad, ref_hidden_grads = _reference(kind, datas, device)
        sp_loss, sp_head_grad, sp_hidden_grads = _run_sp(kind, datas, device, mesh)

        _assert_parity(sp_loss, ref_loss, None, f"{kind}/{case} loss")
        _assert_parity(sp_head_grad, ref_head_grad, None, f"{kind}/{case} head grad")
        for i, (shard_grad, ref_grad) in enumerate(zip(sp_hidden_grads, ref_hidden_grads)):
            local_len = shard_grad.shape[1]
            valid_len = min(local_len, ref_grad.shape[1] - rank * local_len)
            _assert_parity(
                shard_grad,
                ref_grad[:, rank * local_len : rank * local_len + valid_len],
                valid_len,
                f"{kind}/{case} hidden grad {i}",
            )
        dist.destroy_process_group(pg)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))

"""Critic value-clip 计数与 finalize 派生测试。

Good Tests:
- 用手工张量精确控制 old_values 与 values 的偏差，断言 clip/valid 计数的逐 token 结果。
- 覆盖 cliprange 边界（严格大于才算 clipped）与 ignore_idx 掩码。
- 空 rank（全 ignore）保持可反传零损失，并在 old_values 存在时发零计数，保持归约 rank 对称。
- finalize 无 dist 环境下的派生与 config 委托。
- ModelForwardExtraLogInfo 白名单 round-trip：新计数 key 能被求和保留。

Bad Tests:
- 不构造模型/engine；除全掩码零损失回归外，不重复 GPU 侧 test_critic_sft.py 的数值覆盖。
"""

import unittest

import torch

from xtuner.v1.loss.critic_loss import (
    CriticLossConfig,
    CriticLossContext,
    CriticLossKwargs,
    finalize_critic_value_metrics,
)
from xtuner.v1.model.utils import ModelForwardExtraLogInfo


def _loss_fn_with_values(
    values: list[float],
    old_values: list[float] | None,
    labels: list[int],
    cliprange_value: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Run ``CriticLossContext.loss_fn`` with head_weight=1 so values equal
    hidden_states, returning the emitted extra_info."""
    seq_len = len(values)
    loss_cfg = CriticLossConfig(cliprange_value=cliprange_value)
    loss_kwargs = CriticLossKwargs(
        shifted_labels=torch.tensor([labels], dtype=torch.long),
        returns=torch.zeros(1, seq_len, dtype=torch.float32),
        old_values=None if old_values is None else torch.tensor([old_values], dtype=torch.float32).unsqueeze(-1),
    )
    # production 中由 build_batches 填；计数只要求非 None。
    loss_kwargs.loss_weight = torch.ones(1, seq_len, dtype=torch.float32)
    loss_ctx = CriticLossContext(loss_cfg, loss_kwargs)
    hidden_states = torch.tensor([values], dtype=torch.float32).unsqueeze(-1)
    head_weight = torch.ones(1, 1)
    _, (_, extra_info) = loss_ctx.loss_fn(hidden_states, head_weight, None, loss_kwargs)
    return extra_info


class TestCriticValueClipCounts(unittest.TestCase):
    def test_clip_counts_match_hand_computed_tokens(self) -> None:
        # 行为：clip 判据是 |values - old_values| > cliprange（严格大于），且只计 valid token
        extra_info = _loss_fn_with_values(
            values=[0.5, 0.1, 0.9, 0.2, 1.0],
            old_values=[0.0, 0.0, 0.0, 0.0, 1.05],
            labels=[5, 6, -100, 7, 8],
        )
        # t0: |0.5| > 0.2 clipped; t1 unclipped; t2 clipped but ignored (-100);
        # t3: |0.2| == 0.2 exactly -> not clipped; t4: |−0.05| < 0.2 unclipped.
        self.assertEqual(extra_info["reduced_critic_value_clip_count"].item(), 1.0)
        self.assertEqual(extra_info["reduced_critic_value_valid_count"].item(), 4.0)

    def test_unclipped_sft_path_emits_no_counters(self) -> None:
        # 行为：old_values 缺失（SFT MSE 路径）不产生 clip 计数，与 actor 缺 key 行为一致
        extra_info = _loss_fn_with_values(values=[0.5, 0.1], old_values=None, labels=[5, 6])
        self.assertEqual(extra_info, {})

    def test_all_ignored_tokens_still_emit_zero_counters(self) -> None:
        # 回归：全 ignore 的空 rank（old_values 存在）也必须发出计数 key（值为 0）。
        # 若缺失，finalize 的 all-reduce 会按 rank 条件跳过，有 token 的 rank 在
        # NCCL 上永久等待直至 watchdog 击杀 worker（smoke 两连挂的根因）。
        loss_ctx = CriticLossConfig().build(
            {
                "shifted_labels": torch.tensor([[-100]]),
                # Finite targets whose squared error overflows must not affect an ignored pack.
                "returns": torch.tensor([[1e20]]),
                "old_values": torch.tensor([[0.0]]),
            },
            device="cpu",
        )
        assert loss_ctx is not None
        CriticLossContext.build_batches([loss_ctx])
        hidden_states = torch.tensor([[[0.5]]], requires_grad=True)
        head_weight = torch.ones(1, 1, requires_grad=True)
        loss, (_, extra_info) = loss_ctx.forward(hidden_states, head_weight)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(hidden_states.grad, torch.zeros_like(hidden_states))
        torch.testing.assert_close(head_weight.grad, torch.zeros_like(head_weight))
        self.assertEqual(extra_info["reduced_critic_value_clip_count"].item(), 0.0)
        self.assertEqual(extra_info["reduced_critic_value_valid_count"].item(), 0.0)

    def test_mse_path_all_ignored_emits_no_counters(self) -> None:
        # 行为：old_values 缺失（SFT MSE 路径）且全 ignore 时不产生计数
        extra_info = _loss_fn_with_values(values=[0.5], old_values=None, labels=[-100])
        self.assertEqual(extra_info, {})


class TestFinalizeCriticValueMetrics(unittest.TestCase):
    def test_derives_clip_frac_without_dist(self) -> None:
        extra_info = {
            "reduced_critic_value_clip_count": 2.0,
            "reduced_critic_value_valid_count": 10.0,
        }
        finalized = finalize_critic_value_metrics(extra_info, "cpu")
        self.assertEqual(finalized["reduced_critic_value_clip_count"], 2.0)
        self.assertEqual(finalized["reduced_critic_value_valid_count"], 10.0)
        self.assertAlmostEqual(finalized["reduced_critic_value_clip_frac"], 0.2)

    def test_zero_valid_count_guards_division(self) -> None:
        finalized = finalize_critic_value_metrics(
            {"reduced_critic_value_clip_count": 0.0, "reduced_critic_value_valid_count": 0.0},
            "cpu",
        )
        self.assertEqual(finalized["reduced_critic_value_clip_frac"], 0.0)

    def test_missing_keys_pass_through_unchanged(self) -> None:
        # 行为：非 critic 字典（SFT 路径）原样返回
        extra_info = {"reduced_llm_loss": 1.5}
        self.assertEqual(finalize_critic_value_metrics(extra_info, "cpu"), {"reduced_llm_loss": 1.5})

    def test_config_finalize_metrics_delegates(self) -> None:
        finalized = CriticLossConfig().finalize_metrics(
            {"reduced_critic_value_clip_count": 3.0, "reduced_critic_value_valid_count": 4.0},
            "cpu",
        )
        self.assertAlmostEqual(finalized["reduced_critic_value_clip_frac"], 0.75)


class TestExtraLogInfoWhitelistRoundTrip(unittest.TestCase):
    def test_clip_counters_survive_micro_batch_reduction(self) -> None:
        # 行为：白名单内的计数跨 micro-batch 求和保留，白名单外的 key 被丢弃
        log_info = ModelForwardExtraLogInfo()
        log_info.append(
            {
                "reduced_critic_value_clip_count": torch.tensor(1.0),
                "reduced_critic_value_valid_count": torch.tensor(3.0),
                "reduced_critic_value_bogus": torch.tensor(7.0),
            }
        )
        log_info.append(
            {
                "reduced_critic_value_clip_count": torch.tensor(2.0),
                "reduced_critic_value_valid_count": torch.tensor(9.0),
            }
        )
        reduced = log_info.get()
        self.assertEqual(reduced["reduced_critic_value_clip_count"], 3.0)
        self.assertEqual(reduced["reduced_critic_value_valid_count"], 12.0)
        self.assertNotIn("reduced_critic_value_bogus", reduced)


if __name__ == "__main__":
    unittest.main()

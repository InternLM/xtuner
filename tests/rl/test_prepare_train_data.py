"""RLTrainer._prepare_train_data 的 PR-fast contract 测试。

本文件只测试训练数据构造的纯逻辑，不启动 trainer、Ray worker、模型或 rollout backend。
当前测试点：
- 文本样本的 input_ids、shifted_labels、rollout_logprobs、advantage 布局。
- 同一个 prompt 下多个 response 各自使用对应 reward / advantage。
- VLM 样本使用 train_prompt_ids，并保留 multimodal 训练字段。
- VLM M-RoPE：get_train_seq_ctx 用 global amax 续写 response position（对齐 SFT get_rope_index_3）。
- 无效 rollout group 会被跳过。
- 缺失 reward、logprob/mask 长度不一致、pack_max_length 过小时 fail fast。
- 多条样本 pack 成一条序列后，input_ids/shifted_labels/advantages/rollout_logprobs（及 teacher 字段）
  与输入位置逐一对齐（regression: advantage 曾比 input_ids 长 1 导致 pack 后整体错位）。
"""

import unittest
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status, TeacherTargets, reset_rollout_response
from xtuner.v1.rl.distillation import DistillationConfig, DistillationTrainerAdapter, RolloutTeacherConfig
from xtuner.v1.rl.loss import DistillationLossConfig
from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.train.rl_trainer import BaseRLTrainer, get_train_seq_ctx


class _FakeAdvantageEstimator:
    def __init__(self, values: list[float]):
        self.values = values
        self.calls = []

    def compute(self, rewards_tensor, group):
        self.calls.append((rewards_tensor.clone(), group))
        return torch.tensor(self.values[: len(group)], dtype=torch.float32)


class TestPrepareTrainData(unittest.TestCase):
    def _build_trainer(self, advantages: list[float]):
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer._advantage_estimator = _FakeAdvantageEstimator(advantages)
        trainer._distillation = DistillationTrainerAdapter(None)
        trainer.tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[999]])})
        trainer.logger = MagicMock()
        return trainer

    def _state(
        self,
        *,
        uid: int = 1,
        group_id: int = 1,
        prompt_ids: list[int] | None = None,
        response_ids: list[int] | torch.Tensor | None = None,
        logprobs: list[float] | None = None,
        response_mask: list[int] | None = None,
        reward: dict | None = None,
        status: Status = Status.COMPLETED,
        response: str = "response",
        routed_experts=None,
        position_ids: np.ndarray | None = None,
        mm_info: dict | None = None,
        extra_fields: dict | None = None,
        input_ids: list[int] | None = None,
        labels: list[int] | None = None,
        teacher_tokens: list[int] | list[list[int]] | None = None,
        teacher_logprobs: list[float] | list[list[float]] | None = None,
    ) -> RolloutState:
        return RolloutState(
            rollout_id=uid,
            group_id=group_id,
            message=[{"role": "user", "content": f"prompt {group_id}"}],
            prompt_ids=prompt_ids if prompt_ids is not None else [10, 11, 12],
            response=response,
            response_ids=response_ids if response_ids is not None else [20, 21, 22],
            logprobs=logprobs,
            response_mask=response_mask,
            reward=reward if reward is not None else {"score": 1.0},
            status=status,
            finish_reason="stop" if status == Status.COMPLETED else "error",
            routed_experts=routed_experts,
            position_ids=position_ids,
            mm_info=mm_info,
            extra_fields=extra_fields or {},
            input_ids=input_ids,
            labels=labels,
            teacher_targets=(
                TeacherTargets(
                    kind="topk" if teacher_tokens and isinstance(teacher_tokens[0], list) else "sampled",
                    tokens=teacher_tokens,
                    logprobs=teacher_logprobs,
                )
                if teacher_tokens is not None and teacher_logprobs is not None
                else None
            ),
        )

    def _prepare(self, trainer, data_groups, pack_max_length=128):
        with patch("xtuner.v1.train.rl_trainer.XTUNER_DETERMINISTIC", True):
            return trainer._prepare_train_data(data_groups, pack_max_length=pack_max_length)

    @staticmethod
    def _enable_rollout_distillation(trainer, loss_config: DistillationLossConfig) -> None:
        trainer._distillation = DistillationTrainerAdapter(
            DistillationConfig(
                loss_config=loss_config,
                teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
                data_source_teacher_map={"agent_math": "teacher"},
            )
        )

    def test_text_path_builds_shifted_training_tensors(self):
        # 文本主路径固定 token 布局：input_ids 去掉 response 最后一个 token，label/logprob 对齐预测位置。
        trainer = self._build_trainer([1.5])
        routed_experts = np.array([[1, 2], [3, 4]])
        state = self._state(
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 22],
            logprobs=[0.1, 0.2, 0.3],
            response_mask=[1, 0, 1],
            reward={"score": 1.0},
            routed_experts=routed_experts,
        )

        data_batches, info = self._prepare(trainer, [[state]])

        self.assertEqual(len(data_batches), 1)
        batch = data_batches[0]
        self.assertEqual(batch["seq_ctx"].input_ids.tolist(), [[10, 11, 12, 20, 21]])
        self.assertEqual(batch["shifted_labels"].tolist(), [[-100, -100, 20, -100, 22]])
        torch.testing.assert_close(
            batch["rollout_logprobs"],
            torch.tensor([[0.0, 0.0, 0.1, 0.2, 0.3]], dtype=torch.float32),
        )
        # advantage 与 shifted_labels 逐位置对齐: prompt 段为 0, mask=0 处为 0。
        self.assertEqual(batch["advantage"], [0.0, 0.0, 1.5, 0.0, 1.5])
        self.assertEqual(len(batch["advantage"]), batch["shifted_labels"].numel())
        self.assertIs(batch["seq_ctx"].rollout_routed_experts, routed_experts)
        self.assertEqual(info["training_samples"], 1)
        self.assertEqual(info["training_tokens"], 5)
        self.assertEqual(info["rewards/mean"], 1.0)
        self.assertEqual(info["response_len/mean"], 3.0)
        self.assertEqual(info["prompt_len/mean"], 3.0)

    def test_rerolled_state_without_semantic_mask_uses_all_response_tokens(self):
        trainer = self._build_trainer([1.0])
        state = reset_rollout_response(self._state(response_mask=[0, 1, 0]))
        state.response = "rerolled response"
        state.response_ids = [30, 31]
        state.logprobs = [0.1, 0.2]
        state.reward = {"score": 1.0}
        state.status = Status.COMPLETED
        state.finish_reason = "stop"

        data_batches, _ = self._prepare(trainer, [[state]])

        self.assertIsNone(state.response_mask)
        self.assertEqual(data_batches[0]["shifted_labels"].tolist(), [[-100, -100, 30, 31]])
        self.assertEqual(data_batches[0]["advantage"], [0.0, 0.0, 1.0, 1.0])

    def test_multi_sample_group_uses_each_sample_reward_and_advantage(self):
        # 同一个 prompt 下的多个 response 要分别使用自己的 reward 和 advantage。
        trainer = self._build_trainer([1.5, -2.0])
        first = self._state(uid=1, response_ids=[20, 21], reward={"score": 3.0})
        second = self._state(uid=2, response_ids=[30, 31], reward={"score": -1.0})

        data_batches, info = self._prepare(trainer, [[first, second]])

        self.assertEqual(len(data_batches), 2)
        self.assertEqual(data_batches[0]["advantage"], [0.0, 0.0, 1.5, 1.5])
        self.assertEqual(data_batches[1]["advantage"], [0.0, 0.0, -2.0, -2.0])
        self.assertEqual(info["batch_size"], 2)
        self.assertEqual(info["rewards/min"], -1.0)
        self.assertEqual(info["rewards/max"], 3.0)
        self.assertEqual(info["rewards/mean"], 1.0)
        self.assertEqual(info["advantages/min"], -2.0)
        self.assertEqual(info["advantages/max"], 1.5)
        self.assertEqual(trainer._advantage_estimator.calls[0][0].tolist(), [3.0, -1.0])

    def test_advantage_stats_count_only_loss_active_tokens(self):
        # advantages/mean|min|max 只统计 loss-active token: prompt 占位与 mask=0 的 token 不参与。
        trainer = self._build_trainer([2.0, -1.0])
        plain = self._state(
            uid=1,
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 22, 23],
            logprobs=[0.1, 0.2, 0.3, 0.4],
            response_mask=[1, 0, 1, 0],
            reward={"score": 2.0},
        )
        agentic = self._state(
            uid=2,
            input_ids=[30, 31, 40, 41, 42],
            labels=[-100, -100, 40, -100, 42],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
            reward={"score": -1.0},
        )

        _, info = self._prepare(trainer, [[plain, agentic]])

        # plain: 2 个 mask!=0 token 计入 2.0; agentic: 2 个 label!=-100 token 计入 -1.0。
        self.assertEqual(info["advantages/mean"], 0.5)
        self.assertEqual(info["advantages/min"], -1.0)
        self.assertEqual(info["advantages/max"], 2.0)

    def test_vlm_path_uses_train_prompt_ids_and_preserves_multimodal_fields(self):
        # VLM 分支使用 extra_fields["train_prompt_ids"] 作为训练 prompt，并把图像字段带进 SequenceContext。
        trainer = self._build_trainer([0.25])
        pixel_values = np.ones((1, 2, 3), dtype=np.float32)
        image_grid_thw = np.array([[1, 2, 3]], dtype=np.int32)
        position_ids = np.array([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=np.int64)
        state = self._state(
            prompt_ids=[1],
            response_ids=[102, 103],
            response_mask=[1, 1],
            position_ids=position_ids,
            mm_info={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            extra_fields={"train_prompt_ids": [100, 101]},
        )

        data_batches, _ = self._prepare(trainer, [[state]])

        seq_ctx = data_batches[0]["seq_ctx"]
        self.assertEqual(seq_ctx.input_ids.tolist(), [[100, 101, 102]])
        self.assertEqual(tuple(seq_ctx.position_ids.shape), (3, 1, 3))
        self.assertEqual(seq_ctx.position_ids.dtype, torch.long)
        self.assertIs(seq_ctx.pixel_values, pixel_values)
        self.assertEqual(seq_ctx.image_grid_thw.dtype, torch.long)
        self.assertEqual(seq_ctx.image_grid_thw.tolist(), [[1, 2, 3]])

    def test_get_train_seq_ctx_mrope_continues_from_global_amax(self):
        """RL 只拿 prompt 的 3D position，续写 response 须用 global amax（对齐 SFT get_rope_index_3）。

        Fixture 等价于：prompt 以 image tokens 结尾（grid 1x4x4, merge=2 → 2x2），
        此时 T 轴 max=2、H/W 轴 max=3，per-axis max 会在 T 轴分叉；SFT 用 max(T,H,W)+1 续写。
        不直接 import get_rope_index_3：本文件走 lightweight datasets stub。
        """
        # prompt positions from get_rope_index_3([text, vision_start, 4 img tokens])
        prompt_position_ids = np.array(
            [
                [[0, 1, 2, 2, 2, 2]],  # T
                [[0, 1, 2, 2, 3, 3]],  # H
                [[0, 1, 2, 3, 2, 3]],  # W
            ],
            dtype=np.int64,
        )
        response_ids = [20, 21, 22]
        full_ids = torch.tensor([[10, 11, 12, 13, 14, 15] + response_ids], dtype=torch.long)
        # SFT get_rope_index_3 整段结果：response 从 global amax(=3)+1 起
        expected = torch.tensor(
            [
                [[0, 1, 2, 2, 2, 2, 4, 5, 6]],
                [[0, 1, 2, 2, 3, 3, 4, 5, 6]],
                [[0, 1, 2, 3, 2, 3, 4, 5, 6]],
            ],
            dtype=torch.long,
        )

        seq_ctx = get_train_seq_ctx(
            cast(torch.LongTensor, full_ids),
            prompt_position_ids,
            len_response_ids=len(response_ids),
        )
        rl_position_ids = seq_ctx.position_ids
        assert rl_position_ids is not None
        self.assertEqual(tuple(rl_position_ids.shape), (3, 1, 9))
        torch.testing.assert_close(rl_position_ids, expected)
        # 回归：若误用 per-axis max，T 轴会变成 3,4,5 而非 4,5,6
        self.assertFalse(torch.equal(rl_position_ids[0, 0, -3:], torch.tensor([3, 4, 5])))

    def test_mixed_agentic_and_vlm_reasoning_use_3d_position_ids(self):
        trainer = self._build_trainer([0.5])
        reasoning_state = self._state(
            uid=1,
            group_id=1,
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 22],
            position_ids=np.arange(3, dtype=np.int64).reshape(1, 1, -1).repeat(3, axis=0),
        )
        agentic_state = self._state(
            uid=2,
            group_id=2,
            input_ids=[30, 31, 40, 41, 42],
            labels=[-100, -100, 40, 41, 42],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
        )

        data_batches, _ = self._prepare(trainer, [[reasoning_state], [agentic_state]])

        self.assertEqual(tuple(data_batches[0]["seq_ctx"].position_ids.shape), (3, 1, 5))
        agentic_position_ids = data_batches[1]["seq_ctx"].position_ids
        self.assertEqual(tuple(agentic_position_ids.shape), (3, 1, 4))
        torch.testing.assert_close(
            agentic_position_ids,
            torch.arange(4, dtype=torch.long).reshape(1, 1, -1).expand(3, -1, -1),
        )

    def test_agentic_topk_targets_include_token_ids_and_logprobs(self):
        loss_config = DistillationLossConfig(
            policy_loss_cfg={
                "loss_type": "vanilla",
                "cliprange_low": 0.2,
                "cliprange_high": 0.2,
            },
            loss_mode="reverse",
            use_policy_gradient=False,
            top_k=2,
        )
        trainer = self._build_trainer([0.0])
        self._enable_rollout_distillation(trainer, loss_config)
        state = self._state(
            input_ids=[10, 11, 20, 21, 22],
            labels=[-100, -100, 20, -100, 22],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
            teacher_tokens=[[100, 101], [102, 103], [104, 105]],
            teacher_logprobs=[[-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]],
            extra_fields={"origin_data_source": "agent_math"},
        )

        data_batches, _ = self._prepare(trainer, [[state]])

        self.assertEqual(len(data_batches), 1)
        batch = data_batches[0]
        self.assertEqual(batch["shifted_labels"].tolist(), [[-100, 20, -100, 22]])
        self.assertEqual(
            batch["target_token_ids"].tolist(),
            [[[0, 0], [100, 101], [102, 103], [104, 105]]],
        )
        torch.testing.assert_close(
            batch["teacher_logprobs"],
            torch.tensor(
                [[[0.0, 0.0], [-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]]],
                dtype=torch.float32,
            ),
        )
        with patch("xtuner.v1.rl.loss.distillation_loss.DEVICE", "cpu"):
            loss_ctx = loss_config.build(
                {
                    "shifted_labels": batch["shifted_labels"],
                    "advantages": torch.tensor([batch["advantage"]], dtype=torch.float32),
                    "old_logprobs": torch.zeros_like(batch["shifted_labels"], dtype=torch.float32),
                    "teacher_logprobs": batch["teacher_logprobs"],
                    "target_token_ids": batch["target_token_ids"],
                }
            )
        assert loss_ctx is not None
        type(loss_ctx).build_batches([loss_ctx])
        loss, _ = loss_ctx.loss_fn(
            hidden_states=torch.randn(1, 4, 8),
            head_weight=torch.randn(128, 8),
            head_bias=None,
            loss_kwargs=loss_ctx.loss_kwargs,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_plain_topk_targets_include_masked_response_rows(self):
        loss_config = DistillationLossConfig(
            policy_loss_cfg={"loss_type": "vanilla"},
            loss_mode="forward_kl_topk",
            use_policy_gradient=False,
            top_k=2,
        )
        trainer = self._build_trainer([0.0])
        self._enable_rollout_distillation(trainer, loss_config)
        state = self._state(
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 22],
            response_mask=[0, 1, 1],
            teacher_tokens=[[102, 103], [104, 105]],
            teacher_logprobs=[[-0.7, -0.8], [-0.9, -1.0]],
            extra_fields={"origin_data_source": "agent_math"},
        )

        data_batches, _ = self._prepare(trainer, [[state]])

        batch = data_batches[0]
        self.assertEqual(batch["shifted_labels"].tolist(), [[-100, -100, -100, 21, 22]])
        self.assertEqual(
            batch["target_token_ids"].tolist(),
            [[[0, 0], [0, 0], [0, 0], [102, 103], [104, 105]]],
        )
        torch.testing.assert_close(
            batch["teacher_logprobs"],
            torch.tensor(
                [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-0.7, -0.8], [-0.9, -1.0]]],
                dtype=torch.float32,
            ),
        )

    def test_sampled_token_targets_align_for_plain_and_agentic_rollouts(self):
        loss_config = DistillationLossConfig(
            policy_loss_cfg={"loss_type": "vanilla"},
            loss_mode="k1",
            use_policy_gradient=True,
        )
        trainer = self._build_trainer([0.0, 0.0])
        self._enable_rollout_distillation(trainer, loss_config)
        plain_state = self._state(
            uid=1,
            group_id=1,
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 22],
            response_mask=[0, 1, 1],
            teacher_tokens=[21, 22],
            teacher_logprobs=[-0.7, -0.9],
            extra_fields={"origin_data_source": "agent_math"},
        )
        agentic_state = self._state(
            uid=2,
            group_id=2,
            input_ids=[30, 31, 40, 41, 42],
            labels=[-100, -100, 40, -100, 42],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
            teacher_tokens=[40, 41, 42],
            teacher_logprobs=[-1.1, -1.2, -1.3],
            extra_fields={"origin_data_source": "agent_math"},
        )

        data_batches, _ = self._prepare(trainer, [[plain_state], [agentic_state]])

        self.assertEqual(len(data_batches), 2)
        torch.testing.assert_close(
            data_batches[0]["teacher_logprobs"],
            torch.tensor([[0.0, 0.0, 0.0, -0.7, -0.9]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            data_batches[1]["teacher_logprobs"],
            torch.tensor([[0.0, -1.1, -1.2, -1.3]], dtype=torch.float32),
        )
        self.assertNotIn("target_token_ids", data_batches[0])
        self.assertNotIn("target_token_ids", data_batches[1])

    def test_invalid_group_is_skipped(self):
        # FAILED/FILTERED/ABORTED group 不能进入训练 batch，也不能贡献训练样本数。
        trainer = self._build_trainer([1.0])
        valid = self._state(uid=1, response_ids=[20, 21], reward={"score": 2.0})
        failed = self._state(uid=2, status=Status.FAILED, response_ids=[30, 31], reward={"score": 4.0})

        data_batches, info = self._prepare(trainer, [[valid], [failed]])

        self.assertEqual(len(data_batches), 1)
        self.assertEqual(info["training_samples"], 1)
        trainer.logger.error.assert_called_once()

    def test_missing_reward_score_fails_fast(self):
        # reward 必须包含 score，否则 advantage 计算前后语义都不明确。
        trainer = self._build_trainer([1.0])
        state = self._state(reward={"other": 1.0})

        with self.assertRaisesRegex(ValueError, "missing.*score"):
            self._prepare(trainer, [[state]])

    def test_logprobs_must_match_response_ids_length(self):
        # rollout logprobs 和 response_ids 必须逐 token 对齐。
        trainer = self._build_trainer([1.0])
        state = self._state(response_ids=[20, 21, 22], logprobs=[0.1, 0.2])

        with self.assertRaises(AssertionError):
            self._prepare(trainer, [[state]])

    def test_response_mask_must_match_response_ids_length(self):
        # response_mask 参与 label 和 advantage mask，长度不一致时必须直接失败。
        trainer = self._build_trainer([1.0])
        state = self._state(response_ids=[20, 21, 22], response_mask=[1, 0])

        with self.assertRaises(AssertionError):
            self._prepare(trainer, [[state]])

    def test_input_ids_must_not_exceed_pack_max_length(self):
        # pack_max_length 过小时要在进入 packing 前失败，避免后续训练侧报错难定位。
        trainer = self._build_trainer([1.0])
        state = self._state(prompt_ids=[10, 11, 12], response_ids=[20, 21, 22])

        with self.assertRaises(AssertionError):
            self._prepare(trainer, [[state]], pack_max_length=4)


class TestPrepareTrainDataPackAlignment(unittest.TestCase):
    """多条样本 pack 成一条序列后, pg_loss 消费前的张量逐位置对齐。

    regression: advantage 曾按 `[adv]*len(prompt_ids) + [mask处理]` 构造, 比 input_ids 长 1,
    pack 拼接后 advantages 与 shifted_labels/rollout_logprobs 整体错位或形状不匹配。
    """

    def _make_trainer(self, with_teacher_fields: bool = False) -> BaseRLTrainer:
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer.logger = MagicMock()
        distillation = MagicMock()
        distillation.task_adv_weight = 1.0
        if with_teacher_fields:
            # 与真实批次一致的逐位置 teacher 字段形状 (1, seq_len, k) / (1, seq_len)。
            distillation.rollout_teacher_targets.side_effect = lambda state, shifted_labels: {
                "teacher_logprobs": torch.zeros(1, len(shifted_labels), 2),
                "target_token_ids": torch.zeros(1, len(shifted_labels), 2),
                "teacher_indices": torch.full((1, len(shifted_labels)), -1, dtype=torch.int64),
            }
        else:
            distillation.rollout_teacher_targets.return_value = {}
        distillation.reward_scalars.return_value = {}
        trainer._distillation = distillation
        trainer._advantage_estimator = MagicMock()
        # 确定性 advantage: advantage = reward - 0.5, 每个样本值唯一且 float32 精确。
        trainer._advantage_estimator.compute.side_effect = lambda rewards, representatives: rewards - 0.5
        return trainer

    def _make_sample(
        self,
        rollout_id: int,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        logprobs: list[float],
        reward: float,
    ) -> RolloutState:
        return RolloutState(
            rollout_id=rollout_id,
            message=[],
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            logprobs=logprobs,
            response="ok",
            reward={"score": reward},
            status=Status.COMPLETED,
        )

    @staticmethod
    def _expected_segment(
        sample: RolloutState, adv_val: float
    ) -> tuple[list[int], list[int], list[float], list[float]]:
        """单个样本 pack 前的正确布局: input_ids 去掉 response 末位(EOS 只作 label)。"""
        assert sample.prompt_ids is not None
        assert sample.response_ids is not None
        assert sample.response_mask is not None
        assert sample.logprobs is not None
        prompt_len = len(sample.prompt_ids)
        input_ids = list(sample.prompt_ids) + list(sample.response_ids)[:-1]
        labels = [-100] * (prompt_len - 1) + [
            resp_id if mask != 0 else -100 for resp_id, mask in zip(sample.response_ids, sample.response_mask)
        ]
        advantages = [0.0] * (prompt_len - 1) + [0.0 if mask == 0 else adv_val for mask in sample.response_mask]
        logprobs = [0.0] * (prompt_len - 1) + list(sample.logprobs)
        return input_ids, labels, advantages, logprobs

    @staticmethod
    def _pack_and_extract(data_batches: list[dict], pack_max_length: int) -> dict:
        controller = TrainingController.__new__(TrainingController)
        packed_batches = controller._packing(data_batches, pack_max_length, None)
        assert len(packed_batches) == 1, "all samples are expected to fit into a single pack"
        packed = packed_batches[0]
        result = {
            "input_ids": packed["seq_ctx"].input_ids.squeeze(0).tolist(),
            "labels": packed["shifted_labels"].squeeze(0).tolist(),
            "advantages": packed["advantages"].squeeze(0).tolist(),
            "rollout_logprobs": packed["rollout_logprobs"].squeeze(0).tolist(),
        }
        for key in ("teacher_logprobs", "target_token_ids", "teacher_indices"):
            if packed.get(key) is not None:
                result[key] = packed[key].squeeze(0).tolist()
        return result

    def _build_data_groups(self) -> list[list[RolloutState]]:
        """两组样本; 组内共享 prompt(与真实 RL 组一致), 长度/mask/advantage 刻意互不相同。"""
        s1 = self._make_sample(
            0,
            [101, 102, 103, 104],
            [1001, 1002, 1003, 1004, 1005],
            [1, 1, 0, 1, 1],
            [-0.5, -1.0, -1.5, -2.0, -2.5],
            1.0,
        )
        s2 = self._make_sample(1, [101, 102, 103, 104], [2001, 2002, 2003], [1, 0, 1], [-0.5, -1.0, -1.5], 2.0)
        s3 = self._make_sample(
            2, [301, 302, 303, 304, 305], [3001, 3002, 3003, 3004], [1, 1, 1, 0], [-0.5, -1.0, -1.5, -2.0], 4.0
        )
        s4 = self._make_sample(3, [301, 302, 303, 304, 305], [4001, 4002], [1, 0], [-0.5, -1.0], 8.0)
        return [[s1, s2], [s3, s4]]

    def _prepare(self, trainer, data_groups, pack_max_length: int):
        with patch("xtuner.v1.train.rl_trainer.XTUNER_DETERMINISTIC", True):
            return trainer._prepare_train_data(data_groups, pack_max_length=pack_max_length)

    def _assert_real_token_alignment(
        self, samples: list[RolloutState], packed: dict, total_len: int, packed_len: int
    ) -> None:
        self.assertEqual(len(packed["input_ids"]), packed_len)
        self.assertEqual(len(packed["labels"]), packed_len)
        self.assertEqual(len(packed["advantages"]), packed_len)
        self.assertEqual(len(packed["rollout_logprobs"]), packed_len)

        offset = 0
        for sample in samples:
            adv_val = sample.reward["score"] - 0.5
            exp_ids, exp_labels, exp_advs, exp_logprobs = self._expected_segment(sample, adv_val)
            seg = slice(offset, offset + len(exp_ids))
            self.assertEqual(packed["input_ids"][seg], exp_ids)
            self.assertEqual(packed["labels"][seg], exp_labels)
            self.assertEqual(packed["advantages"][seg], exp_advs)
            self.assertEqual(packed["rollout_logprobs"][seg], exp_logprobs)
            offset += len(exp_ids)
        self.assertEqual(offset, total_len)

    def test_packed_fields_align_without_padding(self) -> None:
        trainer = self._make_trainer()
        data_groups = self._build_data_groups()
        samples = [sample for group in data_groups for sample in group]
        total_len = sum(len(s.prompt_ids) + len(s.response_ids) - 1 for s in samples)

        data_batches, _ = self._prepare(trainer, data_groups, pack_max_length=total_len)

        self.assertEqual(len(data_batches), len(samples))
        for item in data_batches:
            self.assertEqual(len(item["advantage"]), item["shifted_labels"].numel())

        packed = self._pack_and_extract(data_batches, total_len)
        self._assert_real_token_alignment(samples, packed, total_len, total_len)

    def test_packed_fields_align_with_padding(self) -> None:
        trainer = self._make_trainer()
        data_groups = self._build_data_groups()
        samples = [sample for group in data_groups for sample in group]
        total_len = sum(len(s.prompt_ids) + len(s.response_ids) - 1 for s in samples)
        pack_max_length = (total_len // 16 + 1) * 16

        data_batches, _ = self._prepare(trainer, data_groups, pack_max_length=pack_max_length)
        packed = self._pack_and_extract(data_batches, pack_max_length)

        self._assert_real_token_alignment(samples, packed, total_len, pack_max_length)
        # pad 区: label 为 -100, advantage 为 controller 的 pad 值 -100。
        self.assertTrue(all(label == -100 for label in packed["labels"][total_len:]))
        self.assertTrue(all(adv == -100 for adv in packed["advantages"][total_len:]))

    def test_packed_fields_align_with_teacher_fields(self) -> None:
        trainer = self._make_trainer(with_teacher_fields=True)
        data_groups = self._build_data_groups()
        samples = [sample for group in data_groups for sample in group]
        total_len = sum(len(s.prompt_ids) + len(s.response_ids) - 1 for s in samples)

        data_batches, _ = self._prepare(trainer, data_groups, pack_max_length=total_len)
        packed = self._pack_and_extract(data_batches, total_len)

        self._assert_real_token_alignment(samples, packed, total_len, total_len)
        self.assertEqual(len(packed["teacher_logprobs"]), total_len)
        self.assertEqual(len(packed["target_token_ids"]), total_len)
        self.assertEqual(len(packed["teacher_indices"]), total_len)


if __name__ == "__main__":
    unittest.main()

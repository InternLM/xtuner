"""RLTrainer._prepare_train_data 的 PR-fast contract 测试。

本文件只测试训练数据构造的纯逻辑，不启动 trainer、Ray worker、模型或 rollout backend。
当前测试点：
- 文本样本的 input_ids、shifted_labels、rollout_logprobs、advantage 布局。
- 同一个 prompt 下多个 response 各自使用对应 reward / advantage。
- VLM 样本使用 train_prompt_ids，并保留 multimodal 训练字段。
- 无效 rollout group 会被跳过。
- 缺失 reward、logprob/mask 长度不一致、pack_max_length 过小时 fail fast。

注意：当前训练 contract 中 data_dict["advantage"] 比 shifted_labels 多 1 个元素；
metric 统计使用 actual_advantages[:-1]，测试会显式固定这个行为。
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.train.rl_trainer import BaseRLTrainer


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
        )

    def _prepare(self, trainer, data_groups, pack_max_length=128):
        with patch("xtuner.v1.train.rl_trainer.XTUNER_DETERMINISTIC", True):
            return trainer._prepare_train_data(data_groups, pack_max_length=pack_max_length)

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
        self.assertEqual(batch["advantage"], [1.5, 1.5, 1.5, 1.5, 0.0, 1.5])
        self.assertEqual(len(batch["advantage"]), batch["shifted_labels"].numel() + 1)
        self.assertIs(batch["seq_ctx"].rollout_routed_experts, routed_experts)
        self.assertEqual(info["training_samples"], 1)
        self.assertEqual(info["training_tokens"], 5)
        self.assertEqual(info["rewards/mean"], 1.0)
        self.assertEqual(info["response_len/mean"], 3.0)
        self.assertEqual(info["prompt_len/mean"], 3.0)

    def test_multi_sample_group_uses_each_sample_reward_and_advantage(self):
        # 同一个 prompt 下的多个 response 要分别使用自己的 reward 和 advantage。
        trainer = self._build_trainer([1.5, -2.0])
        first = self._state(uid=1, response_ids=[20, 21], reward={"score": 3.0})
        second = self._state(uid=2, response_ids=[30, 31], reward={"score": -1.0})

        data_batches, info = self._prepare(trainer, [[first, second]])

        self.assertEqual(len(data_batches), 2)
        self.assertEqual(data_batches[0]["advantage"], [1.5, 1.5, 1.5, 1.5, 1.5])
        self.assertEqual(data_batches[1]["advantage"], [-2.0, -2.0, -2.0, -2.0, -2.0])
        self.assertEqual(info["batch_size"], 2)
        self.assertEqual(info["rewards/min"], -1.0)
        self.assertEqual(info["rewards/max"], 3.0)
        self.assertEqual(info["rewards/mean"], 1.0)
        self.assertEqual(info["advantages/min"], -2.0)
        self.assertEqual(info["advantages/max"], 1.5)
        self.assertEqual(trainer._advantage_estimator.calls[0][0].tolist(), [3.0, -1.0])

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

        with self.assertRaises(AssertionError):
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


if __name__ == "__main__":
    unittest.main()

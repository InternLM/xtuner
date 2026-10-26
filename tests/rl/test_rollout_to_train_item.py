"""RolloutState -> TrainSample 转换的 contract 测试。

``TrainingWorker._get_train_sample``（seq_ctx/labels/logprobs/teacher targets）与
``TrainingWorker._add_sample_fields``（advantage 暂存）是 RolloutState -> 训练样本
的唯一入口：shift、scalar advantage 逐监督位广播、rollout logprob shift、position 布局统一、
seq_ctx 构造、teacher target 对齐在这一步完成、advantage 由 _add_sample_fields 写入。本文件只测该
纯转换逻辑，不启动 Ray、模型或 rollout backend。

当前测试点：
- 文本样本：input_ids 去尾、shifted_labels、rollout_logprobs shift 与形状对齐。
- scalar advantage 广播：prompt 占位与 labels=-100 的语义洞位为 0，监督位为 group advantage。
- use_3d_position_ids=True 时文本样本广播出三轴 MRoPE 位置；False 时保持 1D。
- VLM 样本：prompt 段 3D position 续写 response、mm_info/pixel_values 透传、routed_experts 挂到 seq_ctx。
- 蒸馏：rollout teacher target 对齐结果进入 sample.teacher_targets（sampled 无 target_token_ids，topk 有）。
- token advantage 与 shifted labels 等长，语义洞与 prompt 不参与监督。
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status, TeacherTargets
from xtuner.v1.rl.distillation import DistillationConfig, DistillationTrainerAdapter, RolloutTeacherConfig
from xtuner.v1.rl.loss import DistillationLossConfig, GRPOLossConfig
from xtuner.v1.rl.trainer.actor_worker import ActorWorker
from xtuner.v1.rl.trainer.pack import TokenBatchProto
from xtuner.v1.rl.trainer.worker import TrainingWorker


def _convert(worker, states, advantages, use_3d=False):
    samples = [worker._get_train_sample(state, use_3d) for state in states]
    worker._cached_samples = samples
    if advantages and isinstance(advantages[0], list):
        advantages = TokenBatchProto.concat([torch.tensor(row, dtype=torch.float32) for row in advantages])
    worker._add_sample_fields("advantages", advantages)
    return samples


class TestConvertRolloutItemsToTrainItems(unittest.TestCase):
    def _make_worker(self, distillation: DistillationTrainerAdapter | None = None) -> TrainingWorker:
        # 生产契约：蒸馏开启时 train_worker_cfg.loss_cfg 即 distillation_config.loss_config。
        worker = TrainingWorker.__new__(TrainingWorker)
        worker.config = SimpleNamespace(
            pack_max_length=16,
            loss_cfg=GRPOLossConfig(policy_loss_cfg={"loss_type": "vanilla"}),
        )
        actor = ActorWorker.__new__(ActorWorker)
        actor.config = worker.config
        actor._distillation = distillation if distillation is not None else DistillationTrainerAdapter(None)
        actor._staged = {}
        actor.mtp_config = None
        actor.host = worker
        worker.actor = actor
        return worker

    def _state(
        self,
        *,
        uid: int = 1,
        prompt_ids: list[int] | None = None,
        response_ids: list[int] | None = None,
        logprobs: list[float] | None = None,
        supervised_mask: list[int] | None = None,
        routed_experts=None,
        position_ids: np.ndarray | None = None,
        mm_info: dict | None = None,
        extra_fields: dict | None = None,
        input_ids: list[int] | None = None,
        labels: list[int] | None = None,
        teacher_tokens: list[int] | list[list[int]] | None = None,
        teacher_logprobs: list[float] | list[list[float]] | None = None,
    ) -> RolloutState:
        resolved_prompt_ids = prompt_ids if prompt_ids is not None else [10, 11, 12]
        resolved_response_ids = response_ids if response_ids is not None else [20, 21, 22]
        state = RolloutState(
            rollout_id=uid,
            group_id=1,
            message=[{"role": "user", "content": "prompt"}],
            prompt_ids=resolved_prompt_ids,
            response="response",
            response_ids=resolved_response_ids,
            logprobs=logprobs,
            status=Status.COMPLETED,
            finish_reason="stop",
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
        if input_ids is not None:
            return state
        state.input_ids = list(resolved_prompt_ids) + list(resolved_response_ids)
        if supervised_mask is None:
            state.labels = [-100] * len(resolved_prompt_ids) + list(resolved_response_ids)
        else:
            state.labels = [-100] * len(resolved_prompt_ids) + [
                resp_id if flag else -100 for resp_id, flag in zip(resolved_response_ids, supervised_mask)
            ]
        if state.logprobs is not None:
            state.logprobs = [0.0] * len(resolved_prompt_ids) + list(state.logprobs)
        return state

    def test_text_path_builds_shifted_rl_train_item(self):
        worker = self._make_worker()
        state = self._state(
            response_ids=[20, 21, 22],
            logprobs=[0.1, 0.2, 0.3],
            supervised_mask=[1, 0, 1],
        )

        samples = _convert(worker, [state], [1.5])

        self.assertEqual(len(samples), 1)
        seq_ctx = samples[0].seq_ctx
        sample = samples[0]
        self.assertEqual(seq_ctx.input_ids.tolist(), [[10, 11, 12, 20, 21]])
        self.assertEqual(sample.shifted_labels.tolist(), [[-100, -100, 20, -100, 22]])
        torch.testing.assert_close(
            sample.advantages,
            torch.tensor([[0.0, 0.0, 1.5, 0.0, 1.5]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            sample.rollout_logprobs,
            torch.tensor([[0.0, 0.0, 0.1, 0.2, 0.3]], dtype=torch.float32),
        )
        self.assertIsNone(sample.teacher_targets.get("teacher_logprobs"))
        self.assertIsNone(sample.teacher_targets.get("target_token_ids"))
        self.assertIsNone(sample.teacher_targets.get("teacher_indices"))
        for tensor in (sample.shifted_labels, sample.advantages, sample.rollout_logprobs):
            self.assertEqual(tensor.shape[1], seq_ctx.input_ids.shape[1])

    def test_advantages_follow_each_sample_scalar(self):
        worker = self._make_worker()
        first = self._state(uid=1, response_ids=[20, 21], supervised_mask=[1, 1])
        second = self._state(uid=2, response_ids=[30, 31], supervised_mask=[1, 0])

        samples = _convert(worker, [first, second], [1.5, -2.0])

        torch.testing.assert_close(
            samples[0].advantages,
            torch.tensor([[0.0, 0.0, 1.5, 1.5]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            samples[1].advantages,
            torch.tensor([[0.0, 0.0, -2.0, 0.0]], dtype=torch.float32),
        )

    def test_text_positions_stay_1d_without_3d_flag(self):
        worker = self._make_worker()
        state = self._state(response_ids=[20, 21])

        samples = _convert(worker, [state], [0.0])

        torch.testing.assert_close(samples[0].seq_ctx.position_ids, torch.arange(4).reshape(1, -1))

    def test_text_positions_broadcast_to_3d_in_mixed_batches(self):
        worker = self._make_worker()
        state = self._state(response_ids=[20, 21])

        samples = _convert(worker, [state], [0.0], use_3d=True)

        position_ids = samples[0].seq_ctx.position_ids
        self.assertEqual(tuple(position_ids.shape), (3, 1, 4))
        torch.testing.assert_close(
            position_ids,
            torch.arange(4, dtype=torch.long).reshape(1, 1, -1).expand(3, -1, -1),
        )

    def test_vlm_position_extension_and_multimodal_passthrough(self):
        worker = self._make_worker()
        pixel_values = np.ones((1, 2, 3), dtype=np.float32)
        image_grid_thw = np.array([[1, 2, 3]], dtype=np.int32)
        routed_experts = np.array([[1, 2], [3, 4]])
        state = self._state(
            prompt_ids=[100, 101],
            response_ids=[102, 103],
            position_ids=np.array([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=np.int64),
            mm_info={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            routed_experts=routed_experts,
        )

        samples = _convert(worker, [state], [0.25], use_3d=True)

        seq_ctx = samples[0].seq_ctx
        self.assertEqual(seq_ctx.input_ids.tolist(), [[100, 101, 102]])
        self.assertEqual(tuple(seq_ctx.position_ids.shape), (3, 1, 3))
        self.assertIs(seq_ctx.pixel_values, pixel_values)
        self.assertEqual(seq_ctx.image_grid_thw.tolist(), [[1, 2, 3]])
        self.assertIs(seq_ctx.rollout_routed_experts, routed_experts)

    def _enable_rollout_distillation(self, worker: TrainingWorker, loss_config: DistillationLossConfig) -> None:
        worker.actor._distillation = DistillationTrainerAdapter(
            DistillationConfig(
                loss_config=loss_config,
                teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
                data_source_teacher_map={"agent_math": "teacher"},
            )
        )
        worker.config.loss_cfg = loss_config

    def test_topk_teacher_targets_land_in_teacher_targets(self):
        worker = self._make_worker()
        self._enable_rollout_distillation(
            worker,
            DistillationLossConfig(
                policy_loss_cfg={"loss_type": "vanilla"},
                loss_mode="forward_kl_topk",
                use_policy_gradient=False,
                top_k=2,
            ),
        )
        state = self._state(
            input_ids=[10, 11, 20, 21, 22],
            labels=[-100, -100, 20, -100, 22],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
            teacher_tokens=[[100, 101], [102, 103], [104, 105]],
            teacher_logprobs=[[-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]],
            extra_fields={"origin_data_source": "agent_math"},
        )

        samples = _convert(worker, [state], [0.0])

        sample = samples[0]
        self.assertEqual(sample.shifted_labels.tolist(), [[-100, 20, -100, 22]])
        self.assertEqual(
            sample.teacher_targets["target_token_ids"].tolist(),
            [[[0, 0], [100, 101], [102, 103], [104, 105]]],
        )
        torch.testing.assert_close(
            sample.teacher_targets["teacher_logprobs"],
            torch.tensor([[[0.0, 0.0], [-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]]], dtype=torch.float32),
        )

    def test_sampled_teacher_targets_omit_target_token_ids(self):
        worker = self._make_worker()
        self._enable_rollout_distillation(
            worker,
            DistillationLossConfig(
                policy_loss_cfg={"loss_type": "vanilla"},
                loss_mode="k1",
                use_policy_gradient=True,
            ),
        )
        state = self._state(
            response_ids=[20, 21],
            logprobs=[0.1, 0.2],
            teacher_tokens=[20, 21],
            teacher_logprobs=[-0.7, -0.9],
            extra_fields={"origin_data_source": "agent_math"},
        )

        samples = _convert(worker, [state], [0.0])

        sample = samples[0]
        self.assertIsNone(sample.teacher_targets.get("target_token_ids"))
        torch.testing.assert_close(
            sample.teacher_targets["teacher_logprobs"],
            torch.tensor([[0.0, 0.0, -0.7, -0.9]], dtype=torch.float32),
        )

    def test_token_advantages_mask_prompt_and_semantic_holes(self):
        worker = self._make_worker()
        state = self._state(supervised_mask=[1, 0, 1])
        samples = _convert(worker, [state], [[1, 2, 3, 4, 5]])
        torch.testing.assert_close(samples[0].advantages, torch.tensor([[0.0, 0.0, 3.0, 0.0, 5.0]]))

    def test_token_advantages_require_shifted_length(self):
        worker = self._make_worker()
        with self.assertRaisesRegex(ValueError, "Token advantages length mismatch"):
            _convert(worker, [self._state()], [[1.0]])


if __name__ == "__main__":
    unittest.main()

"""RL 训练数据构造的 contract 测试（loop 侧 canonicalize + controller 侧转换）。

重构后训练数据构造分两段，本文件只测两段的公开纯逻辑，不启动 trainer、Ray worker、模型或
rollout backend：

- loop 侧 ``canonicalize_train_fields``：基类为 prompt+response 型样本构造统一全序列字段
  （``input_ids``/``labels``/``logprobs`` 等长、未 shift；``agent_loop_type`` 由 generate_group
  写入产出 loop 的类名）；localhost/sandbox 各自覆写为
  trace 全序列字段的校验（失败置 FAILED，不上抛）。
- controller 侧 ``TrainingController._convert_rollout_groups``：消费已 canonicalize 且 labels 已定稿
  （语义洞直接烙在 labels 中）的状态，完成 shift、张量化、组级 advantage、seq_ctx 构造与
  ``data_info`` 统计。

当前测试点：
- 文本样本的 input_ids、shifted_labels、rollout_logprobs、advantage 布局。
- 同一个 prompt 下多个 response 各自使用对应 reward / advantage。
- VLM 样本 loop 侧使用 train_prompt_ids 构造；controller 侧保留 multimodal 字段与 3D 位置。
- VLM M-RoPE：get_train_seq_ctx 用 global amax 续写 response position（对齐 SFT get_rope_index_3）。
- 无效 rollout group 会被跳过；缺 reward 在 task_adv_weight>0 时 fail fast，=0 时零 advantage 参训。
- 监督语义：labels 是唯一监督载体（语义洞 = labels 中的 -100）；controller 不做任何掩码处理。
- 多条样本 pack 成一条序列后，input_ids/shifted_labels/advantages/rollout_logprobs（及 teacher 字段）
  与输入位置逐一对齐（regression: advantage 曾比 input_ids 长 1 导致 pack 后整体错位）。
"""

import sys
import unittest
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    Status,
    TeacherTargets,
)
from xtuner.v1.datasets.mllm_tokenize_fn.qwenvl_rope2d import get_rope_index_3
from xtuner.v1.rl.distillation import DistillationConfig, DistillationTrainerAdapter, RolloutTeacherConfig
from xtuner.v1.rl.loss import DistillationLossConfig
from xtuner.v1.rl.trainer.controller import TrainingController, get_train_seq_ctx

# localhost/sandbox loop 模块顶层导入 lagent；测试环境不依赖其真实实现。
# rate_limiter 必须整路径 stub：父级 MagicMock 没有 __path__，子模块 from-import 无法透传。
for _stub_name in ("lagent", "lagent.utils", "lagent.utils.rate_limiter"):
    sys.modules.setdefault(_stub_name, MagicMock())

from xtuner.v1.rl.agent_loop.agent_loop import AgentLoop  # noqa: E402
from xtuner.v1.rl.agent_loop.localhost_agent_loop.agent_in_localhost_loop import AgentInLocalhostLoop  # noqa: E402
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import AgentInSandboxLoop  # noqa: E402


class _FakeAdvantageEstimator:
    def __init__(self, values: list[float]):
        self.values = values
        self.calls = []

    def compute(self, rewards_tensor, group):
        self.calls.append((rewards_tensor.clone(), group))
        return torch.tensor(self.values[: len(group)], dtype=torch.float32)


class _PromptResponseLoop(AgentLoop):
    """最小具体子类：single-turn 等prompt+response loop 通过继承获得基类 canonicalize 实现。"""

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        return rollout_state


class TestAgentLoopCanonicalizeTrainFields(unittest.TestCase):
    """基类默认实现：为 prompt+response 型样本构造统一全序列字段。"""

    def _make_loop(self) -> AgentLoop:
        loop = _PromptResponseLoop.__new__(_PromptResponseLoop)
        loop.logger = MagicMock()
        loop.tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[999, 998]])})
        return loop

    def _state(
        self,
        *,
        uid: int = 1,
        prompt_ids: list[int] | None = None,
        response_ids: list[int] | None = None,
        response: str | None = "response",
        logprobs: list[float] | None = None,
        status: Status = Status.COMPLETED,
        extra_fields: dict | None = None,
        input_ids: list[int] | None = None,
        labels: list[int] | None = None,
    ) -> RolloutState:
        return RolloutState(
            rollout_id=uid,
            group_id=1,
            message=[{"role": "user", "content": "prompt"}],
            prompt_ids=prompt_ids if prompt_ids is not None else [10, 11, 12],
            response=response,
            response_ids=response_ids if response_ids is not None else [20, 21, 22],
            logprobs=logprobs,
            status=status,
            finish_reason="stop" if status == Status.COMPLETED else "error",
            extra_fields=extra_fields or {},
            input_ids=input_ids,
            labels=labels,
        )

    def test_builds_full_sequence_fields(self):
        # prompt+response 样本构造统一约定：三者等长，labels 全监督，logprobs 前补 0。
        loop = self._make_loop()
        state = self._state(response_ids=[20, 21, 22], logprobs=[0.1, 0.2, 0.3])

        returned = loop.canonicalize_train_fields([state])

        self.assertIs(returned[0], state)
        self.assertEqual(state.input_ids, [10, 11, 12, 20, 21, 22])
        self.assertEqual(state.labels, [-100, -100, -100, 20, 21, 22])
        self.assertEqual(state.logprobs, [0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
        self.assertEqual(state.status, Status.COMPLETED)

    def test_tokenizer_fallback_when_response_ids_missing(self):
        loop = self._make_loop()
        state = self._state(response="ok")
        state.response_ids = None

        loop.canonicalize_train_fields([state])

        loop.tokenizer.assert_called_once_with("ok", return_tensors="pt")
        self.assertEqual(state.response_ids, [999, 998])
        self.assertEqual(state.input_ids, [10, 11, 12, 999, 998])
        self.assertEqual(state.labels, [-100, -100, -100, 999, 998])

    def test_flattens_tensor_response_ids(self):
        # rollout backend 可能返回 Tensor 形态的 response_ids，构造前需 flatten。
        loop = self._make_loop()
        state = self._state()
        state.response_ids = torch.tensor([[20, 21, 22]])

        loop.canonicalize_train_fields([state])

        self.assertEqual(state.response_ids, [20, 21, 22])
        self.assertEqual(state.input_ids, [10, 11, 12, 20, 21, 22])

    def test_vlm_uses_train_prompt_ids(self):
        # VLM 分支用 extra_fields["train_prompt_ids"] 作为训练 prompt 构造全序列。
        loop = self._make_loop()
        state = self._state(prompt_ids=[1], response_ids=[102, 103], extra_fields={"train_prompt_ids": [100, 101]})

        loop.canonicalize_train_fields([state])

        self.assertEqual(state.input_ids, [100, 101, 102, 103])
        self.assertEqual(state.labels, [-100, -100, 102, 103])

    def test_logprobs_stay_none_when_missing(self):
        loop = self._make_loop()
        state = self._state(logprobs=None)

        loop.canonicalize_train_fields([state])

        self.assertIsNone(state.logprobs)

    def test_agentic_state_with_input_ids_is_untouched(self):
        # 已有 input_ids 的全序列样本（agentic loop 产出）满足约定，零改动。
        loop = self._make_loop()
        state = self._state(input_ids=[30, 31, 40], labels=[-100, -100, 40], logprobs=[0.0, -0.1, -0.2])
        before = (list(state.input_ids), list(state.labels), list(state.logprobs))

        loop.canonicalize_train_fields([state])

        self.assertEqual((state.input_ids, state.labels, state.logprobs), before)
        loop.tokenizer.assert_not_called()

    def test_non_completed_state_is_skipped(self):
        loop = self._make_loop()
        state = self._state(status=Status.FAILED)

        loop.canonicalize_train_fields([state])

        self.assertIsNone(state.input_ids)
        self.assertIsNone(state.labels)
        self.assertEqual(state.status, Status.FAILED)

    def test_missing_prompt_ids_marks_sample_failed(self):
        # 构造失败容错：单个样本置 FAILED，不上抛到 producer。
        loop = self._make_loop()
        state = self._state()
        state.prompt_ids = None

        loop.canonicalize_train_fields([state])

        self.assertEqual(state.status, Status.FAILED)
        self.assertIn("canonicalize_train_fields failed", state.error_msg)
        loop.logger.error.assert_called_once()

    def test_failure_is_isolated_within_group(self):
        loop = self._make_loop()
        bad = self._state(uid=1)
        bad.prompt_ids = None
        good = self._state(uid=2)

        loop.canonicalize_train_fields([bad, good])

        self.assertEqual(bad.status, Status.FAILED)
        self.assertEqual(good.status, Status.COMPLETED)
        self.assertEqual(good.input_ids, [10, 11, 12, 20, 21, 22])


class TestAgenticLoopCanonicalizeTrainFields(unittest.TestCase):
    """localhost/sandbox 各自的覆写：trace 全序列字段只做统一约定校验。"""

    AGENTIC_LOOP_CLASSES = (AgentInLocalhostLoop, AgentInSandboxLoop)

    def _make_loop(self, loop_cls):
        loop = loop_cls.__new__(loop_cls)
        loop.logger = MagicMock()
        return loop

    def _agentic_state(
        self,
        *,
        uid: int = 1,
        status: Status = Status.COMPLETED,
        input_ids: list[int] | None = None,
        labels: list[int] | None = None,
        logprobs: list[float] | None = None,
    ) -> RolloutState:
        return RolloutState(
            rollout_id=uid,
            group_id=1,
            message=[{"role": "user", "content": "prompt"}],
            prompt_ids=[10, 11],
            response="ok",
            response_ids=[20, 21],
            status=status,
            finish_reason="stop",
            input_ids=input_ids,
            labels=labels,
            logprobs=logprobs,
        )

    def test_valid_full_sequence_state_is_unchanged(self):
        for loop_cls in self.AGENTIC_LOOP_CLASSES:
            with self.subTest(loop_cls=loop_cls.__name__):
                loop = self._make_loop(loop_cls)
                state = self._agentic_state(
                    input_ids=[30, 31, 40, 41, 42],
                    labels=[-100, -100, 40, 41, 42],
                    logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
                )

                returned = loop.canonicalize_train_fields([state])

                self.assertIs(returned[0], state)
                self.assertEqual(state.status, Status.COMPLETED)
                self.assertIsNone(state.error_msg)
                loop.logger.error.assert_not_called()

    def test_labels_length_mismatch_marks_sample_failed(self):
        for loop_cls in self.AGENTIC_LOOP_CLASSES:
            with self.subTest(loop_cls=loop_cls.__name__):
                loop = self._make_loop(loop_cls)
                state = self._agentic_state(input_ids=[30, 31, 40, 41, 42], labels=[-100, -100, 40, 41])

                loop.canonicalize_train_fields([state])

                self.assertEqual(state.status, Status.FAILED)
                self.assertIn("canonicalize_train_fields failed", state.error_msg)
                loop.logger.error.assert_called_once()

    def test_missing_labels_marks_sample_failed(self):
        for loop_cls in self.AGENTIC_LOOP_CLASSES:
            with self.subTest(loop_cls=loop_cls.__name__):
                loop = self._make_loop(loop_cls)
                state = self._agentic_state(input_ids=[30, 31, 40, 41, 42], labels=None)

                loop.canonicalize_train_fields([state])

                self.assertEqual(state.status, Status.FAILED)
                self.assertIn("labels length mismatch", state.error_msg)

    def test_logprobs_length_mismatch_marks_sample_failed(self):
        for loop_cls in self.AGENTIC_LOOP_CLASSES:
            with self.subTest(loop_cls=loop_cls.__name__):
                loop = self._make_loop(loop_cls)
                state = self._agentic_state(
                    input_ids=[30, 31, 40, 41, 42],
                    labels=[-100, -100, 40, 41, 42],
                    logprobs=[0.0, -0.1, -0.2],
                )

                loop.canonicalize_train_fields([state])

                self.assertEqual(state.status, Status.FAILED)
                self.assertIn("logprobs length mismatch", state.error_msg)

    def test_state_without_input_ids_is_skipped(self):
        # eval 态样本显式置空训练字段且 COMPLETED，不能被构造或校验。
        for loop_cls in self.AGENTIC_LOOP_CLASSES:
            with self.subTest(loop_cls=loop_cls.__name__):
                loop = self._make_loop(loop_cls)
                state = self._agentic_state()

                loop.canonicalize_train_fields([state])

                self.assertIsNone(state.input_ids)
                self.assertEqual(state.status, Status.COMPLETED)
                loop.logger.error.assert_not_called()

    def test_non_completed_state_is_skipped(self):
        for loop_cls in self.AGENTIC_LOOP_CLASSES:
            with self.subTest(loop_cls=loop_cls.__name__):
                loop = self._make_loop(loop_cls)
                state = self._agentic_state(status=Status.FAILED)

                loop.canonicalize_train_fields([state])

                self.assertIsNone(state.input_ids)
                self.assertEqual(state.status, Status.FAILED)
                loop.logger.error.assert_not_called()


class TestConvertRolloutGroups(unittest.TestCase):
    """TrainingController._convert_rollout_groups 合同：shift、advantage、张量与统计。"""

    def _build_controller(self, advantages: list[float], task_adv_weight: float = 1.0) -> TrainingController:
        controller = TrainingController.__new__(TrainingController)
        controller.advantage_estimator = _FakeAdvantageEstimator(advantages)
        controller.task_adv_weight = task_adv_weight
        controller.distillation = DistillationTrainerAdapter(None)
        controller.logger = MagicMock()
        return controller

    def _convert(self, controller, data_groups, pack_max_length=128):
        with patch("xtuner.v1.rl.trainer.controller.XTUNER_DETERMINISTIC", True):
            return controller._convert_rollout_groups(data_groups, pack_max_length)

    def _state(
        self,
        *,
        uid: int = 1,
        group_id: int = 1,
        prompt_ids: list[int] | None = None,
        response_ids: list[int] | torch.Tensor | None = None,
        logprobs: list[float] | None = None,
        supervised_mask: list[int] | None = None,
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
        agent_loop_type: str | None = None,
    ) -> RolloutState:
        resolved_prompt_ids = prompt_ids if prompt_ids is not None else [10, 11, 12]
        resolved_response_ids = response_ids if response_ids is not None else [20, 21, 22]
        state = RolloutState(
            rollout_id=uid,
            group_id=group_id,
            message=[{"role": "user", "content": f"prompt {group_id}"}],
            prompt_ids=resolved_prompt_ids,
            response=response,
            response_ids=resolved_response_ids,
            logprobs=logprobs,
            agent_loop_type=agent_loop_type,
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
        if input_ids is not None:
            # agentic 全序列样本由 trace store 直接提供字段，按原样返回。
            return state
        # prompt+response 样本：模拟 loop 侧 canonicalize 后的最终形态（语义洞直接烙在 labels）。
        resp_ids = list(resolved_response_ids)
        state.input_ids = list(resolved_prompt_ids) + resp_ids
        if supervised_mask is None:
            state.labels = [-100] * len(resolved_prompt_ids) + resp_ids
        else:
            state.labels = [-100] * len(resolved_prompt_ids) + [
                resp_id if flag else -100 for resp_id, flag in zip(resp_ids, supervised_mask)
            ]
        if state.logprobs is not None:
            state.logprobs = [0.0] * len(resolved_prompt_ids) + list(state.logprobs)
        return state

    @staticmethod
    def _enable_rollout_distillation(controller, loss_config: DistillationLossConfig) -> None:
        controller.distillation = DistillationTrainerAdapter(
            DistillationConfig(
                loss_config=loss_config,
                teachers=[RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"])],
                data_source_teacher_map={"agent_math": "teacher"},
            )
        )

    def test_text_path_builds_shifted_training_tensors(self):
        # 文本主路径固定 token 布局：input_ids 去掉 response 最后一个 token，label/logprob 对齐预测位置。
        controller = self._build_controller([1.5])
        routed_experts = np.array([[1, 2], [3, 4]])
        state = self._state(
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 22],
            logprobs=[0.1, 0.2, 0.3],
            supervised_mask=[1, 0, 1],
            reward={"score": 1.0},
            routed_experts=routed_experts,
        )

        data_batches, info = self._convert(controller, [[state]])

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

    def test_controller_consumes_labels_as_final_supervision(self):
        # controller 不处理任何掩码语义：labels 是什么就消费什么（语义洞已在生成期烙进 labels）。
        controller = self._build_controller([1.0])
        state = self._state(response_ids=[20, 21, 22], logprobs=[0.1, 0.2, 0.3], supervised_mask=[1, 0, 1])

        data_batches, _ = self._convert(controller, [[state]])

        self.assertEqual(data_batches[0]["shifted_labels"].tolist(), [[-100, -100, 20, -100, 22]])
        self.assertEqual(data_batches[0]["advantage"], [0.0, 0.0, 1.0, 0.0, 1.0])

    def test_multi_sample_group_uses_each_sample_reward_and_advantage(self):
        # 同一个 prompt 下的多个 response 要分别使用自己的 reward 和 advantage。
        controller = self._build_controller([1.5, -2.0])
        first = self._state(uid=1, response_ids=[20, 21], reward={"score": 3.0})
        second = self._state(uid=2, response_ids=[30, 31], reward={"score": -1.0})

        data_batches, info = self._convert(controller, [[first, second]])

        self.assertEqual(len(data_batches), 2)
        self.assertEqual(data_batches[0]["advantage"], [0.0, 0.0, 1.5, 1.5])
        self.assertEqual(data_batches[1]["advantage"], [0.0, 0.0, -2.0, -2.0])
        self.assertEqual(info["batch_size"], 2)
        self.assertEqual(info["rewards/min"], -1.0)
        self.assertEqual(info["rewards/max"], 3.0)
        self.assertEqual(info["rewards/mean"], 1.0)
        self.assertEqual(info["advantages/min"], -2.0)
        self.assertEqual(info["advantages/max"], 1.5)
        self.assertEqual(controller.advantage_estimator.calls[0][0].tolist(), [3.0, -1.0])

    def test_advantage_stats_count_only_loss_active_tokens(self):
        # advantages/mean|min|max 只统计 loss-active token: prompt 占位与 labels=-100 的 token 不参与。
        controller = self._build_controller([2.0, -1.0])
        plain = self._state(
            uid=1,
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 22, 23],
            logprobs=[0.1, 0.2, 0.3, 0.4],
            supervised_mask=[1, 0, 1, 0],
            reward={"score": 2.0},
        )
        agentic = self._state(
            uid=2,
            input_ids=[30, 31, 40, 41, 42],
            labels=[-100, -100, 40, -100, 42],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
            reward={"score": -1.0},
            agent_loop_type="AgentInLocalhostLoop",
        )

        _, info = self._convert(controller, [[plain, agentic]])

        # plain: 2 个 label!=-100 的 response token 计入 2.0; agentic: 同理计入 -1.0。
        self.assertEqual(info["advantages/mean"], 0.5)
        self.assertEqual(info["advantages/min"], -1.0)
        self.assertEqual(info["advantages/max"], 2.0)

    def test_vlm_fields_pass_through_with_3d_position_extension(self):
        # VLM 样本保留 multimodal 字段；3D position_ids 按 len_response_ids 补 response 段。
        controller = self._build_controller([0.25])
        pixel_values = np.ones((1, 2, 3), dtype=np.float32)
        image_grid_thw = np.array([[1, 2, 3]], dtype=np.int32)
        position_ids = np.array([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=np.int64)
        state = self._state(
            prompt_ids=[100, 101],
            response_ids=[102, 103],
            position_ids=position_ids,
            mm_info={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
        )

        data_batches, _ = self._convert(controller, [[state]])

        seq_ctx = data_batches[0]["seq_ctx"]
        self.assertEqual(seq_ctx.input_ids.tolist(), [[100, 101, 102]])
        self.assertEqual(tuple(seq_ctx.position_ids.shape), (3, 1, 3))
        self.assertEqual(seq_ctx.position_ids.dtype, torch.long)
        self.assertIs(seq_ctx.pixel_values, pixel_values)
        self.assertEqual(seq_ctx.image_grid_thw.dtype, torch.long)
        self.assertEqual(seq_ctx.image_grid_thw.tolist(), [[1, 2, 3]])

    def test_get_train_seq_ctx_mrope_continues_from_global_amax(self):
        """RL 只拿 prompt 的 3D position，续写 response 后须与 SFT get_rope_index_3 一致。

        Fixture：prompt 以 image tokens 结尾（grid 1x4x4, merge=2 → 2x2），使 per-axis max
        与 global amax 分叉。
        """
        image_token_id = 151655
        vision_start_token_id = 151652
        prompt_ids = [10, vision_start_token_id, image_token_id, image_token_id, image_token_id, image_token_id]
        response_ids = [20, 21, 22]
        full_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

        sft_position_ids = get_rope_index_3(
            full_ids,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=2,
        )
        prompt_position_ids = get_rope_index_3(
            torch.tensor([prompt_ids], dtype=torch.long),
            image_grid_thw=image_grid_thw,
            spatial_merge_size=2,
        )

        seq_ctx = get_train_seq_ctx(
            cast(torch.LongTensor, full_ids),
            prompt_position_ids.numpy(),
            len_response_ids=len(response_ids),
        )
        rl_position_ids = seq_ctx.position_ids
        assert rl_position_ids is not None
        self.assertEqual(tuple(rl_position_ids.shape), tuple(sft_position_ids.shape))
        torch.testing.assert_close(rl_position_ids, sft_position_ids)

    def test_mixed_agentic_and_vlm_reasoning_use_3d_position_ids(self):
        controller = self._build_controller([0.5])
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
            agent_loop_type="AgentInLocalhostLoop",
        )

        data_batches, _ = self._convert(controller, [[reasoning_state], [agentic_state]])

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
        controller = self._build_controller([0.0])
        self._enable_rollout_distillation(controller, loss_config)
        state = self._state(
            input_ids=[10, 11, 20, 21, 22],
            labels=[-100, -100, 20, -100, 22],
            logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
            teacher_tokens=[[100, 101], [102, 103], [104, 105]],
            teacher_logprobs=[[-0.5, -0.6], [-0.7, -0.8], [-0.9, -1.0]],
            extra_fields={"origin_data_source": "agent_math"},
            agent_loop_type="AgentInLocalhostLoop",
        )

        data_batches, _ = self._convert(controller, [[state]])

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
        controller = self._build_controller([0.0])
        self._enable_rollout_distillation(controller, loss_config)
        state = self._state(
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 22],
            supervised_mask=[0, 1, 1],
            teacher_tokens=[[102, 103], [104, 105]],
            teacher_logprobs=[[-0.7, -0.8], [-0.9, -1.0]],
            extra_fields={"origin_data_source": "agent_math"},
        )

        data_batches, _ = self._convert(controller, [[state]])

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
        controller = self._build_controller([0.0, 0.0])
        self._enable_rollout_distillation(controller, loss_config)
        plain_state = self._state(
            uid=1,
            group_id=1,
            prompt_ids=[10, 11, 12],
            response_ids=[20, 21, 22],
            supervised_mask=[0, 1, 1],
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
            agent_loop_type="AgentInLocalhostLoop",
        )

        data_batches, _ = self._convert(controller, [[plain_state], [agentic_state]])

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
        controller = self._build_controller([1.0])
        valid = self._state(uid=1, response_ids=[20, 21], reward={"score": 2.0})
        failed = self._state(uid=2, status=Status.FAILED, response_ids=[30, 31], reward={"score": 4.0})

        data_batches, info = self._convert(controller, [[valid], [failed]])

        self.assertEqual(len(data_batches), 1)
        self.assertEqual(info["training_samples"], 1)
        controller.logger.error.assert_called_once()

    def test_missing_reward_score_fails_fast(self):
        # reward 必须包含 score，否则 advantage 计算前后语义都不明确。
        controller = self._build_controller([1.0])
        state = self._state(reward={"other": 1.0})

        with self.assertRaisesRegex(ValueError, "missing.*score"):
            self._convert(controller, [[state]])

    def test_missing_reward_with_zero_task_adv_weight_trains_with_zero_advantage(self):
        # task_adv_weight=0 时（纯 OPD）缺 reward 不崩溃：样本以零 advantage 参训，不计入 rewards 统计。
        controller = self._build_controller([1.0], task_adv_weight=0.0)
        state = self._state()
        state.reward = None

        data_batches, info = self._convert(controller, [[state]])

        self.assertEqual(len(data_batches), 1)
        self.assertEqual(data_batches[0]["advantage"], [0.0] * 5)
        self.assertEqual(info["training_samples"], 1)
        self.assertEqual(info["rewards/mean"], 0.0)
        self.assertEqual(controller.advantage_estimator.calls, [])

    def test_group_with_mismatched_full_sequence_logprobs_is_skipped(self):
        # 全序列 logprobs 与 input_ids 不对齐的样本由组校验拦截：整组跳过，不进训练。
        controller = self._build_controller([1.0])
        state = self._state(
            input_ids=[30, 31, 40, 41, 42],
            labels=[-100, -100, 40, 41, 42],
            logprobs=[0.0, -0.1, -0.2, -0.3],
            agent_loop_type="AgentInLocalhostLoop",
        )

        data_batches, info = self._convert(controller, [[state]])

        self.assertEqual(data_batches, [])
        self.assertEqual(info["training_samples"], 0)

    def test_input_ids_must_not_exceed_pack_max_length(self):
        # pack_max_length 过小时要在进入 packing 前失败，避免后续训练侧报错难定位。
        controller = self._build_controller([1.0])
        state = self._state(prompt_ids=[10, 11, 12], response_ids=[20, 21, 22])

        with self.assertRaises(AssertionError):
            self._convert(controller, [[state]], pack_max_length=4)


class TestConvertRolloutGroupsPackAlignment(unittest.TestCase):
    """多条样本 pack 成一条序列后, pg_loss 消费前的张量逐位置对齐。

    regression: advantage 曾按 `[adv]*len(prompt_ids) + [mask处理]` 构造, 比 input_ids 长 1,
    pack 拼接后 advantages 与 shifted_labels/rollout_logprobs 整体错位或形状不匹配。
    """

    def _make_controller(self, with_teacher_fields: bool = False) -> TrainingController:
        controller = TrainingController.__new__(TrainingController)
        controller.logger = MagicMock()
        controller.task_adv_weight = 1.0
        distillation = MagicMock()
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
        controller.distillation = distillation
        controller.advantage_estimator = MagicMock()
        # 确定性 advantage: advantage = reward - 0.5, 每个样本值唯一且 float32 精确。
        controller.advantage_estimator.compute.side_effect = lambda rewards, representatives: rewards - 0.5
        return controller

    def _make_sample(
        self,
        rollout_id: int,
        prompt_ids: list[int],
        response_ids: list[int],
        supervised_mask: list[int],
        logprobs: list[float],
        reward: float,
    ) -> RolloutState:
        state = RolloutState(
            rollout_id=rollout_id,
            message=[],
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            logprobs=logprobs,
            response="ok",
            reward={"score": reward},
            status=Status.COMPLETED,
        )
        # 模拟 loop 侧 canonicalize 后的最终形态（语义洞直接烙在 labels）。
        state.input_ids = list(prompt_ids) + list(response_ids)
        state.labels = [-100] * len(prompt_ids) + [
            resp_id if flag else -100 for resp_id, flag in zip(response_ids, supervised_mask)
        ]
        state.logprobs = [0.0] * len(prompt_ids) + list(logprobs)
        return state

    @staticmethod
    def _expected_segment(
        sample: RolloutState, adv_val: float
    ) -> tuple[list[int], list[int], list[float], list[float]]:
        """单个样本 pack 前的正确布局: input_ids 去掉 response 末位(EOS 只作 label)。"""
        assert sample.prompt_ids is not None
        assert sample.response_ids is not None
        assert sample.labels is not None
        assert sample.logprobs is not None
        prompt_len = len(sample.prompt_ids)
        input_ids = list(sample.prompt_ids) + list(sample.response_ids)[:-1]
        response_labels = list(sample.labels)[len(sample.labels) - len(sample.response_ids) :]
        labels = [-100] * (prompt_len - 1) + response_labels
        advantages = [0.0] * (prompt_len - 1) + [0.0 if label == -100 else adv_val for label in response_labels]
        # canonical logprobs 为全序列对齐（prompt 段补 0），shift 后即 pack 布局。
        logprobs = list(sample.logprobs)[1:]
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
        """两组样本; 组内共享 prompt(与真实 RL 组一致), 长度/监督位/advantage 刻意互不相同。"""
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

    def _convert(self, controller, data_groups, pack_max_length: int):
        with patch("xtuner.v1.rl.trainer.controller.XTUNER_DETERMINISTIC", True):
            return controller._convert_rollout_groups(data_groups, pack_max_length)

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
        controller = self._make_controller()
        data_groups = self._build_data_groups()
        samples = [sample for group in data_groups for sample in group]
        total_len = sum(len(s.prompt_ids) + len(s.response_ids) - 1 for s in samples)

        data_batches, _ = self._convert(controller, data_groups, pack_max_length=total_len)

        self.assertEqual(len(data_batches), len(samples))
        for item in data_batches:
            self.assertEqual(len(item["advantage"]), item["shifted_labels"].numel())

        packed = self._pack_and_extract(data_batches, total_len)
        self._assert_real_token_alignment(samples, packed, total_len, total_len)

    def test_packed_fields_align_with_padding(self) -> None:
        controller = self._make_controller()
        data_groups = self._build_data_groups()
        samples = [sample for group in data_groups for sample in group]
        total_len = sum(len(s.prompt_ids) + len(s.response_ids) - 1 for s in samples)
        pack_max_length = (total_len // 16 + 1) * 16

        data_batches, _ = self._convert(controller, data_groups, pack_max_length=pack_max_length)
        packed = self._pack_and_extract(data_batches, pack_max_length)

        self._assert_real_token_alignment(samples, packed, total_len, pack_max_length)
        # pad 区: label 为 -100, advantage 为 controller 的 pad 值 -100。
        self.assertTrue(all(label == -100 for label in packed["labels"][total_len:]))
        self.assertTrue(all(adv == -100 for adv in packed["advantages"][total_len:]))

    def test_packed_fields_align_with_teacher_fields(self) -> None:
        controller = self._make_controller(with_teacher_fields=True)
        data_groups = self._build_data_groups()
        samples = [sample for group in data_groups for sample in group]
        total_len = sum(len(s.prompt_ids) + len(s.response_ids) - 1 for s in samples)

        data_batches, _ = self._convert(controller, data_groups, pack_max_length=total_len)
        packed = self._pack_and_extract(data_batches, total_len)

        self._assert_real_token_alignment(samples, packed, total_len, total_len)
        self.assertEqual(len(packed["teacher_logprobs"]), total_len)
        self.assertEqual(len(packed["target_token_ids"]), total_len)
        self.assertEqual(len(packed["teacher_indices"]), total_len)


if __name__ == "__main__":
    unittest.main()

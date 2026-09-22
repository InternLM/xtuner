"""RL 训练数据构造的 contract 测试（loop 侧 canonicalize + controller 侧调度准备）。

重构后训练数据构造分三段，本文件只测 loop 侧与 controller 侧的公开纯逻辑，不启动
trainer、Ray worker、模型或 rollout backend：

- loop 侧 ``canonicalize_train_fields``：基类为 prompt+response 型样本构造统一全序列字段
  （``input_ids``/``labels``/``logprobs`` 等长、未 shift）并写轻量 train meta；
  localhost/sandbox 各自覆写为 trace 全序列字段的校验（失败置 FAILED，不上抛）。
- controller 侧 ``TrainingController._prepare_rollout_items``：只消费轻量 meta 字段
  （``num_tokens``/``supervised_tokens``/``train_prompt_length`` 等），完成组校验、组级
  advantage、``data_info`` 统计与 items-advantages 配对 shuffle，不构造任何训练张量。
- token 级张量构造与 pack 物化在 training worker 侧（见 ``test_training_worker_rank.py``）；
  index-only 打包计划见 ``test_pack.py``。

当前测试点：
- 无效 rollout group 会被跳过；缺 reward 在 task_adv_weight>0 时 fail fast，=0 时零 advantage 参训。
- 同一个 prompt 下多个 response 各自使用对应 reward / advantage；advantage 统计只计监督位。
- controller 依赖 write_train_meta 写入的 meta 字段，缺失时 fail fast。
- shuffle 只打乱调度顺序，items 与 advantages 的配对关系保持不变。
- 批级 3D position 布局位由 meta 推导（use_3d_position_ids）。
"""

import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    Status,
    write_train_meta,
)
from xtuner.v1.rl.distillation import DistillationTrainerAdapter
from xtuner.v1.rl.trainer.controller import TrainingController


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
        position_ids: np.ndarray | None = None,
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
            position_ids=position_ids,
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

    def test_writes_train_meta_fields(self):
        # canonicalize 通过后写下游调度消费的轻量统计：训练长度、prompt/response 长度、
        # 监督 token 数与 position 布局。
        loop = self._make_loop()
        state = self._state(response_ids=[20, 21, 22], logprobs=[0.1, 0.2, 0.3])

        loop.canonicalize_train_fields([state])

        self.assertEqual(state.num_tokens, 5)
        self.assertEqual(state.extra_fields["train_prompt_length"], 3)
        self.assertEqual(state.extra_fields["train_response_length"], 3)
        self.assertEqual(state.extra_fields["supervised_tokens"], 3)
        self.assertEqual(state.extra_fields["position_layout"], "1d")

    def test_train_meta_uses_train_prompt_ids_and_mrope_layout(self):
        # VLM 分支 prompt 长度取 train_prompt_ids；3D position_ids 标记 mrope_3d 布局。
        loop = self._make_loop()
        state = self._state(
            prompt_ids=[1],
            response_ids=[102, 103],
            extra_fields={"train_prompt_ids": [100, 101]},
            position_ids=np.array([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]], dtype=np.int64),
        )

        loop.canonicalize_train_fields([state])

        self.assertEqual(state.num_tokens, 3)
        self.assertEqual(state.extra_fields["train_prompt_length"], 2)
        self.assertEqual(state.extra_fields["position_layout"], "mrope_3d")

    def test_train_meta_prompt_length_falls_back_to_unsupervised_count(self):
        # prompt_ids 缺失时 prompt 长度退回 shifted_labels 的非监督位计数（与旧 data_info 口径一致）。
        loop = self._make_loop()
        state = self._state()
        state.prompt_ids = None
        state.input_ids = [30, 31, 40, 41]
        state.labels = [-100, -100, 40, 41]

        loop.canonicalize_train_fields([state])

        self.assertEqual(state.num_tokens, 3)
        self.assertEqual(state.extra_fields["train_prompt_length"], 1)
        self.assertEqual(state.extra_fields["supervised_tokens"], 2)

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

    def test_writes_train_meta_for_valid_states(self):
        for loop_cls in self.AGENTIC_LOOP_CLASSES:
            with self.subTest(loop_cls=loop_cls.__name__):
                loop = self._make_loop(loop_cls)
                state = self._agentic_state(
                    input_ids=[30, 31, 40, 41, 42],
                    labels=[-100, -100, 40, 41, 42],
                    logprobs=[0.0, -0.1, -0.2, -0.3, -0.4],
                )

                loop.canonicalize_train_fields([state])

                self.assertEqual(state.num_tokens, 4)
                self.assertEqual(state.extra_fields["train_prompt_length"], 2)
                self.assertEqual(state.extra_fields["train_response_length"], 2)
                self.assertEqual(state.extra_fields["supervised_tokens"], 3)
                self.assertEqual(state.extra_fields["position_layout"], "1d")

    def test_train_meta_skipped_for_invalid_states(self):
        # FAILED 与无 input_ids 的样本不写轻量统计。
        for loop_cls in self.AGENTIC_LOOP_CLASSES:
            with self.subTest(loop_cls=loop_cls.__name__):
                loop = self._make_loop(loop_cls)
                failed = self._agentic_state(status=Status.FAILED)
                eval_state = self._agentic_state()

                loop.canonicalize_train_fields([failed, eval_state])

                self.assertIsNone(failed.num_tokens)
                self.assertNotIn("supervised_tokens", failed.extra_fields)
                self.assertIsNone(eval_state.num_tokens)
                self.assertNotIn("supervised_tokens", eval_state.extra_fields)

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


class TestPrepareRolloutItems(unittest.TestCase):
    """TrainingController._prepare_rollout_items 合同：组级 advantage 与 data_info 统计。

    controller 只消费 write_train_meta 写入的轻量 meta 字段（缺失即 fail fast；组级
    状态/对齐校验由 replay buffer 查询与 canonicalize_train_fields 保证）；shift/张量化/
    pack 物化在 training worker 侧（见 test_training_worker_rank.py），本类不构造任何训练张量。
    """

    def _build_controller(self, advantages: list[float], task_adv_weight: float = 1.0) -> TrainingController:
        controller = TrainingController.__new__(TrainingController)
        controller.advantage_estimator = _FakeAdvantageEstimator(advantages)
        controller.task_adv_weight = task_adv_weight
        controller.distillation = DistillationTrainerAdapter(None)
        controller.logger = MagicMock()
        return controller

    def _prepare(self, controller, data_groups):
        prepared = controller._prepare_rollout_items(data_groups, rollout_idx=0)
        return (
            prepared["rollout_items"],
            prepared["advantages"],
            controller._build_trainer_log_info(
                prepared, pack_plan={"dp_dispatches": {}, "plan_log": {}}, log_infos=[]
            ),
            prepared["batch_attr"]["use_3d_position_ids"],
        )

    def _state(
        self,
        *,
        uid: int = 1,
        group_id: int = 1,
        prompt_ids: list[int] | None = None,
        response_ids: list[int] | None = None,
        logprobs: list[float] | None = None,
        supervised_mask: list[int] | None = None,
        reward: dict | None = None,
        status: Status = Status.COMPLETED,
        response: str = "response",
        position_ids: np.ndarray | None = None,
        input_ids: list[int] | None = None,
        labels: list[int] | None = None,
        write_meta: bool = True,
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
            reward=reward if reward is not None else {"score": 1.0},
            status=status,
            finish_reason="stop" if status == Status.COMPLETED else "error",
            position_ids=position_ids,
            input_ids=input_ids,
            labels=labels,
        )
        if input_ids is None:
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
        if write_meta:
            write_train_meta(state)
        return state

    def test_returns_items_with_per_sample_advantages(self):
        # controller 只做调度准备：states 原样透传，advantage 逐样本对齐，不构造训练张量。
        controller = self._build_controller([1.5])
        state = self._state(
            response_ids=[20, 21, 22],
            logprobs=[0.1, 0.2, 0.3],
            supervised_mask=[1, 0, 1],
            reward={"score": 1.0},
        )

        items, advantages, info, use_3d = self._prepare(controller, [[state]])

        self.assertEqual(len(items), 1)
        self.assertIs(items[0], state)
        self.assertEqual(advantages, [1.5])
        self.assertFalse(use_3d)
        self.assertEqual(info["training_samples"], 1)
        self.assertEqual(info["training_tokens"], 5)
        self.assertEqual(info["rewards/mean"], 1.0)
        # response_len/prompt_len 由 meta 字段推导（labels 是唯一监督载体：语义洞不计入）。
        self.assertEqual(info["response_len/mean"], 2.0)
        self.assertEqual(info["prompt_len/mean"], 3.0)

    def test_multi_sample_group_uses_each_sample_reward_and_advantage(self):
        # 同一个 prompt 下的多个 response 要分别使用自己的 reward 和 advantage。
        controller = self._build_controller([1.5, -2.0])
        first = self._state(uid=1, response_ids=[20, 21], reward={"score": 3.0})
        second = self._state(uid=2, response_ids=[30, 31], reward={"score": -1.0})

        items, advantages, info, _ = self._prepare(controller, [[first, second]])

        self.assertEqual(len(items), 2)
        self.assertIs(items[0], first)
        self.assertIs(items[1], second)
        self.assertEqual(advantages, [1.5, -2.0])
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
        )

        _, _, info, _ = self._prepare(controller, [[plain, agentic]])

        # plain: 2 个 label!=-100 的 response token 计入 2.0; agentic: 同理计入 -1.0。
        self.assertEqual(info["advantages/mean"], 0.5)
        self.assertEqual(info["advantages/min"], -1.0)
        self.assertEqual(info["advantages/max"], 2.0)

    def test_missing_reward_score_fails_fast(self):
        # reward 必须包含 score，否则 advantage 计算前后语义都不明确。
        controller = self._build_controller([1.0])
        state = self._state(reward={"other": 1.0})

        with self.assertRaisesRegex(ValueError, "missing.*score"):
            self._prepare(controller, [[state]])

    def test_missing_reward_with_zero_task_adv_weight_trains_with_zero_advantage(self):
        # task_adv_weight=0 时（纯 OPD）缺 reward 不崩溃：样本以零 advantage 参训，不计入 rewards 统计。
        controller = self._build_controller([1.0], task_adv_weight=0.0)
        state = self._state()
        state.reward = None

        items, advantages, info, _ = self._prepare(controller, [[state]])

        self.assertEqual(len(items), 1)
        self.assertIs(items[0], state)
        self.assertEqual(advantages, [0.0])
        self.assertEqual(info["training_samples"], 1)
        self.assertEqual(info["rewards/mean"], 0.0)
        self.assertEqual(controller.advantage_estimator.calls, [])

    def test_missing_train_meta_fails_fast(self):
        # 缺 write_train_meta 写入的 meta 字段时 fail fast：这类状态到 worker 转换阶段也会失败。
        controller = self._build_controller([1.0])
        state = self._state(write_meta=False)

        with self.assertRaisesRegex(ValueError, "write_train_meta"):
            self._prepare(controller, [[state]])

    def test_use_3d_position_ids_follows_batch_meta(self):
        # 批级 3D 布局位：任一样本为 mrope_3d 时整批用 3D position（混合批文本样本会补 3D 轴）。
        controller = self._build_controller([1.0, 1.0])
        text = self._state(uid=1, response_ids=[20, 21])
        vlm = self._state(
            uid=2,
            response_ids=[30, 31],
            position_ids=np.array([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=np.int64),
        )

        _, _, _, text_only = self._prepare(controller, [[text]])
        self.assertFalse(text_only)

        _, _, _, mixed = self._prepare(controller, [[text], [vlm]])
        self.assertTrue(mixed)


if __name__ == "__main__":
    unittest.main()

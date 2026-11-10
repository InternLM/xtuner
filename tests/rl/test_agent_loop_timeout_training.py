"""AgentInSandboxLoop 超时轨迹保留训练(train_on_timeout)的行为测试。

本文件不加载 tokenizer/processor、真实 trace store(Ray actor)或 teacher 服务,
runner 用预置结果的假对象注入(作为 rollout_item.pipeline 传入,走真实的
generate_sample / generate_group 代码路径)。

TestTimeoutTraining —— 超时样本的生成与保留训练行为
- test_timeout_sample_is_kept_trainable_with_fixed_reward:
    开关打开且错误为 entry timeout 时,已完成 turn 的部分轨迹被保留为 COMPLETED 样本,
    finish_reason="timeout",reward 固定为 timeout_reward,训练字段全部填充并带
    timeout_trained 标记。
- test_timeout_tool_turns_is_none_without_store_count:
    导出链缺少 tool_call_count 时 agent_tool_turns 保持 None,不从 artifacts 估算。
- test_timeout_falls_back_without_trainable_trace:
    单 token 或 labels 全 -100 的部分轨迹没有可训练 token 时回退 FAILED。
- test_timeout_response_ids_do_not_depend_on_prompt_ids:
    response_ids 由 trace store 从头节点(会话初始 prompt 段)切出,与 dataset
    prompt_ids 的渲染边界无关;两者不一致时样本仍可保留。
- test_timeout_decode_failure_keeps_state_failed:
    保留流程中任何一步失败时样本保持 FAILED,不会被标成 COMPLETED 或写入固定 reward。
- test_non_timeout_failure_keeps_failed:
    非 timeout 失败不触发保留,也不查询 trace store。
- test_timeout_without_flag_keeps_failed:
    开关关闭时保持原有 FAILED 行为,也不查询 trace store。

TestTeacherScoring —— teacher scoring 对超时样本的处理
- test_group_level_teacher_scoring_includes_timeout_samples:
    group 级 teacher scoring(defer 模式)同时收到普通样本和超时样本。
- test_sample_level_teacher_scoring_includes_timeout_samples:
    非 defer 模式下逐样本 teacher scoring 也收到超时样本。
- test_stale_timeout_marker_is_cleared_on_retry:
    上一次 attempt 残留的 timeout_trained 标记会被清理。

TestTimeoutTrainingConfigValidation —— train_on_timeout 与 expired rerollout 的配置期互斥
- test_timeout_training_rejects_expired_rerollout:
    train_on_timeout=True 时 tail_batch_trigger_size=0/1 使 TaskSpecConfig/DisaggTaskSpecConfig
    构造抛 ValueError。
- test_timeout_training_allows_disabled_rerollout:
    tail_batch_trigger_size=-1 时允许打开 train_on_timeout。
- test_rerollout_without_timeout_training_is_allowed:
    未打开 train_on_timeout 时 tail_batch_trigger_size>=0 不受影响。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import (
    AgentInSandboxLoop,
    AgentInSandboxLoopConfig,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem, RolloutError, RolloutStatus
from xtuner.v1.rl.agent_loop_manager import (
    AsyncProduceStrategyConfig,
    DisaggAsyncProduceStrategyConfig,
    DisaggTaskSpecConfig,
    SamplerConfig,
    TaskSpecConfig,
)

LOOP_MODULE = "xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop"


class _FakeRunner:
    """Pre-built runner returning a preset result, injected as ``pipeline``."""

    def __init__(self, result: AgentRolloutItem):
        self._result = result

    def __deepcopy__(self, memo):
        return self  # stay identity-stable across rollout_item.model_copy(deep=True)

    async def run(self, item: AgentRolloutItem) -> AgentRolloutItem:
        return self._result


def _make_loop(**kwargs) -> AgentInSandboxLoop:
    fake_tokenizer = SimpleNamespace(
        decode=lambda token_ids: "decoded",
        apply_chat_template=lambda messages, **kwargs: "rendered prompt",
    )
    # Real loggers accept ``exc_info``; tolerate it so the fallback warning path is exercised.
    fake_logger = SimpleNamespace(
        warning=lambda *args, **kwargs: print(*args),
        error=lambda *args, **kwargs: print(*args),
        info=lambda *args, **kwargs: print(*args),
    )
    with (
        patch("xtuner.v1.rl.agent_loop.agent_loop.load_tokenizer", return_value=fake_tokenizer),
        patch("xtuner.v1.rl.agent_loop.agent_loop.load_processor", return_value=object()),
    ):
        return AgentInSandboxLoop(
            rollout_ctl=None,
            hf_checkpoint="dummy",
            sample_params=None,
            judger=None,
            logger=fake_logger,
            **kwargs,
        )


def _make_state(result: AgentRolloutItem) -> RolloutState:
    # Fake traces start with a one-token prompt, so ``response_ids`` (the contiguous
    # suffix after the prompt) is ``input_ids[1:]``.
    return RolloutState(
        message=[{"role": "user", "content": "hi"}],
        prompt_ids=[1],
        session_id=123,
        extra_fields={"rollout_item": AgentRolloutItem(
            id="src", data_source="test", uid=123, instruction="task.md", pipeline=_FakeRunner(result)
        )},
    )


def _timeout_item() -> AgentRolloutItem:
    item = AgentRolloutItem(id="item-1", data_source="test", uid=123, instruction="task.md")
    item.status = RolloutStatus.FAILED
    item.error = RolloutError(category="timeout", message="entry timed out")
    item.artifacts = {}
    return item


def _timeout_item_with_tool_calls() -> AgentRolloutItem:
    item = _timeout_item()
    item.artifacts = {
        "messages": [{"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "f"}}]},
        ], "tools": []}],
    }
    return item


def _completed_item() -> AgentRolloutItem:
    item = AgentRolloutItem(id="item-2", data_source="test", uid=123, instruction="task.md")
    item.status = RolloutStatus.COMPLETED
    item.reward = 1.0
    item.artifacts = {
        "messages": [{"messages": [{"role": "user", "content": "hi"},
                                    {"role": "assistant", "content": "answer"}],
                      "tools": []}],
        "response_message": {"finish_reason": "stop", "content": "answer"},
    }
    return item


def _fake_store(partial_trace: dict | None = None) -> SimpleNamespace:
    partial = partial_trace or {"input_ids": [], "labels": [], "logprobs": [], "routed_experts": None}
    # Fake traces start with a one-token session prompt, so the store-derived response_ids
    # is the tail after it (the real store slices from its head node).
    partial = {**partial, "response_ids": list(partial["input_ids"][1:])}
    return SimpleNamespace(
        export_partial_training_trace=SimpleNamespace(remote=AsyncMock(return_value=partial)),
        export_training_trace=SimpleNamespace(
            remote=AsyncMock(return_value={"input_ids": [1, 2], "labels": [-100, 5],
                                           "logprobs": [0.0, 0.1], "routed_experts": None,
                                           "response_ids": [2]})
        ),
    )


class TestTimeoutTraining(unittest.IsolatedAsyncioTestCase):
    """超时样本经 generate_sample 公开路径的保留训练行为。"""

    async def test_timeout_sample_is_kept_trainable_with_fixed_reward(self):
        # 行为:开关打开且错误为 entry timeout 时,已完成 turn 的部分轨迹被保留为可训练样本
        loop = _make_loop(train_on_timeout=True, timeout_reward=-1.0)
        state = _make_state(_timeout_item())
        partial_trace = {
            "input_ids": [1, 2, 3],
            "labels": [-100, 5, 6],
            "logprobs": [0.0, 0.1, 0.2],
            "routed_experts": None,
            "tool_call_count": 2,
        }

        with patch(f"{LOOP_MODULE}.get_store", return_value=_fake_store(partial_trace)) as get_store:
            kept_list = await loop.generate_sample(state)

        get_store.assert_called_once_with()
        self.assertEqual(len(kept_list), 1)
        kept = kept_list[0]
        self.assertEqual(kept.status, Status.COMPLETED)
        self.assertEqual(kept.finish_reason, "timeout")
        self.assertEqual(kept.reward, {"score": -1.0})
        self.assertEqual(kept.input_ids, [1, 2, 3])
        self.assertEqual(kept.labels, [-100, 5, 6])
        # response_ids 由 trace store 从头节点切出的连续后缀(与正常路径一致)。
        self.assertEqual(kept.response_ids, [2, 3])
        self.assertEqual(kept.logprobs, [0.0, 0.1, 0.2])
        self.assertIsNone(kept.routed_experts)
        self.assertTrue(kept.extra_fields["timeout_trained"])
        self.assertEqual(kept.extra_fields["agent_status"], "timeout")
        # 优先使用 trace store 沿导出链求和的计数(与训练 token 同源)。
        self.assertEqual(kept.extra_fields["agent_tool_turns"], 2)

    async def test_timeout_tool_turns_is_none_without_store_count(self):
        # 行为:导出的 tool_call_count 为 None(旧节点缺字段)时,agent_tool_turns 保持
        # None——trace store 没有记录就不虚构计数,不从 artifacts 估算
        loop = _make_loop(train_on_timeout=True)
        state = _make_state(_timeout_item_with_tool_calls())
        partial_trace = {
            "input_ids": [1, 2, 3],
            "labels": [-100, 5, 6],
            "logprobs": [0.0, 0.1, 0.2],
            "routed_experts": None,
            "tool_call_count": None,
        }

        with patch(f"{LOOP_MODULE}.get_store", return_value=_fake_store(partial_trace)):
            kept_list = await loop.generate_sample(state)

        self.assertIsNone(kept_list[0].extra_fields["agent_tool_turns"])

    async def test_timeout_response_ids_do_not_depend_on_prompt_ids(self):
        # 行为:response_ids 由 trace store 从头节点切出,与 dataset prompt_ids 的渲染边界无关;
        # 两者不一致时样本仍可保留(agent 请求渲染含 system/tools,与 dataset 渲染不同)
        loop = _make_loop(train_on_timeout=True)
        state = _make_state(_timeout_item())
        state.prompt_ids = [99]  # 与 partial_trace 的 [1, 2, 3] 不一致
        partial_trace = {
            "input_ids": [1, 2, 3],
            "labels": [-100, 5, 6],
            "logprobs": [0.0, 0.1, 0.2],
            "routed_experts": None,
        }

        with patch(f"{LOOP_MODULE}.get_store", return_value=_fake_store(partial_trace)):
            kept_list = await loop.generate_sample(state)

        self.assertEqual(len(kept_list), 1)
        kept = kept_list[0]
        self.assertEqual(kept.status, Status.COMPLETED)
        self.assertEqual(kept.response_ids, [2, 3])

    async def test_timeout_decode_failure_keeps_state_failed(self):
        # 行为:落盘前任何一步失败(这里让 decode 抛错)都不能留下 COMPLETED 状态和固定 reward
        loop = _make_loop(train_on_timeout=True)
        loop.tokenizer.decode = Mock(side_effect=RuntimeError("decode boom"))
        state = _make_state(_timeout_item())
        partial_trace = {
            "input_ids": [1, 2, 3],
            "labels": [-100, 5, 6],
            "logprobs": [0.0, 0.1, 0.2],
            "routed_experts": None,
        }

        with patch(f"{LOOP_MODULE}.get_store", return_value=_fake_store(partial_trace)):
            failed_list = await loop.generate_sample(state)

        failed = failed_list[0]
        self.assertEqual(failed.status, Status.FAILED)
        self.assertEqual(failed.extra_fields["agent_status"], "failed")
        self.assertIsNone(failed.reward)
        self.assertNotIn("timeout_trained", failed.extra_fields)

    async def test_timeout_falls_back_without_trainable_trace(self):
        # 行为:单 token 或 labels 全 -100 的部分轨迹没有可训练 token,回退 FAILED
        cases = {
            "single_token": {"input_ids": [5], "labels": [5], "logprobs": [0.0], "routed_experts": None},
            "all_masked": {
                "input_ids": [1, 2, 3],
                "labels": [-100, -100, -100],
                "logprobs": [0.0, 0.1, 0.2],
                "routed_experts": None,
            },
        }
        for name, partial_trace in cases.items():
            with self.subTest(name):
                loop = _make_loop(train_on_timeout=True)
                state = _make_state(_timeout_item())
                with patch(f"{LOOP_MODULE}.get_store", return_value=_fake_store(partial_trace)):
                    failed_list = await loop.generate_sample(state)
                self.assertEqual(len(failed_list), 1)
                self.assertEqual(failed_list[0].status, Status.FAILED)

    async def test_non_timeout_failure_keeps_failed(self):
        # 行为:错误 category 非 timeout 时不触发保留,也不查询 trace store
        loop = _make_loop(train_on_timeout=True)
        item = _timeout_item()
        item.error = RolloutError(category="entry_failed", message="rc=1")
        state = _make_state(item)

        with patch(f"{LOOP_MODULE}.get_store") as get_store:
            failed_list = await loop.generate_sample(state)

        get_store.assert_not_called()
        self.assertEqual(failed_list[0].status, Status.FAILED)

    async def test_timeout_without_flag_keeps_failed(self):
        # 行为:开关关闭(默认)时保持原有 FAILED 行为,也不查询 trace store
        loop = _make_loop(train_on_timeout=False)
        state = _make_state(_timeout_item())

        with patch(f"{LOOP_MODULE}.get_store") as get_store:
            failed_list = await loop.generate_sample(state)

        get_store.assert_not_called()
        self.assertEqual(failed_list[0].status, Status.FAILED)


class TestTeacherScoring(unittest.IsolatedAsyncioTestCase):
    """teacher scoring 对超时样本的处理行为。"""

    async def test_group_level_teacher_scoring_includes_timeout_samples(self):
        # 行为:defer 模式的 group 级 teacher scoring 同时收到普通样本和超时样本
        loop = _make_loop(train_on_timeout=True)
        normal_state = _make_state(_completed_item())
        timeout_state = _make_state(_timeout_item())
        partial_trace = {
            "input_ids": [1, 2, 3],
            "labels": [-100, 5, 6],
            "logprobs": [0.0, 0.1, 0.2],
            "routed_experts": None,
        }

        scored_states: list[RolloutState] = []

        async def fake_on_group_ready(group: list[RolloutState]) -> list[RolloutState]:
            scored_states.extend(group)
            return group

        loop._teacher_scorer = SimpleNamespace(
            on_sample_ready=AsyncMock(side_effect=lambda state: state),
            on_group_ready=fake_on_group_ready,
        )

        with patch(f"{LOOP_MODULE}.get_store", return_value=_fake_store(partial_trace)):
            result = await loop.generate_group([normal_state, timeout_state])

        self.assertEqual(len(result), 2)
        scored_flags = sorted(state.extra_fields.get("timeout_trained", False) for state in scored_states)
        self.assertEqual(scored_flags, [False, True])

    async def test_sample_level_teacher_scoring_includes_timeout_samples(self):
        # 行为:非 defer 模式下逐样本 teacher scoring 也收到超时样本
        loop = _make_loop(train_on_timeout=True)
        normal_state = _make_state(_completed_item())
        timeout_state = _make_state(_timeout_item())
        partial_trace = {
            "input_ids": [1, 2, 3],
            "labels": [-100, 5, 6],
            "logprobs": [0.0, 0.1, 0.2],
            "routed_experts": None,
        }

        scored_states: list[RolloutState] = []

        async def fake_on_sample_ready(sample: RolloutState) -> RolloutState:
            scored_states.append(sample)
            return sample

        loop._teacher_scorer = SimpleNamespace(
            on_sample_ready=fake_on_sample_ready,
            on_group_ready=AsyncMock(side_effect=lambda group: group),
        )

        with patch(f"{LOOP_MODULE}.get_store", return_value=_fake_store(partial_trace)):
            await loop.generate_group([normal_state, timeout_state])

        self.assertEqual(len(scored_states), 2)
        self.assertEqual(sorted(s.extra_fields.get("timeout_trained", False) for s in scored_states), [False, True])

    async def test_stale_timeout_marker_is_cleared_on_retry(self):
        # 行为:上一次 attempt 残留的 timeout_trained 标记被清理
        loop = _make_loop(train_on_timeout=True)
        state = _make_state(_completed_item())
        state.extra_fields["timeout_trained"] = True  # 上次超时尝试遗留的标记

        loop._teacher_scorer = SimpleNamespace(
            on_sample_ready=AsyncMock(side_effect=lambda state: state),
            on_group_ready=AsyncMock(side_effect=lambda group: group),
        )

        with patch(f"{LOOP_MODULE}.get_store", return_value=_fake_store()):
            result = await loop.generate_group([state])

        self.assertEqual(len(result), 1)
        self.assertNotIn("timeout_trained", result[0].extra_fields)


class TestTimeoutTrainingConfigValidation(unittest.TestCase):
    """train_on_timeout 与 expired rerollout 的配置期互斥校验。"""

    @staticmethod
    def _sampler_config() -> SamplerConfig:
        # 只验证 validator 接线，不构造真实 dataloader。
        return SamplerConfig.model_construct(dataloader_cfg=SimpleNamespace(), prompt_repeat_k=1)

    @staticmethod
    def _agent_loop_config(*, train_on_timeout: bool) -> AgentInSandboxLoopConfig:
        return AgentInSandboxLoopConfig(hf_checkpoint="dummy", train_on_timeout=train_on_timeout)

    def _task_spec(self, *, train_on_timeout: bool, tail_batch_trigger_size: int) -> TaskSpecConfig:
        return TaskSpecConfig(
            task_name="t",
            agent_loop_config=self._agent_loop_config(train_on_timeout=train_on_timeout),
            produce_strategy_config=AsyncProduceStrategyConfig(tail_batch_trigger_size=tail_batch_trigger_size),
            sampler_config=self._sampler_config(),
        )

    def _disagg_task_spec(self, *, train_on_timeout: bool, tail_batch_trigger_size: int) -> DisaggTaskSpecConfig:
        return DisaggTaskSpecConfig(
            task_name="t",
            agent_loop_config=self._agent_loop_config(train_on_timeout=train_on_timeout),
            produce_strategy_config=DisaggAsyncProduceStrategyConfig(
                tail_batch_trigger_size=tail_batch_trigger_size
            ),
            sampler_config=self._sampler_config(),
        )

    def test_timeout_training_rejects_expired_rerollout(self):
        # 行为:train_on_timeout 打开时,任何启用 expired rerollout 的 tail_batch_trigger_size 都在配置期被拒绝
        for spec_factory in (self._task_spec, self._disagg_task_spec):
            for trigger_size in (0, 1):
                with self.subTest(spec=spec_factory.__name__, tail_batch_trigger_size=trigger_size):
                    with self.assertRaisesRegex(ValueError, "train_on_timeout"):
                        spec_factory(train_on_timeout=True, tail_batch_trigger_size=trigger_size)

    def test_timeout_training_allows_disabled_rerollout(self):
        # 行为:tail_batch_trigger_size=-1(禁用 rerollout)时允许打开 train_on_timeout
        for spec_factory in (self._task_spec, self._disagg_task_spec):
            with self.subTest(spec=spec_factory.__name__):
                spec = spec_factory(train_on_timeout=True, tail_batch_trigger_size=-1)
                self.assertTrue(spec.agent_loop_config.train_on_timeout)

    def test_rerollout_without_timeout_training_is_allowed(self):
        # 行为:未打开 train_on_timeout 时,expired rerollout 配置不受影响
        for spec_factory in (self._task_spec, self._disagg_task_spec):
            with self.subTest(spec=spec_factory.__name__):
                spec = spec_factory(train_on_timeout=False, tail_batch_trigger_size=0)
                self.assertFalse(spec.agent_loop_config.train_on_timeout)


if __name__ == "__main__":
    unittest.main()

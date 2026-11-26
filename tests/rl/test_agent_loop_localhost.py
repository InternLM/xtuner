"""AgentInLocalhostLoop 训练字段行为测试。

TestLocalhostTraceFields —— _fill_rollout_state 从 trace store 读取训练字段
- test_response_ids_come_from_store_not_dataset_prompt:
    prompt_ids/response_ids 都使用 store 导出的首段 + 连续后缀(含 tool-result 段),
    满足 input_ids == prompt_ids + response_ids;与 dataset prompt_ids 的渲染边界无关。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop.localhost_agent_loop.agent_in_localhost_loop import AgentInLocalhostLoop
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem, RolloutStatus


LOOP_MODULE = "xtuner.v1.rl.agent_loop.localhost_agent_loop.agent_in_localhost_loop"


def _make_loop() -> AgentInLocalhostLoop:
    loop = AgentInLocalhostLoop.__new__(AgentInLocalhostLoop)
    loop.mode = "train"
    loop.tokenizer = MagicMock()
    loop.tokenizer.apply_chat_template.return_value = "rendered prompt\n"
    return loop


def _make_item() -> AgentRolloutItem:
    return AgentRolloutItem(
        id="item-1",
        data_source="test",
        instruction="task.md",
        status=RolloutStatus.COMPLETED,
        reward=1.0,
        artifacts={
            "messages": [
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "answer"},
                    ],
                    "tools": [],
                }
            ],
            "response_message": {"content": "answer", "finish_reason": "stop"},
        },
    )


def _fake_store(trace: dict) -> SimpleNamespace:
    return SimpleNamespace(
        export_training_trace=SimpleNamespace(remote=AsyncMock(return_value=trace)),
        empty_sessions=SimpleNamespace(remote=AsyncMock(return_value=["123"])),
    )


class TestLocalhostTraceFields(unittest.IsolatedAsyncioTestCase):
    """_fill_rollout_state 使用 store 导出的训练字段。"""

    async def test_response_ids_come_from_store_not_dataset_prompt(self):
        # 行为:prompt_ids/response_ids 都取 store 导出结果,不按 dataset prompt_ids 切片
        loop = _make_loop()
        state = RolloutState(
            message=[{"role": "user", "content": "hi"}],
            prompt_ids=[99],  # 与 trace 的 [1, 2, 3] 不一致,会被 trace 首段覆盖
            session_id=123,
        )
        trace = {
            "input_ids": [1, 2, 3],
            "labels": [-100, 5, 6],
            "logprobs": [0.0, 0.1, 0.2],
            "routed_experts": None,
            "prompt_ids": [1],
            "response_ids": [2, 3],
        }

        with patch(f"{LOOP_MODULE}.get_store", return_value=_fake_store(trace)) as get_store:
            await loop._fill_rollout_state(state, _make_item())

        get_store.assert_called_once_with()
        self.assertEqual(state.status, Status.COMPLETED)
        self.assertEqual(state.finish_reason, "stop")
        self.assertEqual(state.input_ids, [1, 2, 3])
        self.assertEqual(state.labels, [-100, 5, 6])
        self.assertEqual(state.prompt_ids, [1])
        self.assertEqual(state.response_ids, [2, 3])
        self.assertEqual(state.input_ids, state.prompt_ids + state.response_ids)
        self.assertEqual(state.logprobs, [0.0, 0.1, 0.2])
        self.assertIsNone(state.routed_experts)
        self.assertEqual(state.response, "answer")

    async def test_routed_experts_are_detached_without_force_free(self):
        loop = _make_loop()
        state = RolloutState(message=[{"role": "user", "content": "hi"}], session_id=123)
        expert_ref = object()
        trace = {
            "input_ids": [1, 2],
            "labels": [-100, 2],
            "logprobs": [0.0, 0.1],
            "routed_experts": [expert_ref],
            "prompt_ids": [1],
            "response_ids": [2],
        }

        with patch(f"{LOOP_MODULE}.get_store", return_value=_fake_store(trace)):
            await loop._fill_rollout_state(state, _make_item())

        self.assertIs(state.routed_experts[0], expert_ref)
        with patch("xtuner.v1.rl.utils.ray_utils.free_object_refs") as free_refs:
            state.reset_response()

        free_refs.assert_not_called()
        self.assertIsNone(state.routed_experts)


class TestLocalhostGenerateGroup(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_group_preserves_input_order_when_completed_state_is_last(self):
        loop = AgentInLocalhostLoop.__new__(AgentInLocalhostLoop)
        loop.mode = "train"
        loop.sample_params = SampleParams()
        loop._sample_semaphore = None
        loop.is_valid_sample_fn = None
        loop.logger = MagicMock()
        loop._teacher_scorer = SimpleNamespace(
            on_sample_ready=AsyncMock(side_effect=lambda sample: sample),
            on_group_ready=AsyncMock(return_value=None),
        )
        pending = RolloutState(rollout_id=1, message=[], status=Status.EXPIRED)
        completed = RolloutState(rollout_id=2, message=[], status=Status.COMPLETED, input_ids=[2])

        async def generate_sample(state: RolloutState, **kwargs: object) -> RolloutState:
            state.status = Status.COMPLETED
            return state

        loop.generate_sample = AsyncMock(side_effect=generate_sample)
        loop.canonicalize_train_fields = lambda group: group

        samples = await loop.generate_group([pending, completed])

        self.assertEqual([state.rollout_id for state in samples], [1, 2])
        self.assertIs(samples[0], pending)
        self.assertIs(samples[1], completed)


if __name__ == "__main__":
    unittest.main()

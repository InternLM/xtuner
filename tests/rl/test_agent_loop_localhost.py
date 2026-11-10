"""AgentInLocalhostLoop 训练字段行为测试。

TestLocalhostTraceFields —— _fill_rollout_state 从 trace store 读取训练字段
- test_response_ids_come_from_store_not_dataset_prompt:
    response_ids 使用 store 导出的连续后缀(含 tool-result 段),与 dataset prompt_ids
    的渲染边界无关;两者不一致时正常样本仍可产出训练字段。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto.rl_data import RolloutState, Status
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
    return SimpleNamespace(export_training_trace=SimpleNamespace(remote=AsyncMock(return_value=trace)))


class TestLocalhostTraceFields(unittest.IsolatedAsyncioTestCase):
    """_fill_rollout_state 使用 store 导出的训练字段。"""

    async def test_response_ids_come_from_store_not_dataset_prompt(self):
        # 行为:response_ids 直接取 store 导出结果,不按 dataset prompt_ids 切片
        loop = _make_loop()
        state = RolloutState(
            message=[{"role": "user", "content": "hi"}],
            prompt_ids=[99],  # 与 trace 的 [1, 2, 3] 不一致
            session_id=123,
        )
        trace = {
            "input_ids": [1, 2, 3],
            "labels": [-100, 5, 6],
            "logprobs": [0.0, 0.1, 0.2],
            "routed_experts": None,
            "response_ids": [2, 3],
        }

        with patch(f"{LOOP_MODULE}.get_store", return_value=_fake_store(trace)) as get_store:
            await loop._fill_rollout_state(state, _make_item())

        get_store.assert_called_once_with()
        self.assertEqual(state.status, Status.COMPLETED)
        self.assertEqual(state.finish_reason, "stop")
        self.assertEqual(state.input_ids, [1, 2, 3])
        self.assertEqual(state.labels, [-100, 5, 6])
        self.assertEqual(state.response_ids, [2, 3])
        self.assertEqual(state.logprobs, [0.0, 0.1, 0.2])
        self.assertIsNone(state.routed_experts)
        self.assertEqual(state.response, "answer")


if __name__ == "__main__":
    unittest.main()

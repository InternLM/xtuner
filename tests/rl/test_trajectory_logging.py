import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.train.rl_trainer import BaseRLTrainer


with patch.dict(
    sys.modules,
    {
        "lagent": MagicMock(),
        "lagent.utils": MagicMock(),
        "lagent.utils.rate_limiter": MagicMock(),
    },
):
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import AgentInSandboxLoop
    from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem, RolloutStatus


class TestSandboxArtifactOwnership(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _make_state() -> RolloutState:
        return RolloutState(
            group_id=3,
            rollout_id=7,
            session_id=12345678901234567890,
            message=[{"role": "user", "content": "test"}],
            num_tokens=1,
            extra_fields={},
        )

    @staticmethod
    def _make_item(status: RolloutStatus, segment_count: int = 2) -> AgentRolloutItem:
        segments = [
            {
                "messages": [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": f"segment-{index}"},
                ],
                "tools": [{"type": "function", "function": {"name": f"tool-{index}"}}],
            }
            for index in range(segment_count)
        ]
        return AgentRolloutItem(
            id="test-item",
            data_source="test",
            instruction="instruction.md",
            status=status,
            reward=1.0 if status == RolloutStatus.COMPLETED else None,
            artifacts={
                "messages": segments,
                "response_message": {
                    "role": "assistant",
                    "content": "done",
                    "finish_reason": "stop",
                },
                "large_blob": "x" * 1024,
            },
        )

    @staticmethod
    def _make_loop(mode: str = "train") -> AgentInSandboxLoop:
        loop = AgentInSandboxLoop.__new__(AgentInSandboxLoop)
        loop.mode = mode
        loop.tokenizer = MagicMock()
        loop.tokenizer.apply_chat_template.side_effect = lambda messages, **_: f"rendered-{messages[-1]['content']}\n"
        loop.tokenizer.decode.side_effect = lambda token_ids: f"decoded-{token_ids[-1]}"
        return loop

    @staticmethod
    def _trace_store(segment_count: int):
        traces = [
            {
                "input_ids": [1, 10 + index, 20 + index],
                "labels": [-100, 10 + index, 20 + index],
                "logprobs": [0.0, -0.1, -0.2],
                "routed_experts": None,
            }
            for index in range(segment_count)
        ]
        return SimpleNamespace(export_training_trace=SimpleNamespace(remote=AsyncMock(side_effect=traces)))

    async def test_completed_session_assigns_artifacts_only_to_first_segment(self):
        loop = self._make_loop()
        state = self._make_state()
        item = self._make_item(RolloutStatus.COMPLETED)
        original_artifacts = copy.deepcopy(item.artifacts)
        trace_store = self._trace_store(segment_count=2)

        with patch(
            "xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop.get_store",
            return_value=trace_store,
        ):
            segments = await loop._build_rollout_states(state, item)

        self.assertEqual(len(segments), 2)
        self.assertEqual([segment.session_id for segment in segments], [state.session_id, state.session_id])
        self.assertEqual(
            [segment.extra_fields["agent_trace_segment_index"] for segment in segments],
            [0, 1],
        )
        self.assertTrue(all(segment.extra_fields["agent_trace_segment_count"] == 2 for segment in segments))
        self.assertEqual(segments[0].extra_fields["agent_artifacts"], original_artifacts)
        self.assertNotIn("agent_artifacts", segments[1].extra_fields)
        self.assertEqual(
            [segment.extra_fields["agent_messages"][-1]["content"] for segment in segments],
            ["segment-0", "segment-1"],
        )
        self.assertEqual(
            [segment.extra_fields["agent_tools"][0]["function"]["name"] for segment in segments],
            ["tool-0", "tool-1"],
        )
        self.assertEqual([segment.response_ids for segment in segments], [[10, 20], [11, 21]])
        self.assertEqual(item.artifacts, original_artifacts)

    async def test_single_segment_keeps_artifacts(self):
        loop = self._make_loop()
        state = self._make_state()
        item = self._make_item(RolloutStatus.COMPLETED, segment_count=1)

        with patch(
            "xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop.get_store",
            return_value=self._trace_store(segment_count=1),
        ):
            segments = await loop._build_rollout_states(state, item)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].extra_fields["agent_artifacts"], item.artifacts)

    async def test_failed_train_session_keeps_artifacts(self):
        loop = self._make_loop()
        state = self._make_state()
        item = self._make_item(RolloutStatus.FAILED)

        segments = await loop._build_rollout_states(state, item)

        self.assertEqual(segments, [state])
        self.assertEqual(state.status, Status.FAILED)
        self.assertEqual(state.extra_fields["agent_artifacts"], item.artifacts)

    async def test_eval_session_keeps_artifacts(self):
        loop = self._make_loop(mode="eval")
        state = self._make_state()
        item = self._make_item(RolloutStatus.COMPLETED)

        segments = await loop._build_rollout_states(state, item)

        self.assertEqual(segments, [state])
        self.assertEqual(state.extra_fields["agent_artifacts"], item.artifacts)
        self.assertEqual(state.extra_fields["agent_messages"][-1]["content"], "segment-1")


class TestTrajectoryLogging(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        self.trainer.logger = MagicMock()

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    @staticmethod
    def _make_rollout(
        *,
        rollout_id: int,
        session_id: int | None,
        reward: float,
        artifacts: dict | None,
        segment_index: int | None = None,
        segment_count: int | None = None,
        status: Status = Status.COMPLETED,
    ) -> RolloutState:
        messages = [{"role": "assistant", "content": f"segment-{segment_index}"}]
        tools = [{"type": "function", "function": {"name": "shell"}}]
        extra_fields = {
            "agent_name": "test-agent",
            "agent_status": "completed",
            "agent_tool_turns": 1,
            "agent_messages": messages,
            "agent_tools": tools,
        }
        if artifacts is not None:
            extra_fields["agent_artifacts"] = artifacts
        if segment_index is not None:
            extra_fields["agent_trace_segment_index"] = segment_index
        if segment_count is not None:
            extra_fields["agent_trace_segment_count"] = segment_count
        return RolloutState(
            group_id=3,
            rollout_id=rollout_id,
            session_id=session_id,
            message=[{"role": "user", "content": "test"}],
            num_tokens=1,
            response=f"response-{segment_index}",
            response_ids=[10, 11],
            reward={"score": reward},
            status=status,
            extra_fields=extra_fields,
        )

    def test_train_artifacts_are_written_once_per_session(self):
        artifacts = {
            "messages": [
                {"messages": [{"role": "assistant", "content": "segment-0"}], "tools": []},
                {"messages": [{"role": "assistant", "content": "segment-1"}], "tools": []},
            ],
            "response_message": {"role": "assistant", "content": "done"},
            "large_blob": "x" * 1024,
        }
        data_groups = [
            [
                self._make_rollout(
                    rollout_id=7,
                    session_id=12345678901234567890,
                    reward=1.0,
                    artifacts=artifacts,
                    segment_index=0,
                    segment_count=2,
                ),
                self._make_rollout(
                    rollout_id=7,
                    session_id=12345678901234567890,
                    reward=1.0,
                    artifacts=None,
                    segment_index=1,
                    segment_count=2,
                ),
            ]
        ]
        trajectory_path = self.output_dir / "train_rollout_1.jsonl"

        self.trainer._save_trajectories(data_groups, trajectory_path)

        objects = self._load_jsonl(trajectory_path)
        summary, rows = objects[0], objects[1:]
        self.assertNotIn("trajectory_format_version", summary)
        self.assertNotIn("session_artifacts_file", summary)
        self.assertEqual(summary["total_len"], 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["agent"]["artifacts"], artifacts)
        self.assertIsNone(rows[1]["agent"]["artifacts"])
        self.assertEqual(trajectory_path.read_text(encoding="utf-8").count('"large_blob"'), 1)
        self.assertEqual([row["agent"]["segment_index"] for row in rows], [0, 1])
        self.assertTrue(all(row["agent"]["segment_count"] == 2 for row in rows))
        self.assertTrue(all(row["agent"]["session_id"] == "12345678901234567890" for row in rows))
        self.assertEqual(rows[0]["agent"]["messages"][0]["content"], "segment-0")
        self.assertEqual(rows[1]["agent"]["messages"][0]["content"], "segment-1")
        self.assertEqual([path.name for path in self.output_dir.glob("*.jsonl")], ["train_rollout_1.jsonl"])

    def test_eval_keeps_artifacts_in_the_trajectory_file(self):
        first_artifacts = {"large_blob": "first"}
        second_artifacts = {"large_blob": "second"}
        data_groups = [
            [self._make_rollout(rollout_id=10, session_id=None, reward=0.0, artifacts=first_artifacts)],
            [self._make_rollout(rollout_id=11, session_id=222, reward=1.0, artifacts=second_artifacts)],
        ]
        trajectory_path = self.output_dir / "eval_rollout_1.jsonl"

        self.trainer._save_eval_trajectories(data_groups, trajectory_path)

        summary, *rows = self._load_jsonl(trajectory_path)
        self.assertEqual(summary["total_len"], 2)
        self.assertEqual(summary["reward_mean"], 0.5)
        self.assertEqual([row["agent"]["session_id"] for row in rows], [None, "222"])
        self.assertEqual(
            [row["agent"]["artifacts"] for row in rows],
            [first_artifacts, second_artifacts],
        )
        self.assertEqual([path.name for path in self.output_dir.glob("*.jsonl")], ["eval_rollout_1.jsonl"])

    def test_non_agent_rollout_keeps_null_agent_metadata(self):
        data = RolloutState(
            group_id=3,
            rollout_id=20,
            message=[{"role": "user", "content": "test"}],
            num_tokens=1,
            response="response",
            response_ids=[10, 11],
            reward={"score": 1.0},
            status=Status.COMPLETED,
            extra_fields={},
        )
        trajectory_path = self.output_dir / "train_rollout_1.jsonl"

        self.trainer._save_trajectories([[data]], trajectory_path)

        _, row = self._load_jsonl(trajectory_path)
        self.assertIsNone(row["agent"]["session_id"])
        self.assertIsNone(row["agent"]["segment_index"])
        self.assertIsNone(row["agent"]["segment_count"])
        self.assertIsNone(row["agent"]["artifacts"])

    def test_invalid_train_group_is_not_written(self):
        data = self._make_rollout(
            rollout_id=20,
            session_id=123,
            reward=1.0,
            artifacts={"large_blob": "discarded"},
            status=Status.FAILED,
        )
        trajectory_path = self.output_dir / "train_rollout_1.jsonl"

        self.trainer._save_trajectories([[data]], trajectory_path)

        objects = self._load_jsonl(trajectory_path)
        self.assertEqual(objects[0]["total_len"], 0)
        self.assertEqual(len(objects), 1)


if __name__ == "__main__":
    unittest.main()

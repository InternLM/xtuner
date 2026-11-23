import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status


# lagent 是 agent loop 模块的顶层可选依赖；测试只需要 create_object 的符号。
for _name in ("lagent", "lagent.utils", "lagent.utils.rate_limiter"):
    sys.modules.setdefault(_name, MagicMock())

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import AgentInSandboxLoop  # noqa: E402
from xtuner.v1.rl.agent_loop_manager.produce_utils import normalize_retry_group_inputs  # noqa: E402


def _state(
    uid: int,
    session_id: int | None,
    status: Status,
    *,
    response_ids: list[int] | None = None,
) -> RolloutState:
    return RolloutState(
        rollout_id=uid,
        message=[],
        session_id=session_id,
        status=status,
        input_ids=[uid] if status == Status.COMPLETED else None,
        response_ids=response_ids,
    )


class TestSandboxGenerateGroup(unittest.IsolatedAsyncioTestCase):
    async def test_generate_group_does_not_filter_pre_normalized_inputs(self):
        loop = AgentInSandboxLoop.__new__(AgentInSandboxLoop)
        loop.mode = "train"
        loop.sample_params = SampleParams()
        loop._sample_semaphore = None
        loop.min_valid_attempts = None
        loop.is_valid_sample_fn = None
        loop.logger = MagicMock()
        loop._teacher_scorer = SimpleNamespace(
            on_sample_ready=AsyncMock(side_effect=lambda sample: sample),
            on_group_ready=AsyncMock(side_effect=lambda group: group),
        )
        states = [_state(1, 10, Status.EXPIRED), _state(2, 20, Status.COMPLETED)]
        loop.generate_sample = AsyncMock(side_effect=lambda state: [state])
        loop.canonicalize_train_fields = lambda group: group

        samples = await loop.generate_group(states)

        self.assertEqual([state.rollout_id for state in samples], [1, 2])
        self.assertEqual([call.args[0].rollout_id for call in loop.generate_sample.await_args_list], [1])

    async def test_generate_group_expands_fresh_segments_at_original_input_position(self):
        loop = AgentInSandboxLoop.__new__(AgentInSandboxLoop)
        loop.mode = "train"
        loop.sample_params = SampleParams()
        loop._sample_semaphore = None
        loop.min_valid_attempts = None
        loop.is_valid_sample_fn = None
        loop.logger = MagicMock()
        loop._teacher_scorer = SimpleNamespace(
            on_sample_ready=AsyncMock(side_effect=lambda sample: sample),
            on_group_ready=AsyncMock(side_effect=lambda group: group),
        )
        retry = _state(1, 10, Status.EXPIRED)
        reused = _state(2, 20, Status.COMPLETED)
        first_segment = _state(11, 10, Status.COMPLETED)
        second_segment = _state(12, 10, Status.COMPLETED)
        loop.generate_sample = AsyncMock(return_value=[first_segment, second_segment])
        loop.canonicalize_train_fields = lambda group: group

        samples = await loop.generate_group([retry, reused])

        self.assertEqual([state.rollout_id for state in samples], [11, 12, 2])
        loop._teacher_scorer.on_group_ready.assert_awaited_once_with([first_segment, second_segment])


class TestRetryGroupInputNormalization(unittest.TestCase):
    def test_retries_one_expired_state_and_keeps_other_sessions(self):
        expired_first = _state(1, 10, Status.EXPIRED)
        completed_sibling = _state(2, 10, Status.COMPLETED)
        expired_sibling = _state(3, 10, Status.EXPIRED)
        aborted_sibling = _state(4, 10, Status.ABORTED)
        untouched = _state(5, 20, Status.COMPLETED)
        other_expired = _state(6, 30, Status.EXPIRED)
        no_session_expired = _state(7, None, Status.EXPIRED)
        no_session_completed = _state(8, None, Status.COMPLETED)
        states = [
            expired_first,
            completed_sibling,
            expired_sibling,
            aborted_sibling,
            untouched,
            other_expired,
            no_session_expired,
            no_session_completed,
        ]

        normalized = normalize_retry_group_inputs(states)

        self.assertEqual(
            normalized, [expired_first, untouched, other_expired, no_session_expired, no_session_completed]
        )

    def test_keeps_aborted_session_unchanged(self):
        states = [_state(1, 10, Status.COMPLETED), _state(2, 10, Status.ABORTED)]

        self.assertIs(normalize_retry_group_inputs(states), states)

    def test_keeps_one_expired_entry_per_session_in_original_order(self):
        first_session = _state(1, 10, Status.EXPIRED)
        another_session = _state(2, 20, Status.EXPIRED)
        duplicate_session = _state(3, 10, Status.EXPIRED)

        normalized = normalize_retry_group_inputs([first_session, another_session, duplicate_session])

        self.assertEqual(normalized, [first_session, another_session])


if __name__ == "__main__":
    unittest.main()

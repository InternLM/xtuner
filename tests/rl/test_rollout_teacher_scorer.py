import unittest
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.distillation import RolloutTeacherScorer


class TestRolloutTeacherScorer(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _state(uid: int, status: Status = Status.COMPLETED) -> RolloutState:
        return RolloutState(
            rollout_id=uid,
            group_id=uid,
            message=[{"role": "user", "content": f"prompt {uid}"}],
            prompt_ids=[uid],
            tokens=None,
            response=None,
            response_ids=[uid],
            status=status,
            extra_fields={"origin_data_source": "math"},
        )

    @staticmethod
    def _client(events: list[int]):
        client = MagicMock()

        async def compute_logprobs(state: RolloutState) -> RolloutState:
            events.append(state.rollout_id or 0)
            return state

        client.compute_logprobs = AsyncMock(side_effect=compute_logprobs)
        client.aclose = AsyncMock()
        return client

    async def test_scores_samples_immediately_without_filter_deferral(self):
        events: list[int] = []
        client = self._client(events)
        scorer = RolloutTeacherScorer(
            {"teacher": client},
            {"math": "teacher"},
            defers_to_filter=False,
        )

        state = await scorer.on_sample_ready(self._state(1))

        self.assertEqual(state.rollout_id, 1)
        self.assertEqual(events, [1])
        client.compute_logprobs.assert_awaited_once()

    async def test_scores_only_surviving_completed_group_when_deferred(self):
        events: list[int] = []
        client = self._client(events)
        scorer = RolloutTeacherScorer(
            {"teacher": client},
            {"math": "teacher"},
            defers_to_filter=True,
        )
        states = [self._state(1), self._state(2)]

        await scorer.on_sample_ready(states[0])
        result = await scorer.on_group_ready(states)

        self.assertEqual([state.rollout_id for state in result], [1, 2])
        self.assertEqual(sorted(events), [1, 2])
        self.assertEqual(client.compute_logprobs.await_count, 2)

    async def test_skips_non_completed_group(self):
        events: list[int] = []
        client = self._client(events)
        scorer = RolloutTeacherScorer(
            {"teacher": client},
            {"math": "teacher"},
            defers_to_filter=True,
        )

        result = await scorer.on_group_ready([self._state(1), self._state(2, Status.FAILED)])

        self.assertEqual([state.rollout_id for state in result], [1, 2])
        self.assertEqual(events, [])
        client.compute_logprobs.assert_not_awaited()

    async def test_aclose_is_idempotent(self):
        events: list[int] = []
        client = self._client(events)
        scorer = RolloutTeacherScorer(
            {"teacher": client},
            {"math": "teacher"},
            defers_to_filter=False,
        )

        await scorer.aclose()
        await scorer.aclose()

        client.aclose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()

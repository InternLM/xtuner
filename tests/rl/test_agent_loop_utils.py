import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto import RolloutState, SampleParams, Status, refresh_seq_staleness
from xtuner.v1.rl.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from xtuner.v1.rl.agent_loop.utils import PartialRolloutHandler


def _make_rollout_state(
    response_ids: list[int],
    response_model_steps: list[int] | None = None,
    seq_staleness: int = 0,
    status: Status = Status.ABORTED,
    extra_fields: dict | None = None,
):
    return RolloutState(
        uid=1,
        message=[{"role": "user", "content": "hello"}],
        prompt_ids=[101, 102],
        response_ids=response_ids,
        response="resp",
        logprobs=[0.0] * len(response_ids),
        response_mask=[1] * len(response_ids),
        response_model_steps=response_model_steps,
        seq_staleness=seq_staleness,
        sample_params=SampleParams(max_tokens=8),
        status=status,
        extra_fields=extra_fields or {},
    )


class TestAgentLoopUtils(unittest.TestCase):
    def test_refresh_seq_staleness_recomputes_from_response_model_steps(self):
        group = [_make_rollout_state(response_ids=[1, 2], response_model_steps=[3, 4], seq_staleness=0)]

        refresh_seq_staleness(group, current_train_step=8)

        self.assertEqual(group[0].seq_staleness, 4)

    def test_refresh_seq_staleness_resets_without_response_model_steps(self):
        group = [_make_rollout_state(response_ids=[1, 2], response_model_steps=None, seq_staleness=5)]

        refresh_seq_staleness(group, current_train_step=8)

        self.assertEqual(group[0].seq_staleness, 0)

    def test_partial_rollout_postprocess_only_concatenates_history(self):
        handler = PartialRolloutHandler(max_tokens=8)
        rollout_state = _make_rollout_state(
            response_ids=[30, 31],
            response_model_steps=[2, 2],
            seq_staleness=0,
            extra_fields={
                "history_response_dict": {
                    "response_ids": [10, 11],
                    "response": "hi",
                    "logprobs": [0.1, 0.2],
                    "response_mask": [1, 1],
                    "routed_experts": None,
                }
            },
        )

        result = handler.postprocess(rollout_state)

        self.assertEqual(result.response_ids, [10, 11, 30, 31])
        self.assertEqual(result.response_model_steps, [2, 2])
        self.assertEqual(result.seq_staleness, 0)


class TestSingleTurnAgentLoop(unittest.IsolatedAsyncioTestCase):
    def _build_agent_loop(self):
        rollout_ctl = MagicMock()
        rollout_ctl.generate.remote = AsyncMock()
        with (
            patch("xtuner.v1.rl.agent_loop.agent_loop.load_tokenizer", return_value=MagicMock()),
            patch("xtuner.v1.rl.agent_loop.agent_loop.load_processor", return_value=MagicMock()),
        ):
            return SingleTurnAgentLoop(
                rollout_ctl=rollout_ctl,
                sample_params=SampleParams(max_tokens=8),
                hf_checkpoint="dummy",
                judger=None,
                logger=MagicMock(),
            )

    async def test_generate_sample_does_not_update_staleness(self):
        agent_loop = self._build_agent_loop()
        rollout_state = _make_rollout_state(response_ids=[], status=Status.ABORTED)
        generated_state = _make_rollout_state(response_ids=[30, 31], seq_staleness=7, status=Status.ABORTED)
        agent_loop.rollout_ctl.generate.remote.return_value = generated_state

        result = await agent_loop.generate_sample(
            rollout_state,
        )

        self.assertIsNone(result.response_model_steps)
        self.assertEqual(result.seq_staleness, 7)

    async def test_generate_sample_does_not_update_sample_version(self):
        agent_loop = self._build_agent_loop()
        rollout_state = _make_rollout_state(response_ids=[], status=Status.ABORTED)
        generated_state = _make_rollout_state(response_ids=[30, 31], status=Status.ABORTED)
        agent_loop.rollout_ctl.generate.remote.return_value = generated_state

        result = await agent_loop.generate_sample(rollout_state)

        self.assertIsNone(result.response_model_steps)
        self.assertEqual(result.seq_staleness, 0)

    async def test_generate_sample_does_not_require_model_step(self):
        agent_loop = self._build_agent_loop()
        rollout_state = _make_rollout_state(response_ids=[], status=Status.ABORTED)
        generated_state = _make_rollout_state(response_ids=[30, 31], status=Status.ABORTED)
        agent_loop.rollout_ctl.generate.remote.return_value = generated_state

        result = await agent_loop.generate_sample(rollout_state)

        self.assertIsNone(result.response_model_steps)
        self.assertEqual(result.seq_staleness, 0)

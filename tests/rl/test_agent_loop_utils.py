import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from xtuner.v1.rl.agent_loop.utils import PartialRolloutHandler, refresh_seq_staleness


def _make_rollout_state(
    response_ids: list[int],
    response_rollout_steps: list[int] | None = None,
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
        response_rollout_steps=response_rollout_steps,
        seq_staleness=seq_staleness,
        sample_params=SampleParams(max_tokens=8),
        status=status,
        extra_fields=extra_fields or {},
    )


class TestAgentLoopUtils(unittest.TestCase):
    def test_refresh_seq_staleness_recomputes_from_response_rollout_steps(self):
        group = [_make_rollout_state(response_ids=[1, 2], response_rollout_steps=[3, 4], seq_staleness=0)]

        refresh_seq_staleness(group, current_rollout_step=8)

        self.assertEqual(group[0].seq_staleness, 5)

    def test_partial_rollout_postprocess_uses_model_rollout_step(self):
        handler = PartialRolloutHandler(max_tokens=8)
        rollout_state = _make_rollout_state(
            response_ids=[30, 31],
            response_rollout_steps=[2, 2],
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

        result = handler.postprocess(rollout_state, model_rollout_step=5, current_rollout_step=9)

        self.assertEqual(result.response_ids, [10, 11, 30, 31])
        self.assertEqual(result.response_rollout_steps, [2, 2, 5, 5])
        self.assertEqual(result.seq_staleness, 7)


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

    async def test_generate_sample_requires_both_step_kwargs(self):
        agent_loop = self._build_agent_loop()
        rollout_state = _make_rollout_state(response_ids=[], status=Status.ABORTED)

        with self.assertRaises(ValueError):
            await agent_loop.generate_sample(rollout_state, rollout_step=9)

        with self.assertRaises(ValueError):
            await agent_loop.generate_sample(rollout_state, model_rollout_step=5)

    async def test_generate_sample_uses_model_rollout_step_for_tokens_and_rollout_step_for_staleness(self):
        agent_loop = self._build_agent_loop()
        rollout_state = _make_rollout_state(response_ids=[], status=Status.ABORTED)
        generated_state = _make_rollout_state(response_ids=[30, 31], status=Status.ABORTED)
        agent_loop.rollout_ctl.generate.remote.return_value = generated_state

        result = await agent_loop.generate_sample(
            rollout_state,
            model_rollout_step=5,
            rollout_step=9,
        )

        self.assertEqual(result.response_rollout_steps, [5, 5])
        self.assertEqual(result.seq_staleness, 4)

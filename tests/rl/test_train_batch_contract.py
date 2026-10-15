"""Canonical field validation, staleness metadata and reusable packing layouts."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    Status,
    validate_response_model_steps,
    validate_train_fields,
    write_train_meta,
)
from xtuner.v1.rl.agent_loop_manager.produce_utils import _TaskRunner, take_train_batch
from xtuner.v1.rl.agent_loop_manager.producer import ProduceProgress
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.trainer.controller import TrainingController


def canonical_state(uid=1):
    state = RolloutState(
        rollout_id=uid,
        group_id=uid,
        message=[{"role": "user", "content": "prompt"}],
        prompt_ids=[10, 11],
        response_ids=[20, 21, 22, 23],
        input_ids=[10, 11, 20, 21, 22, 23],
        labels=[-100, -100, 20, -100, 22, 23],
        response="answer",
        status=Status.COMPLETED,
        reward={"score": 1.0},
    )
    write_train_meta(state)
    return state


class TestTrainFields(unittest.TestCase):
    def test_optional_logprobs_and_versions(self):
        state = canonical_state()
        validate_train_fields(state)
        validate_response_model_steps(state)
        state.logprobs = [0.0] * 6
        state.response_model_steps = [1] * 4
        validate_train_fields(state)
        validate_response_model_steps(state)

    def test_invalid_training_fields_include_context(self):
        for field, value in (
            ("input_ids", None),
            ("input_ids", [1]),
            ("labels", None),
            ("labels", [1]),
            ("logprobs", [0.0]),
        ):
            with self.subTest(field=field, value=value):
                state = canonical_state()
                setattr(state, field, value)
                with self.assertRaisesRegex(ValueError, f"{field}.*task='math', rollout_id=1"):
                    validate_train_fields(state, task_name="math")


class TestTrainBatchContract(unittest.IsolatedAsyncioTestCase):
    async def take(self, buffer, threshold):
        runner = _TaskRunner(
            task_name="math",
            agent_loop=MagicMock(),
            sampler=MagicMock(),
            produce_strategy=SimpleNamespace(token_stale_threshold=threshold),
        )
        return await take_train_batch(
            task_runners=[runner],
            replay_buffer=buffer,
            logger=MagicMock(),
            manager_name="test",
            task_batch_sizes={"math": 1},
            progress=ProduceProgress.build(task_names=["math"], target_samples={"math": 1}),
            current_train_step=2,
        )

    async def test_staleness_refreshes_only_supervision_count_idempotently(self):
        buffer = SyncReplayBufferConfig().build()
        state = canonical_state()
        # Two fresh supervised tokens, one stale token, and one existing semantic hole.
        state.response_model_steps = [0, 0, 2, 2]
        old_meta = dict(state.extra_fields)
        old_length = state.num_tokens
        for _ in range(2):
            await buffer.put([state], "math")
            result = await self.take(buffer, 1)
            state = result.rollout_states[0][0]
            self.assertEqual(state.labels, [-100, -100, -100, -100, 22, 23])
            self.assertEqual(state.extra_fields["supervised_tokens"], 2)
            self.assertEqual(state.num_tokens, old_length)
            self.assertEqual(state.extra_fields, {**old_meta, "supervised_tokens": 2})

    async def test_training_entry_rejects_custom_loop_fields_without_staleness(self):
        buffer = SyncReplayBufferConfig().build()
        state = canonical_state()
        state.labels = [20]
        await buffer.put([state], "math")
        with self.assertRaisesRegex(ValueError, "labels length mismatch.*task='math'"):
            await self.take(buffer, None)

    async def test_training_entry_checks_versions_before_mask_calculation(self):
        buffer = SyncReplayBufferConfig().build()
        state = canonical_state()
        state.response_model_steps = [2] * 4
        await buffer.put([state], "math")
        # Simulate a payload changed after insertion; the training entry must still reject it.
        state.response_model_steps.append(2)
        with patch("xtuner.v1.rl.agent_loop_manager.produce_utils.calculate_group_effective_response_masks") as masks:
            with self.assertRaisesRegex(ValueError, "response_model_steps length mismatch"):
                await self.take(buffer, 1)
            masks.assert_not_called()

    async def test_replay_fills_partial_versions_before_validation(self):
        buffer = SyncReplayBufferConfig().build()
        state = canonical_state()
        state.response_model_steps = [0]
        await buffer.put([state], "math", model_step=2, current_train_step=2)
        self.assertEqual(state.response_model_steps, [0, 2, 2, 2])
        result = await self.take(buffer, None)
        self.assertEqual(result.rollout_states[0][0].response_model_steps, [0, 2, 2, 2])

    async def test_replay_rejects_bad_versions_before_staleness(self):
        buffer = SyncReplayBufferConfig().build()
        state = canonical_state()
        state.response_model_steps = [2] * 5
        with patch("xtuner.v1.rl.replay_buffer.calculate_group_effective_response_masks") as masks:
            with self.assertRaisesRegex(ValueError, "response_model_steps length mismatch.*task='math'"):
                await buffer.put([state], "math", model_step=2, current_train_step=2)
            masks.assert_not_called()
        self.assertEqual(len(buffer), 0)

    async def test_unfinished_states_do_not_require_training_fields_in_replay(self):
        buffer = SyncReplayBufferConfig().build()
        state = RolloutState(message=[], rollout_id=9, status=Status.ABORTED, response_ids=[])
        await buffer.put([state], "math", model_step=2)
        self.assertEqual(await buffer.count("math", Status.ABORTED), 1)


class TestPackPlanDispatch(unittest.TestCase):
    def test_layout_without_advantages_preserves_mapping_when_targets_change(self):
        controller = TrainingController(workers=[object()] * 4)
        states = [canonical_state(i) for i in range(5)]
        for state, length in zip(states, [6, 4, 7, 3, 8]):
            state.num_tokens = length
        config = SimpleNamespace(optimizer_steps=2, pack_strategy="greedy", pack_seed=None)
        with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
            plan = controller._build_pack_plan(states, 10, config, 2)
        self.assertEqual(plan["dp_plans"][0]["global_indices"], [0, 1, 4])
        self.assertEqual(plan["dp_plans"][0]["plan"], [[[0, 1]], [[2]]])
        self.assertEqual(plan["dp_plans"][1]["plan"], [[[0, 1]], [[]]])
        for advantages in ([1.0, 2.0, 3.0, 4.0, 5.0], [-5.0, -4.0, -3.0, -2.0, -1.0]):
            dispatches = controller._build_dispatches(states, advantages, plan)
            for rank, dp_plan in plan["dp_plans"].items():
                dispatch = dispatches[rank]
                self.assertIs(dispatch["plan"], dp_plan["plan"])
                self.assertEqual(dispatch["advantages"], [advantages[i] for i in dp_plan["global_indices"]])
                self.assertEqual(dispatch["rollout_items"], [states[i] for i in dp_plan["global_indices"]])
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            controller._build_dispatches(states, [], plan)
        with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
            padding_plan = controller._build_pack_plan(states[:1], 10, config, 1)
        dispatches = controller._build_dispatches(states[:1], [1.0], padding_plan)
        self.assertEqual(dispatches[3]["rollout_items"], [])
        self.assertEqual(dispatches[3]["advantages"], [])
        self.assertEqual(dispatches[3]["plan"], [[[]]])

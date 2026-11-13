"""Canonical field validation, staleness metadata and reusable packing layouts."""

import asyncio
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status, validate_rollout_group, validate_train_group
from xtuner.v1.rl.agent_loop_manager.produce_utils import ProduceBatchResult, _TaskRunner, take_train_batch
from xtuner.v1.rl.agent_loop_manager.producer import ProduceProgress
from xtuner.v1.rl.distillation import DistillationTrainerAdapter
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.train.rl_trainer import BaseRLTrainer


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
    state.response_model_steps = [1] * 4
    state.set_training_fields(input_ids=state.input_ids, labels=state.labels, logprobs=state.logprobs)
    return state


class TestTrainFields(unittest.TestCase):
    def admit(self, group):
        return validate_rollout_group(group, task_name="math")

    def test_optional_logprobs(self):
        state = canonical_state()
        validate_train_group([state], task_name="math")
        state.logprobs = [0.0] * 6
        state.set_training_fields(input_ids=state.input_ids, labels=state.labels, logprobs=state.logprobs)
        validate_train_group([state], task_name="math")

    def test_invalid_training_fields_include_context(self):
        for field, value in (
            ("input_ids", None),
            ("input_ids", [1]),
            ("labels", None),
            ("labels", [1]),
            ("logprobs", [0.0]),
        ):
            with self.subTest(field=field):
                state = canonical_state()
                setattr(state, field, value)
                self.admit([state])
                with self.assertRaisesRegex(ValueError, f"task='math'.*rollout_id=1.*{field}"):
                    validate_train_group([state], task_name="math")

    def test_public_validation_does_not_fill_versions(self):
        state = canonical_state()
        state.response_model_steps = [0]
        self.assertIsNone(self.admit([state]))
        self.assertEqual(state.response_model_steps, [0])

    def test_training_validation_does_not_infer_missing_historical_versions(self):
        state = canonical_state()
        state.response_model_steps = [0]
        with self.assertRaisesRegex(ValueError, "response_model_steps length"):
            validate_train_group([state], task_name="math")
        self.assertEqual(state.response_model_steps, [0])

    def test_eval_and_partial_results_need_no_train_payload(self):
        for status in (Status.COMPLETED, Status.ABORTED, Status.EXPIRED):
            with self.subTest(status=status):
                state = RolloutState(message=[], status=status, response_ids=[1])
                self.admit([state])
                self.assertIsNone(state.input_ids)
                self.assertIsNone(state.response_model_steps)

    def test_producer_rejects_inactive_result_without_mutating_it(self):
        for status in (Status.INIT, Status.ARCHIVED):
            state = RolloutState(message=[], status=status, response_ids=[1])
            with self.assertRaisesRegex(ValueError, "invalid generated status"):
                self.admit([state])
            self.assertEqual(state.status, status)
            self.assertIsNone(state.response_model_steps)

    def test_training_validation_rejects_missing_or_stale_metadata(self):
        for key, invalid in (
            ("prompt_length", -1),
            ("response_length", 100),
            ("supervised_tokens", 100),
            ("position_layout", "mrope_3d"),
            ("has_rollout_logprobs", True),
            ("has_routed_experts", True),
        ):
            for missing in (False, True):
                with self.subTest(key=key, missing=missing):
                    state = canonical_state()
                    if missing:
                        state.train_meta = None
                    else:
                        assert state.train_meta is not None
                        state.train_meta = replace(state.train_meta, **{key: invalid})
                    self.admit([state])
                    with self.assertRaisesRegex(
                        ValueError, "must be present|prompt_length|Response suffix|inconsistent"
                    ):
                        validate_train_group([state], task_name="math")

    def test_training_validation_rejects_stale_token_count(self):
        state = canonical_state()
        state.num_tokens += 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_train_group([state], task_name="math")

    def test_public_validation_rejects_versions_longer_than_response(self):
        state = canonical_state()
        state.response_model_steps = [1] * 5
        with self.assertRaisesRegex(ValueError, "response_model_steps length=5.*response_ids=4"):
            self.admit([state])
        self.assertEqual(state.response_model_steps, [1] * 5)


class TestTrainBatchContract(unittest.IsolatedAsyncioTestCase):
    async def take(self, buffer, threshold):
        runner = _TaskRunner(
            task_name="math",
            agent_loop=MagicMock(),
            sampler=MagicMock(),
            produce_strategy=SimpleNamespace(token_stale_threshold=threshold),
        )
        result = await take_train_batch(
            task_runners=[runner],
            replay_buffer=buffer,
            logger=MagicMock(),
            manager_name="test",
            task_batch_sizes={"math": 1},
            progress=ProduceProgress.build(task_names=["math"], target_samples={"math": 1}),
            current_train_step=2,
        )

        trainer = SimpleNamespace(logger=MagicMock())
        await asyncio.to_thread(BaseRLTrainer._prepare_train_batch, trainer, result.rollout_states, 2, result)
        return result

    async def test_staleness_refreshes_only_supervision_count_idempotently(self):
        buffer = SyncReplayBufferConfig().build()
        state = canonical_state()
        # Two fresh supervised tokens, one stale token, and one existing semantic hole.
        state.response_model_steps = [0, 0, 2, 2]
        old_meta = state.train_meta
        old_length = state.num_tokens
        for _ in range(2):
            await buffer.put([state], "math")
            result = await self.take(buffer, 1)
            state = result.rollout_states[0][0]
            self.assertEqual(state.labels, [-100, -100, -100, -100, 22, 23])
            assert state.train_meta is not None and old_meta is not None
            self.assertEqual(state.train_meta.supervised_tokens, 2)
            self.assertEqual(state.num_tokens, old_length)
            self.assertEqual(replace(state.train_meta, supervised_tokens=old_meta.supervised_tokens), old_meta)
            self.assertEqual(state.extra_fields, {})

    async def test_training_boundary_discards_entire_invalid_group(self):
        bad, sibling, good = canonical_state(1), canonical_state(2), canonical_state(3)
        bad.labels = [20]
        invalid_group = [bad, sibling]
        trainer = SimpleNamespace(logger=MagicMock())
        batch = [invalid_group, [good]]
        with patch("xtuner.v1.train.rl_trainer.release_and_discard_rollout_groups", new=AsyncMock()) as release:
            dropped = await asyncio.to_thread(BaseRLTrainer._prepare_train_batch, trainer, batch, 2)
        self.assertEqual(dropped, (1, 2))
        self.assertEqual(batch, [[good]])
        release.assert_awaited_once_with([invalid_group])
        trainer.logger.warning.assert_called_once()

    async def test_all_invalid_groups_are_released_before_empty_batch_error(self):
        state = canonical_state()
        state.num_tokens += 1
        trainer = SimpleNamespace(logger=MagicMock())
        batch = [[state]]
        with patch("xtuner.v1.train.rl_trainer.release_and_discard_rollout_groups", new=AsyncMock()) as release:
            with self.assertRaisesRegex(ValueError, "No valid training data"):
                await asyncio.to_thread(BaseRLTrainer._prepare_train_batch, trainer, batch, 2)
        release.assert_awaited_once()
        self.assertEqual(batch, [])

    async def test_training_check_preserves_existing_label_masks(self):
        state = canonical_state()
        state.task_name = "math"
        state.response_model_steps = [0, 0, 2, 2]
        state.labels = [-100, -100, -100, -100, 22, 23]
        state.set_training_fields(input_ids=state.input_ids, labels=state.labels, logprobs=state.logprobs)
        trainer = SimpleNamespace(logger=MagicMock())
        batch = [[state]]
        await asyncio.to_thread(BaseRLTrainer._prepare_train_batch, trainer, batch, 2)
        self.assertEqual(state.labels, [-100, -100, -100, -100, 22, 23])
        assert state.train_meta is not None
        self.assertEqual(state.train_meta.supervised_tokens, 2)

    async def test_invalid_groups_removed_from_task_results(self):
        bad, good = canonical_state(1), canonical_state(2)
        bad.labels = None
        result = ProduceBatchResult(
            rollout_states=[[bad], [good]],
            task_results={"math": ProduceBatchResult(rollout_states=[[bad], [good]])},
        )
        trainer = SimpleNamespace(logger=MagicMock())
        with patch("xtuner.v1.train.rl_trainer.release_and_discard_rollout_groups", new=AsyncMock()):
            await asyncio.to_thread(BaseRLTrainer._prepare_train_batch, trainer, result.rollout_states, 2, result)
        self.assertEqual(result.rollout_states, [[good]])
        self.assertEqual(result.task_results["math"].rollout_states, [[good]])

    async def test_shared_batch_retrieval_does_not_require_or_mutate_train_fields(self):
        buffer = SyncReplayBufferConfig().build()
        state = RolloutState(message=[], status=Status.COMPLETED, response_ids=[1])
        await buffer.put([state], "math", model_step=0)
        runner = _TaskRunner(
            task_name="math",
            agent_loop=MagicMock(),
            sampler=MagicMock(),
            produce_strategy=SimpleNamespace(token_stale_threshold=1),
        )
        result = await take_train_batch(
            task_runners=[runner],
            replay_buffer=buffer,
            logger=MagicMock(),
            manager_name="eval",
            task_batch_sizes={"math": 1},
            progress=ProduceProgress.build(task_names=["math"], target_samples={"math": 1}),
            current_train_step=2,
        )
        self.assertEqual(result.rollout_states, [[state]])
        self.assertIsNone(state.labels)
        self.assertEqual(state.extra_fields, {})

    async def test_replay_fills_partial_versions_before_validation(self):
        buffer = SyncReplayBufferConfig().build()
        state = canonical_state()
        state.response_model_steps = [0]
        await buffer.put([state], "math", model_step=2, current_train_step=2)
        self.assertEqual(state.response_model_steps, [0, 2, 2, 2])
        result = await self.take(buffer, None)
        self.assertEqual(result.rollout_states[0][0].response_model_steps, [0, 2, 2, 2])

    async def test_bad_versions_are_discarded_at_training_boundary(self):
        buffer = SyncReplayBufferConfig().build()
        state = canonical_state()
        state.response_model_steps = [2] * 5
        await buffer.put([state], "math", model_step=2, current_train_step=2)
        self.assertEqual(len(buffer), 1)
        with self.assertRaisesRegex(ValueError, "No valid training data"):
            await self.take(buffer, None)
        self.assertEqual(len(buffer), 0)


class TestBatchLifecycle(unittest.TestCase):
    @staticmethod
    def _batch_attr():
        return {
            "rollout_idx": 0,
            "use_3d_position_ids": False,
            "pack_loss_keys": [],
            "has_routed_experts": False,
        }

    def test_loaded_batch_maps_states_per_rank_and_releases(self):
        from xtuner.v1.rl.trainer.pack import PackPlan, RankPlan

        workers = [SimpleNamespace(), SimpleNamespace()]
        calls = {"load": [], "release": 0}
        for worker in workers:
            worker.load_train_batch = SimpleNamespace(
                remote=lambda refs, rank, attr: calls["load"].append((refs, rank))
            )

        def release():
            calls["release"] += 1

        for worker in workers:
            worker.release_train_batch = SimpleNamespace(remote=release)

        controller = TrainingController.__new__(TrainingController)
        controller.workers = workers
        controller.distillation = DistillationTrainerAdapter(None)
        controller.logger = MagicMock()
        states = [canonical_state(uid=index) for index in range(3)]
        plan = PackPlan(
            ranks=(
                RankPlan(global_indices=(2, 0), steps=(((0, 1),),)),
                RankPlan(global_indices=(1,), steps=(((0,),),)),
            ),
            metrics={},
        )

        with (
            patch("xtuner.v1.rl.trainer.controller.ray.put", side_effect=lambda value: ("put", value)),
            patch("xtuner.v1.rl.trainer.controller.ray.get", side_effect=lambda value, timeout=None: value),
        ):
            with controller._loaded_batch(states, plan, self._batch_attr(), data_replicate_size=1):
                pass

        self.assertEqual(calls["release"], 2)
        self.assertEqual([rank.global_indices for _, rank in calls["load"]], [(2, 0), (1,)])
        shipped = calls["load"][0][0][0][1]
        self.assertEqual(len(shipped), 2)
        for original, slim in zip((states[2], states[0]), shipped):
            self.assertIsNot(slim, original)
            self.assertEqual(slim.input_ids, original.input_ids)
            self.assertEqual(slim.response, None)

    def test_loaded_batch_releases_on_failure(self):
        from xtuner.v1.rl.trainer.pack import PackPlan, RankPlan

        worker = SimpleNamespace()
        released = []
        worker.load_train_batch = SimpleNamespace(remote=lambda *args: None)
        worker.release_train_batch = SimpleNamespace(remote=lambda: released.append(True))
        controller = TrainingController.__new__(TrainingController)
        controller.workers = [worker]
        controller.distillation = DistillationTrainerAdapter(None)
        controller.logger = MagicMock()
        plan = PackPlan(ranks=(RankPlan(global_indices=(0,), steps=(((0,),),)),), metrics={})

        with (
            patch("xtuner.v1.rl.trainer.controller.ray.put", side_effect=lambda value: value),
            patch("xtuner.v1.rl.trainer.controller.ray.get", side_effect=lambda value, timeout=None: value),
            self.assertRaises(RuntimeError),
        ):
            with controller._loaded_batch([canonical_state()], plan, self._batch_attr(), data_replicate_size=1):
                raise RuntimeError("boom")

        self.assertEqual(released, [True])

    def test_release_train_batch_resources_skips_trace_owned_routed_refs(self):
        from unittest.mock import patch as mock_patch

        import ray

        from xtuner.v1.train.rl_trainer import BaseRLTrainer

        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer.logger = MagicMock()
        trace_ref = ray.ObjectRef(b"a" * 28)
        plain_ref = ray.ObjectRef(b"b" * 28)
        pixel_ref = ray.ObjectRef(b"c" * 28)
        video_ref = ray.ObjectRef(b"d" * 28)
        traced = canonical_state(1)
        traced.session_id = 7
        traced.routed_experts = trace_ref
        plain = canonical_state(2)
        plain.routed_experts = [plain_ref]
        plain.mm_info = {"pixel_values": pixel_ref, "pixel_values_videos": video_ref}

        with (
            mock_patch.object(BaseRLTrainer, "_release_trace_sessions", return_value={"7"}),
            mock_patch("xtuner.v1.train.rl_trainer.free_object_refs") as free_refs,
        ):
            trainer._release_train_batch_resources([[traced], [plain]])

        freed = free_refs.call_args.args[0]
        self.assertIn(plain_ref, freed)
        self.assertIn(pixel_ref, freed)
        self.assertIn(video_ref, freed)
        self.assertNotIn(trace_ref, freed)


class TestPackPlanDispatch(unittest.TestCase):
    def test_layout_maps_global_indices_to_local_packs(self):
        controller = TrainingController(workers=[object()] * 4)
        states = [canonical_state(i) for i in range(5)]
        for state, length in zip(states, [6, 4, 7, 3, 8]):
            state.num_tokens = length
        config = SimpleNamespace(
            optimizer_steps=2, pack_strategy="legacy", pack_seed=None, pack_extra_buffer_size=1000
        )
        with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
            plan = controller._build_pack_plan(states, 10, config, 2)
        self.assertEqual(plan.ranks[0].global_indices, (0, 1, 4))
        self.assertEqual(plan.ranks[0].steps, (((0, 1),), ((2,),)))
        self.assertEqual(plan.ranks[1].steps, (((0, 1),), ((),)))
        with patch("xtuner.v1.rl.trainer.pack.XTUNER_DETERMINISTIC", True):
            padding_plan = controller._build_pack_plan(states[:1], 10, config, 1)
        self.assertEqual(padding_plan.ranks[3].global_indices, ())
        self.assertEqual(padding_plan.ranks[3].steps, (((),),))

    def test_restore_rows_rebuilds_global_order(self):
        from xtuner.v1.rl.trainer.pack import PackPlan, RankPlan, TokenBatchProto

        plan = PackPlan(
            ranks=(
                RankPlan(global_indices=(2, 0), steps=(((0,), (1,)),)),
                RankPlan(global_indices=(1,), steps=(((0,),),)),
            ),
            metrics={},
        )
        rows = plan.restore_rows(
            [
                TokenBatchProto.concat([torch.tensor([2.0]), torch.tensor([0.0])]),
                TokenBatchProto.concat([torch.tensor([1.0])]),
            ]
        )
        self.assertEqual([row.tolist() for row in rows], [[0.0], [1.0], [2.0]])


class TestTrainingPayloadSlimming(unittest.TestCase):
    """``RolloutState.training_copy`` ships only what training workers read."""

    def heavy_state(self):
        state = canonical_state(uid=7)
        state.message = [{"role": "user", "content": "long prompt" * 100}]
        state.prompt_ids = [10, 11]
        state.tokens = [10, 11, 20, 21]
        state.tools = [{"type": "function", "function": {"name": "search"}}]
        state.tool_choice = "auto"
        state.tool_calls = [{"name": "search", "arguments": "{}"}]
        state.reward_model = {"ground_truth": "42"}
        state.extra_fields = {
            "origin_data_source": "gsm8k",
            "agent_messages": [{"role": "assistant", "content": "long trace" * 100}],
            "agent_artifacts": {"log": "x" * 1000},
        }
        return state

    def test_training_copy_strips_trainer_only_payloads(self):
        state = self.heavy_state()
        slim = state.training_copy()

        self.assertIsNot(slim, state)
        self.assertEqual(slim.message, [])
        for field in (
            "prompt_ids",
            "tokens",
            "tools",
            "tool_choice",
            "response",
            "tool_calls",
            "reward_model",
            "response_model_steps",
        ):
            self.assertIsNone(getattr(slim, field), field)
        self.assertEqual(slim.extra_fields, {})

    def test_training_copy_keeps_training_payloads(self):
        state = self.heavy_state()
        slim = state.training_copy()

        self.assertEqual(slim.input_ids, state.input_ids)
        self.assertEqual(slim.labels, state.labels)
        self.assertEqual(slim.logprobs, state.logprobs)
        self.assertEqual(slim.reward, state.reward)
        self.assertEqual(slim.train_meta, state.train_meta)
        self.assertEqual(slim.rollout_id, state.rollout_id)
        self.assertEqual(slim.group_id, state.group_id)
        self.assertEqual(slim.seq_staleness, state.seq_staleness)

    def test_training_copy_does_not_mutate_original(self):
        state = self.heavy_state()
        slim = state.training_copy(extra_field_keys=("origin_data_source",))

        self.assertEqual(state.message, [{"role": "user", "content": "long prompt" * 100}])
        self.assertEqual(state.response, "answer")
        self.assertEqual(state.extra_fields["agent_artifacts"], {"log": "x" * 1000})
        self.assertEqual(slim.extra_fields, {"origin_data_source": "gsm8k"})

    def test_training_copy_missing_extra_field_key_maps_to_none(self):
        slim = self.heavy_state().training_copy(extra_field_keys=("origin_data_source", "absent"))
        self.assertEqual(slim.extra_fields, {"origin_data_source": "gsm8k", "absent": None})

    def test_adapter_needs_rollout_extra_fields(self):
        self.assertFalse(DistillationTrainerAdapter(None).needs_rollout_extra_fields)
        self.assertFalse(DistillationTrainerAdapter(SimpleNamespace(train_teachers=[])).needs_rollout_extra_fields)
        self.assertTrue(
            DistillationTrainerAdapter(SimpleNamespace(train_teachers=["teacher"])).needs_rollout_extra_fields
        )

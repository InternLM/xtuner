import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import ray
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop_manager import AsyncProduceStrategyConfig, ProduceBatchResult
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import AgentLoopManager, _TaskRunner
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SerializedRayObjectRef
from xtuner.v1.rl.rollout import RolloutEndpoint
from xtuner.v1.train.rl_trainer import RLColocateTrainer


class _FakeRolloutState:
    def __init__(self, uid: int):
        self.id = uid
        self.uid = str(uid)
        self.message_uid = uid
        self.status = Status.INIT
        self.seq_staleness = 0
        self.response_ids = []
        self.response = None
        self.reward = None
        self.extra_fields = {}
        self.response_model_steps = []


class _FakeSampler:
    def __init__(self):
        self._next_id = 0

    def __len__(self):
        return 8

    def save(self, checkpoint_path):
        return None

    def resume(self, checkpoint_path):
        return None

    async def sample(self, task_name, group_status=None, **kwargs):
        item = _FakeRolloutState(self._next_id)
        self._next_id += 1
        return [item]


def _build_fake_agent_loop():
    agent_loop = MagicMock()

    async def generate_group(rollout_states, **kwargs):
        model_step = kwargs.get("model_step", kwargs.get("train_step", 0))
        for state in rollout_states:
            state.status = Status.COMPLETED
            state.response_ids = [1, 2, 3]
            state.response = "ok"
            state.reward = {"score": 1.0}
            state.response_model_steps = [model_step]
        return rollout_states

    agent_loop.generate_group = generate_group
    return agent_loop


def _fake_rollout_endpoint():
    rollout_controller = MagicMock()
    rollout_controller.continue_generation.remote = AsyncMock(return_value=None)
    rollout_controller.pause_generation.remote = AsyncMock(return_value=None)
    rollout_controller.get_rollout_metadata.remote = AsyncMock(return_value={"server_url_dict": {}})
    return RolloutEndpoint(kind="worker_extern", base_url="http://rollout-router", rollout_controller=rollout_controller)


class TestRLColocateTrainer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_trainer(self, agent_loop_manager, *, total_train_steps: int = 1, sync_weights_interval: int = 1):
        trainer = RLColocateTrainer.__new__(RLColocateTrainer)
        trainer.logger = MagicMock()
        trainer._total_train_steps = total_train_steps
        trainer._cur_step = 0
        trainer._global_train_step = 0
        trainer.train_batch_size = 1
        trainer._sync_weights_interval = sync_weights_interval
        trainer._debug_rollout = False
        trainer._debug_rollout_dir = None
        trainer._debug_train = False
        trainer._enable_evaluate = False
        trainer._enable_initial_evaluate = False
        trainer._evaluate_step = 1
        trainer._train_worker_cfg = SimpleNamespace(pack_max_length=16)
        trainer._meta = SimpleNamespace(
            latest_exp=SimpleNamespace(exp_dir=str(Path(self.temp_dir.name) / "exp")),
        )
        Path(trainer.exp_dir).mkdir(parents=True, exist_ok=True)
        trainer.agent_loop_manager = agent_loop_manager
        trainer.eval_agent_loop_manager = MagicMock()
        trainer.evaluator = MagicMock(eval_batch_size=1)
        trainer.tokenizer = MagicMock()
        trainer._exp_tracker = MagicMock()
        trainer._display_all_workers_log = False
        trainer._save_trajectories = MagicMock()
        trainer._sync_weights_and_save = MagicMock(
            side_effect=lambda train_step, step_timer_dict: train_step % trainer._sync_weights_interval == 0
        )
        trainer._log_step = MagicMock()
        trainer._prepare_train_data = MagicMock(
            return_value=([{"seq_ctx": "fake"}], {"batch_size": 1, "rewards/mean": 1.0})
        )

        trainer.rollout_controller = SimpleNamespace(
            offload=SimpleNamespace(remote=MagicMock(return_value="rollout_offloaded")),
        )
        trainer.train_controller = SimpleNamespace(
            onload=MagicMock(return_value="train_onloaded"),
            fit=MagicMock(
                return_value=[
                    {
                        "rollout_is_metrics": {},
                        "mismatch_metrics": {},
                        "rollout_entropy": 0.0,
                        "train_entropy": 0.0,
                        "train_metrics": [],
                        "sft_train_metrics": {},
                    }
                ]
            ),
        )
        return trainer

    def test_fit_accepts_async_strategy_manager_on_colocate_path(self):
        replay_buffer = AsyncReplayBufferConfig().build()
        manager = AgentLoopManager(
            rollout_endpoint=_fake_rollout_endpoint(),
            task_runners=[
                _TaskRunner(
                    task_name="train_task",
                    agent_loop=_build_fake_agent_loop(),
                    produce_strategy=AsyncProduceStrategyConfig(over_sample_threshold=0.0).build(),
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                )
            ],
            replay_buffer=replay_buffer,
        )
        trainer = self._make_trainer(manager)

        with (
            patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj: obj),
        ):
            trainer.fit()

        trainer.rollout_controller.offload.remote.assert_called_once_with()
        trainer.train_controller.onload.assert_called_once_with(target="all")
        trainer.train_controller.fit.assert_called_once()
        trainer._prepare_train_data.assert_called_once()
        trainer._save_trajectories.assert_called_once()
        trainer._sync_weights_and_save.assert_called_once()
        trainer._log_step.assert_called_once()
        self.assertEqual(trainer._cur_step, 1)

    def test_fit_requires_non_empty_batch_from_manager(self):
        async def _produce_empty(batch_size, train_step, **kwargs):
            return ProduceBatchResult(rollout_states=[])

        empty_manager = SimpleNamespace(produce_batch=_produce_empty)
        trainer = self._make_trainer(empty_manager)

        with (
            patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj: obj),
        ):
            with self.assertRaisesRegex(AssertionError, "return non-empty rollout_states"):
                trainer.fit()

        trainer.rollout_controller.offload.remote.assert_not_called()
        trainer.train_controller.onload.assert_not_called()
        trainer.train_controller.fit.assert_not_called()
        trainer._prepare_train_data.assert_not_called()
        trainer._save_trajectories.assert_not_called()
        trainer._sync_weights_and_save.assert_not_called()
        trainer._log_step.assert_not_called()
        self.assertEqual(trainer._cur_step, 0)

    def test_fit_uses_sync_interval_and_passes_rollout_model_step(self):
        produce_calls = []

        async def _produce_batch(batch_size, train_step, *, model_step):
            produce_calls.append((batch_size, train_step, model_step))
            return ProduceBatchResult(
                rollout_states=[[SimpleNamespace(message_uid=train_step, uid=train_step)]]
            )

        trainer = self._make_trainer(
            SimpleNamespace(produce_batch=_produce_batch),
            total_train_steps=3,
            sync_weights_interval=2,
        )
        with (
            patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj: obj),
        ):
            trainer.fit()

        self.assertEqual(produce_calls, [(1, 1, 0), (1, 2, 0), (1, 3, 2)])
        self.assertEqual(
            [call.args[0] for call in trainer._sync_weights_and_save.call_args_list],
            [1, 2, 3],
        )
        self.assertEqual(trainer._cur_step, 3)

    def test_debug_rollout_saves_raw_batch_and_skips_training(self):
        rollout_state = RolloutState(
            message=[{"role": "user", "content": "hello"}],
            response="ok",
            response_ids=[1, 2],
            reward={"score": 1.0},
            status=Status.COMPLETED,
        )

        async def _produce_batch(batch_size, train_step, *, model_step):
            return ProduceBatchResult(rollout_states=[[rollout_state]])

        trainer = self._make_trainer(SimpleNamespace(produce_batch=_produce_batch))
        trainer._debug_rollout = True
        trainer._debug_rollout_dir = Path(self.temp_dir.name) / "debug_rollout"

        with (
            patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj: obj),
        ):
            trainer.fit()

        saved_path = trainer._debug_rollout_dir / "debug_rollout_1.pt"
        self.assertTrue(saved_path.exists())
        saved_batch = torch.load(saved_path, map_location="cpu", weights_only=False)
        self.assertEqual(saved_batch[0][0].response, "ok")
        trainer.train_controller.fit.assert_not_called()
        trainer._sync_weights_and_save.assert_not_called()

    def test_debug_train_loads_batches_and_skips_weight_sync(self):
        trainer = self._make_trainer(MagicMock(), total_train_steps=2)
        trainer._debug_train = True
        trainer._load_debug_rollout_batch = MagicMock(
            side_effect=[
                [[SimpleNamespace(uid=1, message_uid=1)]],
                [[SimpleNamespace(uid=2, message_uid=2)]],
            ]
        )
        trainer._train_one_batch = MagicMock(
            return_value={
                "data_info": {"batch_size": 1},
                "workers_log_item": [
                    {
                        "rollout_is_metrics": {},
                        "mismatch_metrics": {},
                        "rollout_entropy": 0.0,
                        "train_entropy": 0.0,
                        "train_metrics": [],
                        "sft_train_metrics": {},
                    }
                ],
            }
        )

        trainer.fit()

        self.assertEqual([call.args[0] for call in trainer._load_debug_rollout_batch.call_args_list], [1, 2])
        self.assertEqual(trainer._train_one_batch.call_count, 2)
        trainer._sync_weights_and_save.assert_not_called()
        self.assertEqual(trainer._cur_step, 2)

    def test_debug_rollout_save_resolves_object_refs_and_load_puts_them_back(self):
        if not ray.is_initialized():
            try:
                ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
            except Exception as exc:
                self.skipTest(f"Ray init failed in this test environment: {exc}")
        try:
            pixel_values = torch.ones(1, 2)
            routed_experts = [1, 2, 3]
            rollout_state = RolloutState(
                message=[{"role": "user", "content": "hello"}],
                mm_info={"pixel_values": ray.put(pixel_values)},
                routed_experts=ray.put(routed_experts),
                response="ok",
                response_ids=[1],
                reward={"score": 1.0},
                status=Status.COMPLETED,
            )
            trainer = self._make_trainer(MagicMock())
            trainer._debug_rollout_dir = Path(self.temp_dir.name) / "debug_refs"
            trainer._save_debug_rollout_batch([[rollout_state]], train_step=1)

            saved_batch = torch.load(
                trainer._debug_rollout_dir / "debug_rollout_1.pt",
                map_location="cpu",
                weights_only=False,
            )
            saved_pixel_values = saved_batch[0][0].mm_info["pixel_values"]
            saved_routed_experts = saved_batch[0][0].routed_experts
            self.assertFalse(isinstance(saved_pixel_values, ray.ObjectRef))
            self.assertFalse(isinstance(saved_routed_experts, ray.ObjectRef))
            self.assertIsInstance(saved_pixel_values, SerializedRayObjectRef)
            self.assertIsInstance(saved_routed_experts, SerializedRayObjectRef)
            self.assertTrue(torch.equal(saved_pixel_values.value, pixel_values))
            self.assertEqual(saved_routed_experts.value, routed_experts)

            trainer._debug_train_files = {1: trainer._debug_rollout_dir / "debug_rollout_1.pt"}
            loaded_batch = trainer._load_debug_rollout_batch(train_step=1)
            self.assertIsInstance(loaded_batch[0][0].mm_info["pixel_values"], ray.ObjectRef)
            self.assertIsInstance(loaded_batch[0][0].routed_experts, ray.ObjectRef)
            self.assertTrue(torch.equal(ray.get(loaded_batch[0][0].mm_info["pixel_values"]), pixel_values))
            self.assertEqual(ray.get(loaded_batch[0][0].routed_experts), routed_experts)
        finally:
            ray.shutdown()

    def test_sync_weights_and_save_can_skip_weight_update_and_restore_rollout(self):
        trainer = RLColocateTrainer.__new__(RLColocateTrainer)
        events = []
        trainer._sync_weights_interval = 2
        trainer._maybe_save_checkpoint = MagicMock(side_effect=lambda step: events.append(f"save:{step}"))
        trainer._maybe_save_hf = MagicMock(side_effect=lambda step: events.append(f"hf:{step}"))
        trainer.train_controller = SimpleNamespace(
            update_weights=MagicMock(side_effect=lambda: events.append("update_weights")),
            offload=MagicMock(side_effect=lambda target="all": events.append(("train_offload", target))),
        )
        trainer.rollout_controller = SimpleNamespace(
            recover_failed_workers=SimpleNamespace(
                remote=MagicMock(side_effect=lambda: events.append("recover_rollout"))
            ),
            onload_weights=SimpleNamespace(remote=MagicMock(side_effect=lambda: events.append("onload_weights"))),
            onload_kvcache=SimpleNamespace(remote=MagicMock(side_effect=lambda: events.append("onload_kvcache"))),
        )

        with (
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj: obj),
            patch(
                "xtuner.v1.train.rl_trainer.bind_train_rollout",
                side_effect=lambda train_controller, rollout_controller: events.append("bind"),
            ),
        ):
            synced = trainer._sync_weights_and_save(train_step=1, step_timer_dict={})

        self.assertFalse(synced)
        self.assertEqual(
            events,
            [
                ("train_offload", "optimizer"),
                "save:1",
                "hf:1",
                "recover_rollout",
                ("train_offload", "model"),
                "onload_weights",
                "onload_kvcache",
            ],
        )

    def test_sync_weights_and_save_updates_weights_on_interval_step(self):
        trainer = RLColocateTrainer.__new__(RLColocateTrainer)
        events = []
        trainer.logger = MagicMock()
        trainer._sync_weights_interval = 2
        trainer._maybe_save_checkpoint = MagicMock(side_effect=lambda step: events.append(f"save:{step}"))
        trainer._maybe_save_hf = MagicMock(side_effect=lambda step: events.append(f"hf:{step}"))
        trainer.train_controller = SimpleNamespace(
            update_weights=MagicMock(side_effect=lambda: events.append("update_weights")),
            offload=MagicMock(side_effect=lambda target="all": events.append(("train_offload", target))),
        )
        trainer.rollout_controller = SimpleNamespace(
            recover_failed_workers=SimpleNamespace(
                remote=MagicMock(side_effect=lambda: events.append("recover_rollout"))
            ),
            onload_weights=SimpleNamespace(remote=MagicMock(side_effect=lambda: events.append("onload_weights"))),
            onload_kvcache=SimpleNamespace(remote=MagicMock(side_effect=lambda: events.append("onload_kvcache"))),
        )

        with (
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj: obj),
            patch(
                "xtuner.v1.train.rl_trainer.bind_train_rollout",
                side_effect=lambda train_controller, rollout_controller: events.append("bind"),
            ),
        ):
            synced = trainer._sync_weights_and_save(train_step=2, step_timer_dict={})

        self.assertTrue(synced)
        self.assertEqual(
            events,
            [
                ("train_offload", "optimizer"),
                "save:2",
                "hf:2",
                "recover_rollout",
                "bind",
                "onload_weights",
                "update_weights",
                ("train_offload", "model"),
                "onload_kvcache",
            ],
        )


if __name__ == "__main__":
    unittest.main()

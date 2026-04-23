import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto import Status
from xtuner.v1.rl.agent_loop_manager import AsyncProduceStrategyConfig, ProduceBatchResult
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import AgentLoopManager, _TaskRunner
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainer


class _FakeRolloutState:
    def __init__(self, uid: int):
        self.id = uid
        self.uid = str(uid)
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
    rollout_ctl = MagicMock()
    rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
    rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
    rollout_ctl.get_rollout_metadata.remote = AsyncMock(return_value={"server_url_dict": {}})
    agent_loop = MagicMock()
    agent_loop.rollout_ctl = rollout_ctl

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
            return ProduceBatchResult(rollout_states=[[f"sample-{train_step}"]])

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

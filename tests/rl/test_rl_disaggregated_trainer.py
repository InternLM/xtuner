import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerStatus,
    ProduceBatchResult,
    ProduceBatchStatus,
)
from xtuner.v1.train.rl_trainer import RLDisaggregatedTrainer, _validate_sync_intervals


class _FakeRolloutState:
    def __init__(self, uid: int):
        self.message_uid = uid
        self.uid = uid


class _FakeManager:
    def __init__(self, get_batch_results):
        self._results = list(get_batch_results)
        self._status = AgentLoopManagerStatus.NORMAL
        self._finish_event = asyncio.Event()
        self.calls: list[object] = []

    async def produce_loop(self, batch_size: int):
        self.calls.append(("produce_loop_start", batch_size))
        await self._finish_event.wait()
        self.calls.append("produce_loop_exit")

    async def get_batch(self, batch_size: int, train_step: int):
        self.calls.append(("get_batch", batch_size, train_step))
        return self._results.pop(0)

    async def pause_produce(self, *, use_global_progress: bool):
        self.calls.append(("pause_produce", use_global_progress))
        return 0.25

    def continue_produce(self, model_step: int):
        self.calls.append(("continue_produce", model_step))


class TestRLDisaggregatedTrainer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_trainer(self, agent_loop_manager):
        trainer = RLDisaggregatedTrainer.__new__(RLDisaggregatedTrainer)
        trainer.logger = MagicMock()
        trainer._cur_step = 0
        trainer._total_train_steps = 1
        trainer._global_train_step = 0
        trainer.train_batch_size = 2
        trainer._sync_weights_interval = 1
        trainer._enable_evaluate = False
        trainer._enable_initial_evaluate = False
        trainer._evaluate_step = 1
        trainer._debug_rollout = False
        trainer._display_all_workers_log = False
        trainer._train_worker_cfg = SimpleNamespace(pack_max_length=16)
        trainer._meta = SimpleNamespace(
            latest_exp=SimpleNamespace(exp_dir=str(Path(self.temp_dir.name) / "exp")),
        )
        Path(trainer.exp_dir).mkdir(parents=True, exist_ok=True)
        trainer.agent_loop_manager = agent_loop_manager
        trainer.eval_agent_loop_manager = SimpleNamespace(produce_batch=AsyncMock())
        trainer.evaluator = MagicMock(eval_batch_size=1, run=MagicMock(return_value={"acc": 1.0}))
        trainer._exp_tracker = MagicMock()
        trainer._prepare_train_data = MagicMock(
            return_value=([{"seq_ctx": "fake"}], {"batch_size": 1, "rewards/mean": 1.0})
        )
        trainer._save_trajectories = MagicMock()
        trainer._log_step = MagicMock()
        trainer._maybe_save_checkpoint = MagicMock()
        trainer._maybe_save_hf = MagicMock()
        trainer.fake_update_weights = MagicMock()
        trainer.train_controller = SimpleNamespace(
            fit=MagicMock(return_value=[{"train_metrics": [], "sft_train_metrics": {}}]),
            onload=MagicMock(return_value="onload"),
            offload=MagicMock(return_value="offload"),
            update_weights=MagicMock(return_value="update"),
        )
        trainer.rollout_controller = SimpleNamespace(
            recover_failed_workers=SimpleNamespace(remote=MagicMock(return_value="recover")),
            onload_weights=SimpleNamespace(remote=MagicMock(return_value="onload_weights")),
            onload_kvcache=SimpleNamespace(remote=MagicMock(return_value="onload_kvcache")),
        )
        return trainer

    def test_sync_weights_and_save_saves_before_fake_update(self):
        manager = _FakeManager([])
        trainer = self._make_trainer(manager)
        events: list[str] = []
        trainer._maybe_save_checkpoint = MagicMock(side_effect=lambda step: events.append(f"save:{step}"))
        trainer._maybe_save_hf = MagicMock(side_effect=lambda step: events.append(f"hf:{step}"))
        trainer.fake_update_weights = MagicMock(side_effect=lambda: events.append("fake_update"))

        with (
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
            patch(
                "xtuner.v1.train.rl_trainer.bind_train_rollout",
                side_effect=lambda train_controller, rollout_controller: events.append("bind"),
            ),
        ):
            asyncio.run(trainer._sync_weights_and_save(train_step=3, step_timer_dict={}))

        self.assertEqual(events, ["save:3", "hf:3", "bind", "fake_update"])
        trainer.train_controller.offload.assert_not_called()

    def test_fit_skips_train_when_batch_is_expired(self):
        manager = _FakeManager(
            [ProduceBatchResult(rollout_states=[], status=ProduceBatchStatus.EXPIRED_BATCH)]
        )
        trainer = self._make_trainer(manager)
        trainer._sync_weights_and_save = AsyncMock()

        asyncio.run(trainer._fit())

        trainer._prepare_train_data.assert_not_called()
        trainer.train_controller.fit.assert_not_called()
        trainer._sync_weights_and_save.assert_awaited_once()
        self.assertIn(("continue_produce", 1), manager.calls)
        self.assertIn("produce_loop_exit", manager.calls)

    def test_fit_runs_eval_before_reset_and_stops_producer(self):
        manager = _FakeManager(
            [ProduceBatchResult(rollout_states=[[_FakeRolloutState(1)]], status=ProduceBatchStatus.NORMAL)]
        )
        trainer = self._make_trainer(manager)
        trainer._enable_evaluate = True
        events: list[str] = []

        async def sync_weights_and_save(train_step: int, step_timer_dict: dict):
            events.append("sync")

        async def eval_produce_batch(batch_size: int, train_step: int, model_step: int):
            events.append("eval")
            return ProduceBatchResult(rollout_states=[[_FakeRolloutState(2)]])

        def continue_produce(model_step: int):
            events.append("continue_produce")
            manager.calls.append(("continue_produce", model_step))

        trainer._sync_weights_and_save = AsyncMock(side_effect=sync_weights_and_save)
        trainer.eval_agent_loop_manager.produce_batch = AsyncMock(side_effect=eval_produce_batch)
        trainer.evaluator.run = MagicMock(return_value={"acc": 1.0})
        manager.continue_produce = continue_produce

        with patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj):
            asyncio.run(trainer._fit())

        trainer._prepare_train_data.assert_called_once()
        trainer.train_controller.fit.assert_called_once()
        trainer.train_controller.onload.assert_not_called()
        self.assertEqual(events, ["sync", "eval", "continue_produce"])
        self.assertTrue(manager._finish_event.is_set())
        self.assertIn("produce_loop_exit", manager.calls)

    def test_fake_update_weights_does_not_onload_rollout(self):
        manager = _FakeManager([])
        trainer = self._make_trainer(manager)

        with patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj):
            trainer.fake_update_weights = RLDisaggregatedTrainer.fake_update_weights.__get__(
                trainer, RLDisaggregatedTrainer
            )
            trainer.fake_update_weights()

        trainer.train_controller.update_weights.assert_called_once_with()
        trainer.rollout_controller.onload_weights.remote.assert_not_called()
        trainer.rollout_controller.onload_kvcache.remote.assert_not_called()

    def test_resume_from_checkpoint_syncs_weights_then_resets_manager(self):
        trainer = RLDisaggregatedTrainer.__new__(RLDisaggregatedTrainer)
        trainer.logger = MagicMock()
        trainer._load_checkpoint_cfg = SimpleNamespace(checkpoint_path=Path(self.temp_dir.name))
        trainer.train_controller = SimpleNamespace(resume=MagicMock(return_value="resume"))
        trainer.rollout_controller = SimpleNamespace()
        events: list[str] = []

        def manager_resume(checkpoint_path):
            events.append(f"manager_resume:{Path(checkpoint_path).name}")
            return 3

        def manager_continue_produce(model_step: int):
            events.append(f"continue_produce:{model_step}")

        trainer.agent_loop_manager = SimpleNamespace(
            resume=MagicMock(side_effect=manager_resume),
            continue_produce=MagicMock(side_effect=manager_continue_produce),
        )
        trainer.fake_update_weights = MagicMock(side_effect=lambda: events.append("fake_update"))

        train_state_path = Path(self.temp_dir.name) / trainer._SAVE_TRAIN_STATE_PATH
        train_state_path.write_text('{"cur_step": 3}')

        with (
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
            patch(
                "xtuner.v1.train.rl_trainer.bind_train_rollout",
                side_effect=lambda train_controller, rollout_controller: events.append("bind"),
            ),
        ):
            trainer._resume_from_checkpoint(self.temp_dir.name)

        trainer.train_controller.resume.assert_called_once_with(trainer._load_checkpoint_cfg)
        self.assertEqual(trainer._cur_step, 3)
        trainer.agent_loop_manager.resume.assert_called_once_with(Path(self.temp_dir.name))
        self.assertTrue(events[0].startswith("manager_resume:"))
        self.assertEqual(events[1:], ["bind", "fake_update", "continue_produce:3"])

    def test_validate_sync_schedule_accepts_multiples(self):
        _validate_sync_intervals(sync_weights_interval=2, checkpoint_interval=4, hf_interval=6)
        _validate_sync_intervals(sync_weights_interval=2, checkpoint_interval=-1, hf_interval=None)
        _validate_sync_intervals(
            sync_weights_interval=2,
            checkpoint_interval=-1,
            hf_interval=None,
            evaluate_step=4,
            enable_evaluate=True,
        )

    def test_validate_sync_schedule_rejects_non_multiple_checkpoint_interval(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_interval=5.*sync_weights_interval=2"):
            _validate_sync_intervals(sync_weights_interval=2, checkpoint_interval=5, hf_interval=-1)

    def test_validate_sync_schedule_rejects_non_multiple_hf_interval(self):
        with self.assertRaisesRegex(ValueError, "hf_interval=5.*sync_weights_interval=2"):
            _validate_sync_intervals(sync_weights_interval=2, checkpoint_interval=4, hf_interval=5)

    def test_validate_sync_schedule_rejects_non_multiple_evaluate_step(self):
        with self.assertRaisesRegex(ValueError, "evaluate_step=5.*sync_weights_interval=2"):
            _validate_sync_intervals(
                sync_weights_interval=2,
                checkpoint_interval=4,
                hf_interval=6,
                evaluate_step=5,
                enable_evaluate=True,
            )

    def test_build_disaggregated_placement_groups_uses_distinct_names(self):
        trainer = RLDisaggregatedTrainer.__new__(RLDisaggregatedTrainer)
        trainer.logger = MagicMock()
        trainer._meta = SimpleNamespace(
            latest_exp=SimpleNamespace(exp_dir=str(Path(self.temp_dir.name) / "20260416130000")),
        )
        train_pg = SimpleNamespace(id="train-pg-id")
        rollout_pg = SimpleNamespace(id="rollout-pg-id")

        with patch(
            "xtuner.v1.train.rl_trainer.AutoAcceleratorWorkers.build_placement_group",
            side_effect=[train_pg, rollout_pg],
        ) as build_pg:
            built_train_pg, built_rollout_pg = trainer._build_disaggregated_placement_groups(
                train_resources=object(),
                rollout_resources=object(),
            )

        self.assertIs(built_train_pg, train_pg)
        self.assertIs(built_rollout_pg, rollout_pg)
        self.assertEqual(
            build_pg.call_args_list[0].kwargs["name"],
            "xtuner_rl_disagg_20260416130000_train",
        )
        self.assertEqual(
            build_pg.call_args_list[1].kwargs["name"],
            "xtuner_rl_disagg_20260416130000_rollout",
        )

    def test_build_disaggregated_placement_groups_rejects_reused_pg(self):
        trainer = RLDisaggregatedTrainer.__new__(RLDisaggregatedTrainer)
        trainer.logger = MagicMock()
        trainer._meta = SimpleNamespace(
            latest_exp=SimpleNamespace(exp_dir=str(Path(self.temp_dir.name) / "20260416130000")),
        )
        shared_pg = SimpleNamespace(id="shared-pg-id")

        with patch(
            "xtuner.v1.train.rl_trainer.AutoAcceleratorWorkers.build_placement_group",
            side_effect=[shared_pg, shared_pg],
        ):
            with self.assertRaisesRegex(RuntimeError, "distinct placement groups"):
                trainer._build_disaggregated_placement_groups(
                    train_resources=object(),
                    rollout_resources=object(),
                )


if __name__ == "__main__":
    unittest.main()

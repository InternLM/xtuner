import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.rl.agent_loop import ProduceBatchResult, ProduceBatchStatus
from xtuner.v1.rl.agent_loop.agent_loop_manager import AgentLoopManagerStatus
from xtuner.v1.train.rl_disaggregated_trainer import RLDisaggregatedTrainer


class _FakeManager:
    def __init__(self, get_batch_results):
        self._results = list(get_batch_results)
        self._status = AgentLoopManagerStatus.NORMAL
        self._finish_event = asyncio.Event()
        self.calls: list[object] = []

    async def produce_loop(self, batch_size: int, start_rollout_step: int = 0):
        self.calls.append(("produce_loop_start", batch_size, start_rollout_step))
        await self._finish_event.wait()
        self.calls.append("produce_loop_exit")

    async def get_batch(self, batch_size: int, rollout_step: int):
        self.calls.append(("get_batch", batch_size, rollout_step))
        return self._results.pop(0)

    async def cleanup_pending_tasks(self, pause_product_for_update: bool = False):
        self.calls.append(("cleanup_pending_tasks", pause_product_for_update))
        return 0.25

    def continue_product(self, model_rollout_step: int):
        self.calls.append(("continue_product", model_rollout_step))


class TestRLDisaggregatedTrainer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_trainer(self, agent_loop_manager):
        trainer = RLDisaggregatedTrainer.__new__(RLDisaggregatedTrainer)
        trainer.logger = MagicMock()
        trainer._cur_step = 0
        trainer._rollout_steps = 1
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
            fit=SimpleNamespace(remote=MagicMock(return_value=[{"train_metrics": [], "sft_train_metrics": {}}])),
            offload=SimpleNamespace(remote=MagicMock(return_value="offload")),
        )
        trainer.rollout_controller = SimpleNamespace(
            recover_failed_workers=SimpleNamespace(remote=MagicMock(return_value="recover")),
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
            patch("xtuner.v1.train.rl_disaggregated_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
            patch(
                "xtuner.v1.train.rl_disaggregated_trainer.bind_train_rollout",
                side_effect=lambda train_controller, rollout_controller: events.append("bind"),
            ),
        ):
            asyncio.run(trainer._sync_weights_and_save(rollout_idx=3, step_timer_dict={}))

        self.assertEqual(events, ["save:3", "hf:3", "bind", "fake_update"])

    def test_fit_skips_train_when_batch_is_expired(self):
        manager = _FakeManager(
            [ProduceBatchResult(rollout_states=[], status=ProduceBatchStatus.EXPIRED_BATCH)]
        )
        trainer = self._make_trainer(manager)
        trainer._sync_weights_and_save = AsyncMock()

        asyncio.run(trainer._fit())

        trainer._prepare_train_data.assert_not_called()
        trainer.train_controller.fit.remote.assert_not_called()
        trainer._sync_weights_and_save.assert_awaited_once()
        self.assertIn(("continue_product", 1), manager.calls)
        self.assertIn("produce_loop_exit", manager.calls)

    def test_fit_runs_eval_before_reset_and_stops_producer(self):
        manager = _FakeManager(
            [ProduceBatchResult(rollout_states=[["sample"]], status=ProduceBatchStatus.NORMAL)]
        )
        trainer = self._make_trainer(manager)
        trainer._enable_evaluate = True
        events: list[str] = []

        async def sync_weights_and_save(rollout_idx: int, step_timer_dict: dict):
            events.append("sync")

        async def eval_produce_batch(batch_size: int, rollout_step: int):
            events.append("eval")
            return ProduceBatchResult(rollout_states=[["eval"]])

        def continue_product(model_rollout_step: int):
            events.append("continue_product")
            manager.calls.append(("continue_product", model_rollout_step))

        trainer._sync_weights_and_save = AsyncMock(side_effect=sync_weights_and_save)
        trainer.eval_agent_loop_manager.produce_batch = AsyncMock(side_effect=eval_produce_batch)
        trainer.evaluator.run = MagicMock(return_value={"acc": 1.0})
        manager.continue_product = continue_product

        with patch("xtuner.v1.train.rl_disaggregated_trainer.ray.get", side_effect=lambda obj, timeout=None: obj):
            asyncio.run(trainer._fit())

        trainer._prepare_train_data.assert_called_once()
        trainer.train_controller.fit.remote.assert_called_once()
        self.assertEqual(events, ["sync", "eval", "continue_product"])
        self.assertTrue(manager._finish_event.is_set())
        self.assertIn("produce_loop_exit", manager.calls)


if __name__ == "__main__":
    unittest.main()

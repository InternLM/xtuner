"""RLDisaggregatedTrainer 的 public 行为测试。

Good Tests:
- 通过 fit()、update_weights()、同步周期校验和资源布局不变量验证行为。
- 用 _FakeManager 和轻量 controller 替代真实 Ray worker。
- 只断言 step 递进、producer 恢复 model_step、checkpoint 文件等可观察结果。
- 对 async producer/consumer 只验证最终业务结果；内部任务编排放到 AgentLoopManager 测试中。

Bad Tests:
- 不直接测试 _sync_weights_and_save、_log_step、_resume_from_checkpoint 等私有 helper。
- 不把 save/sync/eval 的内部调用顺序当成契约。
- 只保留 eval 先于 continue_produce 这类对外可见顺序。
- 不重复验证 RLThroughputBenchmark schema；统一放在 colocate trainer 测试中覆盖。

本文件主要覆盖的 public 行为:
- disaggregated fit 遇到空 EXPIRED_BATCH 时重试同一个 train_step，不推进 _cur_step。
- 非空 EXPIRED_BATCH 仍会训练，并用当前完成的 model_step 恢复 producer。
- checkpoint 保存发生在 fit 完成的 model_step 上，且 manager.save 为 async 调用。
- eval 在 producer 恢复前运行；update_weights 本身不直接 pause/continue rollout controller。
- sync/checkpoint/eval interval 必须是 sync_weights_interval 的整数倍，资源布局必须 fail fast。
"""

import asyncio
import json
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

    async def continue_produce(self, model_step: int):
        self.calls.append(("continue_produce", model_step))

    def shutdown(self):
        self.calls.append("shutdown")
        self._status = AgentLoopManagerStatus.FINISH
        self._finish_event.set()


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
        trainer._num_workers = 1.0
        trainer._rollout_num_workers = 1.0
        trainer._benchmark_start_time_s = 100.0
        trainer._benchmark_training_samples = 0
        trainer._benchmark_training_tokens = 0
        trainer._cpu_resource_manager = None
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
        trainer._maybe_save_checkpoint = AsyncMock()
        trainer._maybe_save_hf = MagicMock()
        trainer.train_controller = SimpleNamespace(
            fit=MagicMock(return_value=[{"train_metrics": [], "sft_train_metrics": {}}]),
            onload=MagicMock(return_value="onload"),
            offload=MagicMock(return_value="offload"),
            update_weights=MagicMock(return_value="update"),
        )
        trainer.rollout_controller = SimpleNamespace(
            recover_failed_workers=SimpleNamespace(remote=MagicMock(return_value="recover")),
            pause_generation=SimpleNamespace(remote=MagicMock(return_value="pause")),
            continue_generation=SimpleNamespace(remote=MagicMock(return_value="continue")),
            onload_weights=SimpleNamespace(remote=MagicMock(return_value="onload_weights")),
            onload_kvcache=SimpleNamespace(remote=MagicMock(return_value="onload_kvcache")),
        )
        return trainer

    def _run_fit(self, trainer):
        with patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run):
            trainer.fit()

    def test_fit_persists_checkpoint_for_completed_model_step(self):
        # 验证 checkpoint 以 fit 完成的 model_step 为准，并通过 async manager.save 落盘。
        train_sample = SimpleNamespace(message_uid=1, uid=1)
        manager = _FakeManager([ProduceBatchResult(rollout_states=[[train_sample]])])
        manager.save = AsyncMock()
        trainer = self._make_trainer(manager)
        trainer._checkpoint_interval = 1
        trainer._checkpoint_maxkeep = None
        trainer._checkpoint_no_save_optimizer = False
        trainer._hf_interval = -1
        trainer._meta_path = ".xtuner_rl_disaggregated_trainer"
        trainer._meta.latest_exp.checkpoint_list = []
        trainer._meta.model_dump_json = MagicMock(return_value="{}")
        trainer.train_controller.save = MagicMock()
        trainer._maybe_save_checkpoint = RLDisaggregatedTrainer._maybe_save_checkpoint.__get__(
            trainer, RLDisaggregatedTrainer
        )

        with (
            patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
            patch("xtuner.v1.train.rl_trainer.bind_train_rollout"),
        ):
            trainer.fit()

        checkpoint_path = Path(trainer.exp_dir) / trainer._CHECKPOINT_DIR / "ckpt-step-1"
        manager.save.assert_awaited_once_with(checkpoint_path, model_step=1)
        trainer.train_controller.save.assert_called_once_with(str(checkpoint_path), False)
        with (checkpoint_path / trainer._SAVE_TRAIN_STATE_PATH).open("r") as f:
            self.assertEqual(json.load(f), {"cur_step": 1})
        self.assertIn(("continue_produce", 1), manager.calls)
        self.assertEqual(trainer._cur_step, 1)

    def test_fit_retries_same_step_after_empty_expired_skip(self):
        # 验证空 expired batch 只同步上一版模型，不推进 train_step，并重试同一步。
        train_sample = SimpleNamespace(message_uid=1, uid=1)
        manager = _FakeManager(
            [
                ProduceBatchResult(rollout_states=[], status=ProduceBatchStatus.EXPIRED_BATCH),
                ProduceBatchResult(rollout_states=[[train_sample]], status=ProduceBatchStatus.NORMAL),
            ]
        )
        trainer = self._make_trainer(manager)
        trainer._total_train_steps = 2
        trainer._cur_step = 1
        trainer._sync_weights_and_save = AsyncMock()

        self._run_fit(trainer)

        self.assertEqual(
            [call for call in manager.calls if isinstance(call, tuple) and call[0] == "get_batch"],
            [("get_batch", 2, 2), ("get_batch", 2, 2)],
        )
        trainer.train_controller.fit.assert_called_once()
        self.assertIn(("continue_produce", 1), manager.calls)
        self.assertIn(("continue_produce", 2), manager.calls)
        self.assertEqual(trainer._cur_step, 2)
        self.assertIn("produce_loop_exit", manager.calls)

    def test_fit_trains_non_empty_expired_batch_then_syncs_current_step(self):
        # 验证非空 expired batch 仍会训练，并用当前完成的 model_step 恢复 producer。
        train_sample = SimpleNamespace(message_uid=1, uid=1)
        manager = _FakeManager(
            [ProduceBatchResult(rollout_states=[[train_sample]], status=ProduceBatchStatus.EXPIRED_BATCH)]
        )
        trainer = self._make_trainer(manager)
        trainer._sync_weights_and_save = AsyncMock()

        self._run_fit(trainer)

        trainer.train_controller.fit.assert_called_once()
        self.assertIn(("continue_produce", 1), manager.calls)
        self.assertEqual(trainer._cur_step, 1)

    def test_fit_runs_eval_before_reset_and_stops_producer(self):
        # 验证 eval 在 producer 恢复前执行，避免生产侧提前抢占 rollout 资源。
        # 确定性排序依赖 RolloutState 的 message_uid 和 uid，测试用轻量对象模拟即可。
        train_sample = SimpleNamespace(message_uid=1, uid=1)
        eval_sample = SimpleNamespace(message_uid=2, uid=2)
        manager = _FakeManager(
            [ProduceBatchResult(rollout_states=[[train_sample]], status=ProduceBatchStatus.NORMAL)]
        )
        trainer = self._make_trainer(manager)
        trainer._enable_evaluate = True
        events: list[str] = []

        async def sync_weights_and_save(train_step: int, step_timer_dict: dict):
            events.append("sync")

        async def eval_produce_batch(batch_size: int, train_step: int, model_step: int):
            events.append("eval")
            return ProduceBatchResult(rollout_states=[[eval_sample]])

        async def continue_produce(model_step: int):
            events.append("continue_produce")
            manager.calls.append(("continue_produce", model_step))

        trainer._sync_weights_and_save = AsyncMock(side_effect=sync_weights_and_save)
        trainer.eval_agent_loop_manager.produce_batch = AsyncMock(side_effect=eval_produce_batch)
        trainer.evaluator.run = MagicMock(return_value={"acc": 1.0})
        manager.continue_produce = continue_produce

        self._run_fit(trainer)

        trainer.train_controller.fit.assert_called_once()
        trainer.train_controller.onload.assert_not_called()
        self.assertEqual(events, ["sync", "eval", "continue_produce"])
        self.assertTrue(manager._finish_event.is_set())
        self.assertIn("produce_loop_exit", manager.calls)

    def test_validate_sync_schedule_accepts_multiples(self):
        # 验证保存、HF 导出、评测周期都可以对齐 sync_weights_interval。
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
        # 验证 checkpoint_interval 不能落在非权重同步 step 上。
        with self.assertRaisesRegex(ValueError, "checkpoint_interval=5.*sync_weights_interval=2"):
            _validate_sync_intervals(sync_weights_interval=2, checkpoint_interval=5, hf_interval=-1)

    def test_validate_sync_schedule_rejects_non_multiple_hf_interval(self):
        # 验证 hf_interval 不能落在非权重同步 step 上。
        with self.assertRaisesRegex(ValueError, "hf_interval=5.*sync_weights_interval=2"):
            _validate_sync_intervals(sync_weights_interval=2, checkpoint_interval=4, hf_interval=5)

    def test_validate_sync_schedule_rejects_non_multiple_evaluate_step(self):
        # 验证 evaluate_step 不能落在非权重同步 step 上。
        with self.assertRaisesRegex(ValueError, "evaluate_step=5.*sync_weights_interval=2"):
            _validate_sync_intervals(
                sync_weights_interval=2,
                checkpoint_interval=4,
                hf_interval=6,
                evaluate_step=5,
                enable_evaluate=True,
            )

    def test_resource_layout_uses_distinct_train_and_rollout_group_names(self):
        # 验证 train/rollout placement group 使用不同名字，避免 Ray 复用同一组资源。
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

    def test_resource_layout_fails_fast_when_train_and_rollout_share_group(self):
        # 验证 Ray 返回同一 placement group 时 fail fast，并打印可排查的不变量。
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

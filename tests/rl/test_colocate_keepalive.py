"""CPU-only contracts: persistent Agent owner and alternating gate phases."""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import AgentLoopManager
from xtuner.v1.rl.agent_loop_manager.colocate_keepalive import ColocateKeepaliveManager, LoopThread
from xtuner.v1.rl.agent_loop_manager.produce_utils import _PendingTasks


class OwnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_task_progresses_while_sync_training_blocks_main_thread(self):
        owner = LoopThread()
        state = {}
        tool_done = threading.Event()

        async def first_window():
            state["loop"] = asyncio.get_running_loop()
            state["pending"] = _PendingTasks()

            async def agent():
                await asyncio.sleep(0.03)
                state["same_loop"] = asyncio.get_running_loop() is state["loop"]
                tool_done.set()
                await asyncio.Event().wait()

            await state["pending"].schedule_one(
                max_pending=1, should_abort=lambda: False, spawn_one=lambda: self.spawn(agent())
            )

        await owner.call(first_window())
        time.sleep(0.1)  # stand in for native synchronous FSDP work
        self.assertTrue(tool_done.is_set())
        self.assertTrue(state["same_loop"])
        self.assertEqual(state["pending"].count(), 1)
        inner = SimpleNamespace(
            task_runners=[SimpleNamespace(produce_strategy=SimpleNamespace(_pending_tasks=state["pending"]))]
        )
        proxy = ColocateKeepaliveManager(inner, owner)
        await proxy.close()
        await proxy.close()
        self.assertFalse(owner.thread.is_alive())

    async def spawn(self, coro):
        return asyncio.create_task(coro)

    async def test_manager_methods_share_owner_loop(self):
        owner = LoopThread()
        loops = []

        async def operation(*args, **kwargs):
            loops.append(asyncio.get_running_loop())
            return 7

        inner = SimpleNamespace(task_runners=[], produce_batch=operation, save=operation, resume=operation)
        proxy = ColocateKeepaliveManager(inner, owner)
        try:
            self.assertEqual(await proxy.produce_batch(), 7)
            await proxy.save()
            await proxy.resume()
            self.assertTrue(all(loop is owner.loop for loop in loops))
        finally:
            await proxy.close()


class BoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_quiesce_finishes_episode_with_open_gate_before_backend_pause(self):
        events = []
        release = asyncio.Event()
        owner = _PendingTasks()

        async def episode():
            await release.wait()
            events.append("episode_finished")
            return []

        async def spawn():
            return asyncio.create_task(episode())

        await owner.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn)
        strategy = SimpleNamespace(
            _pending_tasks=owner,
            produce_batch=AsyncMock(side_effect=lambda ctx: release.set()),
            pending_task_count=owner.count,
        )
        task = SimpleNamespace(
            task_name="task",
            produce_strategy=strategy,
            agent_loop=object(),
            sampler=object(),
            weight=1.0,
            order=0,
            stale_threshold=None,
            token_stale_threshold=None,
            expired_groups_retryable=True,
        )
        manager = AgentLoopManager([task], object(), object(), logger=MagicMock())

        async def resume(*args):
            events.append("gate_open")

        async def pause(*args):
            self.assertEqual(owner.count(), 0)
            events.append("gate_seal_backend_pause")

        manager._keepalive = SimpleNamespace(resume=resume, check_requests=AsyncMock(), pause=pause, quiesce_timeout=1)
        result = SimpleNamespace(rollout_states=[[object()]])
        with (
            patch("xtuner.v1.rl.agent_loop_manager.agent_loop_manager.refresh_for_all_tasks", new=AsyncMock()),
            patch(
                "xtuner.v1.rl.agent_loop_manager.agent_loop_manager.take_train_batch",
                new=AsyncMock(return_value=result),
            ),
            patch("xtuner.v1.rl.agent_loop_manager.producer.ProduceContext.put_generated_group", new=AsyncMock()),
        ):
            await manager.produce_batch(1, 1, model_step=0, quiesce=True)
        self.assertEqual(events, ["gate_open", "episode_finished", "gate_seal_backend_pause"])

    async def test_drain_failure_prevents_return_to_native_train(self):
        strategy = SimpleNamespace(produce_batch=AsyncMock())
        task = SimpleNamespace(
            task_name="task",
            produce_strategy=strategy,
            agent_loop=object(),
            sampler=object(),
            weight=1.0,
            order=0,
            stale_threshold=None,
            token_stale_threshold=None,
            expired_groups_retryable=True,
        )
        manager = AgentLoopManager([task], object(), object(), logger=MagicMock())
        manager._keepalive = SimpleNamespace(
            resume=AsyncMock(),
            check_requests=AsyncMock(),
            pause=AsyncMock(side_effect=TimeoutError("drain")),
            quiesce_timeout=1,
        )
        with (
            patch("xtuner.v1.rl.agent_loop_manager.agent_loop_manager.refresh_for_all_tasks", new=AsyncMock()),
            patch("xtuner.v1.rl.agent_loop_manager.agent_loop_manager.take_train_batch", new=AsyncMock()) as take,
        ):
            with self.assertRaises(TimeoutError):
                await manager.produce_batch(1, 1, model_step=0)
            take.assert_not_awaited()


class CompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_retained_ready_group_is_claimed_once_across_windows(self):
        from test_producer import TestProducer as Fixtures
        from test_producer import make_rollout_state

        from xtuner.v1.data_proto.rl_data import Status
        from xtuner.v1.rl.agent_loop_manager.producer import AsyncProduceStrategyConfig

        fixture = Fixtures()
        fixture.setUp()
        strategy = AsyncProduceStrategyConfig().build()
        strategy.retain_live_tasks = True
        owner = strategy._pending_tasks

        async def done_group():
            return [make_rollout_state(999)]

        async def spawn():
            return asyncio.create_task(done_group())

        await owner.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn)
        await asyncio.sleep(0)
        ctx = fixture._build_context(
            strategy, "retained", fixture._build_agent_loop(), fixture._build_sampler(), batch_size=1
        )
        await strategy.produce_batch(ctx)
        await strategy.produce_batch(ctx)
        self.assertIs(owner, strategy._pending_tasks)
        self.assertEqual(await fixture.replay_buffer.count("retained", Status.COMPLETED), 1)
        self.assertEqual(owner.count(), 0)

    async def test_resume_configuration_and_debug_combinations(self):
        from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig

        config = SimpleNamespace(
            agent_loop_manager_cfg=SimpleNamespace(turn_keepalive=True),
            enable_evaluate=False,
            enable_initial_evaluate=False,
            eval_agent_loop_manager_cfg=None,
            rollout_config=SimpleNamespace(turn_resume_wait_timeout=720, health_check_failure_threshold=0),
            load_checkpoint_cfg=SimpleNamespace(
                checkpoint_path=None, load_optimizer_states=True, load_optimizer_args=True
            ),
            auto_resume=False,
            debug_train=False,
            debug_rollout=False,
        )
        self.assertIs(RLColocateTrainerConfig._validate_turn_keepalive(config), config)
        self.assertTrue(config.rollout_config.turn_resume_initially_closed)
        config.rollout_config.health_check_failure_threshold = 3
        with self.assertRaisesRegex(ValueError, "fixed topology"):
            RLColocateTrainerConfig._validate_turn_keepalive(config)
        config.rollout_config.health_check_failure_threshold = 0
        config.agent_loop_manager_cfg.turn_keepalive = False
        with self.assertRaisesRegex(ValueError, "both request gates"):
            RLColocateTrainerConfig._validate_turn_keepalive(config)
        config.rollout_config.turn_resume_wait_timeout = None
        self.assertIs(RLColocateTrainerConfig._validate_turn_keepalive(config), config)
        config.agent_loop_manager_cfg.turn_keepalive = True
        with self.assertRaisesRegex(ValueError, "both request gates"):
            RLColocateTrainerConfig._validate_turn_keepalive(config)
        config.rollout_config.turn_resume_wait_timeout = 720
        for field in ("enable_evaluate", "enable_initial_evaluate"):
            setattr(config, field, True)
            with self.assertRaises(ValueError):
                RLColocateTrainerConfig._validate_turn_keepalive(config)
            setattr(config, field, False)
        config.auto_resume = True
        RLColocateTrainerConfig._validate_turn_keepalive(config)
        config.debug_rollout = True
        with self.assertRaises(ValueError):
            RLColocateTrainerConfig._validate_turn_keepalive(config)
        config.debug_rollout = False
        config.debug_train = True
        with self.assertRaises(ValueError):
            RLColocateTrainerConfig._validate_turn_keepalive(config)
        config.debug_train = False
        config.auto_resume = False
        config.load_checkpoint_cfg.checkpoint_path = "/checkpoint"
        RLColocateTrainerConfig._validate_turn_keepalive(config)
        config.auto_resume = True
        with self.assertRaises(ValueError):
            RLColocateTrainerConfig._validate_turn_keepalive(config)
        config.auto_resume = False
        config.load_checkpoint_cfg.load_optimizer_states = False
        with self.assertRaises(ValueError):
            RLColocateTrainerConfig._validate_turn_keepalive(config)


class StartupTests(unittest.TestCase):
    def _run_startup(self, step, fail=None):
        from xtuner.v1.rl.agent_loop_manager.agent_turn_pause_policy import AgentTurnPausePolicy
        from xtuner.v1.rl.rollout.request_gate import RequestGate, SettlementReceipt
        from xtuner.v1.rl.utils import asyncio_run
        from xtuner.v1.train.rl_trainer import RLColocateTrainer

        owner = LoopThread()
        events = []
        state = {}

        async def construct():
            state["gate"] = RequestGate(wait_timeout=2, initially_open=False)
            events.append("endpoint_visible_closed")

            async def waiting():
                async with state["gate"].request("new-session") as lease:
                    events.append(("admit", lease.model_step))
                    lease.settlement = SettlementReceipt(True, True, "trace_committed")

            state["queued"] = asyncio.create_task(waiting())

        owner.submit(construct()).result()

        async def prepare_pause(**kwargs):
            self.assertFalse(state["gate"].snapshot()["open"])
            events.append("seal")
            await state["gate"].seal(1)
            await state["gate"].drain(1)
            return {"epoch": 1}

        async def pause():
            events.append("backend_pause")

        async def backend_ready():
            events.append("backend_ready")

        async def publish(barrier, model_step, *, timeout):
            self.assertEqual(timeout, 1)
            await state["gate"].install_version(barrier["epoch"], model_step)
            await backend_ready()
            events.append(("publish", model_step))
            await state["gate"].reopen(barrier["epoch"])

        ctl = SimpleNamespace(
            prepare_pause=SimpleNamespace(remote=prepare_pause),
            pause_generation=SimpleNamespace(remote=pause),
            continue_generation=SimpleNamespace(remote=backend_ready),
            resume_requests=SimpleNamespace(remote=publish),
        )

        async def restore(path):
            self.assertIs(asyncio.get_running_loop(), owner.loop)
            self.assertFalse(state["gate"].snapshot()["open"])
            events.append("sampler_replay_restore")
            if fail == "replay":
                raise RuntimeError("replay")
            return step

        inner = SimpleNamespace(
            _rollout_controller=ctl,
            _keepalive=AgentTurnPausePolicy(freeze_timeout=1, drain_timeout=1),
            task_runners=[],
            resume=restore,
        )
        manager = ColocateKeepaliveManager(inner, owner)
        trainer = RLColocateTrainer.__new__(RLColocateTrainer)
        trainer.agent_loop_manager = manager
        trainer._rollout_config = SimpleNamespace(skip_load_weights=False)
        trainer._cur_step = step
        trainer.logger = MagicMock()
        trainer._resume_agent_loop_manager = lambda path: manager.resume(path)

        def sync():
            self.assertFalse(state["gate"].snapshot()["open"])
            events.append("weights_and_kv_sync")
            if fail == "sync":
                raise RuntimeError("sync")

        trainer._sync_weights_from_train_workers = sync
        try:
            if fail:
                with self.assertRaises(RuntimeError):
                    trainer._initialize_keepalive_rollout("/checkpoint")
                self.assertFalse(state["gate"].snapshot()["open"])
                self.assertFalse(owner.thread.is_alive())
                self.assertFalse(any(isinstance(x, tuple) and x[0] == "admit" for x in events))
            else:
                trainer._initialize_keepalive_rollout("/checkpoint" if step else None)

                async def finish():
                    await state["queued"]

                owner.submit(finish()).result(timeout=2)
                self.assertEqual(events[-2:], [("publish", step), ("admit", step)])
                if step:
                    self.assertLess(events.index("sampler_replay_restore"), events.index("weights_and_kv_sync"))
                    self.assertLess(events.index("weights_and_kv_sync"), events.index("backend_ready"))
        finally:
            asyncio_run(manager.close())
        return events

    def test_fresh_version_zero_and_restored_version_published_after_sync(self):
        self._run_startup(0)
        self._run_startup(6)

    def test_restore_or_sync_failure_keeps_gate_closed_and_closes_owner(self):
        self._run_startup(6, "replay")
        self._run_startup(6, "sync")

    def test_auto_resume_uses_native_selector_then_new_native_exp(self):
        import tempfile
        from pathlib import Path

        from xtuner.v1.train.rl_trainer import BaseRLTrainer, RLColocateTrainer

        for auto, previous, selected in [
            (True, "/ckpt6", "/ckpt6"),
            (True, None, None),
            (False, "/ckpt6", "/explicit"),
        ]:
            with self.subTest(auto=auto, previous=previous), tempfile.TemporaryDirectory() as temp:
                meta_path = Path(temp) / "meta.json"
                meta_path.write_text("{}")
                load = SimpleNamespace(checkpoint_path=None if auto else "/explicit")
                cfg = SimpleNamespace(auto_resume=auto, work_dir=temp, load_checkpoint_cfg=load)
                cfg.model_copy = lambda update: SimpleNamespace(**{**vars(cfg), **update})
                trainer = RLColocateTrainer.__new__(RLColocateTrainer)
                trainer._turn_keepalive = True
                old_exp = SimpleNamespace(latest_checkpoint=previous)
                old = SimpleNamespace(exps=[old_exp, SimpleNamespace(latest_checkpoint=None)])
                old.model_copy = lambda update: SimpleNamespace(exps=update["exps"], latest_exp=update["exps"][-1])
                with (
                    patch("xtuner.v1.train.rl_trainer.XTunerMeta.model_validate_json", return_value=old, create=True),
                    patch.object(BaseRLTrainer, "_init_work_dir_and_meta") as native_build,
                ):
                    trainer._init_work_dir_and_meta(cfg, "meta.json")
                self.assertEqual(load.checkpoint_path, Path(selected) if auto and selected else selected)
                self.assertEqual(native_build.call_args[0][0].auto_resume, False if auto else auto)


class SessionConstructionTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_gate_is_closed_before_http_server_start(self):
        from xtuner.v1.rl.rollout.session_server import SessionServer, SessionServerActor

        seen = []

        async def fake_start(server):
            seen.append(server.request_gate.snapshot()["open"])

        with (
            patch(
                "xtuner.v1.rl.rollout.session_server.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(eos_token=""),
            ),
            patch("xtuner.v1.rl.rollout.session_server.get_store", return_value=object()),
            patch.object(SessionServer, "start", new=fake_start),
        ):
            actor = SessionServerActor(
                "http://backend",
                "fake-tokenizer",
                "localhost",
                8765,
                10,
                turn_resume_wait_timeout=720,
                turn_resume_initially_closed=True,
            )
            await actor.start()
        self.assertEqual(seen, [False])


class CheckpointDataTests(unittest.IsolatedAsyncioTestCase):
    async def test_sampler_and_replay_routes_are_materialized_and_rehydrated(self):
        import json
        import tempfile
        from pathlib import Path

        import numpy as np
        from test_producer import make_rollout_state

        from xtuner.v1.rl.replay_buffer import SerializedRayObjectRef, SyncReplayBufferConfig

        class FakeRef:
            def __init__(self, value):
                self.value = value

        class Sampler:
            def __init__(self):
                self.next_id = 71

            def save(self, path):
                (Path(path) / "sampler.txt").write_text(str(self.next_id))

            def resume(self, path):
                self.next_id = int((Path(path) / "sampler.txt").read_text())

        def manager():
            sampler = Sampler()
            runner = SimpleNamespace(
                task_name="task",
                sampler=sampler,
                weight=1,
                produce_strategy=SimpleNamespace(pending_task_count=lambda: 0),
            )
            result = AgentLoopManager([runner], SyncReplayBufferConfig().build(), object(), logger=MagicMock())
            result._keepalive = object()
            return result

        with (
            tempfile.TemporaryDirectory() as temp,
            patch("xtuner.v1.rl.replay_buffer.ray.ObjectRef", FakeRef),
            patch("xtuner.v1.rl.replay_buffer.ray.get", side_effect=lambda ref: ref.value),
            patch("xtuner.v1.rl.replay_buffer.ray.put", side_effect=FakeRef),
        ):
            before = manager()
            item = make_rollout_state(19)
            routes = np.arange(12, dtype=np.int32).reshape(3, 4)
            item.routed_experts = FakeRef(routes)
            item.response_model_steps = [3, 4, 5]
            await before.replay_buffer.put([item], "task")
            await before.save(temp, model_step=6)
            import torch

            payload = torch.load(Path(temp) / before.replay_buffer._SAVE_PATH, weights_only=False)
            self.assertIsInstance(payload["storage_state"]["items"][0].item[0].routed_experts, SerializedRayObjectRef)
            after = manager()
            after.task_runners[0].sampler.next_id = 0
            self.assertEqual(await after.resume(temp), 6)
            self.assertEqual(after.task_runners[0].sampler.next_id, 71)
            from xtuner.v1.data_proto.rl_data import Status

            groups = await after.replay_buffer.get(1, "task", Status.COMPLETED)
            restored = groups[0][0]
            self.assertIsInstance(restored.routed_experts, FakeRef)
            self.assertIsNot(restored.routed_experts, item.routed_experts)
            np.testing.assert_array_equal(restored.routed_experts.value, routes)
            self.assertEqual(restored.response_model_steps, [3, 4, 5])
            state_path = Path(temp) / "agent_loop_manager_state.json"
            saved = json.loads(state_path.read_text())
            saved["replay_buffer_saved"] = False
            state_path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(RuntimeError, "persisted replay"):
                await manager().resume(temp)


class DriverRestoreTests(unittest.TestCase):
    def test_driver_restore_forwards_full_optimizer_config_and_step(self):
        import json
        import tempfile
        from pathlib import Path

        from xtuner.v1.train.rl_trainer import RLColocateTrainer

        with tempfile.TemporaryDirectory() as temp:
            trainer = RLColocateTrainer.__new__(RLColocateTrainer)
            trainer.logger = MagicMock()
            trainer.train_controller = SimpleNamespace(resume=MagicMock())
            config = SimpleNamespace(checkpoint_path=temp, load_optimizer_states=True, load_optimizer_args=True)
            trainer._load_checkpoint_cfg = config
            (Path(temp) / "train_state.json").write_text(json.dumps({"cur_step": 6}))
            trainer._resume_train_controller_and_state(temp)
            trainer.train_controller.resume.assert_called_once_with(config)
            self.assertEqual(trainer._cur_step, 6)
            (Path(temp) / "train_state.json").unlink()
            with self.assertRaises(FileNotFoundError):
                trainer._resume_train_controller_and_state(temp)

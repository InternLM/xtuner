"""RLColocateTrainer 的 public 行为测试。

Good Tests:
- 通过 fit()、debug rollout/train 工作流和 RLThroughputBenchmark 的公开 schema 验证行为。
- 用轻量 fake AgentLoopManager / TrainController 替代重型 Ray worker。
- 只断言 step、model_step、文件内容等可观察结果。
- 对 debug 文件只验证磁盘内容和 fit() 读回的训练输入，不锁定内部 helper 的调用顺序。

Bad Tests:
- 不直接测试 _sync_weights_and_save、_prepare_train_data、_log_step 等私有 helper。
- 不把私有 mock 的调用次数或调用顺序当作核心契约。
- 不重复测试 AgentLoopManager、ReplayBuffer 或 ProduceStrategy 的状态机。

本文件主要覆盖的 public 行为:
- colocate fit 可以消费 async AgentLoopManager 并完成训练 step。
- fit 对空 rollout batch fail fast，并按 sync interval 传递 rollout model_step。
- debug_rollout 通过 fit() 将 batch 落盘；debug_train 通过 fit() 从落盘 batch 训练。
- debug 文件能保存/恢复 Ray ObjectRef；throughput scalar key schema 保持稳定。
"""

import asyncio
import threading
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
from xtuner.v1.rl.rollout.controller import RolloutController, WorkerInfo
from xtuner.v1.train.rl_trainer import RLColocateTrainer, RLThroughputBenchmark


class _FakeRolloutState:
    def __init__(self, uid: int):
        self.id = uid
        self.uid = str(uid)
        self.message_uid = uid
        self.session_uid = uid
        self.status = Status.INIT
        self.finish_reason = None
        self.seq_staleness = 0
        self.prompt_ids = [10, 11]
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
        trainer._debug_rollout_dir = None
        trainer._debug_train = False
        trainer._enable_evaluate = False
        trainer._enable_initial_evaluate = False
        trainer._evaluate_step = 1
        trainer._cpu_resource_manager = None
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
        trainer._num_workers = 1.0
        trainer._rollout_num_workers = 1.0
        trainer._benchmark_start_time_s = 100.0
        trainer._benchmark_training_samples = 0
        trainer._benchmark_training_tokens = 0
        trainer._rollout_config = SimpleNamespace(rollout_timeout=37.0)
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
            ensure_workers_healthy_before_training=SimpleNamespace(
                remote=MagicMock(return_value="rollout_ready_for_training")
            ),
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
        # 验证 colocate fit 可以通过公开入口消费 async manager 产出的 batch。
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
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
        ):
            trainer.fit()

        trainer.rollout_controller.ensure_workers_healthy_before_training.remote.assert_called_once_with()
        trainer.rollout_controller.offload.remote.assert_called_once_with()
        trainer.train_controller.onload.assert_called_once_with(target="all")
        trainer.train_controller.fit.assert_called_once()
        self.assertEqual(trainer._cur_step, 1)

    def test_fit_requires_non_empty_batch_from_manager(self):
        # 验证空 rollout batch 会 fail fast，并且不会推进训练 step。
        async def _produce_empty(batch_size, train_step, **kwargs):
            return ProduceBatchResult(rollout_states=[])

        empty_manager = SimpleNamespace(produce_batch=_produce_empty)
        trainer = self._make_trainer(empty_manager)

        with (
            patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
        ):
            with self.assertRaisesRegex(AssertionError, "return non-empty rollout_states"):
                trainer.fit()

        trainer.rollout_controller.offload.remote.assert_not_called()
        trainer.train_controller.onload.assert_not_called()
        trainer.train_controller.fit.assert_not_called()
        self.assertEqual(trainer._cur_step, 0)

    def test_fit_does_not_onload_train_when_rollout_training_barrier_fails(self):
        # 验证共卡训练进入训练前必须先通过 rollout phase-switch barrier；
        # 失败时不能 onload 训练。
        async def _produce_batch(batch_size, train_step, *, model_step):
            return ProduceBatchResult(
                rollout_states=[[SimpleNamespace(message_uid=train_step, uid=train_step)]]
            )

        trainer = self._make_trainer(SimpleNamespace(produce_batch=_produce_batch))
        trainer.rollout_controller.ensure_workers_healthy_before_training.remote.side_effect = RuntimeError(
            "inactive rollout workers before training"
        )

        with (
            patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
        ):
            with self.assertRaisesRegex(RuntimeError, "inactive rollout workers"):
                trainer.fit()

        trainer.rollout_controller.ensure_workers_healthy_before_training.remote.assert_called_once_with()
        trainer.rollout_controller.offload.remote.assert_not_called()
        trainer.train_controller.onload.assert_not_called()
        trainer.train_controller.fit.assert_not_called()
        self.assertEqual(trainer._cur_step, 0)

    def test_fit_uses_real_rollout_controller_barrier_to_recover_before_next_rollout(self):
        # Trainer 只负责调用真实 rollout controller barrier；
        # recovery 细节由 RolloutController.ensure_workers_healthy_before_training 覆盖。
        events = []
        worker_url = "http://worker-0"
        worker_state = {"healthy": True}
        workers_info = {}

        class _RemoteAction:
            def __init__(self, action):
                self.remote = MagicMock(side_effect=action)

        def check_health():
            is_healthy = worker_state["healthy"]
            events.append(("check_health", workers_info[0].is_active, is_healthy))
            return is_healthy

        def shutdown():
            events.append(("shutdown", workers_info[0].is_active))
            self.assertFalse(workers_info[0].is_active)

        def init(*args, **kwargs):
            events.append(("init", args, kwargs, worker_url))
            self.assertFalse(workers_info[0].is_active)
            worker_state["healthy"] = True
            return 0, worker_url

        def init_dist_port():
            events.append(("init_dist_port",))
            return "127.0.0.1:12345"

        def assert_worker_available(event_name):
            self.assertEqual(workers_info[0].url, worker_url)
            self.assertTrue(worker_state["healthy"])
            self.assertTrue(workers_info[0].is_active)
            events.append((event_name, worker_state["healthy"], workers_info[0].is_active, worker_url))

        worker = SimpleNamespace(
            check_health=_RemoteAction(check_health),
            shutdown=_RemoteAction(shutdown),
            init=_RemoteAction(init),
            init_dist_port=_RemoteAction(init_dist_port),
            offload=_RemoteAction(lambda: assert_worker_available("rollout_offload")),
            onload_weights=_RemoteAction(lambda: assert_worker_available("rollout_onload_weights")),
            onload_kvcache=_RemoteAction(lambda: assert_worker_available("rollout_onload_kvcache")),
        )
        workers_info[0] = WorkerInfo(actor=worker, url=worker_url, is_active=True)

        controller = RolloutController.__new__(RolloutController)
        controller.rank2info = workers_info
        controller.worker_info_lock = threading.RLock()
        health_checker_state = {"paused": False}

        def is_health_checker_paused():
            events.append(("health_is_paused", health_checker_state["paused"]))
            return health_checker_state["paused"]

        def pause_health_checker():
            events.append(("health_pause",))
            health_checker_state["paused"] = True

        def resume_health_checker():
            events.append(("health_resume",))
            health_checker_state["paused"] = False

        controller.health_checker = SimpleNamespace(
            is_paused=is_health_checker_paused,
            pause=pause_health_checker,
            resume=resume_health_checker,
            stop=lambda: events.append(("health_stop",)),
        )
        controller.logger = MagicMock()

        class _LocalRolloutControllerProxy:
            def __init__(self, real_controller):
                self.ensure_workers_healthy_before_training = _RemoteAction(
                    real_controller.ensure_workers_healthy_before_training
                )
                self.offload = _RemoteAction(real_controller.offload)
                self.onload_weights = _RemoteAction(real_controller.onload_weights)
                self.onload_kvcache = _RemoteAction(real_controller.onload_kvcache)

        rollout_controller = _LocalRolloutControllerProxy(controller)
        produce_calls = []

        async def _produce_batch(batch_size, train_step, *, model_step):
            produce_calls.append((batch_size, train_step, model_step))
            events.append(("rollout", train_step, worker_state["healthy"], workers_info[0].is_active, worker_url))
            if train_step == 1:
                worker_state["healthy"] = False
                events.append(("worker_failed", train_step, worker_url))
            else:
                self.assertTrue(worker_state["healthy"])
                self.assertTrue(workers_info[0].is_active)
                self.assertEqual(workers_info[0].url, worker_url)
            return ProduceBatchResult(
                rollout_states=[[SimpleNamespace(message_uid=train_step, uid=train_step)]]
            )

        trainer = self._make_trainer(
            SimpleNamespace(produce_batch=_produce_batch),
            total_train_steps=2,
            sync_weights_interval=10,
        )
        trainer.rollout_controller = rollout_controller
        trainer.train_controller.onload = MagicMock(side_effect=lambda target: events.append(("train_onload", target)))
        trainer.train_controller.fit = MagicMock(
            side_effect=lambda *args, rollout_idx, **kwargs: events.append(("train_fit", rollout_idx)) or []
        )

        def sync_rollout_after_train(train_step, step_timer_dict):
            events.append(("sync_after_train", train_step))
            rollout_controller.onload_weights.remote()
            rollout_controller.onload_kvcache.remote()
            return False

        trainer._sync_weights_and_save = MagicMock(side_effect=sync_rollout_after_train)

        with (
            patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
        ):
            trainer.fit()

        self.assertEqual(produce_calls, [(1, 1, 0), (1, 2, 0)])
        self.assertEqual(workers_info[0].url, worker_url)
        self.assertTrue(worker_state["healthy"])
        self.assertTrue(workers_info[0].is_active)
        self.assertEqual([event[0] for event in events].count("init"), 1)
        self.assertNotIn(("init_dist_port",), events)
        self.assertEqual(trainer.train_controller.fit.call_count, 2)
        self.assertEqual(trainer._cur_step, 2)
        self.assertLess(
            events.index(("worker_failed", 1, worker_url)),
            events.index(("check_health", True, False)),
        )
        self.assertLess(events.index(("health_is_paused", False)), events.index(("health_pause",)))
        self.assertLess(events.index(("health_pause",)), events.index(("check_health", True, False)))
        self.assertLess(events.index(("check_health", True, False)), events.index(("shutdown", False)))
        restart_event = ("init", (), {}, worker_url)
        self.assertLess(events.index(("shutdown", False)), events.index(restart_event))
        self.assertLess(
            events.index(restart_event),
            events.index(("check_health", False, True)),
        )
        self.assertLess(
            events.index(("check_health", False, True)),
            events.index(("health_resume",)),
        )
        self.assertLess(
            events.index(("health_resume",)),
            events.index(("rollout_offload", True, True, worker_url)),
        )
        self.assertLess(
            events.index(("rollout_offload", True, True, worker_url)),
            events.index(("train_onload", "all")),
        )
        self.assertLess(events.index(("train_onload", "all")), events.index(("train_fit", 1)))
        self.assertLess(events.index(("train_fit", 1)), events.index(("sync_after_train", 1)))
        self.assertLess(
            events.index(("rollout_onload_weights", True, True, worker_url)),
            events.index(("rollout", 2, True, True, worker_url)),
        )
        self.assertLess(
            events.index(("rollout_onload_kvcache", True, True, worker_url)),
            events.index(("rollout", 2, True, True, worker_url)),
        )
        self.assertIn(("rollout", 2, True, True, worker_url), events)

    def test_fit_uses_sync_interval_and_passes_rollout_model_step(self):
        # 验证 rollout 看到的是按 sync interval 推进后的 model_step。
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
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
        ):
            trainer.fit()

        self.assertEqual(produce_calls, [(1, 1, 0), (1, 2, 0), (1, 3, 2)])
        self.assertEqual(trainer._cur_step, 3)

    def test_debug_rollout_saves_raw_batch_and_skips_training(self):
        # 验证 debug_rollout 通过 fit() 将原始 rollout batch 落盘且不启动训练。
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
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
        ):
            trainer.fit()

        saved_path = trainer._debug_rollout_dir / "debug_rollout_1.pt"
        self.assertTrue(saved_path.exists())
        saved_batch = torch.load(saved_path, map_location="cpu", weights_only=False)
        self.assertEqual(saved_batch[0][0].response, "ok")
        trainer.train_controller.fit.assert_not_called()

    def test_throughput_benchmark_exports_stable_scalar_schema(self):
        # throughput key 是日志对外契约；这里固定 schema，避免在流程测试中硬编码。
        scalars = RLThroughputBenchmark(
            e2e_effective_sgs=1.0,
            e2e_effective_tgs=2.0,
            effective_sgs=3.0,
            effective_tgs=4.0,
            training_tgs=5.0,
            rollout_sgs=6.0,
            rollout_tgs=7.0,
        ).to_scalars()

        self.assertEqual(
            scalars,
            {
                "throughput/e2e_effective_sgs": 1.0,
                "throughput/e2e_effective_tgs": 2.0,
                "throughput/effective_sgs": 3.0,
                "throughput/effective_tgs": 4.0,
                "throughput/training_tgs": 5.0,
                "throughput/rollout_sgs": 6.0,
                "throughput/rollout_tgs": 7.0,
            },
        )

    def test_debug_train_loads_batches_and_skips_weight_sync(self):
        # 验证 debug_train 通过 fit() 读取落盘 batch，并只推进训练流程。
        debug_dir = Path(self.temp_dir.name) / "debug_train"
        debug_dir.mkdir()
        torch.save([[SimpleNamespace(uid=1, message_uid=1)]], debug_dir / "debug_rollout_1.pt")
        torch.save([[SimpleNamespace(uid=2, message_uid=2)]], debug_dir / "debug_rollout_2.pt")

        trainer = self._make_trainer(MagicMock(), total_train_steps=2)
        trainer._debug_train = True
        trainer._debug_train_files = {
            1: debug_dir / "debug_rollout_1.pt",
            2: debug_dir / "debug_rollout_2.pt",
        }
        captured_batches = []

        def train_one_batch(train_batch, train_step, step_timer_dict, **kwargs):
            captured_batches.append((train_step, train_batch))
            return {
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

        trainer._train_one_batch = MagicMock(side_effect=train_one_batch)

        trainer.fit()

        self.assertEqual([(step, batch[0][0].uid) for step, batch in captured_batches], [(1, 1), (2, 2)])
        self.assertEqual(trainer._cur_step, 2)

    def test_debug_rollout_fit_serializes_object_refs_and_debug_train_fit_restores_them(self):
        # 验证 debug 文件作为公开调试产物保存值快照，训练回放时恢复为 Ray ObjectRef。
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

            async def _produce_batch(batch_size, train_step, *, model_step):
                return ProduceBatchResult(rollout_states=[[rollout_state]])

            trainer = self._make_trainer(SimpleNamespace(produce_batch=_produce_batch))
            trainer._debug_rollout = True
            trainer._debug_rollout_dir = Path(self.temp_dir.name) / "debug_refs"

            with patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run):
                trainer.fit()

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

            replay_trainer = self._make_trainer(MagicMock())
            replay_trainer._debug_train = True
            replay_trainer._debug_train_files = {1: trainer._debug_rollout_dir / "debug_rollout_1.pt"}
            captured_batches = []

            def train_one_batch(train_batch, train_step, step_timer_dict, **kwargs):
                captured_batches.append(train_batch)
                return {
                    "data_info": {"batch_size": 1},
                    "workers_log_item": [
                        {
                            "train_metrics": [],
                            "sft_train_metrics": {},
                            "train_entropy": 0.0,
                        }
                    ],
                }

            replay_trainer._train_one_batch = MagicMock(side_effect=train_one_batch)
            replay_trainer.fit()

            loaded_batch = captured_batches[0]
            self.assertIsInstance(loaded_batch[0][0].mm_info["pixel_values"], ray.ObjectRef)
            self.assertIsInstance(loaded_batch[0][0].routed_experts, ray.ObjectRef)
            self.assertTrue(torch.equal(ray.get(loaded_batch[0][0].mm_info["pixel_values"]), pixel_values))
            self.assertEqual(ray.get(loaded_batch[0][0].routed_experts), routed_experts)
        finally:
            ray.shutdown()


if __name__ == "__main__":
    unittest.main()

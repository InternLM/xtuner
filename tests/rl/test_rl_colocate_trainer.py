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
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import ray
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop_manager import AsyncProduceStrategyConfig, ProduceBatchResult
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import AgentLoopManager
from xtuner.v1.rl.agent_loop_manager.produce_utils import _TaskRunner
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SerializedRayObjectRef
from xtuner.v1.train.rl_trainer import RLColocateTrainer, RLThroughputBenchmark


class _FakeRolloutState:
    def __init__(self, uid: int):
        self.id = uid
        self.rollout_id = str(uid)
        self.group_id = uid
        self.session_id = uid
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


def _build_fake_rollout_controller():
    rollout_ctl = MagicMock()
    rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
    rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
    rollout_ctl.get_weight_update_targets.remote = AsyncMock(return_value=())
    return rollout_ctl


def _ray_get_none_ref():
    return ray.put(None) if ray.is_initialized() else None


def _build_fake_agent_loop():
    rollout_ctl = _build_fake_rollout_controller()
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
        trainer._save_trajectories = MagicMock()
        trainer._release_trace_store = MagicMock()
        trainer._sync_weights_and_save = MagicMock(
            side_effect=lambda train_step, step_timer_dict: train_step % trainer._sync_weights_interval == 0
        )
        trainer._log_step = MagicMock()
        trainer._prepare_train_data = MagicMock(
            return_value=([{"seq_ctx": "fake"}], {"batch_size": 1, "rewards/mean": 1.0})
        )

        trainer.rollout_controller = SimpleNamespace(
            check_and_shutdown_inactive_workers=SimpleNamespace(
                remote=MagicMock(return_value="rollout_inactive_workers_shutdown")
            ),
            offload=SimpleNamespace(remote=MagicMock(return_value="rollout_offloaded")),
            restart_inactive_workers=SimpleNamespace(remote=MagicMock(return_value="rollout_restarted")),
            onload_weights=SimpleNamespace(remote=MagicMock(return_value="weights_loaded")),
            onload_kvcache=SimpleNamespace(remote=MagicMock(return_value="kvcache_loaded")),
            validate_registered_workers_to_proxy=SimpleNamespace(remote=MagicMock(side_effect=_ray_get_none_ref)),
        )
        trainer.train_controller = SimpleNamespace(
            onload=MagicMock(return_value="train_onloaded"),
            offload=MagicMock(return_value="train_offloaded"),
            update_weights=MagicMock(return_value="weights_updated"),
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
            rollout_controller=_build_fake_rollout_controller(),
        )
        trainer = self._make_trainer(manager)

        with (
            patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
        ):
            trainer.fit()

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
            return ProduceBatchResult(rollout_states=[[_FakeRolloutState(train_step)]])

        trainer = self._make_trainer(SimpleNamespace(produce_batch=_produce_batch))
        trainer.rollout_controller.check_and_shutdown_inactive_workers.remote.side_effect = RuntimeError(
            "inactive rollout workers after recovery"
        )

        with (
            patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
        ):
            with self.assertRaisesRegex(RuntimeError, "inactive rollout workers"):
                trainer.fit()

        trainer.rollout_controller.check_and_shutdown_inactive_workers.remote.assert_called_once_with()
        trainer.rollout_controller.offload.remote.assert_not_called()
        trainer.train_controller.onload.assert_not_called()
        trainer.train_controller.fit.assert_not_called()
        self.assertEqual(trainer._cur_step, 0)

    def test_fit_uses_sync_interval_and_passes_rollout_model_step(self):
        # 验证 rollout 看到的是按 sync interval 推进后的 model_step。
        produce_calls = []

        async def _produce_batch(batch_size, train_step, *, model_step):
            produce_calls.append((batch_size, train_step, model_step))
            return ProduceBatchResult(rollout_states=[[_FakeRolloutState(train_step)]])

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
        torch.save([[SimpleNamespace(rollout_id=1, group_id=1)]], debug_dir / "debug_rollout_1.pt")
        torch.save([[SimpleNamespace(rollout_id=2, group_id=2)]], debug_dir / "debug_rollout_2.pt")

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

        self.assertEqual([(step, batch[0][0].rollout_id) for step, batch in captured_batches], [(1, 1), (2, 2)])
        self.assertEqual(trainer._cur_step, 2)

    def test_debug_rollout_fit_serializes_object_refs_and_debug_train_fit_restores_them(self):
        # 验证 debug 文件作为公开调试产物保存值快照，训练回放时恢复为 Ray ObjectRef。
        if not ray.is_initialized():
            try:
                ray.init(address="local", num_cpus=1, ignore_reinit_error=True, include_dashboard=False)
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

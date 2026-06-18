"""RL trainer checkpoint save/resume 的 public 行为测试。

Good Tests:
- 通过 RLColocateTrainerConfig / RLDisaggregatedTrainerConfig 的 build() 和 trainer.fit() 验证行为。
- 使用真实 AgentLoopManager、Sampler、ReplayBuffer、XTunerMeta 和 checkpoint 文件。
- 只 mock placement group、train controller、rollout controller 这些耗时外部边界。

Bad Tests:
- 不直接调用 _maybe_save_checkpoint、_resume_from_checkpoint 等私有 helper。
- 不断言 save/resume 的内部调用顺序，只验证 checkpoint 文件和 resume 后继续训练的结果。
- 不在这个文件重复测试 ProduceStrategy、ReplayBuffer 的内部状态机。

本文件主要覆盖的 public 行为:
- RL trainer 能在 fit() 中保存 train state、AgentLoopManager state 和训练 worker checkpoint。
- auto_resume=True 能从最新 checkpoint 恢复原 exp_dir 和 cur_step。
- resume 后再次 fit() 只训练剩余 step，并继续写新的 checkpoint。
"""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams, Status
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.model.dense.qwen3 import Qwen3Dense4BConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    DisaggAgentLoopManagerConfig,
    DisaggAsyncProduceStrategyConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
)
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig, RLDisaggregatedTrainerConfig

QWEN3_4B_PATH = os.environ.get("QWEN3_4B_PATH")
CHECKPOINT_DIR = "checkpoints"
TRAIN_STATE_PATH = "train_state.json"
MANAGER_STATE_PATH = "agent_loop_manager_state.json"


class _RemoteMethod:
    def __init__(self, func=None, *, async_result: bool = False, return_value=None):
        self.func = func
        self.async_result = async_result
        self.return_value = return_value
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if not self.async_result:
            if self.func is None:
                return self.return_value
            return self.func(*args, **kwargs)

        async def _run():
            if self.func is None:
                return self.return_value
            return self.func(*args, **kwargs)

        return _run()


class _FakeCPUResourceManager:
    def __init__(self, accelerator_placement_groups=None):
        self.accelerator_placement_groups = accelerator_placement_groups

    def log_initial_snapshot(self):
        return None

    def log_registered_summary(self):
        return None


class _FakeRolloutController:
    def __init__(self):
        self.generate = _RemoteMethod(self._generate, async_result=True)
        self.pause_generation = _RemoteMethod(async_result=True)
        self.continue_generation = _RemoteMethod(async_result=True)
        self.offload = _RemoteMethod(return_value="rollout_offloaded")
        self.check_and_shutdown_inactive_workers = _RemoteMethod(return_value="rollout_inactive_workers_shutdown")
        self.restart_inactive_workers = _RemoteMethod(return_value="rollout_restarted")
        self.onload_weights = _RemoteMethod(return_value="weights_loaded")
        self.onload_kvcache = _RemoteMethod(return_value="kvcache_loaded")
        self.get_rollout_metadata = _RemoteMethod(return_value={"server_url_dict": {}})
        self.set_enable_partial_rollout = _RemoteMethod(return_value=None)

    def _generate(self, rollout_state):
        # 生成侧只补齐训练真正需要的可观察 rollout 结果，不加载真实推理服务。
        rollout_state.status = Status.COMPLETED
        rollout_state.response = "ok"
        rollout_state.response_ids = [100, 101]
        reward_score = 1.0 if int(rollout_state.rollout_id) % 2 == 0 else 0.5
        rollout_state.reward = {"score": reward_score}
        return rollout_state


class _FakeTrainController:
    def __init__(self):
        self.fit_steps: list[int] = []
        self.saved_checkpoints: list[Path] = []
        self.resume_checkpoint_paths: list[Path] = []
        self.train_rollout_mode = None
        self.update_weights_count = 0
        self.rollout_info = None

    def set_train_rollout_mode(self, mode: str):
        self.train_rollout_mode = mode

    def update_rollout_info(self, info):
        self.rollout_info = info

    def onload(self, target="all"):
        return f"onload:{target}"

    def offload(self, target="all"):
        return f"offload:{target}"

    def update_weights(self):
        self.update_weights_count += 1
        return "updated"

    def fit(self, data_batches, pack_max_length: int, rollout_idx: int):
        self.fit_steps.append(rollout_idx)
        return [
            {
                "rollout_is_metrics": {},
                "mismatch_metrics": {},
                "rollout_entropy": 0.0,
                "train_entropy": 0.0,
                "train_metrics": [],
                "sft_train_metrics": {},
            }
        ]

    def save(self, checkpoint_path: str, no_save_optimizer: bool):
        path = Path(checkpoint_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "fake_train_controller_checkpoint.txt").write_text(
            f"no_save_optimizer={no_save_optimizer}",
            encoding="utf-8",
        )
        self.saved_checkpoints.append(path)

    def resume(self, load_checkpoint_cfg):
        self.resume_checkpoint_paths.append(Path(load_checkpoint_cfg.checkpoint_path))

    def save_hf(self, hf_path: str):
        Path(hf_path).mkdir(parents=True, exist_ok=True)


class TestRLTrainerCheckpoint(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name) / "work_dir"
        self.dataset_path = Path(self.temp_dir.name) / "rollout_data.jsonl"
        self._write_rollout_dataset(self.dataset_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_rollout_dataset(self, dataset_path: Path):
        rows = []
        for idx in range(8):
            rows.append(
                {
                    "data_source": "unit",
                    "prompt": [{"role": "user", "content": f"question {idx}?"}],
                    "reward_model": {"style": "rule", "ground_truth": "ok"},
                    "extra_info": {"index": idx},
                }
            )
        dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    @contextmanager
    def _patched_runtime(self):
        runtime = SimpleNamespace(train_controllers=[], rollout_controllers=[])

        def build_pg(resources_config, name="train"):
            idx = len(getattr(build_pg, "built", []))
            build_pg.built = getattr(build_pg, "built", []) + [name]
            return SimpleNamespace(id=f"pg-{idx}", bundle_specs=[])

        def build_train_controller(worker_cfg, placement_group):
            controller = _FakeTrainController()
            runtime.train_controllers.append(controller)
            return controller

        def build_rollout_controller(rollout_cfg, placement_group):
            controller = _FakeRolloutController()
            runtime.rollout_controllers.append(controller)
            return controller

        with (
            patch("xtuner.v1.rl.utils.ray_accelerator_worker.ray.is_initialized", return_value=True),
            patch(
                "xtuner.v1.rl.utils.ray_accelerator_worker.ray.available_resources",
                return_value={"CPU": 64, "memory": 128 * 1024**3, "GPU": 8},
            ),
            patch("xtuner.v1.train.rl_trainer.AutoAcceleratorWorkers.build_placement_group", side_effect=build_pg),
            patch("xtuner.v1.train.rl_trainer.CPUResourceManager", _FakeCPUResourceManager),
            patch("xtuner.v1.train.rl_trainer.set_cpu_resource_manager", lambda manager: None),
            patch("xtuner.v1.train.rl_trainer.get_rollout_engine_version", return_value={}),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=lambda obj, timeout=None: obj),
            patch("xtuner.v1.train.rl_trainer.BaseRLTrainer._release_trace_store", return_value=None),
            patch.object(WorkerConfig, "build", autospec=True, side_effect=build_train_controller),
            patch.object(RolloutConfig, "build", autospec=True, side_effect=build_rollout_controller),
        ):
            yield runtime

    def _build_train_worker_config(self, model_path: str) -> WorkerConfig:
        return WorkerConfig(
            model_cfg=Qwen3Dense4BConfig(),
            optim_cfg=AdamWConfig(lr=1e-6, weight_decay=0.0),
            loss_cfg=GRPOLossConfig(
                policy_loss_cfg={
                    "loss_type": "vanilla",
                    "cliprange_low": 0.2,
                    "cliprange_high": 0.2,
                }
            ),
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0.0, lr_min=1e-6),
            fsdp_cfg=FSDPConfig(torch_compile=False, cpu_offload=False),
            load_from=model_path,
            optimizer_steps=1,
            pack_max_length=256,
        )

    def _build_agent_loop_manager_config(
        self,
        model_path: str,
        *,
        mode: str = "colocate",
        produce_strategy_config=None,
    ) -> AgentLoopManagerConfig | DisaggAgentLoopManagerConfig:
        dataloader_cfg = DataloaderConfig(
            dataset_config_list=[
                {
                    "dataset": DatasetConfig(
                        name="unit",
                        anno_path=self.dataset_path,
                        enable_sequential_sampler=True,
                        disable_filter=True,
                    ),
                    "tokenize_fn": RLTextTokenizeFnConfig(max_length=128),
                }
            ],
            collator="fake_collator",
            pack_level="none",
            pack_to_max_length=False,
            pack_max_length=256,
            num_workers=0,
            round_up=False,
        )
        manager_config_cls = DisaggAgentLoopManagerConfig if mode == "disaggregated" else AgentLoopManagerConfig
        produce_strategy_config = produce_strategy_config or (
            DisaggAsyncProduceStrategyConfig() if mode == "disaggregated" else SyncProduceStrategyConfig()
        )
        return manager_config_cls(
            tasks=[
                {
                    "task_name": "unit_task",
                    "agent_loop_config": SingleTurnAgentLoopConfig(
                        hf_checkpoint=model_path,
                        sample_params=SampleParams(max_tokens=2, temperature=0.0, top_k=1),
                    ),
                    "produce_strategy_config": produce_strategy_config,
                    "sampler_config": SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=2),
                }
            ],
        )

    def _build_rollout_config(self, model_path: str) -> RolloutConfig:
        return RolloutConfig(
            model_path=model_path,
            tokenizer_path=model_path,
            model_name="qwen3-4b-test",
            context_length=256,
            tensor_parallel_size=1,
            expert_parallel_size=1,
        )

    def _build_colocate_config(
        self,
        *,
        total_train_steps: int,
        auto_resume: bool,
    ) -> RLColocateTrainerConfig:
        assert QWEN3_4B_PATH is not None
        return RLColocateTrainerConfig(
            resources=AcceleratorResourcesConfig(
                accelerator="GPU",
                num_workers=1,
                num_cpus_per_worker=1,
                cpu_memory_per_worker=0,
            ),
            train_worker_cfg=self._build_train_worker_config(QWEN3_4B_PATH),
            rollout_config=self._build_rollout_config(QWEN3_4B_PATH),
            tokenizer_path=QWEN3_4B_PATH,
            replay_buffer_config=SyncReplayBufferConfig(),
            agent_loop_manager_cfg=self._build_agent_loop_manager_config(QWEN3_4B_PATH),
            load_from=QWEN3_4B_PATH,
            total_train_steps=total_train_steps,
            train_batch_size=1,
            sync_weights_interval=1,
            enable_evaluate=False,
            enable_initial_evaluate=False,
            work_dir=self.work_dir,
            auto_resume=auto_resume,
            checkpoint_interval=1,
            checkpoint_maxkeep=None,
            hf_interval=-1,
            seed=42,
            exp_tracker="jsonl",
        )

    def _build_disaggregated_config(
        self,
        *,
        total_train_steps: int,
        auto_resume: bool,
    ) -> RLDisaggregatedTrainerConfig:
        assert QWEN3_4B_PATH is not None
        resource_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=1,
            num_cpus_per_worker=1,
            cpu_memory_per_worker=0,
        )
        return RLDisaggregatedTrainerConfig(
            train_resources=resource_cfg,
            rollout_resources=resource_cfg,
            train_worker_cfg=self._build_train_worker_config(QWEN3_4B_PATH),
            rollout_config=self._build_rollout_config(QWEN3_4B_PATH),
            tokenizer_path=QWEN3_4B_PATH,
            replay_buffer_config=AsyncReplayBufferConfig(),
            agent_loop_manager_cfg=self._build_agent_loop_manager_config(
                QWEN3_4B_PATH,
                mode="disaggregated",
                produce_strategy_config=DisaggAsyncProduceStrategyConfig(over_sample_threshold=0.0),
            ),
            load_from=QWEN3_4B_PATH,
            total_train_steps=total_train_steps,
            train_batch_size=1,
            sync_weights_interval=1,
            enable_evaluate=False,
            enable_initial_evaluate=False,
            work_dir=self.work_dir,
            auto_resume=auto_resume,
            checkpoint_interval=1,
            checkpoint_maxkeep=None,
            hf_interval=-1,
            seed=42,
            exp_tracker="jsonl",
        )

    def _checkpoint_path(self, trainer, step: int) -> Path:
        return trainer.exp_dir / CHECKPOINT_DIR / f"ckpt-step-{step}"

    def _assert_checkpoint_saved_for_step(self, checkpoint_path: Path, step: int):
        self.assertEqual(checkpoint_path.name, f"ckpt-step-{step}")
        self.assertTrue((checkpoint_path / "fake_train_controller_checkpoint.txt").exists())
        with (checkpoint_path / TRAIN_STATE_PATH).open("r", encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"cur_step": step})
        with (checkpoint_path / MANAGER_STATE_PATH).open("r", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["model_step"], step)

    @unittest.skipUnless(QWEN3_4B_PATH, "QWEN3_4B_PATH is required for RL trainer checkpoint tests")
    def test_colocate_save_and_auto_resume_continue_from_latest_checkpoint(self):
        # 验证 colocate trainer 通过公开 build/fit 保存 checkpoint，并能 auto_resume 继续训练。
        with self._patched_runtime() as runtime:
            trainer = self._build_colocate_config(total_train_steps=2, auto_resume=False).build()
            trainer.fit()

            first_exp_dir = trainer.exp_dir
            checkpoint_path = self._checkpoint_path(trainer, step=2)
            self._assert_checkpoint_saved_for_step(checkpoint_path, step=2)
            self.assertEqual(runtime.train_controllers[0].fit_steps, [1, 2])

            resume_trainer = self._build_colocate_config(total_train_steps=3, auto_resume=True).build()
            self.assertEqual(resume_trainer.exp_dir, first_exp_dir)
            self.assertEqual(runtime.train_controllers[1].resume_checkpoint_paths, [checkpoint_path])

            resume_trainer.fit()

            self.assertEqual(runtime.train_controllers[1].fit_steps, [3])
            self._assert_checkpoint_saved_for_step(self._checkpoint_path(resume_trainer, step=3), step=3)

    @unittest.skipUnless(QWEN3_4B_PATH, "QWEN3_4B_PATH is required for RL trainer checkpoint tests")
    def test_disaggregated_save_and_auto_resume_continue_from_latest_checkpoint(self):
        # 验证 disaggregated trainer resume 后会恢复 producer，并继续完成剩余 step。
        with self._patched_runtime() as runtime:
            trainer = self._build_disaggregated_config(total_train_steps=2, auto_resume=False).build()
            trainer.fit()

            first_exp_dir = trainer.exp_dir
            checkpoint_path = self._checkpoint_path(trainer, step=2)
            self._assert_checkpoint_saved_for_step(checkpoint_path, step=2)
            self.assertEqual(runtime.train_controllers[0].fit_steps, [1, 2])

            resume_trainer = self._build_disaggregated_config(total_train_steps=3, auto_resume=True).build()
            self.assertEqual(resume_trainer.exp_dir, first_exp_dir)
            self.assertEqual(runtime.train_controllers[1].resume_checkpoint_paths, [checkpoint_path])

            resume_trainer.fit()

            self.assertEqual(runtime.train_controllers[1].fit_steps, [3])
            self._assert_checkpoint_saved_for_step(self._checkpoint_path(resume_trainer, step=3), step=3)


if __name__ == "__main__":
    unittest.main()

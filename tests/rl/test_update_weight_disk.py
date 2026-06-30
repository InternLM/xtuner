import asyncio
import os
import tempfile
import unittest
from pathlib import Path

import ray

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLDense4BConfig
from xtuner.v1.rl.loss import GRPOLossConfig as LossConfig
from xtuner.v1.rl.rollout.constants import ROLLOUT_RAY_GENERATE_MAX_CONCURRENCY
from xtuner.v1.rl.rollout.controller import RolloutController
from xtuner.v1.rl.rollout.sglang import SGLangWorker
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import (
    TrainingController,
    TrainingWorker as BaseTrainingWorker,
    WorkerConfig,
)
from xtuner.v1.rl.utils import (
    AcceleratorResourcesConfig,
    AutoAcceleratorWorkers,
    CPUResourcesConfig,
    CPUResourceManager,
    clear_cpu_resource_manager,
    register_cpu_resources,
    set_cpu_resource_manager,
)


TEST_TEXT_MESSAGES = [{"role": "user", "content": "Hello!"}]
MODEL_PATH = os.environ.get("QWEN3_VL_DENSE_PATH")


class DiskUpdateTestSGLangWorker(SGLangWorker):
    """Test-only SGLang worker that keeps recovered workers onloaded."""

    def offload(self):
        # restart_inactive_workers finally calls worker.actor.offload, but disaggregated recovery does not need offload.
        # The production recovery path offloads restarted workers to match the
        # colocated rollout baseline. This disk-recovery test immediately loads
        # weights from disk, so keep the test server alive and probeable.
        return {"success": True}


class DiskUpdateTestRolloutController(RolloutController):
    """Rollout controller with test hooks for targeting one recovered engine."""

    def _build_remote_worker_cls(self, worker_base_cls):
        # Force this E2E to use the test worker above. The passed worker class is
        # the production backend class selected from RolloutConfig.
        del worker_base_cls
        return super()._build_remote_worker_cls(DiskUpdateTestSGLangWorker)

    async def generate_from_weight_update_endpoint(
        self,
        *,
        endpoint_rank: int,
        rollout_state: RolloutState,
    ) -> RolloutState:
        # Production routing is session-based and may choose any active engine.
        # This test must verify the exact engine that was killed and restarted.
        worker = self.registry.active_entrypoint_by_rank(endpoint_rank)
        if worker is None:
            raise RuntimeError(f"Rollout endpoint rank={endpoint_rank} is not active.")

        response_ref = worker.actor.generate.remote(rollout_state=rollout_state)  # type: ignore[attr-defined]
        return await asyncio.wait_for(response_ref, timeout=self.config.rollout_timeout * self.timeout_multiplier)

    def shutdown_weight_update_endpoint(self, endpoint_rank: int | None = None) -> tuple[int, ...]:

        # Tests need a deterministic failure injection point. Mark the lifecycle
        # group inactive before shutdown so restart_inactive_workers() can claim it.
        targets = [target for target in self.registry.weight_update_targets() if target.is_active]
        if not targets:
            raise RuntimeError("No active rollout weight-update endpoint can be shut down.")

        target = targets[0] if endpoint_rank is None else next(
            (target for target in targets if target.endpoint_rank == endpoint_rank),
            None,
        )
        if target is None:
            raise RuntimeError(f"No active rollout weight-update endpoint rank={endpoint_rank} can be shut down.")

        with self.health_manager._paused_lifecycle_operation():
            groups = self.registry.mark_unhealthy_ranks({target.endpoint_rank})
            shutdown_ranks: list[int] = []
            for group in groups:
                if not self.health_manager._shutdown_worker_group(group):
                    raise RuntimeError(f"Failed to shut down rollout worker group ranks={group.ranks}.")
                shutdown_ranks.extend(group.ranks)
        return tuple(shutdown_ranks)

class TestUpdateWeightDisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if MODEL_PATH is None:
            raise unittest.SkipTest("QWEN3_VL_DENSE_PATH is not set")
        os.environ["XTUNER_USE_FA3"] = "1"
        # TODO(shipengcheng): SGLang disaggregated weight update cannot use
        # NCCL_CUMEM for now. Remove this after the root cause is fixed.
        os.environ["NCCL_CUMEM_ENABLE"] = "0"

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("XTUNER_USE_FA3", None)

    def setUp(self):
        self._original_pytorch_cuda_alloc_conf = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        ray.init(num_cpus=128, ignore_reinit_error=True)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.init_config()
        self.train_pg = AutoAcceleratorWorkers.build_placement_group(
            self.train_resources_cfg,
            name=f"test_update_weight_disk_train_{id(self)}",
        )
        self.rollout_pg = AutoAcceleratorWorkers.build_placement_group(
            self.rollout_resources_cfg,
            name=f"test_update_weight_disk_rollout_{id(self)}",
        )
        set_cpu_resource_manager(CPUResourceManager(accelerator_placement_groups=[self.train_pg, self.rollout_pg]))

    def tearDown(self):
        clear_cpu_resource_manager()
        ray.shutdown()
        self.temp_dir.cleanup()
        if self._original_pytorch_cuda_alloc_conf is not None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self._original_pytorch_cuda_alloc_conf
        else:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    def init_config(self):
        train_num_workers = int(os.environ.get("TRAIN_NUM_WORKERS", "4"))
        rollout_tp_size = int(os.environ.get("ROLLOUT_TP_SIZE", "2"))
        # Use at least two rollout engines by default so the test exercises
        # recovery of one failed engine while another engine remains active.
        rollout_num_workers = int(os.environ.get("ROLLOUT_NUM_WORKERS", str(rollout_tp_size * 2)))
        if rollout_num_workers < rollout_tp_size * 2:
            raise unittest.SkipTest("Disk recovery E2E requires at least two rollout engines.")
        if rollout_num_workers % rollout_tp_size != 0:
            raise unittest.SkipTest("ROLLOUT_NUM_WORKERS must be divisible by ROLLOUT_TP_SIZE.")

        self.train_resources_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=train_num_workers,
            num_cpus_per_worker=12,
            cpu_memory_per_worker=16 * 1024**3,
        )
        self.rollout_resources_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=rollout_num_workers,
            num_cpus_per_worker=12,
            cpu_memory_per_worker=16 * 1024**3,
        )
        self.rollout_cfg = RolloutConfig(
            env="test_rollout_disk",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            rollout_cross_node_comm=False,
            tensor_parallel_size=rollout_tp_size,
            expert_parallel_size=1,
            gpus_per_node=int(os.environ.get("GPUS_PER_NODE", "8")),
            dtype="bfloat16",
            skip_load_weights=False,
            context_length=256,
            worker_log_dir=self.worker_log_dir,
            gpu_memory_utilization=float(os.environ.get("ROLLOUT_GPU_MEMORY_UTILIZATION", "0.5")),
        )

        self.worker_cfg = WorkerConfig(
            model_cfg=Qwen3VLDense4BConfig(),
            optim_cfg=AdamWConfig(lr=5e-7, foreach=False),
            loss_cfg=LossConfig(
                policy_loss_cfg=dict(
                    cliprange_high=0.28,
                    cliprange_low=0.2,
                    loss_type="vanilla",
                ),
                ignore_idx=-100,
                use_kl_loss=False,
                kl_loss_coef=0.001,
                kl_loss_type="low_var_kl",
                mode="eager",
            ),
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=5e-7),
            fsdp_cfg=FSDPConfig(ep_size=1),
            load_from=MODEL_PATH,
            sp_size=1,
            pack_max_length=1024,
        )

    def _build_training_controller(self) -> TrainingController:
        TrainingWorker = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                }
            },
        )(BaseTrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(
            TrainingWorker,
            self.worker_cfg,
            self.train_pg,
        )
        ray.get([worker.test_all_reduce.remote() for worker in train_workers])
        return TrainingController(workers=train_workers)

    def _build_rollout_controller(self):
        num_workers = 1
        register_cpu_resources(
            name="rollout_controller",
            cpu_resources=CPUResourcesConfig(num_workers=num_workers),
        )
        test_dir = Path(__file__).resolve().parent
        pythonpath = os.pathsep.join(
            path for path in (str(test_dir), os.environ.get("PYTHONPATH", "")) if path
        )
        return (
            ray.remote(
                concurrency_groups={
                    "generate": ROLLOUT_RAY_GENERATE_MAX_CONCURRENCY,
                },
                runtime_env={"env_vars": {"PYTHONPATH": pythonpath}},
            )(DiskUpdateTestRolloutController)
            .options(num_cpus=num_workers)
            .remote(self.rollout_cfg, self.rollout_pg)
        )

    def _update_weights(
        self,
        train_controller,
        rollout_controller,
        weight_transport_type,
        target_endpoint_ranks: set[int] | None = None,
        **kwargs,
    ) -> None:
        targets = ray.get(rollout_controller.get_weight_update_targets.remote())
        if target_endpoint_ranks is not None:
            targets = tuple(target for target in targets if target.endpoint_rank in target_endpoint_ranks)
            self.assertGreater(
                len(targets),
                0,
                f"No rollout weight-update targets matched endpoint ranks={sorted(target_endpoint_ranks)}.",
            )
        train_controller.bind_rollout_weight_update(
            targets=targets,
            rollout_config=self.rollout_cfg,
            weight_transport_type=weight_transport_type,
            **kwargs,
        )
        train_controller.update_weights()

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "sglang backend is not enabled")
    def test_sglang_disaggregated_disk_update_after_engine_recovery(self):
        train_controller = self._build_training_controller()
        rollout_controller = self._build_rollout_controller()

        try:
            sample_params = SampleParams(temperature=0.0, max_tokens=128, top_k=1)
            input_state = RolloutState(message=TEST_TEXT_MESSAGES, sample_params=sample_params)

            self._update_weights(train_controller, rollout_controller, "nccl")

            initial_targets = ray.get(rollout_controller.get_weight_update_targets.remote())
            self.assertGreaterEqual(
                len(initial_targets),
                2,
                "This test requires multiple rollout engines. Set ROLLOUT_NUM_WORKERS >= 2 * ROLLOUT_TP_SIZE.",
            )
            failed_endpoint_rank = initial_targets[0].endpoint_rank

            # Warm up every rollout engine before killing one of them.
            endpoint_results = {
                target.endpoint_rank: ray.get(
                    rollout_controller.generate_from_weight_update_endpoint.remote(
                        endpoint_rank=target.endpoint_rank,
                        rollout_state=input_state.model_copy(deep=True),
                    )
                )
                for target in initial_targets
            }
            baseline = endpoint_results[failed_endpoint_rank]

            hf_dir = Path(self.temp_dir.name) / "hf-disk-update"
            hf_dir.mkdir(parents=True, exist_ok=True)
            train_controller.save_hf(str(hf_dir))

            shutdown_ranks = ray.get(
                rollout_controller.shutdown_weight_update_endpoint.remote(endpoint_rank=failed_endpoint_rank)
            )
            self.assertGreater(len(shutdown_ranks), 0)

            targets_after_shutdown = ray.get(rollout_controller.get_weight_update_targets.remote())
            self.assertTrue(any(not target.is_active for target in targets_after_shutdown))

            ray.get(rollout_controller.restart_inactive_workers.remote())
            targets_after_restart = ray.get(rollout_controller.get_weight_update_targets.remote())
            self.assertTrue(all(target.is_active for target in targets_after_restart))
            self.assertTrue(any(target.endpoint_rank == failed_endpoint_rank for target in targets_after_restart))

            self._update_weights(
                train_controller,
                rollout_controller,
                "disk",
                target_endpoint_ranks={failed_endpoint_rank},
                disk_weight_path=str(hf_dir),
            )

            recovered = ray.get(
                rollout_controller.generate_from_weight_update_endpoint.remote(
                    endpoint_rank=failed_endpoint_rank,
                    rollout_state=input_state.model_copy(deep=True),
                )
            )
            self.assertEqual(recovered.response, baseline.response)
        finally:
            ray.get(rollout_controller.shutdown.remote(), timeout=60)


if __name__ == "__main__":
    unittest.main()

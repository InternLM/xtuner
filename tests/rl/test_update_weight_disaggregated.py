import os
import tempfile
import unittest

import ray
import requests

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLDense4BConfig
from xtuner.v1.rl.loss import GRPOLossConfig as LossConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import (
    TrainingController,
    TrainingWorker as BaseTrainingWorker,
    WorkerConfig,
)
from xtuner.v1.rl.utils import (
    AcceleratorResourcesConfig,
    AutoAcceleratorWorkers,
    CPUResourceManager,
    clear_cpu_resource_manager,
    set_cpu_resource_manager,
)

TEST_TEXT_MESSAGES = [{"role": "user", "content": "Hello!"}]
MODEL_PATH = os.environ["QWEN3_VL_DENSE_PATH"]

class TestUpdateWeightDisaggregated(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if MODEL_PATH is None:
            raise unittest.SkipTest("MODEL_PATH is not set")
        os.environ["XTUNER_USE_FA3"] = "1"
        # TODO(shipengcheng): SGLang disaggregated weight update cannot use
        # NCCL_CUMEM for now. Remove this after the root cause is fixed.
        os.environ["NCCL_CUMEM_ENABLE"] = "0"

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]

    def setUp(self):
        ray.init(num_cpus=128, ignore_reinit_error=True)
        self.model_path = MODEL_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.init_config()
        self.train_pg = AutoAcceleratorWorkers.build_placement_group(self.train_resources_cfg,
        name=f"test_update_weight_train_{id(self)}")
        self.rollout_pg = AutoAcceleratorWorkers.build_placement_group(self.rollout_resources_cfg,
        name=f"test_update_weight_rollout_{id(self)}")
        set_cpu_resource_manager(
            CPUResourceManager(accelerator_placement_groups=[self.train_pg, self.rollout_pg])
        )

    def tearDown(self):
        clear_cpu_resource_manager()
        ray.shutdown()
        self.temp_dir.cleanup()

    def init_config(self):
        train_num_workers = int(os.environ.get("TRAIN_NUM_WORKERS", "4"))
        rollout_num_workers = int(os.environ.get("ROLLOUT_NUM_WORKERS", "4"))

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
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            rollout_cross_node_comm=False,
            tensor_parallel_size=int(os.environ.get("ROLLOUT_TP_SIZE", "4")),
            expert_parallel_size=1,
            gpus_per_node=int(os.environ.get("GPUS_PER_NODE", "8")),
            dtype="bfloat16",
            skip_load_weights=True,
            context_length=256,
            worker_log_dir=self.worker_log_dir,
            gpu_memory_utilization=float(os.environ.get("ROLLOUT_GPU_MEMORY_UTILIZATION", "0.5")),
        )

        model_cfg = Qwen3VLDense4BConfig()
        optim_cfg = AdamWConfig(lr=5e-7, foreach=False)
        fsdp_cfg = FSDPConfig(ep_size=1)
        lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=5e-7)
        self.worker_cfg = WorkerConfig(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
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
            lr_cfg=lr_cfg,
            fsdp_cfg=fsdp_cfg,
            load_from=MODEL_PATH,
            sp_size=1,
            pack_max_length=1024,
        )

    def _check_sglang_weights(self, rollout_controller, action):
        info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())
        active_urls = [
            url
            for url, is_active in info_dict["worker_server_urls_status"].items()
            if is_active
        ]
        self.assertGreater(len(active_urls), 0)
        results = []
        for url in active_urls:
            response = requests.post(
                f"{url}/weights_checker",
                json={"action": action},
                timeout=300,
            )
            response.raise_for_status()
            results.append(response.json())
        return results

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "sglang backend is not enabled")
    def test_sglang_disaggregated_update_weight_and_generate(self):
        TrainingWorker = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                }
            },
        )(BaseTrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(
            TrainingWorker, self.worker_cfg, self.train_pg
        )
        ray.get([worker.test_all_reduce.remote() for worker in train_workers])
        train_controller = TrainingController(workers=train_workers)
        
        self.rollout_cfg.skip_load_weights = False
        rollout_controller = self.rollout_cfg.build(self.rollout_pg)

        sample_params = SampleParams(temperature=0.0, max_tokens=128, top_k=1)
        input_state = RolloutState(message=TEST_TEXT_MESSAGES, sample_params=sample_params)
        res_baseline = ray.get(rollout_controller.generate.remote(rollout_state=input_state))

        info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())
        train_controller.update_rollout_info(info_dict, train_rollout_mode="disaggregated")
        train_controller.update_weights()

        res_update_weight = ray.get(rollout_controller.generate.remote(rollout_state=input_state))
        self.assertEqual(res_update_weight.response, res_baseline.response)
        ray.get(rollout_controller.shutdown.remote(), timeout=60)

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "sglang backend is not enabled")
    def test_sglang_disaggregated_update_weight_equal_after_reset(self):
        # This test verifies SGLang rollout weight update correctness with a parameter-only check.
        # The SGLang parameter-only WeightChecker actions are implemented in 
        # https://github.com/PengchengShi00/sglang/commit/05e89d63b5a1a80671b267ff4494ad950b2aba75.
        # Flow: snapshot_parameters -> reset_parameters -> update_weights -> compare_parameters.
        TrainingWorker = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                }
            },
        )(BaseTrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(
            TrainingWorker, self.worker_cfg, self.train_pg
        )
        ray.get([worker.test_all_reduce.remote() for worker in train_workers])
        train_controller = TrainingController(workers=train_workers)

        self.rollout_cfg.skip_load_weights = False
        rollout_controller = self.rollout_cfg.build(self.rollout_pg)

        try:
            self._check_sglang_weights(rollout_controller, action="snapshot_parameters")
            self._check_sglang_weights(rollout_controller, action="reset_parameters")

            info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())
            train_controller.update_rollout_info(info_dict, train_rollout_mode="disaggregated")
            train_controller.update_weights()

            self._check_sglang_weights(rollout_controller, action="compare_parameters")
        finally:
            ray.get(rollout_controller.shutdown.remote(), timeout=60)

    @unittest.skipIf(
        os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0",
        "lmdeploy backend is not enabled",
    )
    def test_lmdeploy_disaggregated_update_weight_and_generate(self):
        TrainingWorker = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                }
            },
        )(BaseTrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(
            TrainingWorker, self.worker_cfg, self.train_pg
        )
        ray.get([worker.test_all_reduce.remote() for worker in train_workers])
        train_controller = TrainingController(workers=train_workers)

        self.rollout_cfg.skip_load_weights = False
        self.rollout_cfg.extra_rollout_config = {
            "lmdeploy_backend": "pytorch",
            "lmdeploy_distributed_executor_backend": "ray",
        }
        rollout_controller = self.rollout_cfg.build(self.rollout_pg)

        sample_params = SampleParams(temperature=0.0, max_tokens=128, top_k=1)
        input_state = RolloutState(message=TEST_TEXT_MESSAGES, sample_params=sample_params)
        res_baseline = ray.get(rollout_controller.generate.remote(rollout_state=input_state))

        info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())
        train_controller.update_rollout_info(info_dict, train_rollout_mode="disaggregated")
        train_controller.update_weights()

        res_update_weight = ray.get(rollout_controller.generate.remote(rollout_state=input_state))
        self.assertEqual(res_update_weight.response, res_baseline.response)
        ray.get(rollout_controller.shutdown.remote(), timeout=60)

if __name__ == "__main__":
    unittest.main()

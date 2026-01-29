import os
import unittest
import tempfile
import ray

from xtuner.v1.ray.base import AutoAcceleratorWorkers
from xtuner.v1.ray.rollout import RolloutController
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.model.moe.moe import BalancingLossConfig, ZLossConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.rl.base import WorkerConfig, TrainingController, TrainingWorker as BaseTrainingWorker
from xtuner.v1.rl.grpo.loss import GRPOLossConfig as LossConfig
from xtuner.v1.model import get_model_config_from_hf

TEST_TEXT_MESSAGES=[{"role": "user", "content": "Hello!"}]
MODEL_PATH = os.environ["QWEN3_MOE_PATH"]

class TestUpdateWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["XTUNER_USE_FA3"] = "1"
        os.environ["LMD_SKIP_WARMUP"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]
        del os.environ["LMD_SKIP_WARMUP"]
        
    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.model_path = MODEL_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.init_config()
        self.pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)

    def tearDown(self):
        ray.shutdown()
        self.temp_dir.cleanup()

    def init_config(self):
        self.resources_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=4,
            num_cpus_per_worker=12,
            cpu_memory_per_worker=16 * 1024 ** 3,  # 16 GB
        )
        self.rollout_cfg = RolloutConfig(
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            rollout_cross_node_comm=False,
            tensor_parallel_size=4,
            expert_parallel_size=1,
            gpus_per_node=8, # gpu: 8, npu: 16
            dtype="bfloat16",
            skip_load_weights=True,
            context_length=256,
            worker_log_dir=self.worker_log_dir,
            gpu_memory_utilization=0.5,
        )

        # training config
        model_cfg = get_model_config_from_hf(model_path=MODEL_PATH)
        if hasattr(model_cfg, 'z_loss_cfg'):
            model_cfg.z_loss_cfg = ZLossConfig()
        if hasattr(model_cfg, 'balancing_loss_cfg'):
            model_cfg.balancing_loss_cfg = BalancingLossConfig()
        optim_cfg: AdamWConfig = AdamWConfig(lr=5e-7, foreach=False)
        fsdp_cfg: FSDPConfig = FSDPConfig(ep_size=4)
        model_cfg.ep_size = fsdp_cfg.ep_size
        lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=5e-7)
        self.worker_cfg: WorkerConfig = WorkerConfig(
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
                mode="eager"),
            lr_cfg=lr_cfg,
            fsdp_cfg=fsdp_cfg,
            load_from=MODEL_PATH,
            sp_size=1,
            pack_max_length=1024,
        )

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_update_weight_and_generate(self):
        # init train
        TrainingWorker = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                }
            },
        )(BaseTrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(
            TrainingWorker, self.worker_cfg, self.pg
        )
        futures = [ worker.test_all_reduce.remote() for worker in train_workers ]
        ray.get(futures)
        train_controller = TrainingController.remote(
            workers=train_workers,
        )
        ray.get(train_controller.__ray_ready__.remote())

        # fixed sample params
        sample_params = SampleParams(temperature=0.0, max_tokens=128, top_k=1)

        # init rollout_update
        rollout_controller = ray.remote(RolloutController).remote(
            self.rollout_cfg,
            self.pg,
        )
        info_dict = ray.get(rollout_controller.get_rollout_info.remote())
        ray.get(train_controller.update_rollout_info.remote(info_dict))
        
        # update weights
        ray.get(rollout_controller.offload.remote())
        ray.get(rollout_controller.onload_weights.remote())
        ray.get(train_controller.offload.remote(["optimizer"]))
        ray.get(train_controller.update_weights.remote())
        ray.get(train_controller.offload.remote(["model"]))
        ray.get(rollout_controller.onload_kvcache.remote())

        res_update_weight = ray.get(rollout_controller.rollout.remote(prompt=TEST_TEXT_MESSAGES, sample_params=sample_params))       
        ray.get(rollout_controller.shutdown.remote(), timeout=60)

        # init rollout_ref
        self.rollout_cfg.skip_load_weights = False
        rollout_controller_ref = ray.remote(RolloutController).remote(
            self.rollout_cfg,
            self.pg,
        )

        res_ref = ray.get(rollout_controller_ref.rollout.remote(prompt=TEST_TEXT_MESSAGES, sample_params=sample_params))  
        ray.get(rollout_controller_ref.shutdown.remote(), timeout=60)

        self.assertEqual(res_update_weight.response, res_ref.response)


if __name__ == "__main__":
    test_instance = TestUpdateWeight()
    test_instance.setUp()
    try:
        test_instance.test_lmdeploy_update_weight_and_generate()
    finally:
        test_instance.tearDown()

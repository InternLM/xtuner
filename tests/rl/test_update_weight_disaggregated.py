import os
import hashlib
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

import ray
import torch
import torch.distributed as dist

from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.data_proto.rl_data import SampleParams, RolloutState
from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.rl.trainer import WorkerConfig, TrainingController, TrainingWorker as BaseTrainingWorker
from xtuner.v1.rl.loss import GRPOLossConfig as LossConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.utils import ray_method

TEST_TEXT_MESSAGES = [{"role": "user", "content": "Hello!"}]
MODEL_PATH = os.environ.get("MODEL_PATH") or os.environ.get("QWEN3_VL_DENSE_PATH")


class HashingTrainingWorker(BaseTrainingWorker):
    def _init_update_weighter(self):
        super()._init_update_weighter()
        self._test_update_weight_sha256 = hashlib.sha256()
        self._test_update_weight_bucket_count = 0

    @ray_method
    def reset_update_weight_sha256(self):
        self._test_update_weight_sha256 = hashlib.sha256()
        self._test_update_weight_bucket_count = 0

    @ray_method
    def get_update_weight_sha256(self):
        return {
            "rank": self.rank,
            "sha256": self._test_update_weight_sha256.hexdigest(),
            "bucket_count": self._test_update_weight_bucket_count,
        }

    def request_update_params(self, state_dict, train_enable_ep=False, finished=False, profile_context=None):
        if state_dict and dist.get_rank() == 0:
            for name in sorted(state_dict):
                tensor = state_dict[name].detach().contiguous().cpu()
                self._test_update_weight_sha256.update(name.encode("utf-8"))
                self._test_update_weight_sha256.update(str(tensor.dtype).encode("utf-8"))
                self._test_update_weight_sha256.update(str(tuple(tensor.shape)).encode("utf-8"))
                self._test_update_weight_sha256.update(tensor.view(torch.uint8).numpy().tobytes())
            self._test_update_weight_bucket_count += 1
        return super().request_update_params(
            state_dict,
            train_enable_ep=train_enable_ep,
            finished=finished,
        )


class TestUpdateWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if MODEL_PATH is None:
            raise unittest.SkipTest("MODEL_PATH is not set")
        os.environ["XTUNER_USE_FA3"] = "1"

    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.model_path = MODEL_PATH
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.init_config()
        self.pg = AutoAcceleratorWorkers.build_placement_group(
            self.train_resources_cfg,
            name=f"test_update_weight_train_{id(self)}",
        )

    def tearDown(self):
        ray.shutdown()
        self.temp_dir.cleanup()

    def init_config(self):
        train_num_workers = int(os.environ.get("TRAIN_NUM_WORKERS", "4"))
        rollout_num_workers = int(os.environ.get("ROLLOUT_NUM_WORKERS", "4"))
        rollout_tp_size = int(os.environ.get("ROLLOUT_TP_SIZE", str(rollout_num_workers)))
        rollout_ep_size = int(os.environ.get("ROLLOUT_EP_SIZE", "1"))
        train_ep_size = int(os.environ.get("TRAIN_EP_SIZE", "1"))

        self.train_resources_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=train_num_workers,
            num_cpus_per_worker=float(os.environ.get("TRAIN_CPUS_PER_WORKER", "12")),
            cpu_memory_per_worker=int(os.environ.get("TRAIN_CPU_MEMORY_PER_WORKER", str(16 * 1024**3))),
        )
        self.rollout_resources_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=rollout_num_workers,
            num_cpus_per_worker=float(os.environ.get("ROLLOUT_CPUS_PER_WORKER", "12")),
            cpu_memory_per_worker=int(os.environ.get("ROLLOUT_CPU_MEMORY_PER_WORKER", str(16 * 1024**3))),
        )
        self.rollout_cfg = RolloutConfig(
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            rollout_cross_node_comm=False,
            tensor_parallel_size=rollout_tp_size,
            expert_parallel_size=rollout_ep_size,
            gpus_per_node=int(os.environ.get("GPUS_PER_NODE", "8")),  # gpu: 8, npu: 16
            dtype="bfloat16",
            skip_load_weights=True,
            context_length=int(os.environ.get("ROLLOUT_CONTEXT_LENGTH", "256")),
            worker_log_dir=self.worker_log_dir,
            gpu_memory_utilization=float(os.environ.get("ROLLOUT_GPU_MEMORY_UTILIZATION", "0.5")),
        )

        # training config
        model_cfg = get_model_config_from_hf(Path(MODEL_PATH))
        optim_cfg: AdamWConfig = AdamWConfig(lr=5e-7, foreach=False)
        fsdp_cfg: FSDPConfig = FSDPConfig(ep_size=train_ep_size)
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

    def _build_train_controller(self, worker_cls=BaseTrainingWorker):
        TrainingWorker = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                }
            },
        )(worker_cls)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(
            TrainingWorker, self.worker_cfg, self.pg
        )
        ray.get([worker.test_all_reduce.remote() for worker in train_workers])
        train_controller = TrainingController(workers=train_workers)
        train_controller.set_train_rollout_mode("disaggregated")
        return train_controller

    def _build_sglang_rollout_controller(self):
        rollout_pg = AutoAcceleratorWorkers.build_placement_group(
            self.rollout_resources_cfg,
            name=f"test_update_weight_rollout_{id(self)}",
        )
        self.rollout_cfg.skip_load_weights = False
        return ray.remote(RolloutController).remote(
            self.rollout_cfg,
            rollout_pg,
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
        train_controller = TrainingController(
            workers=train_workers,
        )
        # fixed sample params
        sample_params = SampleParams(temperature=0.0, max_tokens=128, top_k=1)

        # init rollout_controller and rollout baseline
        self.rollout_cfg.skip_load_weights = False
        rollout_controller = ray.remote(RolloutController).remote(
            self.rollout_cfg,
            self.pg,
        )

        input_state = RolloutState(message=TEST_TEXT_MESSAGES, sample_params=sample_params)
        res_baseline = ray.get(rollout_controller.generate.remote(rollout_state=input_state)) 
        
        # start update weight test
        info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())
        train_controller.update_rollout_info(info_dict)
        
        # update weights
        ray.get(rollout_controller.offload.remote())
        train_controller.onload(target="all")
        train_controller.offload("optimizer")
        ray.get(rollout_controller.onload_weights.remote())
        train_controller.update_weights()
        train_controller.offload("model")
        ray.get(rollout_controller.onload_kvcache.remote())

        res_update_weight = ray.get(rollout_controller.generate.remote(rollout_state=input_state)) 
        self.assertEqual(res_update_weight.response, res_baseline.response)
        ray.get(rollout_controller.shutdown.remote(), timeout=60)

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "sglang backend is not enabled")
    def test_sglang_disaggregated_update_weight_and_generate(self):
        # init train on a dedicated placement group
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
        futures = [worker.test_all_reduce.remote() for worker in train_workers]
        ray.get(futures)
        train_controller = TrainingController(workers=train_workers)
        train_controller.set_train_rollout_mode("disaggregated")

        # init rollout on a separate placement group
        rollout_pg = AutoAcceleratorWorkers.build_placement_group(
            self.rollout_resources_cfg,
            name=f"test_update_weight_rollout_{id(self)}",
        )
        self.rollout_cfg.skip_load_weights = False
        rollout_controller = ray.remote(RolloutController).remote(
            self.rollout_cfg,
            rollout_pg,
        )

        sample_params = SampleParams(temperature=0.0, max_tokens=128, top_k=1)
        input_state = RolloutState(message=TEST_TEXT_MESSAGES, sample_params=sample_params)
        res_baseline = ray.get(rollout_controller.generate.remote(rollout_state=input_state))

        info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())
        train_controller.update_rollout_info(info_dict)

        train_controller.update_weights()

        res_update_weight = ray.get(rollout_controller.generate.remote(rollout_state=input_state))
        self.assertEqual(res_update_weight.response, res_baseline.response)
        ray.get(rollout_controller.shutdown.remote(), timeout=60)

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "sglang backend is not enabled")
    def test_sglang_disaggregated_update_weight_after_pause_and_generate(self):
        train_controller = self._build_train_controller()
        rollout_controller = self._build_sglang_rollout_controller()

        sample_params = SampleParams(temperature=0.0, max_tokens=128, top_k=1)
        input_state = RolloutState(message=TEST_TEXT_MESSAGES, sample_params=sample_params)
        res_baseline = ray.get(rollout_controller.generate.remote(rollout_state=input_state))

        info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())
        train_controller.update_rollout_info(info_dict)

        ray.get(rollout_controller.pause_generation.remote())
        time.sleep(float(os.environ.get("XTUNER_UPDATE_WEIGHT_PAUSE_SLEEP", "2")))
        train_controller.update_weights()
        ray.get(rollout_controller.continue_generation.remote())

        res_update_weight = ray.get(rollout_controller.generate.remote(rollout_state=input_state))
        self.assertEqual(res_update_weight.response, res_baseline.response)
        ray.get(rollout_controller.shutdown.remote(), timeout=60)

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "sglang backend is not enabled")
    def test_sglang_disaggregated_update_weight_sha256_is_stable(self):
        train_controller = self._build_train_controller(worker_cls=HashingTrainingWorker)
        rollout_controller = self._build_sglang_rollout_controller()

        info_dict = ray.get(rollout_controller.get_rollout_metadata.remote())
        train_controller.update_rollout_info(info_dict)

        ray.get([worker.reset_update_weight_sha256.remote() for worker in train_controller.workers])
        train_controller.update_weights()
        first_hashes = ray.get([worker.get_update_weight_sha256.remote() for worker in train_controller.workers])

        ray.get([worker.reset_update_weight_sha256.remote() for worker in train_controller.workers])
        train_controller.update_weights()
        second_hashes = ray.get([worker.get_update_weight_sha256.remote() for worker in train_controller.workers])

        first_rank0_hash = next(item for item in first_hashes if item["rank"] == 0)
        second_rank0_hash = next(item for item in second_hashes if item["rank"] == 0)
        self.assertGreater(first_rank0_hash["bucket_count"], 0)
        self.assertEqual(first_rank0_hash["sha256"], second_rank0_hash["sha256"])
        self.assertEqual(first_rank0_hash["bucket_count"], second_rank0_hash["bucket_count"])
        ray.get(rollout_controller.shutdown.remote(), timeout=60)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest

import numpy as np
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
from xtuner.v1.rl.rollout.worker_registry import WorkerLifecycleState

RL_TRAINER_RAY_GET_TIMEOUT = 3600
TEST_TEXT_MESSAGES = [{"role": "user", "content": "Hello!"}]
# Tokens can still match when a few shards are wrong; sampled-token logprobs
# catch that. Keep this tight: identical weights should stay near exact.
GENERATE_LOGPROB_RTOL = 1e-5
GENERATE_LOGPROB_ATOL = 1e-5
# SGLang: temperature=0 is greedy. LMDeploy /generate always sets do_sample=True
# and divides logits by temperature, so greedy is top_k=1 with temperature=1.0.
SGLANG_GREEDY_SAMPLE_PARAMS = SampleParams(
    temperature=0.0,
    max_tokens=128,
    top_k=1,
    return_logprob=True,
    return_token_ids=True,
)
LMDEPLOY_GREEDY_SAMPLE_PARAMS = SampleParams(
    temperature=1.0,
    max_tokens=128,
    top_k=1,
    return_logprob=True,
    return_token_ids=True,
)
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
        if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

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
            weight_transport_type="nccl",
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
        targets = ray.get(rollout_controller.get_weight_update_targets.remote())
        active_urls = [target.server_url for target in targets if target.lifecycle_state == WorkerLifecycleState.ACTIVE.value]
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

    def _assert_generate_outputs_match(
        self,
        actual: RolloutState,
        expected: RolloutState,
        err_msg: str,
    ) -> None:
        self.assertEqual(actual.response, expected.response)
        self.assertEqual(actual.response_ids, expected.response_ids)
        self.assertIsNotNone(expected.logprobs)
        self.assertIsNotNone(actual.logprobs)
        self.assertGreater(len(expected.logprobs), 0)
        self.assertEqual(len(actual.logprobs), len(expected.logprobs))
        self.assertEqual(len(actual.logprobs), len(actual.response_ids or []))
        np.testing.assert_allclose(
            actual.logprobs,
            expected.logprobs,
            rtol=GENERATE_LOGPROB_RTOL,
            atol=GENERATE_LOGPROB_ATOL,
            err_msg=err_msg,
        )

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

        sample_params = SGLANG_GREEDY_SAMPLE_PARAMS

        def _generate() -> RolloutState:
            return ray.get(
                rollout_controller.generate.remote(
                    rollout_state=RolloutState(message=TEST_TEXT_MESSAGES, sample_params=sample_params),
                ),
                timeout=RL_TRAINER_RAY_GET_TIMEOUT,
            )

        res_baseline = _generate()
        res_repeat = _generate()
        self._assert_generate_outputs_match(
            res_repeat,
            res_baseline,
            err_msg="rollout logprobs changed between repeated generates before weight update",
        )

        # 1) 清 KV + 释放旧权重（sleep level=2 -> meta）
        ray.get(
            rollout_controller.offload.remote(),
            timeout=RL_TRAINER_RAY_GET_TIMEOUT,
        )
        # 2) 只 wakeup weights（empty_init），此时不要 onload_kvcache/warmup
        ray.get(
            rollout_controller.onload_weights.remote(),
            timeout=RL_TRAINER_RAY_GET_TIMEOUT,
        )
        targets = ray.get(rollout_controller.get_weight_update_targets.remote())
        train_controller.bind_rollout_weight_update(
            targets=targets,
            rollout_config=self.rollout_cfg,
        )
        # 3) 先把权重更新完（含 finished=True finalize）
        train_controller.weight_update()
        # 4) 最后再 onload_kvcache（这里才会 warmup）
        ray.get(
            rollout_controller.onload_kvcache.remote(),
            timeout=RL_TRAINER_RAY_GET_TIMEOUT,
        )

        res_update_weight = _generate()
        self._assert_generate_outputs_match(
            res_update_weight,
            res_baseline,
            err_msg="rollout logprobs changed after weight update",
        )
        ray.get(rollout_controller.shutdown.remote(), timeout=60)

    @unittest.skip("skip sglang parameter-only weight check test until the parameter-check-only patch is applied")
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

            targets = ray.get(rollout_controller.get_weight_update_targets.remote())
            train_controller.bind_rollout_weight_update(
                targets=targets,
                rollout_config=self.rollout_cfg,
            )
            train_controller.weight_update()

            self._check_sglang_weights(rollout_controller, action="compare_parameters")
        finally:
            ray.get(rollout_controller.shutdown.remote(), timeout=60)

    def test_lmdeploy_disaggregated_update_weight_and_generate(self):
        # TODO(shipengcheng): Remove skip when CI update lmdeploy.
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

        sample_params = LMDEPLOY_GREEDY_SAMPLE_PARAMS

        def _generate() -> RolloutState:
            return ray.get(
                rollout_controller.generate.remote(
                    rollout_state=RolloutState(message=TEST_TEXT_MESSAGES, sample_params=sample_params),
                ),
                timeout=RL_TRAINER_RAY_GET_TIMEOUT,
            )

        res_baseline = _generate()
        res_repeat = _generate()
        self._assert_generate_outputs_match(
            res_repeat,
            res_baseline,
            err_msg="rollout logprobs changed between repeated generates before weight update",
        )

        # 1) 清 KV + 释放旧权重（sleep level=2 -> meta）
        ray.get(
            rollout_controller.offload.remote(),
            timeout=RL_TRAINER_RAY_GET_TIMEOUT,
        )
        # 2) 只 wakeup weights（empty_init），此时不要 onload_kvcache/warmup
        ray.get(
            rollout_controller.onload_weights.remote(),
            timeout=RL_TRAINER_RAY_GET_TIMEOUT,
        )
        targets = ray.get(rollout_controller.get_weight_update_targets.remote())
        train_controller.bind_rollout_weight_update(
            targets=targets,
            rollout_config=self.rollout_cfg,
        )
        # 3) 先把权重更新完（含 finished=True finalize）
        train_controller.weight_update()
        # 4) 最后再 onload_kvcache（这里才会 warmup）
        ray.get(
            rollout_controller.onload_kvcache.remote(),
            timeout=RL_TRAINER_RAY_GET_TIMEOUT,
        )

        res_update_weight = _generate()
        self._assert_generate_outputs_match(
            res_update_weight,
            res_baseline,
            err_msg="rollout logprobs changed after weight update",
        )
        ray.get(rollout_controller.shutdown.remote(), timeout=60)

if __name__ == "__main__":
    unittest.main()

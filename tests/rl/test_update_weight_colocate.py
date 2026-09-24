"""共卡部署下的模型权重更新正确性测试。

覆盖 IPC 与 checkpoint-engine：训练侧权重写入推理引擎后，检查引擎状态是否与
更新前一致。两类用例互补，不是重复。

Generate 检查（默认启用）
    greedy generate 两次，确认引擎自身可复现；再走 IPC 更新后第三次 generate。
    比较 response / response_ids / sampled-token logprobs。
    SGLang greedy 用 temperature=0；LMDeploy /generate 用 top_k=1 且 temperature=1.0。
    - test_sglang_colocate_ipc_update_weight_and_generate
    - test_lmdeploy_colocate_ipc_update_weight_and_generate

参数逐点检查（当前 skip，依赖 SGLang WeightChecker patch）
    snapshot_parameters -> reset_parameters -> update_weights -> compare_parameters。
    不 generate，直接比对引擎参数。WeightChecker:
    https://github.com/PengchengShi00/sglang/commit/05e89d63b5a1a80671b267ff4494ad950b2aba75
    - test_sglang_colocate_ipc_update_weight
    - test_sglang_colocate_checkpoint_engine_update_weight_train_register
"""

import os
import tempfile
import unittest

import numpy as np
import ray
import requests

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.module.mtp import MTPConfig

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
MODEL_PATH = os.environ["QWEN3_5_MOE_PATH"]


class TestUpdateWeightColocate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if MODEL_PATH is None:
            raise unittest.SkipTest("MODEL_PATH is not set")
        os.environ["XTUNER_USE_FA3"] = "1"
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["NCCL_IB_HCA"] = "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
        os.environ["PS_P2P_STORE_RDMA_DEVICES"] = "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
        if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
        elif os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "1":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        else:
            raise unittest.SkipTest("XTUNER_USE_SGLANG or XTUNER_USE_LMDEPLOY is not set")
    @classmethod
    def tearDownClass(cls) -> None:
        del os.environ["XTUNER_USE_FA3"]
        del os.environ["NCCL_CUMEM_ENABLE"]
        del os.environ["NCCL_IB_HCA"]
        del os.environ["PS_P2P_STORE_RDMA_DEVICES"]
        del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    def setUp(self):
        self.model_path = MODEL_PATH
        self.temp_dir = None
        self.train_controller = None
        self.rollout_controller = None

    def tearDown(self):
        if self.train_controller is not None:
            self.train_controller = None
        if self.rollout_controller is not None:
            ray.get(self.rollout_controller.shutdown.remote(), timeout=60)
            self.rollout_controller = None
        clear_cpu_resource_manager()
        if ray.is_initialized():
            ray.shutdown()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None

    def init_config(self, *, weight_transport_type: str, extra_rollout_config: dict | None = None):
        nnodes = int(os.environ.get("WORLD_SIZE", "1"))
        num_workers = int(os.environ.get("COLOCATE_NUM_WORKERS", str(8 * nnodes)))
        rollout_tp_size = 4
        rollout_ep_size = 1

        self.resources_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=num_workers,
            num_cpus_per_worker=12,
            cpu_memory_per_worker=32 * 1024**3,
        )
        self.rollout_cfg = RolloutConfig(
            env="test_rollout",
            device=self.resources_cfg.accelerator,
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            rollout_cross_node_comm=False,
            tensor_parallel_size=rollout_tp_size,
            expert_parallel_size=rollout_ep_size,
            gpus_per_node=int(os.environ.get("GPUS_PER_NODE", "8")),
            dtype="bfloat16",
            skip_load_weights=False,
            weight_transport_type=weight_transport_type,
            checkpoint_name_prefix=f"test-update-weight-colocate-{id(self)}",
            context_length=int(os.environ.get("ROLLOUT_CONTEXT_LENGTH", "10240")),
            worker_log_dir=self.worker_log_dir,
            gpu_memory_utilization=float(os.environ.get("ROLLOUT_GPU_MEMORY_UTILIZATION", "0.8")),
            extra_rollout_config=extra_rollout_config or {},
        )

        model_cfg = Qwen3_5_VLMoE35BA3Config(freeze_vision=True, freeze_projector=True)
        model_cfg.text_config.mtp_config = MTPConfig(num_layers=1)
        model_cfg.text_config.ep_size = 1

        optim_cfg = AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1)
        fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1)
        lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
        self.worker_cfg = WorkerConfig(
            model_cfg=model_cfg,
            load_from=MODEL_PATH,
            optim_cfg=optim_cfg,
            loss_cfg=LossConfig(
                policy_loss_cfg=dict(
                    cliprange_high=0.28,
                    cliprange_low=0.2,
                    loss_type="vanilla",
                    clip_ratio_c=10.0,
                    log_prob_diff_min=-20.0,
                    log_prob_diff_max=20.0,
                ),
                ignore_idx=-100,
                use_kl_loss=False,
                kl_loss_coef=0.0,
                kl_loss_type="low_var_kl",
                mode="chunk",
                chunk_size=512,
            ),
            lr_cfg=lr_cfg,
            fsdp_cfg=fsdp_cfg,
            sp_size=1,
            optimizer_steps=1,
            pack_max_length=int(os.environ.get("PACK_MAX_LENGTH", str(10 * 1024))),
        )

    def _setup_engines(self, *, weight_transport_type: str, extra_rollout_config: dict | None = None):
        ray.init(num_cpus=128, ignore_reinit_error=True)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")
        self.init_config(
            weight_transport_type=weight_transport_type,
            extra_rollout_config=extra_rollout_config,
        )
        self.pg = AutoAcceleratorWorkers.build_placement_group(
            self.resources_cfg,
            name=f"test_update_weight_colocate_{id(self)}",
        )
        set_cpu_resource_manager(CPUResourceManager(accelerator_placement_groups=[self.pg]))

        TrainingWorker = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                }
            },
        )(BaseTrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(TrainingWorker, self.worker_cfg, self.pg)
        ray.get([worker.test_all_reduce.remote() for worker in train_workers])
        self.train_controller = TrainingController(workers=train_workers)
        self.train_controller.offload(target="all")

        self.rollout_controller = self.rollout_cfg.build(self.pg)
        return self.train_controller, self.rollout_controller

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
        self.assertEqual(actual.status, Status.COMPLETED, actual.error_msg)
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

    def _run_colocate_ipc_update_weight_and_generate(
        self,
        extra_rollout_config: dict | None = None,
        sample_params: SampleParams | None = None,
    ):
        train_controller, rollout_controller = self._setup_engines(
            weight_transport_type="ipc",
            extra_rollout_config=extra_rollout_config,
        )

        sample_params = sample_params or SGLANG_GREEDY_SAMPLE_PARAMS

        def _generate() -> RolloutState:
            return ray.get(
                rollout_controller.generate.remote(
                    rollout_state=RolloutState(message=TEST_TEXT_MESSAGES, sample_params=sample_params),
                ),
                timeout=RL_TRAINER_RAY_GET_TIMEOUT,
            )

        res_baseline = _generate()
        self.assertEqual(res_baseline.status, Status.COMPLETED, res_baseline.error_msg)
        res_repeat = _generate()
        self._assert_generate_outputs_match(
            res_repeat,
            res_baseline,
            err_msg="rollout logprobs changed between repeated generates before weight update",
        )

        # Colocate IPC: free rollout, bring train weights online, then update.
        ray.get(rollout_controller.offload.remote(), timeout=RL_TRAINER_RAY_GET_TIMEOUT)
        train_controller.onload(target="model")
        ray.get(rollout_controller.onload_weights.remote(), timeout=RL_TRAINER_RAY_GET_TIMEOUT)
        targets = ray.get(rollout_controller.get_weight_update_targets.remote())
        train_controller.bind_rollout_weight_update(
            targets=targets,
            rollout_config=self.rollout_cfg,
        )
        train_controller.weight_update()
        train_controller.offload(target="model")
        ray.get(rollout_controller.onload_kvcache.remote(), timeout=RL_TRAINER_RAY_GET_TIMEOUT)

        res_update_weight = _generate()
        self._assert_generate_outputs_match(
            res_update_weight,
            res_baseline,
            err_msg="rollout logprobs changed after weight update",
        )

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "sglang backend is not enabled")
    def test_sglang_colocate_ipc_update_weight_and_generate(self):
        self._run_colocate_ipc_update_weight_and_generate()

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_colocate_ipc_update_weight_and_generate(self):
        self._run_colocate_ipc_update_weight_and_generate(
            extra_rollout_config={"lmdeploy_backend": "pytorch"},
            sample_params=LMDEPLOY_GREEDY_SAMPLE_PARAMS,
        )

    @unittest.skip("skip sglang parameter-only weight check test until the parameter-check-only patch is applied")
    def test_sglang_colocate_ipc_update_weight(self):
        train_controller, rollout_controller = self._setup_engines(weight_transport_type='ipc')

        self._check_sglang_weights(rollout_controller, action="snapshot_parameters")
        self._check_sglang_weights(rollout_controller, action="reset_parameters")
        
        targets = ray.get(rollout_controller.get_weight_update_targets.remote())
        train_controller.bind_rollout_weight_update(
            targets=targets,
            rollout_config=self.rollout_cfg,
        )

        ray.get(rollout_controller.offload.remote(), timeout=300)
        ray.get(self.rollout_controller.onload_weights.remote(), timeout=300)
        train_controller.onload(target="model")
        train_controller.weight_update()

        self._check_sglang_weights(rollout_controller, action="compare_parameters")


    @unittest.skip("skip sglang parameter-only weight check test until the parameter-check-only patch is applied")
    def test_sglang_colocate_checkpoint_engine_update_weight_train_register(self):
        train_controller, rollout_controller = self._setup_engines(weight_transport_type="checkpoint_engine")

        self._check_sglang_weights(rollout_controller, action="snapshot_parameters")
        self._check_sglang_weights(rollout_controller, action="reset_parameters")
        
        targets = ray.get(rollout_controller.get_weight_update_targets.remote())
        train_controller.bind_rollout_weight_update(
            targets=targets,
            rollout_config=self.rollout_cfg,
        )
        ray.get(rollout_controller.offload.remote(), timeout=300)
        train_controller.onload(target="model")
        train_controller.weight_update(need_register=True, need_update=False)
        train_controller.offload(target="model")
        ray.get(self.rollout_controller.onload_weights.remote(), timeout=300)
        train_controller.weight_update(need_register=False, need_update=True)

        self._check_sglang_weights(rollout_controller, action="compare_parameters")

if __name__ == "__main__":
    unittest.main()

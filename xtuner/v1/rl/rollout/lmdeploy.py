import os
from argparse import Namespace
from typing import Any, Dict, List, Mapping

import numpy as np
import ray
import requests
from ray.util.placement_group import placement_group_table

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams

from .rollout_topology import RolloutEngine, RolloutServerProcess, RolloutTopology
from .worker import RolloutConfig, RolloutWorker


SHARED_STORE = "shared_store"
SHARED_STORE_NAMESPACE = "lmdeploy"


def run_lmdeploy_server_wrapper(lmdeploy_config_namespace: Namespace):
    """Wrapper function to run the LMDeploy API server.

    This function unpacks the configuration and starts the server. It also
    handles environment variable setup for the PyTorch backend.

    Args:
        lmdeploy_config_namespace (Namespace): A namespace object containing
            the configuration for the LMDeploy server.
    """
    # unload_module("torch")
    from lmdeploy.serve.openai.api_server import serve

    lmdeploy_serve_kwargs = vars(lmdeploy_config_namespace)
    env = lmdeploy_serve_kwargs.get("env", {})
    if lmdeploy_serve_kwargs.get("backend") == "pytorch":
        for k, v in env.items():
            os.environ[k] = str(v)
    else:  # turbomind
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    serve(**lmdeploy_serve_kwargs)


class LMDeployWorker(RolloutWorker):
    """A Ray actor that runs a text generation server using LMDeploy."""

    def __init__(
        self,
        config: RolloutConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        """Initialize the LMDeployWorker.

        Args:
            config (RolloutConfig): The configuration for the rollout worker.
            rank (int): The rank of this worker in the distributed setup.
            master_addr (str): The address of the master worker.
            master_port (int): The port of the master worker.
            world_size (int): The total number of workers.
            accelerator (str): The type of accelerator to use (e.g., "GPU").
                Defaults to "GPU".
        """
        super().__init__(config, rank, master_addr, master_port, world_size, accelerator)
        self.server_func = run_lmdeploy_server_wrapper
        self.router_func_str = "lmdeploy.serve.proxy.proxy.proxy"
        self.endpoints["health_generate"] = "health"
        self.endpoints["generate"] = "generate"
        self.endpoints["v1/chat/completions"] = "v1/chat/completions"
        self.endpoints["output_ids"] = "output_ids"
        self.endpoints["response"] = "text"
        self.endpoints["sleep"] = "sleep"
        self.endpoints["wake_up"] = "wakeup"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, trust_remote_code=True)
        self.api_keys = self.config.api_key
        self.model_name = self.config.model_name
        self.enable_return_routed_experts = self.config.enable_return_routed_experts
        self.lmdeploy_actor = None

    @classmethod
    def build_rollout_topology(
        cls,
        config: RolloutConfig,
        rank_bundle_idx_list: list[tuple[int, int]],
        rank_to_dist_init_addr: Mapping[int, str],
    ) -> RolloutTopology:
        """Build LMDeploy rollout topology with bound engine dist-init
        addresses.

        ``rank_bundle_idx_list`` stores ``(worker_rank, bundle_idx)`` pairs.

        Example with ranks [(0, 0), (1, 1), (2, 2), (3, 3)] and addrs
        {0: "addr0", 1: "addr1", 2: "addr2", 3: "addr3"}:

        +------+------------------------------------------------------------------+
        | Mode | RolloutEngine topology                                           |
        +------+------------------------------------------------------------------+
        | TP   | RolloutEngine(                                                   |
        |      |     engine_ranks=(0, 1),                                         |
        |      |     dist_init_addr="addr0",                                      |
        |      |     server_processes=(                                           |
        |      |         RolloutServerProcess(                                    |
        |      |             worker_rank=0,                                       |
        |      |             placement_group_bundle_idxs=(0, 1),                  |
        |      |             weight_update_ranks=(0, 1),                          |
        |      |         ),                                                       |
        |      |     ),                                                           |
        |      | )                                                                |
        |      | RolloutEngine(                                                   |
        |      |     engine_ranks=(2, 3),                                         |
        |      |     dist_init_addr="addr2",                                      |
        |      |     server_processes=(                                           |
        |      |         RolloutServerProcess(                                    |
        |      |             worker_rank=2,                                       |
        |      |             placement_group_bundle_idxs=(2, 3),                  |
        |      |             weight_update_ranks=(2, 3),                          |
        |      |         ),                                                       |
        |      |     ),                                                           |
        |      | )                                                                |
        +------+------------------------------------------------------------------+
        | EP   | RolloutEngine(                                                   |
        |      |     engine_ranks=(0, 1),                                         |
        |      |     dist_init_addr="addr0",                                      |
        |      |     server_processes=(                                           |
        |      |         RolloutServerProcess(                                    |
        |      |             worker_rank=0,                                       |
        |      |             placement_group_bundle_idxs=(0,),                    |
        |      |             weight_update_ranks=(0,),                            |
        |      |         ),                                                       |
        |      |         RolloutServerProcess(                                    |
        |      |             worker_rank=1,                                       |
        |      |             placement_group_bundle_idxs=(1,),                    |
        |      |             weight_update_ranks=(1,),                            |
        |      |         ),                                                       |
        |      |     ),                                                           |
        |      | )                                                                |
        |      | RolloutEngine(                                                   |
        |      |     engine_ranks=(2, 3),                                         |
        |      |     dist_init_addr="addr2",                                      |
        |      |     server_processes=(                                           |
        |      |         RolloutServerProcess(                                    |
        |      |             worker_rank=2,                                       |
        |      |             placement_group_bundle_idxs=(2,),                    |
        |      |             weight_update_ranks=(2,),                            |
        |      |         ),                                                       |
        |      |         RolloutServerProcess(                                    |
        |      |             worker_rank=3,                                       |
        |      |             placement_group_bundle_idxs=(3,),                    |
        |      |             weight_update_ranks=(3,),                            |
        |      |         ),                                                       |
        |      |     ),                                                           |
        |      | )                                                                |
        +------+------------------------------------------------------------------+
        """
        engines: list[RolloutEngine] = []
        num_workers = len(rank_bundle_idx_list)
        if config.expert_parallel_size <= 1:
            num_gpus_per_engine = config.num_gpus_per_engine
            if num_workers % num_gpus_per_engine != 0:
                raise ValueError(
                    f"num_rollout_workers={num_workers} must be divisible by "
                    f"num_gpus_per_engine={num_gpus_per_engine}."
                )
            for engine_start in range(0, num_workers, num_gpus_per_engine):
                engine_meta = rank_bundle_idx_list[engine_start : engine_start + num_gpus_per_engine]
                engine_ranks = tuple(rank for rank, _ in engine_meta)
                engine_bundle_idxs = tuple(bundle_idx for _, bundle_idx in engine_meta)
                dist_init_addr_owner_rank = engine_ranks[0]
                engines.append(
                    RolloutEngine(
                        engine_ranks=engine_ranks,
                        dist_init_addr=rank_to_dist_init_addr[dist_init_addr_owner_rank],
                        server_processes=(
                            RolloutServerProcess(
                                worker_rank=engine_ranks[0],
                                placement_group_bundle_idxs=engine_bundle_idxs,
                                weight_update_ranks=engine_ranks,
                            ),
                        ),
                    )
                )
        else:
            ep_size = config.expert_parallel_size
            if num_workers % ep_size != 0:
                raise ValueError(
                    f"num_rollout_workers={num_workers} must be divisible by expert_parallel_size={ep_size}."
                )
            for engine_start in range(0, num_workers, ep_size):
                engine_meta = rank_bundle_idx_list[engine_start : engine_start + ep_size]
                engine_ranks = tuple(rank for rank, _ in engine_meta)
                dist_init_addr_owner_rank = engine_ranks[0]
                engines.append(
                    RolloutEngine(
                        engine_ranks=engine_ranks,
                        dist_init_addr=rank_to_dist_init_addr[dist_init_addr_owner_rank],
                        server_processes=tuple(
                            RolloutServerProcess(
                                worker_rank=server_rank,
                                placement_group_bundle_idxs=(bundle_idx,),
                                weight_update_ranks=(server_rank,),
                            )
                            for server_rank, bundle_idx in engine_meta
                        ),
                    )
                )

        return RolloutTopology(engines=tuple(engines))

    def offload(self):
        """Offloads the model weights and KV cache."""
        return self._sleep(level=2)

    def onload_weights(self):
        """Onloads the model weights by waking up the model."""
        return self._wake_up(tags=["weights"])

    def onload_kvcache(self):
        """Onloads the KV cache by waking up the model."""
        return self._wake_up(tags=["kv_cache"])

    def _get_request_payload(self, rollout_state: RolloutState) -> dict:
        tools = rollout_state.tools
        tool_choice = rollout_state.tool_choice
        sample_params = rollout_state.sample_params
        message = rollout_state.message
        input_tokens = rollout_state.tokens

        optional_fields: dict[str, object] = {}
        if tools is not None:
            optional_fields["tools"] = tools
        if tool_choice is not None:
            optional_fields["tool_choice"] = tool_choice

        if sample_params.return_token_ids:
            payload = {"model": self.model_name, **optional_fields}

            if "image_data" in rollout_state.extra_fields:
                assert input_tokens is not None, "input_tokens is required when image_data is provided."
                payload["image_data"] = rollout_state.extra_fields["image_data"]

            if input_tokens is not None:
                payload["input_ids"] = input_tokens
            else:
                text_prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                prompt_token_ids = self.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
                payload["input_ids"] = prompt_token_ids
            lmdeploy_sample_params = self._transform_sample_params(
                sample_params.model_copy(
                    update={
                        "return_routed_experts": (
                            self.enable_return_routed_experts and sample_params.return_routed_experts
                        )
                    }
                )
            )
            payload.update(lmdeploy_sample_params)
        else:
            payload = {
                "model": self.model_name,
                "messages": rollout_state.message,
                **optional_fields,
            }
            lmdeploy_sample_params = {
                "temperature": sample_params.temperature,
                "top_p": sample_params.top_p,
                "n": sample_params.n,
                "stream": sample_params.stream,
                "max_tokens": sample_params.max_tokens,
                "repetition_penalty": sample_params.repetition_penalty,
                "top_k": sample_params.top_k,
                "skip_special_tokens": sample_params.skip_special_tokens,
            }
            if sample_params.stops:
                lmdeploy_sample_params["stop"] = sample_params.stops
            if sample_params.min_tokens > 0:
                lmdeploy_sample_params["min_new_tokens"] = sample_params.min_tokens
            payload.update(lmdeploy_sample_params)
        return payload

    def _sleep(self, level: int = 1):
        """Put the model into a sleep state to save resources.

        Args:
            level (int): The sleep level. Defaults to 1.

        Returns:
            str: The response text from the server.
        """
        url = f"{self.server_url}/{self.endpoints['sleep']}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_keys}"}
        data = {"level": level}
        response = requests.post(url, headers=headers, params=data, timeout=600)
        assert response.status_code == 200, response.status_code
        return response.text

    def _wake_up(self, tags: List[str] | None = None):
        """Wakes up the model from a sleep state.

        Args:
            tags (List[str] | None, optional): A list of tags to specify what
                to wake up. Defaults to None.

        Returns:
            str: The response text from the server.
        """
        url = f"{self.server_url}/{self.endpoints['wake_up']}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_keys}"}
        data = {"tags": tags}
        response = requests.post(url, headers=headers, params=data, timeout=600)
        assert response.status_code == 200, response.status_code
        return response.text

    def _request_server_terminate(self) -> bool:
        """Ask the inference server to terminate itself if it supports it."""
        terminate_url = f"{self.server_url}/terminate"
        try:
            response = requests.get(terminate_url, timeout=5.0)
        except requests.RequestException as e:
            self.logger.debug(f"Worker {self.rank} terminate request failed for {terminate_url}: {e}")
            return False

        if response.status_code == 200:
            self.logger.debug(f"Worker {self.rank} terminate request accepted by {terminate_url}.")
            return True

        self.logger.debug(
            f"Worker {self.rank} terminate request to {terminate_url} returned status {response.status_code}."
        )
        return False

    async def _decode_routed_experts(self, routed_experts: Any) -> Any:
        if isinstance(routed_experts, str):
            if self.lmdeploy_actor is None:
                self.lmdeploy_actor = ray.get_actor(SHARED_STORE, namespace=SHARED_STORE_NAMESPACE)
            assert self.lmdeploy_actor is not None, "LMDeploy actor should be available in the shared store."
            routed_experts_data = await self.lmdeploy_actor.get.remote(routed_experts)
            return ray.put(np.asarray(routed_experts_data))
        return np.asarray(routed_experts)

    def _transform_rollout_config_to_server_configs(self) -> Namespace:
        """Transform the RolloutConfig into a Namespace suitable for the
        LMDeploy server.

        This method configures the backend engine (PyTorch or Turbomind),
        sets up distributed training parameters, and prepares environment
        variables for Ray integration.

        Returns:
            Namespace: A namespace object containing the server configuration.
        """
        from lmdeploy.messages import PytorchEngineConfig, SpeculativeConfig, TurbomindEngineConfig

        accelerator_to_device_type = {
            "GPU": "cuda",
            "NPU": "ascend",
        }

        extra_config = self.config.extra_rollout_config
        lmdeploy_config_kwargs = {
            k.replace("lmdeploy_", ""): v for k, v in extra_config.items() if k.startswith("lmdeploy_")
        }

        backend = lmdeploy_config_kwargs.get("backend", "pytorch")
        tp_size = self.config.tensor_parallel_size
        dp_size = ep_size = self.config.expert_parallel_size
        # RolloutController plays the role of proxy server in LMDeploy, which balances dp requests.
        # Therefore, each server only needs to handle 1 / dp_size of the total requests
        max_batch_size = self.config.rollout_max_batch_size_per_instance // dp_size
        distributed_executor_backend = lmdeploy_config_kwargs.get("distributed_executor_backend", "ray")
        lmdeploy_config_kwargs["allow_terminate_by_client"] = True
        lmdeploy_config_kwargs["log_level"] = lmdeploy_config_kwargs.pop("log_level", "ERROR")
        lmdeploy_config_kwargs["uvicorn_log_level"] = lmdeploy_config_kwargs.pop("uvicorn_log_level", "ERROR")
        lmdeploy_config_kwargs["tm_log_level"] = lmdeploy_config_kwargs.pop("tm_log_level", "ERROR")

        speculative_config = None
        speculative_algorithm = lmdeploy_config_kwargs.pop("speculative_algorithm", None)
        speculative_num_draft_tokens = lmdeploy_config_kwargs.pop("speculative_num_draft_tokens", None)
        if speculative_algorithm is not None:
            assert speculative_num_draft_tokens is not None, (
                "lmdeploy_speculative_num_draft_tokens is required when speculative_algorithm is set"
            )
            speculative_config = SpeculativeConfig(
                method=speculative_algorithm,
                num_speculative_tokens=speculative_num_draft_tokens,
            )

        extra_engine_config: Dict[str, Any] = {}
        if backend == "pytorch" and self.config.enable_return_routed_experts:
            extra_engine_config["enable_return_routed_experts"] = True
        if backend == "pytorch" and self.config.router_n_groups:
            hf_overrides = extra_engine_config.setdefault("hf_overrides", {})
            hf_overrides.update(router_n_groups=self.config.router_n_groups)
        if backend == "pytorch" and self.config.fp32_lm_head:
            hf_overrides = extra_engine_config.setdefault("hf_overrides", {})
            hf_overrides.update(fp32_lm_head=self.config.fp32_lm_head)
        if backend == "pytorch" and self.config.max_prefill_token_num:
            extra_engine_config["max_prefill_token_num"] = self.config.max_prefill_token_num

        assert self.server_launch_spec is not None
        dp_rank = 0
        if backend == "pytorch":
            # currently only support ep > 1 and tp == 1 / ep == 1 and tp > 1
            assert ep_size == 1 or tp_size == 1
            if ep_size > 1:
                dp_rank = self.server_launch_spec.engine_rank

        backend_config = (
            PytorchEngineConfig(
                tp=tp_size,
                ep=ep_size,
                dp=dp_size,
                dp_rank=dp_rank,
                max_batch_size=max_batch_size,
                empty_init=self.config.skip_load_weights,
                distributed_executor_backend=distributed_executor_backend,
                mp_engine_backend="ray",  # force ray to pass placement group
                device_type=accelerator_to_device_type[self.accelerator],
                logprobs_mode="raw_logprobs",
                session_len=self.config.context_length,
                model_format="fp8" if self.config.enable_float8 else None,
                cache_max_entry_count=self.config.gpu_memory_utilization,
                **extra_engine_config,
            )
            if backend == "pytorch"
            else TurbomindEngineConfig(
                tp=tp_size,
                max_batch_size=self.config.rollout_max_batch_size_per_instance,
                devices=[
                    bundle_idx % self.config.gpus_per_node
                    for bundle_idx in self.server_launch_spec.placement_group_bundle_idxs
                ],
                empty_init=self.config.skip_load_weights,
                session_len=self.config.context_length,
                model_format="fp8" if self.config.enable_float8 else None,
                cache_max_entry_count=self.config.gpu_memory_utilization,
            )
        )
        if backend == "pytorch" and self.accelerator == "NPU":
            backend_config.eager_mode = True

        env = dict()
        if backend == "pytorch":
            ray_runtime_ctx = ray.get_runtime_context()
            current_pg = ray.util.get_current_placement_group()
            current_pg_name = placement_group_table(current_pg).get("name")
            env = {
                "LMDEPLOY_RAY_EXTERNAL_NS": ray_runtime_ctx.namespace,
                "LMDEPLOY_RAY_EXTERNAL_PG_NAME": current_pg_name,
                "LMDEPLOY_RAY_EXTERNAL_PG_BUNDLES": ",".join(
                    map(str, self.server_launch_spec.placement_group_bundle_idxs)
                ),
            }

            if self.accelerator == "NPU":
                env.update(
                    {
                        "ASCEND_SET_RT_VISIBLE_DEVICES_BY_RAY": "1",
                        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                        "DLINFER_RESET_MOE_UPDATE_WEIGHTS": "1",
                    }
                )

            if tp_size > 1:
                dist_addr, dist_port = self.server_launch_spec.dist_init_addr.split(":")[:2]
                env.update(
                    {
                        "LMDEPLOY_DIST_MASTER_ADDR": dist_addr,
                        "LMDEPLOY_DIST_MASTER_PORT": dist_port,
                    }
                )
            elif ep_size > 1:
                dist_addr, dist_port = self.server_launch_spec.dist_init_addr.split(":")[:2]
                if speculative_num_draft_tokens is not None:
                    deepep_max_tokens_per_rank = max_batch_size * (1 + speculative_num_draft_tokens)
                else:
                    deepep_max_tokens_per_rank = max_batch_size
                env.update(
                    {
                        "LMDEPLOY_DP_MASTER_ADDR": dist_addr,
                        "LMDEPLOY_DP_MASTER_PORT": dist_port,
                        # DEEPEP_MAX_TOKENS_PER_RANK is required by DLBlas's DeepEP
                        # token dispatcher used in lmdeploy EP mode. Without it,
                        # lmdeploy will fail during warmup.
                        # Ref: https://github.com/DeepLink-org/DLBlas/blob/aae23445/dlblas/layers/moe/token_dispatcher.py#L81
                        # Ref: https://github.com/InternLM/lmdeploy/blob/81627e3d/lmdeploy/utils.py#L375
                        "DEEPEP_MAX_TOKENS_PER_RANK": str(deepep_max_tokens_per_rank),
                    }
                )
            if "uvicorn_log_level" in lmdeploy_config_kwargs:
                env.update({"UVICORN_LOG_LEVEL": lmdeploy_config_kwargs["uvicorn_log_level"]})
            skip_warmup = os.environ.get("LMDEPLOY_SKIP_WARMUP", os.environ.get("LMD_SKIP_WARMUP"))
            if skip_warmup is not None:
                env["LMDEPLOY_SKIP_WARMUP"] = skip_warmup
        else:
            env.update({"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"})
            if "tm_log_level" in lmdeploy_config_kwargs:
                env.update({"TM_LOG_LEVEL": lmdeploy_config_kwargs["tm_log_level"]})

        if "backend" in lmdeploy_config_kwargs:
            lmdeploy_config_kwargs.pop("backend")

        return Namespace(
            model_path=self.config.model_path,
            model_name=self.model_name,
            backend=backend,
            backend_config=backend_config,
            server_name=self.host,
            server_port=self.server_port,
            api_key=self.api_keys,
            api_keys=self.api_keys,
            ray_runtime_env={"env_vars": env},
            enable_abort_handling=True,
            speculative_config=speculative_config,
            **lmdeploy_config_kwargs,
        )

    def _transform_sample_params(self, sample_params: SampleParams) -> dict:
        return sample_params.model_dump(exclude_none=True)

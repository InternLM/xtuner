import asyncio
import os
import traceback
from argparse import Namespace
from typing import Any, Dict, List, Union

import ray
import requests
import torch
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils import FlexibleArgumentParser

from xtuner.v1.data_proto.rl_data import RLRolloutResponseItem, RolloutState
from xtuner.v1.ray.config import RolloutConfig
from xtuner.v1.ray.rollout.worker import RolloutWorker
from xtuner.v1.utils.device import get_device, get_torch_device_module


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """VLLM provides `StatelessProcessGroup` to create a process group without
    considering the global process group in torch.distributed.

    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)

    if DEVICE == "npu":
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

        pynccl = PyHcclCommunicator(pg, device=device)
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WorkerWrap:
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="hccl", use_ray=False
    ):
        """Init torch process group for model weights update."""
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_with_ray = use_ray
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                self.device,
            )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight_npu_ipc(self, data):
        import base64
        import json
        from multiprocessing.reduction import ForkingPickler

        if isinstance(data, str):
            data = json.loads(data)

        def _construct(item):
            func, args = item
            args = list(args)
            args[6] = DEVICE_MODULE.current_device()
            return func(*args)

        serialized_data = data["serialized_named_tensors"]
        if isinstance(serialized_data, list):
            serialized_data = serialized_data[self.global_rank]
        weights = ForkingPickler.loads(base64.b64decode(serialized_data))
        weights = [(k, _construct(v)) for k, v in weights]
        DEVICE_MODULE.synchronize()
        self.model_runner.model.load_weights(weights=weights)
        del weights
        DEVICE_MODULE.synchronize()
        DEVICE_MODULE.empty_cache()

    def get_worker_pids(self):
        current_pid = os.getpid()
        return current_pid


@ray.remote
class VllmServerWrapper:
    def __init__(self, server_namespace: Namespace):
        cli_env_setup()
        server_args = getattr(server_namespace, "args", Namespace())
        env = getattr(server_namespace, "env", {})
        for k, v in env.items():
            os.environ[k] = str(v)
        try:
            asyncio.run(run_server(server_args))
        except Exception as e:
            error_msg = f"Failed to start server in VllmServerWrapper: {type(e).__name__}: {str(e)}"
            stack_trace = traceback.format_exc()
            print(error_msg)
            print(stack_trace)
            raise  # Re-raise the exception to prevent silent failure

    def actor_health(self):
        return "healthy"


# Add a dummy task.
def run_lmdeploy_server_wrapper(server_namespace: Namespace):
    return ray.get(VllmServerWrapper.remote(server_namespace).actor_health.remote())  # type: ignore


class vLLMWorker(RolloutWorker):
    def __init__(
        self,
        config: RolloutConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        super().__init__(config, rank, master_addr, master_port, world_size, accelerator)
        self.router_func = ""
        self.server_func = run_lmdeploy_server_wrapper
        self.endpoints["health_generate"] = "health"
        self.endpoints["v1/chat/completions"] = "v1/chat/completions"
        self.endpoints["generate"] = "v1/chat/completions"
        self.endpoints["sleep"] = "sleep"
        self.endpoints["wake_up"] = "wake_up"
        self.endpoints["models"] = "models"
        self.endpoints["update_weights"] = "update_weights"
        # self.endpoints['abort_request'] = "abort_request"
        self.api_keys = self.config.api_key
        self.model_name = self.config.model_name
        self.enable_return_routed_experts = self.config.enable_return_routed_experts
        self.dp_size = self.config.data_parallel_size
        assert self.dp_size > 0, "data_parallel_size must be > 0"
        assert self.config.tensor_parallel_size % self.dp_size == 0, (
            f"tensor_parallel_size ({self.config.tensor_parallel_size}) must be divisible by data_parallel_size ({self.dp_size})"
        )
        self.tp_size = self.config.tensor_parallel_size // self.dp_size

    async def _create_request(
        self,
        url: str,
        prompt: Union[str, List[Dict[str, Any]]] | None,
        input_ids: List[int] | None,
        tools: List,  # reserved for agent tool use
        tool_choice: str,  # reserved for agent tool use
        sample_params: dict,
        extra_params: dict,
        extra_info: dict,
    ):
        stream = extra_params["stream"]
        headers = {"Content-Type": "application/json"}

        if "image_data" in extra_info:
            if not isinstance(prompt, list):
                raise ValueError("image_data requires prompt to be a list of messages")

            image_index = 0
            for message in prompt:
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "user":
                    new_content = []
                    for content_part in message.get("content", []):
                        if not isinstance(content_part, dict):
                            new_content.append(content_part)
                            continue
                        if content_part.get("type") == "image_url":
                            content_part["image_url"]["url"] = f"file://{extra_info['image_data'][image_index]}"
                            content_part["image_url"].pop("image_wh", None)
                            image_index += 1
                            new_content.append(content_part)
                        else:
                            new_content.append(content_part)

                    message["content"] = new_content

            assert image_index == len(extra_info["image_data"]), (
                f"Expected {len(extra_info['image_data'])} images, but processed {image_index}."
            )

        payload = {
            "model": self.config.model_path,
            "messages": prompt,
            "stream": stream,
        }
        if "train_prompt_ids" in extra_info:
            payload["input_ids"] = extra_info["train_prompt_ids"]

        vllm_sample_params = self._transform_sample_params(sample_params, extra_params)
        payload.update(vllm_sample_params)

        return await self._safe_post_request(url, headers, payload)

    def _transform_sample_params(self, sample_params: Dict, extra_params: Dict = {}):
        import copy

        vllm_sample_params = copy.deepcopy(sample_params)
        if extra_params:
            vllm_sample_params.update(extra_params)
        if "stops" in vllm_sample_params:
            vllm_sample_params["stop"] = vllm_sample_params.pop("stops")
        if "no_stop_trim" in vllm_sample_params:
            vllm_sample_params["include_stop_str_in_output"] = vllm_sample_params.pop("no_stop_trim")
        if "top_logprobs" in vllm_sample_params and "return_logprob" in vllm_sample_params:
            vllm_sample_params["logprobs"] = vllm_sample_params.pop("return_logprob")
        return vllm_sample_params

    def get_logprobs(self, input_ids, sampling_params):
        pass

    def generate(self, input_ids, sampling_params):
        pass

    def sleep(self, level=1):
        url = f"{self.server_url}/{self.endpoints['sleep']}"
        headers = {"Content-Type": "application/json"}
        params = {}
        params["level"] = level
        response = requests.post(url, headers=headers, params=params)
        assert response.status_code == 200, response.status_code
        return response.text

    def wake_up(self, tags: List[str] | None = None):
        url = f"{self.server_url}/{self.endpoints['wake_up']}"
        headers = {"Content-Type": "application/json"}
        params = {}
        if tags is not None:
            params["tags"] = tags
        response = requests.post(url, headers=headers, params=params)
        assert response.status_code == 200, response.status_code
        return response.text

    def pause_generation(self):
        pass

    def continue_generation(self):
        pass

    def onload_weights(self):
        """Onloads the model weights by waking up the model."""
        return self.wake_up(tags=["weights"])

    def onload_kvcache(self):
        """Onloads the KV cache by waking up the model."""
        return self.wake_up(tags=["kv_cache"])

    def offload(self):
        """Offloads the model weights and KV cache."""
        return self.sleep(level=2)

    def reset_prefix_cache(self, tags: List[str] | None = None):
        raise NotImplementedError("The 'reset_prefix_cache' API is not yet implemented in the vLLM server.")

    def _decode_routed_experts(self, routed_experts: Any):
        raise NotImplementedError

    def _transform_rollout_config_to_server_configs(self) -> Namespace:
        # use vllm FlexibleArgumentParser to parse the config
        # and return the args as the default server config
        # vllm server_args: vllm/vllm/engine/arg_utils.py
        parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        args_ = parser.parse_args([])

        args = {}
        args["host"] = self.host
        args["port"] = self.server_port
        args["api_key"] = self.api_keys
        args["api_keys"] = self.api_keys
        args["model"] = self.config.model_path
        args["log_level"] = "info"
        args["data_parallel_size"] = self.dp_size
        args["tensor_parallel_size"] = self.tp_size
        args["enable_expert_parallel"] = False

        args["distributed_executor_backend"] = "ray"
        args["max_model_len"] = self.config.context_length
        args["enforce_eager"] = False
        args["enable_sleep_mode"] = True
        args["worker_extension_cls"] = "xtuner.v1.ray.rollout.vllm.WorkerWrap"
        args["trust_remote_code"] = True
        args["enable_prefix_caching"] = False
        args["allowed_local_media_path"] = "/"
        args["mm_processor_cache_gb"] = 0
        args["max_num_batched_tokens"] = 4096
        args["max_num_seqs"] = self.config.rollout_max_batch_size_per_instance // self.dp_size
        args["block_size"] = 128
        args["gpu_memory_utilization"] = self.config.gpu_memory_utilization
        args["compilation_config"] = {
            "cudagraph_capture_sizes": [16, 12, 8, 4, 2, 1],
            "cudagraph_mode": "FULL_DECODE_ONLY",
        }
        args["additional_config"] = {"enable_cpu_binding": True}
        args["limit_mm_per_prompt"] = {"image": 10, "video": 0}
        args["enable_log_requests"] = False
        args["uvicorn_log_level"] = "error"
        env = {
            "VLLM_VERSION": "0.11.0",
            "TASK_QUEUE_ENABLE": "0",
            "CPU_AFFINITY_CONF": "2",
            "VLLM_USE_V1": "1",
            "VLLM_RAY_PER_WORKER_GPUS": "0.1",
            "VLLM_RAY_BUNDLE_INDICES": ",".join(map(str, self.engine_bundle_idxs)),
            "VLLM_MONITOR": "1",
            "VLLM_ACCU_MONITOR": "0",
            "CUSTOM_SCHEDULE_KV_LIMIT": "0.9",
            "HCCL_BUFFSIZE": "512",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0",
            "SHM_BARRIER": "true",
            "USE_TOKEN_IN": "1",
            "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HCCL_CONNECT_TIMEOUT": "7200",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "INTERNS1_VIT_USE_TP": "1",
            "VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ASCEND_ENABLE_NZ": "0",
        }

        # Apply extra_rollout_config overrides for vLLM parameters (prefix: "vllm_")
        extra_cfg = getattr(self.config, "extra_rollout_config", None) or {}
        for key, value in extra_cfg.items():
            if key.startswith("vllm_"):
                real_key = key[5:]
                args[real_key] = value

        args_.__dict__.update(args)
        validate_parsed_serve_args(args_)

        return Namespace(
            args=args_,
            env=env,
            api_key=self.api_keys,
            api_keys=self.api_keys,
            ray_runtime_env={"env_vars": env},
        )

    async def _handle_stream_response(self, uid, sample_params, extra_params, response) -> RLRolloutResponseItem:
        raise NotImplementedError

    async def _handle_non_stream_response(
        self, root_id, action_id, sample_params, extra_params, response, input_extra_info
    ) -> RLRolloutResponseItem:
        uid = action_id
        last_token_ids = []
        last_logprobs = []

        response = response.json()["choices"][0]
        if "logprobs" in response:
            last_token_ids = response["token_ids"]
            last_logprobs = [item["logprob"] for item in response["logprobs"]["content"]]
            assert len(last_token_ids) == len(last_logprobs)
            assert len(last_token_ids) <= sample_params["max_tokens"], (
                f"Generation length exceeds limit: generated {len(last_token_ids)}, limit {sample_params['max_tokens']}"
            )
        last_trajectory = response["message"]["content"]
        finish_reason = response["finish_reason"]
        if finish_reason == "abort" and self.receive_abort_request.is_set() is False:
            self.receive_abort_request.set()
            self.logger.info(f"Setting receive_abort_request to True for rank {self.rank}")

        if finish_reason != "abort" and (len(last_token_ids) == 0 or len(last_logprobs) == 0):
            self.logger.error(f"Invalid rollout response for request {uid}: {response}")
            return RLRolloutResponseItem(state=RolloutState.SKIPPED)

        rollout_response = RLRolloutResponseItem(
            response=last_trajectory,
            response_ids=last_token_ids if len(last_token_ids) > 0 else None,
            num_return_tokens=len(last_token_ids) if len(last_token_ids) > 0 else None,
            finish_reason=finish_reason,
            logprobs=last_logprobs,
            state=RolloutState.ABORTED if finish_reason == "abort" else RolloutState.COMPLETED,
        )

        return rollout_response

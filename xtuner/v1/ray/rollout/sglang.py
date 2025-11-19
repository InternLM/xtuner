import os
from typing import Any, Dict, List, Union

import requests
from urllib3.exceptions import NewConnectionError

from transformers import AutoTokenizer
from xtuner.v1.ray.config import RolloutConfig

from .worker import RolloutWorker


class SGLangWorker(RolloutWorker):
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
        from sglang.srt.entrypoints.http_server import launch_server

        self.server_func = launch_server
        self.endpoints["health_generate"] = "health_generate"
        self.endpoints["generate"] = "generate"
        self.endpoints["v1/chat/completions"] = "v1/chat/completions"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        self.api_keys = self.config.api_key
        self.model_name = self.config.model_name
        self.enable_return_routed_experts = self.config.enable_return_routed_experts

    async def _create_request(
        self,
        url: str,
        prompt: Union[str, List[Dict[str, Any]]] | None,
        input_ids: List[int] | None,
        tools: List,
        tool_choice: str,
        sample_params: dict,
        extra_params: dict,
        extra_info: dict,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys}",  # 如果需要鉴权
        }
        payload = {"model": self.model_name}
        sglang_sample_params = self._transform_sample_params(sample_params)
        sglang_extra_params = self._transform_extra_params(extra_params)
        if self.enable_return_routed_experts:
            sglang_extra_params["return_routed_experts"] = True

        payload.update(sglang_extra_params)

        if "return_token_ids" in extra_params and extra_params["return_token_ids"]:
            # 多模态场景下，由于 input_ids 处理比较复杂，现在不支持 prompt 输入，必须要有 input_ids
            if "image_data" in extra_info:
                assert input_ids is not None, "input_ids is required when image_data is provided."
            if input_ids is not None:
                payload["input_ids"] = input_ids
                if "image_data" in extra_info:
                    payload["image_data"] = extra_info["image_data"]
            else:
                text_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                prompt_token_ids = self.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
                payload["input_ids"] = prompt_token_ids
            payload["sampling_params"] = sglang_sample_params
        else:
            payload["messages"] = prompt
            payload.update(sglang_sample_params)
            # note: chat completions 接口需要传入 max_tokens 和 min_tokens 参数
            payload["max_tokens"] = sglang_sample_params["max_new_tokens"]
            payload["min_tokens"] = sglang_sample_params["min_new_tokens"]
            payload.pop("max_new_tokens", None)
            payload.pop("min_new_tokens", None)

        return await self._safe_post_request(url, headers, payload)

    def _make_request(self, endpoint: str, payload=None):
        # TODO: 支持 tp
        url = f"{self.server_url}/{endpoint}"
        response = requests.post(url, json=payload or {})
        response.raise_for_status()
        return response.json()

    def flush_cache(self):
        """Flush the cache of the server."""
        # TODO: 支持 tp
        # flush cache will not return status_code 200 when there are pending requests
        while True:
            try:
                response = requests.get(f"{self.server_url}/flush_cache", timeout=60)
                if response.status_code == 200:
                    break
            except requests.exceptions.Timeout:
                print("Timeout occurred while flushing cache. Exiting loop.")
                break
            except NewConnectionError as e:
                raise e
            except Exception as e:
                print(f"Error flushing cache: {e}")
                continue

    def get_logprobs(self, input_ids, sampling_params):
        return self._make_request(
            "generate",
            {"input_ids": input_ids, "sampling_params": sampling_params, "stream": False, "return_logprob": True},
        )

    def offload(self):
        """Offloads the model weights and KV cache."""
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def onload_weights(self):
        """Onloads the model weights by waking up the model."""
        return self._make_request("resume_memory_occupation", {"tags": ["weights"]})

    def onload_kvcache(self):
        return self._make_request("resume_memory_occupation", {"tags": ["kv_cache"]})

    def pause_generation(self):
        return self._make_request("pause_generation")

    def continue_generation(self):
        return self._make_request("continue_generation")

    def reset_prefix_cache(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def _transform_rollout_config_to_server_configs(self):
        # remove the CUDA_VISIBLE_DEVICES set by ray and use base_gpu_id
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from sglang.srt.server_args import ServerArgs

        extra_config = self.config.extra_rollout_config or dict()
        sglang_config_kwargs = {
            k.replace("sglang_", ""): v for k, v in extra_config.items() if k.startswith("sglang_")
        }
        grammar_backend = sglang_config_kwargs.get(
            "grammar_backend", None
        )  # for intern-s1 series models, have to set the grammar_backend to "none"
        log_level = sglang_config_kwargs.get("log_level", "critical")
        log_level_http = sglang_config_kwargs.get("log_level_http", "critical")
        enable_deterministic_inference = sglang_config_kwargs.get("enable_deterministic_inference", False)

        sglang_server_args = ServerArgs(model_path=self.config.model_path, trust_remote_code=True)
        num_gpus_per_engine = (
            self.config.expert_parallel_size
            if self.config.expert_parallel_size > 1
            else self.config.tensor_parallel_size
        )
        sglang_server_args.host = self.host
        sglang_server_args.port = self.server_port
        sglang_server_args.nccl_port = self.nccl_port
        sglang_server_args.dist_init_addr = self.dist_init_addr
        sglang_server_args.base_gpu_id = self.rank % self.config.gpus_per_node
        sglang_server_args.gpu_id_step = 1
        sglang_server_args.nnodes = max(1, num_gpus_per_engine // self.config.gpus_per_node)
        sglang_server_args.skip_server_warmup = True

        sglang_server_args.mem_fraction_static = self.config.gpu_memory_utilization
        # note: 非共卡模式下无需设置,共卡模式下需要offload必须设置，否则显存释放不了
        sglang_server_args.enable_memory_saver = True

        if self.enable_return_routed_experts:
            sglang_server_args.enable_return_routed_experts = True

        sglang_server_args.max_running_requests = self.config.rollout_max_batch_size_per_instance
        sglang_server_args.log_level = log_level
        sglang_server_args.log_level_http = log_level_http
        sglang_server_args.enable_deterministic_inference = enable_deterministic_inference

        if self.config.expert_parallel_size > 1:
            sglang_server_args.tp_size = num_gpus_per_engine
            sglang_server_args.ep_size = num_gpus_per_engine
        else:
            sglang_server_args.tp_size = self.config.tensor_parallel_size
            sglang_server_args.ep_size = self.config.expert_parallel_size

        if grammar_backend is not None:
            sglang_server_args.grammar_backend = grammar_backend

        if self.config.context_length is not None:
            sglang_server_args.context_length = self.config.context_length

        if sglang_server_args.nnodes > 1:
            sglang_server_args.node_rank = self.rank // self.config.gpus_per_node
        else:
            sglang_server_args.node_rank = 0

        return sglang_server_args

    def _transform_sample_params(self, sample_params: Dict):
        if sample_params["top_p"] > 0:
            sample_params["top_k"] = -1  # top_p优先级更高
        sglang_sample_params = {
            "n": sample_params["n"],
            "top_k": sample_params["top_k"],
            "top_p": sample_params["top_p"],
            "temperature": sample_params["temperature"],
            "repetition_penalty": sample_params["repetition_penalty"],
            "presence_penalty": sample_params["presence_penalty"],
            "frequency_penalty": sample_params["frequency_penalty"],
            "max_new_tokens": sample_params["max_tokens"],
            "min_new_tokens": sample_params["min_tokens"],
            "stop": sample_params["stops"],
            "stop_token_ids": sample_params["stop_token_ids"],
            "skip_special_tokens": sample_params["skip_special_tokens"],
        }
        return sglang_sample_params

    def _transform_extra_params(self, extra_params: Dict):
        sglang_extra_params = {
            "stream": extra_params["stream"],
            "return_logprob": extra_params["return_logprob"],
            "include_stop_str_in_output": extra_params["include_stop_str_in_output"],
            "no_stop_trim": extra_params.get("no_stop_trim", False),
            "spaces_between_special_tokens": extra_params.get("spaces_between_special_tokens", False),
        }
        return sglang_extra_params

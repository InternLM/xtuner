import base64
import os
from typing import Any, Dict, List, Union

import numpy as np
import requests
import torch
from urllib3.exceptions import NewConnectionError

from transformers import AutoConfig, AutoTokenizer
from xtuner.v1.ray.config import RolloutConfig
from xtuner.v1.utils import XTUNER_DETERMINISTIC

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
        self.endpoints["health_generate"] = "health"
        self.endpoints["generate"] = "generate"
        self.endpoints["v1/chat/completions"] = "v1/chat/completions"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        self.model_config = AutoConfig.from_pretrained(self.config.model_path, trust_remote_code=True)
        text_config = getattr(self.model_config, "text_config", self.model_config)
        self.model_type = getattr(text_config, "model_type", getattr(self.model_config, "model_type", None))
        self.routed_experts_num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
        self.routed_experts_num_experts_per_tok = getattr(text_config, "num_experts_per_tok", None)
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
        if self.enable_return_routed_experts and not extra_params.get("disable_routed_experts", False):
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

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.server_url}/{self.endpoints['health_generate']}", timeout=5.0)
            return response.status_code == 200
        except requests.RequestException as e:
            self.logger.error(f"Health check failed for server {self.server_url}: {e}")
            return False

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

    def _decode_routed_experts(self, routed_experts: Any, meta_info: Dict[str, Any]):
        if not isinstance(routed_experts, str):
            return super()._decode_routed_experts(routed_experts, meta_info)

        prompt_tokens = meta_info.get("prompt_tokens", 0)
        completion_tokens = meta_info.get("completion_tokens", 0)
        num_tokens = prompt_tokens + completion_tokens - 1
        assert num_tokens > 0, (
            f"Unexpected routed_experts token count: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"
        )
        assert self.routed_experts_num_hidden_layers is not None, (
            "num_hidden_layers is required to decode routed_experts"
        )
        assert self.routed_experts_num_experts_per_tok is not None, (
            "num_experts_per_tok is required to decode routed_experts"
        )

        routed_experts_flat = np.frombuffer(base64.b64decode(routed_experts), dtype=np.int32)
        expected_size = num_tokens * self.routed_experts_num_hidden_layers * self.routed_experts_num_experts_per_tok
        assert routed_experts_flat.size == expected_size, (
            f"Unexpected routed_experts size {routed_experts_flat.size}, expected {expected_size}. "
            f"num_tokens={num_tokens}, num_hidden_layers={self.routed_experts_num_hidden_layers}, "
            f"num_experts_per_tok={self.routed_experts_num_experts_per_tok}"
        )
        routed_experts_array = routed_experts_flat.reshape(
            num_tokens,
            self.routed_experts_num_hidden_layers,
            self.routed_experts_num_experts_per_tok,
        )
        return torch.from_numpy(routed_experts_array.copy())

    def _transform_rollout_config_to_server_configs(self):
        # remove the CUDA_VISIBLE_DEVICES set by ray and use base_gpu_id
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from sglang.srt.server_args import ServerArgs

        extra_config = self.config.extra_rollout_config or dict()
        sglang_config_kwargs = {
            k.replace("sglang_", ""): v for k, v in extra_config.items() if k.startswith("sglang_")
        }
        log_level = sglang_config_kwargs.get("log_level", "error")
        log_level_http = sglang_config_kwargs.get("log_level_http", "error")
        num_gpus_per_engine = (
            self.config.expert_parallel_size
            if self.config.expert_parallel_size > 1
            else self.config.tensor_parallel_size
        )
        tp_size = num_gpus_per_engine if self.config.expert_parallel_size > 1 else self.config.tensor_parallel_size
        ep_size = num_gpus_per_engine if self.config.expert_parallel_size > 1 else self.config.expert_parallel_size
        nnodes = max(1, num_gpus_per_engine // self.config.gpus_per_node)
        node_rank = self.rank // self.config.gpus_per_node if nnodes > 1 else 0
        init_kwargs = dict(
            model_path=self.config.model_path,
            trust_remote_code=True,
            host=self.host,
            port=self.server_port,
            nccl_port=self.nccl_port,
            dist_init_addr=self.dist_init_addr,
            base_gpu_id=self.rank % self.config.gpus_per_node,
            gpu_id_step=1,
            nnodes=nnodes,
            node_rank=node_rank,
            skip_server_warmup=True,
            mem_fraction_static=self.config.gpu_memory_utilization,
            enable_memory_saver=True,
            max_running_requests=self.config.rollout_max_batch_size_per_instance,
            log_level=log_level,
            log_level_http=log_level_http,
            tp_size=tp_size,
            ep_size=ep_size,
        )
        if self.enable_return_routed_experts:
            init_kwargs["enable_return_routed_experts"] = True
        if XTUNER_DETERMINISTIC:
            init_kwargs["enable_deterministic_inference"] = True
            init_kwargs["rl_on_policy_target"] = "fsdp"
            init_kwargs["attention_backend"] = "fa3"
            init_kwargs["random_seed"] = self.config.random_seed
            # SGLang's deterministic mode does not currently force-disable every
            # performance-oriented runtime path. For long MoE rollouts we still
            # observed rare trajectory divergence, so explicitly turn off the
            # scheduler/cache/graph features that can perturb execution order.
            init_kwargs["disable_radix_cache"] = True
            init_kwargs["disable_overlap_schedule"] = True
            init_kwargs["disable_cuda_graph"] = True

        # Forward supported sglang_* extra configs to ServerArgs directly.
        server_arg_fields = getattr(ServerArgs, "__dataclass_fields__", {})
        for key, value in sglang_config_kwargs.items():
            if key in server_arg_fields:
                init_kwargs[key] = value
            else:
                self.logger.warning(f"Ignore unknown SGLang server arg: {key}={value!r}")

        # Qwen3-MoE in sglang 0.5.9 can hit native rotary + fused KV buffer incompatibility
        # during server startup unless fused qk_norm_rope is enabled.
        if self.model_type == "qwen3_moe" and "enable_fused_qk_norm_rope" not in sglang_config_kwargs:
            init_kwargs["enable_fused_qk_norm_rope"] = True
            self.logger.info("Auto enable SGLang enable_fused_qk_norm_rope for qwen3_moe.")

        if self.config.context_length is not None:
            init_kwargs["context_length"] = self.config.context_length

        sglang_server_args = ServerArgs(**init_kwargs)

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
        if XTUNER_DETERMINISTIC:
            sglang_sample_params["sampling_seed"] = sample_params["sampling_seed"]
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

import os
from typing import Any, Dict, List, Union

import ray
import requests
from urllib3.exceptions import NewConnectionError

from transformers import AutoTokenizer
from xtuner.v1.ray.config import RolloutConfig

from .worker import RolloutWorker


@ray.remote
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

    async def _create_request(
        self,
        url: str,
        prompt: Union[str, List[Dict[str, Any]]] | None,
        input_ids: List[int] | None,
        tools: List,
        tool_choice: str,
        sample_params: dict,
        extra_params: dict,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys}",  # 如果需要鉴权
        }
        stream = extra_params["stream"]
        # note: 此处默认使用tokne_id的话，则不使用流式；异步rollout+token_id进出后续修复
        payload = {"model": self.model_name}
        if stream:
            self.logger.warning("Using stream mode for SGLangWorker is not supported yet.")
            raise NotImplementedError("Streaming mode is not supported for SGLangWorker.")
        else:
            if "return_token_ids" in extra_params and extra_params["return_token_ids"]:
                if input_ids is not None:
                    payload["input_ids"] = input_ids
                else:
                    text_prompt = self.tokenizer.apply_chat_template(
                        prompt, tokenize=False, add_generation_prompt=True
                    )
                    prompt_token_ids = self.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
                    payload["input_ids"] = prompt_token_ids
            else:
                payload["messages"] = prompt

        sglang_sample_params = self._transform_sample_params(sample_params)
        payload["sampling_params"] = sglang_sample_params
        sglang_extra_params = self._transform_extra_params(extra_params)
        payload.update(sglang_extra_params)
        # self.logger.info(f"Request payload: {payload}")
        req = self.client.build_request(
            "POST",
            url,
            headers=headers,
            json=payload,
        )
        r = await self.client.send(req, stream=stream)
        r.raise_for_status()
        return r

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
                response = requests.get(f"{self.server_url}/flush_cache")
                if response.status_code == 200:
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
        pass

    def continue_generation(self):
        pass

    def shutdown(self):
        pass

    def reset_prefix_cache(self):
        pass

    def _transform_rollout_config_to_server_configs(self):
        # remove the CUDA_VISIBLE_DEVICES set by ray and use base_gpu_id
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from sglang.srt.server_args import ServerArgs

        sglang_server_args = ServerArgs(model_path=self.config.model_path)
        sglang_server_args.host = self.host
        sglang_server_args.port = self.server_port
        sglang_server_args.nccl_port = self.nccl_port
        sglang_server_args.dist_init_addr = self.dist_init_addr
        sglang_server_args.base_gpu_id = self.rank % self.config.gpus_per_node
        sglang_server_args.gpu_id_step = 1
        sglang_server_args.nnodes = max(1, self.config.tensor_parallel_size // self.config.gpus_per_node)
        sglang_server_args.skip_server_warmup = True
        sglang_server_args.tp_size = self.config.tensor_parallel_size
        sglang_server_args.mem_fraction_static = self.config.gpu_memory_utilization
        # note: 非共卡模式下无需设置,共卡模式下需要offload必须设置，否则显存释放不了
        sglang_server_args.enable_memory_saver = True

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

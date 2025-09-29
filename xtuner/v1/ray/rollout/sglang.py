import os
from typing import Any, Dict, List, Union
import requests
import ray
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from urllib3.exceptions import NewConnectionError

from xtuner.v1.ray.config import RolloutConfig
from transformers import AutoTokenizer
from .worker import RolloutWorker
import os


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
        self.server_func = launch_server
        self.endpoints["health_generate"] = "health_generate"
        if os.environ.get("ID_INPUT_OUTPUT", '0') == '1':
            self.endpoints["generate"] = "generate"
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        else:
            self.endpoints["generate"] = "v1/chat/completions"
        self.api_keys = self.config.api_key
        self.model_name = self.config.model_name

    async def _create_request(
            self,
            url: str,
            prompt: Union[str, List[Dict[str, Any]]],
            tools: List,
            tool_choice: str,
            sample_params: dict,
            extra_params: dict,
    ):
        sample_params['top_k'] = -1  # TODO： 暂时写死

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys}",  # 如果需要鉴权
        }
        if os.environ.get("ID_INPUT_OUTPUT", '0') == '1':
            payload = {"model": self.model_name, "stream": True, "return_logprob": True}
            text_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            prompt_token_ids = self.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
            payload["input_ids"] = prompt_token_ids

            new_sample_params = {"max_new_tokens": sample_params['max_tokens'],
                                 "temperature": sample_params['temperature'],
                                 "top_p": sample_params['top_p'],
                                 "top_k": sample_params['top_k'],
                                 "no_stop_trim": True,
                                 "skip_special_tokens": False,
                                 "spaces_between_special_tokens":False,
                                 }
            payload['sampling_params'] = new_sample_params
        else:
            payload = {
                "model": self.model_name,
                "messages": prompt,
                "stream": True,
            }
            payload.update(sample_params)
            payload.update(extra_params)
        req = self.client.build_request(
            "POST",
            url,
            headers=headers,
            json=payload,
        )
        r = await self.client.send(req, stream=True)
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
        # return self._make_request("pause_generation")

    def continue_generation(self):
        pass
        # return self._make_request("continue_generation")

    def shutdown(self):
        pass

    def reset_prefix_cache(self):
        pass

    def _transform_rollout_config_to_server_configs(self):
        # remove the CUDA_VISIBLE_DEVICES set by ray and use base_gpu_id
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
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
        sglang_server_args.mem_fraction_static = 0.7  # 关键
        sglang_server_args.enable_memory_saver = True  # 关键，否则显存释放不了

        if sglang_server_args.nnodes > 1:
            sglang_server_args.node_rank = self.rank // self.config.gpus_per_node
        else:
            sglang_server_args.node_rank = 0
        return sglang_server_args

    def _transform_rollout_config_to_router_configs(self, infer_config):
        pass

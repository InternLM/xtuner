from argparse import Namespace
from typing import Dict, List

import ray
import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser

from xtuner.v1.ray.config import RolloutConfig

from .worker import RolloutWorker


def run_vllm_server_wrapper(server_args):
    uvloop.run(run_server(server_args))


@ray.remote
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
        self.server_func = run_vllm_server_wrapper
        self.router_func = ""
        self.endpoints["health_generate"] = "health"
        self.endpoints["generate"] = "v1/chat/completions"
        self.endpoints["output_ids"] = "output_ids"
        self.endpoints["response"] = "text"
        self.endpoints["sleep"] = "sleep"
        self.endpoints["wake_up"] = "wakeup"
        self.api_keys = self.config.api_key

    async def _create_request(
        self,
        url: str,
        prompt: List[Dict[str, str]],
        tools: List,  # reserved for agent tool use
        tool_choice: str,  # reserved for agent tool use
        sample_params: dict,
        extra_params: dict,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys}",  # 如果需要鉴权
        }
        payload = {
            "model": self.config.model_name,
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

    def get_logprobs(self, input_ids, sampling_params):
        pass

    def generate(self, input_ids, sampling_params):
        pass

    def sleep(self, level=1, tags: List[str] | None = None):
        import requests

        url = f"{self.server_url}/{self.endpoints['sleep']}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_keys}"}
        data = {"tags": tags}
        response = requests.post(url, headers=headers, json=data)
        assert response.status_code == 200, response.status_code
        return response.json()

    def wake_up(self, tags: List[str] | None = None):
        import requests

        url = f"{self.server_url}/{self.endpoints['wake_up']}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_keys}"}
        data = {"tags": tags}
        response = requests.post(url, headers=headers, json=data)
        assert response.status_code == 200, response.status_code
        return response.json()

    def pause_generation(self):
        pass

    def continue_generation(self):
        pass

    def update_weights(self, ipc_handles):
        # todo
        pass

    def reset_prefix_cache(self):
        # todo
        pass

    def _transform_rollout_config_to_server_configs(self) -> Namespace:
        # use vllm FlexibleArgumentParser to parse the config
        # and return the args as the default server config
        parser = FlexibleArgumentParser()
        parser = make_arg_parser(parser)
        args = parser.parse_args([])
        args.__dict__.update(vars(self.config))

        args.host = self.host
        args.port = self.server_port
        args.model = self.config.model_path
        args.disable_log_requests = True
        args.disable_log_stats = True
        args.tensor_parallel_size = self.config.tensor_parallel_size
        if args.expert_parallel_size > 1:
            args.tensor_parallel_size = self.config.expert_parallel_size
            args.enable_expert_parallel = True

        return args

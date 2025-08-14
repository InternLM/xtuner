import os
from argparse import Namespace
from copy import deepcopy
from typing import List

import ray
import torch
from ray.util.placement_group import placement_group_table

from xtuner.v1.ray.config import RolloutConfig

from .worker import RolloutWorker


def run_lmdeploy_server_wrapper(lmdeploy_config_namespace: Namespace):
    # unload_module("torch")
    from lmdeploy.serve.openai.api_server import serve

    lmdeploy_serve_kwargs = vars(lmdeploy_config_namespace)
    env = lmdeploy_serve_kwargs.get("env", {})
    if lmdeploy_serve_kwargs.get("backend") == "pytorch":
        for k, v in env.items():
            os.environ[k] = str(v)
    serve(**lmdeploy_serve_kwargs)


@ray.remote
class LMDeployWorker(RolloutWorker):
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
        self.server_func = run_lmdeploy_server_wrapper
        self.router_func_str = "lmdeploy.serve.proxy.proxy.proxy"
        self.endpoints["health_generate"] = "health"
        self.endpoints["generate"] = "v1/chat/completions"
        self.endpoints["output_ids"] = "output_ids"
        self.endpoints["response"] = "text"
        self.endpoints["sleep"] = "sleep"
        self.endpoints["wake_up"] = "wakeup"
        self.api_keys = self.config.api_key
        self.model_name = self.config.model_name

    async def _create_request(
        self,
        url: str,
        uid: str,
        prompt: str,
        sample_params: dict = dict(),
        extra_params: dict = dict(),
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys}",  # 如果需要鉴权
        }
        payload = {
            "request_id": uid,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "model": self.model_name,
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
        # 直接调用engine.generate方法
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

        self.paused = False
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
        from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig

        backend = (self.config.extra_rollout_config or dict()).get("lmdeploy_backend", "pytorch")
        tp_size = self.config.tensor_parallel_size
        dp_size = ep_size = self.config.expert_parallel_size
        backend_config = (
            PytorchEngineConfig(
                tp=tp_size,
                ep=ep_size,
                dp=dp_size,
                empty_init=self.config.skip_load_weights,
            )
            if backend == "pytorch"
            else TurbomindEngineConfig(
                tp=tp_size,
                dp=dp_size,
                empty_init=self.config.skip_load_weights,
            )
        )

        env = dict()
        if backend == "pytorch":
            ray_runtime_ctx = ray.get_runtime_context()
            current_pg = ray.util.get_current_placement_group()
            current_pg_name = placement_group_table(current_pg).get("name")
            env = {
                "LMDEPLOY_RAY_EXTERNAL_NS": ray_runtime_ctx.namespace,
                "LMDEPLOY_RAY_EXTERNAL_PG_NAME": current_pg_name,
                "LMDEPLOY_RAY_EXTERNAL_PG_BUNDLES": ",".join(map(str, self.engine_bundle_idxs)),
            }
            local_rank = self.rank % torch.accelerator.device_count()
            if tp_size > 1:
                dist_addr, dist_port = self.dist_init_addr.split(":")[:2]
                devices = [str(r) for r in range(local_rank, local_rank + tp_size)]
                env.update(
                    {
                        self.device_visible_env_name: ",".join(devices),
                        "LMDEPLOY_DIST_MASTER_ADDR": dist_addr,
                        "LMDEPLOY_DIST_MASTER_PORT": dist_port,
                    }
                )
            elif dp_size > 1:
                dist_addr, dist_port = self.dist_init_addr.split(":")[:2]
                env.update(
                    {
                        self.device_visible_env_name: str(local_rank),
                        "LMDEPLOY_DP_MASTER_ADDR": dist_addr,
                        "LMDEPLOY_DP_MASTER_PORT": dist_port,
                    }
                )

        extra_kwargs = deepcopy(self.config.extra_rollout_config) or dict()
        if "lmdeploy_backend" in extra_kwargs:
            extra_kwargs.pop("lmdeploy_backend")

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
            **extra_kwargs,
        )

    def _transform_rollout_config_to_router_configs(self):
        pass

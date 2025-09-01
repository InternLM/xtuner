import os
from argparse import Namespace
from typing import Dict, List

import ray
import requests
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
            "model": self.model_name,
            "messages": prompt,
            "tools": tools,
            "tool_choice": tool_choice,
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

    def sleep(self, level: int = 1):
        url = f"{self.server_url}/{self.endpoints['sleep']}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_keys}"}
        data = {"level": level}
        response = requests.post(url, headers=headers, params=data)
        assert response.status_code == 200, response.status_code
        return response.text

    def offload_weights(self):
        return self.sleep(level=1)

    def offload_weights_and_kvcache(self):
        return self.sleep(level=2)

    def wake_up(self, tags: List[str] | None = None):
        url = f"{self.server_url}/{self.endpoints['wake_up']}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_keys}"}
        data = {"tags": tags}
        response = requests.post(url, headers=headers, params=data)
        assert response.status_code == 200, response.status_code
        return response.text

    def onload_weights(self):
        return self.wake_up(tags=["weights"])

    def onload_kvcache(self):
        return self.wake_up(tags=["kv_cache"])

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

        extra_config = self.config.extra_rollout_config or dict()
        lmdeploy_config_kwargs = {
            k.replace("lmdeploy_", ""): v for k, v in extra_config.items() if k.startswith("lmdeploy_")
        }

        backend = lmdeploy_config_kwargs.get("backend", "pytorch")
        tp_size = self.config.tensor_parallel_size
        dp_size = ep_size = self.config.expert_parallel_size
        distributed_executor_backend = lmdeploy_config_kwargs.get("distributed_executor_backend", "ray")
        backend_config = (
            PytorchEngineConfig(
                tp=tp_size,
                ep=ep_size,
                dp=dp_size,
                empty_init=self.config.skip_load_weights,
                distributed_executor_backend=distributed_executor_backend,
                mp_engine_backend="ray",  # force ray to pass placement group
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
            if tp_size > 1:
                dist_addr, dist_port = self.dist_init_addr.split(":")[:2]
                env.update(
                    {
                        "LMDEPLOY_DIST_MASTER_ADDR": dist_addr,
                        "LMDEPLOY_DIST_MASTER_PORT": dist_port,
                    }
                )
            elif dp_size > 1:
                dist_addr, dist_port = self.dist_init_addr.split(":")[:2]
                env.update(
                    {
                        "LMDEPLOY_DP_MASTER_ADDR": dist_addr,
                        "LMDEPLOY_DP_MASTER_PORT": dist_port,
                    }
                )
            if "uvicorn_log_level" in lmdeploy_config_kwargs:
                env["UVICORN_LOG_LEVEL"] = lmdeploy_config_kwargs["uvicorn_log_level"]

        if "backend" in lmdeploy_config_kwargs:
            lmdeploy_config_kwargs.pop("backend")

        lmdeploy_config_kwargs["log_level"] = "CRITICAL"  # disable logging
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
            **lmdeploy_config_kwargs,
        )

    def _transform_rollout_config_to_router_configs(self):
        pass

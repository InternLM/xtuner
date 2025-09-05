import os
from argparse import Namespace
from typing import Any, Dict, List, Union

import ray
import requests
from ray.util.placement_group import placement_group_table

from xtuner.v1.ray.config import RolloutConfig

from .worker import RolloutWorker


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


@ray.remote
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
        prompt: Union[str, List[Dict[str, Any]]],
        tools: List,  # reserved for agent tool use
        tool_choice: str,  # reserved for agent tool use
        sample_params: dict,
        extra_params: dict,
    ):
        """Create and send a streaming generation request to the server.

        Args:
            url (str): The URL of the generation endpoint.
            prompt (List[Dict[str, str]]): The input prompt for generation,
                formatted as a list of messages.
            tools (List): A list of tools the model can call.
            tool_choice (str): The tool choice strategy.
            sample_params (dict): Parameters for sampling. Defaults to {}.
            extra_params (dict): Extra parameters for the request.
                Defaults to {}.

        Returns:
            An httpx.Response object for streaming the response.
        """
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
        """This method will be implemented for the LMDeploy worker in the
        future."""
        pass

    def generate(self, input_ids, sampling_params):
        """This method will be implemented for the LMDeploy worker in the
        future."""
        pass

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
        response = requests.post(url, headers=headers, params=data)
        assert response.status_code == 200, response.status_code
        return response.text

    def offload(self):
        """Offloads the model weights and KV cache."""
        return self._sleep(level=2)

    def wake_up(self, tags: List[str] | None = None):
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
        response = requests.post(url, headers=headers, params=data)
        assert response.status_code == 200, response.status_code
        return response.text

    def onload_weights(self):
        """Onloads the model weights by waking up the model."""
        return self.wake_up(tags=["weights"])

    def onload_kvcache(self):
        """Onloads the KV cache by waking up the model."""
        return self.wake_up(tags=["kv_cache"])

    def pause_generation(self):
        """It will implemented for LMDeploy worker in the future."""
        pass

    def continue_generation(self):
        """It will implemented for LMDeploy worker in the future."""
        pass

    def reset_prefix_cache(self):
        """It will implemented for LMDeploy worker in the future."""
        pass

    def _transform_rollout_config_to_server_configs(self) -> Namespace:
        """Transform the RolloutConfig into a Namespace suitable for the
        LMDeploy server.

        This method configures the backend engine (PyTorch or Turbomind),
        sets up distributed training parameters, and prepares environment
        variables for Ray integration.

        Returns:
            Namespace: A namespace object containing the server configuration.
        """
        from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig

        accelerator_to_device_type = {
            "GPU": "cuda",
            "NPU": "ascend",
        }

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
                device_type=accelerator_to_device_type[self.accelerator],
            )
            if backend == "pytorch"
            else TurbomindEngineConfig(
                tp=tp_size,
                devices=[bundle_idxs % self.config.gpus_per_node for bundle_idxs in self.engine_bundle_idxs],
                empty_init=self.config.skip_load_weights,
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
                "LMDEPLOY_RAY_EXTERNAL_PG_BUNDLES": ",".join(map(str, self.engine_bundle_idxs)),
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

        # todo(@duanyanhui): remove qwen3_chat_template when lmdeploy support tokenizer
        # apply_chat_template in PR: https://github.com/InternLM/lmdeploy/pull/3845
        from lmdeploy.model import ChatTemplateConfig

        qwen3_chat_template = ChatTemplateConfig(
            model_name="qwen3",
            system="",
            eosys="",
            user="<|im_start|>user\n",
            eoh="<|im_end|>\n",
            assistant="<|im_start|>assistant\n",
            eoa="<|im_end|>",
            separator="\n",
            capability="chat",
            stop_words=["<|im_end|>"],
        )
        assert "qwen3" in self.config.model_path.lower(), (
            "qwen3_chat_template only for qwen3 model, you should provide ChatTemplateConfig for other model"
        )

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
            chat_template_config=qwen3_chat_template,
            **lmdeploy_config_kwargs,
        )

    def _transform_rollout_config_to_router_configs(self):
        """This method will be implemented for the LMDeploy worker in the
        future."""
        pass

import copy
import os
from argparse import Namespace
from typing import Any, Dict, List, Union

import ray
import requests
from ray.util.placement_group import placement_group_table

from transformers import AutoTokenizer
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
            "tools": tools if len(tools) > 0 else None,
            "tool_choice": tool_choice if tool_choice else None,
        }
        if "return_token_ids" in extra_params and extra_params["return_token_ids"]:
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
        else:
            payload["messages"] = prompt

        if self.enable_return_routed_experts:
            extra_params["return_routed_experts"] = True

        lmdeploy_sample_params = self._transform_sample_params(sample_params, extra_params)
        payload.update(lmdeploy_sample_params)

        return await self._safe_post_request(url, headers, payload)

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
        lmdeploy_config_kwargs["log_level"] = lmdeploy_config_kwargs.pop("log_level", "WARNING")
        lmdeploy_config_kwargs["uvicorn_log_level"] = lmdeploy_config_kwargs.pop("uvicorn_log_level", "CRITICAL")
        lmdeploy_config_kwargs["tm_log_level"] = lmdeploy_config_kwargs.pop("tm_log_level", "CRITICAL")

        extra_engine_config = {}
        if backend == "pytorch" and self.config.enable_return_routed_experts:
            extra_engine_config["enable_return_routed_experts"] = True

        backend_config = (
            PytorchEngineConfig(
                tp=tp_size,
                ep=ep_size,
                dp=dp_size,
                max_batch_size=self.config.rollout_max_batch_size_per_instance,
                empty_init=self.config.skip_load_weights,
                distributed_executor_backend=distributed_executor_backend,
                mp_engine_backend="ray",  # force ray to pass placement group
                device_type=accelerator_to_device_type[self.accelerator],
                logprobs_mode="raw_logprobs",
                session_len=self.config.context_length,
                **extra_engine_config,
            )
            if backend == "pytorch"
            else TurbomindEngineConfig(
                tp=tp_size,
                max_batch_size=self.config.rollout_max_batch_size_per_instance,
                devices=[bundle_idxs % self.config.gpus_per_node for bundle_idxs in self.engine_bundle_idxs],
                empty_init=self.config.skip_load_weights,
                session_len=self.config.context_length,
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
                env.update({"UVICORN_LOG_LEVEL": lmdeploy_config_kwargs["uvicorn_log_level"]})
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
            **lmdeploy_config_kwargs,
        )

    def _transform_sample_params(self, sample_params: Dict, extra_params: Dict = {}):
        lmdeploy_sample_params = copy.deepcopy(sample_params)
        if extra_params:
            lmdeploy_sample_params.update(extra_params)
        return lmdeploy_sample_params

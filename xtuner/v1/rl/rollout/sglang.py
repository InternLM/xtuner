import base64
import os
from typing import Any, Dict, List, Mapping, Union

import numpy as np
import ray
import requests
from urllib3.exceptions import NewConnectionError

from transformers import AutoConfig, AutoTokenizer
from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.utils import XTUNER_DETERMINISTIC

from .rollout_topology import RolloutEngine, RolloutServerProcess, RolloutTopology
from .worker import RolloutConfig, RolloutWorker


SHARED_STORE = "shared_store"
SHARED_STORE_NAMESPACE = "sglang"


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
        self.endpoints["health"] = "health"
        self.endpoints["health_generate"] = "health_generate"
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
        self.sglang_actor = None

    @classmethod
    def build_rollout_topology(
        cls,
        config: RolloutConfig,
        rank_bundle_idx_list: list[tuple[int, int]],
        rank_to_dist_init_addr: Mapping[int, str],
    ) -> RolloutTopology:
        """Build SGLang rollout topology with bound engine dist-init addresses.

        The normal SGLang topology starts one server process for each logical
        engine. Cross-node engines are the special case: SGLang starts one
        server process per node, but only node 0 accepts rollout requests and
        owns the weight-update endpoint.

        Example with ``expert_parallel_size=2`` on one node:
            RolloutTopology(
                engines=(
                    RolloutEngine(
                        engine_ranks=(0, 1),
                        dist_init_addr="addr0",
                        server_processes=(
                            RolloutServerProcess(
                                worker_rank=0,
                                placement_group_bundle_idxs=(0, 1),
                                accepts_rollout_requests=True,
                                weight_update_ranks=(0, 1),
                                node_rank=0,
                                nnodes=1,
                            ),
                        ),
                    ),
                    RolloutEngine(
                        engine_ranks=(2, 3),
                        dist_init_addr="addr2",
                        server_processes=(
                            RolloutServerProcess(
                                worker_rank=2,
                                placement_group_bundle_idxs=(2, 3),
                                accepts_rollout_requests=True,
                                weight_update_ranks=(2, 3),
                                node_rank=0,
                                nnodes=1,
                            ),
                        ),
                    ),
                ),
            )

        Example with ``expert_parallel_size=16`` across two 8-GPU nodes:
            RolloutTopology(
                engines=(
                    RolloutEngine(
                        engine_ranks=(0, 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 10, 11, 12, 13, 14, 15),
                        dist_init_addr="addr0",
                        server_processes=(
                            RolloutServerProcess(
                                worker_rank=0,
                                placement_group_bundle_idxs=(0, 1, 2, 3, 4, 5, 6, 7),
                                accepts_rollout_requests=True,
                                weight_update_ranks=(0, 1, 2, 3, 4, 5, 6, 7,
                                                     8, 9, 10, 11, 12, 13, 14, 15),
                                node_rank=0,
                                nnodes=2,
                            ),
                            RolloutServerProcess(
                                worker_rank=8,
                                placement_group_bundle_idxs=(8, 9, 10, 11, 12, 13, 14, 15),
                                accepts_rollout_requests=False,
                                weight_update_ranks=(),
                                node_rank=1,
                                nnodes=2,
                            ),
                        ),
                    ),
                ),
            )
        """
        num_workers = len(rank_bundle_idx_list)
        num_gpus_per_engine = config.num_gpus_per_engine
        if num_workers % num_gpus_per_engine != 0:
            raise ValueError(
                f"num_rollout_workers={num_workers} must be divisible by num_gpus_per_engine={num_gpus_per_engine}."
            )
        if num_gpus_per_engine > config.gpus_per_node and num_gpus_per_engine % config.gpus_per_node != 0:
            raise ValueError(
                "SGLang cross-node rollout requires num_gpus_per_engine to be divisible by gpus_per_node."
            )

        nnodes = max(1, num_gpus_per_engine // config.gpus_per_node)
        engines = []
        for engine_start in range(0, num_workers, num_gpus_per_engine):
            engine_meta = rank_bundle_idx_list[engine_start : engine_start + num_gpus_per_engine]
            engine_ranks = tuple(rank for rank, _ in engine_meta)
            engine_bundle_idxs = tuple(bundle_idx for _, bundle_idx in engine_meta)
            # SGLang cross-node launch starts one server process per node. The
            # first rank of each node owns that node's bundles, while only node
            # 0 is exposed as the rollout request entrypoint.
            server_ranks = engine_ranks[:: config.gpus_per_node]
            dist_init_addr_owner_rank = server_ranks[0]
            server_processes = []
            for node_rank, server_rank in enumerate(server_ranks):
                node_bundle_start = node_rank * config.gpus_per_node
                node_bundle_end = node_bundle_start + config.gpus_per_node
                server_processes.append(
                    RolloutServerProcess(
                        worker_rank=server_rank,
                        placement_group_bundle_idxs=engine_bundle_idxs[node_bundle_start:node_bundle_end],
                        accepts_rollout_requests=node_rank == 0,
                        weight_update_ranks=engine_ranks if node_rank == 0 else (),
                        node_rank=node_rank,
                        nnodes=nnodes,
                    )
                )
            engines.append(
                RolloutEngine(
                    engine_ranks=engine_ranks,
                    dist_init_addr=rank_to_dist_init_addr[dist_init_addr_owner_rank],
                    server_processes=tuple(server_processes),
                )
            )
        return RolloutTopology(
            engines=tuple(engines),
        )

    def _get_request_payload(self, rollout_state: RolloutState) -> dict:
        sample_params = rollout_state.sample_params
        payload: dict[str, Any] = {"model": self.model_name}

        if rollout_state.tools is not None:
            payload["tools"] = rollout_state.tools
        if rollout_state.tool_choice is not None:
            payload["tool_choice"] = rollout_state.tool_choice

        sglang_sample_params = self._transform_sample_params(sample_params.model_dump())
        sglang_extra_params = self._transform_extra_params(sample_params.model_dump())
        payload.update(sglang_extra_params)

        if (
            self.enable_return_routed_experts
            and sample_params.return_routed_experts
            and not rollout_state.extra_fields.get("disable_routed_experts", False)
        ):
            payload["return_routed_experts"] = True

        if sample_params.return_token_ids:
            if "image_data" in rollout_state.extra_fields:
                assert rollout_state.tokens is not None, "input_ids is required when image_data is provided."
                payload["image_data"] = rollout_state.extra_fields["image_data"]

            if rollout_state.tokens is not None:
                payload["input_ids"] = rollout_state.tokens
            else:
                text_prompt = self.tokenizer.apply_chat_template(
                    rollout_state.message, tokenize=False, add_generation_prompt=True
                )
                payload["input_ids"] = self.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]

            payload["sampling_params"] = sglang_sample_params
            return payload

        payload["messages"] = rollout_state.message
        payload.update(sglang_sample_params)
        # The chat-completions API uses OpenAI-style names.
        payload["max_tokens"] = sglang_sample_params["max_new_tokens"]
        payload["min_tokens"] = sglang_sample_params["min_new_tokens"]
        payload.pop("max_new_tokens", None)
        payload.pop("min_new_tokens", None)
        return payload

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
        if (
            self.enable_return_routed_experts
            and sample_params.get("return_routed_experts", False)
            and not extra_params.get("disable_routed_experts", False)
        ):
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
            response = requests.get(
                f"{self.server_url}/{self.endpoints['health']}",
                timeout=self.config.health_check_timeout_seconds,
            )
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

    async def pause_generation(self):
        # SGLang PauseGeneration支持三种模式（https://github.com/sgl-project/sglang/blob/8d27ce7371da617a671f62e78dde66d64b7ad6cb/python/sglang/srt/managers/io_struct.py#L1353）：
        # abort    = 丢弃 waiting 和 running 请求，
        # retract  = 保留waiting请求和running请求（保留已生成 token），释放 KV，恢复时重算 KV 后继续
        # in_place = 保留waiting请求和running请求（保留已生成 token）、已生成 token、KV，恢复时直接继续
        self.receive_abort_request.set()
        await self._send_abort_request()
        return self._make_request("pause_generation", {"mode": "abort"})

    def continue_generation(self):
        # 恢复生成时必须清掉上一轮 abort 标志，否则新请求会在发送前被本地直接标成 ABORTED。
        self.receive_abort_request.clear()
        return self._make_request("continue_generation")

    def reset_prefix_cache(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    async def _decode_routed_experts(self, routed_experts: Any):
        if isinstance(routed_experts, str):
            try:
                if self.sglang_actor is None:
                    self.sglang_actor = ray.get_actor(SHARED_STORE, namespace=SHARED_STORE_NAMESPACE)
                assert self.sglang_actor is not None, "SGLang actor should be available in the shared store."
                routed_experts_data = await self.sglang_actor.get.remote(routed_experts)
                if hasattr(routed_experts_data, "detach"):
                    routed_experts_data = routed_experts_data.detach().cpu().numpy()
                return ray.put(np.asarray(routed_experts_data))
            except Exception:
                self.logger.debug("Failed to resolve SGLang routed_experts from Ray shared store; trying base64.")
            routed_experts_flat = np.frombuffer(base64.b64decode(routed_experts), dtype=np.int32)
            routed_experts_array = routed_experts_flat.reshape(
                -1,
                self.routed_experts_num_hidden_layers,
                self.routed_experts_num_experts_per_tok,
            )
            return routed_experts_array.copy()
        return np.asarray(routed_experts)

    def _transform_rollout_config_to_server_configs(self):
        # remove the CUDA_VISIBLE_DEVICES set by ray and use base_gpu_id
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from sglang.srt.server_args import ServerArgs

        extra_config = self.config.extra_rollout_config
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
        assert self.server_launch_spec is not None
        assigned_gpu_id = int(ray.get_runtime_context().get_accelerator_ids()[self.accelerator][0])

        # SGLang 0.5.10 默认启用的 Piecewise CUDA Graph 在启动 warmup compile 阶段会报错。sglang的文档提到这个功能还是实验功能，可能还不太稳定(https://sgl-project-sglang-93.mintlify.app/optimization/cuda-graph#bug-report)。暂时先通过disable_piecewise_cuda_graph=True关掉改功能
        init_kwargs = dict(
            model_path=self.config.model_path,
            trust_remote_code=True,
            host=self.host,
            port=self.server_port,
            nccl_port=self.nccl_port,
            dist_init_addr=self.server_launch_spec.dist_init_addr,
            base_gpu_id=assigned_gpu_id,
            gpu_id_step=1,
            nnodes=self.server_launch_spec.nnodes,
            node_rank=self.server_launch_spec.node_rank,
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

        if self.config.skip_load_weights and "load_format" not in sglang_config_kwargs:
            init_kwargs["load_format"] = "dummy"

        sglang_server_args = ServerArgs(**init_kwargs)

        return sglang_server_args

    def _request_server_terminate(self) -> bool:
        self.logger.warning("SGLang server does not support terminate request, will directly kill the process.")
        return True

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
        sampling_seed = sample_params.get("sampling_seed")
        if sampling_seed is None and XTUNER_DETERMINISTIC:
            sampling_seed = self.config.random_seed
        if sampling_seed is not None:
            sglang_sample_params["sampling_seed"] = sampling_seed
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

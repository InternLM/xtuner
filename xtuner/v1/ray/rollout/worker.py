import asyncio
import copy
import json
import multiprocessing
import os
import time
import traceback
import uuid
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import httpx
import numpy as np
import ray
import requests  # type: ignore[import-untyped]
import torch
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import RLRolloutResponseItem, RolloutState
from xtuner.v1.ray import find_master_addr_and_port
from xtuner.v1.ray.base import AutoAcceleratorWorkers, SingleAcceleratorWorker
from xtuner.v1.ray.config import RolloutConfig
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType, HttpRequestResult


def get_eos_token(model_path: str) -> int | List[int]:
    from xtuner.v1.utils.logger import get_logger

    logger = get_logger()
    generation_config_path = os.path.join(model_path, "generation_config.json")
    if not os.path.exists(generation_config_path):
        logger.warning(
            f"Config {generation_config_path} does not exist and thus cannot get eos_token. You must provide eos_token manually."
        )
        return []
    with open(generation_config_path) as f:
        generation_config = json.load(f)
    eos_token_id = generation_config.get("eos_token_id")
    return eos_token_id


class RolloutWorker(SingleAcceleratorWorker):
    """Base class for a rollout worker that runs an inference server.

    This class manages the lifecycle of a distributed inference server, including initialization, launching, and
    handling generation requests. It is designed to be subclassed for specific inference backends like LMDeploy, vLLM
    or SGLang.
    """

    def __init__(
        self,
        config: RolloutConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        """Initialize the RolloutWorker.

        Args:
            config (RolloutConfig): The configuration for the rollout.
            rank (int): The rank of this worker in the distributed setup.
            master_addr (str): The address of the Ray master node.
            master_port (int): The port of the Ray master node.
            world_size (int): The total number of workers.
            accelerator (str): The type of accelerator to use.
                Defaults to "GPU".
        """
        self.config = config
        self.rank = rank
        self.master_addr = master_addr  # ray master
        self.master_port = master_port
        self.world_size = world_size
        self.accelerator = accelerator
        self.server_func: Callable
        self.endpoints: dict[str, str] = dict()
        self.engine_rank_mesh_array: list[list[int]]
        # http_concurrency is calculated based on the max batch size per engine and the total number of engines
        assert config.rollout_max_batch_size_per_instance, (
            "rollout_max_batch_size_per_instance must be set in RolloutConfig"
        )
        http_concurrency = config.rollout_max_batch_size_per_instance * config.allow_over_concurrency_ratio
        limits = httpx.Limits(max_connections=http_concurrency, max_keepalive_connections=100)
        self.client = httpx.AsyncClient(limits=limits, timeout=self.config.rollout_timeout)
        self.paused = False
        self.server_task = None
        self.engine_bundle_idxs: list[int] = []
        self.server_process: Optional[multiprocessing.Process] = None
        self.logger = get_logger(log_dir=config.worker_log_dir, tag="RolloutWorker")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, trust_remote_code=True)
        self.check_flag = True  # only print once
        self.enable_return_routed_experts = self.config.enable_return_routed_experts
        if self.rank == 0:
            self.logger.info(f"RolloutConfig:\n{self.config.model_dump_json(indent=2)}")
        eos_token = get_eos_token(self.config.model_path)
        self.logger.info(f"Using eos_token: {eos_token} for model at {self.config.model_path}")
        self.eos_token: List[int] = [eos_token] if isinstance(eos_token, int) else eos_token
        self.receive_abort_request = asyncio.Event()
        self.abort_timeout = 5.0

    def init_dist_port(self):
        """Initialize distributed communication ports.

        This method acquires three free ports for the distributed setup:
        one for the inference server, one for NCCL, and one for Ray's
        distributed communication.

        Returns:
            str: The distributed initialization address (host:port).
        """
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=ray.util.get_current_placement_group(),
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=self.engine_bundle_idxs[0],
        )

        local_rank = int(ray.get_runtime_context().get_accelerator_ids()[self.accelerator][0])
        interval = 1024
        start_port = self.config.dist_port_base + local_rank * interval
        end_port = start_port + interval
        self.host, self.ports = ray.get(
            find_master_addr_and_port.options(scheduling_strategy=scheduling_strategy).remote(
                nums=3,
                start_port=start_port,
                end_port=end_port,
            )
        )

        self.dist_port = self.ports[0]
        self.server_port = self.ports[1]
        self.nccl_port = self.ports[2]
        self.dist_init_addr = f"{self.host}:{self.dist_port}"
        self.server_url = f"http://{self.host}:{self.server_port}"
        return self.dist_init_addr

    def init(self, dist_init_addr: str = ""):
        """Initialize the worker and launch the server.

        Args:
            dist_init_addr (str): The distributed initialization address.
                If not provided, the one generated by `init_dist_port` is used.

        Returns:
            Tuple[int, str]: A tuple containing the worker's rank and its
                server URL.
        """
        self.dist_init_addr = dist_init_addr if dist_init_addr else self.dist_init_addr
        self.receive_abort_request.clear()
        self.launch_server()
        return (self.rank, self.server_url)

    def set_engine_rank_mesh_array(self, engine_rank_mesh_array: list[list[int]]):
        self.engine_rank_mesh_array = engine_rank_mesh_array

    def set_engine_bundle_idxs(self, engine_bundle_idxs: list[int]):
        """Set the bundle indices for the inference engine.

        This is used by some backends (like LMDeploy with Ray executor) to
        know which bundles in the placement group belong to this engine.

        Args:
            engine_bundle_idxs (list[int]): A list of bundle indices.
        """
        self.engine_bundle_idxs = engine_bundle_idxs

    def launch_server(self):
        """Launch the inference server as a separate process or Ray task.

        It waits for the server to become healthy before returning.

        Raises:
            TimeoutError: If the server fails to start within the specified
                timeout.
            Exception: If the server task terminates unexpectedly.
        """
        server_configs = self._transform_rollout_config_to_server_configs()
        timeout = 3600.0  # Increased timeout to 5 minutes for downloading large models
        start_time = time.perf_counter()
        last_log_time = start_time
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {server_configs.api_key}",
        }

        self.logger.info(f"Launch server task on server_url: {self.server_url}")

        # note(@duanyanhui): launch server as multiprocessing for sglang temporarily
        if self.config.launch_server_method == "multiprocessing":
            mp_ctx = multiprocessing.get_context("spawn")
            process = mp_ctx.Process(target=self.server_func, args=(server_configs,))
            process.start()
            self.server_process = process
            time.sleep(60)  # Wait for the server to start
            with requests.Session() as session:
                while time.perf_counter() - start_time < timeout:
                    try:
                        response = session.get(
                            f"{self.server_url}/{self.endpoints['health_generate']}", headers=headers
                        )
                        if response.status_code == 200:
                            return
                    except requests.RequestException as e:
                        self.logger.error(
                            f"can't connect to server url {self.server_url}/{self.endpoints['health_generate']} because {e}"
                        )

                    current_time = time.perf_counter()
                    if current_time - last_log_time >= 15:
                        self.logger.info(
                            f"Waiting for server to start, Elapsed time: {current_time - start_time:.2f}s"
                        )
                        last_log_time = current_time

                    time.sleep(5)
            process.terminate()
            raise TimeoutError("Server failed to start within the timeout period.")
        else:
            # launch the server as ray task
            # so that the lmdeploy backend could get externl pg
            current_pg = ray.util.get_current_placement_group()
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=current_pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=self.engine_bundle_idxs[0],
            )
            assert ray.is_initialized()
            ray_kwargs = (
                {"runtime_env": server_configs.ray_runtime_env} if hasattr(server_configs, "ray_runtime_env") else {}
            )
            self.server_task = (
                ray.remote(self.server_func)
                .options(
                    scheduling_strategy=scheduling_strategy,
                    **AutoAcceleratorWorkers.get_pg_options(current_pg),
                    **ray_kwargs,
                )
                .remote(server_configs)
            )

            with requests.Session() as session:
                while time.perf_counter() - start_time < timeout:
                    try:
                        response = session.get(
                            f"{self.server_url}/{self.endpoints['health_generate']}", headers=headers
                        )
                        if response.status_code == 200:
                            return
                    except requests.RequestException:
                        pass

                    try:
                        ray.get(self.server_task, timeout=0.1)
                        raise Exception("Server task terminated unexpectedly.")
                    except ray.exceptions.GetTimeoutError:
                        pass
                    except Exception as e:
                        raise e

                    current_time = time.perf_counter()
                    if current_time - last_log_time >= 15:
                        self.logger.info(
                            f"Waiting for server to start... Elapsed time: {current_time - start_time:.2f}s"
                        )
                        last_log_time = current_time

            ray.cancel(self.server_task)
            raise TimeoutError("Server failed to start within the timeout period.")

    def _adapt_input_to_openai_spec(self, prompts, tools, tool_choice):
        openai_prompts = []
        openai_tools = []
        # transform claude spec to openai spec
        # 1. transform system prompt: concat provided system_prompt to input prompt
        system_prompt = self.config.system_prompt
        if system_prompt:
            system_prompt_json = {"role": "system", "content": f"{system_prompt}"}
            prompts.insert(0, system_prompt_json)
        # 2. transform multi-modal usage
        for prompt in prompts:
            content = prompt["content"]
            openai_content = []
            for item in content:
                if item["type"] == "image":
                    if item["source"]["type"] == "base64":
                        openai_url = f"data:{item['source']['media_type']};base64,{item['source']['data']}"
                    if item["source"]["type"] == "url":
                        openai_url = item["source"]["url"]
                    new_prompt = {"type": "image_url", "image_url": {"url": openai_url}}
                    openai_content.append(new_prompt)
                elif item["type"] == "text":
                    openai_content.append(item)
            new_prompt = copy.deepcopy(prompt)
            new_prompt["content"] = openai_content
            openai_prompts.append(new_prompt)
        # 3. transform tool use
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            openai_tools.append(openai_tool)
        return openai_prompts, openai_tools

    def _check_infer_engine_version(self, return_token_ids: bool):
        # TODO(@duanyanhui): remove this check when all backends support return_token_ids
        if self.check_flag:
            if os.environ.get("XTUNER_USE_VLLM", "0") == "1":
                if return_token_ids:
                    self.logger.error(
                        "VLLM backend does not support return_token_ids or generate with input_ids as input in Xtuner now"
                    )
            elif os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "1":
                import lmdeploy

                lmdeploy_version = lmdeploy.__version__
                if return_token_ids and Version(lmdeploy_version) < Version("0.10.2"):
                    self.logger.error(
                        f"You should use lmdeploy >= v0.10.2 to support return_token_ids, but current version is {lmdeploy_version}"
                    )
            self.check_flag = False

    async def _safe_post_request(self, url, headers, payload) -> HttpRequestResult:
        try:
            if self.receive_abort_request.is_set():
                self.logger.debug(f"Request to {url} was cancelled before sending due to an abort signal.")
                return HttpRequestResult(error_type=HttpRequestErrorType.REQUEST_ABORTED, url=url, payload=payload)
            req = self.client.build_request(
                "POST",
                url,
                headers=headers,
                json=payload,
            )
            r = await self.client.send(req)
            r.raise_for_status()
            return HttpRequestResult(response=r)

        except Exception as e:
            error_type = HttpRequestErrorType.from_exception(e)
            result = HttpRequestResult(error_type=error_type, exception=e, url=url, payload=payload)
            return result

    async def rollout_task(
        self,
        prompts: Union[str, List[Dict[str, Any]]] | None,
        input_ids: List[int] | None,
        tools: List,
        tool_choice: str,
        sample_params: dict,
        extra_params: dict,
        format: str,
        extra_info: dict,
    ) -> RLRolloutResponseItem:
        uid = extra_info.get("action_id", str(uuid.uuid4()))
        action_id = extra_info.get("action_id", str(uuid.uuid4()))
        root_id = extra_info.get("action_id", str(uuid.uuid4()))
        response = None
        cur_retry_times = 0

        if format == "openai":
            openai_prompts, openai_tools = prompts, tools
        else:
            openai_prompts, openai_tools = self._adapt_input_to_openai_spec(prompts, tools, tool_choice)

        if "return_token_ids" in extra_params and extra_params["return_token_ids"]:
            endpoint_url = f"{self.server_url}/{self.endpoints['generate']}"
        else:
            endpoint_url = f"{self.server_url}/{self.endpoints['v1/chat/completions']}"

        while True:
            # 当拼接后的response_ids长度已经达到了max_tokens时，则不需要发送数据，直接返回
            if extra_info.get("partial_rollout_input_ids", None) is not None:
                if sample_params["max_tokens"] == 0:
                    self.logger.info(
                        f"Request {uid} reached max context length {self.config.context_length}, no need to rollout more."
                    )
                    return RLRolloutResponseItem(
                        response=None,
                        response_ids=None,
                        logprobs=None,
                        num_return_tokens=0,
                        finish_reason="length",
                        state=RolloutState.COMPLETED,
                    )
                if extra_info["partial_rollout_input_ids"][-1] in self.eos_token:
                    self.logger.info(
                        f"Request {uid} already ends with eos token {extra_info['partial_rollout_input_ids'][-1]}, no need to rollout more"
                    )
                    return RLRolloutResponseItem(
                        response=None,
                        response_ids=None,
                        logprobs=None,
                        num_return_tokens=0,
                        finish_reason="stop",
                        state=RolloutState.COMPLETED,
                    )

            http_result = await self._create_request(
                endpoint_url,
                openai_prompts,
                input_ids,
                openai_tools,
                tool_choice,
                sample_params=sample_params,
                extra_params=extra_params,
                extra_info=extra_info,
            )
            # Case 1: Request was successful
            if http_result.response is not None:  # 推理完成：completed状态：finish_reason为abort/stop/length, 退出
                response = await self._handle_non_stream_response(
                    root_id, action_id, sample_params, extra_params, http_result.response, extra_info
                )
                if response.state == RolloutState.SKIPPED:
                    # retry
                    cur_retry_times += 1
                    if cur_retry_times < self.config.max_retry_per_sample:
                        self.logger.warning(
                            f"Invalid rollout response for request {uid}, retrying {cur_retry_times}/{self.config.max_retry_per_sample}."
                        )
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        return RLRolloutResponseItem(state=RolloutState.SKIPPED)
                return response

            # Case2: Return aborted error if receive abort signal
            if http_result.error_type == HttpRequestErrorType.REQUEST_ABORTED:
                return RLRolloutResponseItem(finish_reason="abort", state=RolloutState.ABORTED)

            # Case 3: A fatal, non-retryable error occurred
            elif http_result.is_unknown_error:
                raise RuntimeError(
                    f"Unexpected error during rollout request {uid} to {http_result.url}: {http_result.exception}"
                )

            # Case 4: A retryable error occurred, and we still have retries left
            elif http_result.is_retryable and cur_retry_times < self.config.max_retry_per_sample:
                cur_retry_times += 1
                self.logger.warning(
                    f"Retrying rollout request {uid} to {http_result.url} due to {http_result.error_type} with {http_result.error_msg}. "
                    f"Retry {cur_retry_times}/{self.config.max_retry_per_sample}."
                )
                await asyncio.sleep(0.1)
                continue

            elif http_result.is_retryable and cur_retry_times >= self.config.max_retry_per_sample:
                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} was skipped due to max retries reached"
                )
                return RLRolloutResponseItem(state=RolloutState.SKIPPED)
            elif http_result.is_client_error:
                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} was skipped due to client error {http_result.error_type} with {http_result.error_msg}"
                )
                return RLRolloutResponseItem(state=RolloutState.SKIPPED)
            elif http_result.is_server_error:
                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} failed due to server error {http_result.error_type} with {http_result.error_msg}"
                )
                return RLRolloutResponseItem(state=RolloutState.FAILED)
            else:
                raise RuntimeError(
                    f"Unhandled error case for rollout request {uid} to {http_result.url}: {http_result.exception}"
                )

    async def _handle_stream_response(self, uid, sample_params, extra_params, response) -> RLRolloutResponseItem:
        last_trajectory = ""
        last_token_ids = []
        last_logprobs = []
        finish_reason = ""
        async for chunk in response.aiter_lines():
            if not chunk.startswith("data:"):
                continue
            try:
                chunk_data_str = chunk[len("data:") :].strip()
                if self.paused or chunk_data_str == "[DONE]":
                    finish_reason = "paused" if self.paused else finish_reason
                    break
                if not (chunk_data_str.startswith("{") and chunk_data_str.endswith("}")):
                    continue

                chunk_data = json.loads(chunk_data_str)

                if "return_token_ids" in extra_params and extra_params["return_token_ids"]:
                    last_trajectory = last_trajectory + chunk_data.get("text", "")
                    finish_reason = chunk_data["meta_info"].get("finish_reason")
                    if finish_reason is not None:
                        finish_reason = finish_reason["type"]

                    output_token_logprobs = chunk_data["meta_info"].get("output_token_logprobs")
                    if output_token_logprobs is not None:
                        for token_logprob in output_token_logprobs:
                            last_logprobs.append(token_logprob[0])
                            last_token_ids.append(token_logprob[1])
                else:
                    delta_content = chunk_data["choices"][0]["delta"].get("content")
                    last_trajectory = last_trajectory + delta_content if delta_content else last_trajectory
                    last_token_id = chunk_data["choices"][0]["delta"].get("gen_tokens")
                    if last_token_id is not None:
                        last_token_ids.extend(last_token_id)
                    finish_reason = chunk_data["choices"][0].get("finish_reason")
                    logprobs_content = chunk_data["choices"][0]["logprobs"]
                    if logprobs_content is not None:
                        for content_item in logprobs_content["content"]:
                            last_logprobs.append(content_item["logprob"])

            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error for chunk in request {uid}: {chunk}, error: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Error processing chunk for {uid}: {chunk}, error: {e}")
                return RLRolloutResponseItem(
                    response="",
                    finish_reason="failed",
                )

        assert finish_reason in ["stop", "length", "tool_call", "abort"], f"Unexpected finish_reason: {finish_reason}"
        rollout_response = RLRolloutResponseItem(
            response=last_trajectory,
            response_ids=last_token_ids if len(last_token_ids) > 0 else None,
            num_return_tokens=len(last_token_ids) if len(last_token_ids) > 0 else None,
            finish_reason=finish_reason,
            logprobs=last_logprobs,
        )
        return rollout_response

    async def _handle_non_stream_response(
        self, root_id, action_id, sample_params, extra_params, response, input_extra_info
    ) -> RLRolloutResponseItem:
        response = response.json()
        uid = action_id
        if "return_token_ids" in extra_params and extra_params["return_token_ids"]:
            last_logprobs: list[float] = []
            try:
                extra_info = {}
                finish_reason = response["meta_info"]["finish_reason"]["type"]
                if finish_reason == "abort" and self.receive_abort_request.is_set() is False:
                    self.receive_abort_request.set()
                    self.logger.info(f"Setting receive_abort_request to True for rank {self.rank}")
                if "output_token_logprobs" in response["meta_info"]:
                    if response["meta_info"]["output_token_logprobs"] is None:
                        last_token_ids = []
                        last_logprobs = []
                    else:
                        last_token_ids = [item[1] for item in response["meta_info"]["output_token_logprobs"]]
                        last_logprobs = [item[0] for item in response["meta_info"]["output_token_logprobs"]]
                        assert len(last_token_ids) <= sample_params["max_tokens"], (
                            f"Generation length exceeds the limit: generated length is {len(last_token_ids)}, limit is {sample_params['max_tokens']}"
                        )
                else:
                    num_return_tokens = response["meta_info"].get("completion_tokens", 0)
                    last_token_ids = response["output_ids"][-num_return_tokens:] if num_return_tokens > 0 else []

                if self.enable_return_routed_experts:
                    assert "routed_experts" in response["meta_info"], (
                        "enable_return_routed_experts is True, but routed_experts is not in meta_info"
                    )
                    routed_experts = response["meta_info"]["routed_experts"]  # token[layer[expert]]
                    # 处理当前专家
                    if routed_experts is not None:
                        if isinstance(routed_experts, str):
                            import base64

                            data = base64.b64decode(routed_experts)
                            routed_experts = ray.cloudpickle.loads(data)
                        else:
                            routed_experts = torch.tensor(routed_experts)  # n,layer,expert
                            routed_experts = ray.put(routed_experts)

                        # 处理历史专家
                        if "routed_experts" in input_extra_info and input_extra_info["routed_experts"] is not None:
                            exist_routed_experts = await input_extra_info["routed_experts"]  # n, layer, expert
                            cur_routed_experts = await routed_experts  # [n, layer, expert]
                            ray._private.internal_api.free(routed_experts)
                            ray._private.internal_api.free(input_extra_info["routed_experts"])
                            del input_extra_info["routed_experts"]
                            assert (exist_routed_experts.shape[0] - 1) > 0 and exist_routed_experts.shape[
                                0
                            ] - 1 <= cur_routed_experts.shape[0], (
                                f"Existing routed_experts shape: {exist_routed_experts.shape}, current routed_experts shape: {cur_routed_experts.shape}"
                            )
                            init_cur_roued_experts = cur_routed_experts.shape[0]
                            cur_routed_experts = cur_routed_experts[exist_routed_experts.shape[0] :, :, :]
                            concat_routed_experts = np.concatenate((exist_routed_experts, cur_routed_experts), axis=0)
                            prompt_tokens = response["meta_info"].get("prompt_tokens", 0)
                            response_tokens = response["meta_info"].get("completion_tokens", 0)
                            self.logger.debug(
                                f"[{root_id}/{action_id}] Partial Rollout Stats: "
                                f"Tokens(prompt={prompt_tokens}, response={response_tokens}, total={prompt_tokens + response_tokens}) | "
                                f"Experts(exist={exist_routed_experts.shape}, init_cur={init_cur_roued_experts}, cur={cur_routed_experts.shape}, concat={concat_routed_experts.shape})"
                            )
                            extra_info["routed_experts"] = ray.put(concat_routed_experts)
                        else:
                            extra_info["routed_experts"] = routed_experts

                    else:
                        assert finish_reason == "abort", (
                            f"routed_experts is None, but finish_reason is {finish_reason}, expected abort. response: {response}"
                        )
                # NOTE: When set return_token_ids = True, the response must contain valid token_ids/logprobs.
                # If not, we consider it as an invalid response and retry it.
                # NOTE: !!! When finish_reason is abort, some queries may not return token_ids or logprobs. !!!
                if finish_reason != "abort" and (len(last_token_ids) == 0 or len(last_logprobs) == 0):
                    self.logger.error(f"Invalid rollout response for request {uid}: {response}")
                    return RLRolloutResponseItem(state=RolloutState.SKIPPED)
                else:
                    rollout_response = RLRolloutResponseItem(
                        response=response["text"],
                        response_ids=last_token_ids,
                        num_return_tokens=len(last_token_ids),
                        finish_reason=finish_reason,
                        logprobs=last_logprobs,
                        extra_info=extra_info,
                        state=RolloutState.ABORTED if finish_reason == "abort" else RolloutState.COMPLETED,
                    )
                    # self.logger.info(f"Rollout response for request {uid}: finish_reason={finish_reason}, num_return_tokens={len(last_token_ids)}")
                return rollout_response
            except KeyError as e:
                error_msg = f"Missing expected key {e} in response {response} for {uid}"
                raise RuntimeError(error_msg)
            except IndexError as e:
                error_msg = f"Index error {e} while processing response {response} for {uid}"
                raise RuntimeError(error_msg)
            except AssertionError as e:
                error_msg = f"AssertionError: {e} when processing response {response} for {uid}"
                raise RuntimeError(error_msg)
            except json.JSONDecodeError as e:
                error_msg = f"JSONDecodeError: {e} when processing response {response} for {uid}"
                raise RuntimeError(error_msg)
            except TypeError as e:
                error_msg = f"TypeError: {e} when processing response {response} for {uid}"
                raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error: {e} when processing response {response} for {uid}\nTraceback: {traceback.format_exc()}"
                raise RuntimeError(error_msg)
        else:
            # v1/chat/completions API response
            try:
                last_trajectory = response["choices"][0]["message"]["content"]
                finish_reason = response["choices"][0]["finish_reason"]
                rollout_response = RLRolloutResponseItem(
                    response=last_trajectory,
                    finish_reason=finish_reason,
                    num_return_tokens=response["usage"]["completion_tokens"],
                )
                return rollout_response
            except KeyError as e:
                error_msg = f"Missing expected key {e} in response {response} for {uid}"
                raise RuntimeError(error_msg)
            except IndexError as e:
                error_msg = f"Index error {e} while processing response {response} for {uid}"
                raise RuntimeError(error_msg)
            except AssertionError as e:
                error_msg = f"AssertionError: {e} when processing response {response} for {uid}"
                raise RuntimeError(error_msg)
            except json.JSONDecodeError as e:
                error_msg = f"JSONDecodeError: {e} when processing response {response} for {uid}"
                raise RuntimeError(error_msg)
            except TypeError as e:
                error_msg = f"TypeError: {e} when processing response {response} for {uid}"
                raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error: {e} when processing response {response} for {uid}\nTraceback: {traceback.format_exc()}"
                raise RuntimeError(error_msg)

    async def rollout(
        self,
        prompt: Union[str, List[Dict[str, Any]]] | None = None,
        input_ids: Optional[List[int]] | None = None,
        tools: List = [],
        tool_choice: str = "auto",
        sample_params: dict = dict(),
        extra_params: dict = dict(),
        format: str = "openai",
        extra_info: dict = dict(),
    ) -> RLRolloutResponseItem:
        """Public method to initiate a rollout.

        Args:
            prompt (str): The input prompt for generation.
            sample_params (dict): Parameters for sampling.

        Returns:
            The result of the `rollout_task`.
        """
        return await self.rollout_task(
            prompt, input_ids, tools, tool_choice, sample_params, extra_params, format=format, extra_info=extra_info
        )

    def pause(self):
        """Pause the worker's generation process."""
        self.paused = True

    def restart(self):
        """Resume the worker's generation process."""
        self.receive_abort_request.clear()

    def check_health(self) -> bool:
        """Check the health of the worker's server.

        Returns:
            bool: True if the server is healthy, False otherwise.
        """
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self.config.api_key}",
            }
            response = requests.get(
                f"{self.server_url}/{self.endpoints['health_generate']}", headers=headers, timeout=5.0
            )
            return response.status_code == 200
        except requests.RequestException as e:
            self.logger.error(f"Health check failed for server {self.server_url}: {e}")
            return False

    def shutdown(self):
        """Shut down the worker, its server task, and any child processes."""
        if self.server_task is not None:
            ray.cancel(self.server_task)
            return

        if self.server_process is not None:
            import psutil

            parent = psutil.Process(self.server_process.pid)
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            gone, alive = psutil.wait_procs(children, timeout=5)
            for child in alive:
                child.kill()
            parent.terminate()
            parent.wait(timeout=5)
            self.logger.debug(f"Worker {self.rank} server process and its children terminated.")
            return

    @abstractmethod
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
        """Abstract method to create a generation request.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("_create_request must be implemented in subclass")

    @abstractmethod
    def _transform_rollout_config_to_server_configs(self):
        """Abstract method to transform rollout config to server configs.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("_transform_rollout_config_to_server_configs must be implemented in subclass")

    @abstractmethod
    def _transform_sample_params(self, sample_params: Dict):
        """Abstract method to transform rollout config to server configs.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("_transform_rollout_config_to_server_configs must be implemented in subclass")

    @abstractmethod
    def get_logprobs(self, input_ids, sampling_params):
        """Abstract method to get log probabilities.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("get_logprobs must be implemented in subclass")

    @abstractmethod
    def update_weights(self):
        """Abstract method to update model weights.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("update_weights must be implemented in subclass")

    @abstractmethod
    def reset_prefix_cache(self):
        """Abstract method to reset the prefix cache.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("reset_prefix_cache must be implemented in subclass")

    @abstractmethod
    def offload(self):
        """Abstract method to offload the model and KVcache.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("reset_prefix_cache must be implemented in subclass")

    @abstractmethod
    def onload_weights(self):
        """Abstract method to onload the model weights.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def onload_kvcache(self):
        """Abstract method to onload the KV cache.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def pause_generation(self):
        """Abstract method to pause the generation process.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("pause_generation must be implemented in subclass")

    @abstractmethod
    def continue_generation(self):
        """Abstract method to continue the generation process.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("continue_generation must be implemented in subclass")

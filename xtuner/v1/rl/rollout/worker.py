import asyncio
import copy
import json
import multiprocessing
import os
import socket
import time
import traceback
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Union

import httpx
import ray
import requests  # type: ignore[import-untyped]
import torch
from cyclopts import Group, Parameter
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, PrivateAttr
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from typing_extensions import Annotated

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status, update_status_from_finish_reason
from xtuner.v1.rl.utils import (
    AutoAcceleratorWorkers,
    SingleAcceleratorWorker,
    find_master_addr_and_port,
    get_eos_token,
)
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType, HttpRequestResult


if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup


infer_group = Group("inference", help="Inference worker configuration.")


class RolloutConfig(BaseModel):
    """Rollout worker configuration for XTuner.

    This class defines comprehensive configuration parameters for rollout workers in XTuner,
    supporting multiple inference backends with distributed computing and optimization features.

    Args:
        env (str): Environment variables for the rollout worker. Defaults to "".
        backend (str): Backend framework ('vllm', 'lmdeploy', etc.). Defaults to "lmdeploy".
        model_path (str | Path): Path to the inference model.
        model_name (str): Model name for the backend engine.
        tokenizer_path (str): Path to the model tokenizer. Defaults to "".
        api_key (Optional[Union[List[str], str]]): API keys for rollout service. Supports single key or
            list of keys. Defaults to None.
        api_port (Optional[int]): Port number for the rollout API server. If not set, it will find an
            available port starting from 8000. Defaults to 8000.
        gpus_per_node (int): Number of GPUs per node. Defaults to 8.
        dtype (str): Model data type ('bfloat16', 'float16', 'int8'). Defaults to "bfloat16".
        gpu_memory_utilization (float): GPU memory utilization ratio. Defaults to 0.85.
        random_seed (int): Random seed for reproducible generation. Defaults to 1024.
        rollout_cross_node_comm (bool): Enable cross-node communication. Defaults to False.
        rollout_max_batch_size_per_instance (int): Maximum batch size for the rollout worker. If not set, it
            will be determined automatically based on `context_length`. Defaults to 512.
        allow_over_concurrency_ratio (float): Factor to allow over-concurrency in HTTP requests for the
            rollout worker to improve GPU utilization. Defaults to 1.2.
        tensor_parallel_size (int): GPUs per inference engine (tensor parallelism). Defaults to 1.
        expert_parallel_size (int): Experts per inference engine (expert parallelism). Defaults to 1.
        enable_chunked_prefill (bool): Enable chunked prefill for memory efficiency. Defaults to False.
        chunked_prefill_size (int): Chunk size for prefill operations. Defaults to 128.
        skip_load_weights (bool): Skip weight loading for rollout worker. Defaults to False.
        rollout_timeout (float): Timeout duration in seconds for rollout requests. Defaults to 3600.0.
        context_length (int): Context length for the rollout worker.
        launch_server_method (Literal["ray", "multiprocessing"]): Server launch method. Defaults to "ray".
        system_prompt (Optional[str]): System prompt to guide generation behavior. Defaults to None.
        extra_rollout_config (Optional[dict]): Backend-specific configurations using engine prefixes
            (e.g., 'vllm_enable_chunked_prefill', 'lmdeploy_max_batch_size'). Defaults to empty dict.

    **Examples:**

    Example configuration with LMDeploy backend::

        config = RolloutConfig(
            env="test_env",
            model_path="Qwen/Qwen3-8B",
            model_name="Qwen3-8B",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.6,
            gpus_per_node=8,
            backend="lmdeploy",
        )
    """

    model_config = ConfigDict(extra="forbid")

    # base config
    env: Annotated[
        str,
        Parameter(group=infer_group, help="Environment variables to set for the rollout."),
    ] = ""
    device: Annotated[str, Parameter(group=infer_group, help="Device to be used for the rollout worker.")] = "GPU"
    model_path: Annotated[str | Path, Parameter(group=infer_group, help="Path to the SGLang model.")]
    model_name: Annotated[
        str | None, Parameter(group=infer_group, help="Name of the model to be used in the LMDeploy.")
    ] = None
    tokenizer_path: Annotated[
        str | None, Parameter(group=infer_group, help="Path to the tokenizer for the model.")
    ] = None
    api_key: Annotated[
        Optional[Union[List[str], str]],
        Parameter(
            group=infer_group,
            help="API keys for the rollout service. Can be a single key or a list of keys.",
        ),
    ] = None
    api_port: Annotated[
        int,
        Parameter(group=infer_group, help="Port number for the rollout API server. If not set, 8000 will be used."),
    ] = 8000
    api_host: Annotated[
        str,
        Parameter(group=infer_group, help="Host for the rollout API server."),
    ] = "0.0.0.0"
    gpus_per_node: Annotated[int, Parameter(group=infer_group, help="Number of GPUs allocated per node.")] = 8
    dtype: Annotated[
        str,
        Parameter(group=infer_group, help="Data type for the model, e.g., 'bfloat16', 'float16', 'int8'."),
    ] = "bfloat16"
    gpu_memory_utilization: Annotated[
        float, Parameter(group=infer_group, help="GPU memory utilization for the rollout worker.")
    ] = 0.85
    random_seed: Annotated[int, Parameter(group=infer_group, help="Random seed for the rollout worker.")] = 1024
    # distributed config
    rollout_cross_node_comm: Annotated[
        bool,
        Parameter(
            group=infer_group,
            help="Whether to enable cross-node communication for the rollout worker.",
        ),
    ] = False
    dist_port_base: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Base port number for distributed communication among rollout workers.",
        ),
    ] = 35000
    rollout_max_batch_size_per_instance: Annotated[
        Optional[int],
        Parameter(
            group=infer_group,
            help="Maximum batch size for the rollout worker. If not set, it will be determined automatically based on the model and GPU memory.",
        ),
    ] = None
    allow_over_concurrency_ratio: Annotated[
        float,
        Parameter(
            group=infer_group,
            help="Factor to allow over concurrency in the http request for rollout worker to improve GPU utilization.",
        ),
    ] = 1.2
    tensor_parallel_size: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Number of GPUs allocated for each inference engine in the rollout worker.",
        ),
    ] = 1
    expert_parallel_size: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Number of experts allocated for each inference engine in the rollout worker.",
        ),
    ] = 1
    # optimization config
    enable_chunked_prefill: Annotated[
        bool,
        Parameter(
            group=infer_group,
            help="Whether to enable chunked prefill for the rollout worker.",
        ),
    ] = False
    chunked_prefill_size: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Chunked prefill size for the rollout worker.",
        ),
    ] = 128
    skip_load_weights: Annotated[
        bool,
        Parameter(
            group=infer_group,
            help="Whether to skip loading weights for the rollout worker.",
        ),
    ] = False
    enable_return_routed_experts: Annotated[
        bool,
        Parameter(
            group=infer_group,
            help="Whether to enable returning routed experts for the rollout worker.",
        ),
    ] = False
    launch_server_method: Annotated[
        Literal["ray", "multiprocessing"],
        Parameter(
            group=infer_group,
            help="Method to launch the rollout server, either 'ray' or 'multiprocessing'.",
        ),
    ] = "ray"
    rollout_timeout: Annotated[
        float,
        Parameter(
            group=infer_group,
            help="Timeout duration (in seconds) for rollout requests.",
        ),
    ] = 1200.0
    context_length: Annotated[
        Optional[int],
        Parameter(
            group=infer_group,
            help="Context length for the rollout worker.",
        ),
    ] = None
    tool_call_parser: Annotated[
        Literal["none", "qwen3", "qwen3p5"],
        Parameter(
            group=infer_group,
            help='Structured tool-call parser to apply to rollout output. Use "none" to disable parsing, "qwen3" to enable Qwen3 tool-call parsing, or "qwen3p5" to enable Qwen3.5 coder-style tool-call parsing.',
        ),
    ] = "none"
    reasoning_parser: Annotated[
        Literal["none", "qwen3"],
        Parameter(
            group=infer_group,
            help='Reasoning parser to apply to rollout output. Use "none" to disable parsing or "qwen3" to enable Qwen3 <think> parsing.',
        ),
    ] = "none"
    enable_float8: Annotated[
        bool,
        Parameter(
            group=infer_group,
            help="Whether to enable float8 quantization for the rollout worker.",
        ),
    ] = False
    extra_rollout_config: Annotated[
        dict,
        Parameter(
            group=infer_group,
            help='Extra configuration for different rollout worker. vllm parameters will start with prefix "vllm", etc.',
        ),
    ] = {}
    max_retry_per_worker: Annotated[
        Optional[int],
        Parameter(
            group=infer_group,
            help="Maximum number of retries per rollout worker before deactivation.",
        ),
    ] = None
    max_retry_per_sample: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Maximum number of retries per sample before marking it as failed.",
        ),
    ] = 1
    worker_log_dir: Annotated[Path, Parameter(help="Directory to save worker logs.")] = Path.cwd() / "work_dir"
    health_check_interval_seconds: Annotated[
        float,
        Parameter(
            group=infer_group,
            help="Interval in seconds between rollout worker health checks.",
        ),
    ] = 30.0
    health_check_failure_threshold: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Number of consecutive health check failures required before marking a worker inactive.",
        ),
    ] = 3
    _logged_server_urls_per_engine: bool = PrivateAttr(default=False)

    @property
    def rollout_backend(self) -> str:
        backend = ""
        if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
            backend = "sglang"
        elif os.environ.get("XTUNER_USE_VLLM", "0") == "1":
            backend = "vllm"
        elif os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "1":
            backend = "lmdeploy"

        assert backend in ["sglang", "vllm", "lmdeploy"], (
            f"Unsupported rollout backend: {backend}. Please set XTUNER_USE_SGLANG, XTUNER_USE_VLLM, or XTUNER_USE_LMDEPLOY to 1."
        )
        return backend

    @property
    def server_urls_per_engine(self) -> int:
        # server_urls_per_engine is introduced for lmdeploy ep settings
        # for now only lmdeploy pytorch backend with ep > 1 requires multiple server urls per engine
        if self.rollout_backend == "lmdeploy" and self.expert_parallel_size > 1:
            # when expert parallelism is used, lmdeploy requires `expert_parallel_size` server instances per engine
            if not self._logged_server_urls_per_engine:
                self._logged_server_urls_per_engine = True
                get_logger().info(
                    f"Setting server_urls_per_engine={self.expert_parallel_size} due to expert parallelism in LMDeploy."
                )
            return self.expert_parallel_size
        else:
            return 1

    def model_post_init(self, __context: Any) -> None:
        if self.model_name is None:
            model_name_from_config = None
            config_json_path = Path(self.model_path) / "config.json"
            try:
                with open(config_json_path, encoding="utf-8") as f:
                    config_data = json.load(f)
                    model_name_from_config = config_data.get("model_type")
            except (json.JSONDecodeError, OSError):
                pass
            self.model_name = model_name_from_config or Path(self.model_path).name

        if self.tokenizer_path is None:
            self.tokenizer_path = str(self.model_path)

        port = self.api_port
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((self.api_host if self.api_host != "0.0.0.0" else "localhost", port))
                    break
                except OSError:
                    port += 1
        self.api_port = port

        if self.device == "NPU":
            self.gpus_per_node = 16

        if self.rollout_backend == "sglang":
            self.launch_server_method = "multiprocessing"
            self.rollout_cross_node_comm = False
        else:
            self.launch_server_method = "ray"
            self.rollout_cross_node_comm = True

        if self.rollout_max_batch_size_per_instance is None:
            assert self.context_length is not None, (
                "context_length must be set if rollout_max_batch_size_per_instance is not provided."
            )
            # TODO(@duanyanhui): Provide better suggestions for different models/input-output lengths
            if self.context_length <= 4096:
                self.rollout_max_batch_size_per_instance = 1024
            elif self.context_length <= 8192:
                self.rollout_max_batch_size_per_instance = 512
            else:
                self.rollout_max_batch_size_per_instance = 128

        if self.max_retry_per_worker is None:
            self.max_retry_per_worker = self.rollout_max_batch_size_per_instance

        self.worker_log_dir.mkdir(parents=True, exist_ok=True)

    def build(self, placement_group: "PlacementGroup"):
        """Build and return a Ray remote RolloutController from this config.

        Args:
            placement_group: The placement group for scheduling RolloutWorker actors.

        Returns:
            A Ray actor handle (proxy) of RolloutController.
        """
        import ray

        from xtuner.v1.rl.rollout.controller import RolloutController

        return (
            ray.remote(RolloutController)
            .options(max_concurrency=int(os.environ.get("RAY_MAX_CONCURRENCY", 1000)))
            .remote(self, placement_group)
        )


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
        self.dist_init_addr: str = ""
        self.serverl_url: str = ""

    def init(self, dist_init_addr: str) -> tuple[int, str]:
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
        self._launch_server()
        return (self.rank, self.server_url)

    def init_dist_port(self) -> str:
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

    def shutdown(self):
        """Shut down the worker, its server task, and any child processes."""
        if self.server_task is not None:
            ray.cancel(self.server_task, force=True)
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

    def pause_generation(self):
        """Pause the worker's generation process."""
        self.paused = True

    def continue_generation(self):
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

    async def generate(self, rollout_state: RolloutState) -> RolloutState:
        # TODO(@duanyanhui):
        # 1. support claude format input
        # 2. 需要看下新的输入输出(RolloutState)怎么适配PartialRollout的逻辑，先跑起来
        # 3. 对于流式返回的response先删掉，目前还用不上，等需要的时候再加上

        uid = rollout_state.uid
        sample_params: SampleParams = rollout_state.sample_params

        if sample_params.return_token_ids:
            endpoint_url = f"{self.server_url}/{self.endpoints['generate']}"
        else:
            endpoint_url = f"{self.server_url}/{self.endpoints['v1/chat/completions']}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        max_retries = self.config.max_retry_per_sample
        payload = self._get_request_payload(rollout_state)

        # 早退逻辑 1：检查是否已被标记为完成
        if rollout_state.status == Status.COMPLETED:
            self.logger.debug(f"Request {uid} is already marked as COMPLETED, skipping generation.")
            return rollout_state

        # 早退逻辑 2：检测输入是否还需要 generation (安全获取变量)
        input_ids = payload.get("input_ids", [])
        max_tokens = payload.get("max_tokens")

        last_id = input_ids[-1] if len(input_ids) > 0 else "None"
        is_max_tokens_zero = max_tokens is not None and max_tokens <= 0
        is_eos_reached = len(input_ids) > 0 and input_ids[-1] in self.eos_token

        if is_max_tokens_zero or is_eos_reached:
            self.logger.debug(
                f"No generation needed for request {uid}: max_tokens={max_tokens} or last input_id={last_id} is in eos_token."
            )
            rollout_state.status = Status.COMPLETED
            rollout_state.response_ids = []
            rollout_state.response = ""
            rollout_state.logprobs = []
            rollout_state.response_mask = []
            rollout_state.response_rollout_steps = []
            rollout_state.finish_reason = "stop" if is_eos_reached else "length"
            return rollout_state

        for attempt in range(max_retries + 1):
            is_last_attempt = attempt == max_retries
            http_result = await self._safe_post_request(endpoint_url, headers=headers, payload=payload)

            # Case 1: HTTP Request is Successful
            if http_result.response:
                # Case 1.1: Valid rollout response
                rollout_state = await self._safe_handle_response(rollout_state, http_result.response)
                if rollout_state.status in [Status.COMPLETED, Status.ABORTED]:
                    return rollout_state

                if is_last_attempt:
                    # Case 1.2: Invalid rollout response and no retries left, so we return FAILED
                    self.logger.warning(
                        f"Invalid rollout response for request {uid} after {max_retries} attempts, marking as FAILED."
                    )
                    rollout_state.status = Status.FAILED
                    rollout_state.error_msg = f"Invalid rollout response after {max_retries} attempts."
                    return rollout_state

                # Case 1.3: Invalid rollout response but we have retries left
                self.logger.warning(
                    f"Invalid rollout response for request {uid}, retrying {attempt + 1}/{max_retries}."
                )
                await asyncio.sleep(0.1)
                continue

            # Case 2: Error occurred during HTTP Request
            if http_result.error_type == HttpRequestErrorType.REQUEST_ABORTED:
                # Case 2.1: The request was aborted due to an signal set by `receive_abort_request`
                rollout_state.finish_reason = "abort"
                rollout_state.status = update_status_from_finish_reason("abort")
                return rollout_state

            if http_result.is_client_error:
                # Case 2.2: A non-retryable client error occurred (such as 4xx HTTP status)
                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} was skipped due to client error {http_result.error_type} with {http_result.error_msg}"
                )
                rollout_state.error_msg = (
                    f"Client error {http_result.error_type} with message: {http_result.error_msg}"
                )
                rollout_state.status = Status.FAILED
                return rollout_state

            if http_result.is_server_error:
                # Case 2.3: A non-retryable server error occurred (such as 5xx HTTP status)
                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} failed due to server error {http_result.error_type} with {http_result.error_msg}"
                )
                rollout_state.error_msg = (
                    f"Server error {http_result.error_type} with message: {http_result.error_msg}"
                )
                rollout_state.status = Status.FAILED
                return rollout_state

            # Case 3: Retryable error occurred during HTTP Request
            if http_result.is_retryable:
                if is_last_attempt:
                    self.logger.warning(
                        f"rollout request {uid} to {http_result.url} failed after {max_retries} attempts due to retryable error {http_result.error_type} with {http_result.error_msg}"
                    )
                    rollout_state.error_msg = f"Request failed after {max_retries} attempts due to retryable error {http_result.error_type} with message: {http_result.error_msg}"
                    rollout_state.status = Status.FAILED
                    return rollout_state

                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} failed due to retryable error {http_result.error_type} with {http_result.error_msg}, retrying {attempt + 1}/{max_retries}."
                )
                await asyncio.sleep(0.1)
                continue

            # Case 4: Unknown error occurred during HTTP Request and stop the rollout
            if http_result.is_unknown_error:
                raise RuntimeError(
                    f"Unexpected error during rollout request {uid} to {http_result.url}: {http_result.exception}"
                )
        return rollout_state

    def _launch_server(self):
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

    async def _safe_handle_response(self, rollout_state: RolloutState, http_response: httpx.Response) -> RolloutState:
        uid = rollout_state.message_uid
        sample_params = rollout_state.sample_params
        is_token_out = sample_params.return_token_ids
        response = http_response.json()
        if is_token_out:
            response_ids: list[int] = []
            logprobs: list[float] = []
            routed_experts = None
            returned_response = ""
            finish_reason = response["meta_info"]["finish_reason"]["type"]
            if finish_reason == "abort" and self.receive_abort_request.is_set() is False:
                self.receive_abort_request.set()
                self.logger.info(f"Setting receive_abort_request to True for rank {self.rank}")
            try:
                returned_response = response.get("text", "")
                # 获取response_ids && respoonse_ids
                if (
                    "output_token_logprobs" in response["meta_info"]
                    and response["meta_info"]["output_token_logprobs"] is not None
                ):
                    response_ids = [item[1] for item in response["meta_info"]["output_token_logprobs"]]
                    logprobs = [item[0] for item in response["meta_info"]["output_token_logprobs"]]
                else:
                    num_return_tokens = response["meta_info"].get("completion_tokens", 0)
                    response_ids = response["output_ids"][-num_return_tokens:] if num_return_tokens > 0 else []

                # 获取 routed_experts
                if self.enable_return_routed_experts:
                    assert "routed_experts" in response["meta_info"], (
                        "enable_return_routed_experts is True, but routed_experts is not in meta_info"
                    )
                    routed_experts = response["meta_info"]["routed_experts"]  # token[layer[expert]]
                    if routed_experts is not None:
                        if isinstance(routed_experts, str):
                            import base64

                            data = base64.b64decode(routed_experts)
                            routed_experts = ray.cloudpickle.loads(data)
                        else:
                            routed_experts = torch.tensor(routed_experts)  # n,layer,expert
                            routed_experts = ray.put(routed_experts)

                # 获取 status
                rollout_status = update_status_from_finish_reason(finish_reason)

                # 检查输出结果
                if rollout_status == Status.COMPLETED:
                    validation_errors = []

                    if not response_ids:
                        validation_errors.append("empty response_ids")

                    if not response:
                        validation_errors.append("empty response text")

                    if sample_params.return_logprob and not logprobs:
                        validation_errors.append("missing logprobs")

                    if self.enable_return_routed_experts and routed_experts is None:
                        validation_errors.append("missing routed_experts")

                    if validation_errors:
                        error_msg = f"Incomplete rollout data for msg {uid}: {', '.join(validation_errors)}"
                        self.logger.error(error_msg)
                        rollout_state.status = Status.FAILED
                        rollout_state.error_msg = error_msg
                        return rollout_state
                elif rollout_status == Status.FAILED:
                    error_msg = f"Rollout failed for msg {uid} with finish_reason {finish_reason}"
                    self.logger.error(error_msg)
                    rollout_state.status = Status.FAILED
                    rollout_state.error_msg = error_msg
                    return rollout_state

                rollout_state.response = returned_response
                rollout_state.response_ids = response_ids
                rollout_state.logprobs = logprobs
                rollout_state.routed_experts = routed_experts
                rollout_state.finish_reason = finish_reason
                rollout_state.status = rollout_status
                return rollout_state
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
                returned_response = response["choices"][0]["message"]["content"]
                finish_reason = response["choices"][0]["finish_reason"]
                rollout_status = update_status_from_finish_reason(finish_reason)
                if rollout_status == Status.COMPLETED and not returned_response:
                    self.logger.error(f"Empty response text for msg {uid} with finish_reason {finish_reason}")
                    rollout_state.status = Status.FAILED
                    rollout_state.error_msg = "Empty response text"
                    return rollout_state

                rollout_state.response = returned_response
                rollout_state.finish_reason = finish_reason
                rollout_state.status = rollout_status
                return rollout_state
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

    def _set_engine_rank_mesh_array(self, engine_rank_mesh_array: list[list[int]]):
        self.engine_rank_mesh_array = engine_rank_mesh_array

    def _set_engine_bundle_idxs(self, engine_bundle_idxs: list[int]):
        """Set the bundle indices for the inference engine.

        This is used by some backends (like LMDeploy with Ray executor) to
        know which bundles in the placement group belong to this engine.

        Args:
            engine_bundle_idxs (list[int]): A list of bundle indices.
        """
        self.engine_bundle_idxs = engine_bundle_idxs

    @abstractmethod
    def _get_request_payload(self, rollout_state: RolloutState) -> dict:
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
    def _transform_sample_params(self, sample_params: SampleParams) -> dict:
        """Abstract method to transform rollout config to server configs.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("_transform_rollout_config_to_server_configs must be implemented in subclass")

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

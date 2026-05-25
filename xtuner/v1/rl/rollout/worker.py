import asyncio
import json
import multiprocessing
import os
import socket
import threading
import time
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Union

import httpx
import ray
from cyclopts import Group, Parameter
from pydantic import BaseModel, ConfigDict, PrivateAttr
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from typing_extensions import Annotated

from transformers import AutoTokenizer
from xtuner.v1.rl.utils import (
    AutoAcceleratorWorkers,
    CPUResourcesConfig,
    SingleAcceleratorWorker,
    find_master_addr_and_port,
    get_eos_token,
    register_cpu_resources,
)
from xtuner.v1.utils import get_logger

from .session_server import SessionServerActor
from .utils import ROLLOUT_RAY_GET_TIMEOUT


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
    data_parallel_size: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Number of GPUs allocated for processing data batches in parallel (Data Parallelism).",
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
    max_retry_per_sample: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Maximum number of retries per sample before marking it as failed.",
        ),
    ] = 1
    max_prefill_token_num: Annotated[
        Optional[int],
        Parameter(
            group=infer_group,
            help="The number of tokens each iteration during prefill.",
        ),
    ] = None
    router_n_groups: Annotated[
        Optional[int],
        Parameter(
            group=infer_group,
            help="The number of groups in MoE model with group router, e.g. Intern-S1-Pro.",
        ),
    ] = None
    fp32_lm_head: Annotated[
        bool,
        Parameter(
            group=infer_group,
            help="Use float32 for language model head.",
        ),
    ] = False
    worker_log_dir: Annotated[Path, Parameter(help="Directory to save worker logs.")] = Path.cwd() / "work_dir"
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

    @property
    def num_gpus_per_engine(self) -> int:
        return self.expert_parallel_size if self.expert_parallel_size > 1 else self.tensor_parallel_size

    def get_active_servers_count(self, num_rollout_workers: int) -> tuple[int, int]:
        """Calculate the number of active servers and nodes per engine."""
        # NOTE: Since different inference engines have different launch methods,
        # the number of nodes contained in each engine is not consistent.
        # For example, sglang requires starting an inference engine for each node,
        # while lmdeploy and vllm do not. Therefore, calculate active servers from the rollout config.
        nodes_per_engine = (
            1
            if self.rollout_cross_node_comm or self.num_gpus_per_engine < self.gpus_per_node
            else self.num_gpus_per_engine // self.gpus_per_node
        )
        active_servers_count = max(
            1,
            int((num_rollout_workers // self.num_gpus_per_engine) * nodes_per_engine * self.server_urls_per_engine),
        )
        return active_servers_count, nodes_per_engine

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
        from xtuner.v1.rl.rollout.rollout_worker_build import build_rollout_runtime

        num_workers = 1
        register_cpu_resources(
            name="rollout_controller",
            cpu_resources=CPUResourcesConfig(num_workers=num_workers),
        )
        runtime = build_rollout_runtime(self, placement_group)
        return ray.remote(RolloutController).options(num_cpus=num_workers).remote(self, runtime=runtime)


class RolloutWorker(SingleAcceleratorWorker):
    """Base class for a rollout worker that runs an inference server.

    This class manages the lifecycle of a distributed inference server, including initialization, launching, weight
    updates, and backend control. Runtime generation is handled by RolloutWorkerGenerator.
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
        assert config.rollout_max_batch_size_per_instance, (
            "rollout_max_batch_size_per_instance must be set in RolloutConfig"
        )
        self.server_task = None
        self.engine_bundle_idxs: list[int] = []
        self.server_process: Optional[multiprocessing.Process] = None
        self.session_server_actor: Any | None = None
        self.session_server_url: str | None = None
        self.logger = get_logger(log_dir=config.worker_log_dir, tag="RolloutWorker")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, trust_remote_code=True)
        self.check_flag = True  # only print once
        self.enable_return_routed_experts = self.config.enable_return_routed_experts
        if self.rank == 0:
            self.logger.info(f"RolloutConfig:\n{self.config.model_dump_json(indent=2)}")
        eos_token = get_eos_token(self.config.model_path)
        self.logger.info(f"Using eos_token: {eos_token} for model at {self.config.model_path}")
        self.eos_token: List[int] = [eos_token] if isinstance(eos_token, int) else eos_token
        self.receive_abort_request = threading.Event()
        # After an abort signal, wait this long for an in-flight rollout request to return before cancelling the
        # client-side request task.
        self.abort_timeout = 10.0
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
        self._stop_session_server()
        self._launch_server()
        self._start_session_server()
        return (self.rank, self.server_url)

    def init_dist_port(self) -> str:
        """Initialize distributed communication ports.

        This method acquires four free ports for the distributed setup:
        one for Ray's distributed communication, one for the inference server,
        one for NCCL, and one for the per-worker SessionServer proxy.

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
                nums=4,
                start_port=start_port,
                end_port=end_port,
            )
        )

        self.dist_port = self.ports[0]
        self.server_port = self.ports[1]
        self.nccl_port = self.ports[2]
        self.session_server_port = self.ports[3]
        self.dist_init_addr = f"{self.host}:{self.dist_port}"
        self.server_url = f"http://{self.host}:{self.server_port}"
        return self.dist_init_addr

    def shutdown(self):
        """Shut down the worker, its server task, and any child processes."""
        self._stop_session_server()

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

    def _start_session_server(self) -> None:
        """Start the per-worker SessionServer proxy."""
        if self.session_server_actor is not None:
            return

        current_pg = ray.util.get_current_placement_group()
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=current_pg,
            placement_group_capture_child_tasks=False,
            placement_group_bundle_index=self.engine_bundle_idxs[0],
        )
        self.session_server_actor = (
            ray.remote(SessionServerActor)
            .options(
                scheduling_strategy=scheduling_strategy,
                num_cpus=0,
            )
            .remote(
                worker_base_url=self.server_url,
                tokenizer_path=str(self.config.tokenizer_path or self.config.model_path),
                host=self.host,
                port=self.session_server_port,
            )
        )
        self.session_server_url = ray.get(
            self.session_server_actor.start.remote(),
            timeout=ROLLOUT_RAY_GET_TIMEOUT,
        )

    def _stop_session_server(self) -> None:
        if self.session_server_actor is not None:
            try:
                ray.get(self.session_server_actor.stop.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)
            finally:
                ray.kill(self.session_server_actor)
                self.session_server_actor = None
            self.session_server_url = None

    def get_session_server_info(self) -> tuple[int, str | None]:
        return self.rank, self.session_server_url

    async def pause_generation(self):
        """Pause the worker's generation process."""
        self.receive_abort_request.set()
        return await self._send_abort_request()

    async def cleanup_after_pause(self) -> None:
        """Run backend-specific cleanup after paused requests are drained."""
        return None

    async def _send_abort_request(self) -> bool:
        url = f"{self.server_url}/abort_request"
        try:
            async with httpx.AsyncClient(timeout=self.abort_timeout) as client:
                response = await client.post(url, json={"abort_all": True})
            response.raise_for_status()
            self.logger.debug(f"Successfully sent abort request to {self.server_url}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send abort request to {self.server_url}: {e}")
            return False

    async def _wait_abort_request(self) -> None:
        while not self.receive_abort_request.is_set():
            await asyncio.sleep(1)

    def continue_generation(self):
        """Resume the worker's generation process."""
        self.receive_abort_request.clear()

    def _launch_server(self):
        """Launch the inference server as a separate process or Ray task."""
        server_configs = self._transform_rollout_config_to_server_configs()
        self.logger.info(f"Launch server task on server_url: {self.server_url}")

        # note(@duanyanhui): launch server as multiprocessing for sglang temporarily
        if self.config.launch_server_method == "multiprocessing":
            mp_ctx = multiprocessing.get_context("spawn")
            process = mp_ctx.Process(target=self.server_func, args=(server_configs,))
            process.start()
            self.server_process = process
            time.sleep(60)
            return
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
            return

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
    def _transform_rollout_config_to_server_configs(self):
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

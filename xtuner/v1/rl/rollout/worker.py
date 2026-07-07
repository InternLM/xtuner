import asyncio
import copy
import json
import multiprocessing
import os
import threading
import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, TypeAlias, Union, cast

import httpx
import ray
import requests  # type: ignore[import-untyped]
from cyclopts import Group, Parameter
from packaging.version import Version
from pydantic import BaseModel, ConfigDict
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from typing_extensions import Annotated

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    SampleParams,
    Status,
    reset_rollout_response,
    update_status_from_finish_reason,
)
from xtuner.v1.rl.trace import inject_trace_context, trace_span, traced_rollout_endpoint
from xtuner.v1.rl.utils import (
    AutoAcceleratorWorkers,
    CPUResourcesConfig,
    SingleAcceleratorWorker,
    get_eos_token,
    register_cpu_resources,
    with_trace_runtime_env,
)
from xtuner.v1.rl.utils.trace_utils import (
    TRACE_ATTR_ROLLOUT_BACKEND,
    TRACE_SPAN_ROLLOUT_WORKER_GENERATE,
    rollout_state_trace_attributes,
)
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType, HttpRequestResult

from .constants import ROLLOUT_HTTP_MAX_CONNECTIONS, ROLLOUT_RAY_GENERATE_MAX_CONCURRENCY
from .health_manager import ROLLOUT_RAY_GET_TIMEOUT
from .session_server import SessionServerActor
from .utils import PartialRolloutHandler


if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup


infer_group = Group("inference", help="Inference worker configuration.")
ROLLOUT_CONCURRENCY_GROUP_GENERATE = "generate"


@dataclass(frozen=True)
class ServerProcessSpec:
    """How to start one rollout server process."""

    # Worker rank that owns this server process.
    worker_rank: int
    # Placement-group bundle indexes assigned to this server process.
    placement_group_bundle_idxs: tuple[int, ...]
    # Distributed init address used by every server process in the same engine.
    # Filled after init_dist_port initializes worker-local ports.
    dist_init_addr: str | None = None
    # Whether this server is exposed as a rollout request entrypoint. Some
    # backends launch extra server processes that must participate in
    # lifecycle/health operations but must not be added to worker_server_urls_map
    # or receive normal rollout traffic.
    accepts_rollout_requests: bool = True
    # Node index of this server inside a multi-node logical engine.
    node_rank: int = 0
    # Number of nodes used by this logical engine.
    nnodes: int = 1


@dataclass(frozen=True)
class EngineLaunchSpec:
    """How to launch rollout servers for one logical inference engine."""

    # All worker ranks that form this logical inference engine.
    engine_ranks: tuple[int, ...]
    # Server processes required by this engine.
    server_processes: tuple[ServerProcessSpec, ...]

    @property
    def server_worker_ranks(self) -> tuple[int, ...]:
        return tuple(server.worker_rank for server in self.server_processes)

    @property
    def request_entrypoint_servers(self) -> tuple[ServerProcessSpec, ...]:
        return tuple(server for server in self.server_processes if server.accepts_rollout_requests)

    @property
    def request_entrypoint_worker_ranks(self) -> tuple[int, ...]:
        return tuple(server.worker_rank for server in self.request_entrypoint_servers)

    @property
    def placement_group_bundle_idxs(self) -> tuple[int, ...]:
        return tuple(
            bundle_idx for server in self.server_processes for bundle_idx in server.placement_group_bundle_idxs
        )


EngineLaunchSpecs: TypeAlias = tuple[EngineLaunchSpec, ...]


def get_rollout_worker_base_cls(config: "RolloutConfig") -> type["RolloutWorker"]:
    if config.rollout_backend == "lmdeploy":
        from .lmdeploy import LMDeployWorker

        return LMDeployWorker
    elif config.rollout_backend == "vllm":
        from .vllm import vLLMWorker

        return vLLMWorker
    elif config.rollout_backend == "sglang":
        from .sglang import SGLangWorker

        return SGLangWorker
    else:
        raise NotImplementedError(
            f"Rollout backend is not supported: {config.rollout_backend}. "
            "Please set XTUNER_USE_LMDEPLOY or XTUNER_USE_VLLM or XTUNER_USE_SGLANG environment variable."
        )


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
        gpus_per_node (int): Number of GPUs per node. Defaults to 8.
        dtype (str): Model data type ('bfloat16', 'float16', 'int8'). Defaults to "bfloat16".
        gpu_memory_utilization (float): GPU memory utilization ratio. Defaults to 0.85.
        random_seed (int): Random seed for reproducible generation. Defaults to 1024.
        rollout_cross_node_comm (bool): Enable cross-node communication. Defaults to False.
        weight_update_host (Optional[str]): Host used by train rank 0 to initialize the external NCCL weight update
            group. Defaults to None.
        weight_update_port (Optional[int]): Port used by train rank 0 to initialize the external NCCL weight update
            group. Defaults to 30000.
        rollout_max_batch_size_per_instance (int): Maximum batch size for the rollout worker. If not set, it
            will be determined automatically based on `context_length`. Defaults to 512.
        allow_over_concurrency_ratio (float): Deprecated compatibility option. Rollout runtime concurrency is
            controlled by fixed caps in xtuner.v1.rl.rollout.constants. Defaults to 1.2.
        tensor_parallel_size (int): GPUs per inference engine (tensor parallelism). Defaults to 1.
        expert_parallel_size (int): Experts per inference engine (expert parallelism). Defaults to 1.
        enable_chunked_prefill (bool): Enable chunked prefill for memory efficiency. Defaults to False.
        chunked_prefill_size (int): Chunk size for prefill operations. Defaults to 128.
        skip_load_weights (bool): Skip weight loading for rollout worker. Defaults to False.
        rollout_timeout (float): Timeout duration in seconds for rollout requests. Defaults to 1200.0.
        session_server_timeout (float): Timeout duration in seconds for SessionServer requests forwarded to rollout
            workers. Defaults to 1200.0.
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
    ] = 25000
    weight_update_host: Annotated[
        Optional[str],
        Parameter(
            group=infer_group,
            help=(
                "Host used by train rank 0 to initialize the external NCCL weight update group. "
                "Only used for NCCL weight update."
            ),
        ),
    ] = None
    weight_update_port: Annotated[
        Optional[int],
        Parameter(
            group=infer_group,
            help=(
                "Port used by train rank 0 to initialize the external NCCL weight update group. "
                "Only used for NCCL weight update."
            ),
        ),
    ] = 30000
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
            help=(
                "Deprecated compatibility option. Rollout runtime concurrency is controlled by fixed caps in "
                "xtuner.v1.rl.rollout.constants."
            ),
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
    session_server_timeout: Annotated[
        float,
        Parameter(
            group=infer_group,
            help="Timeout duration (in seconds) for SessionServer requests forwarded to rollout workers.",
        ),
    ] = 1200.0
    context_length: Annotated[
        Optional[int],
        Parameter(
            group=infer_group,
            help="Context length for the rollout worker.",
        ),
    ] = None
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
    health_check_interval_seconds: Annotated[
        float,
        Parameter(
            group=infer_group,
            help="Interval in seconds between rollout worker health checks.",
        ),
    ] = 30.0
    # LMDeploy /health returns an EngineHealthMonitor snapshot. The monitor's
    # backend probe timeout defaults to 10s and its poll interval defaults to
    # 12s, so XTuner's HTTP read timeout needs to be longer than 10s to avoid
    # turning a slow but informative /health response into a client-side
    # timeout.
    health_check_timeout_seconds: Annotated[
        float,
        Parameter(
            group=infer_group,
            help=(
                "HTTP timeout in seconds for rollout worker health check requests. "
                "The default is longer than LMDeploy's 10s backend health probe timeout."
            ),
        ),
    ] = 15.0
    health_check_failure_threshold: Annotated[
        int,
        Parameter(
            group=infer_group,
            help="Number of consecutive health check failures required before marking a worker inactive.",
        ),
    ] = 3
    enable_proxy: Annotated[
        bool,
        Parameter(
            group=infer_group,
            help="Register rollout session servers to routed API proxy and keep registrations in sync with health.",
        ),
    ] = False
    routed_proxy_url: Annotated[
        str,
        Parameter(
            group=infer_group,
            help="Routed API proxy base URL used to validate proxy chat completions after registration.",
        ),
    ] = "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn"
    routed_proxy_admin_url: Annotated[
        str,
        Parameter(
            group=infer_group,
            help="Routed API proxy admin base URL used for model registration and deletion.",
        ),
    ] = "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000"

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
    def num_gpus_per_engine(self) -> int:
        return self.expert_parallel_size if self.expert_parallel_size > 1 else self.tensor_parallel_size

    def model_post_init(self, __context: Any) -> None:
        default_allow_over_concurrency_ratio = type(self).model_fields["allow_over_concurrency_ratio"].default
        if self.allow_over_concurrency_ratio != default_allow_over_concurrency_ratio:
            get_logger().warning(
                "rollout_config.allow_over_concurrency_ratio is deprecated and no longer controls runtime "
                "rollout concurrency. The configured value "
                f"{self.allow_over_concurrency_ratio} will be ignored; fixed rollout concurrency caps from "
                "xtuner.v1.rl.rollout.constants are used instead."
            )

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

        num_workers = 1
        register_cpu_resources(
            name="rollout_controller",
            cpu_resources=CPUResourcesConfig(num_workers=num_workers),
        )
        ray_kwargs = {
            "concurrency_groups": {
                ROLLOUT_CONCURRENCY_GROUP_GENERATE: ROLLOUT_RAY_GENERATE_MAX_CONCURRENCY,
            },
        }
        return (
            ray.remote(**ray_kwargs)(RolloutController)
            .options(**with_trace_runtime_env({"num_cpus": num_workers}))
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
        self._default_skip_load_weights = config.skip_load_weights
        self.rank = rank
        self.master_addr = master_addr  # ray master
        self.master_port = master_port
        self.world_size = world_size
        self.accelerator = accelerator
        self.server_func: Callable
        self.endpoints: dict[str, str] = dict()
        self.engine_rank_mesh_array: list[list[int]] = []
        self.engine_launch_spec: EngineLaunchSpec | None = None
        # Keep this deliberately large so requests do not queue in the
        # RolloutWorker/httpx client; the inference engine owns rollout request
        # scheduling and queueing.
        http_concurrency = ROLLOUT_HTTP_MAX_CONNECTIONS
        limits = httpx.Limits(max_connections=http_concurrency, max_keepalive_connections=100)
        self.client = httpx.AsyncClient(limits=limits, timeout=self.config.rollout_timeout)
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
        self.dist_init_addr: str = ""
        self.serverl_url: str = ""
        self.partial_rollout_handler = PartialRolloutHandler()
        self.enable_partial_rollout: bool = False

    @staticmethod
    def _get_num_gpus_per_engine(config: RolloutConfig) -> int:
        return config.num_gpus_per_engine

    @classmethod
    def validate_engine_launch_specs(
        cls,
        engine_launch_specs: EngineLaunchSpecs,
        *,
        known_worker_ranks: tuple[int, ...] | None = None,
    ) -> EngineLaunchSpecs:
        """Validate backend launch layout before the controller launches
        servers."""
        if not engine_launch_specs:
            raise ValueError("engine_launch_specs must define at least one engine.")

        known_worker_rank_set = set(known_worker_ranks) if known_worker_ranks is not None else None
        seen_engine_ranks: set[int] = set()
        seen_server_ranks: set[int] = set()
        seen_bundle_idxs: set[int] = set()
        for engine_index, engine_spec in enumerate(engine_launch_specs):
            if not engine_spec.engine_ranks:
                raise ValueError(f"EngineLaunchSpec[{engine_index}] must define at least one engine rank.")
            engine_rank_set = set(engine_spec.engine_ranks)
            if len(engine_rank_set) != len(engine_spec.engine_ranks):
                raise ValueError(
                    f"EngineLaunchSpec[{engine_index}] has duplicate engine ranks: {engine_spec.engine_ranks}."
                )
            if known_worker_rank_set is not None:
                unknown_engine_ranks = sorted(
                    rank for rank in engine_spec.engine_ranks if rank not in known_worker_rank_set
                )
                if unknown_engine_ranks:
                    raise ValueError(
                        f"EngineLaunchSpec[{engine_index}] references unknown engine ranks: {unknown_engine_ranks}."
                    )
            duplicated_engine_ranks = sorted(rank for rank in engine_spec.engine_ranks if rank in seen_engine_ranks)
            if duplicated_engine_ranks:
                raise ValueError(
                    f"EngineLaunchSpec[{engine_index}] engine ranks appear in more than one engine: "
                    f"{duplicated_engine_ranks}."
                )
            seen_engine_ranks.update(engine_spec.engine_ranks)

            if not engine_spec.server_processes:
                raise ValueError(f"EngineLaunchSpec[{engine_index}] must define at least one server process.")

            for server_process in engine_spec.server_processes:
                server_rank = server_process.worker_rank
                if server_rank not in engine_rank_set:
                    raise ValueError(
                        f"EngineLaunchSpec[{engine_index}] server worker_rank={server_rank} "
                        f"must be part of engine_ranks={engine_spec.engine_ranks}."
                    )
                if server_rank in seen_server_ranks:
                    raise ValueError(f"Server worker_rank={server_rank} appears in more than one server process.")
                seen_server_ranks.add(server_rank)

                if not server_process.placement_group_bundle_idxs:
                    raise ValueError(f"Server worker_rank={server_rank} must own at least one placement-group bundle.")
                if len(set(server_process.placement_group_bundle_idxs)) != len(
                    server_process.placement_group_bundle_idxs
                ):
                    raise ValueError(
                        f"Server worker_rank={server_rank} has duplicate placement-group bundles: "
                        f"{server_process.placement_group_bundle_idxs}."
                    )
                duplicated_bundle_idxs = sorted(
                    bundle_idx
                    for bundle_idx in server_process.placement_group_bundle_idxs
                    if bundle_idx in seen_bundle_idxs
                )
                if duplicated_bundle_idxs:
                    raise ValueError(
                        f"Placement-group bundles are assigned to multiple server processes: {duplicated_bundle_idxs}."
                    )
                seen_bundle_idxs.update(server_process.placement_group_bundle_idxs)

                if server_process.nnodes < 1:
                    raise ValueError(f"Server worker_rank={server_rank} must have nnodes >= 1.")
                if server_process.node_rank < 0 or server_process.node_rank >= server_process.nnodes:
                    raise ValueError(
                        f"Server worker_rank={server_rank} has invalid node_rank={server_process.node_rank} "
                        f"for nnodes={server_process.nnodes}."
                    )

            if not engine_spec.request_entrypoint_servers:
                raise ValueError(f"EngineLaunchSpec[{engine_index}] must expose at least one request entrypoint.")

        if known_worker_rank_set is not None:
            missing_engine_ranks = sorted(known_worker_rank_set - seen_engine_ranks)
            if missing_engine_ranks:
                raise ValueError(
                    f"EngineLaunchSpecs do not cover known worker ranks in engine_ranks: {missing_engine_ranks}."
                )

        return engine_launch_specs

    @classmethod
    def build_engine_launch_specs(
        cls,
        config: RolloutConfig,
        rank_bundle_idx_list: list[tuple[int, int]],
        rank_to_dist_init_addr: dict[int, str] | None = None,
    ) -> EngineLaunchSpecs:
        """Build default launch spec: one request-serving server per engine."""
        num_gpus_per_engine = cls._get_num_gpus_per_engine(config)
        num_workers = len(rank_bundle_idx_list)
        if num_workers % num_gpus_per_engine != 0:
            raise ValueError(
                f"num_rollout_workers={num_workers} must be divisible by num_gpus_per_engine={num_gpus_per_engine}."
            )

        engine_launch_specs: list[EngineLaunchSpec] = []
        for engine_start in range(0, num_workers, num_gpus_per_engine):
            engine_meta = rank_bundle_idx_list[engine_start : engine_start + num_gpus_per_engine]
            engine_ranks = tuple(rank for rank, _ in engine_meta)
            engine_bundle_idxs = tuple(bundle_idx for _, bundle_idx in engine_meta)
            engine_dist_init_addr = None if rank_to_dist_init_addr is None else rank_to_dist_init_addr[engine_ranks[0]]
            engine_launch_specs.append(
                EngineLaunchSpec(
                    engine_ranks=engine_ranks,
                    server_processes=(
                        ServerProcessSpec(
                            worker_rank=engine_ranks[0],
                            placement_group_bundle_idxs=engine_bundle_idxs,
                            dist_init_addr=engine_dist_init_addr,
                        ),
                    ),
                )
            )
        return cls.validate_engine_launch_specs(
            tuple(engine_launch_specs),
            known_worker_ranks=tuple(rank for rank, _ in rank_bundle_idx_list),
        )

    @classmethod
    def build_metadata_engine_rank_mesh_array(
        cls,
        engine_launch_specs: EngineLaunchSpecs,
    ) -> list[list[int]]:
        """Build the public engine mesh returned in rollout metadata.

        By default, the public metadata mesh matches the logical engine topology. Backends with multiple request
        servers per logical engine can override this to preserve their legacy update-weight mesh semantics.
        """
        return [list(engine_spec.engine_ranks) for engine_spec in engine_launch_specs]

    def _get_current_server_process_spec(
        self,
        engine_launch_spec: EngineLaunchSpec | None = None,
    ) -> ServerProcessSpec | None:
        engine_launch_spec = engine_launch_spec or self.engine_launch_spec
        if engine_launch_spec is None:
            return None

        for server_process_spec in engine_launch_spec.server_processes:
            if server_process_spec.worker_rank == self.rank:
                return server_process_spec
        raise RuntimeError(
            f"Engine launch spec does not include rollout worker rank={self.rank} "
            f"in server_worker_ranks={engine_launch_spec.server_worker_ranks}."
        )

    def set_enable_partial_rollout(self, enable: bool) -> None:
        self.enable_partial_rollout = enable

    def init(
        self,
        *,
        engine_launch_spec: EngineLaunchSpec | None = None,
    ) -> tuple[int, str]:
        """Initialize the worker and launch the server.

        Returns:
            Tuple[int, str]: A tuple containing the worker's rank and its
                server URL.
        """
        if engine_launch_spec is not None:
            # Initial controller startup passes the immutable launch spec and caches
            # it on the actor. Recovery calls init() without arguments after
            # shutdown, intentionally reusing this cached placement/dist layout.
            self.engine_launch_spec = engine_launch_spec
            server_process_spec = cast(
                ServerProcessSpec,
                self._get_current_server_process_spec(engine_launch_spec),
            )
            self.engine_bundle_idxs = list(server_process_spec.placement_group_bundle_idxs)
            if server_process_spec.dist_init_addr is not None:
                self.dist_init_addr = server_process_spec.dist_init_addr
        self.receive_abort_request.clear()
        self._launch_server()
        self._start_session_server()
        return (self.rank, self.server_url)

    def set_skip_load_weights(self, skip_load_weights: bool) -> None:
        self.config = self.config.model_copy(update={"skip_load_weights": skip_load_weights})

    def restore_skip_load_weights(self) -> None:
        self.config = self.config.model_copy(update={"skip_load_weights": self._default_skip_load_weights})

    def init_dist_port(self) -> str:
        """Initialize distributed communication ports.

        This method initializes four fixed ports for the distributed setup:
        one for distributed communication, one for the inference server, one
        for NCCL, and one for the session server.

        Returns:
            str: The distributed initialization address (host:port).
        """
        local_rank = int(ray.get_runtime_context().get_accelerator_ids()[self.accelerator][0])
        base_port = self.config.dist_port_base + local_rank * 4
        self.host = ray.util.get_node_ip_address()
        self.dist_port = base_port
        self.server_port = base_port + 1
        self.nccl_port = base_port + 2
        self.session_server_port = base_port + 3
        self.dist_init_addr = f"{self.host}:{self.dist_port}"
        self.server_url = f"http://{self.host}:{self.server_port}"
        return self.dist_init_addr

    def shutdown(self, *, stop_session_server: bool = False):
        """Shut down the worker, its server task, and any child processes."""
        if stop_session_server:
            self._stop_session_server()

        if self.server_task is not None:
            server_task = self.server_task
            self._request_server_terminate()
            ray.cancel(server_task, force=True, recursive=True)
            try:
                ray.get(server_task, timeout=60)
            except ray.exceptions.GetTimeoutError:
                self.logger.warning(f"Worker {self.rank} server task did not stop within shutdown timeout.")
                raise
            except Exception as e:
                self.logger.debug(f"Worker {self.rank} server task stopped after shutdown: {e}")
            self.server_task = None
            return

        if self.server_process is not None:
            import psutil

            try:
                parent = psutil.Process(self.server_process.pid)
            except psutil.NoSuchProcess:
                self.server_process = None
                return
            children = parent.children(recursive=True)
            for child in children:
                child.terminate()
            gone, alive = psutil.wait_procs(children, timeout=5)
            for child in alive:
                child.kill()
            parent.terminate()
            parent.wait(timeout=5)
            self.server_process = None
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
                **with_trace_runtime_env(
                    {
                        "scheduling_strategy": scheduling_strategy,
                        "num_cpus": 0,
                    }
                )
            )
            .remote(
                worker_base_url=self.server_url,
                tokenizer_path=str(self.config.tokenizer_path or self.config.model_path),
                host=self.host,
                port=self.session_server_port,
                request_timeout=self.config.session_server_timeout,
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

    async def _send_abort_request(self) -> bool:
        url = f"{self.server_url}/abort_request"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json={"abort_all": True})
            response.raise_for_status()
            return True
        except Exception:
            return False

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
            health_url = f"{self.server_url}/{self.endpoints['health_generate']}"
            response = requests.get(
                health_url,
                headers=headers,
                timeout=self.config.health_check_timeout_seconds,
            )
            if response.status_code == 200:
                return True
            health_message = ""
            try:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("message"):
                    health_message = f", message={payload['message']!r}"
            except ValueError:
                pass
            self.logger.warning(
                f"Health check returned non-200 for server {health_url}: "
                f"status_code={response.status_code}{health_message}"
            )
            return False
        except requests.RequestException as e:
            self.logger.error(f"Health check failed for server {self.server_url}: {e}")
            return False

    async def _decode_routed_experts(self, routed_experts: Any) -> Any:
        return routed_experts

    @ray.method(concurrency_group=ROLLOUT_CONCURRENCY_GROUP_GENERATE)
    @traced_rollout_endpoint(TRACE_SPAN_ROLLOUT_WORKER_GENERATE)
    async def generate(self, rollout_state: RolloutState) -> RolloutState:
        request_max_tokens = rollout_state.sample_params.max_tokens
        try:
            # TODO(@duanyanhui):
            # 1. support claude format input
            # 2. 需要看下新的输入输出(RolloutState)怎么适配PartialRollout的逻辑，先跑起来
            # 3. 对于流式返回的response先删掉，目前还用不上，等需要的时候再加上

            if self.receive_abort_request.is_set():
                rollout_state.finish_reason = "abort"
                rollout_state.status = Status.ABORTED
                return rollout_state

            uid = rollout_state.rollout_id
            sample_params: SampleParams = rollout_state.sample_params
            if sample_params.return_token_ids:
                endpoint_url = f"{self.server_url}/{self.endpoints['generate']}"
            else:
                endpoint_url = f"{self.server_url}/{self.endpoints['v1/chat/completions']}"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            }

            rollout_state, payload = self._prepare_request_payload(rollout_state, request_max_tokens)
            max_retries = self.config.max_retry_per_sample

            # 早退逻辑 1：检查是否已被标记为完成
            if rollout_state.status == Status.COMPLETED:
                self.logger.debug(f"Request {uid} is already marked as COMPLETED, skipping generation.")
                return rollout_state

            # 早退逻辑 2：检测输入是否还需要 generation (安全获取变量)
            input_ids = payload.get("input_ids", [])
            payload_max_tokens = cast(int, payload.get("max_tokens"))

            last_id = input_ids[-1] if len(input_ids) > 0 else "None"
            is_max_tokens_zero = payload_max_tokens is not None and payload_max_tokens <= 0
            is_eos_reached = len(input_ids) > 0 and input_ids[-1] in self.eos_token

            if is_max_tokens_zero or is_eos_reached:
                self.logger.debug(
                    f"No generation needed for request {uid}: max_tokens={payload_max_tokens} or last input_id={last_id} is in eos_token."
                )
                finish_reason = "stop" if is_eos_reached else "length"
                # 对于是否开 partial rollout 的情况都直接标记为完成并返回，因为本轮 rollout 未开始，也不需要拼接
                rollout_state.finish_reason = finish_reason
                rollout_state.status = Status.COMPLETED
                return rollout_state

            for attempt in range(max_retries + 1):
                is_last_attempt = attempt == max_retries
                http_result = await self._safe_post_request(
                    endpoint_url,
                    headers=headers,
                    payload=payload,
                    rollout_state=rollout_state,
                )

                # Case 1: HTTP Request is Successful
                if http_result.response:
                    # Case 1.1: Valid rollout response
                    rollout_state = await self._safe_handle_response(rollout_state, http_result.response)
                    if self.receive_abort_request.is_set():
                        rollout_state.finish_reason = "abort"
                        rollout_state.status = Status.ABORTED
                        rollout_state.sample_params = rollout_state.sample_params.model_copy(
                            update={"max_tokens": request_max_tokens}
                        )
                        return rollout_state
                    if rollout_state.status == Status.COMPLETED:
                        return rollout_state
                    if rollout_state.status == Status.ABORTED:
                        rollout_state.sample_params = rollout_state.sample_params.model_copy(
                            update={"max_tokens": request_max_tokens}
                        )
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
                    rollout_state, payload = self._prepare_request_payload(
                        rollout_state, request_max_tokens, discard_response=True
                    )
                    await asyncio.sleep(0.1)
                    continue

                # Case 2: Error occurred during HTTP Request
                if http_result.error_type == HttpRequestErrorType.REQUEST_ABORTED:
                    # Case 2.1: The request was aborted due to an signal set by `receive_abort_request`
                    rollout_state.finish_reason = "abort"
                    rollout_state.status = update_status_from_finish_reason("abort")
                    rollout_state.sample_params = rollout_state.sample_params.model_copy(
                        update={"max_tokens": request_max_tokens}
                    )
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
                    rollout_state, payload = self._prepare_request_payload(
                        rollout_state, request_max_tokens, discard_response=True
                    )
                    await asyncio.sleep(0.1)
                    continue

                # Case 4: Unknown error occurred during HTTP Request and stop the rollout
                if http_result.is_unknown_error:
                    raise RuntimeError(
                        f"Unexpected error during rollout request {uid} to {http_result.url}: {http_result.exception}"
                    )
            return rollout_state
        finally:
            if rollout_state.status == Status.FAILED:
                error_msg = rollout_state.error_msg
                status = rollout_state.status
                reset_rollout_response(rollout_state)
                rollout_state.status = status
                rollout_state.error_msg = error_msg
                rollout_state.sample_params = rollout_state.sample_params.model_copy(
                    update={"max_tokens": request_max_tokens}
                )

    def _prepare_request_payload(
        self,
        rollout_state: RolloutState,
        request_max_tokens: int,
        *,
        discard_response: bool = False,
    ) -> tuple[RolloutState, dict]:
        """Prepare rollout state and payload for one generation request.

        Args:
            discard_response: Only used by retry paths. When true, the previous
                request's response is considered incomplete or invalid, so any
                response/logprob/routed-expert state already attached to
                ``rollout_state`` must be discarded before rebuilding the
                payload from the original prompt and the request entry
                ``max_tokens``.
        """
        if discard_response:
            rollout_state = reset_rollout_response(rollout_state)
            rollout_state.sample_params = rollout_state.sample_params.model_copy(
                update={"max_tokens": request_max_tokens}
            )
            rollout_state.status = Status.INIT
        elif not self.enable_partial_rollout and rollout_state.status == Status.ABORTED:
            # ABORTED samples can be replayed; without partial rollout, rerun from the original prompt.
            rollout_state = reset_rollout_response(rollout_state)
            rollout_state.sample_params = rollout_state.sample_params.model_copy(
                update={"max_tokens": request_max_tokens}
            )
            rollout_state.status = Status.INIT

        if self.enable_partial_rollout:
            rollout_state = self.partial_rollout_handler.preprocess(rollout_state, request_max_tokens)
        return rollout_state, self._get_request_payload(rollout_state)

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
            runtime_env = copy.deepcopy(getattr(server_configs, "ray_runtime_env", None)) or {}
            ray_options = with_trace_runtime_env(
                {
                    "scheduling_strategy": scheduling_strategy,
                    **AutoAcceleratorWorkers.get_pg_options(current_pg),
                    "runtime_env": runtime_env,
                }
            )
            self.server_task = (
                ray.remote(self.server_func)
                .options(
                    **ray_options,
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

    async def _safe_post_request(
        self,
        url,
        headers,
        payload,
        *,
        rollout_state: RolloutState,
    ) -> HttpRequestResult:
        try:
            if self.receive_abort_request.is_set():
                return HttpRequestResult(error_type=HttpRequestErrorType.REQUEST_ABORTED, url=url, payload=payload)
            trace_attributes = {
                TRACE_ATTR_ROLLOUT_BACKEND: self.config.rollout_backend,
                "http.method": "POST",
                "http.url": url,
            }
            trace_attributes.update(rollout_state_trace_attributes(rollout_state))
            request_headers = dict(headers)
            with trace_span("infer_engine.generate", attributes=trace_attributes):
                inject_trace_context(request_headers)
                req = self.client.build_request(
                    "POST",
                    url,
                    headers=request_headers,
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
        uid = rollout_state.group_id

        sample_params = rollout_state.sample_params
        is_token_out = sample_params.return_token_ids
        response = http_response.json()

        if is_token_out:
            response_ids: list[int] = []
            logprobs: list[float] = []
            routed_experts = None
            returned_response = ""
            should_return_routed_experts = self.enable_return_routed_experts and sample_params.return_routed_experts
            try:
                meta_info = response.get("meta_info") or {}
                finish_reason_info = meta_info.get("finish_reason") or {}
                finish_reason = finish_reason_info.get("type")
                if finish_reason is None:
                    if self.receive_abort_request.is_set():
                        rollout_state.finish_reason = "abort"
                        rollout_state.status = Status.ABORTED
                        self.logger.warning(
                            f"finish_reason is missing in response meta_info when waiting for aborted message {uid}, defaulting to 'abort'. Response: {response}"
                        )
                    else:
                        rollout_state.finish_reason = "error"
                        rollout_state.status = Status.FAILED
                        self.logger.warning(
                            f"finish_reason is missing in response meta_info for message {uid}, defaulting to 'error'. Response: {response}"
                        )
                    rollout_state.error_msg = "Missing finish_reason in response meta_info"
                    return rollout_state
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
                if should_return_routed_experts:
                    assert "routed_experts" in response["meta_info"], (
                        "enable_return_routed_experts is True, but routed_experts is not in meta_info"
                    )
                    routed_experts = response["meta_info"]["routed_experts"]  # token[layer[expert]]
                    if routed_experts is not None:
                        routed_experts = await self._decode_routed_experts(routed_experts)
                        if not isinstance(routed_experts, ray.ObjectRef):
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

                    if should_return_routed_experts and routed_experts is None:
                        validation_errors.append("missing routed_experts")

                    if validation_errors:
                        error_msg = f"Incomplete rollout data for msg {uid}: {', '.join(validation_errors)}"
                        self.logger.error(error_msg)
                        rollout_state.routed_experts = routed_experts
                        rollout_state.status = Status.FAILED
                        rollout_state.error_msg = error_msg
                        return rollout_state
                elif rollout_status == Status.FAILED:
                    error_msg = f"Rollout failed for msg {uid} with finish_reason {finish_reason}"
                    self.logger.error(error_msg)
                    rollout_state.routed_experts = routed_experts
                    rollout_state.status = Status.FAILED
                    rollout_state.error_msg = error_msg
                    return rollout_state

                if self.enable_partial_rollout:
                    prompt_tokens = response["meta_info"]["prompt_tokens"]
                    completion_tokens = response["meta_info"]["completion_tokens"]
                    rollout_state = await self.partial_rollout_handler.postprocess(
                        rollout_state,
                        response=returned_response,
                        response_ids=response_ids,
                        logprobs=logprobs,
                        routed_experts=routed_experts,
                        finish_reason=finish_reason,
                        status=rollout_status,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                else:
                    rollout_state.response = returned_response
                    rollout_state.response_ids = response_ids
                    rollout_state.logprobs = logprobs
                    rollout_state.routed_experts = routed_experts
                    rollout_state.finish_reason = finish_reason
                    rollout_state.status = rollout_status
                return rollout_state
            except KeyError as e:
                response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
                error_msg = f"Missing expected key {e} in response {response_for_log} for {uid}"
                raise RuntimeError(error_msg)
            except IndexError as e:
                response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
                error_msg = f"Index error {e} while processing response {response_for_log} for {uid}"
                raise RuntimeError(error_msg)
            except AssertionError as e:
                response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
                error_msg = f"AssertionError: {e} when processing response {response_for_log} for {uid}"
                raise RuntimeError(error_msg)
            except json.JSONDecodeError as e:
                error_msg = f"JSONDecodeError: {e} when processing response {response} for {uid}"
                raise RuntimeError(error_msg)
            except TypeError as e:
                response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
                error_msg = f"TypeError: {e} when processing response {response_for_log} for {uid}"
                raise RuntimeError(error_msg)
            except Exception as e:
                response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
                error_msg = f"Unexpected error: {e} when processing response {response_for_log} for {uid}\nTraceback: {traceback.format_exc()}"
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
                response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
                error_msg = f"Missing expected key {e} in response {response_for_log} for {uid}"
                raise RuntimeError(error_msg)
            except IndexError as e:
                response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
                error_msg = f"Index error {e} while processing response {response_for_log} for {uid}"
                raise RuntimeError(error_msg)
            except AssertionError as e:
                response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
                error_msg = f"AssertionError: {e} when processing response {response_for_log} for {uid}"
                raise RuntimeError(error_msg)
            except json.JSONDecodeError as e:
                error_msg = f"JSONDecodeError: {e} when processing response {response} for {uid}"
                raise RuntimeError(error_msg)
            except TypeError as e:
                response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
                error_msg = f"TypeError: {e} when processing response {response_for_log} for {uid}"
                raise RuntimeError(error_msg)
            except Exception as e:
                response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
                error_msg = f"Unexpected error: {e} when processing response {response_for_log} for {uid}\nTraceback: {traceback.format_exc()}"
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

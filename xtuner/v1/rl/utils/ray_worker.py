import os
from typing import Any, Dict, List, Literal, Tuple, TypeVar

import ray
import torch
import torch.distributed as dist
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, field_validator
from ray.actor import ActorClass, ActorProxy
from ray.util.placement_group import (
    VALID_PLACEMENT_GROUP_STRATEGIES,
    PlacementGroup,
    placement_group,
    placement_group_table,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from typing_extensions import Annotated

from .ray_utils import find_master_addr_and_port, get_accelerator_ids


PG_READY_TIMEOUT = os.getenv("XTUNER_PG_READY_TIMEOUT", 30)  # default 30 seconds
AcceleratorType = Literal["GPU", "NPU"]
T = TypeVar("T")


class CPUResourcesConfig(BaseModel):
    """Configuration for CPU resources in a placement group for XTuner.

    This class provide specific configuration options for CPU-based workers in Ray placement groups.

    Args:
        num_cpus_per_worker (float): Number of CPUs to allocate per worker in the
            placement group. Defaults to 8.
        cpu_memory_per_worker (int): Amount of CPU memory (in bytes) to allocate
            for each worker in the placement group.
        num_workers (int): Total number of workers in the placement group.
    """

    model_config = ConfigDict(extra="forbid")
    num_workers: Annotated[int, Parameter(help="Number of workers in the placement group.")] = 1
    num_cpus_per_worker: Annotated[float, Parameter(help="Number of CPUs to allocate for the placement group.")] = 1
    cpu_memory_per_worker: Annotated[
        int, Parameter(help="Amount of memory (in bytes) to allocate for the placement group.")
    ] = 1024**3  # 1 GB
    pg_pack_strategy: Annotated[
        str,
        Parameter(help="Placement group packing strategy, options: " + ", ".join(VALID_PLACEMENT_GROUP_STRATEGIES)),
    ] = "SPREAD"

    @field_validator("pg_pack_strategy")
    @classmethod
    def check_pg_pack_strategy(cls, v):
        if v not in VALID_PLACEMENT_GROUP_STRATEGIES:
            raise ValueError(f"pg_pack_strategy must be one of {VALID_PLACEMENT_GROUP_STRATEGIES}")
        return v

    def model_post_init(self, __context: Any) -> None:
        assert ray.is_initialized(), "Ray must be initialized before creating CPUResourcesConfig."
        available_resources = ray.available_resources()
        available_cpus = available_resources.get("CPU", 0)
        available_memory = available_resources.get("memory", 0)
        # TODO: manage single controller's cpu resource to replace "10" here
        needed_cpus = (self.num_cpus_per_worker * self.num_workers) + 10
        assert needed_cpus <= available_cpus, (
            f"Not enough available CPUs in Ray cluster, available_cpus is {available_cpus} but xtuner needs {needed_cpus}."
        )
        needed_memory = self.cpu_memory_per_worker * self.num_workers + 10 * 1024**3
        assert needed_memory <= available_memory, (
            f"Not enough available memory in Ray cluster, available_memory is {available_memory} but xtuner needs {needed_memory}."
        )
        # TODO: check all resources sum in cluster to avoid over allocation

    @classmethod
    def from_total(
        cls, total_cpus: float | int, total_memory: int, num_workers: int, pg_pack_strategy: str = "SPREAD"
    ):
        """Create a CPUResourcesConfig from total CPU and memory resources.

        Args:
            total_cpus (float | int): Total number of CPUs to allocate across all workers.
            total_memory (int): Total amount of memory (in bytes) to allocate across all workers.
            num_workers (int): Number of workers in the placement group.

        Returns:
            CPUResourcesConfig: The created CPUResourcesConfig object.
        """
        assert num_workers > 0, "Number of workers must be positive."
        return cls(
            num_workers=num_workers,
            num_cpus_per_worker=total_cpus / num_workers,
            cpu_memory_per_worker=total_memory / num_workers,
            pg_pack_strategy=pg_pack_strategy,
        )

    def build_placement_group(self) -> PlacementGroup:
        """Build a Ray PlacementGroup based on this resource configuration.

        Returns:
            PlacementGroup: The created Ray PlacementGroup.
        """
        return CPUActorLauncher.build_placement_group(self)


class AcceleratorResourcesConfig(BaseModel):
    """Configuration for accelerator resources in a placement group for XTuner.

    This class defines the fundamental configuration parameters for managing
    accelerator resources in Ray placement groups, including resource allocation
    per worker, memory management, and accelerator type specification. It provides
    a unified interface for controlling distributed training resource allocation
    and hardware utilization.

    Args:
        num_accelerators_per_worker (float): Number of accelerators to allocate for
            each worker in the placement group. Defaults to 1.
        num_cpus_per_worker (float): Number of CPUs to allocate per worker in the
            placement group. Defaults to 8.
        cpu_memory_per_worker (int): Amount of CPU memory (in bytes) to allocate
            for each worker in the placement group.
        num_workers (int): Total number of workers in the placement group.
        accelerator (AcceleratorType): Type of accelerator architecture to use
            (e.g., 'GPU', 'NPU').

    **Examples:**

    Example configuration for resources::

        resources = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_accelerators_per_worker=1,
            num_cpus_per_worker=12,
            num_workers=args.num_workers,
            cpu_memory_per_worker=16 * 1024**3,  # 16 GB
        )
    """

    model_config = ConfigDict(extra="forbid")
    accelerator: Annotated[
        AcceleratorType, Parameter(help="Architecture of accelerator to use (e.g., 'GPU', 'NPU').")
    ] = "GPU"
    num_workers: Annotated[int, Parameter(help="Number of accelerators in the placement group.")]
    num_cpus_per_worker: Annotated[float, Parameter(help="Number of CPUs to allocate for the placement group.")] = 12
    cpu_memory_per_worker: Annotated[
        int, Parameter(help="Amount of memory (in bytes) to allocate for the placement group.")
    ] = 16 * 1024**3  # 16 GB
    num_accelerators_per_node: Annotated[int, Parameter(help="Number of accelerators available per node.")] = 8
    num_accelerators_per_worker: Annotated[
        float,
        Parameter(help="Number of accelerators to allocate for each worker in the placement group."),
    ] = 1
    pg_pack_strategy: Annotated[
        str,
        Parameter(help="Placement group packing strategy, options: " + ", ".join(VALID_PLACEMENT_GROUP_STRATEGIES)),
    ] = "PACK"

    @field_validator("pg_pack_strategy")
    @classmethod
    def check_pg_pack_strategy(cls, v):
        if v not in VALID_PLACEMENT_GROUP_STRATEGIES:
            raise ValueError(f"pg_pack_strategy must be one of {VALID_PLACEMENT_GROUP_STRATEGIES}")
        return v

    def model_post_init(self, __context: Any) -> None:
        if self.accelerator == "NPU":
            # NOTE: Ascend 910 has 16 NPUs per node
            self.num_accelerators_per_node = 16

        assert ray.is_initialized(), "Ray must be initialized before creating AcceleratorResourcesConfig."

        available_resources = ray.available_resources()
        available_cpus = available_resources.get("CPU", 0)
        available_memory = available_resources.get("memory", 0)
        if self.accelerator == "GPU":
            available_gpus = available_resources.get("GPU", 0)
            assert self.num_workers <= available_gpus, (
                f"Not enough available GPUS in Ray cluster, {available_gpus} less than {self.num_workers}."
            )
        else:  # NPU
            available_npus = available_resources.get("NPU", 0)
            assert self.num_workers <= available_npus, (
                f"Not enough available NPUS in Ray cluster, {available_npus} less than {self.num_workers}."
            )
        # TODO: manage single controller's cpu resource to replace "10" here
        needed_cpu = self.num_cpus_per_worker * self.num_workers + 10
        assert needed_cpu <= available_cpus, (
            f"Not enough available CPUs in Ray cluster, {available_cpus} less than {needed_cpu}."
        )
        needed_memory = self.cpu_memory_per_worker * self.num_workers + 10 * 1024**3
        assert needed_memory <= available_memory, (
            f"Not enough available memory in Ray cluster, {available_memory} less than {needed_memory}."
        )

    @classmethod
    def from_total(
        cls,
        accelerator: AcceleratorType,
        total_accelerators: float | int,
        total_cpus: float | int,
        total_memory: int,
        num_workers: int,
        pg_pack_strategy: str = "PACK",
    ):
        """Create an AcceleratorResourcesConfig from total accelerator, CPU,
        and memory resources.

        Args:
            accelerator (AcceleratorType): Type of accelerator architecture to use
                (e.g., 'GPU', 'NPU').
            total_accelerators (float | int): Total number of accelerators to allocate
                across all workers.
            total_cpus (float | int): Total number of CPUs to allocate across all workers.
            total_memory (int): Total amount of memory (in bytes) to allocate across all workers.
            num_workers (int): Number of workers in the placement group.

        Returns:
            AcceleratorResourcesConfig: A configuration object with the specified resource allocations.
        """
        return cls(
            accelerator=accelerator,
            num_workers=num_workers,
            num_accelerators_per_worker=total_accelerators / num_workers,
            num_cpus_per_worker=total_cpus / num_workers,
            cpu_memory_per_worker=total_memory / num_workers,
            pg_pack_strategy=pg_pack_strategy,
        )


class SingleAcceleratorWorker:
    """A base class for a worker that utilizes a single accelerator and
    initializes a distributed process group."""

    def __init__(
        self,
        config,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        """Initialize the SingleAcceleratorWorker.

        Args:
            config: The configuration object for the worker.
            rank (int): The rank of this worker in the distributed world.
            master_addr (str): The address of the master node for distributed
                initialization.
            master_port (int): The port of the master node.
            world_size (int): The total number of workers in the distributed
                world.
            accelerator (str): The type of accelerator being used ('GPU' or
                'NPU'). Defaults to "GPU".
        """
        self.config = config
        self.accelerator = accelerator
        self.setup_distributed(rank, master_addr, master_port, world_size)

    @property
    def device_visible_env_name(self):
        """Get the environment variable name for device visibility based on the
        accelerator type.

        Returns:
            str: The name of the environment variable (e.g.,
                'CUDA_VISIBLE_DEVICES').

        Raises:
            ValueError: If the accelerator type is unsupported.
        """
        if self.accelerator == "GPU":
            return "CUDA_VISIBLE_DEVICES"
        elif self.accelerator == "NPU":
            return "ASCEND_RT_VISIBLE_DEVICES"
        else:
            raise ValueError(f"Unsupported accelerator type: {self.accelerator}")

    def get_logical_local_rank(self) -> int:
        """Resolve the assigned accelerator id to the logical local rank.

        Ray reports accelerator ids in the physical numbering space. Torch selects devices from the current visible-
        device list, which is indexed logically from zero after applying visibility masks.
        """
        accelerator_id = str(ray.get_runtime_context().get_accelerator_ids()[self.accelerator][0])
        visible_devices = os.environ.get(self.device_visible_env_name)
        if visible_devices is None:
            return int(accelerator_id)

        visible_device_ids = [device_id.strip() for device_id in visible_devices.split(",") if device_id.strip()]
        if accelerator_id not in visible_device_ids:
            raise ValueError(
                f"Assigned accelerator id {accelerator_id} is not present in "
                f"{self.device_visible_env_name}={visible_devices}."
            )
        return visible_device_ids.index(accelerator_id)

    def setup_distributed(self, rank: int, master_addr: str, master_port: int, world_size: int):
        """Set up the distributed environment for the worker.

        This method configures the necessary environment variables and
        initializes the `torch.distributed` process group.

        Args:
            rank (int): The rank of this worker.
            master_addr (str): The address of the master node.
            master_port (int): The port of the master node.
            world_size (int): The total number of workers.

        Raises:
            ValueError: If the accelerator architecture is unsupported.
        """
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(self.get_logical_local_rank())

        # backend 参数是指定通信后端，不是从环境变量获取
        # - 'nccl': NVIDIA GPU 间通信（推荐用于 GPU）
        # - 'gloo': CPU 通信或跨平台
        # - 'mpi': 需要 MPI 环境
        if self.accelerator == "GPU":
            backend = "nccl"
        elif self.accelerator == "NPU":
            backend = "hccl"
        else:
            raise ValueError(f"Unsupported accelerator architecture: {self.accelerator}")
        # 使用环境变量初始化
        dist.init_process_group(
            backend=backend,
            init_method="env://",  # 这告诉 PyTorch 从环境变量读取配置
        )

    def test_all_reduce(self):
        """Perform a test all-reduce operation to verify distributed setup.

        Returns:
            torch.Tensor: The tensor after the all-reduce operation.
        """
        tensor = torch.tensor([1.0], device=torch.accelerator.current_accelerator())
        dist.all_reduce(tensor)
        return tensor


class AutoAcceleratorWorkers:
    """A utility class for automatically creating and managing distributed
    workers on accelerators within a Ray PlacementGroup."""

    @staticmethod
    def build_placement_group(resources_config: AcceleratorResourcesConfig, name="train"):
        """Build a Ray PlacementGroup based on the provided resource
        configuration.

        Args:
            resources_config (AcceleratorResourcesConfig): The configuration
                specifying the resources for each worker bundle.

        Returns:
            PlacementGroup: The created Ray PlacementGroup.
        """
        bundles = [
            {
                "CPU": resources_config.num_cpus_per_worker,
                "memory": resources_config.cpu_memory_per_worker,
                resources_config.accelerator: resources_config.num_accelerators_per_worker,
            }
        ] * resources_config.num_workers

        pg_info = ray.util.placement_group_table()
        names = [i["name"] for i in pg_info.values() if i.get("state") not in ["REMOVED", "REMOVING"]]

        if name in names:
            pg = ray.util.get_placement_group(name)
        else:
            pg = placement_group(bundles=bundles, strategy=resources_config.pg_pack_strategy, name=name)
            ray.get(pg.ready(), timeout=PG_READY_TIMEOUT)
        return pg

    @staticmethod
    def get_device_type(pg: PlacementGroup) -> AcceleratorType:
        """Determine the type of accelerator used in a PlacementGroup.

        Args:
            pg (PlacementGroup): The placement group to inspect.

        Returns:
            AcceleratorType: The type of accelerator ('GPU' or 'NPU').

        Raises:
            ValueError: If mixed accelerator types are detected or no
                accelerators are found.
        """
        bundles = pg.bundle_specs
        if all("GPU" in bundle for bundle in bundles):
            return "GPU"
        elif all("NPU" in bundle for bundle in bundles):
            return "NPU"
        elif any("GPU" in bundle for bundle in bundles) or any("NPU" in bundle for bundle in bundles):
            raise ValueError("Mixed accelerator types detected in the placement group.")
        else:
            raise ValueError("No accelerators found in the placement group.")

    @staticmethod
    def get_pg_options(pg: PlacementGroup) -> Dict:
        """Provide a dictionary of resource requests for Ray tasks or actors
        that need to be scheduled on a node with a specific accelerator.

        By requesting a fractional amount (e.g., 0.01) of a CPU and an accelerator,
        it ensures the task is co-located with the target hardware without
        blocking the main workload from acquiring the full accelerator.

        Args:
            pg (PlacementGroup): The placement group to get options for.

        Returns:
            Dict: A dictionary of Ray resource options for `task.options()`.

        Raises:
            ValueError: If the accelerator architecture is unsupported.
        """
        accelerator = AutoAcceleratorWorkers.get_device_type(pg)

        if accelerator == "GPU":
            return {"num_cpus": 0.01, "num_gpus": 0.01}
        elif accelerator == "NPU":
            return {"num_cpus": 0.01, "resources": {"NPU": 0.01}}
        else:
            raise ValueError(f"Unsupported accelerator architecture: {accelerator}")

    @staticmethod
    def get_spmd_info(pg: PlacementGroup) -> Tuple[List[int], str, int, int]:
        """Get SPMD (Single Program, Multiple Data) info from the placement
        group.

        This includes the sorted bundle indices, master address, master port,
        and world size, which are essential for initializing a distributed
        process group.

        Args:
            pg (PlacementGroup): The placement group.

        Returns:
            Tuple[List[int], str, int, int]: A tuple containing:
                - sorted_bundle_idxs (List[int]): The bundle indices sorted by
                  node and local rank.
                - master_addr (str): The address of the master worker (rank 0).
                - master_port (int): The port for distributed communication.
                - world_size (int): The total number of workers.

        Raises:
            RuntimeError: If Ray is not initialized.
            AssertionError: If a bundle does not have exactly one accelerator.
        """
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Please initialize Ray before calling this method.")
        accelerator = AutoAcceleratorWorkers.get_device_type(pg)
        pg_options = AutoAcceleratorWorkers.get_pg_options(pg)
        pg_info = placement_group_table(pg)
        bundles_to_node = pg_info.get("bundles_to_node_id", {})

        node_accelerator_infos: Dict[str, Dict[int, int]] = {}

        for bundle_idx, node_id in bundles_to_node.items():
            accelerator_ids_ref = get_accelerator_ids.options(
                placement_group=pg, placement_group_bundle_index=int(bundle_idx), **pg_options
            ).remote(accelerator)

            accelerator_ids = ray.get(accelerator_ids_ref)
            assert len(accelerator_ids) == 1, "Expected exactly one accelerator ID per bundle."

            local_rank = int(accelerator_ids[0])

            if node_id not in node_accelerator_infos:
                node_accelerator_infos[node_id] = {}
            node_accelerator_infos[node_id][local_rank] = bundle_idx

        sorted_bundle_idxs = []
        for node_id, infos in node_accelerator_infos.items():
            for rank in sorted(infos.keys()):
                bundle_idx = infos[rank]
                sorted_bundle_idxs.append(bundle_idx)

        master_addr, master_port = ray.get(
            find_master_addr_and_port.options(
                placement_group=pg, placement_group_bundle_index=sorted_bundle_idxs[0], **pg_options
            ).remote()
        )

        world_size = len(sorted_bundle_idxs)

        return sorted_bundle_idxs, master_addr, master_port, world_size

    @classmethod
    def from_config(cls, worker_cls, worker_config, accelerator_config: AcceleratorResourcesConfig):
        """Create workers and a placement group from configuration objects.

        Args:
            worker_cls: The class of the worker to instantiate.
            worker_config: The configuration for each worker instance.
            accelerator_config (AcceleratorResourcesConfig): The configuration
                for the accelerator resources.

        Returns:
            Tuple[List[T], List[Tuple[int, int]], PlacementGroup]: A tuple containing a list
                of worker instances, a list of their corresponding
                (rank, bundle_index) and placement group.
        """
        pg = AutoAcceleratorWorkers.build_placement_group(accelerator_config)
        workers_list, rank_bundle_idx_list = cls.from_placement_group(worker_cls, worker_config, pg)

        return workers_list, rank_bundle_idx_list, pg

    @classmethod
    def from_placement_group(
        cls, worker_cls: ActorClass[T], worker_config, pg: PlacementGroup
    ) -> tuple[list[ActorProxy[T]], list[tuple[int, int]]]:
        """Create workers from an existing placement group.

        Args:
            worker_cls: The class of the worker to instantiate.
            worker_config: The configuration for each worker instance.
            pg (PlacementGroup): The existing placement group to use.

        Returns:
            Tuple[List[T], List[Tuple[int, int]]]: A tuple containing a list
                of worker instances and a list of their corresponding
                (rank, bundle_index).
        """
        pg_options = cls.get_pg_options(pg)
        device_type = cls.get_device_type(pg)
        sorted_bundle_idxs, master_addr, master_port, world_size = cls.get_spmd_info(pg)

        workers_list: list[ActorProxy[T]] = []
        rank_bundle_idx_list: list[tuple[int, int]] = []
        for rank, bundle_idx in enumerate(sorted_bundle_idxs):
            worker = worker_cls.options(
                placement_group=pg,
                placement_group_bundle_index=bundle_idx,
                **pg_options,
            ).remote(worker_config, rank, master_addr, master_port, world_size, device_type)
            workers_list.append(worker)
            rank_bundle_idx_list.append((rank, bundle_idx))

        return workers_list, rank_bundle_idx_list


class BaseCPUWorker:
    """The BaseCPUWorker class serves as a foundational structure for CPU-based
    workers within the XTuner framework.

    This class is designed to be extended by specific CPU worker implementations.
    It provides a constructor that accepts a configuration object, allowing
    subclasses to initialize with custom settings.

    Args:
        config: The configuration object for the CPU worker.
        num_cpus (float | int): The number of CPUs allocated to this worker.
            Defaults to 1.
    """

    def __init__(self, config, num_cpus: float | int = 1):
        self.config = config
        self.num_cpus = num_cpus


class CPUActorLauncher:
    """Infrastructure for launching CPU Ray actors from plain Python classes.

    This class owns the generic actorization flow for CPU-only components:
    building homogeneous CPU placement groups, converting plain classes into
    Ray actor classes, validating bundle resources, and launching one or more
    actors on specific bundles.
    """

    _ACTOR_CLASS_CACHE: dict[type, ActorClass] = {}

    @staticmethod
    def build_placement_group(resources_config: CPUResourcesConfig):
        """Build a Ray PlacementGroup based on the provided resource
        configuration.

        Args:
            resources_config (CPUResourcesConfig): The configuration
                specifying the resources for each worker bundle.

        Returns:
            PlacementGroup: The created Ray PlacementGroup.
        """
        bundles = [
            {
                "CPU": resources_config.num_cpus_per_worker,
                "memory": resources_config.cpu_memory_per_worker,
            }
        ] * resources_config.num_workers

        pg = placement_group(bundles=bundles, strategy=resources_config.pg_pack_strategy)

        ray.get(pg.ready(), timeout=PG_READY_TIMEOUT)
        return pg

    @staticmethod
    def get_pg_options(pg: PlacementGroup, num_cpus: int | float = -1) -> Dict:
        """Provide a dictionary of resource requests for Ray tasks or actors
        with specific cpu requirements.

        Args:
            pg (PlacementGroup): The placement group to get options for.
            num_cpus (float): The number of CPUs to request. If set to -1,
                the default CPU allocation from the placement group bundle
                will be used. Defaults to -1.

        Returns:
            Dict: A dictionary of Ray resource options for `task.options()`.
        """
        assert len(pg.bundle_specs) > 0, "Placement group has no bundles defined."
        default_cpu = pg.bundle_specs[0].get("CPU", 1)
        return {"num_cpus": num_cpus if num_cpus >= 0 else default_cpu}

    @classmethod
    def to_actor_class(cls, worker_cls):
        """Convert a plain Python class into a Ray actor class.

        If ``worker_cls`` is already a Ray actor class, it is returned as-is.
        """
        if hasattr(worker_cls, "remote") and hasattr(worker_cls, "options"):
            return worker_cls

        if worker_cls not in cls._ACTOR_CLASS_CACHE:
            cls._ACTOR_CLASS_CACHE[worker_cls] = ray.remote(worker_cls)
        return cls._ACTOR_CLASS_CACHE[worker_cls]

    @staticmethod
    def _get_bundle_resources(pg: PlacementGroup, bundle_idx: int) -> dict[str, float | int]:
        assert len(pg.bundle_specs) > bundle_idx, f"Placement group does not have bundle index {bundle_idx}."
        return pg.bundle_specs[bundle_idx]

    @classmethod
    def _resolve_actor_resources(
        cls,
        pg: PlacementGroup,
        bundle_idx: int,
        actor_num_cpus: int | float | None = None,
        actor_memory: int | None = None,
    ) -> tuple[float | int, int]:
        bundle = cls._get_bundle_resources(pg, bundle_idx)
        resolved_num_cpus = actor_num_cpus if actor_num_cpus is not None else bundle.get("CPU", 1)
        resolved_memory = actor_memory if actor_memory is not None else int(bundle.get("memory", 0))
        assert bundle.get("CPU", 1) >= resolved_num_cpus, (
            f"Placement group bundle {bundle_idx} does not have enough CPU resources."
        )
        assert bundle.get("memory", 0) >= resolved_memory, (
            f"Placement group bundle {bundle_idx} does not have enough memory resources."
        )
        return resolved_num_cpus, resolved_memory

    @classmethod
    def build_actor(
        cls,
        worker_cls,
        *init_args,
        pg: PlacementGroup | None = None,
        bundle_idx: int = 0,
        actor_num_cpus: int | float | None = None,
        actor_memory: int | None = None,
        pg_pack_strategy: str = "SPREAD",
        capture_child_tasks: bool = False,
        **init_kwargs,
    ):
        """Build a single CPU actor from a plain class or Ray actor class."""
        resolved_num_cpus = 1 if actor_num_cpus is None else actor_num_cpus
        resolved_memory = 1024**3 if actor_memory is None else actor_memory

        actor_cls = cls.to_actor_class(worker_cls)
        actor_options = {
            "num_cpus": resolved_num_cpus,
        }
        if resolved_memory > 0:
            actor_options["memory"] = resolved_memory

        if pg is None:
            return actor_cls.options(**actor_options).remote(*init_args, **init_kwargs)

        resolved_num_cpus, resolved_memory = cls._resolve_actor_resources(
            pg=pg,
            bundle_idx=bundle_idx,
            actor_num_cpus=actor_num_cpus,
            actor_memory=actor_memory,
        )
        actor_options["num_cpus"] = resolved_num_cpus
        if resolved_memory > 0:
            actor_options["memory"] = resolved_memory
        actor_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundle_idx,
            placement_group_capture_child_tasks=capture_child_tasks,
        )
        return actor_cls.options(**actor_options).remote(*init_args, **init_kwargs)

    @classmethod
    def build_actors(
        cls,
        worker_cls,
        *init_args,
        pg: PlacementGroup | None = None,
        start_bundle_idx: int = 0,
        num_workers: int = 1,
        actor_num_cpus_per_worker: int | float | None = None,
        actor_memory_per_worker: int | None = None,
        pg_pack_strategy: str = "SPREAD",
        capture_child_tasks: bool = False,
        **init_kwargs,
    ):
        """Build multiple homogeneous CPU actors from a plain class or Ray
        actor class."""
        workers_list = []
        for idx in range(num_workers):
            workers_list.append(
                cls.build_actor(
                    worker_cls,
                    *init_args,
                    pg=pg,
                    bundle_idx=start_bundle_idx + idx,
                    actor_num_cpus=actor_num_cpus_per_worker,
                    actor_memory=actor_memory_per_worker,
                    capture_child_tasks=capture_child_tasks,
                    **init_kwargs,
                )
            )
        return workers_list


class AutoCPUWorkers(CPUActorLauncher):
    """Convenience wrapper for BaseCPUWorker-style homogeneous worker pools.

    `CPUActorLauncher` is the generic actorization layer. `AutoCPUWorkers`
    keeps the legacy worker-centric API that instantiates one worker per bundle
    using the conventional `(worker_config, num_cpus=...)` constructor shape.
    """

    @classmethod
    def from_config(cls, worker_cls, worker_config, cpu_config: CPUResourcesConfig):
        """Create workers and a placement group from configuration objects.

        Args:
            worker_cls: The class of the worker to instantiate.
            worker_config: The configuration for each worker instance.
            cpu_config (CPUResourcesConfig): The configuration
                for the cpu resources.

        Returns:
            List[T]: List of created worker instances.
        """
        pg = cls.build_placement_group(cpu_config)
        workers_list = cls.from_placement_group(worker_cls, worker_config, pg)

        return workers_list, pg

    @classmethod
    def from_placement_group(
        cls,
        worker_cls,
        worker_config,
        pg: PlacementGroup,
        num_workers: int = -1,
        start_bundle_idx: int = 0,
    ):
        """Create workers from an existing placement group.

        Args:
            worker_cls: The class of the worker to instantiate.
            worker_config: The configuration for each worker instance.
            pg (PlacementGroup): The existing placement group to use.
            num_workers (int): The number of workers to create. Defaults to -1,
                the number of bundles in the placement group will be used.

        Returns:
            List[T]: List of created worker instances.
        """
        num_workers = num_workers if num_workers > 0 else pg.bundle_count - start_bundle_idx
        default_cpu = cls._get_bundle_resources(pg, start_bundle_idx).get("CPU", 1)
        return cls.build_actors(
            worker_cls,
            worker_config,
            num_cpus=default_cpu,
            pg=pg,
            start_bundle_idx=start_bundle_idx,
            num_workers=num_workers,
            actor_num_cpus_per_worker=default_cpu,
            actor_memory_per_worker=None,
        )

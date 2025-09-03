import os
from typing import Dict, List, Literal, Tuple, TypeVar

import ray
import torch
import torch.distributed as dist
from cyclopts import Parameter
from pydantic import BaseModel
from ray.util.placement_group import PlacementGroup, placement_group, placement_group_table
from typing_extensions import Annotated

from .utils import find_master_addr_and_port, get_accelerator_ids


AcceleratorType = Literal["GPU", "NPU"]
T = TypeVar("T")


class AcceleratorResourcesConfig(BaseModel):
    """Configuration for accelerator resources in a placement group."""

    num_accelerators_per_worker: Annotated[
        float,
        Parameter(help="Number of accelerators to allocate for each worker in the placement group."),
    ] = 1

    num_cpus_per_worker: Annotated[float, Parameter(help="Number of CPUs to allocate for the placement group.")] = 8

    cpu_memory_per_worker: Annotated[
        int, Parameter(help="Amount of memory (in bytes) to allocate for the placement group.")
    ]

    num_workers: Annotated[int, Parameter(help="Number of accelerators in the placement group.")]

    accelerator: Annotated[AcceleratorType, Parameter(help="Architecture of accelerator to use (e.g., 'GPU', 'NPU').")]


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
        os.environ["LOCAL_RANK"] = str(ray.get_runtime_context().get_accelerator_ids()[self.accelerator][0])

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
    def build_placement_group(resources_config: AcceleratorResourcesConfig):
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

        pg = placement_group(bundles=bundles, strategy="PACK", name="train")

        ray.get(pg.ready())
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
            for rank in range(len(infos.keys())):
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
            Tuple[Dict[T, Tuple[int, int]], PlacementGroup]: A tuple
                containing a map of worker instances to their (rank,
                bundle_index) and the created placement group.
        """
        pg = AutoAcceleratorWorkers.build_placement_group(accelerator_config)
        workers_bundle_idx_map = cls.from_placement_group(worker_cls, worker_config, pg)

        return workers_bundle_idx_map, pg

    @classmethod
    def from_placement_group(cls, worker_cls, worker_config, pg: PlacementGroup):
        """Create workers from an existing placement group.

        Args:
            worker_cls: The class of the worker to instantiate.
            worker_config: The configuration for each worker instance.
            pg (PlacementGroup): The existing placement group to use.

        Returns:
            Dict[T, Tuple[int, int]]: A map of worker instances to their
                (rank, bundle_index).
        """
        pg_options = cls.get_pg_options(pg)
        device_type = cls.get_device_type(pg)
        sorted_bundle_idxs, master_addr, master_port, world_size = cls.get_spmd_info(pg)

        workers_bundle_idx_map = dict()
        for rank, bundle_idx in enumerate(sorted_bundle_idxs):
            worker = worker_cls.options(
                placement_group=pg, placement_group_bundle_index=bundle_idx, **pg_options
            ).remote(worker_config, rank, master_addr, master_port, world_size, device_type)
            workers_bundle_idx_map[worker] = (rank, bundle_idx)

        return workers_bundle_idx_map

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
    def __init__(
        self,
        config,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        self.config = config
        self.accelerator = accelerator
        self.setup_distributed(rank, master_addr, master_port, world_size)

    @property
    def device_visible_env_name(self):
        if self.accelerator == "GPU":
            return "CUDA_VISIBLE_DEVICES"
        elif self.accelerator == "NPU":
            return "ASCEND_RT_VISIBLE_DEVICES"
        else:
            raise ValueError(f"Unsupported accelerator type: {self.accelerator}")

    def setup_distributed(self, rank: int, master_addr: str, master_port: int, world_size: int):
        """Setup method to initialize the worker."""

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
        """Perform all-reduce operation on the given tensor."""
        tensor = torch.tensor([1.0], device=torch.accelerator.current_accelerator())
        dist.all_reduce(tensor)
        return tensor


class AutoAcceleratorWorkers:
    @staticmethod
    def build_placement_group(resources_config: AcceleratorResourcesConfig):
        """Build a placement group based on the provided resources
        configuration."""
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
        accelerator = AutoAcceleratorWorkers.get_device_type(pg)

        if accelerator == "GPU":
            return {"num_cpus": 0.01, "num_gpus": 0.01}
        elif accelerator == "NPU":
            return {"num_cpus": 0.01, "resources": {"NPU": 0.01}}
        else:
            raise ValueError(f"Unsupported accelerator architecture: {accelerator}")

    @staticmethod
    def get_spmd_info(pg: PlacementGroup) -> Tuple[List[int], str, int, int]:
        """Get the SPMD (Single Program Multiple Data) information from the
        placement group."""

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
        """Create workers based on the provided configuration."""
        pg = AutoAcceleratorWorkers.build_placement_group(accelerator_config)
        workers_bundle_idx_map = cls.from_placement_group(worker_cls, worker_config, pg)

        return workers_bundle_idx_map, pg

    @classmethod
    def from_placement_group(cls, worker_cls, worker_config, pg: PlacementGroup):
        """Create workers based on the provided configuration."""

        pg_options = cls.get_pg_options(pg)
        sorted_bundle_idxs, master_addr, master_port, world_size = cls.get_spmd_info(pg)

        workers_bundle_idx_map = dict()
        for rank, bundle_idx in enumerate(sorted_bundle_idxs):
            worker = worker_cls.options(
                placement_group=pg, placement_group_bundle_index=bundle_idx, **pg_options
            ).remote(worker_config, rank, master_addr, master_port, world_size)
            workers_bundle_idx_map[worker] = (rank, bundle_idx)

        return workers_bundle_idx_map

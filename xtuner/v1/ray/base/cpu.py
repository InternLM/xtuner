from typing import Any, Dict, TypeVar

import ray
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, field_validator
from ray.util.placement_group import VALID_PLACEMENT_GROUP_STRATEGIES, PlacementGroup, placement_group
from typing_extensions import Annotated


PG_READY_TIMEOUT = 30  # seconds
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


class AutoCPUWorkers:
    """A utility class for automatically creating and managing cpu actors
    within a Ray PlacementGroup."""

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
        pg = AutoCPUWorkers.build_placement_group(cpu_config)
        workers_list = cls.from_placement_group(worker_cls, worker_config, pg)

        return workers_list, pg

    @classmethod
    def from_placement_group(cls, worker_cls, worker_config, pg: PlacementGroup, num_workers: int = -1):
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
        pg_options = cls.get_pg_options(pg)

        num_workers = num_workers if num_workers > 0 else pg.bundle_count
        workers_list = []
        for _ in range(num_workers):
            worker = worker_cls.options(placement_group=pg, **pg_options).remote(
                worker_config, num_cpus=pg_options.get("num_cpus", 1)
            )  # type: ignore[attr-defined]
            workers_list.append(worker)

        return workers_list

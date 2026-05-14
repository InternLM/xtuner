from __future__ import annotations

import os
import threading
from typing import TypeAlias

import ray
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ray.actor import ActorClass
from ray.util.placement_group import (
    VALID_PLACEMENT_GROUP_STRATEGIES,
    PlacementGroup,
    placement_group,
    placement_group_table,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from typing_extensions import Annotated

from xtuner.v1.utils.logger import get_logger


PG_READY_TIMEOUT = os.getenv("XTUNER_PG_READY_TIMEOUT", 30)  # default 30 seconds
PlacementGroups: TypeAlias = PlacementGroup | list[PlacementGroup] | tuple[PlacementGroup, ...] | None
logger = get_logger()


class CPUResourcesConfig(BaseModel):
    """Configuration for CPU worker resources in XTuner.

    This class provides CPU resource options for Ray CPU workers. When used by
    ``AutoCPUWorkers`` the workers are launched in a CPU placement group. When
    used as ``cpu_resources`` by judgers or agent loops, the workers are managed
    by ``CPUResourceManager`` outside accelerator placement groups.

    Args:
        num_workers (int): Total number of CPU workers. Defaults to 1.
        num_cpus_per_worker (float): Number of CPUs to allocate per worker in the
            placement group or Ray actor. Defaults to 1.
        cpu_memory_per_worker (int): Amount of CPU memory (in bytes) to allocate
            for each worker. Defaults to 1 GiB.
        pg_pack_strategy (str): Ray placement group strategy used only when a
            CPU placement group is built. Defaults to "SPREAD".

    **Examples:**

    Example CPU resource configuration::

        resources = CPUResourcesConfig(
            num_workers=4,
            num_cpus_per_worker=2,
            cpu_memory_per_worker=4 * 1024**3,
        )
    """

    model_config = ConfigDict(extra="forbid")
    num_workers: Annotated[int, Parameter(help="Number of workers."), Field(ge=1)] = 1
    num_cpus_per_worker: Annotated[float, Parameter(help="Number of CPUs to allocate per worker."), Field(gt=0)] = 1
    cpu_memory_per_worker: Annotated[
        int, Parameter(help="Amount of memory (in bytes) to allocate per worker."), Field(gt=0)
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


class CPUActorLauncher:
    """Low-level helper for launching CPU Ray actors.

    ``CPUActorLauncher`` only answers one question: given a class and explicit
    Ray resource/scheduling options, how do we create one or more CPU actors?
    It converts plain Python classes to Ray actor classes, optionally binds
    actors to a placement-group bundle, and forwards constructor args.

    It does **not** remember which placement-group bundles have already been
    used. If ``build_actors`` is called twice with the same placement group and
    no explicit ``start_bundle_idx``, both calls start from bundle 0.

    Example:

    .. code-block:: python

        workers_a = CPUActorLauncher.build_actors(
            WorkerA,
            cfg_a,
            pg=pg,
            start_bundle_idx=0,
            num_workers=2,
            actor_num_cpus_per_worker=1,
        )
        workers_b = CPUActorLauncher.build_actors(
            WorkerB,
            cfg_b,
            pg=pg,
            start_bundle_idx=2,  # caller must manage this offset
            num_workers=2,
            actor_num_cpus_per_worker=1,
        )

    Use this class when the caller already knows exactly where the actor(s)
    should be placed, or when no placement group is involved.
    """

    _ACTOR_CLASS_CACHE: dict[type, ActorClass] = {}

    @staticmethod
    def build_placement_group(resources_config: CPUResourcesConfig):
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
    def get_pg_options(pg: PlacementGroup, num_cpus: int | float = -1) -> dict:
        assert len(pg.bundle_specs) > 0, "Placement group has no bundles defined."
        default_cpu = pg.bundle_specs[0].get("CPU", 1)
        return {"num_cpus": num_cpus if num_cpus >= 0 else default_cpu}

    @classmethod
    def to_actor_class(cls, worker_cls):
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
        capture_child_tasks: bool = False,
        **init_kwargs,
    ):
        resolved_num_cpus = 1 if actor_num_cpus is None else actor_num_cpus
        resolved_memory = actor_memory

        actor_cls = cls.to_actor_class(worker_cls)
        actor_options = {
            "num_cpus": resolved_num_cpus,
        }
        if resolved_memory is not None and resolved_memory > 0:
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
        capture_child_tasks: bool = False,
        **init_kwargs,
    ):
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
    """Convenience wrapper for homogeneous CPU worker pools.

    ``AutoCPUWorkers`` builds on ``CPUActorLauncher`` for the common case where
    a CPU placement group contains one bundle per worker and each worker should
    consume the next available bundle.

    Its main extra behavior is a per-process placement-group cursor. When
    ``from_placement_group`` is called without ``start_bundle_idx``, it starts
    from the first bundle not previously consumed through ``AutoCPUWorkers`` for
    that placement group. This prevents repeated calls from accidentally placing
    different worker pools on the same bundles.

    Example:

    .. code-block:: python

        workers_a = AutoCPUWorkers.from_placement_group(
            WorkerA,
            cfg_a,
            pg,
            num_workers=2,
        )  # uses bundle 0 and 1
        workers_b = AutoCPUWorkers.from_placement_group(
            WorkerB,
            cfg_b,
            pg,
            num_workers=2,
        )  # automatically uses bundle 2 and 3

    ``from_config`` is an even higher-level shortcut: it first creates a CPU
    placement group from ``CPUResourcesConfig`` and then launches workers from
    it.

    Example:

    .. code-block:: python

        workers, pg = AutoCPUWorkers.from_config(
            MyCPUWorker,
            worker_config,
            CPUResourcesConfig(num_workers=4, num_cpus_per_worker=2),
        )

    Use this class for homogeneous CPU worker pools backed by a CPU placement
    group. Use ``CPUActorLauncher`` directly when placement is explicit or when
    the actor should run outside a placement group.
    """

    _PG_NEXT_BUNDLE_INDEX: dict[str, int] = {}
    _PG_NEXT_BUNDLE_INDEX_LOCK = threading.Lock()

    @staticmethod
    def _get_pg_key(pg: PlacementGroup) -> str:
        return str(pg.id)

    @classmethod
    def _reserve_bundle_range(
        cls,
        pg: PlacementGroup,
        num_workers: int,
        start_bundle_idx: int | None,
    ) -> tuple[int, int]:
        pg_key = cls._get_pg_key(pg)

        with cls._PG_NEXT_BUNDLE_INDEX_LOCK:
            current_cursor = cls._PG_NEXT_BUNDLE_INDEX.get(pg_key, 0)
            resolved_start_bundle_idx = current_cursor if start_bundle_idx is None else start_bundle_idx
            resolved_num_workers = num_workers if num_workers > 0 else pg.bundle_count - resolved_start_bundle_idx

            assert resolved_num_workers > 0, "At least one worker must be created from the placement group."
            assert resolved_start_bundle_idx >= 0, "start_bundle_idx must be non-negative."
            assert resolved_start_bundle_idx + resolved_num_workers <= pg.bundle_count, (
                "Placement group does not have enough remaining bundles for the requested CPU workers."
            )

            cls._PG_NEXT_BUNDLE_INDEX[pg_key] = max(current_cursor, resolved_start_bundle_idx + resolved_num_workers)

        return resolved_start_bundle_idx, resolved_num_workers

    @classmethod
    def from_config(cls, worker_cls, worker_config, cpu_config: CPUResourcesConfig):
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
        start_bundle_idx: int | None = None,
    ):
        start_bundle_idx, num_workers = cls._reserve_bundle_range(
            pg=pg, num_workers=num_workers, start_bundle_idx=start_bundle_idx
        )
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


class CPUResourceManager:
    """Validates and serves PG-external CPU actor allocations."""

    def __init__(
        self,
        accelerator_placement_groups: PlacementGroups = None,
    ):
        self.pools: dict[str, CPUResourcesConfig] = {}
        self._registration_counts: dict[str, int] = {}
        if accelerator_placement_groups is None:
            self._accelerator_placement_groups: tuple[PlacementGroup, ...] = ()
        elif isinstance(accelerator_placement_groups, PlacementGroup) or hasattr(
            accelerator_placement_groups, "bundle_specs"
        ):
            self._accelerator_placement_groups = (accelerator_placement_groups,)
        else:
            self._accelerator_placement_groups = tuple(accelerator_placement_groups)

    def get(self, name: str) -> CPUResourcesConfig:
        if name not in self.pools:
            raise KeyError(f"Unknown external CPU resource pool {name!r}. Available pools: {sorted(self.pools)}")
        return self.pools[name]

    def register(self, name: str, config: CPUResourcesConfig) -> None:
        registered_name = self._make_unique_registration_name(name)
        self.pools[registered_name] = config
        try:
            self.validate_or_raise()
        except Exception:
            del self.pools[registered_name]
            raise

    def log_initial_snapshot(self) -> None:
        resource_summary = self._build_resource_summary()
        self._log_resource_summary(
            resource_summary,
            include_registered=False,
            title=(
                "External CPU initial cluster snapshot. "
                "Registered external CPU pools are not shown yet and may change as components are built"
            ),
        )

    def log_registered_summary(self) -> None:
        resource_summary = self._build_resource_summary()
        self._log_resource_summary(
            resource_summary,
            include_registered=True,
            title="External CPU resource summary before rollout/training",
        )

    def validate_or_raise(self) -> None:
        resource_summary = self._build_resource_summary()

        cluster_cpus = resource_summary["cluster_cpus"]
        cluster_memory = int(resource_summary["cluster_memory"])
        accelerator_cpus = resource_summary["accelerator_cpus"]
        accelerator_memory = int(resource_summary["accelerator_memory"])
        external_cpus = resource_summary["external_capacity_cpus"]
        external_memory = int(resource_summary["external_capacity_memory"])
        requested_cpus = resource_summary["registered_external_cpus"]
        requested_memory = int(resource_summary["registered_external_memory"])
        max_node_external_cpus = resource_summary["max_node_external_cpus"]

        errors: list[str] = []
        if requested_cpus > external_cpus:
            errors.append(
                f"CPU requested={requested_cpus:g}, available_outside_accelerator_pg={external_cpus:g}, "
                f"cluster_total={cluster_cpus:g}, accelerator_pg_reserved={accelerator_cpus:g}"
            )
        if requested_memory > 0 and requested_memory > external_memory:
            errors.append(
                "memory requested="
                f"{requested_memory}, available_outside_accelerator_pg={external_memory}, "
                f"cluster_total={cluster_memory}, accelerator_pg_reserved={accelerator_memory}"
            )
        for name, pool in self.pools.items():
            if pool.num_cpus_per_worker > max_node_external_cpus:
                errors.append(
                    f"pool {name!r} requests {pool.num_cpus_per_worker:g} CPU per worker, "
                    f"but the largest node has only {max_node_external_cpus:g} CPU outside accelerator PGs"
                )
        if errors:
            self._log_resource_summary(
                resource_summary,
                include_registered=True,
                title="External CPU resource summary at validation failure",
            )
            pool_lines = [
                f"  - {name}: {pool.num_workers} workers * {pool.num_cpus_per_worker:g} CPU"
                f", {pool.cpu_memory_per_worker} memory each"
                for name, pool in self.pools.items()
            ]
            raise RuntimeError(
                "Insufficient PG-external Ray resources for registered external CPU pools:\n"
                + "\n".join(errors)
                + "\nRegistered pools:\n"
                + ("\n".join(pool_lines) if pool_lines else "  <none>")
            )

    def _make_unique_registration_name(self, name: str) -> str:
        count = self._registration_counts.get(name, 0) + 1
        self._registration_counts[name] = count
        if count == 1 and name not in self.pools:
            return name

        while True:
            candidate = f"{name}#{count}"
            if candidate not in self.pools:
                return candidate
            count += 1
            self._registration_counts[name] = count

    def _accelerator_pg_resource_total(self, resource_name: str) -> float:
        total = 0.0
        for pg in self._accelerator_placement_groups:
            for bundle in pg.bundle_specs:
                total += float(bundle.get(resource_name, 0))
        return total

    def _build_resource_summary(self) -> dict[str, float]:
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        cluster_cpus = float(cluster_resources.get("CPU", 0))
        available_cpus = float(available_resources.get("CPU", 0))
        cluster_memory = float(cluster_resources.get("memory", 0))
        available_memory = float(available_resources.get("memory", 0))
        accelerator_cpus = self._accelerator_pg_resource_total("CPU")
        accelerator_memory = self._accelerator_pg_resource_total("memory")
        registered_external_cpus = sum(pool.num_workers * pool.num_cpus_per_worker for pool in self.pools.values())
        registered_external_memory = float(
            sum(pool.num_workers * pool.cpu_memory_per_worker for pool in self.pools.values())
        )
        external_capacity_cpus = cluster_cpus - accelerator_cpus
        external_capacity_memory = cluster_memory - accelerator_memory
        return {
            "cluster_cpus": cluster_cpus,
            "available_cpus": available_cpus,
            "accelerator_cpus": accelerator_cpus,
            "external_capacity_cpus": external_capacity_cpus,
            "ray_external_in_use_cpus": max(0.0, external_capacity_cpus - available_cpus),
            "registered_external_cpus": registered_external_cpus,
            "remaining_after_registered_cpus": external_capacity_cpus - registered_external_cpus,
            "cluster_memory": cluster_memory,
            "available_memory": available_memory,
            "accelerator_memory": accelerator_memory,
            "external_capacity_memory": external_capacity_memory,
            "ray_external_in_use_memory": max(0.0, external_capacity_memory - available_memory),
            "registered_external_memory": registered_external_memory,
            "remaining_after_registered_memory": external_capacity_memory - registered_external_memory,
            "max_node_external_cpus": self._max_node_external_resource("CPU"),
        }

    def _log_resource_summary(self, summary: dict[str, float], *, include_registered: bool, title: str) -> None:
        rows = [
            (
                "cluster_total",
                summary["cluster_cpus"],
                summary["cluster_memory"],
                "Ray cluster_resources total.",
            ),
            (
                "accelerator_pg_reserved",
                summary["accelerator_cpus"],
                summary["accelerator_memory"],
                "Reserved by accelerator placement group(s).",
            ),
            (
                "external_capacity",
                summary["external_capacity_cpus"],
                summary["external_capacity_memory"],
                "cluster_total - accelerator_pg_reserved.",
            ),
        ]
        if include_registered:
            rows.extend(
                [
                    (
                        "registered_external",
                        summary["registered_external_cpus"],
                        summary["registered_external_memory"],
                        "External CPU pools managed by XTuner.",
                    ),
                    (
                        "remaining_after_registered",
                        summary["remaining_after_registered_cpus"],
                        summary["remaining_after_registered_memory"],
                        "XTuner budget after registered pools.",
                    ),
                ]
            )
            rows.extend(
                [
                    (
                        f"pool:{name}",
                        pool.num_workers * pool.num_cpus_per_worker,
                        pool.num_workers * pool.cpu_memory_per_worker,
                        f"{pool.num_workers} worker(s) * {pool.num_cpus_per_worker:g} CPU.",
                    )
                    for name, pool in self.pools.items()
                ]
            )
        logger.info(f"{title}:\n{self._format_resource_table(rows)}")

    @staticmethod
    def _format_resource_table(rows: list[tuple[str, float, float, str]]) -> str:
        headers = ("item", "CPU", "memory(GB)", "note")
        rendered_rows = [
            (
                item,
                f"{cpu:g}",
                "-" if memory == 0 else f"{memory / 1024**3:.2f}",
                note,
            )
            for item, cpu, memory, note in rows
        ]
        widths = [max(len(str(row[idx])) for row in (headers, *rendered_rows)) for idx in range(len(headers))]

        def render(row: tuple[str, str, str, str]) -> str:
            return " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))

        separator = "-+-".join("-" * width for width in widths)
        return "\n".join([render(headers), separator, *(render(row) for row in rendered_rows), separator])

    def _max_node_external_resource(self, resource_name: str) -> float:
        node_totals = {
            node["NodeID"]: float(node.get("Resources", {}).get(resource_name, 0))
            for node in ray.nodes()
            if node.get("Alive", True)
        }
        if not node_totals:
            return 0.0

        for pg in self._accelerator_placement_groups:
            table = placement_group_table(pg)
            bundles_to_node_id = table.get("bundles_to_node_id", {})
            for bundle_idx, node_id in bundles_to_node_id.items():
                try:
                    bundle = pg.bundle_specs[int(bundle_idx)]
                except (IndexError, ValueError):
                    continue
                if node_id in node_totals:
                    node_totals[node_id] -= float(bundle.get(resource_name, 0))
        return max(node_totals.values())


_CPU_RESOURCE_MANAGER: CPUResourceManager | None = None


def format_cpu_resource_manager_uninitialized_error(owner: str) -> str:
    return (
        f"{owner} sets cpu_resources, but CPUResourceManager is not initialized.\n"
        "This usually means the config is being built outside RLTrainer. In normal training, build the trainer first "
        "so XTuner can initialize CPUResourceManager after accelerator placement groups are created.\n"
        "For standalone tests or scripts, initialize it explicitly before building this component:\n"
        "    from xtuner.v1.rl.utils import CPUResourceManager, set_cpu_resource_manager\n"
        "    set_cpu_resource_manager(CPUResourceManager(accelerator_placement_groups=None))\n"
        "Note: standalone initialization does not account for accelerator placement group reservation unless you pass "
        "the placement group(s)."
    )


def set_cpu_resource_manager(manager: CPUResourceManager | None) -> None:
    global _CPU_RESOURCE_MANAGER
    _CPU_RESOURCE_MANAGER = manager


def get_cpu_resource_manager() -> CPUResourceManager | None:
    return _CPU_RESOURCE_MANAGER


def register_cpu_resources(name: str, cpu_resources: CPUResourcesConfig) -> None:
    if _CPU_RESOURCE_MANAGER is None:
        raise ValueError(format_cpu_resource_manager_uninitialized_error(name))

    _CPU_RESOURCE_MANAGER.register(name=name, config=cpu_resources)


def clear_cpu_resource_manager() -> None:
    set_cpu_resource_manager(None)

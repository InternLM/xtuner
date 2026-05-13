from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import ray
from pydantic import BaseModel, ConfigDict, Field
from ray.util.placement_group import PlacementGroup, placement_group_table

from xtuner.v1.utils.logger import get_logger


PlacementGroups: TypeAlias = PlacementGroup | list[PlacementGroup] | tuple[PlacementGroup, ...] | None
logger = get_logger()


class CPUActorPoolConfig(BaseModel):
    """CPU requirements for one PG-external Ray actor pool."""

    model_config = ConfigDict(extra="forbid")

    num_actors: int = Field(ge=1)
    num_cpus_per_actor: float = Field(default=1, gt=0)
    memory_per_actor: int | None = Field(default=1024**3, gt=0)

    @property
    def total_cpus(self) -> float:
        return self.num_actors * self.num_cpus_per_actor

    @property
    def total_memory(self) -> int:
        if self.memory_per_actor is None:
            return 0
        return self.num_actors * self.memory_per_actor


class CPUResourceManagerConfig(BaseModel):
    """Registry for Ray CPU actors that run outside accelerator placement
    groups.

    This config is a bookkeeping and validation layer. It does not reserve or isolate all PG-external CPUs; Ray actors
    that bypass this config can still run if the Ray cluster has ordinary CPU resources available.
    """

    model_config = ConfigDict(extra="forbid")

    strict: bool = True
    pools: dict[str, CPUActorPoolConfig] = Field(default_factory=dict)

    @property
    def total_cpus(self) -> float:
        return sum(pool.total_cpus for pool in self.pools.values())

    @property
    def total_memory(self) -> int:
        return sum(pool.total_memory for pool in self.pools.values())


@dataclass(frozen=True)
class CPUActorPoolAllocation:
    name: str
    config: CPUActorPoolConfig

    @property
    def num_actors(self) -> int:
        return self.config.num_actors

    @property
    def num_cpus_per_actor(self) -> float:
        return self.config.num_cpus_per_actor

    @property
    def memory_per_actor(self) -> int | None:
        return self.config.memory_per_actor

    def actor_options(self) -> dict:
        options: dict = {"num_cpus": self.num_cpus_per_actor}
        if self.memory_per_actor is not None:
            options["memory"] = self.memory_per_actor
        return options


class CPUResourceManager:
    """Validates and serves PG-external CPU actor allocations."""

    def __init__(
        self,
        config: CPUResourceManagerConfig | None,
        accelerator_placement_groups: PlacementGroups = None,
    ):
        self.config = config or CPUResourceManagerConfig()
        self._registration_counts: dict[str, int] = {}
        if accelerator_placement_groups is None:
            self._accelerator_placement_groups: tuple[PlacementGroup, ...] = ()
        elif isinstance(accelerator_placement_groups, PlacementGroup) or hasattr(
            accelerator_placement_groups, "bundle_specs"
        ):
            self._accelerator_placement_groups = (accelerator_placement_groups,)
        else:
            self._accelerator_placement_groups = tuple(accelerator_placement_groups)

    def get(self, name: str) -> CPUActorPoolAllocation:
        if name not in self.config.pools:
            raise KeyError(
                f"Unknown external CPU resource pool {name!r}. Available pools: {sorted(self.config.pools)}"
            )
        return CPUActorPoolAllocation(name=name, config=self.config.pools[name])

    def register(self, name: str, config: CPUActorPoolConfig) -> CPUActorPoolAllocation:
        registered_name = self._make_unique_registration_name(name)
        self.config.pools[registered_name] = config
        try:
            self.validate_or_raise()
        except Exception:
            del self.config.pools[registered_name]
            raise
        return CPUActorPoolAllocation(name=registered_name, config=config)

    def log_initial_snapshot(self) -> None:
        resource_summary = self._build_resource_summary()
        self._log_resource_summary(resource_summary, include_registered=False)

    def validate_or_raise(self) -> None:
        resource_summary = self._build_resource_summary()
        self._log_resource_summary(resource_summary, include_registered=True)
        if not self.config.strict:
            return

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
        for name, pool in self.config.pools.items():
            if pool.num_cpus_per_actor > max_node_external_cpus:
                errors.append(
                    f"pool {name!r} requests {pool.num_cpus_per_actor:g} CPU per actor, "
                    f"but the largest node has only {max_node_external_cpus:g} CPU outside accelerator PGs"
                )
        if errors:
            pool_lines = [
                f"  - {name}: {pool.num_actors} actors * {pool.num_cpus_per_actor:g} CPU"
                + (f", {pool.memory_per_actor} memory each" if pool.memory_per_actor is not None else "")
                for name, pool in self.config.pools.items()
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
        if count == 1 and name not in self.config.pools:
            return name

        while True:
            candidate = f"{name}#{count}"
            if candidate not in self.config.pools:
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
        registered_external_cpus = self.config.total_cpus
        registered_external_memory = float(self.config.total_memory)
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

    def _log_resource_summary(self, summary: dict[str, float], *, include_registered: bool) -> None:
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
                        pool.total_cpus,
                        pool.total_memory,
                        f"{pool.num_actors} actor(s) * {pool.num_cpus_per_actor:g} CPU.",
                    )
                    for name, pool in self.config.pools.items()
                ]
            )
            title = "External CPU resource summary"
        else:
            title = (
                "External CPU initial cluster snapshot. "
                "Registered external CPU pools are not shown yet and may change as components are built"
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
        f"{owner} sets external_cpu, but CPUResourceManager is not initialized.\n"
        "This usually means the config is being built outside RLTrainer. In normal training, build the trainer first "
        "so XTuner can initialize CPUResourceManager after accelerator placement groups are created.\n"
        "For standalone tests or scripts, initialize it explicitly before building this component:\n"
        "    from xtuner.v1.rl.utils import CPUResourceManager, set_cpu_resource_manager\n"
        "    set_cpu_resource_manager(CPUResourceManager(None, accelerator_placement_groups=None))\n"
        "Note: standalone initialization does not account for accelerator placement group reservation unless you pass "
        "the placement group(s)."
    )


def set_cpu_resource_manager(manager: CPUResourceManager | None) -> None:
    global _CPU_RESOURCE_MANAGER
    _CPU_RESOURCE_MANAGER = manager


def get_cpu_resource_manager() -> CPUResourceManager | None:
    return _CPU_RESOURCE_MANAGER


def clear_cpu_resource_manager() -> None:
    set_cpu_resource_manager(None)

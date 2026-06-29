from __future__ import annotations

from dataclasses import dataclass, field


__all__ = [
    "RolloutEngine",
    "RolloutServerProcess",
    "RolloutTopology",
    "ServerLaunchSpec",
]


@dataclass(frozen=True)
class RolloutServerProcess:
    """Static topology for one worker-owned rollout server process."""

    # Worker rank that owns and starts this server process.
    worker_rank: int
    # Placement-group bundles assigned to this server process.
    placement_group_bundle_idxs: tuple[int, ...]
    # Whether this server process can receive rollout generation requests.
    accepts_rollout_requests: bool = True
    # Node index used by backends that launch one server process per node.
    node_rank: int = 0
    # Number of nodes participating in this server launch.
    nnodes: int = 1


@dataclass(frozen=True)
class RolloutEngine:
    """Static topology for one logical inference engine."""

    # Rollout ranks that jointly form this logical inference engine.
    engine_ranks: tuple[int, ...]
    # Rendezvous address shared by every server process in this engine.
    dist_init_addr: str
    # Server processes that expose this engine to rollout traffic or control paths.
    server_processes: tuple[RolloutServerProcess, ...]


@dataclass(frozen=True)
class ServerLaunchSpec:
    """Worker-facing launch data projected from rollout topology."""

    # Worker rank that should receive this launch spec.
    worker_rank: int
    # Placement-group bundles assigned to the launched server process.
    placement_group_bundle_idxs: tuple[int, ...]
    # Engine rendezvous address resolved by RolloutTopology.
    dist_init_addr: str
    # Rank of this worker inside the logical inference engine.
    engine_rank: int
    # Node index for multi-node backend launches.
    node_rank: int = 0
    # Number of nodes for multi-node backend launches.
    nnodes: int = 1


@dataclass(frozen=True)
class RolloutTopology:
    """Immutable rollout engine layout after dist-init addresses are resolved.

    Actor handles, server URLs, session URLs, and lifecycle state belong to RolloutWorkerRegistry.
    """

    # Logical inference engines and their server-process topology.
    engines: tuple[RolloutEngine, ...]
    # Legacy trainer/update-weight mesh kept until the update path moves to explicit targets.
    training_engine_mesh: tuple[tuple[int, ...], ...]
    # Server-process lookup keyed by worker rank.
    _server_process_by_rank: dict[int, RolloutServerProcess] = field(init=False, repr=False, compare=False)
    # Lifecycle group lookup keyed by server-process worker rank.
    _lifecycle_group_by_rank: dict[int, tuple[int, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        server_process_by_rank: dict[int, RolloutServerProcess] = {}
        lifecycle_group_by_rank: dict[int, tuple[int, ...]] = {}
        for engine in self.engines:
            lifecycle_group = tuple(server.worker_rank for server in engine.server_processes)
            for server in engine.server_processes:
                server_process_by_rank[server.worker_rank] = server
                lifecycle_group_by_rank[server.worker_rank] = lifecycle_group

        object.__setattr__(self, "_server_process_by_rank", server_process_by_rank)
        object.__setattr__(self, "_lifecycle_group_by_rank", lifecycle_group_by_rank)

    def server_launch_specs(self) -> tuple[ServerLaunchSpec, ...]:
        return tuple(
            ServerLaunchSpec(
                worker_rank=server.worker_rank,
                placement_group_bundle_idxs=server.placement_group_bundle_idxs,
                dist_init_addr=engine.dist_init_addr,
                engine_rank=engine.engine_ranks.index(server.worker_rank),
                node_rank=server.node_rank,
                nnodes=server.nnodes,
            )
            for engine in self.engines
            for server in engine.server_processes
        )

    def lifecycle_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(dict.fromkeys(self._lifecycle_group_by_rank.values()))

    def is_request_entrypoint_rank(self, rank: int) -> bool:
        server = self._server_process_by_rank.get(rank)
        return server is not None and server.accepts_rollout_requests

    def lifecycle_group_for_server_rank(self, rank: int) -> tuple[int, ...]:
        try:
            return self._lifecycle_group_by_rank[rank]
        except KeyError:
            raise KeyError(f"rank={rank} does not own a rollout server process.") from None

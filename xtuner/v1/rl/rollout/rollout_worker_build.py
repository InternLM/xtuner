from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from xtuner.v1.rl.utils import AutoAcceleratorWorkers
from xtuner.v1.utils import get_logger

from ._generation.rollout_worker_generator import RayRolloutWorkerGenerator
from ._generation.session_worker_selector import RolloutWorkerHandle
from .worker import RolloutConfig, RolloutWorker


@dataclass
class RolloutWorkerRuntime:
    """Runtime handles for one rollout worker."""

    worker_actor: RolloutWorker
    backend_url: str
    session_server_url: str | None = None
    generator_actor: Any | None = None
    bundle_idx: int | None = None


@dataclass
class RolloutRuntime:
    engine_rank_mesh_array: list[list[int]]
    worker_server_urls_map: dict[int, str]
    rank2worker: dict[int, RolloutWorkerRuntime]
    worker_handles: list[RolloutWorkerHandle]


class RolloutWorkerBuilder:
    """Bootstrap rollout workers and worker generation actors."""

    def __init__(self, config: RolloutConfig, placement_group: PlacementGroup) -> None:
        self.config = config
        self.placement_group = placement_group
        self.num_gpus_per_engine = config.num_gpus_per_engine
        self.logger = get_logger(log_dir=config.worker_log_dir, tag="RolloutWorkerBuilder")
        self.num_active_workers = 0

    def build(self) -> RolloutRuntime:
        engine_rank_mesh_array, worker_server_urls_map, rank2worker = self._init_workers()
        self._init_rollout_worker_generators(rank2worker)
        worker_handles = self._build_worker_handles(rank2worker)
        return RolloutRuntime(
            engine_rank_mesh_array=engine_rank_mesh_array,
            worker_server_urls_map=worker_server_urls_map,
            rank2worker=rank2worker,
            worker_handles=worker_handles,
        )

    def _build_worker_handles(self, rank2worker: dict[int, RolloutWorkerRuntime]) -> list[RolloutWorkerHandle]:
        worker_handles = []
        for rank, worker in rank2worker.items():
            if worker.generator_actor is None:
                raise RuntimeError(f"Missing RolloutWorkerGenerator for rollout worker rank {rank}.")
            worker_handles.append(
                RolloutWorkerHandle(
                    rank=rank,
                    worker_actor=worker.worker_actor,
                    backend_url=worker.backend_url,
                    generator_actor=worker.generator_actor,
                    session_server_url=worker.session_server_url,
                )
            )
        return worker_handles

    def _init_rollout_worker_generators(self, rank2worker: dict[int, RolloutWorkerRuntime]) -> None:
        for rank, worker in rank2worker.items():
            if worker.bundle_idx is None:
                raise RuntimeError(f"Missing placement bundle index for active rollout worker rank {rank}.")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.placement_group,
                placement_group_capture_child_tasks=False,
                placement_group_bundle_index=worker.bundle_idx,
            )
            worker.generator_actor = RayRolloutWorkerGenerator.options(
                scheduling_strategy=scheduling_strategy,
                num_cpus=0,
            ).remote(self.config, rank, worker.backend_url)

    def _get_worker_cls(self):
        if os.environ.get("XTUNER_USE_LMDEPLOY") == "1":
            from .lmdeploy import LMDeployWorker

            worker_cls = LMDeployWorker
        elif os.environ.get("XTUNER_USE_VLLM") == "1":
            from .vllm import vLLMWorker

            worker_cls = vLLMWorker
        elif os.environ.get("XTUNER_USE_SGLANG") == "1":
            from .sglang import SGLangWorker

            worker_cls = SGLangWorker
        else:
            raise NotImplementedError(
                "Rollout backend is not supported."
                "Please set XTUNER_USE_LMDEPLOY or XTUNER_USE_VLLM"
                " or XTUNER_USE_SGLANG environment variable."
            )
        return ray.remote(worker_cls)

    def _update_dist_init_addr(self, nodes_per_engine, server_urls_per_engine, dist_init_addrs, tp_size):
        assert server_urls_per_engine == 1 or nodes_per_engine == 1
        if nodes_per_engine > 1:
            index = list(range(0, self.num_active_workers + 1, tp_size)) + [self.num_active_workers]
            for i in range(1, len(index)):
                dist_init_addrs[index[i - 1] : index[i]] = [dist_init_addrs[index[i - 1]]] * (index[i] - index[i - 1])
        if server_urls_per_engine > 1:
            activate_servers = len(dist_init_addrs)
            for i in range(0, activate_servers, server_urls_per_engine):
                dist_init_addrs[i : i + server_urls_per_engine] = [dist_init_addrs[i]] * server_urls_per_engine
        return dist_init_addrs

    def _update_active_workers_and_urls_map(self, active_rollout_workers, worker_server_urls_map):
        if self.config.rollout_cross_node_comm or self.num_gpus_per_engine < self.config.gpus_per_node:
            return active_rollout_workers, worker_server_urls_map

        active_worker_interval = self.num_gpus_per_engine // self.config.gpus_per_node
        active_rank = list(worker_server_urls_map.keys())[::active_worker_interval]
        active_worker_server_urls = list(worker_server_urls_map.values())[::active_worker_interval]
        return active_rollout_workers[::active_worker_interval], dict(zip(active_rank, active_worker_server_urls))

    def _init_workers(self):
        workers, rank_bundle_idx_list = AutoAcceleratorWorkers.from_placement_group(
            self._get_worker_cls(), self.config, self.placement_group
        )
        active_servers_count, nodes_per_engine = self.config.get_active_servers_count(len(workers))
        interval = len(workers) // active_servers_count
        active_rollout_workers = workers[::interval]
        self.num_active_workers = len(active_rollout_workers)
        server_urls_per_engine = self.config.server_urls_per_engine

        set_bundle_idxs_objectref = []
        engine_rank_mesh_array = []
        activate_worker_idx = 0
        for active_worker in active_rollout_workers:
            head_rank, _ = rank_bundle_idx_list[activate_worker_idx]
            engine_workers_meta = rank_bundle_idx_list[head_rank : head_rank + interval]
            engine_bundle_idxs = [meta[1] for meta in engine_workers_meta]
            set_bundle_idxs_objectref.append(active_worker._set_engine_bundle_idxs.remote(engine_bundle_idxs))  # type: ignore[attr-defined]
            engine_rank_mesh_array.append([meta[0] for meta in engine_workers_meta])
            activate_worker_idx += interval
        ray.get(set_bundle_idxs_objectref)
        ray.get(
            [worker._set_engine_rank_mesh_array.remote(engine_rank_mesh_array) for worker in active_rollout_workers]
        )  # type: ignore[attr-defined]

        init_dist_init_addrs = ray.get([worker.init_dist_port.remote() for worker in active_rollout_workers])  # type: ignore[attr-defined]
        dist_init_addrs = self._update_dist_init_addr(
            nodes_per_engine, server_urls_per_engine, init_dist_init_addrs, self.num_gpus_per_engine
        )
        init_results = ray.get(
            [worker.init.remote(dist_init_addrs[i]) for i, worker in enumerate(active_rollout_workers)]
        )
        worker_server_urls_map = dict(init_results)
        worker_session_url_dict = dict(
            ray.get([worker.get_session_server_info.remote() for worker in active_rollout_workers])
        )
        active_rollout_workers, worker_server_urls_map = self._update_active_workers_and_urls_map(
            active_rollout_workers, worker_server_urls_map
        )
        active_ranks = list(worker_server_urls_map.keys())
        worker_session_url_dict = {rank: worker_session_url_dict[rank] for rank in active_ranks}

        worker_runtimes = {}
        rank_to_bundle_idx = {rank: bundle_idx for rank, bundle_idx in rank_bundle_idx_list}
        for i, rank in enumerate(worker_server_urls_map.keys()):
            worker_runtimes[rank] = RolloutWorkerRuntime(
                worker_actor=active_rollout_workers[i],
                backend_url=worker_server_urls_map[rank],
                session_server_url=worker_session_url_dict[rank],
                bundle_idx=rank_to_bundle_idx[rank],
            )
        self.logger.info(f"Rollout worker server URLs: {[worker.backend_url for worker in worker_runtimes.values()]}")
        self.logger.info(
            f"Rollout worker session server URLs: {[worker.session_server_url for worker in worker_runtimes.values()]}"
        )
        return engine_rank_mesh_array, worker_server_urls_map, worker_runtimes


def build_rollout_runtime(config: RolloutConfig, placement_group: PlacementGroup) -> RolloutRuntime:
    return RolloutWorkerBuilder(config, placement_group).build()

from typing import Dict, List, Tuple

import ray
import ray.exceptions
import ray.util.queue

from transformers import AutoTokenizer
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.utils import get_logger

from .worker import RolloutWorker


logger = get_logger()


@ray.remote
class RolloutController:
    def __init__(
        self,
        infer_config: RolloutConfig,
        workers_bundle_idx_map: Dict[RolloutWorker, Tuple[int, int]],
    ):
        self.config = infer_config
        self.num_workers = 0
        self.worker_server_urls: List[str] = []
        self.active_rollout_workers: List[RolloutWorker] = []
        self.tokenizer = (
            AutoTokenizer.from_pretrained(infer_config.model_path, trust_remote_code=True)
            if infer_config.tokenizer_path
            else None
        )
        self.workers_bundle_idx_map = workers_bundle_idx_map
        self.engine_mesh_list, self.server_url_dict = self.init_workers()
        # todo(@duanyanhui): add router to replace native round robin
        self.worker_index = 0  # round robin index

    def get_rollout_info(self):
        return dict(
            engine_mesh_list=self.engine_mesh_list,
            server_url_dict=self.server_url_dict,
            rollout_config=self.config,
        )

    def init_workers(self):
        workers = list(self.workers_bundle_idx_map.keys())
        active_servers_count, nodes_per_engine = self._get_active_servers_count(self.config, len(workers))
        interval = len(workers) // active_servers_count
        self.active_rollout_workers = workers[::interval]
        self.num_workers = len(self.active_rollout_workers)

        set_bundle_idxs_objectref = []
        engine_mesh_list = []
        for active_worker in self.active_rollout_workers:
            head_rank, start_bundle_idx = self.workers_bundle_idx_map[active_worker]
            engine_workers = workers[start_bundle_idx : start_bundle_idx + interval]
            engine_bundle_idxs = [self.workers_bundle_idx_map[worker][1] for worker in engine_workers]
            set_bundle_idxs_objectref.append(active_worker.set_engine_bundle_idxs.remote(engine_bundle_idxs))  # type: ignore[attr-defined]
            engine_mesh_list.append([self.workers_bundle_idx_map[worker][0] for worker in engine_workers])
        ray.get(set_bundle_idxs_objectref)

        logger.info(f"active_rollout_workers: {self.active_rollout_workers}")
        # init dist_init_addr for each worker according to parallel settings
        init_dist_init_addrs = ray.get([worker.init_dist_port.remote() for worker in self.active_rollout_workers])  # type: ignore[attr-defined]
        dist_init_addrs = self._update_dist_init_addr(
            nodes_per_engine, init_dist_init_addrs, self.config.tensor_parallel_size
        )
        # launch rollout servers
        worker_server_urls_map = dict(
            ray.get(
                [
                    worker.init.remote(dist_init_addrs[i])  # type: ignore[attr-defined]
                    for i, worker in enumerate(self.active_rollout_workers)
                ]
            )
        )
        self.worker_server_urls = list(worker_server_urls_map.values())
        return engine_mesh_list, worker_server_urls_map

    async def rollout(self, prompt: str, sample_params):
        index = self.worker_index % len(self.active_rollout_workers)
        response_ref = self.active_rollout_workers[index].rollout.remote(prompt, sample_params)  # type: ignore[attr-defined]
        self.worker_index += 1
        return await response_ref

    def pause(self):
        return ray.get([worker.pause.remote() for worker in self.active_rollout_workers])

    def restart(self):
        return ray.get([worker.restart.remote() for worker in self.active_rollout_workers])

    # internal functions
    def _update_dist_init_addr(self, nodes_per_engine, dist_init_addrs, tp_size):
        if nodes_per_engine > 1:
            index = list(range(0, self.num_workers + 1, tp_size)) + [self.num_workers]
            for i in range(1, len(index)):
                dist_init_addrs[index[i - 1] : index[i]] = [dist_init_addrs[index[i - 1]]] * (index[i] - index[i - 1])
        return dist_init_addrs

    def _get_active_servers_count(self, infer_config: RolloutConfig, gpu_nums: int):
        # NOTE：Since different inference engines have different launch methods,
        # the number of nodes contained in each engine is not consistent.
        # For example: sglang requires starting an inference engine for each node,
        # while lmdeploy and vllm does not. Therefore, we calculate the number
        # of active servers based on the configuration.
        support_cross_node_comm = infer_config.rollout_cross_node_comm
        gpus_per_node = infer_config.gpus_per_node
        tp_size = infer_config.tensor_parallel_size
        nodes_per_engine = 1 if support_cross_node_comm or tp_size < gpus_per_node else tp_size // gpus_per_node
        active_servers_count = int((gpu_nums // tp_size) * nodes_per_engine)
        return active_servers_count, nodes_per_engine

    def reset_prefix_cache(self):
        return [worker.reset_prefix_cache.remote() for worker in self.active_rollout_workers]

    def offload(self):
        ray.get([worker.sleep.remote() for worker in self.active_rollout_workers])
        return

    def onload(self, *args, **kwargs):
        ray.get([worker.wake_up.remote(*args, **kwargs) for worker in self.active_rollout_workers])
        return

    def shutdown(self):
        return ray.get([worker.shutdown.remote() for worker in self.active_rollout_workers])

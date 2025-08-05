import uuid
from typing import List

import ray
import ray.util.queue

from transformers import AutoTokenizer
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import RolloutMeta, SampleParams

from .worker import RolloutWorker


@ray.remote
class RolloutController:
    def __init__(
        self,
        infer_config: RolloutConfig,
        workers: List[RolloutWorker],
        inqueue: ray.util.queue.Queue = None,
        outqueue: ray.util.queue.Queue = None,
    ):
        self.config = infer_config
        self.num_workers = 0
        self.worker_server_urls: List[str] = []
        self.active_rollout_workers: List[ray.actor.ActorHandle] = []
        self.inqueue = inqueue if inqueue is not None else ray.util.queue.Queue(maxsize=10)
        self.outqueue = outqueue if outqueue is not None else ray.util.queue.Queue(maxsize=1000)
        self.tokenizer = (
            AutoTokenizer.from_pretrained(infer_config.model_path, trust_remote_code=True)
            if infer_config.tokenizer_path
            else None
        )
        self.init_workers(self.config, workers)

    def init_workers(self, infer_config: RolloutConfig, workers: List[RolloutWorker]):
        active_servers_count, nodes_per_engine = self._get_active_servers_count(infer_config, len(workers))
        interval = len(workers) // active_servers_count
        self.active_rollout_workers = workers[::interval]
        self.num_workers = len(self.active_rollout_workers)

        print(f"self.active_rollout_workers: {self.active_rollout_workers}")
        # init dist_init_addr for each worker according to parallel settings
        init_dist_init_addrs = ray.get([worker.init_dist_port.remote() for worker in self.active_rollout_workers])  # type: ignore[attr-defined]
        dist_init_addrs = self._update_dist_init_addr(
            nodes_per_engine, init_dist_init_addrs, infer_config.tensor_parallel_size
        )
        # launch rollout servers
        self.worker_server_urls = ray.get(
            [
                worker.init.remote(infer_config, dist_init_addrs[i])  # type: ignore[attr-defined]
                for i, worker in enumerate(self.active_rollout_workers)
            ]
        )
        return [worker.rollout.remote(self.inqueue, self.outqueue) for worker in self.active_rollout_workers]  # type: ignore[attr-defined]

    def init_router(self, infer_config: RolloutConfig):
        self.active_rollout_workers[0].launch_router.remote(infer_config)

    # call workers functions
    def rollout(self, prompt: str, label: str, sample_params: SampleParams = SampleParams()):
        for worker in self.active_rollout_workers:
            worker.restart.remote(self.inqueue, self.outqueue)
        input_ids = self.tokenizer.encode(prompt) if self.tokenizer else []
        uid = str(uuid.uuid4())
        rollout_meta = RolloutMeta(
            uid=uid,
            prompt=prompt,
            input_ids=input_ids,
            response="",
            output_ids=[],
            label=label,
            sample_params=sample_params,
        )
        self.inqueue.put((ray.put(rollout_meta),))

    def pause(self):
        return [worker.pause.remote() for worker in self.active_rollout_workers]

    def get_worker_status(self):
        return [worker.get_status.remote() for worker in self.active_rollout_workers]

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
        return [worker.sleep.remote() for worker in self.active_rollout_workers]

    def onload(self):
        return [worker.wake_up.remote() for worker in self.active_rollout_workers]

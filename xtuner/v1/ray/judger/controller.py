from typing import List

import ray
import ray.util.queue

from .worker import JudgerWorker


@ray.remote
class JudgerController:
    def __init__(self, workers: list[JudgerWorker], config: dict = dict()):
        self.config = config
        self.worker_server_urls: List[str] = []
        # todo(@duanyanhui): single judger controller support multiple workers
        self.workers = workers
        self.num_workers = len(self.workers)
        self.worker_index = 0  # round robin index

    async def judge(self, response, label):
        index = self.worker_index % len(self.workers)
        reward_ref = self.workers[index].judge.remote(response, label)
        self.worker_index += 1
        return await reward_ref

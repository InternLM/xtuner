from typing import List

import ray

from .worker import JudgerWorker


@ray.remote
class JudgerController:
    def __init__(self, config: dict, workers: list[JudgerWorker]):
        self.config = config
        self.worker_server_urls: List[str] = []
        self.workers: List[JudgerWorker] = []
        self.init_workers(config, workers)
        self.workers = workers
        self.num_workers = len(self.workers)

    def init_workers(
        self,
        config: dict,
        workers: List[JudgerWorker],
        launch_method: str = "function",
    ):
        if launch_method == "function":
            return
        else:
            # todo(@duanyanhui): Wrap the judge_function as an OpenAI API server
            return

    def judge(self, inqueue: ray.util.queue.Queue, outqueue: ray.util.queue.Queue):
        return [worker.judge.remote(inqueue, outqueue) for worker in self.workers]  # type: ignore[attr-defined]

    def pause(self):
        return [worker.pause.remote() for worker in self.workers]

import ray

from .worker import RolloutWorker


@ray.remote
class LMDeployWorker(RolloutWorker):
    pass

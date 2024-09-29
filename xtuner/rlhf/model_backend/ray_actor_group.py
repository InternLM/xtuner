import ray

from .cuda_memory_stats import merge_cuda_memory_stats_list
from .ray_actor_mixin import RayActorMixin


class RayActorGroup:

    def __init__(self, name: str, config: dict):
        self.config = config
        self.name = name  # name_prefix for ray_actors
        self.ray_actors: list[RayActorMixin] = []

    def get_cuda_mem_stats(self):
        return merge_cuda_memory_stats_list(
            ray.get([
                ray_actor.get_memory_stats_of_visible_devices.remote()
                for ray_actor in self.ray_actors
            ]))

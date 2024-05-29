import uuid
from typing import TypeVar

from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

DEFAULT_NUM_CPUS = 1
DEFAULT_NUM_GPUS = 1
T = TypeVar('T')
UUID = uuid.uuid4()  # may called multiple times in different ray instances


# Create Ray Actors
def create_ray_actors(
    name_prefix: str,
    config: dict,
    placement_group: PlacementGroup,
    trainer_class: T,
) -> list[T]:
    ray_actors = [_ for _ in range(placement_group.bundle_count)]
    for index in range(placement_group.bundle_count):
        ray_actors[index] = trainer_class.options(
            name=f'{name_prefix}_rank_{index}',
            namespace=f'{UUID}_{trainer_class.__class__.__name__}',
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=index,
            ),
            runtime_env=set_runtime_env(),
        ).remote(config)
    return ray_actors


def set_runtime_env():
    runtime_env = {'env_vars': {'HF_ENDPOINT': 'https://hf-mirror.com'}}
    return runtime_env

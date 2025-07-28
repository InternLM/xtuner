from typing import Optional

from cyclopts import Group, Parameter
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from xtuner.v1.config import EngineConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig
from xtuner.v1.ray.config.worker import InfrerenceWorkerConfig


grpo_group = Group("GRPO", sort_key=1, help="GRPO Trainer Configuration")

actor_worker_group = Group("Actor Workers", sort_key=90, help="Configuration for the rollout worker.")
actor_resources_group = Group("Actor Resources", sort_key=90, help="Configuration for the actor resources.")
rollout_worker_group = Group("Rollout Workers", sort_key=90, help="Configuration for the rollout worker.")
rollout_resources_group = Group("Rollout Resources", sort_key=90, help="Configuration for the rollout resources.")


class GRPOTrainerConfig(BaseModel):
    """Configuration for the GRPO Ray Trainer."""

    actor: Annotated[
        EngineConfig,
        Parameter(group=actor_worker_group, help="Configuration for the rollout worker."),
    ]

    critic: Annotated[
        EngineConfig,
        Parameter(group=actor_worker_group, help="Configuration for the rollout worker."),
    ]

    actor_resources: Annotated[
        AcceleratorResourcesConfig, Parameter(group=actor_resources_group, help="Resources allocated for the actor.")
    ]

    rollout: Annotated[
        InfrerenceWorkerConfig,
        Parameter(group=rollout_worker_group, help="Configuration for the rollout worker."),
        # Discriminator('type')
    ]
    rollout_resources: Annotated[
        Optional[AcceleratorResourcesConfig],
        Parameter(group=rollout_resources_group, help="Resources allocated for the rollout."),
    ] = None

    enrionment: Annotated[str, Parameter(group=grpo_group, help="Environment for the GRPO training.")] = "default"

    global_batch_size: Annotated[int, Parameter(group=grpo_group, help="Batch size for training.")] = Field(
        32, help="Batch size for training."
    )

    micro_batch_size: Annotated[int, Parameter(group=grpo_group, help="Micro batch size for training.")] = Field(
        8, help="Micro batch size for training."
    )

    num_mini_batches: Annotated[int, Parameter(group=grpo_group, help="Number of mini-batches for training.")] = Field(
        4, help="Number of mini-batches for training."
    )

    total_steps: Annotated[int, Parameter(group=grpo_group, help="Total number of training steps.")] = Field(
        100000, help="Total number of training steps."
    )

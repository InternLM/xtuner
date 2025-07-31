import ray

from xtuner.v1.config.trainer import TrainerConfig
from xtuner.v1.train.trainer import Trainer

from ..accelerator import SingleAcceleratorWorker


@ray.remote(
    runtime_env={
        "env_vars": {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        }
    },
)
class TrainingWorker(SingleAcceleratorWorker):
    """Worker class for training tasks."""

    def __init__(
        self,
        # config: EngineConfig,
        config: TrainerConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        super().__init__(config, rank, master_addr, master_port, world_size, accelerator)
        # Additional initialization for training can be added here
        self.config = config
        self.trainer = Trainer.from_config(config)

    def get_data_replicate_size(self) -> int:
        """Get the data parallel size for the training worker."""
        return 1

    def fit(self):
        self.trainer.fit()

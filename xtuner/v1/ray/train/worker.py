import ray

from xtuner.v1.config import EngineConfig
from xtuner.v1.engine import build_engine

from ..accelerator import SingleAcceleratorWorker


@ray.remote
class TrainingWorker(SingleAcceleratorWorker):
    """Worker class for training tasks."""

    def __init__(
        self,
        config: EngineConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        super().__init__(config, rank, master_addr, master_port, world_size, accelerator)
        # Additional initialization for training can be added here
        self.config = config
        self.engine = build_engine(config)

    def get_data_replicate_size(self) -> int:
        """Get the data parallel size for the training worker."""
        return 1

    def train_step(self, data_batches, sp_size: int = 1):
        """Perform a single training step with the provided data."""
        # Here you would implement the actual training logic
        # For demonstration, we will just return a dummy loss
        log = self.engine.train_step(data_batches)
        return log

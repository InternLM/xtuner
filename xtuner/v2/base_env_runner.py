
import asyncio
from typing import AsyncIterator, Optional
from xtuner.v1.datasets import DataloaderConfig

from .utils import load_function
from .rollout_controller import RolloutController
from .rollout_state import Trajectory, ProcessorUtilState


class BaseEnvRunner:
    def __init__(self,
                rollout_controller: RolloutController,
                processor_utils_state: ProcessorUtilState,
                dataloader_cfg: DataloaderConfig | None = None, # none 是为了这个 envruner 可以独立运行
                judger: callable | None = None, # none 是为了这个 envruner 可以独立运行
                generate_external: callable | None = None, 
            ):
        
        self.dataloader = None
        if dataloader_cfg is not None:
            self.dataloader = dataloader_cfg.build()  
        self.rollout_controller = rollout_controller
        self.judger = judger
        self.processor_utils_state = processor_utils_state
        
        self.generate_external = generate_external
        if self.generate_external is not None:
            self.generate_external = load_function(self.generate_external)

    def sample(self) -> dict:
        try:
            data = next(self.dataloader_iter)[0] 
        except StopIteration:
            self.cur_epoch += 1
            self.dataloader.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.dataloader)
            data = next(self.dataloader_iter)[0]
        self.reduced_consumed_samples += 1
        return data
    
    async def generate(self, data_item) -> Optional[Trajectory]:
        if self.generate_external is not None:
            return await self.generate_external(data_item, self.processor_utils_state, self.rollout_controller, self.judger)
        else:
            raise NotImplementedError
 
    async def generate_batch(self, batch_size: int) -> AsyncIterator[Trajectory]:
        data_concurrency = batch_size
        assert self.dataloader is not None, "Dataloader must be provided for batch generation."

        pending_tasks = []
        for _ in range(data_concurrency):
            data_item = self.sample()
            task = asyncio.create_task(self.generate(data_item))
            pending_tasks.append(task)

        completed_sample_count = 0
        batch_trajectories = []
        while completed_sample_count < batch_size:
            if not pending_tasks:
                print("All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            for task in done_tasks:
                try:
                    traj =  await task
                    if traj is not None:
                        batch_trajectories.append(traj)
                        completed_sample_count += 1
                except Exception as e:
                    print(f"Error in generating trajectory: {e}")
        
        # TODO: 如果有超发
        return batch_trajectories

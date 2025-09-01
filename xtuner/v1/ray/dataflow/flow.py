import asyncio
from typing import List, Optional

import ray
from cyclopts import Parameter
from pydantic import BaseModel
from tqdm.auto import tqdm
from typing_extensions import Annotated

from xtuner.v1.datasets.data_item import RLTextDataItem
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.rollout.controller import SampleParams
from xtuner.v1.ray.utils import create_task
from xtuner.v1.utils import get_logger

from .replay_buffer import ReplayBuffer, ReplayBufferConfig


class DataFlowConfig(BaseModel):
    env: Annotated[
        str,
        Parameter(help="Environment name to set for the dataflow."),
    ] = ""
    max_concurrent: Annotated[
        int,
        Parameter(help="Maximum number of concurrent tasks."),
    ] = 8
    prompt_repeat_k: Annotated[
        int,
        Parameter(help="Number of times to repeat each prompt."),
    ] = 1
    global_batch_size: Annotated[
        int,
        Parameter(help="Target number of samples to collect before stopping."),
    ] = 8
    max_retry_times: Annotated[
        int,
        Parameter(help="Maximum number of retry task for failed samples."),
    ] = 1
    # async generate config
    replay_ratio: Annotated[
        float,
        Parameter(help="Ratio of samples to replay from the buffer."),
    ] = 0
    replay_weights: Annotated[
        dict,
        Parameter(help="Weights for different states in the replay buffer."),
    ] = {}
    enable_partial_rollout: Annotated[
        int, Parameter(help="Whether to enable async rollout_controller. 1 for enabled, 0 for disabled")
    ] = 0
    sample_params: Annotated[SampleParams, Parameter(help="Parameters for sampling from the model.")] = SampleParams()


@ray.remote
class DataFlow:
    def __init__(
        self,
        env: str,
        dataflow_cfg: DataFlowConfig,
        replay_buffer_cfg: ReplayBufferConfig,
        environment: SingleTurnEnvironment,
    ):
        self.env = env
        self.config = dataflow_cfg
        self.replay_buffer = ReplayBuffer.remote(replay_buffer_cfg)  # type: ignore[attr-defined]
        self.env_controller = environment
        self.send_samples_count = 0
        self.finished_samples_count = 0
        self.unfinished_samples_count = 0
        self.failed_samples_count = 0
        self.logger = get_logger()
        self.target_batch_size = self.config.global_batch_size

    def get_train_dataset_length(self):
        return ray.get(self.replay_buffer.get_train_dataset_length.remote())

    async def worker_task(self, group_samples_for_retry: Optional[List[RLTextDataItem]] = None):
        group_samples = group_samples_for_retry
        try:
            # step 1: sample
            if group_samples is None:
                group_samples = await self.replay_buffer.sample.remote(  # type: ignore[attr-defined]
                    self.env,
                    self.config.enable_partial_rollout,
                    self.config.prompt_repeat_k,
                    self.config.replay_ratio,
                    self.config.replay_weights,
                )
                self.send_samples_count += 1
                self.logger.debug(
                    f"Get 1 sample and dataflow have sent {self.send_samples_count} to rollout_controller"
                )
            else:
                self.logger.debug("Retrying the failed sample")
            # step 2: env generate
            group_samples = await self.env_controller.run.remote(group_samples, self.sample_params)  # type: ignore[attr-defined]
            # step 3: filter
            filtered_group_samples = await self.replay_buffer.post_processor.remote(group_samples)  # type: ignore[attr-defined]
            # step 4: add to replay buffer
            await self.replay_buffer.add.remote(filtered_group_samples)  # type: ignore[attr-defined]

        except Exception as e:
            if group_samples is not None and len(group_samples) > 0:
                self.logger.error(f"Worker task failed with exception: {e}. Returning meta for retry.", exc_info=True)
                for sample in group_samples:
                    if "retry_times" not in sample:
                        sample["retry_times"] = 0
                    sample["retry_times"] += 1
                return group_samples
            else:
                self.logger.warning(f"Worker task failed with exception: {e}. No samples to return.")

    async def concurrent_task_runner(self):
        waiting_tasks = set()
        with tqdm(total=self.target_batch_size, desc="rollout_controller for training samples") as pbar:
            update_step = max(1, int(self.target_batch_size * 0.1))
            next_update_threshold = update_step
            while self.finished_samples_count < self.target_batch_size:
                if self.finished_samples_count >= next_update_threshold:
                    pbar.n = self.finished_samples_count
                    pbar.refresh()
                    next_update_threshold += update_step
                while len(waiting_tasks) < self.config.max_concurrent:
                    # In async mode, we keep spawning. In sync mode, we stop if we have enough tasks in flight.
                    if (
                        not self.config.enable_partial_rollout
                        and self.finished_samples_count + len(waiting_tasks) >= self.target_batch_size
                    ):
                        break
                    task = create_task(self.worker_task())
                    waiting_tasks.add(task)

                done_tasks, pending_tasks = await asyncio.wait(
                    waiting_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done_tasks:
                    result = task.result()
                    if result is not None:
                        if result[0]["retry_times"] < self.config.max_retry_times:
                            # If the retry count is less than max_retry_times, retry the task
                            retry_task = create_task(self.worker_task(group_samples_for_retry=result))
                            pending_tasks.add(retry_task)
                        else:
                            self.logger.error(f"Max retry reached for {result[0]['prompt_id']}. Not retrying.")
                            self.failed_samples_count += 1

                self.finished_samples_count = ray.get(self.replay_buffer.get_finished_samples.remote())
                waiting_tasks = pending_tasks

            pbar.n = self.finished_samples_count
            pbar.refresh()

        self.logger.info("Target batch size reached. Pausing env controller.")
        ray.get(self.env_controller.pause.remote())

        if waiting_tasks:
            await asyncio.wait_for(asyncio.gather(*waiting_tasks, return_exceptions=True), timeout=10)

        self.unfinished_samples_count = ray.get(self.replay_buffer.get_unfinished_samples.remote())
        self.logger.info(
            f"send_samples_count: {self.send_samples_count}, unfinished_samples_count:{self.unfinished_samples_count}, finished_samples: {self.finished_samples_count}, failed_samples: {self.failed_samples_count}"
        )

    async def run(self, num: Optional[int] = None, sample_params: Optional[SampleParams] = None):
        ray.get(self.env_controller.restart.remote())  # type: ignore[attr-defined]
        self.send_samples_count = 0
        self.finished_samples_count = 0
        self.unfinished_samples_count = 0
        self.failed_samples_count = 0
        self.target_batch_size = num if num and num > 0 else self.config.global_batch_size
        self.logger.info(f"Target batch size set to {self.target_batch_size}.")
        self.sample_params = sample_params if sample_params else self.config.sample_params
        self.logger.info(f"Sample parameters set to {self.sample_params}.")
        await self.concurrent_task_runner()
        return await self.replay_buffer.get_samples.remote(self.target_batch_size)  # type: ignore[attr-defined]

    def state(self):
        ray.get(self.replay_buffer.print.remote())

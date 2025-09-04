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
    """Data flow configuration for XTuner.

    Simple configuration for managing concurrent data generation workflows
    in reinforcement learning training.

    Args:
        env (str): Environment identifier. Defaults to "".
        max_concurrent (int): Maximum concurrent tasks. Defaults to 8.
        prompt_repeat_k (int): Times to repeat each prompt. Defaults to 1.
        global_batch_size (int): Target samples to collect. Defaults to 8.
        max_retry_times (int): Maximum retry attempts. Defaults to 1.
        enable_partial_rollout (int): Enable async mode (1) or disable (0). Defaults to 0.
        sample_params (SampleParams): Model sampling parameters. Defaults to SampleParams().

    **Examples:**

    Example configuration for dataflow::

        config = DataFlowConfig(
            env="test_env",
            max_concurrent=256,
            global_batch_size=1024,
            prompt_repeat_k=8,
            sample_params=SampleParams(max_tokens=2048),
        )
    """

    env: Annotated[
        str,
        Parameter(help="Environment name to set for the dataflow."),
    ] = ""
    max_concurrent: Annotated[
        int,
        Parameter(help="Maximum number of concurrent tasks."),
    ] = 8
    max_retry_times: Annotated[
        int,
        Parameter(help="Maximum number of retry task for failed samples."),
    ] = 1
    prompt_repeat_k: Annotated[
        int,
        Parameter(help="Number of times to repeat each prompt."),
    ] = 1
    global_batch_size: Annotated[
        int,
        Parameter(help="Target number of samples to collect before stopping."),
    ] = 8
    enable_partial_rollout: Annotated[
        int, Parameter(help="Whether to enable async rollout_controller. 1 for enabled, 0 for disabled")
    ] = 0
    sample_params: Annotated[SampleParams, Parameter(help="Parameters for sampling from the model.")] = SampleParams()


@ray.remote
class DataFlow:
    """A Ray actor that manages the data flow for reinforcement learning.

    This class is responsible for sampling prompts, interacting with the environment or to generate responses,
    processing the results, and storing them in a replay buffer. It orchestrates the asynchronous generation of
    training data.
    """

    def __init__(
        self,
        env: str,
        dataflow_cfg: DataFlowConfig,
        replay_buffer_cfg: ReplayBufferConfig,
        environment: SingleTurnEnvironment,
    ):
        """Initializes the DataFlow actor.

        Args:
            env (str): The name of the environment.
            dataflow_cfg (DataFlowConfig): Configuration for the data flow.
            replay_buffer_cfg (ReplayBufferConfig): Configuration for the
                replay buffer.
            environment (EnvController): The environment controller actor.
            postprocessor (Optional[Callable]): An optional function to
                post-process the generated samples.
        """
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
        """Gets the length of the training dataset from the replay buffer."""
        return ray.get(self.replay_buffer.get_train_dataset_length.remote())

    async def worker_task(self, group_samples_for_retry: Optional[List[RLTextDataItem]] = None):
        """A single worker task to generate and process a group of samples.

        This task performs the following steps:
        1. Samples a prompt from the replay buffer (or uses a sample for retry).
        2. Calls the environment controller or rollout controller to generate a response.
        3. Post-processes the generated samples use default postprocessor and custom postprocessor.
        4. Adds the filtered samples to the replay buffer.

        Args:
            group_samples_for_retry (Optional[List[RLTextDataItem]]): A group
                of samples to retry if a previous attempt failed. Defaults to
                None.

        Returns:
            Optional[List[RLTextDataItem]]: The group of samples if the task
            fails and needs to be retried, otherwise None.
        """
        group_samples = group_samples_for_retry
        try:
            # step 1: sample
            if group_samples is None:
                group_samples = await self.replay_buffer.sample.remote(  # type: ignore[attr-defined]
                    self.env,
                    self.config.enable_partial_rollout,
                    self.config.prompt_repeat_k,
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
        """Orchestrates the concurrent execution of worker tasks to generate a
        batch of training data.

        This method manages a pool of asynchronous worker tasks to collect a
        specified number of samples (`self.global_batch_size`). It handles
        task scheduling, retries for failed tasks, and progress tracking.

        The process is as follows:
        1.  Continuously spawns new `worker_task` instances until the
            number of in-flight tasks reaches `self.config.max_concurrent`.
        2.  Uses `asyncio.wait` to efficiently handle completed tasks.
        3.  If a task fails but is retryable, it is rescheduled with the same
            data, up to `self.config.max_retry_times`.
        4.  If a task fails permanently, it is logged and counted.
        5.  A progress bar (`tqdm`) is updated as samples are successfully
            processed.
        6.  Once `global_batch_size` is reached, the environment controller is
            paused, and the method waits for any remaining tasks to finish
            before completing.
        """
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
        """Starts the data generation process.

        This method resets the internal state and runs the concurrent task
        runner to collect a new batch of samples.

        Returns:
            List[RLTextDataItem]: A list of collected training samples.
        """
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

import asyncio
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import ray
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from tqdm.auto import tqdm
from typing_extensions import Annotated

from xtuner.v1.data_proto.rl_data import RLDataFlowItem, RolloutState
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.rollout.controller import SampleParams
from xtuner.v1.ray.utils import create_task
from xtuner.v1.utils import get_logger

from .replay_buffer import ReplayBuffer, ReplayBufferConfig, determine_group_state


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

    model_config = ConfigDict(extra="forbid")

    env: Annotated[
        str,
        Parameter(help="Environment name to set for the dataflow."),
    ] = ""
    # NOTE: max_concurrent / max_retry_times 直接删了，还是兼容下逻辑比较好？
    max_concurrent: Annotated[
        Optional[int],
        Parameter(help="Maximum number of concurrent tasks."),
    ] = None
    max_retry_times: Annotated[
        int,
        Parameter(help="Maximum number of retry task for failed samples."),
    ] = 3
    prompt_repeat_k: Annotated[
        int,
        Parameter(help="Number of times to repeat each prompt."),
    ] = 1
    global_batch_size: Annotated[
        int,
        Parameter(help="Target number of samples to collect before stopping."),
    ] = 8
    sample_params: Annotated[SampleParams, Parameter(help="Parameters for sampling from the model.")] = SampleParams()
    extra_params: Annotated[Dict, Parameter(help="Extra parameters for rollout.")] = {}
    # async params
    staleness_threshold: Annotated[
        float,
        Parameter(
            help="The maximum allowed threshold of stale (expired) samples in a training batch. Must be between 0.0 and 1.0."
        ),
    ] = 0.0
    enable_partial_rollout: Annotated[
        bool,
        Parameter(help="Whether to enable partial rollout for asynchronous data generation."),
    ] = False
    tail_batch_candidate_steps: Annotated[
        int,
        Parameter(
            help="Number of rollout steps after which a sample becomes a candidate for the tail batch. Set to 0 to disable."
        ),
    ] = 0
    tail_batch_trigger_size: Annotated[
        Optional[int],
        Parameter(
            help="Number of candidate samples needed in the queue to trigger a tail batch operation. Set to 0 to disable."
        ),
    ] = None
    worker_log_dir: Annotated[Path, Parameter(help="Directory to save worker logs.")] = Path.cwd() / "work_dir"

    def model_post_init(self, __context: Any) -> None:
        self.worker_log_dir.mkdir(parents=True, exist_ok=True)
        if self.tail_batch_trigger_size is None:
            self.tail_batch_trigger_size = self.global_batch_size


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
        self.logger = get_logger(log_dir=dataflow_cfg.worker_log_dir, tag="DataFlow")
        self.env = env
        self.config = dataflow_cfg
        replay_buffer_cfg.worker_log_dir = self.config.worker_log_dir
        replay_buffer_cfg.enable_partial_rollout = self.config.enable_partial_rollout
        replay_buffer_cfg.tail_batch_candidate_steps = self.config.tail_batch_candidate_steps
        replay_buffer_cfg.tail_batch_trigger_size = self.config.tail_batch_trigger_size
        self.replay_buffer = ReplayBuffer.remote(replay_buffer_cfg)  # type: ignore[attr-defined]
        self.env_controller = environment
        self.finished_samples_count = 0
        self.skipped_sample_count = 0
        self.failed_sample_count = 0
        self.filtered_samples_count = 0
        self.target_batch_size = self.config.global_batch_size
        rollout_info = ray.get(self.env_controller.get_rollout_info.remote())  # type: ignore[attr-defined]
        self.worker_url_list = list(rollout_info["server_url_dict"].values())
        self.logger.info(f"DataFlow connected to active rollout workers url: {self.worker_url_list}")
        rollout_config = rollout_info["rollout_config"]
        max_concurrent = int(
            (
                rollout_config.rollout_max_batch_size_per_instance
                * len(self.worker_url_list)
                / self.config.prompt_repeat_k
            )
            * rollout_config.allow_over_concurrency_ratio
        )

        if self.config.max_concurrent is None:
            self.config.max_concurrent = max_concurrent
            self.logger.info(
                f"Set Dataflow max_concurrent to {self.config.max_concurrent} based on rollout max batch size and number of workers."
            )
        else:
            self.logger.warning(
                f"Dataflow max_concurrent is set to {self.config.max_concurrent}, we proposed to set max_concurrent to {max_concurrent} based on rollout_max_batch_size_per_instance."
            )
        self.logger.info(f"DataFlowConfig:\n{self.config.model_dump_json(indent=2)}")
        self.cleanup_task_time = 5 * 60  # 5 minutes

    def _prepare(
        self,
        global_batch_size: Optional[int] = None,
        sample_params: Optional[SampleParams] = None,
        extra_params: Optional[Dict] = None,
    ):
        """Resets all internal state variables of DataFlow."""
        if global_batch_size and global_batch_size > 0:
            self.target_batch_size = global_batch_size
        else:
            self.target_batch_size = self.config.global_batch_size

        self.sample_from_expired_storage, self.finished_samples_count = ray.get(
            self.replay_buffer.get_prerun_state.remote(self.target_batch_size)
        )
        self.skipped_sample_count = 0
        self.failed_sample_count = 0
        self.filtered_samples_count = 0

        self.sample_params = sample_params if sample_params else self.config.sample_params
        self.extra_params = extra_params if extra_params else self.config.extra_params
        logger_msg = (
            f"DataFlow states for new generations: target_batch_size={self.target_batch_size}, "
            f"sample_params: {self.sample_params}, extra_params: {self.extra_params}, "
            f"sample_from_expired_storage={self.sample_from_expired_storage}, finished_samples_count={self.finished_samples_count}, "
        )
        self.logger.info(logger_msg)

    def get_train_dataset_length(self):
        """Gets the length of the training dataset from the replay buffer."""
        return ray.get(self.replay_buffer.get_train_dataset_length.remote())

    async def worker_task(self, group_samples_for_retry: Optional[List[RLDataFlowItem]] = None):
        """A single worker task to generate and process a group of samples.

        This task performs the following steps:
        1. Samples a prompt from the replay buffer (or uses a sample for retry).
        2. Calls the environment controller or rollout controller to generate a response.
        3. Post-processes the generated samples use default postprocessor and custom postprocessor.
        4. Adds the filtered samples to the replay buffer.

        Args:
            group_samples_for_retry (Optional[List[RLDataFlowItem]]): A group
                of samples to retry if a previous attempt failed. Defaults to
                None.

        Returns:
            Optional[List[RLDataFlowItem]]: The group of samples if the task
            fails and needs to be retried, otherwise None.
        """
        tast_start_time = time.perf_counter()
        # step 1: sample
        # TODO(@duanyanhui): More fine-grained control over group data generation:
        # Pass n to the inference engine to ensure that the same data is processed by the same server, improving efficiency.
        group_data_items = await self.replay_buffer.sample.remote(  # type: ignore[attr-defined]
            self.env, self.config.prompt_repeat_k
        )
        assert len(group_data_items) > 0, "Sampled empty group data items from replay buffer."
        action_id = group_data_items[0].uid.action_id
        # step 2: env generate
        group_data_items = await self.env_controller.run.remote(  # type: ignore[attr-defined]
            group_data_items,
            sample_params=self.sample_params,
            extra_params=self.extra_params,
        )

        # Step 3: Determine the sample's state and act accordingly.
        group_state = determine_group_state(group_data_items)
        self.logger.debug(f"Determined replay state for {action_id}: {group_state}")
        if group_state == RolloutState.COMPLETED:
            group_data_items = await self.replay_buffer.post_processor.remote(group_data_items)  # type: ignore[attr-defined]
            if len(group_data_items) > 0:
                await self.replay_buffer.add.remote(group_data_items)  # type: ignore[attr-defined]
            else:
                self.filtered_samples_count += 1
            self.logger.debug(f"Worker task completed successfully for {action_id}.")
        elif group_state == RolloutState.ABORTED:
            await self.replay_buffer.add.remote(group_data_items)  # type: ignore[attr-defined]
            self.logger.debug(f"Adding aborted sample {action_id} to aborted storage")
        elif group_state == RolloutState.SKIPPED:
            self.skipped_sample_count += 1
            self.logger.info(f"Total skipped samples count: {self.skipped_sample_count}")
        elif group_state == RolloutState.FAILED:
            self.failed_sample_count += 1
            self.logger.info(f"Total failed samples count: {self.failed_sample_count}")
        else:
            self.logger.error(f"Unexpected group state '{group_state}' for action_id {action_id}.")

        return time.perf_counter() - tast_start_time

    async def concurrent_task_runner(self):
        """Orchestrates the concurrent execution of worker tasks.

        This method manages a pool of asynchronous worker tasks to collect a
        target number of samples (`self.target_batch_size`). It dynamically
        adjusts the number of concurrent tasks based on progress and a
        staleness threshold, ensuring efficient data generation.

        The process is as follows:
        1.  Initializes a set of worker tasks, potentially over-provisioning
            based on `self.config.staleness_threshold` to account for
            variability in task completion times.
        2.  Enters a main loop that continues until `target_batch_size`
            samples are collected.
        3.  Inside the loop, it periodically checks the number of pending
            tasks and launches new ones if the current number is insufficient
            to meet the target, maintaining a steady flow of data generation.
        4.  Uses `asyncio.wait` with a short timeout to efficiently monitor
            for completed tasks without blocking execution.
        5.  A progress bar (`tqdm`) is updated as samples are collected.
        6.  Once `target_batch_size` is reached, it sends a pause/abort
            signal to all rollout workers to prevent them from starting new
            computations.
        7.  It then waits for any remaining in-flight tasks to complete, with
            a configurable timeout to prevent indefinite hanging. Tasks that
            do not finish within the timeout are forcefully cancelled.
        """
        waiting_tasks = set()
        dataflow_start_time = time.perf_counter()
        task_completion_times = []
        with tqdm(total=self.target_batch_size, desc="rollout_controller for training samples") as pbar:
            update_step = max(1, int(self.target_batch_size * 0.1))
            next_update_threshold = update_step

            if self.sample_from_expired_storage:
                # 如果是从过期的存储中采样数据，需要禁用staleness_threshold
                data_concurrency = self.target_batch_size - self.finished_samples_count
                staleness_threshold = 0.0  # disable staleness when sampling from expired storage
                self.logger.info(
                    f"Sampling from expired storage, starting {data_concurrency} worker tasks from expired samples."
                )
            else:
                data_concurrency = math.ceil(
                    (1 + self.config.staleness_threshold) * (self.target_batch_size - self.finished_samples_count)
                )
                staleness_threshold = self.config.staleness_threshold
                self.logger.info(
                    f"Starting dataflow concurrent task runner with data_concurrency: {data_concurrency}, target_batch_size: {self.target_batch_size}, finished_samples_count: {self.finished_samples_count}, staleness_threshold: {staleness_threshold}"
                )

            for _ in range(data_concurrency):
                task = create_task(self.worker_task())
                waiting_tasks.add(task)

            while self.finished_samples_count < self.target_batch_size:
                # 每生成10%数据，打印一次状态日志，并且判断waiting_tasks的数量，补充新的task
                if self.finished_samples_count >= next_update_threshold:
                    pbar.n = self.finished_samples_count
                    pbar.refresh()
                    next_update_threshold += update_step
                    self.logger.info(
                        f"waiting_tasks: {len(waiting_tasks)}, finished_samples_count: {self.finished_samples_count}"
                    )
                    if len(waiting_tasks) < self.target_batch_size - self.finished_samples_count:
                        # 当在执行的task的数量不足以满足需要的数量的时候，补充新的task, 补充的方式是超发当前需要数量的staleness_threshold比例的task
                        increment_data_concurrency = math.ceil(
                            (1 + staleness_threshold)
                            * (self.target_batch_size - self.finished_samples_count - len(waiting_tasks))
                        )
                        self.logger.info(
                            f"Increment data concurrency to {increment_data_concurrency} tasks based on staleness_threshold: {staleness_threshold}, current waiting_tasks: {len(waiting_tasks)}, finished_samples_count: {self.finished_samples_count}"
                        )
                        for _ in range(increment_data_concurrency):
                            task = create_task(self.worker_task())
                            waiting_tasks.add(task)
                        self.logger.info(f"After increment, waiting_tasks: {len(waiting_tasks)}")

                if len(waiting_tasks) == 0:
                    if (
                        self.failed_sample_count > self.target_batch_size
                        or self.skipped_sample_count > self.target_batch_size
                    ):
                        self.logger.error(
                            f"Too many failed or skipped samples, aborting dataflow. failed_sample_count: {self.failed_sample_count}, skipped_sample_count: {self.skipped_sample_count}, target_batch_size: {self.target_batch_size}"
                        )
                        break
                    increment_data_concurrency = math.ceil(
                        (1 + staleness_threshold)
                        * (self.target_batch_size - self.finished_samples_count - len(waiting_tasks))
                    )
                    self.logger.info(
                        f"Length of waiting task is 0 and increment data concurrency to {increment_data_concurrency} tasks based on staleness_threshold: {staleness_threshold}, current waiting_tasks: {len(waiting_tasks)}, finished_samples_count: {self.finished_samples_count}"
                    )
                    for _ in range(increment_data_concurrency):
                        task = create_task(self.worker_task())
                        waiting_tasks.add(task)
                    self.logger.info(f"After increment, waiting_tasks: {len(waiting_tasks)}")

                done_tasks, pending_tasks = await asyncio.wait(
                    waiting_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )

                for done_task in done_tasks:
                    task_time = done_task.result()
                    task_completion_times.append(task_time)

                self.finished_samples_count = ray.get(self.replay_buffer.get_completed_samples_count.remote())
                waiting_tasks = pending_tasks

            pbar.n = self.finished_samples_count
            pbar.refresh()

        generation_time = time.perf_counter() - dataflow_start_time
        pause_start_time = time.perf_counter()

        if len(waiting_tasks) > 0:
            self.logger.info(f"Start pausing env controller for remaining worker tasks {len(waiting_tasks)}.")
            await self.pause()
            cleanup_start_time = time.perf_counter()
            while len(waiting_tasks) > 0:
                elapsed_time = time.perf_counter() - cleanup_start_time
                if elapsed_time > self.cleanup_task_time:
                    self.logger.warning(
                        f"Cleanup timeout of {self.cleanup_task_time}s reached. "
                        f"Forcefully cancelling {len(waiting_tasks)} remaining tasks."
                    )
                    for task in waiting_tasks:
                        task.cancel()
                    # Wait for cancellations to complete
                    await asyncio.gather(*waiting_tasks, return_exceptions=True)
                    break  # Exit the cleanup loop
                # NOTE: Keep sending pause requests because the inference engine only marks currently running requests as aborted.
                # When a waiting request starts running, it still needs another pause request to be marked as aborted.
                _, pending_tasks = await asyncio.wait(waiting_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                if len(pending_tasks) > 0:
                    await self.pause()
                waiting_tasks = pending_tasks
            self.logger.info("All worker tasks have completed after pausing env controller.")

        pause_time = time.perf_counter() - pause_start_time
        dataflow_time = time.perf_counter() - dataflow_start_time
        self.logger.info(
            f"dataflow task finished, generation_time: {generation_time:.2f}s, pause_time: {pause_time:.2f}s, total_time: {dataflow_time:.2f}s"
        )
        self._log_task_completion_stats(task_completion_times)

    async def pause(self, timeout: float = 60.0):
        """Asynchronously sends abort requests to all rollout workers."""
        if not self.worker_url_list:
            self.logger.info("No active rollout workers to pause.")
            return

        async with httpx.AsyncClient() as client:
            tasks = [self._send_abort_request(client, url, timeout=timeout) for url in self.worker_url_list]
            results = await asyncio.gather(*tasks)

        failed_workers = [url for url, success in results if not success]
        succeeded_count = len(self.worker_url_list) - len(failed_workers)

        if failed_workers:
            self.logger.warning(
                f"Abort requests completed. Succeeded: {succeeded_count}, "
                f"Failed: {len(failed_workers)}. Failed workers: {failed_workers}"
            )
        else:
            self.logger.info(f"All {succeeded_count} abort requests sent successfully.")

    async def run(
        self,
        num: Optional[int] = None,
        sample_params: Optional[SampleParams] = None,
        extra_params: Optional[Dict] = None,
        dump: bool = False,
        dump_path: Optional[str] = None,
        resume: bool = False,
        resume_path: Optional[str] = None,
    ):
        """Starts the data generation process.

        This method resets the internal state and runs the concurrent task
        runner to collect a new batch of samples.

        Returns:
            List[RLDataFlowItem]: A list of collected training samples.
        """
        self._prepare(global_batch_size=num, sample_params=sample_params, extra_params=extra_params)
        self.logging_replaybuffer_state("DataFlow run started. ")
        if resume:
            assert resume_path, "Resuming is enabled but no resume path is provided."
            self.logger.info(f"Resuming replay buffer from {resume_path}")
            await self.replay_buffer.resume.remote(resume_path)

        await self.concurrent_task_runner()
        self.logging_replaybuffer_state("DataFlow run completed. ")

        if dump:
            assert dump_path, "Dumping is enabled but no dump path is provided."
            self.logger.info(f"Dump replay buffer from {dump_path}")
            await self.replay_buffer.dump.remote(dump_path)

        return await self.replay_buffer.get_samples.remote(self.target_batch_size)  # type: ignore[attr-defined]

    def logging_replaybuffer_state(self, logging_msg: Optional[str] = None):
        status = self.get_replaybuffer_status()
        logging_msg = logging_msg if logging_msg else ""
        logging_msg += f"ReplayBuffer Status: {status}"
        logging_msg += f", finished_samples_count: {self.finished_samples_count}, "
        logging_msg += f"skipped_samples_count: {self.skipped_sample_count}, "
        logging_msg += f"failed_samples_count: {self.failed_sample_count}, "
        logging_msg += f"filtered_samples_count: {self.filtered_samples_count}, "
        self.logger.info(logging_msg)

    def get_replaybuffer_status(self):
        return ray.get(self.replay_buffer.status.remote())

    def clear_replaybuffer(self):
        return ray.get(self.replay_buffer.clear.remote())

    async def _send_abort_request(self, client, url, timeout):
        worker_url = f"{url}/abort_request"
        try:
            response = await client.post(worker_url, json={"abort_all": True}, timeout=timeout)
            response.raise_for_status()
            self.logger.debug(f"Successfully sent abort request to {url}")
            return url, True
        except Exception as e:
            self.logger.error(f"Failed to send abort request to {url}: {e}")
            return url, False

    def _log_task_completion_stats(self, task_times: List[float]):
        if not task_times:
            self.logger.info("No task completion times to report.")
            return

        import numpy as np

        p50 = np.percentile(task_times, 50)
        p90 = np.percentile(task_times, 90)
        p95 = np.percentile(task_times, 95)
        p99 = np.percentile(task_times, 99)
        max_time = np.max(task_times)
        avg_time = np.mean(task_times)
        std_dev = np.std(task_times)

        task_completions_report = (
            "Task Completions Time:\n"
            f"  - Task Count: {len(task_times)}, Avg Time: {avg_time:.2f}s, Std: {std_dev:.2f}s\n"
            f"  - P50 (Median): {p50:.2f}s, P90: {p90:.2f}s, P95: {p95:.2f}s, P99: {p99:.2f}s\n"
            f"  - Max Time: {max_time:.2f}s, Ratio (P99 / P50): {p99 / p50 if p50 > 0 else float('inf'):.2f}\n"
        )
        self.logger.info(task_completions_report)

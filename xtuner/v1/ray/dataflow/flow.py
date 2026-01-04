import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import ray
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from ray.actor import ActorProxy
from tqdm.auto import tqdm
from typing_extensions import Annotated

from xtuner.v1.data_proto.rl_data import RLDataFlowItem, RolloutState
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.rollout.controller import SampleParams
from xtuner.v1.ray.utils import create_task
from xtuner.v1.utils import get_logger, ray_method

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
    enable_partial_rollout: Annotated[
        int, Parameter(help="Whether to enable async rollout_controller. 1 for enabled, 0 for disabled")
    ] = 0
    sample_params: Annotated[SampleParams, Parameter(help="Parameters for sampling from the model.")] = SampleParams()
    extra_params: Annotated[Dict, Parameter(help="Extra parameters for rollout.")] = {}
    worker_log_dir: Annotated[Path, Parameter(help="Directory to save worker logs.")] = Path.cwd() / "work_dir"

    def model_post_init(self, __context: Any) -> None:
        self.worker_log_dir.mkdir(parents=True, exist_ok=True)


class RawDataFlow:
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
        replay_buffer_cfg.worker_log_dir = self.config.worker_log_dir
        self.replay_buffer = ReplayBuffer.remote(replay_buffer_cfg)  # type: ignore[attr-defined]
        self.env_controller = environment
        self.finished_samples_count = 0
        self.skipped_sample_count = 0
        self.failed_sample_count = 0
        self.logger = get_logger(log_dir=self.config.worker_log_dir, tag="DataFlow")
        self.target_batch_size = self.config.global_batch_size
        self.logger.info(f"DataFlowConfig:\n{self.config.model_dump_json(indent=2)}")
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
        self.enable_partial_rollout = self.config.enable_partial_rollout

    def _reset_internal_states(
        self,
        global_batch_size: Optional[int] = None,
        sample_params: Optional[SampleParams] = None,
        extra_params: Optional[Dict] = None,
    ):
        """Resets all internal state variables of DataFlow."""
        self.finished_samples_count = 0
        self.skipped_sample_count = 0
        self.failed_sample_count = 0
        if global_batch_size and global_batch_size > 0:
            self.target_batch_size = global_batch_size
        else:
            self.target_batch_size = self.config.global_batch_size

        self.sample_params = sample_params if sample_params else self.config.sample_params
        self.extra_params = extra_params if extra_params else self.config.extra_params
        logger_msg = (
            f"DataFlow internal states reset for new run: target_batch_size={self.target_batch_size}, "
            f"sample_params: {self.sample_params}, extra_params: {self.extra_params}, "
            f"enable_partial_rollout={self.enable_partial_rollout}."
        )
        self.logger.info(logger_msg)

    @ray_method
    def get_train_dataset_length(self):
        """Gets the length of the training dataset from the replay buffer."""
        return ray.get(self.replay_buffer.get_train_dataset_length.remote())

    @ray_method
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
        # step 1: sample
        # TODO(@duanyanhui): More fine-grained control over group data generation:
        # Pass n to the inference engine to ensure that the same data is processed by the same server, improving efficiency.
        group_data_items = await self.replay_buffer.sample.remote(  # type: ignore[attr-defined]
            self.env, self.enable_partial_rollout, self.config.prompt_repeat_k
        )
        assert len(group_data_items) > 0, "Sampled empty group data items from replay buffer."
        action_id = group_data_items[0].uid.action_id
        # step 2: env generate
        group_data_items = await self.env_controller.run.remote(  # type: ignore[attr-defined]
            group_data_items, sample_params=self.sample_params, extra_params=self.extra_params
        )

        # Step 3: Determine the sample's state and act accordingly.
        group_state = determine_group_state(group_data_items)
        self.logger.debug(f"Determined replay state for {action_id}: {group_state}")
        if group_state == RolloutState.COMPLETED:
            group_data_items = await self.replay_buffer.post_processor.remote(group_data_items)  # type: ignore[attr-defined]
            if len(group_data_items) > 0:
                await self.replay_buffer.add.remote(group_data_items)  # type: ignore[attr-defined]
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
            update_step = max(1, int(self.target_batch_size * 0.01))
            next_update_threshold = update_step
            while (
                self.finished_samples_count < self.target_batch_size
                and self.skipped_sample_count < self.target_batch_size
                and self.failed_sample_count < self.target_batch_size
            ):
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

                _, pending_tasks = await asyncio.wait(waiting_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
                self.finished_samples_count = await self.replay_buffer.get_finished_samples.remote()
                waiting_tasks = pending_tasks

            pbar.n = self.finished_samples_count
            pbar.refresh()

        if self.finished_samples_count >= self.target_batch_size:
            self.logger.info(
                f"Target batch size {self.target_batch_size} reached with {self.finished_samples_count} finished samples."
            )
        elif self.skipped_sample_count >= self.target_batch_size:
            self.logger.info(
                f"Stopping data generation as skipped samples {self.skipped_sample_count} reached target batch size {self.target_batch_size}."
            )
        elif self.failed_sample_count >= self.target_batch_size:
            self.logger.info(
                f"Stopping data generation as failed samples {self.failed_sample_count} reached target batch size {self.target_batch_size}."
            )

        if self.enable_partial_rollout:
            self.logger.info("Start pausing env controller.")
            await self.pause()
            while len(waiting_tasks) > 0:
                # NOTE: Keep sending pause requests because the inference engine only marks currently running requests as aborted.
                # When a waiting request starts running, it still needs another pause request to be marked as aborted.
                done_tasks, pending_tasks = await asyncio.wait(
                    waiting_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )
                if len(pending_tasks) > 0:
                    await self.pause()
                waiting_tasks = pending_tasks
            self.logger.info("All worker tasks have completed after pausing env controller.")

        replay_buffer_stats = await self.replay_buffer.print.remote()  # type: ignore[attr-defined]
        rollout_stats = await self.env_controller.get_rollout_stats.remote()  # type: ignore[attr-defined]
        self.logger.info(
            f"Data generation completed. Replay Buffer Stats: {replay_buffer_stats}, Rollout Stats: {rollout_stats}"
        )

    @ray_method
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
            self.logger.debug(f"All {succeeded_count} abort requests sent successfully.")

    @ray_method
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
        runner to collect a new batch of samples from the environment.

         Args:
            num (Optional[int]): The target number of samples to collect for this run.
                Overrides the existing global_batch_size in DataFlowConfig if provided.
            sample_params (Optional[SampleParams]): Parameters for model sampling.
                Overrides the existing sample_params in DataFlowConfig if provided.
            extra_params (Optional[Dict]): Additional parameters for rollout.
                Overrides the existing extra_params in DataFlowConfig if provided.
            enable_partial_rollout (Optional[bool]): Whether to enable partial rollout mode.
                This is primarily intended for unit testing, allowing the dataflow to pause
                and resume partway through a rollout for checkpointing and recovery tests.Returns:
        Returns:
            List[RLDataFlowItem]: A list of collected training samples.
        """
        self._reset_internal_states(global_batch_size=num, sample_params=sample_params, extra_params=extra_params)

        if resume:
            assert resume_path, "Resuming is enabled but no resume path is provided."
            self.logger.info(f"Resuming replay buffer from {resume_path}")
            await self.replay_buffer.resume_storage.remote(resume_path)

        await self.concurrent_task_runner()

        if dump:
            assert dump_path, "Dumping is enabled but no dump path is provided."
            self.logger.info(f"Dump replay buffer from {dump_path}")
            await self.replay_buffer.dump_storage.remote(dump_path)

        return await self.replay_buffer.get_samples.remote(self.target_batch_size)  # type: ignore[attr-defined]

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

    def save(self, save_path: Path | str):
        """Saves the replay buffer to the specified path.

        Args:
            save_path (str): The path to the checkpoint file to save to.
        """
        ray.get(self.replay_buffer.save.remote(save_path))

    def resume(self, resume_path: Path | str):
        """Resumes the replay buffer from the specified path.

        Args:
            resume_path (str): The path to the checkpoint file to resume from.
        """
        ray.get(self.replay_buffer.resume.remote(resume_path))


DataFlow = ray.remote(RawDataFlow)
DataFlowProxy = ActorProxy[RawDataFlow]

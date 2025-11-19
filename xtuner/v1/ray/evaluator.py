import asyncio
from pathlib import Path
from typing import Callable, List, Optional, Sized, Union

import ray
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from tqdm.auto import tqdm
from typing_extensions import Annotated

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RLDataFlowItem, RLDatasetItem, SampleParams, check_dataflow_item
from xtuner.v1.datasets import build_dataloader, build_datasets
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfigList
from xtuner.v1.ray.environment import BaseEnvironment
from xtuner.v1.ray.utils import create_task
from xtuner.v1.utils import get_logger


class EvaluatorConfig(BaseModel):
    """Configuration for the Evaluator in XTuner.

    This class defines the configuration parameters for model evaluation in XTuner, including four main aspects:

    - Dataset configuration: Specifies the evaluation dataset and tokenizer for text processing

    - Evaluator control logic: Manages concurrency levels and retry mechanisms for robust evaluation

    - Evaluation scheduling: Controls evaluation step intervals and sample size (either by ratio or absolute count)

    - Custom metric computation: Supports user-defined functions for specialized metric calculations

    Args:
        dataset_cfg (DatasetConfigList): Configuration for the evaluation dataset.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer used for text processing.
        evaluate_step (int): Step interval for triggering evaluation. Defaults to 1.
        eval_sample_ratio (float): Ratio of samples to evaluate from the generated samples. If > 0, overrides eval_sample_num. Defaults to 0 (use all samples).
        eval_sample_num (int): Number of samples to evaluate from the generated samples. Used if eval_sample_ratio is 0. Defaults to 0 (use all samples).
        max_concurrent (int): Maximum number of concurrent evaluation tasks. Defaults to 8.
        max_retry_times (int): Maximum number of retry attempts for failed evaluation tasks. Defaults to 2.
        compute_metric_func (Optional[Callable]): Optional function to compute or filter metrics for generated data groups. If None, uses default metric computation.

    **Examples:**

    Example configuration for evaluator with GSM8K dataset::

        from transformers import AutoTokenizer

        config = EvaluatorConfig(
            dataset_cfg=[{
                "dataset": DatasetConfig(name="gsm8k", anno_path="test_data.json"),
                "tokenize_fn": RLTokenizeFnConfig(max_length=512)
            }],
            tokenizer=AutoTokenizer.from_pretrained("model_path"),
            max_concurrent=32,
            eval_sample_ratio=0.8,  # Use 80% of samples
            evaluate_step=10,
            compute_metric_func=custom_accuracy_metric
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    enable_evaluate: Annotated[
        bool,
        Parameter(help="Flag to enable or disable evaluation during training."),
    ] = False
    enable_initial_evaluate: Annotated[
        bool,
        Parameter(help="Flag to enable or disable initial evaluation before training starts."),
    ] = False
    dataset_cfg: Annotated[
        DatasetConfigList,
        Parameter(help="Configuration for the dataset."),
    ]
    dataloader_cfg: Annotated[
        Optional[DataloaderConfig], Parameter(help="The PyTorch DataLoader for iterating over the dataset.")
    ] = None

    tokenizer: Annotated[
        Union[PreTrainedTokenizer, PreTrainedTokenizerFast, str],
        Parameter(help="Tokenizer for text processing."),
    ]
    max_concurrent: Annotated[
        int,
        Parameter(help="Maximum number of concurrent tasks."),
    ] = 512
    eval_sample_ratio: Annotated[
        float,
        Parameter(help="Ratio of samples to evaluate from the generated samples."),
    ] = 0
    eval_sample_num: Annotated[
        int,
        Parameter(help="Number of samples to evaluate from the generated samples."),
    ] = 0
    max_retry_times: Annotated[int, Parameter(help="Maximum number of retry attempts for failed tasks.")] = 2
    evaluate_step: Annotated[int, Parameter(help="Step interval for evaluation.")] = 1
    compute_metric_func: Annotated[
        Optional[Callable],
        Parameter(help="An optional function to filter or modify data groups after they are generated."),
    ] = None
    sample_params: Annotated[
        SampleParams,
        Parameter(help="Sampling parameters for evaluation."),
    ] = SampleParams()
    worker_log_dir: Annotated[Path, Parameter(help="Directory to save worker logs.")] = Path.cwd() / "work_dir"


@ray.remote
class Evaluator:
    """A Ray actor for evaluating a model's performance on a given dataset.

    The Evaluator generates responses using an environment controller or rollout controller, then it use default or
    custom computes metrics function to compute scores for generated samples. It returns the evaluation scores and
    generated samples.
    """

    def __init__(self, config: EvaluatorConfig, env_controller: BaseEnvironment):
        """Initialize the Evaluator.

        Args:
            config (EvaluatorConfig): The configuration for the evaluator.
            env_controller (EnvController): The environment controller used for
                generating responses.
        """
        self.config = config
        self.sample_params = self.config.sample_params
        self.dataset = (
            build_datasets(config.dataset_cfg, config.tokenizer)
            if isinstance(config.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
            else build_datasets(
                config.dataset_cfg, AutoTokenizer.from_pretrained(config.tokenizer, trust_remote_code=True)
            )
        )

        if config.dataloader_cfg is not None:
            self.dataloader_cfg = config.dataloader_cfg
        else:
            self.dataloader_cfg = DataloaderConfig(
                collator="fake_collator",
                pack_level="none",
            )
        self.dataloader = build_dataloader(
            dataloader_config=self.dataloader_cfg,
            datasets=self.dataset,
            global_batch_size=1,
            micro_batch_size=1,
            seed=1,
        )
        assert isinstance(self.dataloader, Sized)

        self.env_controller = env_controller
        self.failed_samples_count = 0
        self.return_list: List[RLDataFlowItem] = []
        if self.config.eval_sample_ratio > 0:
            self.eval_batch_size = int(len(self.dataloader) * self.config.eval_sample_ratio)
        elif self.config.eval_sample_num > 0:
            self.eval_batch_size = self.config.eval_sample_num
        else:
            self.eval_batch_size = len(self.dataloader)
        if self.config.compute_metric_func is not None:
            self.compute_metric = self.config.compute_metric_func
        else:
            self.compute_metric = self.default_compute_metric
        self.logger = get_logger(log_dir=config.worker_log_dir, tag="Evaluator")

    def default_compute_metric(self, samples):
        """Default metric computation function.

        Calculates accuracy based on whether the reward is positive.

        Args:
            samples (list): A list of RLDataFlowItem samples.

        Returns:
            dict: A dictionary containing the accuracy score.
        """
        return {"accuracy": sum(s.env.judger.reward["score"] > 0 for s in samples) / len(samples)}

    async def eval_worker_task(self, sample: RLDataFlowItem):
        """A single worker task to evaluate one sample.

        This task calls the environment controller to run the model on a
        sample. If it fails, it returns the sample with an incremented
        retry count.

        Args:
            sample (RLDataFlowItem): The data item to evaluate.

        Returns:
            RLDataFlowItem or None: The sample with retry information if it
                failed, or None if it succeeded or failed without a sample.
        """
        try:
            # note: In the evaluator, we convert the input sample to a list to adapt to the input format of single_turn_env
            group_sample = await self.env_controller.run.remote([sample], sample_params=self.sample_params)  # type: ignore[attr-defined]
            check_result, msg = check_dataflow_item(group_sample)
            if not check_result and len(group_sample) > 0:
                group_sample[0].extra_info.retry_times += 1
                self.logger.info(f"check_dataflow_item failed for {msg} and returning meta for retry.")
                return group_sample[0]
            self.return_list.append(group_sample[0])
        except Exception as e:
            self.logger.error(f"Worker task failed with exception: {e}. Returning meta for retry.")
            sample.extra_info.retry_times += 1
            return sample

    async def concurrent_eval_task_runner(self):
        """Runs evaluation tasks concurrently to generate a batch of samples.

        This method orchestrates the evaluation process by creating and managing
        a pool of asynchronous worker tasks. It continuously fetches data from
        the dataloader and submits evaluation tasks until the desired number of
        samples (`self.eval_batch_size`) has been successfully processed.
        """
        waiting_tasks = set()
        self.logger.info(f"Start to generate {self.eval_batch_size} samples for evaluate")
        self.logger.info(f"Evaluate sample parameters set to {self.sample_params}.")
        data_iter = iter(self.dataloader)
        with tqdm(total=self.eval_batch_size, desc="Rollout for eval samples") as pbar:
            update_step = max(1, int(self.eval_batch_size * 0.1))
            next_update_threshold = update_step
            while len(self.return_list) < self.eval_batch_size and self.failed_samples_count < self.eval_batch_size:
                if len(self.return_list) >= next_update_threshold:
                    pbar.n = len(self.return_list)
                    pbar.refresh()
                    next_update_threshold += update_step
                while len(waiting_tasks) < self.config.max_concurrent:
                    if len(self.return_list) + len(waiting_tasks) >= self.eval_batch_size:
                        break
                    try:
                        data = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self.dataloader)
                        data = next(data_iter)
                        self.logger.warning("Restarting the evaluation dataset.")
                    data_item = RLDataFlowItem(data=RLDatasetItem(**data[0]))
                    task = create_task(self.eval_worker_task(data_item))
                    waiting_tasks.add(task)

                if len(waiting_tasks) == 0:
                    break

                done_tasks, pending_tasks = await asyncio.wait(
                    waiting_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done_tasks:
                    result = task.result()
                    if result is not None:
                        if result.extra_info.retry_times < self.config.max_retry_times:
                            # If the retry count is less than max_retry_times, retry the task
                            retry_task = create_task(self.eval_worker_task(result))
                            pending_tasks.add(retry_task)
                        else:
                            self.logger.error(f"Max retry reached for {result.uid.action_id}. Not retrying.")
                            self.failed_samples_count += 1

                waiting_tasks = pending_tasks

            pbar.n = len(self.return_list)
            pbar.refresh()

        if self.failed_samples_count == self.eval_batch_size:
            self.logger.warning(
                f"{self.failed_samples_count} samples failed and were skipped. Pausing rollout controller."
            )
        else:
            self.logger.info(
                f"Target batch size reached, and {self.failed_samples_count} samples failed and were skipped. Pausing rollout controller."
            )
        ray.get(self.env_controller.pause.remote())

        if waiting_tasks:
            await asyncio.wait_for(asyncio.gather(*waiting_tasks, return_exceptions=True), timeout=10)

    async def run(self, return_samples=False):
        """Run the full evaluation process.

        This method resets the state, runs the concurrent task runner,
        computes the final metrics, and returns the results.

        Args:
            sample_params (Optional[SampleParams]): Sampling parameters for
                generation. Defaults to a greedy strategy.
            return_samples (bool): Whether to return the generated samples
                along with the scores. Defaults to False.

        Returns:
            dict or Tuple[dict, list]: The evaluation scores, and optionally
                the generated samples.
        """
        self.return_list = []
        ray.get(self.env_controller.restart.remote())  # type: ignore[attr-defined]
        await self.concurrent_eval_task_runner()
        if len(self.return_list) == 0:
            self.logger.warning("No valid samples were generated during evaluation.")
            return {} if not return_samples else ({}, [])
        scores = self.compute_metric(self.return_list)
        # To match the training format : each group's data is a list
        self.eval_samples = [[sample] for sample in self.return_list]
        if return_samples:
            return scores, self.eval_samples
        return scores

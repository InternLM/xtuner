import asyncio
from typing import Callable, List, Optional, Union

import ray
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from tqdm.auto import tqdm
from typing_extensions import Annotated

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.config import DatasetConfigList
from xtuner.v1.datasets import build_datasets
from xtuner.v1.datasets.data_item import RLTextDataItem
from xtuner.v1.ray.environment import BaseEnvironment
from xtuner.v1.ray.rollout import SampleParams
from xtuner.v1.ray.utils import create_task
from xtuner.v1.utils import get_logger


class EvaluatorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_cfg: Annotated[
        DatasetConfigList,
        Parameter(help="Configuration for the dataset."),
    ]
    tokenizer: Annotated[
        Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        Parameter(help="Tokenizer for text processing."),
    ]
    max_concurrent: Annotated[
        int,
        Parameter(help="Maximum number of concurrent tasks."),
    ] = 8
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


@ray.remote
class Evaluator:
    def __init__(self, config: EvaluatorConfig, env_controller: BaseEnvironment):
        self.config = config
        self.dataset = build_datasets(config.dataset_cfg, config.tokenizer)[0]
        self.dataloader = iter(self.dataset)
        self.env_controller = env_controller
        self.return_list: List[RLTextDataItem] = []
        if self.config.eval_sample_ratio > 0:
            self.eval_batch_size = int(len(self.dataset) * self.config.eval_sample_ratio)
        elif self.config.eval_sample_num > 0:
            self.eval_batch_size = self.config.eval_sample_num
        else:
            self.eval_batch_size = len(self.dataset)
        if self.config.compute_metric_func is not None:
            self.compute_metric = self.config.compute_metric_func
        else:
            self.compute_metric = self.default_compute_metric
        self.logger = get_logger()

    def default_compute_metric(self, samples):
        return {"accuracy": sum(s["reward"] > 0 for s in samples) / len(samples)}

    async def eval_worker_task(self, sample: RLTextDataItem):
        try:
            sample = await self.env_controller.run.remote(sample, self.sample_params)  # type: ignore[attr-defined]
            self.return_list.append(sample)
        except Exception as e:
            if sample is not None:
                self.logger.error(f"Worker task failed with exception: {e}. Returning meta for retry.", exc_info=True)
                if "retry_times" not in sample:
                    sample["retry_times"] = 0
                sample["retry_times"] += 1
                return sample
            else:
                self.logger.warning(f"Worker task failed with exception: {e}. No samples to return.")

    async def concurrent_eval_task_runner(self):
        waiting_tasks = set()
        self.logger.info(f"Start to generate {self.eval_batch_size} samples for evaluate")
        self.logger.info(f"Evaluate sample parameters set to {self.sample_params}.")
        with tqdm(total=self.eval_batch_size, desc="Rollout for eval samples") as pbar:
            update_step = max(1, int(self.eval_batch_size * 0.1))
            next_update_threshold = update_step
            while len(self.return_list) < self.eval_batch_size:
                if len(self.return_list) >= next_update_threshold:
                    pbar.n = len(self.return_list)
                    pbar.refresh()
                    next_update_threshold += update_step
                while len(waiting_tasks) < self.config.max_concurrent:
                    if len(self.return_list) + len(waiting_tasks) >= self.eval_batch_size:
                        break
                    try:
                        sample = next(self.dataloader)
                    except StopIteration:
                        break
                    task = create_task(self.eval_worker_task(sample))
                    waiting_tasks.add(task)

                assert len(waiting_tasks) > 0
                done_tasks, pending_tasks = await asyncio.wait(
                    waiting_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done_tasks:
                    result = task.result()
                    if result is not None:
                        if result["retry_times"] < self.config.max_retry_times:
                            # If the retry count is less than max_retry_times, retry the task
                            retry_task = create_task(self.eval_worker_task(result))
                            pending_tasks.add(retry_task)
                        else:
                            self.logger.error(f"Max retry reached for {result['prompt_id']}. Not retrying.")
                            self.failed_samples_count += 1

                waiting_tasks = pending_tasks

            pbar.n = len(self.return_list)
            pbar.refresh()

        self.logger.info("Target batch size reached. Pausing rollout controller.")
        ray.get(self.env_controller.pause.remote())

        if waiting_tasks:
            await asyncio.wait_for(asyncio.gather(*waiting_tasks, return_exceptions=True), timeout=10)

    async def run(self, sample_params: Optional[SampleParams] = None, return_samples=False):
        self.return_list = []
        self.dataloader = iter(self.dataset)
        self.sample_params = sample_params if sample_params else SampleParams()
        # set greedy sample for evaluator
        self.sample_params.temperature = 0.0
        self.sample_params.top_k = 1
        ray.get(self.env_controller.restart.remote())  # type: ignore[attr-defined]
        await self.concurrent_eval_task_runner()
        scores = self.compute_metric(self.return_list)
        # To match the training format : each group's data is a list
        self.eval_samples = [[sample] for sample in self.return_list]
        if return_samples:
            return scores, self.eval_samples
        return scores

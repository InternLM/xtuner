import asyncio
from typing import Callable, List, Tuple

import ray
import ray.util.queue
from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated

from xtuner.v1.ray.judger.controller import JudgerController
from xtuner.v1.ray.rollout.controller import RolloutController
from xtuner.v1.ray.utils import create_task
from xtuner.v1.utils import get_logger

from .replay_buffer import ReplayBuffer, ReplayMeta


class DataFlowConfig(BaseModel):
    env: Annotated[
        str,
        Parameter(help="Environment name to set for the dataflow."),
    ] = ""
    max_concurrent: Annotated[
        int,
        Parameter(help="Maximum number of concurrent tasks."),
    ] = 10
    prompt_repeat_k: Annotated[
        int,
        Parameter(help="Number of times to repeat each prompt."),
    ] = 1
    replay_ratio: Annotated[
        float,
        Parameter(help="Ratio of samples to replay from the buffer."),
    ] = 0
    replay_weights: Annotated[
        dict,
        Parameter(help="Weights for different states in the replay buffer."),
    ] = {}
    global_batch_size: Annotated[
        int,
        Parameter(help="Target number of samples to collect before stopping."),
    ] = 8
    partial_rollout: Annotated[
        bool,
        Parameter(help="Whether to use partial rollout for the rollout."),
    ] = False


class SampleParams(BaseModel):
    n: Annotated[int, Parameter(help="Number of samples to generate.")] = 1
    top_k: Annotated[
        int, Parameter(help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    ] = 50
    top_p: Annotated[float, Parameter(help="The cumulative probability for nucleus sampling.")] = 0.95
    temperature: Annotated[float, Parameter(help="The value used to module the next token probabilities.")] = 0.6
    repetition_penalty: Annotated[float, Parameter(help="The parameter for repetition penalty.")] = 1.0
    presence_penalty: Annotated[float, Parameter(help="The parameter for presence penalty.")] = 0.0
    frequency_penalty: Annotated[float, Parameter(help="The parameter for frequency penalty.")] = 0.0
    min_tokens: Annotated[int, Parameter(help="Minimum number of tokens to generate.")] = 2
    max_tokens: Annotated[int, Parameter(help="Maximum number of tokens to generate.")] = 2048
    stops: Annotated[List[str], Parameter(help="List of stop sequences.")] = []
    stop_token_ids: Annotated[List[int], Parameter(help="List of stop token IDs.")] = []
    logprobs: Annotated[int, Parameter(help="Number of log probabilities to return.")] = 0
    skip_special_tokens: Annotated[bool, Parameter(help="Whether to skip special tokens.")] = True


class DataProcessor:
    def __init__(self):
        pass

    def process(
        self, response_list: List[str], reward_list: List[int]
    ) -> tuple[List[str], List[int]]:  # response, reward
        # reward = 0
        # reward += sum(data for data in reward_list)
        # if reward == 0 or reward == len(reward_list):
        #     print("filter input data")
        #     return [], []
        return response_list, reward_list


@ray.remote
class DataFlow:
    def __init__(
        self,
        config: DataFlowConfig,
        replay_buffer: ReplayBuffer,
        data_processor: DataProcessor,
        rollout_controller: RolloutController,
        judger_controller: JudgerController,
        mapping_dataset_func: Callable[[ReplayMeta], Tuple[str, str]],
    ):
        self.config = config
        self.rollout_controller = rollout_controller
        self.judger_controller = judger_controller
        self.replay_buffer = replay_buffer
        self.mapping_dataset_func = mapping_dataset_func
        self.data_processor = data_processor
        self.tasks: List[asyncio.Task] = []
        self.return_list: List[str] = []
        self.send_samples = 0
        self.logger = get_logger()

    def sample(self) -> ReplayMeta:
        return ray.get(self.replay_buffer.sample.remote(self.config.replay_ratio, self.config.replay_weights))  # type: ignore[attr-defined]

    async def worker_task(self):
        # step 1: sample data and map to rollout and judge input
        replay_meta = self.sample()
        self.logger.debug(f"replay_meta: {replay_meta}")
        if replay_meta is None:
            raise RuntimeError("Failed to sample from replay buffer (got None).")
        rollout_input, reward_input = self.mapping_dataset_func(replay_meta)
        self.send_samples += 1
        self.logger.debug(f"send {self.send_samples} sample to env")

        # step 2: rollout
        rollout_futures = []
        observation_ref_list = ray.get(replay_meta.observation_refs)
        observation_id_list = ray.get(replay_meta.observation_ids)
        len_observation = (
            0 if observation_id_list is None and self.config.partial_rollout else len(observation_id_list)
        )
        remain_prompt_repeat_k = self.config.prompt_repeat_k - len_observation
        observation_id_list.extend([None] * remain_prompt_repeat_k)
        rollout_futures.extend(
            [
                self.rollout_controller.rollout.remote(rollout_input + observation_ref_list[i], SampleParams().dict())
                for i in range(len_observation)
            ]
        )
        rollout_futures.extend(
            [
                self.rollout_controller.rollout.remote(rollout_input, SampleParams().dict())
                for i in range(remain_prompt_repeat_k)
            ]
        )
        results = await asyncio.gather(*rollout_futures)
        states_list, rollout_list = zip(*[(r[1], r[0]) for r in results])

        # step 3: skip unfinished response
        if "unfinished" in states_list:
            replay_meta.update(
                observation_id_list=observation_id_list, observation_ref_list=rollout_list, state="unfinished"
            )
            await self.replay_buffer.add.remote(replay_meta)
            self.logger.debug("get unfinished response, put back to replay buffer")
            return

        # step 4: judger
        reward_futures = [
            self.judger_controller.judge.remote(rollout_res, reward_input) for rollout_res in rollout_list
        ]
        reward_list = await asyncio.gather(*reward_futures)
        # step5: filter
        filtered_res_list, filtered_reward_list = self.data_processor.process(rollout_list, reward_list)
        if len(filtered_res_list) > 0 and len(filtered_reward_list) > 0:
            replay_meta.update(
                observation_id_list=observation_id_list, observation_ref_list=filtered_res_list, state="filtered"
            )
            await self.replay_buffer.add.remote(replay_meta)
            self.return_list.append(
                {
                    "prompt": rollout_input,
                    "response": filtered_res_list,
                    "label": reward_input,
                    "reward": filtered_reward_list,
                }
            )
        self.logger.info(f"get finished response, put to return list, current size is {len(self.return_list)}")
        return

    async def concurrent_task_runner(self):
        failed = False
        while len(self.return_list) < self.config.global_batch_size:
            if len(self.tasks) < self.config.max_concurrent and not failed:
                self.tasks.append(create_task(self.worker_task()))
            done, pending = await asyncio.wait(self.tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                self.tasks.remove(finished)
                # 如果有失败的task，则不再创建新的task
                if finished.done() and finished.exception():
                    failed = True
                    self.logger.info("setting failed to True")
                    self.logger.error(f"Task {finished} failed with exception: {finished.exception()}")

        ray.get(self.rollout_controller.pause.remote())

        try:
            await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), timeout=10)
        except asyncio.TimeoutError:
            self.logger.error("gather tasks timeout!")

    async def run(self):
        self.return_list = []
        self.tasks = []
        await self.concurrent_task_runner()
        return self.return_list

    def shutdown(self):
        return ray.get(self.rollout_controller.shutdown.remote())

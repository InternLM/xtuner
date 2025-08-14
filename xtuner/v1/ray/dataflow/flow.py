import asyncio
from typing import List

import ray
import ray.util.queue
from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated

from xtuner.v1.ray.environment import EnvController
from xtuner.v1.ray.utils import create_task
from xtuner.v1.utils import get_logger

from .replay_buffer import ReplayBuffer, ReplayMeta


logger = get_logger()


class DataFlowConfig(BaseModel):
    env: Annotated[
        str,
        Parameter(help="Environment name to set for the dataflow."),
    ]
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
    ] = 1
    # todo(@duanyanhui): support partial_rollout logic in replaybuffer
    partial_rollout: Annotated[
        bool,
        Parameter(help="Whether to use partial rollout for the environment."),
    ] = False


class DataProcessor:
    def __init__(self):
        pass

    def process(self, data_list: List[tuple[str, str]]) -> List[tuple[str, str]]:  # response, reward
        # reward = 0
        # reward += sum(data[1] for data in data_list)
        # if reward == 0 or reward == len(data_list):
        #     return []
        return data_list


@ray.remote
class Flow:
    def __init__(
        self,
        config: DataFlowConfig,
        environment: EnvController,
        replay_buffer: ReplayBuffer,
        data_processor: DataProcessor,
        mapping_dataset_func,
    ):
        self.config = config
        self.environment = environment
        self.replay_buffer = replay_buffer
        self.mapping_dataset_func = mapping_dataset_func
        self.data_processor = data_processor
        self.tasks: List[asyncio.Task] = []
        self.return_list: List[str] = []
        self.send_samples = 0

    def sample(self) -> ReplayMeta:
        return ray.get(self.replay_buffer.sample.remote(self.config.replay_ratio, self.config.replay_weights))  # type: ignore[attr-defined]

    async def worker_task(self):
        replay_meta = self.sample()
        if replay_meta is None:
            raise RuntimeError("Failed to sample from replay buffer (got None).")
        rollout_input, reward_input = self.mapping_dataset_func(replay_meta)
        self.send_samples += 1
        logger.debug(f"send {self.send_samples} sample to env")
        observation_ref_list = ray.get(replay_meta.observation_refs)
        observation_id_list = ray.get(replay_meta.observation_ids)
        len_observation = 0 if observation_id_list is None else len(observation_id_list)
        remain_prompt_repeat_k = self.config.prompt_repeat_k - len_observation
        observation_id_list.extend([None] * remain_prompt_repeat_k)
        # send prompt with observation
        result_futures = [
            self.environment.run.remote(rollout_input + observe, reward_input) for observe in observation_ref_list
        ]
        # send prompt without observation
        result_futures.extend(
            [self.environment.run.remote(rollout_input, reward_input) for i in range(remain_prompt_repeat_k)]
        )
        results = await asyncio.gather(*result_futures)

        if results:
            states_list, response_list = zip(*[(r["state"], (r["response"], r.get("reward"))) for r in results])
        else:
            states_list, response_list = [], []

        if "unfinished" in states_list:
            replay_meta.update(
                observation_id_list=observation_id_list, observation_ref_list=response_list, state="unfinished"
            )
            await self.replay_buffer.add.remote(replay_meta)
            logger.debug("get unfinished response, put back to replay buffer")
            return
        else:
            filtered_res_list = self.data_processor.process(response_list)
            if len(filtered_res_list) > 0:
                replay_meta.update(
                    observation_id_list=observation_id_list, observation_ref_list=filtered_res_list, state="filtered"
                )
                await self.replay_buffer.add.remote(replay_meta)
                self.return_list.append(
                    {
                        "prompt": rollout_input,
                        "response": [res[0] for res in filtered_res_list],
                        "label": reward_input,
                        "reward": [res[1] for res in filtered_res_list],
                    }
                )
            logger.info(f"get finished response, put to return list, current size is {len(self.return_list)}")
            return

    async def concurrent_task_runner(self):
        failed = False

        while len(self.return_list) < self.config.global_batch_size and not failed:
            if len(self.tasks) < self.config.max_concurrent:
                self.tasks.append(create_task(self.worker_task()))
            done, pending = await asyncio.wait(self.tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                self.tasks.remove(finished)
                # 如果有失败的task，则不再创建新的task
                if finished.done() and finished.exception():
                    failed = True
                    logger.error(f"Task {finished} failed with exception: {finished.exception()}")
                    break

        ray.get(self.environment.pause.remote())

        try:
            await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), timeout=10)
        except asyncio.TimeoutError:
            logger.error("gather tasks timeout!")

    async def run(self):
        self.return_list = []
        self.tasks = []
        await self.concurrent_task_runner()
        return self.return_list

    def shutdown(self):
        return ray.get(self.environment.shutdown.remote())

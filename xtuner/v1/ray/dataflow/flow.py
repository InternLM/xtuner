import asyncio
import json
from typing import List

import ray
import ray.util.queue
from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated

from xtuner.v1.ray.environment import EnvController
from xtuner.v1.ray.utils import create_task
from xtuner.v1.utils import get_logger

from .replay_buffer import ReplayBuffer


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

    async def sample(self):
        return await self.replay_buffer.sample.remote(self.config.replay_ratio, self.config.replay_weights)

    async def worker_task(self):
        replay_meta = await self.sample()
        try:
            rollout_input, reward_input = self.mapping_dataset_func(replay_meta)
        except Exception as e:
            logger.error(f"Error occurred while mapping dataset: {e}")
            return
        self.send_samples += 1
        logger.info(f"send {self.send_samples} response to env")
        results = [
            await self.environment.run.remote(rollout_input, reward_input) for i in range(self.config.prompt_repeat_k)
        ]
        parsed_results = [json.loads(result) for result in results]
        if parsed_results:
            states_list, response_list = zip(*[(r["state"], (r["response"], r.get("reward"))) for r in parsed_results])
        else:
            states_list, response_list = [], []

        if "unfinished" in states_list:
            replay_meta.update(observation_ref_list=response_list, state="unfinished")
            await self.replay_buffer.add.remote(replay_meta)
            logger.debug("get unfinished response, put back to replay buffer")
            return
        else:
            filtered_res_list = self.data_processor.process(response_list)
            if len(filtered_res_list) > 0:
                replay_meta.update(observation_ref_list=filtered_res_list, state="filtered")
                await self.replay_buffer.add.remote(replay_meta)
                self.return_list.append(
                    json.dumps(
                        {
                            "prompt": rollout_input,
                            "response": [res[0] for res in filtered_res_list],
                            "label": reward_input,
                            "reward": [res[1] for res in filtered_res_list],
                        }
                    )
                )
            logger.info(f"get finished response, put to return list, current size is {len(self.return_list)}")
            return

    async def concurrent_task_runner(self):
        while len(self.return_list) < self.config.global_batch_size:
            if len(self.tasks) < self.config.max_concurrent:
                self.tasks.append(create_task(self.worker_task()))
            done, pending = await asyncio.wait(self.tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                self.tasks.remove(finished)

        ray.get(self.environment.pause.remote())
  
        try:
            result = await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), timeout=10)
            print("result: ", result)
        except asyncio.TimeoutError:
            logger.error("gather tasks timeout!")

    async def run(self):
        await self.concurrent_task_runner()
        return self.return_list

    def shutdown(self):
        return ray.get(self.environment.shutdown.remote())

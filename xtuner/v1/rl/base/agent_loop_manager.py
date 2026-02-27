from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto import Status
from xtuner.v1.rl.base.producer import ProduceStrategy, Sampler
from xtuner.v1.rl.base.replay_buffer import ReplayBuffer

from .agent_loop import AgentLoop


class AgentLoopManager:
    def __init__(
        self,
        agent_loop: AgentLoop,
        produce_strategy: ProduceStrategy,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
        task_name: str,
    ):
        self._agent_loop: AgentLoop = agent_loop  # 负责一条或者一组样本生成
        # TODO(@duanyanhui): ProduceStrategy是用config来build还是直接使用实例，后续再统一改，目前先使用实例
        self._scheduler: ProduceStrategy = produce_strategy
        self._replay_buffer: ReplayBuffer = replay_buffer
        self._data_sampler: Sampler = sampler  # 负责采样数据，提供给 ProduceStrategy 来生成样本
        self.task_name = task_name

    # 共卡
    async def produce_batch(self, batch_size: int):
        await self._scheduler.produce_batch(
            self._agent_loop, self._data_sampler, self._replay_buffer, batch_size, self.task_name
        )
        return await self._replay_buffer.get(batch_size, self.task_name, Status.COMPLETED)

    # # 非共卡
    # async def disaggregate_produce_batch(self, batch_size: int):
    #     self._scheduler.produce_batch(batch_size, self._data_sampler, ...)

    # async def disaggregate_get_batch(self, task_name: str, batch_size: int):
    #     # 从不同的 replay_buffer 中采样，然后训练
    #     return self._replay_buffer.get(batch_size, task_name, Status.COMPLETED)


class AgentLoopManagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_name: str

    def build(
        self,
        agent_loop: AgentLoop,
        produce_strategy: ProduceStrategy,
        sampler: Sampler,
        replay_buffer: ReplayBuffer,
    ) -> AgentLoopManager:
        return AgentLoopManager(
            agent_loop=agent_loop,
            produce_strategy=produce_strategy,
            sampler=sampler,
            replay_buffer=replay_buffer,
            task_name=self.task_name,
        )

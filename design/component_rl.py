################################### imports ######################################
from typing import Any, Callable
import asyncio
from enum import Enum
from torch.utils.data import DataLoader
import threading
from typing import List
from collections import deque

from xtuner.v1.ray.rollout.controller import SampleParams
from xtuner.v1.data_proto.rl_data import SampleParams  # TODO: 删掉一个？
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.loss.base_loss_ctx import BaseLossContext

def load_tokenizer(hf_checkpoint, trust_remote_code=True): ...
def load_processor(hf_checkpoint, trust_remote_code=True): ...

class PlacementGroup: ...

def log_metrics(metrics: dict): ...

class TrainItem:
    seq_ctx: SequenceContext
    loss_ctxs: BaseLossContext # 考虑更通用的多 loss 场景，时间原因暂时不改


################################### Main components ######################################
class Status(Enum):
    INIT = "init"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class RolloutState:  # RolloutState:
    # message: list
    tokens: list[int] # 每一次实际输入
    
    uid: int
    session_id: int | None = None
    prompt_ids: list[int]
    response: str
    response_ids: list[int] # 每一次实际输出，覆盖写
    logprobs: list[float] 
    routed_experts: list[int] | None = None
    reward: float | list[float] | list[dict] | None = None
    loss_mask: list[int] | None = None # tokens + response_ids的长度
    state: Status = Status.INIT
    sample_parms: SampleParams | None = None
    tools: list | None = None
    tool_choice: str | None = None
    mm_infer_info: dict[str, Any]
    mm_train_info: dict[str, Any]
    finish_reason: str | None = None
    staleness: int = 0
    extra_fields: dict[str, Any] = {}


class RolloutController:
    async def generate(self, rollout_state: RolloutState) -> RolloutState: ...


class Judge:
    def judge(self, rollout_state: RolloutState) -> RolloutState: ...


# 负责一条和一组轨迹生成，非常简单
class AgentLoop:  
    def __init__(self, rollout_ctl: RolloutController, hf_checkpoint, sample_params=SampleParams(), judge_cfg: dict = None) -> None:
        self.rollout_ctl = rollout_ctl
        self.hf_checkpoint = hf_checkpoint
        self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
        self.sample_params = sample_params
        self.judge = Judge() if judge_cfg is not None else None
        self.task_name = 'aa'

    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState: ...

    async def generate_group(self, rollout_states: list[RolloutState]) -> list[RolloutState]:
        pending_tasks = []

        for rollout_state in rollout_states:
            task = asyncio.create_task(self.generate_sample(rollout_state))
            pending_tasks.append(task)
        
        generated_samples = asyncio.gather(*pending_tasks)

        group_samples = await generated_samples
        return group_samples


class SingleTurnAgentLoop(AgentLoop):
    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState:
        rollout_state = await self.rollout_ctl.generate(rollout_state)
        if self.judge is not None:
            rollout_state = self.judge.judge(rollout_state)
        return rollout_state


class MultiTurnAgentLoop(AgentLoop):
    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState:
        ...


# 中心化管理所有 rollout 过程中的数据，暂时训练中途数据不会放到其中，后续可能会统一为全局数据管理器
# 只管理数据，不控制数据
# 后续可能会抽象一层 backend interface，支持不同存储后端
# 是否需要是 ray 对象？

class Storage:
    async def put(self, items: list[RolloutState], storage_indices: StorageIndices): ...
    async def get(self, count: int, storage_indices: StorageIndices) -> list[RolloutState]: ...
    def __len__(self): ...

class FIFOBackend(Storage):  # 同步RL用
    limit: int = 0
    _storage: deque[RolloutState] = deque(maxlen=limit)


class StalenessBackend(Storage):  # 异步RL用
    max_staleness : int
    min_staleness : int

class BaseReplayBuffer:
    def __init__(self, storage_backend: Storage):
        self._storage = storage_backend
    
    async def put(self, items: list[RolloutState], task_name: str, **kwargs): ...

    async def get(self, batch_size: int, task_name: str, group_state: Status, **kwargs) -> list[RolloutState]: ...
          
class Sampler:
    prompt_k: int

    async def sample(self) -> list[RolloutState]: ...

class SamplerWithReplayBuffer(Sampler):
    replay_buffer: BaseReplayBuffer
    async def sample(self) -> list[RolloutState]: ...

class ProduceStrategy:  # Scheduler负责调度多个样本的生成，里面可以有超发、异步、重排长短样本等优化
    replay_buffer: BaseReplayBuffer

    async def produce_batch(self, agent_loop: AgentLoop, sampler: Sampler, batch_size: int): ...


class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(self, agent_loop: AgentLoop, sampler: Sampler, batch_size: int):
        # TODO: 将 batch_size 封装成 while stop_condition ?
        data_concurrency = batch_size
        
        rollout_states = await sampler.sample()
        
        pending_tasks = []
        for _ in range(data_concurrency):
            task = asyncio.create_task(agent_loop.generate_group(rollout_states))
            pending_tasks.append(task)

        completed_sample_count = 0
        while completed_sample_count < data_concurrency:
            if not pending_tasks:
                print("All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)

            # 如果要过滤，在这个地方处理，然后加入到 replay buffer
            # 如果被过滤的数据就放到 put_to_filtered pool 中
            for task in done_tasks:
                try:
                    await self.replay_buffer.put(task.result(), task_name=agent_loop.task_name)
                    completed_sample_count += 1
                except Exception as e:
                    print(f"Error in generating trajectory: {e}")


class AsyncProduceStrategy(ProduceStrategy):
    staleness_threshold: float = 0.0
    enable_partial_rollout: bool = False
    tail_batch_trigger_size: int = 0
    tail_batch_candidate_step: int = 0

    async def produce_batch(self, agent_loop: AgentLoop, sampler: SamplerWithReplayBuffer, batch_size: int):
        # hack sampler with replay buffer
        rollout_state = await sampler.sample()
        ...
        

# 支持单 task
class AgentLoopManager:
    def __init__(self, agent_loop: AgentLoop, produce_strategy: ProduceStrategy, sampler: Sampler, replay_buffer: BaseReplayBuffer):
        # 一一绑定
        self._agent_loop: AgentLoop = agent_loop # 负责一条或者一组样本生成
        self._produce_strategy: ProduceStrategy = produce_strategy # 负责一批样本生成+调度
        self._sampler: Sampler = sampler # 负责采样
        self._replay_buffer: BaseReplayBuffer = replay_buffer
    
    # 共卡
    async def produce_batch(self, batch_size: int):
        await self._produce_strategy.produce_batch(self._agent_loop, self._sampler, batch_size)
        return await self._replay_buffer.get(batch_size, task_name=self._agent_loop.task_name, group_state=Status.COMPLETED)
    
    # 非共卡
    async def disaggregate_produce_batch(self, batch_size: int):
        # 起一个单独线程不断生成
        self._produce_strategy.produce_batch(self._agent_loop, self._sampler, batch_size)

    async def disaggregate_get_batch(self, batch_size: int):
        # 从不同的 replay_buffer 中采样，然后训练
        return await self._replay_buffer.get(batch_size, task_name=self._agent_loop.task_name, group_state=Status.COMPLETED)


# 多 task 自己写
class MulitiAgentLoopManager(AgentLoopManager):
    def __init__(self, 
                 agent_loop_managers: list[AgentLoopManager]):
          self._agent_loop_managers = agent_loop_managers

    async def produce_batch(self, batch_size: int):
        pass

    async def disaggregate_produce_batch(self, batch_size: int):
        self._produce_strategy[0].produce_batch(self._agent_loop, self._sampler, batch_size)
        self._produce_strategy[1].produce_batch(self._agent_loop, self._sampler, batch_size)

    async def disaggregate_get_batch(self, batch_size: int):
        pass


# 1. grpo 算法是在 TrainController 中调用AdvantageEstimator
# 2. ppo 算法是在 Trainworker 中调用AdvantageEstimator，并新增额外方法实现(如 compute_ref_logprobs, compute_values)，无需重写 TrainController 和 Trainworker
class AdvantageEstimator:
    def compute_advantages(self, batch: list[TrainItem]) -> list[TrainItem]: ...


# ppo 算法是通过在 Trainworker 中新增额外方法实现，无需重写 TrainController 和 Trainworker
class TrainController:
    advantage_estimator: AdvantageEstimator
    # high level API
    def fit(self, batch: list[TrainItem]) -> dict: ...
    def train(self, batch: list[TrainItem]) -> dict: ...


class CheckpointEngine:
    rollout_ctl: RolloutController
    train_ctl: TrainController
    def update_weights(self): ...

class Evaluator:  # 根据rollout输出的batch，计算评估指标。本身并不负责rollout。
    def evaluate(self, batch: list[RolloutState]) -> dict: ...


################################### Usage example with components #########################################
# 弱化Trainer：Trainer中代码尽量少，尽量用componet来组织代码。下面是几种典型Trainer的组织方式。

class Packer:
    def pack_pad_dispatch(self, samples: list[RolloutState]) -> list[TrainItem]: ...


def main_colocate_with_train_highlevel():
    # rollout_ctl, train_ctl, data_mgr, env, evaluator等对象都是主进程中本地对象，并不是ray actor。这样：
    # 1. 保证一大部分的数据传递无需跨机传输，方便统一管理
    # 2. 减少ray引入的debug和维护难度
    pg: PlacementGroup
    rollout_ctl: RolloutController(pg)
    train_ctl: TrainController(pg)
    checkpoint_engine: CheckpointEngine(rollout_ctl, train_ctl)

    global_batch_size: int
    dataloader: DataLoader
    hf_checkpoint: str
    agent_loop: AgentLoop = AgentLoop(rollout_ctl, hf_checkpoint)
    produce_strategy: AsyncProduceStrategy = AsyncProduceStrategy(replay_buffer)
    sampler: SamplerWithReplayBuffer
    replay_buffer: BaseReplayBuffer
    env: AgentLoopManager = AgentLoopManager(agent_loop, produce_strategy, sampler, replay_buffer)

    eval_batch_size: int
    eval_produce_strategy: ProduceStrategy = SyncProduceStrategy(eval_replay_buffer)
    eval_sampler: Sampler = Sampler(replay_buffer)
    eval_replay_buffer: BaseReplayBuffer
    eval_env: AgentLoopManager = AgentLoopManager(agent_loop, eval_produce_strategy, eval_sampler, eval_replay_buffer)
    evaluator: Evaluator
    total_rollouts: int

    for i in range(total_rollouts):
        train_batch: list[RolloutState] = asyncio.run(env.produce_batch(global_batch_size))

        train_batch = Packer.pack_pad_dispatch(train_batch)

        metrics = train_ctl.fit(train_batch)
        log_metrics(metrics)

        checkpoint_engine.update_weights()

        eval_batch: list[RolloutState] = asyncio.run(eval_env.produce_batch(eval_batch_size))
        eval_metrics = evaluator.evaluate(eval_batch)
        log_metrics(eval_metrics)
        

def main_colocate_with_train_lowlevel():
    data_mgr: DataManager
    pg: PlacementGroup
    rollout_ctl: RolloutController(pg)
    env: AgentLoopManager(rollout_ctl)
    train_ctl: TrainController(pg)
    checkpoint_engine: CheckpointEngine(rollout_ctl, train_ctl)

    eval_data_mgr: DataManager
    evaluator: Evaluator
    total_rollouts: int

    for i in range(total_rollouts):
        asyncio.run(env.produce_batch(data_mgr))

        batch: list[TrainItem] = data_mgr.get_batch()

        # below is equivalent to train_ctl.fit(batch)
        batch = Packer.pack_pad_dispatch(batch)
        batch = train_ctl.compute_old_logprobs(batch)
        batch = train_ctl.compute_ref_logprobs(batch)
        batch = train_ctl.compute_values(batch)
        batch = train_ctl.compute_advantages(batch)  # TODO: AdvEstimator
        metrics = train_ctl.train(batch)

        log_metrics(metrics)

        checkpoint_engine.update_weights()

        env.produce_batch(eval_data_mgr)
        eval_metrics = evaluator.evaluate(eval_data_mgr.get_batch())
        log_metrics(eval_metrics)
        

def main_separate():
    data_mgr: DataManager
    pg1: PlacementGroup
    rollout_ctl: RolloutController(pg1)
    # pg1_2: PlacementGroup
    # rollout_ctl_2: RolloutController(pg1_2)
    env: AgentLoopManager(rollout_ctl)  # Environment(rollout_ctl, rollout_ctl_2)

    pg2: PlacementGroup
    train_ctl: TrainController(pg2)
    checkpoint_engine: CheckpointEngine(rollout_ctl, train_ctl)

    eval_data_mgr: DataManager
    evaluator: Evaluator

    producer_thread = threading.Thread(target=env.produce_loop, args=(data_mgr,))
    producer_thread.start()

    total_rollouts: int
    for i in range(total_rollouts):
        batch: list[TrainItem] = data_mgr.get_batch()
        metrics = train_ctl.fit(batch)
        log_metrics(metrics)

        checkpoint_engine.update_weights()

        env.produce_batch(eval_data_mgr)  # 优先级高于env.produce_loop
        eval_metrics = evaluator.evaluate(eval_data_mgr.get_batch())
        log_metrics(eval_metrics)

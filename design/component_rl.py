################################### imports ######################################
from typing import Any, Callable
import asyncio
from enum import Enum
from torch.utils.data import DataLoader
import threading

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
    loss_ctx: BaseLossContext


################################### Main components ######################################
class Status(Enum):
    INIT = "init"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    SKIPPED = "skipped"


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


class RolloutController:
    async def generate_sample(self, sample: RolloutState) -> RolloutState: ...


class Judge:
    def judge(self, sample: RolloutState) -> RolloutState: ...


class Agent:  # Agent负责一条轨迹样本的生成
    async def generate_sample(self, sample: RolloutState) -> RolloutState: ...
    async def generate_group(self, sample_fn: Callable[[], list[RolloutState]], data_mgr: "DataManager") -> list[RolloutState]: ...

class SingleTurnAgent(Agent):
    def __init__(self, rollout_ctl: RolloutController, hf_checkpoint, sample_params=SampleParams(), judge_cfg: dict = None) -> None:
        # persistent state for the generation process
        self.rollout_ctl = rollout_ctl
        self.hf_checkpoint = hf_checkpoint
        self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
        self.sample_params = sample_params
        self.judge = Judge() if judge_cfg is not None else None

    async def generate_sample(self, sample: RolloutState) -> RolloutState:
        sample = await self.rollout_ctl.generate_sample(sample)
        if self.judge is not None:
            sample = self.judge.judge(sample)
        return sample

    async def generate_group(self, sample_fn: Callable[[], list[RolloutState]], data_mgr: "DataManager") -> Status:
        pending_tasks = []

        group_samples: list[RolloutState] = sample_fn()  # list of prompt_k Sample
        for sample in group_samples:
            task = asyncio.create_task(self.generate_sample(sample))
            pending_tasks.append(task)
        
        generated_samples = asyncio.gather(*pending_tasks)

        group_samples = await generated_samples
        data_mgr.add_to_replay_buffer(group_samples)
        return Status.COMPLETED


class MultiTurnAgent(Agent):
    ...


class MultiTurnToolAgent(Agent):
    ...


class DataManager:
    dataloader: DataLoader
    replay_buffer: list[list[RolloutState]]

    def sample_from_dataset(self) -> list[RolloutState]: ...  # get from dataloader

    def add_to_replay_buffer(self, samples: list[RolloutState]): ...

    def get_batch(self) -> list[TrainItem]: ...  # get from replay_buffer and convert to TrainItem

class ProduceStrategy:  # Scheduler负责调度多个样本的生成，里面可以有超发、异步、重排长短样本等优化
    async def produce_batch(self, batch_size: int, data_mgr: DataManager, agent: Agent): ...

class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(self, batch_size: int, data_mgr: DataManager, agent: Agent):
        data_concurrency = batch_size

        pending_tasks = []
        for _ in range(data_concurrency):
            task = asyncio.create_task(agent.generate_group(data_mgr.sample_from_dataset, data_mgr))
            pending_tasks.append(task)

        completed_sample_count = 0
        while completed_sample_count < data_concurrency:
            if not pending_tasks:
                print("All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            for task in done_tasks:
                try:
                    status: Status =  await task
                    if status == Status.COMPLETED:
                        completed_sample_count += 1
                except Exception as e:
                    print(f"Error in generating trajectory: {e}")


class AsyncProduceStrategy(ProduceStrategy):
    def __init__(
            self, 
            staleness_threshold: float = 0.0,
            enable_partial_rollout: bool = False,
            tail_batch_trigger_size: int = 0,
            tail_batch_candidate_step: int = 0,
        ):
        class _Buffer: ...
        self.buffer = _Buffer(enable_partial_rollout, tail_batch_candidate_step, tail_batch_trigger_size)

    async def produce_batch(self, batch_size: int, data_mgr: DataManager, agent: Agent):
        # hack sample_fn from data_mgr.sample_from_dataset and self.buffer.sample()
        pass


class Environment:
    def __init__(self, rollout_ctl: RolloutController):
        self._agent: Agent = SingleTurnAgent(rollout_ctl)
        self._scheduler: ProduceStrategy = SyncProduceStrategy()
    
    async def produce_batch(self, data_mgr: DataManager, batch_size: int):
        await self._scheduler.produce_batch(batch_size, data_mgr, self._agent)

    def produce_loop(self, data_mgr: DataManager):
        pass


class TrainController:
    # high level API
    def fit(self, batch: list[TrainItem]) -> dict: ...
    # low level API
    def compute_old_logprobs(self, batch: list[TrainItem]) -> list[TrainItem]: ...
    def compute_ref_logprobs(self, batch: list[TrainItem]) -> list[TrainItem]: ...
    def compute_values(self, batch: list[TrainItem]) -> list[TrainItem]: ...
    def compute_advantages(self, batch: list[TrainItem]) -> list[TrainItem]: ...
    def train(self, batch: list[TrainItem]) -> dict: ...
    def sync_weights(self, rollout_ctl: RolloutController): ...


class Evaluator:  # 根据rollout输出的batch，计算评估指标。本身并不负责rollout。
    def evaluate(self, batch: list[RolloutState]) -> dict: ...


################################### Usage example with components #########################################
# 弱化Trainer：Trainer中代码尽量少，尽量用componet来组织代码。下面是几种典型Trainer的组织方式。

def main_colocate_with_train_highlevel():
    # rollout_ctl, train_ctl, data_mgr, env, evaluator等对象都是主进程中本地对象，并不是ray actor。这样：
    # 1. 保证一大部分的数据传递无需跨机传输，方便统一管理
    # 2. 减少ray引入的debug和维护难度
    pg: PlacementGroup
    rollout_ctl: RolloutController(pg)
    train_ctl: TrainController(pg)

    data_mgr: DataManager
    env: Environment(rollout_ctl)
    eval_data_mgr: DataManager
    evaluator: Evaluator
    total_rollouts: int

    for i in range(total_rollouts):
        env.produce_batch(data_mgr)

        train_batch: list[TrainItem] = data_mgr.get_batch()
        metrics = train_ctl.fit(train_batch)
        log_metrics(metrics)

        train_ctl.sync_weights(rollout_ctl)

        env.produce_batch(eval_data_mgr)
        eval_metrics = evaluator.evaluate(eval_data_mgr.get_batch())
        log_metrics(eval_metrics)
        

class Packer:
    def pack_pad_dispatch(self, samples: list[RolloutState]) -> list[RolloutState]: ...

def main_colocate_with_train_lowlevel():
    data_mgr: DataManager
    pg: PlacementGroup
    rollout_ctl: RolloutController(pg)
    env: Environment(rollout_ctl)
    train_ctl: TrainController(pg)

    eval_data_mgr: DataManager
    evaluator: Evaluator
    total_rollouts: int

    for i in range(total_rollouts):
        env.produce_batch(data_mgr)

        batch: list[TrainItem] = data_mgr.get_batch()

        # below is equivalent to train_ctl.fit(batch)
        batch = Packer.pack_pad_dispatch(batch)
        batch = train_ctl.compute_old_logprobs(batch)
        batch = train_ctl.compute_ref_logprobs(batch)
        batch = train_ctl.compute_values(batch)
        batch = train_ctl.compute_advantages(batch)  # TODO: AdvEstimator
        metrics = train_ctl.train(batch)

        log_metrics(metrics)

        train_ctl.sync_weights(rollout_ctl)

        env.produce_batch(eval_data_mgr)
        eval_metrics = evaluator.evaluate(eval_data_mgr.get_batch())
        log_metrics(eval_metrics)
        

def main_separate():
    data_mgr: DataManager
    pg1: PlacementGroup
    rollout_ctl: RolloutController(pg1)
    pg1_2: PlacementGroup
    rollout_ctl_2: RolloutController(pg1_2)
    env: Environment(rollout_ctl, rollout_ctl_2)

    pg2: PlacementGroup
    train_ctl: TrainController(pg2)

    eval_data_mgr: DataManager
    evaluator: Evaluator

    producer_thread = threading.Thread(target=env.produce_loop, args=(data_mgr,))
    producer_thread.start()

    total_rollouts: int
    for i in range(total_rollouts):
        batch: list[TrainItem] = data_mgr.get_batch()
        metrics = train_ctl.fit(batch)
        log_metrics(metrics)

        train_ctl.sync_weights(rollout_ctl)

        env.produce_batch(eval_data_mgr)  # 优先级高于env.produce_loop
        eval_metrics = evaluator.evaluate(eval_data_mgr.get_batch())
        log_metrics(eval_metrics)

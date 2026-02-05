import asyncio
from .simple_env_runner import SimpleEnvRunner
from .rollout_state import RolloutState, Status
from typing import List, Dict
from collections import defaultdict
from .simple_env_runner import DataSampler

# 用户无感
class ExpiredBuffer:
    def __init__(self):
        self.buffer: List[List[RolloutState]] = []

    def add(self, rollout_state: List[RolloutState]):
        for rs in rollout_state:
            rs.response = ""
            rs.response_ids = []
            rs.logprobs = []
            if rs.routed_experts is not None:
                # 需要注意新的routed_experts最好不用用ray._internal.free
                del res.router_experts
            rs.routed_experts = None
            
            # 2. 重置评价与统计字段
            rs.reward = None
            rs.staleness = 0
            
            # 3. 重置生命周期状态
            from .rollout_state import Status
            rs.state = Status.INIT
        self.buffer.append(rollout_state)

    def pop(self) -> list[RolloutState]:
        assert self.buffer, "ExpiredBuffer is empty!"
        return self.buffer.pop(0)


class AbortedBuffer:
    def __init__(self):
        self.buffer: Dict[int, List[List[RolloutState]]] = defaultdict(list)

    def add(self, rollout_state: List[RolloutState]):
        group_staleness = max([rs.staleness for rs in rollout_state]) 
        self.buffer[group_staleness].append(rollout_state)

    def pop(self, enable_partial_rollout) -> list[RolloutState]:
        assert self.buffer, "AbortedBuffer is empty!"
        highest_staleness = max(self.buffer.keys())
        rollout_states = self.buffer[highest_staleness]
        data = rollout_states.pop(0)
        if enable_partial_rollout:
            for rs in data:
                rs.tokens = rs.prompt_ids + rs.response_ids
                rs.sample_params.max_tokens = rs.sample_params.max_tokens - len(rs.response_ids)
        else:
            for rs in data:
                rs.response = ""
                rs.response_ids = []
                rs.logprobs = []
                if rs.routed_experts is not None:
                    del rs.routed_experts
                rs.routed_experts = None
                rs.reward = None
                rs.staleness = 0
                from .rollout_state import Status
                rs.state = Status.INIT
        return data
    
    def update(self):
        new_buffer = defaultdict(list)
        for staleness, rollout_states in self.buffer.items():
            new_staleness = staleness + 1
            new_buffer[new_staleness].extend(rollout_states)
        self.buffer = new_buffer

class CompletedBuffer:
    def __init__(self):
        self.buffer: Dict[int, List[List[RolloutState]]] = defaultdict(list)

    def add(self, rollout_state: List[RolloutState]):
        group_staleness = max([rs.staleness for rs in rollout_state]) 
        self.buffer[group_staleness].append(rollout_state)
    
    def pop(self) -> list[RolloutState]:
        highest_staleness = max(self.buffer.keys())
        rollout_states = self.buffer[highest_staleness]
        return rollout_states.pop(0)

    def update(self):
        new_buffer = defaultdict(list)
        for staleness, rollout_states in self.buffer.items():
            new_staleness = staleness + 1
            new_buffer[new_staleness].extend(rollout_states)
        self.buffer = new_buffer

    @property
    def length(self) -> int:
        return sum(len(v) for v in self.buffer.values())
    
class Buffer:
    # 这个功能独立的作为一个类的想法是：expired buffer 和 aborted buffer 可能会有不同的优先级管理，
    # 例如，我们现在是根据版本进行管理，版本越旧的样本越先出队，可能未来还有其他的方式，例如按照长度？
    # 同时，将过期的样本的管理进行独立，使代码可读性更高一点
    # 每个变量控制的内容也要更加独立：
    # enable_partial_rollout: 下次rollout是否进行拼接 
    # tail_batch_candidate_step: 多老的样本才会进入expired buffer
    # tail_batch_trigger_size: expired buffer的触发采样大小阈值

    def __init__(self, 
                 enable_partial_rollout: bool = False,
                 tail_batch_candidate_step: int = 1,
                 tail_batch_trigger_size: int = 10):
        self.expired_buffer: ExpiredBuffer = ExpiredBuffer()
        self.aborted_buffer: AbortedBuffer = AbortedBuffer()
        self.completed_buffer: CompletedBuffer = CompletedBuffer()
        self.enable_tail_batch = tail_batch_candidate_step > 0
        self.enable_partial_rollout = enable_partial_rollout
        self.tail_batch_candidate_step = tail_batch_candidate_step

    def add(self, rollout_state: List[RolloutState]):
        # rollout_state的版本管理放在哪里？例如一次权重更新后，对Buffer里所有样本的版本进行一次更新，而不是在这里进行更新
        group_staleness = max([rs.staleness for rs in rollout_state])
        group_states = [rs.state for rs in rollout_state]
        if self.enable_tail_batch and group_staleness > self.tail_batch_candidate_step:
            self.expired_buffer.add(rollout_state)
        elif all(state == Status.COMPLETED for state in group_states):
            self.completed_buffer.add(rollout_state)
        else:  
            self.aborted_buffer.add(rollout_state)

    def get_sample_func(self, data_sampler: DataSampler) -> RolloutState:
        use_expired = self.enable_tail_batch and len(self.expired_buffer) > 0
        self.update()

        def _sample():
            if use_expired and self.expired_buffer:
                return self.expired_buffer.pop()
            elif self.aborted_buffer:
                return self.aborted_buffer.pop(self.enable_partial_rollout)
            else:
                return data_sampler.sample_from_dataset()
        
        return _sample 
    
    def update(self):
        if self.enable_partial_rollout:
            self.completed_buffer.update()
        else:
            while self.completed_buffer.length > 0:
                state = self.completed_buffer.pop()
                self.aborted_buffer.add(state)
        self.aborted_buffer.update()


class AsyncProxyEnvRuner:
    def __init__(
            self, 
            staleness_threshold: float = 0.0,
            enable_partial_rollout: bool = False,
            tail_batch_trigger_size: int = 0,
            tail_batch_candidate_step: int = 0,
        ):
        self.base_env_runner: SimpleEnvRunner = None
        self.buffer = Buffer(enable_partial_rollout, tail_batch_candidate_step, tail_batch_trigger_size)
        self.staleness_threshold = staleness_threshold

    def set_base_env_runner(self, base_env_runner: SimpleEnvRunner):
        self.base_env_runner = base_env_runner

    # 这个方法应该可以实现所有异步功能的
    async def async_generate_batch(self, 
                                    data_sampler: DataSampler,
                                    batch_size: int,
                                    prompt_repeat_k: int,
                                    ):
        # 基于当前内部管理的状态，就可以下一次应该从哪个池子中采样
        # 高度内聚功能模块
        last_step_remain_completed_samples = self.buffer.completed_buffer.length
        data_concurrency = (1 + self.staleness_threshold) * (batch_size - last_step_remain_completed_samples)
        completed_sample_count = 0
        sample_func = self.buffer.get_sample_func(data_sampler, prompt_repeat_k)

        pending_tasks = []

        for _ in range(last_step_remain_completed_samples):
            traj = self.buffer.completed_buffer.pop()
            yield traj

        for _ in range(data_concurrency):
            task = asyncio.create_task(self.base_env_runner.generate_group(sample_func)) 
            # task = asyncio.create_task(self.generate_group(data_sampler.sample(), prompt_repeat_k)) 
            pending_tasks.append(task)

        while completed_sample_count < data_concurrency:
            if not pending_tasks:
                print("All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            for task in done_tasks:
                try:
                    traj =  await task
                    if traj is not None:
                        completed_sample_count += 1
                        if completed_sample_count <= batch_size:
                            yield traj
                        else:
                            self.buffer.add(traj)
                except Exception as e:
                    print(f"Error in generating trajectory: {e}")
        
        await self.base_env_runner.rollout_controller.pause()
        while len(pending_tasks) > 0:
            done_tasks, pending_tasks = await asyncio.wait(pending_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
            for task in done_tasks:
                try:
                    abort_traj = await task
                    self.buffer.add(abort_traj)
                except Exception as e:
                    print(f"Error while pausing task: {e}")
            if len(pending_tasks) > 0:
                await self.base_env_runner.rollout_controller.pause()
                await asyncio.sleep(1)

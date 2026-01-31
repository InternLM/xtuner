import asyncio
from .simple_env_runner import SimpleEnvRunner
from .rollout_state import RolloutState

# 用户无感
class AsyncProxyEnvRuner:

    def __init__(self):
        self.base_env_runner: SimpleEnvRunner = None
        
        # 先简单点写
        self.expired_buffer: list[RolloutState] = []
        self.aborted_buffer: list[RolloutState] = []

    def set_base_env_runner(self, base_env_runner: SimpleEnvRunner):
        self.base_env_runner = base_env_runner
    
    def sample_from_expired_buffer(self) -> RolloutState:
        pass

    def sample_from_aborted_buffer(self) -> RolloutState:
        pass
    
    # 这个方法应该可以实现所有异步功能的
    async def async_generate_batch(self, 
                                    batch_size: int,
                                    prompt_repeat_k: int,
                                    # 这些可能是类输入参数，而不是通过参数传入
                                    staleness_threshold: float = 0.0,
                                    enable_partial_rollout: bool =False,
                                    ) -> list[RolloutState]:
        # 基于当前内部管理的状态，就可以下一次应该从哪个池子中采样
        # 高度内聚功能模块
        data_concurrency = (1+staleness_threshold)*batch_size

        # 仅仅考虑 partial_rollout 场景
        pending_tasks = []
        if enable_partial_rollout:
            # 先从 abort buffer 里采样
            for _ in range(len(self.aborted_buffer)):
                rollout_state = self.sample_from_aborted_buffer()
                task = asyncio.create_task(self.generate_group(rollout_state, prompt_repeat_k))
                pending_tasks.append(task)

        data_concurrency -= len(pending_tasks)    
        for _ in range(data_concurrency):
            # 最后从数据集中采样
            rollout_state = self.sample_from_dataset()
            task = asyncio.create_task(self.generate_group(rollout_state, prompt_repeat_k))
            pending_tasks.append(task)
        
        completed_sample_count = 0
        batch_trajectories = []
        while completed_sample_count < batch_size:
            if not pending_tasks:
                print("All tasks are done but not enough samples collected.")
                break
            done_tasks, pending_tasks = await asyncio.wait(pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            for task in done_tasks:
                try:
                    traj =  await task
                    if traj is not None:
                        batch_trajectories.append(traj)
                        completed_sample_count += 1
                except Exception as e:
                    print(f"Error in generating trajectory: {e}")
        
        # 被 abort 的样本放入 buffer
        # 好像并不是设置啥额外的例如 save_partial 这种方法来保存中间内容，因为所有东西都可以在 rollout_state 里复原才对。
        # 即使 agent 内部有一套复杂的格式，只要他返回的 rollout_state 带有这部分信息，那就可以复原的。
        self.aborted_buffer.extend([ts for ts in batch_trajectories if ts.is_aborted()])

        return batch_trajectories # 返回的数据一定是可以训练的

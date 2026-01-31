

import asyncio
from xtuner.v1.datasets import DataloaderConfig
from .utils import load_function
from .rollout_controller import RolloutController
from .rollout_state import ProcessorUtilState, RolloutState



# 这个类负责所以定义接口，同时提供一个满足大部分需求的同步运行 rollout。支持单轮，多轮，agent 等场景
# 异步功能我们假设有两套完全不同的实现，则分别继承这个类进行扩展即可
class SimpleEnvRunner:
    def __init__(self,
                rollout_controller: RolloutController,
                processor_utils_state: ProcessorUtilState | None = None,
                judger: callable | None = None, # none 是为了这个 envruner 可以独立运行, 可以是简单的 callable, 也可以是 actor worker
                dataloader_cfg: DataloaderConfig | None = None, # none 是为了这个 envruner 可以独立运行
                generate_external: callable | None = None, 
                # 最理想状态是：这个类用户是完全无感的，用于只要基于 simple_env_runner 定制化自己的逻辑后
                # 然后传入类似这个 proxy 类就可以实现一种异步策略，实现解耦目的
                async_proxy_runner = None, # 用于异步场景的代理 runner
            ):
        self.rollout_controller = rollout_controller
        self.judger = judger
        self.processor_utils_state = processor_utils_state

        self.dataloader = None
        if dataloader_cfg is not None:
            self.dataloader = dataloader_cfg.build()  

        self.generate_external = generate_external
        if self.generate_external is not None:
            self.generate_external = load_function(self.generate_external)
        
        self.async_proxy_runner = async_proxy_runner
        if self.async_proxy_runner is not None:
            # 循环引用，会不会有问题
            self.async_proxy_runner.set_base_env_runner(self)
    
    def sample_from_dataset(self) -> RolloutState:
        try:
            data = next(self.dataloader_iter)[0] 
        except StopIteration:
            self.cur_epoch += 1
            self.dataloader.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.dataloader)
            data = next(self.dataloader_iter)[0]
        return data

    # 生成一条样本
    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState:
        # 默认走最简单的单轮模式
        rollout_state = await self.rollout_controller.generate(rollout_state)
            
        reward = 0.0
        if self.judger is not None:
            if asyncio.iscoroutinefunction(self.judger):
                reward = await self.judger(rollout_state)
            else:
                reward = self.run_judger_worker(rollout_state)
            rollout_state.reward = reward
        return rollout_state
    
    async def run_judger_worker(self,rollout_state):
        # 可能有多个 judge worker,此时就涉及到调度问题
        # 用户可以重载这个方法，自定义自己调度策略。
        reward = self.judger(rollout_state)
        return reward

    async def generate(self, rollout_state: RolloutState) -> RolloutState:
        if self.generate_external is not None:
            # TODO: 如果走这个分支，估计没有走 partial rollout
            return await self.generate_external(rollout_state, self.processor_utils_state, self.rollout_controller, self.judger)
        else:
            return await self.generate_sample(rollout_state)
    
    # 生成一组样本
    async def generate_group(self, rollout_state: RolloutState, prompt_repeat_k: int) -> list[RolloutState]:
        pending_tasks = []
        for _ in range(prompt_repeat_k):
            task = asyncio.create_task(self.generate(rollout_state))
            pending_tasks.append(task)
        
        trajectories = asyncio.gather(*pending_tasks)
        return await trajectories
    
    # 不可打断式生成一批样本，用于同步场景
    async def generate_batch(self, 
                             batch_size: int, 
                             prompt_repeat_k: int,
                             ) -> list[RolloutState]:
        data_concurrency = batch_size
        assert self.dataloader is not None, "Dataloader must be provided for batch generation."

        pending_tasks = []
        for _ in range(data_concurrency):
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
        
        return batch_trajectories
    
    # =====================================================================
    # 以下接口都是异步 rollout 相关的接口

    # 用于可中断生成场景
    async def async_generate_batch(self, 
                                    batch_size: int, 
                                    prompt_repeat_k: int,
                                    staleness_threshold: float = 0.0,
                                    enable_partial_rollout: bool =False,
                                    ) -> list[RolloutState]:
        return await self.async_proxy_runner.async_generate_batch(
                                                            batch_size, 
                                                            prompt_repeat_k, 
                                                            staleness_threshold, 
                                                            enable_partial_rollout)

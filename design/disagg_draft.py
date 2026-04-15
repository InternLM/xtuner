"""非共卡训练方案的伪代码草稿。

这个文件不是可运行实现，而是把本次设计里的关键接口改动、
状态机、以及主流程串起来，方便后续正式编码时对照。
"""

from dataclasses import dataclass
from enum import Enum, auto
import asyncio


class RLDisaggregatedTrainer:
    def fit(self):
        # 对外仍保留同步接口，兼容当前 CLI 的 `trainer.fit()` 调用方式。
        # 内部真实逻辑放到 async `_fit()`，这样才能同时管理后台 producer task。
        return asyncio_run(self._fit())

    async def _fit(self):
        # 非共卡训练的核心思路：
        # 1. 单独起一个后台生产循环，不断往 replay buffer 填数据；
        # 2. 前台训练循环按需从 buffer 取 batch；
        # 3. 到权重同步点时，显式打断 producer，等待 pending rollout 收尾；
        # 4. 完成权重同步后，再恢复 producer。
        producer_task = create_task(
            self.agent_loop_manager.produce_loop(
                batch_size=self.train_batch_size,
                start_rollout_step=self._cur_step,
            )
        )
        try:
            for rollout_step in range(self._cur_step + 1, self._total_train_steps + 1):
                # 训练侧只负责“取数”，不再直接驱动 rollout 生成。
                # 真正的生成已经在后台 `produce_loop()` 中异步进行。
                produce_result = await self.agent_loop_manager.get_batch(
                    self.train_batch_size,
                    rollout_step=rollout_step,
                )

                # `EXPIRED_BATCH` 表示：
                # 当前 rollout 使用的模型权重已经整体过期，无法再产出满足 staleness
                # 约束的一整批训练样本。此时直接跳过训练，进入权重同步阶段。
                if produce_result.status != ProduceBatchStatus.EXPIRED_BATCH:
                    # 正常路径：从 replay buffer 里取到可训练 batch，
                    # 先做数据整理，再下发到 train controller。
                    train_data = self._prepare_train_data(produce_result.rollout_states, ...)
                    await self.train_controller.fit.remote(train_data, ...)

                # 无论这一轮是否真正训练，只要到了同步点，都执行一次权重更新流程。
                # 这样在 `EXPIRED_BATCH` 情况下，producer 才能尽快拿到更新后的权重继续工作。
                await self._sync_weights_and_save(rollout_step)

                if self._need_eval(rollout_step):
                    # 设计上让 eval 优先级高于 background producer：
                    # 先 eval，再 reset producer，避免评测和后台生成竞争 rollout 资源。
                    await self._run_eval(rollout_step)

                # reset 的语义不是“清空一切”，而是：
                # 1. 清除 update_event；
                # 2. 状态回到 NORMAL；
                # 3. 记录 rollout 侧刚刚切换到的最新模型版本步数。
                self.agent_loop_manager.reset(model_rollout_step=rollout_step)
        finally:
            # 训练结束时，显式通知后台 producer 退出，避免留下悬空协程。
            self.agent_loop_manager._status = AgentLoopManagerStatus.FINISH
            self.agent_loop_manager._finish_event.set()
            await producer_task

    async def _sync_weights_and_save(self, rollout_step: int):
        # 权重同步前必须先停止 rollout 生成：
        # 否则 rollout 侧可能仍在使用旧权重继续产生样本，导致 staleness 失控。
        pause_time_s = await self.agent_loop_manager.cleanup_pending_tasks(for_weight_update=True)
        self.agent_loop_manager._pause_time_s = pause_time_s
        # 这里先用 fake 接口占位，后续再替换成真实的跨卡权重同步实现。
        self.fake_update_weights()

    def fake_update_weights(self):
        pass


class ProduceBatchStatus(Enum):
    """单次 producer 调度的返回状态。

    这是“某次调度调用”的结果，不是整个 AgentLoopManager 的全局状态。

    状态含义：
    - NORMAL:
        本次调度是正常结束的。可能已经向 buffer 放入了新样本，
        或者发现 buffer 中已有足够可用样本。
    - UPDATE_ABORT:
        外部触发了权重更新，producer 应尽快停止继续补发新任务，
        并把剩余 pending rollout 留给外层显式 cleanup。
    - EXPIRED_BATCH:
        以当前 rollout 模型版本继续生成，已经无法满足 staleness 约束，
        因而不能再得到一个有效 batch，必须等待权重更新。
    """

    NORMAL = auto()
    UPDATE_ABORT = auto()
    EXPIRED_BATCH = auto()


class AgentLoopManagerStatus(Enum):
    """AgentLoopManager 的全局运行状态。

    可以把它理解为 producer 主循环的状态机：
    - 初始为 NORMAL
    - NORMAL --(开始权重更新/cleanup)--> UPDATE_ABORT
    - UPDATE_ABORT --(reset 后)--> NORMAL
    - NORMAL --(整体过期，无法再 produce)--> EXPIRED_BATCH
    - EXPIRED_BATCH --(开始权重更新/cleanup)--> UPDATE_ABORT
    - 任意状态 --(训练结束)--> FINISH
    """

    NORMAL = auto()
    UPDATE_ABORT = auto()
    EXPIRED_BATCH = auto()
    FINISH = auto()


@dataclass
class ProduceBatchResult:
    """训练侧一次取 batch 的结果。

    这里的结果是“消费侧视角”的结果：
    - `rollout_states` 是最终交给训练的数据
    - `status` 反映这次 `get_batch()` 是否拿到了有效 batch
    - timing / leftover 统计用于日志与诊断
    """

    rollout_states: list[list]
    status: ProduceBatchStatus = ProduceBatchStatus.NORMAL
    group_gen_count: int | None = None
    group_gen_mean_s: float | None = None
    group_gen_p50_s: float | None = None
    group_gen_p99_s: float | None = None
    group_gen_p99_p50_ratio: float | None = None
    group_gen_pause_time_s: float | None = None
    leftover_completed: int = 0
    leftover_aborted: int = 0
    leftover_expired: int = 0
    task_batch_sizes: dict[str, int] | None = None
    task_results: dict[str, "ProduceBatchResult"] | None = None


def refresh_seq_staleness(group: list, current_rollout_step: int) -> list:
    """在复用旧的 COMPLETED leftover 前，按当前训练步重新计算 staleness。

    设计动机：
    - 非共卡场景下，producer 和 trainer 解耦后，buffer 中的数据会“存活更久”；
    - 不能只看样本上一次写入时的 `seq_staleness`，因为那只是历史快照；
    - 所以在真正复用前，需要结合当前训练步重新计算一次，决定是否仍然可用。
    """
    return group


class PartialRolloutHandler:
    def postprocess(self, rollout_state, model_rollout_step: int):
        # 改动点：
        # `response_rollout_steps` 不再记录“producer 当前循环跑到了第几轮”，
        # 而是记录“这段 token 对应的模型权重版本步数”。
        # 这样 staleness 的语义才稳定，尤其适合非共卡下 producer 连续运行的场景。
        return rollout_state


class SingleTurnAgentLoop:
    async def generate_sample(self, rollout_state, **kwargs):
        # 改动点：
        # rollout sample 的后处理中，使用 `model_rollout_step` 写 token 来源版本。
        model_rollout_step = kwargs["model_rollout_step"]
        enable_partial_rollout = kwargs.get("enable_partial_rollout", False)
        return rollout_state


class AsyncProduceStrategy:
    def __init__(self, ...):
        # 改动点：
        # pending tasks 不再是局部变量，而是 strategy 的持久状态。
        # 因为非共卡下 `produce_batch()` 会被反复调用，
        # 调用之间未收尾的 rollout 任务需要跨轮保存并显式 cleanup。
        self._pending_tasks: set[asyncio.Task] = set()

    async def cleanup_pending_tasks(self, agent_loop, replay_buffer, task_name: str) -> float:
        """公开的 pending 清理接口。

        设计动机：
        - 旧逻辑由 `AsyncProduceStrategy.produce_batch()` 在内部自动 cleanup；
        - 新逻辑把 cleanup 提升为外层显式调用；
        - 这样 trainer 就可以在权重同步前主动打断 rollout，并等待尾部任务收尾。

        返回值：
        - `pause_time_s`，表示本次停止并回收 pending 任务的耗时，
          供后续日志统计使用。
        """
        return 0.0

    async def produce_batch(
        self,
        agent_loop,
        sampler,
        replay_buffer,
        batch_size: int,
        task_name: str,
        *,
        rollout_step: int,
        model_rollout_step: int,
        update_event: asyncio.Event,
    ) -> ProduceBatchStatus:
        # 核心流程：
        # 1. 先检查上一轮遗留在 `self._pending_tasks` 中、已经完成的任务，
        #    回收到 replay buffer，避免这些结果丢失。
        # 2. 对 buffer 中旧的 COMPLETED leftovers 按当前训练步重算 staleness，
        #    真正过期的转成 EXPIRED。
        # 3. 如果当前 buffer 中已经有足够多的 COMPLETED 样本，直接返回 NORMAL，
        #    不必额外发新 rollout。
        # 4. 如果当前模型版本已经老到不可能再产出满足要求的一整批数据，
        #    则返回 EXPIRED_BATCH，交给外部触发权重更新。
        # 5. 否则继续异步补发 rollout 任务，直到：
        #    - 收集到足够多的 completed groups，或
        #    - 外部设置了 `update_event`，此时返回 UPDATE_ABORT。
        # 6. 每个 group 的生成耗时不再通过返回值上抛，
        #    而是写入 `rollout_state.extra_fields["group_generate_time_s"]`，
        #    后续由 manager 在 get_batch 阶段重新聚合统计。
        return ProduceBatchStatus.NORMAL


class AgentLoopManager:
    def __init__(self, ...):
        # 当 trainer 准备做权重更新时置位，用来通知 producer 停止继续补任务。
        self._update_event = asyncio.Event()

        # 训练结束时置位，用来让 `produce_loop()` 正常退出。
        self._finish_event = asyncio.Event()

        # 当前 rollout 侧正在使用的模型权重，对应的是哪一次 rollout_step 更新得到的。
        self._model_rollout_step = 0

        # AgentLoopManager 的全局运行状态，见 `AgentLoopManagerStatus` 注释。
        self._status = AgentLoopManagerStatus.NORMAL

        # 最近一次 cleanup_pending_tasks 的耗时，下一次 get_batch 时带给上层并清零。
        self._pause_time_s = 0.0

    async def _produce_batch_to_buffer(self, batch_size: int, rollout_step: int) -> ProduceBatchStatus:
        # 这是新的“只生产、不取数”的内部工具函数。
        #
        # 单 task：
        # - 直接调用该 task 对应的 AsyncProduceStrategy.produce_batch()
        #
        # 多 task：
        # - 先按照权重分配各 task 的 batch_size
        # - 再并发 gather 各 task 的 produce_batch()
        # - 最后聚合出一个总的 ProduceBatchStatus
        return ProduceBatchStatus.NORMAL

    async def cleanup_pending_tasks(self, for_weight_update: bool = False) -> float:
        if for_weight_update:
            # 权重更新场景下，先发停止信号，再切 manager 全局状态。
            # 后续 producer loop 看到这个信号，会停止继续补任务。
            self._update_event.set()
            self._status = AgentLoopManagerStatus.UPDATE_ABORT

        # 单 task / 多 task 两种情况下，都统一下沉到各 task strategy 的
        # `cleanup_pending_tasks()`，由具体 strategy 去回收 pending rollout。
        return 0.0

    async def _get_batch_from_buffer(self, batch_size: int, rollout_step: int) -> ProduceBatchResult:
        # 这是新的“只取数、不生产”的内部工具函数。
        #
        # 核心职责：
        # 1. 从 replay buffer 中按 task_name / COMPLETED 取出训练 batch；
        # 2. 从每个 group 的 extra_fields 中重建 timing 统计；
        # 3. 把上一轮 cleanup 产生的 `self._pause_time_s` 挂到结果里；
        # 4. 返回给 trainer 训练侧消费。
        return ProduceBatchResult(rollout_states=[])

    def reset(self, model_rollout_step: int) -> None:
        # reset 表示“权重已经同步完成，producer 可以恢复工作了”。
        # 这里不重置 replay buffer，只重置控制状态。
        self._update_event.clear()
        self._status = AgentLoopManagerStatus.NORMAL
        self._model_rollout_step = model_rollout_step

    async def produce_batch(self, batch_size: int, rollout_step: int = 0) -> ProduceBatchResult:
        # 共卡路径仍然保留这个接口，但内部改成三段式：
        # 1. `_produce_batch_to_buffer()`：调度 rollout，把数据写入 buffer
        # 2. `cleanup_pending_tasks()`：显式收尾 pending rollout
        # 3. `_get_batch_from_buffer()`：再从 buffer 中把 batch 取出来
        #
        # 这样共卡和非共卡都会复用同一组底层工具函数。
        return ProduceBatchResult(rollout_states=[])

    async def produce_loop(self, batch_size: int, start_rollout_step: int = 0) -> None:
        # 非共卡新增的后台生产循环。
        #
        # 它不直接返回 batch，而是不断往 replay buffer 里“喂数据”。
        # 训练侧通过 `get_batch()` 异步消费这些数据。
        rollout_step = start_rollout_step
        while not self._finish_event.is_set():
            status = await self._produce_batch_to_buffer(batch_size, rollout_step)

            if self._status == AgentLoopManagerStatus.FINISH:
                break

            if status == ProduceBatchStatus.EXPIRED_BATCH:
                # 当前权重版本整体过期，先挂起 producer，等待 trainer 做权重更新。
                self._status = AgentLoopManagerStatus.EXPIRED_BATCH
                await asyncio.sleep(0.1)
                continue

            if status == ProduceBatchStatus.UPDATE_ABORT:
                # 这里不自己 cleanup，避免和 trainer 的同步流程重复收尾。
                # producer 只需要停下来，等待外部 reset 即可。
                await asyncio.sleep(0.1)
                continue

            # 只有在正常产出数据时，producer 自己维护的 rollout_step 才递增。
            rollout_step += 1

    async def get_batch(self, batch_size: int, rollout_step: int) -> ProduceBatchResult:
        if self._status == AgentLoopManagerStatus.EXPIRED_BATCH:
            # 特殊语义：
            # 当 manager 已进入 EXPIRED_BATCH，全局表示“当前权重下已无法拿到有效 batch”，
            # 所以训练侧不再从 buffer 里强行取数，而是直接收到一个状态信号，
            # 然后跳去做权重更新。
            return ProduceBatchResult(
                rollout_states=[],
                status=ProduceBatchStatus.EXPIRED_BATCH,
            )

        # 正常情况下，训练侧只是一个 buffer 消费者。
        return await self._get_batch_from_buffer(batch_size, rollout_step)

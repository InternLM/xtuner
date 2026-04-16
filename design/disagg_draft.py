"""非共卡训练方案的伪代码草稿。

这个文件不是可运行实现，而是把本次设计里的关键接口改动、
状态机、以及主流程串起来，方便后续正式编码时对照。
"""

from dataclasses import dataclass
from enum import Enum, auto
import asyncio


class RLDisaggregatedTrainer:
    # 复用 colocate trainer 的整体目录约定，便于后续接入现有 meta / auto_resume 逻辑。
    _CHECKPOINT_DIR = "checkpoints"
    _SAVE_TRAIN_STATE_PATH = "train_state.json"

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
                # 当前 rollout 使用的模型权重已经过旧。
                # 在这个设计里，一旦当前 rollout model 过旧，就立即停止本轮训练消费，
                # 直接进入权重同步阶段，优先让 rollout 侧尽快切到新权重，
                # 避免 rollout 占用的卡继续空等。
                if produce_result.status != ProduceBatchStatus.EXPIRED_BATCH:
                    # 正常路径：只有在 producer 没有报告 `EXPIRED_BATCH` 时，
                    # 才继续从 replay buffer 中取 batch 做训练。
                    #
                    # 一旦 producer 报告当前 rollout model 已经过旧，
                    # 即使 buffer 里可能还残留旧 completed 数据，也不再优先消费，
                    # 而是直接推进权重更新。
                    #
                    # 这样做的目的是尽快让 rollout 侧恢复工作，而不是为了多榨取一点旧数据。
                    #
                    # 从 replay buffer 里取到可训练 batch 后，
                    # 先做数据整理，再下发到 train controller。
                    train_data = self._prepare_train_data(produce_result.rollout_states, ...)
                    await self.train_controller.fit.remote(train_data, ...)

                # 无论这一轮是否真正训练，只要到了同步点，都执行一次权重更新流程。
                # 这样在 `EXPIRED_BATCH` 情况下，producer 才能尽快拿到更新后的权重继续工作。
                await self.agent_loop_manager.pause_product(for_weight_update=True)
                await self._sync_weights_and_save(rollout_step)

                if self._need_eval(rollout_step):
                    # 设计上让 eval 优先级高于 background producer：
                    # 先 eval，再 continue_product，避免评测和后台生成竞争 rollout 资源。
                    await self._run_eval(rollout_step)

                # continue_product 的语义不是“清空一切”，而是：
                # 1. 清除 update_event；
                # 2. 状态回到 NORMAL；
                # 3. 记录 rollout 侧刚刚切换到的最新模型版本步数。
                self.agent_loop_manager.continue_product(model_rollout_step=rollout_step)
        finally:
            # 训练结束时，显式通知后台 producer 退出，避免留下悬空协程。
            self.agent_loop_manager._status = AgentLoopManagerStatus.FINISH
            self.agent_loop_manager._finish_event.set()
            await producer_task

    async def _sync_weights_and_save(self, rollout_step: int):
        # 这里默认外层已经先调用过 `pause_product(for_weight_update=True)`，
        # 所以进入这个函数时，producer 已经停下，系统处于静止态。
        #
        # checkpoint 的安全保存点放在这里：
        # 1. producer 已经停下，pending rollout 已收尾；
        # 2. replay buffer 不再被后台并发写入；
        # 3. trainer 正准备进入权重同步。
        #
        # 这样保存出来的是一个“静止态”快照，恢复时不需要考虑未完成的 rollout task。
        self._maybe_save_checkpoint(rollout_step)

        # 这里先用 fake 接口占位，后续再替换成真实的跨卡权重同步实现。
        self.fake_update_weights()

    def fake_update_weights(self):
        pass

    def _maybe_save_checkpoint(self, rollout_step: int):
        # 设计目标：沿用 colocate trainer 的三层保存结构。
        #
        # 1. AgentLoopManager.save(...)
        #    - 保存 task sampler 状态
        #    - 保存 replay buffer
        #    - 保存 manager 自身的控制状态
        #    - 必须显式传入 `model_rollout_step_override=rollout_step`
        #      作为 resume 后 rollout 应恢复到的目标权重版本
        #    - 原因是 checkpoint 的保存时机在：
        #      `pause_product(for_weight_update=True)` 之后、
        #      `continue_product(model_rollout_step=rollout_step)` 之前
        #    - 也就是说，此时 manager 内部原有的 `self._model_rollout_step`
        #      仍然还是“旧 rollout 权重版本”，还没被 continue_product
        #      推进到新的 `rollout_step`
        #    - 如果这里不显式 override，而是直接保存旧的
        #      `self._model_rollout_step`，那么 resume 后 producer 会以旧模型版本
        #      继续恢复流程，语义就和保存前预期不一致
        #
        # 2. train_controller.save(...)
        #    - 保存 train model / optimizer 等训练态
        #
        # 3. trainer_state.json
        #    - 保存 `cur_step`
        #    - 可额外保存 `global_train_step`
        #    - 可额外重复保存 `model_rollout_step` 便于诊断
        #
        # 注意：
        # - save 时不保存 eval manager 状态，延续现有 colocate 语义；
        # - save 前要求 producer 已 pause 完成，所有 strategy._pending_tasks 为空。
        pass

    def _resume_from_checkpoint(self, checkpoint_path: str):
        # resume 的源头仍然是 train checkpoint + replay buffer + sampler 状态。
        #
        # 恢复顺序建议：
        # 1. train_controller.resume(...)
        # 2. saved_model_rollout_step = agent_loop_manager.resume(...)
        # 3. 用 train 侧权重重新同步 rollout（当前先走 fake_update_weights）
        # 4. agent_loop_manager.continue_product(
        #        model_rollout_step=saved_model_rollout_step
        #    )
        #
        # 这里的关键点是：
        # - resume 取回的 `saved_model_rollout_step` 应该是 checkpoint 保存时
        #   显式 override 进去的“新 rollout_step”
        # - 而不是 save 那一刻 manager 内部还没来得及 continue_product 的旧
        #   `self._model_rollout_step`
        #
        # 这里不会恢复 producer_task 本身；
        # producer loop 会在 fit() 启动时重新 create_task。
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
        并把剩余 pending rollout 留给外层显式 pause。
    - EXPIRED_BATCH:
        当前 rollout 模型版本已经过旧。
        在本设计中，这不是“旧样本还够不够再拼一批”的问题，
        而是一个更强的停止信号：只要当前 model 过旧，就立即停止，
        优先触发权重更新，让 rollout 侧尽快恢复工作。
    """

    NORMAL = auto()
    UPDATE_ABORT = auto()
    EXPIRED_BATCH = auto()


class AgentLoopManagerStatus(Enum):
    """AgentLoopManager 的全局运行状态。

    可以把它理解为 producer 主循环的状态机：
    - 初始为 NORMAL
    - NORMAL --(开始权重更新/pause)--> UPDATE_ABORT
    - UPDATE_ABORT --(continue_product 后)--> NORMAL
    - NORMAL --(整体过期，无法再 produce)--> EXPIRED_BATCH
    - EXPIRED_BATCH --(开始权重更新/pause)--> UPDATE_ABORT
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
    """按当前训练步重算已有样本的 staleness。

    设计动机：
    - 非共卡场景下，producer 和 trainer 解耦后，buffer 中的数据会“存活更久”；
    - 不能只看样本上一次写入时的 `seq_staleness`，因为那只是历史快照；
    - 所以在需要检查样本新鲜度时，要结合当前训练步重新计算一次。
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
        # 调用之间未收尾的 rollout 任务需要跨轮保存并显式 pause。
        self._pending_tasks: set[asyncio.Task] = set()

    async def pause_product(self, agent_loop, replay_buffer, task_name: str) -> float:
        """公开的 producer 暂停接口。

        设计动机：
        - 旧逻辑由 `AsyncProduceStrategy.produce_batch()` 在内部自动 cleanup；
        - 新逻辑把暂停和 pending 回收提升为外层显式调用；
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
        # 2. 立即检查当前 rollout model 是否已经过旧。
        #    只要过旧，就直接返回 EXPIRED_BATCH，
        #    不再优先尝试消费 buffer 中残留的旧 completed 样本。
        #    设计目标是：让 rollout 权重尽快更新，避免 rollout 占卡等待。
        # 3. 如果当前 model 仍然新鲜，再继续检查 buffer / 发新 rollout。
        # 4. 必要时继续异步补发 rollout 任务，直到：
        #    - 收集到足够多的 completed groups，或
        #    - 外部设置了 `update_event`，此时返回 UPDATE_ABORT。
        # 5. 每个 group 的生成耗时不再通过返回值上抛，
        #    而是写入 `rollout_state.extra_fields["group_generate_time_s"]`，
        #    后续由 manager 在 get_batch 阶段重新聚合统计。
        return ProduceBatchStatus.NORMAL


class AgentLoopManager:
    _MANAGER_STATE_PATH = "agent_loop_manager_state.json"

    def __init__(self, ...):
        # 当 trainer 准备做权重更新时置位，用来通知 producer 停止继续补任务。
        self._update_event = asyncio.Event()

        # 训练结束时置位，用来让 `produce_loop()` 正常退出。
        self._finish_event = asyncio.Event()

        # 当前 rollout 侧正在使用的模型权重，对应的是哪一次 rollout_step 更新得到的。
        self._model_rollout_step = 0

        # AgentLoopManager 的全局运行状态，见 `AgentLoopManagerStatus` 注释。
        self._status = AgentLoopManagerStatus.NORMAL

        # 最近一次 pause_product 的耗时，下一次 get_batch 时带给上层并清零。
        self._pause_time_s = 0.0

    async def _produce_batch_to_buffer(self, batch_size: int, rollout_step: int) -> ProduceBatchStatus:
        # 这是新的“只生产、不取数”的内部工具函数。
        #
        # `model_rollout_step` 不再作为这个函数的显式参数传入，
        # 而是统一从 `self._model_rollout_step` 读取。
        #
        # 单 task：
        # - 直接调用该 task 对应的 AsyncProduceStrategy.produce_batch()
        #
        # 多 task：
        # - 先按照权重分配各 task 的 batch_size
        # - 再并发 gather 各 task 的 produce_batch()
        # - 最后聚合出一个总的 ProduceBatchStatus
        return ProduceBatchStatus.NORMAL

    async def pause_product(
        self, for_weight_update: bool = False
    ) -> float:
        if for_weight_update:
            # 权重更新场景下，先发停止信号，再切 manager 全局状态。
            # 后续 producer loop 看到这个信号，会停止继续补任务。
            self._update_event.set()
            self._status = AgentLoopManagerStatus.UPDATE_ABORT

        # 单 task / 多 task 两种情况下，都统一下沉到各 task strategy 的
        # `pause_product()`，由具体 strategy 去回收 pending rollout。
        return 0.0

    async def _get_batch_from_buffer(self, batch_size: int, rollout_step: int) -> ProduceBatchResult:
        # 这是新的“只取数、不生产”的内部工具函数。
        #
        # 核心职责：
        # 1. 从 replay buffer 中按 task_name / COMPLETED 取出训练 batch；
        # 2. 从每个 group 的 extra_fields 中重建 timing 统计；
        # 3. 把上一轮 pause 产生的 `self._pause_time_s` 挂到结果里；
        # 4. 返回给 trainer 训练侧消费。
        return ProduceBatchResult(rollout_states=[])

    def continue_product(self, model_rollout_step: int) -> None:
        # continue_product 表示“权重已经同步完成，producer 可以恢复工作了”。
        # 这里不重置 replay buffer，只重置控制状态。
        self._update_event.clear()
        self._status = AgentLoopManagerStatus.NORMAL
        self._model_rollout_step = model_rollout_step

    def save(self, checkpoint_path: str, model_rollout_step_override: int | None = None) -> None:
        # 与当前 colocate `AgentLoopManager.save()` 相比，这里除了 sampler / replay buffer，
        # 还需要额外保存 manager 的控制状态。
        #
        # 推荐保存内容：
        # - 各 task sampler 状态
        # - replay_buffer 状态
        # - manager_state:
        #   - `model_rollout_step`
        #   - `status`
        #
        # 设计约束：
        # - `_update_event` / `_finish_event` 本身不序列化；
        # - `status` 保存为 `UPDATE_ABORT`，表示这个 checkpoint 是在 producer 已暂停的安全点拍的；
        # - `pause_time_s` 不必持久化，resume 后置 0 即可；
        # - 不保存 strategy._pending_tasks，save 前必须已经 pause 完成。
        #
        # `model_rollout_step_override` 的作用：
        # - 在非共卡主流程里，checkpoint 保存点位于 pause 之后、
        #   continue_product 之前；
        # - 因而 save 发生时，`self._model_rollout_step` 还是旧值；
        # - 但保存完成后，主流程马上会调用
        #   `continue_product(model_rollout_step=rollout_step)`，
        #   把 rollout 使用的模型版本
        #   推进到新的 `rollout_step`；
        # - 所以这里不是“可选优化”，而是必须显式传入
        #   `model_rollout_step_override=rollout_step`；
        # - resume 时希望恢复的是“新的 rollout_step 对应的模型版本”，
        #   而不是 save 瞬间那个还未更新的旧 `self._model_rollout_step`。
        pass

    def resume(self, checkpoint_path: str) -> int:
        # 恢复内容：
        # - 各 task sampler 状态
        # - replay_buffer
        # - manager_state
        #
        # 恢复后推荐状态：
        # - `_model_rollout_step = saved_model_rollout_step`
        # - `_status = UPDATE_ABORT`
        # - `_update_event.set()`
        # - `_finish_event.clear()`
        # - `_pause_time_s = 0.0`
        #
        # 这样做的原因是：
        # - checkpoint 恢复后的 manager 应先处于“暂停态”
        # - 等 trainer 重新把 train 权重同步到 rollout 后，
        #   再通过 `continue_product()` 恢复到 NORMAL
        #
        # 返回值：
        # - `saved_model_rollout_step`，供 trainer resume 逻辑继续使用。
        return 0

    async def produce_batch(self, batch_size: int, rollout_step: int = 0) -> ProduceBatchResult:
        # 共卡路径仍然保留这个接口，但内部改成三段式：
        # 1. `continue_product(model_rollout_step=rollout_step)`：恢复到干净状态并对齐当前权重版本
        # 2. `_produce_batch_to_buffer()`：调度 rollout，把数据写入 buffer
        # 3. `pause_product(for_weight_update=False)`：显式收尾 pending rollout
        # 4. `_get_batch_from_buffer()`：再从 buffer 中把 batch 取出来
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
                # 当前 rollout model 一旦过旧，就立刻进入 EXPIRED_BATCH，
                # 不再试图继续“榨干”旧 buffer 中的 completed 数据。
                # 这样 trainer 会尽快走权重同步，让 rollout 尽早恢复。
                self._status = AgentLoopManagerStatus.EXPIRED_BATCH
                await asyncio.sleep(0.1)
                continue

            if status == ProduceBatchStatus.UPDATE_ABORT:
                # 这里不自己 pause，避免和 trainer 的同步流程重复收尾。
                # producer 只需要停下来，等待外部 continue_product 即可。
                await asyncio.sleep(0.1)
                continue

            # 只有在正常产出数据时，producer 自己维护的 rollout_step 才递增。
            rollout_step += 1

    async def get_batch(self, batch_size: int, rollout_step: int) -> ProduceBatchResult:
        if self._status == AgentLoopManagerStatus.EXPIRED_BATCH:
            # 特殊语义：
            # 当 manager 已进入 EXPIRED_BATCH，全局表示“当前 rollout model 已经过旧”，
            # 所以训练侧不再继续消费 buffer 里的旧数据，而是直接收到一个状态信号，
            # 然后跳去做权重更新。
            return ProduceBatchResult(
                rollout_states=[],
                status=ProduceBatchStatus.EXPIRED_BATCH,
            )

        # 正常情况下，训练侧只是一个 buffer 消费者。
        return await self._get_batch_from_buffer(batch_size, rollout_step)

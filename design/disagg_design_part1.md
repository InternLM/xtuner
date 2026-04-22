# 非共卡训练设计说明

## 1. 设计目标

当前 xtuner 已经有一套共卡训练流程：

- rollout 和 train 共用同一批卡
- 每一轮训练前先做 rollout
- rollout 完成后切换到训练
- 训练结束后同步权重，再进入下一轮 rollout

这套流程实现简单，但 rollout 与 train 强同步，两个阶段会互相等待。

本次设计要补的是“非共卡训练”路径：

- rollout 和 train 使用不同的卡组
- rollout 在后台持续生成数据
- trainer 前台按需消费 replay buffer 中的数据
- 到权重同步点时，由 trainer 显式中断 producer
- 权重同步完成后，再恢复 producer

一句话概括：从“串行切换式 rollout/train”改成“后台生产 + 前台训练 + 显式同步点”的模式。

---

## 2. 总体思路

为了尽量少改现有代码，本方案不推翻已有 `AgentLoopManager + AsyncProduceStrategy + ReplayBuffer` 的结构，而是把现在耦合在一次 `produce_batch()` 调用中的逻辑拆开。

现有生产大致是：

1. 调度 rollout
2. 回收 pending
3. 从 replay buffer 取出训练 batch

新的设计会把它拆成三个可复用步骤：

- `_produce_batch_to_buffer(...)`
  只负责调度 rollout，把结果写入 replay buffer
- `pause_product(...)`
  显式停止并回收尚未收尾的 rollout 任务
- `_get_batch_from_buffer(...)`
  只负责从 replay buffer 取训练 batch，并组装统计信息

这样有两个直接好处：

- 共卡路径仍然可以继续复用 `produce_batch()`，只是内部改成三段式
- 非共卡路径可以单独复用“只生产”和“只取数”的步骤，形成后台 producer + 前台 consumer

目前设计的伪代码见 `design/disagg_draft.py`

---

## 3. 关键状态与状态机

### 3.1 `ProduceBatchStatus`

`ProduceBatchStatus` 表示“某一次 producer 调度调用”的结果，不是 manager 的全局状态。

- `NORMAL`
  - 本次调度正常结束
  - 可能已经新生成了一批样本写入 buffer
  - 也可能发现 buffer 里已经有足够样本，因此不需要继续发新 rollout

- `UPDATE_ABORT`
  - 外部准备进行权重同步
  - producer 不应再继续补发新任务
  - 剩余 pending rollout 交给外层显式 pause

- `EXPIRED_BATCH`
  - 当前 rollout 使用的模型版本已经过旧
  - 在本设计里，这会被当成一个立即停止信号
  - 不再优先尝试消费旧 completed leftovers，而是立刻进行权重更新，这样是为了尽早更新rollout权重避免其占卡空转

### 3.2 `AgentLoopManagerStatus`

`AgentLoopManagerStatus` 表示 manager 的全局运行状态，可以把它理解为后台 producer 主循环的状态机。

- `NORMAL`
  - 正常生成状态

- `UPDATE_ABORT`
  - 已收到权重更新信号
  - producer 暂停继续补任务
  - 等待 trainer 完成 pause 和权重同步

- `EXPIRED_BATCH`
  - 当前 rollout model 已经过旧
  - trainer 看到这个状态后会立刻跳过训练，直接进入权重同步

- `FINISH`
  - 整个训练结束
  - producer loop 应退出

### 3.3 状态流转

全局状态按下面的路径流转：

- 初始状态是 `NORMAL`
- `NORMAL -> UPDATE_ABORT`
  - trainer 开始做权重同步前触发
- `UPDATE_ABORT -> NORMAL`
  - 权重同步完成后调用 `continue_product()`
- `NORMAL -> EXPIRED_BATCH`
  - 当前 rollout model 已经过旧
- `EXPIRED_BATCH -> UPDATE_ABORT`
  - trainer 检测到过期后，进入权重同步阶段
- 任意状态 -> `FINISH`
  - 训练结束

这里有一个重要区分：

- `ProduceBatchStatus` 是“单次调度调用的局部结果”
- `AgentLoopManagerStatus` 是“后台 producer 的全局运行状态”

---

## 4. 关键接口改动

### 4.1 `ProduceBatchResult`

`ProduceBatchResult` 新增：

- `status: ProduceBatchStatus`

用途：

- 共卡路径下，通常返回 `NORMAL`
- 非共卡路径下，`get_batch()` 如果发现 manager 已经处于 `EXPIRED_BATCH`，可以直接返回一个空 batch，并通过 `status` 告诉 trainer “当前 rollout model 已经过旧，这轮不要训练，直接去做权重同步”

其余 timing / leftover 字段继续保留，用于训练日志与调试。

### 4.2 `AgentLoopManager`

新增成员：

- `_update_event: asyncio.Event`
  - trainer 触发权重更新时置位
  - producer 看到后尽快停止继续补任务

- `_finish_event: asyncio.Event`
  - 用于训练结束时安全退出 producer loop

- `_model_rollout_step: int`
  - rollout 当前使用的是哪一版权重
  - 注意它和 producer 自己循环的 `rollout_step` 不是同一个概念

- `_status: AgentLoopManagerStatus`
  - manager 的全局运行状态

- `_pause_time_s: float`
  - 最近一次 `pause_product()` 的耗时
  - 在下次 `ProduceBatchResult` 中上报后清零

新增接口：

- `pause_product(for_weight_update: bool = False) -> float`
- `continue_product(model_rollout_step: int) -> None`
- `produce_loop(batch_size: int, start_rollout_step: int) -> Awaitable[None]`
- `get_batch(batch_size: int, rollout_step: int) -> Awaitable[ProduceBatchResult]`

### 4.3 `AsyncProduceStrategy`

改动点：

- `pause_product()` 从私有方法改成公开接口
- `pending_tasks` 改成实例属性 `self._pending_tasks`
- `produce_batch()` 改成“只生产到 buffer”，返回 `ProduceBatchStatus`
- `produce_batch()` 不再在内部自动 pause

这里的核心意图是：把“停 producer”这个动作交回外层控制，而不是由 strategy 自己决定何时收尾。

---

## 5. 生产侧设计

### 5.1 为什么 `self._pending_tasks` 要持久化

旧设计里，`pending_tasks` 是一次 `produce_batch()` 调用内的局部变量。

这适合共卡训练，因为：

- 一次 `produce_batch()` 调完就要收尾
- 不存在跨多轮调用还继续保留 pending rollout 的需求

但非共卡场景不同：

- producer loop 会持续运行
- 多次 `AsyncProduceStrategy.produce_batch()` 调用之间，可能有 rollout 任务还没结束
- 这些 pending 任务需要跨调用保留，并在合适的时候统一 pause

因此 `pending_tasks` 必须提升为实例属性 `self._pending_tasks`。

### 5.2 `AsyncProduceStrategy.produce_batch()` 的新职责

新的 `produce_batch()` 不再返回训练 batch，而只负责“往 replay buffer 里生产数据”。

调用开始时需要做这些事：

1. 回收 `self._pending_tasks` 中已经完成的任务
2. 立即判断当前模型版本是否已经过旧
3. 如果模型仍然新鲜，再判断 buffer / 是否需要继续补发 rollout
4. 把新的结果写回 replay buffer

返回结果是一个 `ProduceBatchStatus`：

- `NORMAL`
- `UPDATE_ABORT`
- `EXPIRED_BATCH`

### 5.3 `pause_product()` 外提

旧逻辑里，strategy 会在 `produce_batch()` 末尾自动 cleanup pending rollout。

新逻辑把 pause 提升到 manager / trainer 外层，原因是：

- 非共卡下，什么时候停 producer，不应由 strategy 自己决定
- trainer 需要在权重同步前明确地说“现在停下来”
- pause 的耗时也应作为一次显式操作被记录

因此：

- `AsyncProduceStrategy.pause_product()` 负责真正停止并回收 pending rollout
- `AgentLoopManager.pause_product()` 负责在单 task / 多 task 情况下统一调度各 strategy 的 pause

### 5.4 生成耗时统计改法

原本 `ProducerTimings.generate_times_s` 是 `produce_batch()` 的返回值之一。

但现在 `AsyncProduceStrategy.produce_batch()` 已不再直接返回训练 batch，也不再适合携带这一类结果统计。

新的做法是：

- 每个 group 的 generate time 直接写到对应 `RolloutState.extra_fields["group_generate_time_s"]`
- `AgentLoopManager._get_batch_from_buffer()` 取 batch 时再重建 `group_gen_count / mean / p50 / p99`
- `pause_product()` 返回 `pause_time_s`
- `AgentLoopManager` 暂存到 `_pause_time_s`，下一次 `ProduceBatchResult` 带出后清零

这样统计信息仍然保留，但不再强依赖 strategy 的同步返回值。

---

## 6. Staleness 设计

### 6.1 为什么要区分 `rollout_step` 和 `model_rollout_step`

非共卡下，producer 是后台持续运行的，它自己的循环步数会不断前进。

但样本真正重要的信息不是“producer 在第几轮循环里生成了它”，而是：

- 这段 token 是用哪一版权重生成的

因此：

- `response_rollout_steps` 不应再记录“当前 producer 调度步数”
- 而应记录“当前 rollout 使用的模型版本步数”，即 `_model_rollout_step`

所以 `SingleTurnAgentLoop` / `PartialRolloutHandler` 需要改成接收 `model_rollout_step`。

### 6.2 为什么仍然保留样本 staleness 重算能力

非共卡场景下，buffer 中的样本可能停留更久。

如果只看样本写入时的 `seq_staleness`，有问题：

- 它只是历史快照
- 到了真正训练时，这个样本可能已经又老了很多

因此设计里仍建议保留一个轻量 helper，例如：

- `refresh_seq_staleness(group, current_rollout_step)`

它的职责是：

- 在需要检查旧样本新鲜度时
- 根据 `response_rollout_steps` 和当前训练步
- 重新计算 staleness

### 6.3 `ExpiredBatch` 的判定

`ExpiredBatch` 不是简单地说“当前没有数据”，也不是“先尽量消化旧 completed leftovers 再决定要不要停”。

它的真实含义是：

- 当前 rollout 使用的模型版本已经过旧
- 为了让 rollout 侧尽快切到新权重，不再继续占卡等待
- producer 应立即停止并要求外层尽快做权重同步

因此这里采用的是“更激进的停机策略”：

- 只要当前 rollout model 过旧，就直接返回 `EXPIRED_BATCH`
- 不再优先尝试复用 buffer 里的旧 completed 数据
- trainer 收到信号后，直接跳过训练并推进权重更新

这样做的原因是：

- 非共卡场景下，rollout 卡组本来就和 train 卡组解耦
- 如果 rollout 明知已经过旧，还继续等待 trainer 去消化旧数据，
  rollout 侧会白白占卡、拖慢权重切换
- 所以这里优先保证 rollout 尽快更新，而不是优先榨取旧样本

---

## 7. AgentLoopManager 设计

### 7.1 `_produce_batch_to_buffer(...)`

这是新的内部工具函数，只负责生产，不负责取数。

这里有一个实现约束：

- `model_rollout_step` 不再作为 `_produce_batch_to_buffer(...)` 的显式参数传入
- 而是统一从 `self._model_rollout_step` 读取
- 共卡路径在 `produce_batch()` 入口通过
  `continue_product(model_rollout_step=rollout_step)` 对齐这个状态
- 非共卡路径则在外部权重同步完成后通过 `continue_product(...)` 更新它

单 task：

- 直接调用该 task 对应的 `AsyncProduceStrategy.produce_batch()`

多 task：

- 继续沿用 `get_task_batch_sizes()` 进行 batch 分配
- 用 `asyncio.gather()` 并发调度各 task 的 produce
- 对返回的 `ProduceBatchStatus` 做聚合

状态聚合优先级：

- `UPDATE_ABORT` > `EXPIRED_BATCH` > `NORMAL`

原因是：

- 如果有任何 task 收到了权重更新信号，整个 producer 就应优先停下来
- 其次才是某些 task 因当前 rollout model 过旧而直接停机

### 7.2 `_get_batch_from_buffer(...)`

这是新的内部工具函数，只负责取数。

职责：

1. 从 replay buffer 中按 `COMPLETED` 取训练 batch
2. 统计 leftover 的 `COMPLETED / ABORTED / EXPIRED`
3. 从 sample 的 `extra_fields` 中重建 generate timing
4. 把最近一次 pause 的 `pause_time_s` 带给结果

### 7.3 `pause_product(...)`

当 `for_weight_update=True` 时：

1. 先置 `_update_event`
2. 再把 manager 状态切到 `UPDATE_ABORT`
3. 最后调用各 task strategy 的 `pause_product()`

顺序很重要。

先置事件的原因是：

- 防止 producer 在 pause 开始前又继续补发新任务

### 7.4 `continue_product(...)`

`continue_product(model_rollout_step=...)` 的作用是恢复 producer 控制状态：

- 清 `_update_event`
- `_status = NORMAL`
- `_model_rollout_step = 当前训练步`

这样 producer 才知道：“现在 rollout 侧已经切换到新权重，可以继续生成了”。

---

## 8. 共卡路径怎么适配

虽然这次目标是非共卡，但共卡路径也需要适配接口拆分，避免维护两套不同的逻辑。

新的共卡 `AgentLoopManager.produce_batch()` 内部改成：

1. `continue_generation()`
2. `continue_product(model_rollout_step=rollout_step)`
3. `_produce_batch_to_buffer(...)`
4. `pause_product(for_weight_update=False)`
5. `_get_batch_from_buffer(...)`
6. `pause_generation()`

这样做的意义是：

- 共卡与非共卡都复用同一套底层逻辑
- 差别只在于：
  - 共卡路径是“一次调用里生产+收尾+取数”
  - 非共卡路径是“后台持续生产，前台单独取数”

---

## 9. 非共卡 producer loop

`produce_loop(batch_size, start_rollout_step)` 是非共卡新增的后台生产循环。

主逻辑：

1. 持续调用 `_produce_batch_to_buffer(batch_size, rollout_step)`
2. 根据返回状态决定下一步动作

当返回：

- `NORMAL`
  - 表示本轮生产逻辑正常结束
  - producer 自己维护的本地 `rollout_step += 1`

- `EXPIRED_BATCH`
  - manager 进入 `EXPIRED_BATCH`
  - producer 立即暂停继续工作
  - 不再继续尝试消化旧 completed buffer
  - 等待 trainer 进行权重同步并 `continue_product()`

- `UPDATE_ABORT`
  - 表示 trainer 正在准备做权重同步
  - producer 不再自己 pause，避免与 trainer 竞争
  - 只等待外部 `continue_product()`

- `FINISH`
  - 退出 producer loop

这里有一个重要约束：

- producer loop 收到 `UPDATE_ABORT` 后不做二次 pause
- 只有 trainer 的权重同步路径显式调用 `pause_product()`

这样可以避免重复回收 pending rollout 的竞态。

---

## 10. 非共卡 `get_batch(...)`

`get_batch(batch_size, rollout_step)` 是训练侧的消费接口。

正常情况：

- 它只调用 `_get_batch_from_buffer(...)`
- 本身不驱动新的 rollout 生成

特殊情况：

- 如果当前 `AgentLoopManager._status == EXPIRED_BATCH`
- 则直接返回 `ProduceBatchResult(status=EXPIRED_BATCH, rollout_states=[])`

这样 trainer 就能收到一个很明确的信号：

- 这轮不要再训练
- 直接进入权重同步

---

## 11. 非共卡 Trainer 设计

### 11.1 新 trainer 入口

新增：

- `xtuner/v1/train/rl_disaggregated_trainer.py`

它的配置形状保持和现有 `rl_disagg_single.py` / `rl_disagg_multi.py` 一致。

### 11.2 为什么 `fit()` 仍保持同步

当前 CLI 仍然使用：

- `trainer = trainer_cfg.build()`
- `trainer.fit()`

为了不改 CLI，设计上保留同步 `fit()`，内部再用 `asyncio_run(self._fit())` 包一层。

这样对外接口不变，但内部可以自然地管理：

- 后台 producer task
- 前台 async 取数
- eval 与 producer 的优先级关系

### 11.3 `_fit()` 主流程

训练主流程分成两条并行逻辑：

- 后台：`producer_task = create_task(agent_loop_manager.produce_loop(...))`
- 前台：训练循环不断 `get_batch()`

前台每一轮大致是：

1. `produce_result = await agent_loop_manager.get_batch(...)`
2. 如果 `status != EXPIRED_BATCH`
   - `_prepare_train_data(...)`
   - `train_controller.fit(...)`
3. 如果到达同步点
   - `agent_loop_manager.pause_product(for_weight_update=True)`
   - `_sync_weights_and_save(...)`
4. 如果这一轮需要 eval
   - 先做 eval
5. `agent_loop_manager.continue_product(model_rollout_step=current_step)`

### 11.4 为什么 `EXPIRED_BATCH` 时直接跳过训练

`EXPIRED_BATCH` 的语义不是“这一轮没有数据”，而是：

- 当前 rollout 权重版本已经过旧
- 应优先让 rollout 侧尽快更新权重，而不是继续等待 trainer 消化旧样本

这时如果还继续从 buffer 中拿旧数据训练，会带来两个问题：

- rollout 侧还要继续等待，不能尽快切到新权重
- stale 数据仍可能被继续消费

因此策略是：

- 直接跳过 `_prepare_train_data`
- 跳过 `train_controller.fit`
- 直接进入权重同步

---

## 12. 权重同步与评测优先级

### 12.1 `_sync_weights_and_save(...)`

非共卡下，权重同步前的顺序必须是：

1. `agent_loop_manager.pause_product(for_weight_update=True)`
2. `_sync_weights_and_save(rollout_step)`

而 `_sync_weights_and_save(rollout_step)` 内部再做：

1. `_maybe_save_checkpoint(rollout_step)`
2. `bind_train_rollout(...)`
3. `fake_update_weights()`

这里暂时不走真实的 `train_controller.update_weights()`，而是保留一个显式占位函数：

- `fake_update_weights()`

这样后续接入真实跨卡同步实现时，不需要改 trainer 主流程。

### 12.2 为什么 eval 要优先于 producer continue_product

如果同步权重后立刻 `continue_product()`，producer 会马上恢复后台生成。

但如果这一步还要做 eval，就会出现：

- eval 和 background producer 同时竞争 rollout 资源

因此本设计固定采用：

- 先同步权重
- 若本轮需要 eval，则先跑 eval
- eval 完成后再 `continue_product()`

即：eval 的优先级高于 background producer。

---

## 13. Checkpoint 保存与恢复

### 13.1 为什么 checkpoint 需要专门细化

共卡训练里，rollout 和 train 是串行切换的，checkpoint 保存点天然比较清晰。

但非共卡训练里，多了一个持续运行的后台 producer：

- replay buffer 可能正在被 producer 写入
- strategy 内可能还挂着未收尾的 pending rollout
- manager 还维护了 `_model_rollout_step` / `_status` / `_update_event` 这些运行时状态

如果不把 save / resume 逻辑说清楚，恢复后很容易出现下面的问题：

- producer 恢复得太早，在 rollout 权重同步前就继续生成
- replay buffer 恢复了，但 manager 的 `model_rollout_step` 不对
- checkpoint 拍下来时还有 pending rollout 没收尾，导致恢复语义不一致

所以这里要求 checkpoint 必须在“静止态”拍摄。

### 13.2 安全保存点

checkpoint 的安全保存点固定放在 `_sync_weights_and_save(...)` 中，且必须满足：

1. `agent_loop_manager.pause_product(for_weight_update=True)` 已完成
2. producer 不会继续补发新任务
3. replay buffer 不再被后台并发写入
4. `continue_product()` 尚未发生

因此保存点的语义是：

- trainer 当前步的训练结果已经稳定
- producer 已暂停
- rollout 仍需在 resume 后由 train 侧重新同步权重

这里特意把 checkpoint 放在 `continue_product()` 之前，是为了避免 producer 恢复后台生成后，又把系统带回“动态变化态”。

这里还要再强调一个容易误解的点：

- `_maybe_save_checkpoint(rollout_step)` 中，必须显式传入
  `model_rollout_step_override=rollout_step`
- 不能偷懒直接把当时 manager 内部的 `self._model_rollout_step` 原样存盘

原因是：

- save 的时机在 `pause_product(for_weight_update=True)` 之后
- 但在 `continue_product(model_rollout_step=rollout_step)` 之前
- 所以 save 那一刻，manager 里的 `self._model_rollout_step` 仍然还是旧的 rollout 权重版本
- 而主流程的真实意图，是保存“本轮同步完成后，resume 应该继续使用的新 rollout_step”

换句话说，这个 override 不是建议项，而是恢复语义正确性的必要条件。

### 13.3 保存内容

沿用当前 colocate trainer 的三层保存结构：

#### 1. `AgentLoopManager.save(...)`

除现有的 sampler / replay buffer 外，非共卡路径还要保存 manager 自身状态。

建议保存：

- 各 task sampler 状态
- replay buffer
- `agent_loop_manager_state.json`

`agent_loop_manager_state.json` 至少包含：

- `model_rollout_step`
- `status`

其中 `model_rollout_step` 的来源要特别注意：

- 在 `_maybe_save_checkpoint(rollout_step)` 里，必须通过
  `model_rollout_step_override=rollout_step` 显式写入
- 不应直接落盘 save 瞬间那个尚未经过 `continue_product()` 推进的旧
  `self._model_rollout_step`

推荐语义：

- `status` 保存为 `UPDATE_ABORT`
- 表示这个 checkpoint 拍摄时，producer 已经被暂停

不建议保存：

- `_update_event`
- `_finish_event`
- `_pause_time_s`
- `AsyncProduceStrategy._pending_tasks`

原因：

- event 是运行时同步原语，不适合直接持久化
- `pause_time_s` 是一次性日志字段，resume 后清零即可
- `pending_tasks` 是内存里的协程对象，checkpoint 前必须已经 pause 并清空

#### 2. `train_controller.save(...)`

和 colocate trainer 语义一致，保存训练态：

- model
- optimizer
- 其他 DCP 状态

#### 3. `trainer_state.json`

建议至少保存：

- `cur_step`

建议额外保存：

- `global_train_step`
- `model_rollout_step`

其中：

- `cur_step` 决定训练主循环恢复到哪一步
- `model_rollout_step` 主要用于恢复校验和排障

### 13.4 `model_rollout_step` 为什么要单独保存

`model_rollout_step` 不能只依赖 `cur_step` 间接推导。

原因是：

- checkpoint 是在 `pause_product()` 后、`continue_product()` 前拍摄的
- 这个时间点的 manager 状态，不一定能直接由训练步数唯一还原
- 后续如果 sync 策略变化，`cur_step` 与 rollout 实际使用的权重版本也不一定严格一一对应

因此这里更准确的做法是：

- 在 manager state 中显式保存“resume 目标版本”的 `model_rollout_step`
- 这个值在 `_maybe_save_checkpoint(rollout_step)` 时通过
  `model_rollout_step_override=rollout_step` 传入
- resume 后直接按这个保存值恢复

这里之所以不能直接依赖 `self._model_rollout_step`，不是因为它永远不可信，
而是因为当前设计的 checkpoint 保存点刚好卡在：

1. `pause_product(...)` 已完成
2. `continue_product(model_rollout_step=rollout_step)` 尚未执行

所以 save 瞬间的 `self._model_rollout_step` 在语义上仍代表“旧 rollout 模型版本”，
而 resume 后我们真正希望恢复的是“新的 rollout_step 对应版本”。

### 13.5 `AgentLoopManager.save(...)` 的约束

为了保证 checkpoint 可恢复，保存前应满足：

- 所有 `AsyncProduceStrategy._pending_tasks` 为空
- `AgentLoopManager._status == UPDATE_ABORT`
- `_update_event` 已经置位
- producer 当前不会继续写 replay buffer
- 调用方显式以 `model_rollout_step_override=rollout_step` 传入
  本次 checkpoint 对应的目标 rollout 版本

如果这些条件不满足，建议：

- 拒绝保存，或
- 在 save 前先强制执行 pause

### 13.6 Resume 顺序

resume 的 source of truth 不是 rollout 的运行时内存，而是：

- train checkpoint
- replay buffer
- sampler 状态
- manager state

推荐恢复顺序如下：

1. `train_controller.resume(checkpoint_path)`
2. `agent_loop_manager.resume(checkpoint_path)`
3. `bind_train_rollout(...)`
4. `fake_update_weights()` 或后续真实权重同步
5. `agent_loop_manager.continue_product(model_rollout_step=saved_model_rollout_step)`
6. `fit()` 启动新的 `producer_task = create_task(produce_loop(...))`

这里有两个重要点：

- 不恢复旧的 producer task
  - producer task 是运行时协程，进程重启后必须重新创建
- rollout 权重不作为 checkpoint source of truth
  - resume 后总是从 train 侧重新同步一次 rollout
- `saved_model_rollout_step` 应该对应新的 `rollout_step`
  - 也就是 `_maybe_save_checkpoint(rollout_step)` 时显式 override 写进去的值
  - 不能退回到 save 瞬间那个旧的 `self._model_rollout_step`

### 13.7 `AgentLoopManager.resume(...)` 的目标状态

`AgentLoopManager.resume(...)` 恢复 sampler / replay buffer 后，推荐把 manager 留在一个“暂停态”：

- `_model_rollout_step = saved_model_rollout_step`
- `_status = UPDATE_ABORT`
- `_update_event.set()`
- `_finish_event.clear()`
- `_pause_time_s = 0.0`

这样做的原因是：

- resume 完成时 producer 还不应立即继续生成
- 必须等 trainer 先重新把 train 权重同步到 rollout
- 然后再由 `continue_product()` 切回 `NORMAL`

### 13.8 eval manager 是否保存

默认不保存 `eval_agent_loop_manager`，保持与当前 colocate trainer 一致的语义。

原因：

- eval 数据流不影响训练正确性
- eval sampler 从头开始通常可接受
- 这样能减少 checkpoint 内容和恢复复杂度

如果后续需要“精确恢复 eval 进度”，再单独扩展即可。

---

## 14. 测试建议

建议至少覆盖以下场景。

### 14.1 Producer / Strategy

- `AsyncProduceStrategy.produce_batch()` 能正确返回：
  - `NORMAL`
  - `UPDATE_ABORT`
  - `EXPIRED_BATCH`
- `self._pending_tasks` 能跨调用保留
- `pause_product()` 外提后，pending 回收逻辑仍正确

### 14.2 Manager

- 单 task / 多 task 的 `_produce_batch_to_buffer()` 行为一致
- 多 task 下 `task_batch_sizes` 仍正确分配
- `get_batch()` 在 `EXPIRED_BATCH` 状态下直接返回空 batch + 状态
- `pause_product(for_weight_update=True)` 先置 `_update_event`
- `save()` 前若仍有 pending tasks，会拒绝保存或先 pause
- `resume()` 后 manager 先处于 `UPDATE_ABORT`，而不是直接 `NORMAL`

### 14.3 Trainer

- `EXPIRED_BATCH` 会跳过训练，直接进入同步
- eval 步上 continue_product 发生在 eval 之后
- `FINISH` 时 producer task 能正确退出
- checkpoint 保存点发生在 pause 之后、continue_product 之前
- resume 后会先做一次 rollout 权重同步，再启动新的 producer_task

### 14.4 端到端测试
- 对于配置示例 examples/v1/config/rl_disagg_multi.py 和 examples/v1/config/rl_disagg_single.py 跑通基本训练流程不报错
- 运行脚本参考 zdev/rl_design_disagg.sh

---

## 15. 当前明确的设计取舍

- `ExpiredBatch` 采用更激进的策略：
  - 只要当前 rollout model 过旧，就直接停
  - 不再优先尝试消费旧 completed leftovers
  - 优先让 rollout 侧尽快完成权重更新

- `produce_loop` 的 batch size 采用显式传参：
  - `produce_loop(batch_size, start_rollout_step)`

- pause 只由 trainer 的权重同步路径显式触发一次
  - producer 收到 `UPDATE_ABORT` 后不再自行二次 pause

- rollout 当前只支持 `abort_all`
  - 不做按 request 的精细化取消
- 活跃 pending 统一在 pause 阶段处理

- `fit()` 对外保持同步
  - 内部通过 async `_fit()` 实现非共卡调度

- checkpoint 只在 producer 已暂停的静止态拍摄
  - 不保存运行中的 pending rollout task
  - resume 后始终重新创建 producer_task
  - rollout 权重始终从 train 侧重新同步

---

## 16. 对应实现锚点

本设计主要落在这些模块：

- `xtuner/v1/rl/agent_loop/producer.py`
  - `AsyncProduceStrategy`

- `xtuner/v1/rl/agent_loop/agent_loop_manager.py`
  - `ProduceBatchResult`
  - `AgentLoopManager`

- `xtuner/v1/rl/agent_loop/single_turn_agent_loop.py`
  - `generate_sample(...)`

- `xtuner/v1/rl/agent_loop/utils.py`
  - `PartialRolloutHandler.postprocess(...)`

- `xtuner/v1/train/rl_disaggregated_trainer.py`
  - 新增的非共卡 trainer

如果需要进一步实现，可以优先按照这个顺序推进：

1. 先把 `AgentLoopManager` 和 `AsyncProduceStrategy` 的接口拆开
2. 再补 `RLDisaggregatedTrainer`
3. 最后补测试和配置示例的细化

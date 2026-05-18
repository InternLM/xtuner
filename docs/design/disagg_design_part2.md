# `produce_loop` / `AsyncProduceStrategy` 重设计

## 目标

在尽量少改现有结构的前提下，解决四件事：

1. 保留 `AgentLoopManager.produce_loop` 本地的 `future_step`，继续按 future step 逐个预取 batch。
2. `AsyncProduceStrategy.produce_batch` 的动态控制从“当前 buffer 中有多少 completed”改成“消费者已消费 + buffer fresh + pending”的累计口径，避免消费者取走 batch1 后，生产 batch2 时误补 batch1。
3. staleness / expired 状态只在两个地方写：
   - strategy 在 `replay_buffer.put` 前，根据 `progress.next_consumer_step` 刷新。
   - manager 在 `get_batch` 入口按当前 `rollout_step` 刷新 buffer 中已有 completed，并在成功取出 batch 后推进 `progress.next_consumer_step = rollout_step + 1` 再刷新 leftover completed。
4. `_pending_tasks` 不再用整体赋值覆盖，改成 snapshot + claim 的增量认领，避免 producer 和 pause 并发 drain 同一 task。

## 主要考虑点

### 1. `consumed_samples` 和 `consumer_step` 不能只传值

Opus 方案里 `produce_batch(..., consumed_samples, consumer_step)` 是进入 strategy 时的一次性快照。

这仍然有竞态：

- producer 进入 `produce_batch` 时，`consumed_samples = 0, fresh = batch1`。
- producer 正在等待 batch2 rollout 完成。
- consumer 并发取走 batch1，此时 manager 中 `consumed_samples = batch1, fresh = 0`。
- 如果 strategy 仍使用旧的 `consumed_samples = 0`，它会误以为 batch1 缺失，继续多补一批。

因此 strategy 内每次计算动态控制、以及每次 put 前刷新 staleness 时，都必须读取 live 值。

推荐接口不是传一组 getter，而是传一个可变 progress 引用：

```python
progress: ProduceProgress
```

只要 Manager 原地更新这个对象，strategy 每次读取 `progress.next_consumer_step` / `progress.consumed_samples[task_name]` 时拿到的就是最新值。

`next_consumer_step` 不是“已经完成训练的最新 step”，而是 producer 在 put 新样本时应该面向的消费 step：

- `get_batch(i)` 开始时，训练侧正在等待 step `i` 的 batch，因此设置 `next_consumer_step = i`。
- `get_batch(i)` 成功取出非空 batch 后，训练侧即将消费 step `i`，producer 后续应面向 step `i + 1`，因此返回前设置 `next_consumer_step = i + 1`。
- `EXPIRED_BATCH` 或 finish 空返回没有成功消费 batch，不推进到 `i + 1`。

### 2. over-sample 不应放大全部历史累计目标

Opus 方案使用：

```python
desired_window = ceil((1 + over_sample) * target_abs)
```

这会把已经消费过的历史目标也一起放大。假设 batch size = `B`，当前在预取 batch10，前 9 个 batch 都已消费，`over_sample=0.5`：

- 必要目标只缺 batch10 的 `B`。
- 但上式要求窗口达到 `15B`，等价于对前 9 个已经消费掉的 batch 也重新保留超发窗口。

修正为“按当前 task batch size 给本轮 target 增加固定超发预算”：

```python
available = consumed_abs + fresh
target_abs = progress.target_samples[task_name]
oversample_budget = ceil(over_sample * task_batch_size)
scheduled_target = target_abs + oversample_budget
```

返回条件仍然是：

```python
available >= target_abs
```

`pending` 只用于决定还要不要继续发任务：

```python
scheduled_effective = available + pending_count
if scheduled_effective < scheduled_target:
    schedule_more()
```

这样 over-sample 只给当前 task batch 一个固定 ahead window，不会让历史累计目标反复膨胀。tail-batch mode 下 `oversample_budget = 0`，本轮新增任务固定从 `Status.EXPIRED` pool 取样，不主动停止已有 pending，也不强制清空 expired pool。

### 3. `_pending_tasks` 不能只靠循环顶部检查 `_update_event`

`pause_produce(use_global_progress=True)` 会先 set `_update_event`，但随后会立刻进入各 task strategy 的 `pause_produce`。此时后台 producer 可能还停在 `asyncio.wait(self._pending_tasks, ...)` 中。

所以不能只假设 producer 会先返回，也不能只在 `produce_batch()` 循环顶部检查一次 event。需要在 strategy 内保证：

- 同一个 done task 只能被一方认领。
- `_pending_tasks` 只能增量 add / discard，不能 `self._pending_tasks = set(pending)` 整体覆盖。
- `_schedule_one()` 在 pending lock 内检查 `update_event.is_set()`；如果 pause 发生在调度中途，本次已创建的 task 必须先加入 pending，再由 pause drain 收尾。
- `_schedule_tasks_until()` 返回后还要再次检查 `update_event`，避免 pending 已被 pause drain 清空后误返回 `NORMAL`。

## 核心状态

### Manager 侧

新增一个 manager 持有的可变进度对象：

```python
@dataclass
class ProduceProgress:
    next_consumer_step: int
    producer_future_step: int
    consumed_samples: dict[str, int]
    target_samples: dict[str, int]
    target_upto_future_step: int

self._produce_progress: ProduceProgress
```

`ProduceProgress` 可以放在 `producer.py` 或一个小的 shared module 中；`agent_loop_manager.py` 已经依赖 `producer.py`，因此由 manager 构造并传给 strategy 不会引入新的反向依赖。

含义：

- `progress.next_consumer_step`：producer 当前应按哪个训练 step 计算新样本的 staleness。fresh disagg 训练初始化为 `1`；`get_batch(i)` 开始时设为 `i`，成功取出 batch 后设为 `i + 1`。
- `progress.consumed_samples[task]`：consumer 已经从 buffer 取走并用于训练的 group 数，按 task 绝对累计。
- `progress.producer_future_step`：producer 当前正在预取的 future step。它属于 manager，不属于 strategy。
- `progress.target_samples[task]`：截至 `progress.target_upto_future_step`，该 task 应该累计生产出的目标 group 数。
- `progress.target_upto_future_step`：`target_samples` 已经覆盖到的最大 future step；初始化为 `0`。

动态控制使用绝对累计口径：

```python
available(task) = progress.consumed_samples[task] + fresh_completed(task)
required(task) = max(0, progress.target_samples[task] - available(task))
```

只要 target 和 consumed 都是绝对累计量，就不需要维护 Progress Window，也不需要在 sync 后重置窗口。

关键约束：

- `self._produce_progress` 的对象引用应保持稳定，初始化后不要在运行中整体替换。
- resume/load 时也优先原地更新字段，而不是 `self._produce_progress = ProduceProgress(...)` 后让 strategy 持有旧引用。
- `consumed_samples` / `target_samples` 也按 key 原地更新；如果必须整体替换 dict，要保证 strategy 没有缓存旧 dict。
- `progress` 的写入方应收敛在 Manager / 调用方初始化与消费入口：
  - Manager 构造、resume、`_ensure_target_upto()`、`get_batch()` 消费计数负责维护全局 `progress`。
  - Strategy 不在传入的 `progress` 上补 key，也不通过 `setdefault()` 修复缺失状态。
  - 传入 `progress` 时，`consumed_samples[task_name]` 和 `target_samples[task_name]` 必须已经存在；缺失应 fail fast。
- `progress` 的读取方必须显式读取：
  - Strategy 内使用 `progress.consumed_samples[task_name]` / `progress.target_samples[task_name]`。
  - 不使用 `dict.get(task_name, 0)` 这类兜底，避免把初始化或 checkpoint 漂移问题隐藏成“目标为 0 / 已消费为 0”。
  - 除了本轮 `target_abs` / `scheduled_target` 这种语义上需要冻结的调度目标，不把 `progress` 字段先复制到局部标量或局部 dict 再使用，例如不要写 `current_rollout_step = progress.next_consumer_step` 或 `target_by_task = dict(progress.target_samples)`；需要字段值时直接读 `progress.xxx`，让并发更新能尽早生效。
  - `progress = self._produce_progress` 这类对象引用别名可以保留；它不复制字段值。
- 所有 strategy 调用都必须显式传入已经初始化好的 `progress`，不再支持 `progress=None` 的本地兜底。

### Colocate 路径的 progress 约束

`AsyncProduceStrategy` 内部只保留一套语义：`available = consumed + fresh_completed`，并和 `target_samples[task_name]` 比较。区别只在 progress 的来源：

- 非共卡 `produce_loop()` 使用 Manager 的全局 `_produce_progress`，target/consumed 都是跨 step 的绝对累计值。
- 共卡 `AgentLoopManager.produce_batch()` 不复用非共卡全局进度窗口；它为本次同步调用构造一个局部 `ProduceProgress`，`next_consumer_step` 等于本次 `rollout_step`，含义是“本次同步调用生产出的 batch 要服务的训练 step”。
- 共卡取走 batch 后，如需要记录 consumed，也应写入这次调用的局部 `ProduceProgress`，不要污染非共卡全局 `_produce_progress`。
- 直接调用 `AsyncProduceStrategy.produce_batch(...)` 也必须传入 `progress`；测试或临时调用如需同步语义，应由调用方构造一次性 local progress。

### Strategy 侧

`AsyncProduceStrategy` 保留 task 私有的 pending 集合：

```python
self._pending_tasks: set[asyncio.Task]
self._pending_lock: asyncio.Lock
```

`pending_count` 不建议再单独维护成第二份可变状态，直接使用：

```python
pending_count = len(self._pending_tasks)
```

如果实现上为了日志或性能保留 `_pending_count`，也必须只在同一个 helper 中和 `_pending_tasks` 同步更新，不能分散手写。

## Cumulative Target

Manager 为当前 `progress.producer_future_step` 维护每个 task 的绝对累计目标。

推荐不要每次从 step 1 重新求和，而是维护一个单调前进的 target 计数器，并 checkpoint：

```python
def _ensure_target_upto(self, batch_size: int, current_future_step: int) -> None:
    progress = self._produce_progress
    if current_future_step <= progress.target_upto_future_step:
        return

    for fs in range(progress.target_upto_future_step + 1, current_future_step + 1):
        if len(self.task_runners) == 1:
            progress.target_samples[self.task_runners[0].task_name] += batch_size
        else:
            sizes = self.get_task_batch_sizes(batch_size, fs)
            self._validate_task_batch_sizes(sizes, batch_size)
            for task_name, n in sizes.items():
                progress.target_samples[task_name] += n

    progress.target_upto_future_step = current_future_step
```

Manager 把该 task 从 step 1 到当前 future step 的绝对累计目标维护在 `progress.target_samples[task_name]` 中；strategy 直接从 `progress` 读取，不通过第二份 target 快照驱动生产。

`progress.target_samples` 需要 checkpoint。这样即使后续自定义的 `get_task_batch_sizes` 不是纯函数，也不会在 resume 后因为重算历史分配而漂移。

strategy 内部实时计算：

```python
fresh = await replay_buffer.count(task_name, Status.COMPLETED)
available = progress.consumed_samples[task_name] + fresh
required = max(0, progress.target_samples[task_name] - available)
```

这里 `progress` 是 Manager 传入的可变引用；strategy 不缓存 `consumed` 或 `next_consumer_step`，每次循环现场读取。`target_abs` / `scheduled_target` 是本轮 produce_batch 的静态调度决策，进入循环前冻结。

## AsyncProduceStrategy 动态控制

### 新接口

`AsyncProduceStrategy.produce_batch` 的进度入口改为必传 `progress`：

```python
async def produce_batch(
    self,
    agent_loop,
    sampler,
    replay_buffer,
    batch_size: int,
    task_name: str,
    rollout_step: int = 0,              # disagg 下传 current_future_step
    update_event: asyncio.Event | None = None,
    *,
    model_rollout_step: int,
    progress: ProduceProgress,
) -> ProduceBatchStatus:
```

入口只做 fail fast 校验，不做缺省初始化：

```python
# fail fast：调用方必须完整初始化 progress。
if task_name not in progress.consumed_samples:
    raise KeyError(...)
if task_name not in progress.target_samples:
    raise KeyError(...)
```

### 主循环

伪代码：

```python
async def produce_batch(...):
    current_future_step = rollout_step
    if update_event is None:
        update_event = asyncio.Event()
    _validate_progress_for_task(progress, task_name)
    if progress.target_samples[task_name] <= 0:
        return ProduceBatchStatus.NORMAL

    if update_event.is_set():
        return ProduceBatchStatus.UPDATE_ABORT
    if self.is_model_expired(current_future_step, model_rollout_step):
        return ProduceBatchStatus.EXPIRED_BATCH

    # 只在进入本轮时回收一次跨调用遗留的 done task，避免 done task 长期留在 pending 集合。
    claimed_done = await self._claim_already_done()
    await self._put_claimed_tasks(claimed_done, replay_buffer, task_name, progress)

    if update_event.is_set():
        return ProduceBatchStatus.UPDATE_ABORT
    if self.is_model_expired(current_future_step, model_rollout_step):
        return ProduceBatchStatus.EXPIRED_BATCH

    expired_count = await replay_buffer.count(task_name=task_name, group_status=Status.EXPIRED)
    sample_from_expired = (
        self.tail_batch_trigger_size > 0
        and expired_count >= self.tail_batch_trigger_size
    )
    target_abs = progress.target_samples[task_name]
    oversample_budget = 0 if sample_from_expired else math.ceil(self.over_sample_threshold * batch_size)
    scheduled_target = target_abs + oversample_budget

    while True:
        if update_event.is_set():
            return ProduceBatchStatus.UPDATE_ABORT
        if self.is_model_expired(current_future_step, model_rollout_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        fresh = await replay_buffer.count(task_name=task_name, group_status=Status.COMPLETED)
        available = progress.consumed_samples[task_name] + fresh
        if available >= target_abs:
            return ProduceBatchStatus.NORMAL

        pending_count = await self._pending_count()
        desired_pending = max(0, scheduled_target - available)
        if available + pending_count < scheduled_target:
            await self._schedule_tasks_until(
                agent_loop=agent_loop,
                sampler=sampler,
                task_name=task_name,
                desired_pending=desired_pending,
                sample_from_expired=sample_from_expired,
                model_rollout_step=model_rollout_step,
                update_event=update_event,
            )
            if update_event.is_set():
                return ProduceBatchStatus.UPDATE_ABORT

        pending_snapshot = await self._snapshot_pending()
        if update_event.is_set():
            return ProduceBatchStatus.UPDATE_ABORT
        if not pending_snapshot:
            # sampler 无数据或当前没有 pending；交回 manager，避免忙等。
            return ProduceBatchStatus.NORMAL

        done, _ = await asyncio.wait(
            pending_snapshot,
            timeout=1,
            return_when=asyncio.FIRST_COMPLETED,
        )
        claimed_done = await self._claim_done(done)
        await self._put_claimed_tasks(claimed_done, replay_buffer, task_name, progress)
```

注意：

- `available >= target_abs` 才表示当前 future step 的必要目标达成。
- `pending` 不参与返回条件，只参与“是否要继续调度”的判断。
- `sample_from_expired` 和 `scheduled_target` 是本轮静态决策，放在循环前；循环中只更新 live `available` / `pending_count`。
- tail-batch mode 下 `scheduled_target == target_abs`，本轮新增任务只从 `Status.EXPIRED` pool 取样，不主动停止已有 pending。
- 失败、filtered、aborted 的 group 会被 put，但不会增加 `fresh`，下一轮自然会补发。
- 如果 consumer 在本循环中间取走了 batch，下一次读取 `progress.consumed_samples[task_name]` 会看到新值，不会误补已消费的部分。

## staleness / expired 写入策略

只保留两个写入点。

### 写入点 1：strategy put 前

新增一个集中 helper，替代 scattered 的 `update_expired_status` 调用：

```python
async def _put_generated_group(
    self,
    items: list[RolloutState],
    replay_buffer: ReplayBuffer,
    task_name: str,
    current_rollout_step: int,
) -> None:
    refresh_seq_staleness(items, current_rollout_step)
    expire_group_if_needed(items, self.tail_batch_stale_threshold)
    await replay_buffer.put(items, task_name)
```

`expire_group_if_needed` 的语义应覆盖 completed 和 aborted：

```python
def expire_group_if_needed(group: list[RolloutState], threshold: int) -> list[RolloutState]:
    if threshold <= 0:
        return group
    group_status = update_group_status(group)
    if group_status not in (Status.COMPLETED, Status.ABORTED):
        return group
    if any(sample.seq_staleness >= threshold for sample in group):
        for sample in group:
            sample.status = Status.EXPIRED
    return group
```

为什么不直接复用当前 `update_expired_status`：

- 当前实现只检查 `sample.status == Status.ABORTED` 的样本。
- 本次需求要求 buffer 中 completed 样本也会因 train_step 推进而过期。
- 因此需要一个对 completed / aborted group 都生效的 group-level 过期 helper，或扩展 `update_expired_status` 的语义。

### 写入点 2：`AgentLoopManager.get_batch`

`get_batch(i)` 在等待当前 step batch 前，先把 producer 的 staleness 基准切到 `i`，并刷新 buffer 中已有 completed / aborted；成功取出非空 batch 后，再把基准推进到 `i + 1` 并刷新 leftover completed / aborted：

```python
async def get_batch(self, batch_size: int, rollout_step: int) -> ProduceBatchResult:
    progress = self._produce_progress
    progress.next_consumer_step = rollout_step

    for task in self.task_runners:
        threshold = getattr(task.produce_strategy, "tail_batch_stale_threshold", 0)
        await self.replay_buffer.refresh_staleness(
            task_name=task.task_name,
            current_rollout_step=rollout_step,
            tail_batch_stale_threshold=threshold,
        )

    while not self._finish_event.is_set():
        ...
        if ready:
            result = await self._get_batch_from_buffer(..., consume_progress=progress)
            if result.rollout_states:
                progress.next_consumer_step = rollout_step + 1
                await self._refresh_staleness_for_all_tasks(rollout_step + 1)
                return result
```

`_get_single_task_batch_from_buffer` 中对返回 batch 调 `refresh_seq_staleness(group, rollout_step)` 可以保留；那只是刷新即将交给训练侧的数据对象，不再写回 buffer，不算第三个 buffer 状态写入点。

这里接受 eventual consistency：`progress.next_consumer_step = i + 1` 与 `refresh_staleness(i + 1)` 不是和 producer `count(COMPLETED)` 共享的全局事务。极短窗口内 producer 可能看到已经推进到 `i + 1` 的 progress，同时 buffer 中还有尚未刷新为 expired 的 completed / aborted leftover，并因此短暂低估缺口。后续的 `refresh_staleness` 和下一轮 produce 会修正 fresh count；当前方案接受这种最终一致性，不为它引入跨 progress / replay buffer 的全局锁。

## ReplayBuffer 改动

新增：

```python
async def refresh_staleness(
    self,
    task_name: str,
    current_rollout_step: int,
    tail_batch_stale_threshold: int,
) -> int:
    ...
```

语义：

- 在 `ReplayBuffer._lock` 下查询该 task 的 `Status.COMPLETED` / `Status.ABORTED` groups。
- 对每个 group 调 `refresh_seq_staleness(group, current_rollout_step)`。
- 更新 `StorageItem.staleness = max(sample.seq_staleness for sample in group)`。
- 如果 `tail_batch_stale_threshold > 0` 且 group 中任意样本 `seq_staleness >= threshold`，则整组样本置 `Status.EXPIRED`，`StorageItem.status = Status.EXPIRED`。
- 返回本次新翻转为 expired 的 group 数。

为避免 destructive `get -> mutate -> put`，storage 增加最小 update 能力：

```python
class StorageBackend:
    async def update(self, items: list[StorageItem]) -> None: ...
```

实现：

- `NaiveStorage.update`：按 `uid` 覆盖 `_items[uid]`，保留原 `timestamp_id`。
- `PandasStorage.update`：flush 后按 `uid` 更新 `status / staleness / item` 列。

这样刷新 completed / aborted staleness 不会和 consumer 抢样本，也不会改变 FIFO / staleness policy 的时间顺序。

## `_pending_tasks` 并发控制

### helper

strategy 内新增小锁保护的 helper：

```python
async def _snapshot_pending(self) -> set[asyncio.Task]:
    async with self._pending_lock:
        return set(self._pending_tasks)

async def _pending_count(self) -> int:
    async with self._pending_lock:
        return len(self._pending_tasks)

async def _claim_done(self, done: set[asyncio.Task]) -> set[asyncio.Task]:
    async with self._pending_lock:
        claimed = done & self._pending_tasks
        self._pending_tasks.difference_update(claimed)
        return claimed

async def _claim_already_done(self) -> set[asyncio.Task]:
    async with self._pending_lock:
        done = {task for task in self._pending_tasks if task.done()}
        self._pending_tasks.difference_update(done)
        return done
```

所有 done task 都必须先 claim，再 `task.result()` / `replay_buffer.put`。

### schedule

`_schedule_tasks_until` 不再直接裸写 set。为避免 pause 正在开始时出现“采样后未纳入 pending”的缝隙，把“检查 `update_event` + sample + create task + add pending”包在同一个 pending lock 中。

这个 lock 不覆盖真正的 rollout generate，只覆盖一次轻量采样和 task 创建：

```python
async def _schedule_one(..., update_event: asyncio.Event | None):
    async with self._pending_lock:
        if update_event is not None and update_event.is_set():
            return False
        if len(self._pending_tasks) >= desired_pending:
            return False

        group_status = Status.EXPIRED if sample_from_expired else Status.ABORTED
        rollout_state = await sampler.sample(task_name=task_name, group_status=group_status)
        task = create_task(
            _timed_generate_group(
                agent_loop,
                rollout_state,
                enable_partial_rollout=self.enable_partial_rollout,
            )
        )
        self._pending_tasks.add(task)
        return True
```

如果不希望在 lock 内 `await sampler.sample(...)`，也可以引入 `_scheduling_count` 防止 pause 在“采样中但尚未 add pending”时提前返回；但这会增加状态。按“尽量少改且易维护”的约束，短锁方案更直接。

`update_event` 是 manager 级暂停信号。`AsyncProduceStrategy` 不再维护 `_pausing` 作为第二套暂停状态；pause drain 只依赖 pending snapshot / claim helper。

不再为 pending task 额外维护 `task -> model_step` 映射。当前约束是：无论共卡还是非共卡，权重更新前都必须先通过 `pause_produce` 清空 `_pending_tasks`，随后才允许 `continue_produce(model_step=...)` 更新 manager 的 `_model_step`。因此一个 `_pending_tasks` 生命周期只对应一个 model step；strategy 在回收 pending 结果时使用本次 `produce_batch` / `pause_produce` 显式传入的 `model_step` 即可。

partial rollout 样本可能已有更早版本的 prefix；`update_sample_version` 只会为新增 token 补当前 model step，最终 staleness 仍按 `min(response_model_steps)` 计算。

### pause

`ProduceStrategy.pause_produce` 使用统一接口；Manager 对 sync / async strategy 使用同一调用形态：

```python
async def pause_produce(
    self,
    agent_loop,
    replay_buffer,
    task_name: str,
    *,
    model_step: int,
    progress: ProduceProgress,
) -> float:
    ...
```

Manager 侧按 `use_global_progress` 选择 progress：

- `use_global_progress=True`：非共卡后台 `produce_loop` 在权重同步点前暂停，传全局 `_produce_progress`，因为后台 producer / trainer consumer 共享同一窗口。
- `use_global_progress=False`：共卡同步 `produce_batch()` 的本次调用收尾，传本次调用的局部 progress。
- Sync strategy 的默认实现忽略 `progress` 并返回 `0.0`，因此无需在 Manager 侧按子类分支。

`AsyncProduceStrategy.pause_produce` 的收尾逻辑为：

```python
async def pause_produce(..., *, model_step: int, progress: ProduceProgress) -> float:
    pause_start = time.perf_counter()

    if await self._pending_count() == 0:
        return 0.0

    rollout_ctl = await get_agent_loop_rollout_ctl(agent_loop)
    await pause_generation(rollout_ctl)

    while True:
        pending_snapshot = await self._snapshot_pending()
        if not pending_snapshot:
            break

        done, _ = await asyncio.wait(
            pending_snapshot,
            timeout=1,
            return_when=asyncio.FIRST_COMPLETED,
        )
        claimed_done = await self._claim_done(done)
        for task in claimed_done:
            await self._put_generated_group(
                task.result(),
                replay_buffer,
                task_name,
                current_rollout_step=progress.next_consumer_step,
                model_rollout_step=model_step,
            )

        if await self._pending_count() > 0:
            await pause_generation(rollout_ctl)
            await asyncio.sleep(1)

    return time.perf_counter() - pause_start
```

关键点：

- producer 和 pause 可以同时 `asyncio.wait` 同一份 snapshot，但只有先 `_claim_done` 的一方会处理结果。
- 另一方拿到的 done task 因为已经不在 `_pending_tasks` 里，`claimed_done` 为空，不会重复 put。
- 不再出现 `self._pending_tasks = set(pending_tasks)` 覆盖新 task 或复活旧 task。

## Manager 生产流程

### `produce_loop`

`produce_loop` 默认从 `progress.producer_future_step` 继续生产，不再接受 `start_rollout_step` 覆盖入口。测试 / resume 如需指定起点，应直接恢复或设置 `progress.producer_future_step`，保证生产 step 只有一个状态来源。manager 初始化时把生产进度放到绝对坐标系原点：

```python
self._produce_progress = ProduceProgress(
    next_consumer_step=1,
    producer_future_step=1,
    consumed_samples={task.task_name: 0 for task in self.task_runners},
    target_samples={task.task_name: 0 for task in self.task_runners},
    target_upto_future_step=0,
)
```

resume 时从 checkpoint 恢复这些状态；trainer 不再把 `self._cur_step` 传进 `produce_loop`。

```python
async def produce_loop(self, batch_size: int):
    while not self._finish_event.is_set():
        if self._status == AgentLoopManagerStatus.FINISH:
            break
        if self._status == AgentLoopManagerStatus.UPDATE_ABORT:
            await self._wait_for_status_exit(AgentLoopManagerStatus.UPDATE_ABORT)
            continue
        if self._status == AgentLoopManagerStatus.EXPIRED_BATCH:
            await self._wait_for_status_exit(AgentLoopManagerStatus.EXPIRED_BATCH)
            continue

        rollout_ctl = await get_agent_loop_rollout_ctl(self.task_runners[0].agent_loop)
        await continue_generation(rollout_ctl)

        status = await self._produce_batch_to_buffer(
            batch_size=batch_size,
            progress=self._produce_progress,
        )

        if status == ProduceBatchStatus.NORMAL:
            self._produce_progress.producer_future_step += 1
        elif status == ProduceBatchStatus.EXPIRED_BATCH:
            self._status = AgentLoopManagerStatus.EXPIRED_BATCH

        await asyncio.sleep(0)
```

`RLDisaggregatedTrainer._fit` 对应改成：

```python
producer_task = create_task(
    self.agent_loop_manager.produce_loop(batch_size=self.train_batch_size)
)
```

恢复训练时，由 `agent_loop_manager.resume(...)` 原地恢复 `self._produce_progress` 的各字段，不再依赖 trainer 传入 `_cur_step`。

### 多 task ExpiredBatch 提前停止

多 task 下，如果任一 task 在当前 `future_step` 上已经整体过期，其他 task 不应继续生产。

因为整体过期只依赖：

```python
current_future_step
model_rollout_step
task.produce_strategy.tail_batch_stale_threshold
```

所以 manager 可以在 `asyncio.gather` 前统一预检查：

实现时把当前 strategy 内部的 `_is_model_expired` 提升为 public wrapper，例如 `is_model_expired`，供 manager 做这个预检查。

```python
expired_tasks = [
    task.task_name
    for task in self.task_runners
    if isinstance(task.produce_strategy, AsyncProduceStrategy)
    and task.produce_strategy.is_model_expired(
        current_future_step,
        self._model_rollout_step,
    )
]
if expired_tasks:
    self.logger.info(f"Expired future_step={current_future_step}, tasks={expired_tasks}")
    return ProduceBatchStatus.EXPIRED_BATCH
```

strategy 内仍保留同样检查，作为单 task / 兼容路径的兜底。

### `_produce_batch_to_buffer`

伪代码：

```python
async def _produce_batch_to_buffer(
    self,
    batch_size: int,
    progress: ProduceProgress,
    *,
    task_batch_sizes: dict[str, int] | None = None,
):
    current_future_step = progress.producer_future_step
    if progress is self._produce_progress:
        # 只有后台生产循环使用全局 progress，需要在这里推进累计 target；
        # colocate 路径传入的是一次性本地 progress，不能污染全局计数。
        self._ensure_target_upto(batch_size, current_future_step)

    # 当前 step 的 task batch sizes 用于本轮 over-sample 预算；active task 由 progress target 决定。
    current_sizes = (
        self._get_task_batch_sizes_for_step(batch_size, current_future_step)
        if task_batch_sizes is None
        else task_batch_sizes
    )
    self._validate_task_batch_sizes(current_sizes, batch_size)

    if self._any_task_model_expired(current_future_step):
        return ProduceBatchStatus.EXPIRED_BATCH

    async def run_task(task):
        name = task.task_name
        return await task.produce_strategy.produce_batch(
            task.agent_loop,
            task.sampler,
            self.replay_buffer,
            current_sizes[name],
            name,
            rollout_step=current_future_step,
            model_rollout_step=self._model_rollout_step,
            update_event=self._update_event,
            progress=progress,
        )

    # 注意：即使 current_sizes[name] == 0，该 task 也可能需要补之前因 expired/failed 造成的缺口。
    tasks_to_run = [
        task
        for task in self.task_runners
        if progress.target_samples[task.task_name] > 0
    ]
    statuses = await asyncio.gather(*(run_task(task) for task in tasks_to_run))
    return _aggregate_status(statuses)
```

## Manager 消费流程

### `get_batch`

`get_batch` 做三件工作：

1. 函数开始时设置 `progress.next_consumer_step = rollout_step`，表示当前正在等待 step `rollout_step` 的训练 batch。
2. 入口刷新一次 buffer 中 completed / aborted 样本的 staleness / expired，避免直接消费进入函数时已经过期的 completed。
3. 成功取出 batch 后，按实际返回数量更新 `progress.consumed_samples`，再设置 `progress.next_consumer_step = rollout_step + 1` 并按下一 step 刷新 leftover completed / aborted。

这里接受 eventual consistency：`refresh_staleness`、producer 的 fresh count 和 consumer 的 get 不是全局事务。为了让逻辑更简单，`get_batch` 不在等待循环里反复 refresh；如果某个 completed / aborted 在等待期间才变 stale，它最多会在本次入口 refresh 与成功消费后 refresh 之间存在一个短暂窗口，下一次入口 / producer 计数 / 成功消费后的 refresh 会收敛状态。

伪代码：

```python
async def get_batch(self, batch_size: int, rollout_step: int) -> ProduceBatchResult:
    progress = self._produce_progress
    progress.next_consumer_step = rollout_step
    await self._refresh_staleness_for_all_tasks(rollout_step)

    while not self._finish_event.is_set():
        if self._status == AgentLoopManagerStatus.EXPIRED_BATCH:
            return ProduceBatchResult(
                rollout_states=[],
                status=ProduceBatchStatus.EXPIRED_BATCH,
            )

        if await self._is_batch_ready(batch_size, rollout_step):
            result = await self._get_batch_from_buffer(
                batch_size,
                rollout_step,
                consume_progress=progress,
            )
            if result.rollout_states:
                progress.next_consumer_step = rollout_step + 1
                await self._refresh_staleness_for_all_tasks(rollout_step + 1)
                return result

        await asyncio.sleep(self._STATUS_POLL_INTERVAL_S)

    return ProduceBatchResult(rollout_states=[])
```

`_get_batch_from_buffer(..., consume_progress=progress)` 应按实际结果计数：

```python
consume_progress.consumed_samples[task_runner.task_name] += len(batch_rollout_states)
```

不要只按 `task_batch_sizes` 加，因为实际返回结果才是 buffer 被消费的权威事实。

### `pause_produce`

manager 侧用 `use_global_progress` 区分使用全局 progress 还是本次调用的局部 progress：

```python
async def pause_produce(
    *,
    use_global_progress: bool,
    progress: ProduceProgress | None = None,
) -> float:
    ...
```

`pause_produce` 入口先校验参数并选择 progress，再置 event / status：

```python
if use_global_progress:
    pause_progress = self._produce_progress

self._update_event.set()
self._status = AgentLoopManagerStatus.UPDATE_ABORT
```

`use_global_progress=True` 是 sticky pause：状态保持到 trainer 完成权重同步 / 评测后调用 `continue_produce()`。

`use_global_progress=False` 用于共卡 `produce_batch()` 的显式收尾，必须传入本次调用的局部 progress；它也会 set `_update_event` / `UPDATE_ABORT`，由下一次 `produce_batch()` 入口的 `continue_produce()` 清理：

```python
else:
    if progress is None:
        raise ValueError(...)
    pause_progress = progress
```

随后调用 strategy 时传同一个 live progress 引用：

```python
pause_time_s += await strategy.pause_produce(
    task.agent_loop,
    self.replay_buffer,
    task.task_name,
    model_step=self._model_step,
    progress=pause_progress,
)
```

## 不再需要 Progress Window

本版使用绝对累计 target 和绝对累计 consumed：

```python
available_abs = consumed_abs + fresh
required = max(0, target_abs - available_abs)
```

因此不需要 `_target_base_step` / `_target_base_consumed`，也不需要在 resume / sync 后重置生产窗口。

权重同步后的 `continue_produce(model_rollout_step=...)` 只负责恢复状态机和更新 rollout 侧模型版本：

```python
def continue_produce(self, model_rollout_step: int) -> None:
    self._status = AgentLoopManagerStatus.NORMAL
    self._model_rollout_step = model_rollout_step
    self._update_event.clear()
```

`progress.producer_future_step`、`progress.target_samples`、`progress.target_upto_future_step` 和 `progress.consumed_samples` 都是训练全局绝对进度，不随 sync 重置。

## Checkpoint

保存 manager state 时追加：

```json
{
  "next_consumer_step": 1,
  "consumed_samples": {"task": 0},
  "producer_future_step": 1,
  "target_samples": {"task": 0},
  "target_upto_future_step": 0
}
```

恢复时：

- 读回上述状态后，原地写回 `self._produce_progress` 的字段；这些字段都是新 checkpoint 的必需字段，缺失时直接 fail fast。
- `_pending_tasks` 不保存；保存前仍要求 pending 为空。
- 每个 strategy 初始化 `_pending_tasks = set()`；不再保存或恢复 strategy-local pause flag。
- 保持现有 `resume()` 进入 `UPDATE_ABORT` 且 `_update_event.set()` 的语义，让 trainer 显式 `continue_produce` 后再恢复生产。

## 共卡路径

`AgentLoopManager.produce_batch` 继续作为 colocate 的同步入口，不改变外部契约。

实现方式：

- 共卡路径不使用 disagg 的全局绝对 target 状态，避免污染后台 producer 的累计进度。
- 共卡路径在调用 `_produce_batch_to_buffer(..., progress=local_progress)` 前构造本次调用的局部 `ProduceProgress`：
  - `next_consumer_step = rollout_step`
  - `producer_future_step = rollout_step`
  - `consumed_samples = {task_name: 0}`
  - `target_samples = current_task_batch_sizes`

共卡路径每次生产后仍调用 `pause_produce(use_global_progress=False, progress=local_progress)` 收尾 pending，然后 `_get_batch_from_buffer` 返回训练 batch。

共卡模式约束：同一个 `AgentLoopManager` 实例只用一种数据提供模式。`SYNC_PRODUCE_BATCH` 收尾会让 manager 保持 `UPDATE_ABORT` / `_update_event.set()`，下一次 `produce_batch()` 入口先调用 `continue_produce(model_rollout_step=rollout_step - 1)` 恢复。不要在两次 sync `produce_batch()` 之间混用 `produce_loop()` / `get_batch()`。

## 删除 / 收敛的旧逻辑

删除或停止使用：

- `AsyncProduceStrategy._process_leftover_samples`
  - 它 destructive `get -> mutate -> put`，会和 consumer 抢 completed。
  - completed / aborted staleness 刷新统一交给 `ReplayBuffer.refresh_staleness`。
- 所有 `self._pending_tasks = set(pending_tasks)`。
- strategy 中基于 `previously_completed_count = replay_buffer.count(COMPLETED)` 的局部 batch 判断。
- strategy 内 `progress=None` 时构造 local progress 的兜底逻辑。
- `AsyncProduceStrategy._pending_task_model_steps`；pending 生命周期与 manager 当前 `_model_step` 周期一致，回收时由调用方显式传入 `model_step`。
- `AsyncProduceStrategy._current_rollout_step` 以及 `pause_produce` 中基于它的 fallback。
- `AsyncProduceStrategy.produce_batch(..., model_rollout_step=None)` 的 fallback；调用方必须显式传入合法 `model_rollout_step`。
- `AgentLoopManager.produce_loop(start_rollout_step=...)` 覆盖入口；producer 起点只来自 `progress.producer_future_step`。
- `_produce_batch_to_buffer(..., rollout_step=...)` / `current_future_step=...` / `use_global_progress` / `progress_override` 这些多入口参数；内部统一使用必传的 `progress.producer_future_step`。
- `_refresh_staleness_for_all_tasks` 中判断 replay buffer 是否存在刷新接口的 fallback；`ReplayBuffer.refresh_staleness` 是固定依赖，缺失应 fail fast。
- `get_batch` while 循环内的重复 completed refresh。
- `_refresh_leftover_counts` 这类只为日志字段再次 recount 的逻辑。
- resume 时读取 `latest_consumer_step` 或用 `manager_state.get(..., default)` 隐藏字段缺失的兼容逻辑。

收敛到：

- `ReplayBuffer.refresh_staleness` 负责 buffer 中 completed / aborted 的 in-place 刷新。
- strategy `_put_generated_group` 负责新生成 / pause drain 结果 put 前刷新。
- strategy `_claim_done` 负责 pending task 的唯一认领。

## 正确性小结

### 消费者取走 batch 不会导致 producer 误补

生产 batch2 时：

```python
available_abs = consumed_abs + fresh
```

consumer 取走 batch1 后：

- `fresh` 减少 `B`
- `consumed_abs` 增加 `B`

所以 `available_abs` 不变，producer 不会把已消费的 batch1 当成缺口。

### completed 样本过期会触发补发

`get_batch` 在入口按当前 step 刷新 completed，成功取出 batch 后也会按下一 step 刷新 leftover：

- 过期样本从 `COMPLETED` 翻成 `EXPIRED`
- `fresh` 下降
- 下一轮 strategy 动态控制看到 `required > 0`
- 自动补发

这不是强事务保证：如果刷新后、消费前又有新 completed 变 stale，当前实现允许它短暂保持 completed。这个窗口通过后续入口 refresh 或成功消费后的下一 step refresh 收敛，换取更少的重复 count / refresh 和更简单的状态维护。

partial rollout 的 staleness 使用 `min(response_rollout_steps)`，仍由 `refresh_seq_staleness` 统一计算。

### 多 task 某 task 整 batch 过期时，全局尽早停

manager 在 gather 前检查所有 task 的 `is_model_expired(current_future_step, model_rollout_step)`。

只要一个 task expired：

- 当前 `_produce_batch_to_buffer` 直接返回 `EXPIRED_BATCH`
- 其他 task 不再新发 rollout
- `produce_loop` 设置 manager status 为 `EXPIRED_BATCH`
- consumer 的 `get_batch` 返回空 batch + `EXPIRED_BATCH`，trainer 优先同步权重

### `_pending_tasks` 不重复 put、不丢 task

两边可以同时 wait snapshot，但 done task 必须先 claim：

```python
claimed = done & self._pending_tasks
self._pending_tasks.difference_update(claimed)
```

同一 task 只有一个协程能 claim 成功，因此不会重复 `task.result()` / `replay_buffer.put`。

新增 task 只通过 helper add，不再用整体赋值覆盖集合，因此不会抹掉新 task 或复活已完成 task。

## 建议测试

1. 单 task：producer 已完成 batch1，consumer 取走 batch1，producer 生产 batch2 时只补 batch2，不额外补 batch1。
2. 单 task：producer 进入 `produce_batch` 后 consumer 中途取走 batch，strategy 通过 live `progress.consumed_samples` 不误补。
3. completed stale：buffer 里已有 completed partial rollout，`get_batch(rollout_step)` 后超过 threshold 的 group in-place 变成 expired。
4. put 前 stale：新生成 group 在 put 前按最新 `progress.next_consumer_step` 刷新；如果已经 stale，直接以 expired 入 buffer。
5. 多 task：任一 task 在当前 future step 上 `EXPIRED_BATCH`，其他 task 不再 schedule。
6. pending race：让 `produce_batch` 和 `pause_produce` 同时 wait 同一个 pending task，确认 replay_buffer 只 put 一次。
7. checkpoint：保存 / 恢复后 `progress.producer_future_step`、`progress.target_samples`、`progress.target_upto_future_step` 和 `progress.consumed_samples` 不回退，buffer leftovers 仍可被后续 train step 消费。
8. fixed over-sample budget：当前只缺 1 个样本时，`over_sample=1, task_batch_size=4` 应调度到 `target_abs + 4`，而不是按缺口只调度到 `available + 2`。
9. tail-batch static mode：进入 tail-batch mode 后，本轮新增任务只从 `Status.EXPIRED` pool 取样，且 `scheduled_target == target_abs` 不超发。

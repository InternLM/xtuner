# AgentLoopManager / AsyncProduceStrategy 重设计

## 1. 目标

这份设计用于指导后续基于当前代码库的增量修改。核心目标是：

- 降低 `AgentLoopManager` 和 `AsyncProduceStrategy` 的认知负荷。
- 保持共卡 / 非共卡外部训练流程稳定。
- 尽量少新增类，避免把复杂度从大类转移成一堆浅接口。
- 把真正通用的能力下沉到已有模块，把业务特定逻辑保留在原有业务层。

对应伪代码见 `design/redesign_disagg.py`。

## 2. 当前最难改的地方

当前 `AgentLoopManager` 最难改的不是单个函数长，而是多个函数共同维护同一组隐藏不变量：

- `ProduceProgress` 是全局累计窗口，但共卡路径又需要局部窗口。
- `target_samples / consumed_samples` 必须使用绝对累计口径。
- consumer 从 replay buffer 取走样本后，producer 不能把这批样本误判为缺口。
- `next_consumer_step` 同时影响 producer put 前的 staleness 刷新。
- checkpoint / resume 必须恢复 progress 字段，且对象引用最好保持稳定。

这些知识目前散落在：

- `AgentLoopManager.__init__`
- `_ensure_target_upto`
- `_build_local_produce_progress`
- `_produce_batch_to_buffer`
- `_get_single_task_batch_from_buffer`
- `get_batch`
- `save / resume`

这会带来三个复杂性症状：

- **变更放大**：修改 target/consumed 口径时，需要同时改生产、消费、checkpoint 和测试。
- **认知负荷**：阅读 `_produce_batch_to_buffer` 时必须知道 `progress is self._produce_progress` 的特殊含义。
- **隐藏未知**：消费 batch 的函数里顺手更新 consumed，调用方很难一眼看出 progress 的写入点。

## 3. 设计原则

本设计按 `software_design.md` 的几个原则约束：

- **深模块优先**：新增接口必须隐藏真实复杂度，不能只是把参数搬到另一个类。
- **信息隐藏**：把同一设计决策只放在一个地方，例如 progress 的累计口径只放在 `ProduceProgress`。
- **接口简单**：尽量复用已有模块，少新增类。
- **容易阅读优先**：入口流程应该能直接读出“生产、暂停、取数、同步”的顺序。
- **渐进修改**：先改最容易出错的知识边界，再考虑更大的结构拆分。

## 4. 总体方案

本设计包含两个主要结构变化：

1. 扩展现有 `ProduceProgress`。
2. 给现有 `ReplayBuffer` 增加通用 batch 方法。

同时新增一个业务接口和一个私有并发 helper：

3. 新增 `ProduceContext`，重点服务 `AsyncProduceStrategy` 的复杂运行时契约。
4. 新增 `_PendingTasks`，只服务 `AsyncProduceStrategy` 的 pending task 并发集合语义。

## 5. ProduceProgress 扩展

`ProduceProgress` 从纯数据对象变成带不变量方法的对象。

建议新增方法：

```python
@classmethod
def build(cls, task_names: list[str]) -> "ProduceProgress": ...

@classmethod
def build_local(
    cls,
    task_names: list[str],
    task_sizes: dict[str, int],
    train_step: int,
) -> "ProduceProgress": ...

def ensure_target_upto(
    self,
    *,
    batch_size: int,
    future_step: int,
    allocate_batch_sizes: Callable[[int, int], dict[str, int]],
) -> dict[str, int]: ...

def begin_consume(self, train_step: int) -> None: ...
def mark_consumed(self, consumed_counts: dict[str, int]) -> None: ...
def finish_consume(self, train_step: int) -> None: ...
def advance_future_step(self) -> None: ...
def state_dict(self) -> dict: ...
def load_state_dict(self, state: dict) -> None: ...
```

关键点：

- batch size 分配策略仍由 `AgentLoopManager.get_task_batch_sizes(batch_size, step)` 提供。
- `ProduceProgress` 只负责累计 target 和 consumed，不理解 task runner。
- `load_state_dict` 原地更新 dict，避免旧引用失效。

这样可以删除或收敛：

- `_ensure_target_upto`
- `_build_local_produce_progress`
- `get_batch` 里直接写 `progress.next_consumer_step`
- `_get_single_task_batch_from_buffer` 里直接更新 consumed

## 6. ReplayBuffer 扩展

把通用 batch 能力加入 `ReplayBuffer`。

建议新增：

```python
async def refresh_staleness(
    self,
    *,
    task_stale_thresholds: dict[str, int],
    current_train_step: int,
    statuses: list[Status] | None = None,
) -> dict[str, int]: ...

async def is_ready(
    self,
    task_batch_sizes: dict[str, int],
    *,
    group_status: Status = Status.COMPLETED,
) -> bool: ...

async def take_batch(
    self,
    task_batch_sizes: dict[str, int],
    *,
    group_status: Status = Status.COMPLETED,
) -> tuple[dict[str, list[list[RolloutState]]], dict[str, int]]: ...

async def put(
    self,
    items: list[RolloutState],
    task_name: str,
    *,
    model_step: int | None = None,
    current_train_step: int | None = None,
    stale_threshold: int | None = None,
) -> None: ...

async def count_statuses(
    self,
    task_names: list[str],
    statuses: list[Status],
) -> dict[str, dict[Status, int]]: ...
```

这些方法适合放进 `ReplayBuffer`，因为它们只依赖存储语义：

- 批量刷新 staleness。
- 判断每个 task 是否有足够 completed groups。
- 按 task size 取 batch。
- 批量统计 leftover。
- 对“生成结果”执行入库前标准化：写入 model step、刷新 staleness，并在有阈值时转 expired。该逻辑只在 `put(...)` 显式传入 `model_step / current_train_step` 时启用。

不放进 `ReplayBuffer` 的逻辑：

- `ProduceBatchResult` 组装。
- generate timing / pause time 的训练日志格式。
- `is_valid_sample_fn` 过滤判断。新的约束是：只对 group status 为 `COMPLETED` 的生成结果执行；`ABORTED / EXPIRED / FAILED` 等非完整样本保留原状态，用于重试、过期池或统计。
- sampler 从 `EXPIRED / ABORTED` 池取样的重试策略。

这些逻辑仍属于 manager / producer 层。

`ReplayBuffer.put(...)` 的默认行为必须保持兼容：不传 `model_step / current_train_step` 时，只按样本当前 `status / staleness` 入库，用于测试和 sampler 手工注入 `ABORTED / EXPIRED / COMPLETED` 样本。生成结果入库时由 producer 显式传入 `model_step / current_train_step`，并在 task 有 stale threshold 时一并传入 `stale_threshold`，`put(...)` 内部再统一执行 version / staleness / expire。group status 聚合使用 `get_group_status(...)`，它只读不改样本；状态翻转只在显式过滤或过期逻辑里发生。

生成结果的顺序调整为：

1. producer 先用 `get_group_status(group)` 判断生成结果是否仍为 `COMPLETED`。
2. 只有 completed group 执行 `is_valid_sample_fn(group)`；无效则把整组标记为 `FILTERED`。
3. producer 调用 `replay_buffer.put(..., model_step=..., current_train_step=..., stale_threshold=...)`。
4. `ReplayBuffer.put` 对生成结果统一执行 version / staleness，并在有阈值时执行 expire，再用 `get_group_status(...)` 按最终状态入库。
5. producer 返回前再次判断 `get_group_status(group) == Status.COMPLETED`，因为 stale group 可能在 put 内被转为 `EXPIRED`。

## 7. ProduceContext

`ProduceContext` 是本设计唯一新增的业务类。

它不是参数袋，而是给 `ProduceStrategy` 提供 task-level 操作界面，主要隐藏异步生产最容易传错的上下文：

```python
ctx.should_abort()
ctx.expired_count()
ctx.available_count()
ctx.sample_group(from_expired_pool=True)
ctx.generate_group(group, enable_partial_rollout=True)
ctx.put_generated_group(group)
```

这样 `AsyncProduceStrategy` 不再直接理解：

- `progress.target_samples[task_name]`
- `progress.consumed_samples[task_name]`
- `update_event.is_set()`
- `replay_buffer.count(task_name, Status.COMPLETED)`
- sampler 的 `group_status` 组合
- agent loop 本地/Ray 调用差异和 generate timing 字段写入
- 生成结果过滤和入库前标准化的顺序

模型过期判断不放在 `ProduceContext` 中重复维护，而是统一由 `ProduceStrategy.is_model_expired(train_step, model_step)` 提供。`AgentLoopManager` 在调度前先检查该入口，strategy 内部如果需要再次判断也调用自己的策略方法。

`ProduceContext` 的目标是隐藏调用契约，不是缩短参数列表。

## 8. _PendingTasks

`_PendingTasks` 是 `AsyncProduceStrategy` 的私有 helper，不是业务抽象，也不进入 manager / trainer 的公开接口。

它要隐藏的是 pending task 集合的并发协议：

```python
class _PendingTasks:
    def count(self) -> int: ...
    async def claim_ready(self) -> set[asyncio.Task]: ...
    async def wait_and_claim(self, *, timeout_s: float) -> set[asyncio.Task]: ...
    async def schedule_one(
        self,
        *,
        max_pending: int,
        should_abort: Callable[[], bool],
        spawn_one: Callable[[], Awaitable[asyncio.Task]],
    ) -> bool: ...
    async def _claim_all(self) -> set[asyncio.Task]: ...
    async def cancel_all(self) -> int: ...
```

放进 `_PendingTasks` 的逻辑：

- `asyncio.wait(snapshot)` 后必须再次 `claim`，避免 `produce_batch` 和 `pause_produce` 重复处理同一个 done task。
- cancel 前先原子取出并清空集合，避免 cancel 后又被其他路径 claim。
- schedule one 时在锁内同时检查 `should_abort()` 和 pending 数量，避免 pause 信号已触发后继续新增 task。

不放进 `_PendingTasks` 的逻辑：

- 从 sampler 取哪个 pool 的样本。
- 如何调用 rollout generate。
- 生成结果是否有效、如何写入 replay buffer。
- 是否触发 tail batch / oversample。

`SyncProduceStrategy` 不需要使用 `_PendingTasks`。同步路径里的 pending set 是单次 `produce_batch` 的局部变量，没有后台 producer 和 consumer pause 的并发访问；强行复用会让简单路径承担异步路径的复杂度。

## 9. AgentLoopManager 保留的职责

`AgentLoopManager` 仍负责流程编排：

- 共卡 `produce_batch`
- 非共卡 `produce_loop`
- 非共卡 `get_batch`
- `pause_produce`
- `continue_produce`
- `shutdown`

状态字段暂时仍保留在 manager：

- `_status`
- `_update_event`
- `_finish_event`
- `_model_step`
- `_pause_time_s`

这些状态转移目前仍比较简单，先保留在 `AgentLoopManager` 内部，并新增公开方法 `shutdown()`，避免 trainer 直接写私有字段。

保存 checkpoint 前，`AgentLoopManager.save()` 通过 `produce_strategy.pending_task_count()` 检查各 task 是否还有未收尾的 rollout task；manager 不直接读取 `_PendingTasks` 私有集合。

`_refresh_for_all_tasks(...)` 不再受环境变量控制，colocate / disagg 都统一刷新传入状态的 staleness；异步策略使用自己的 `stale_threshold`，同步策略没有 `stale_threshold` 时使用默认值 1。

## 10. 核心流程

### 10.1 共卡 produce_batch

```python
self.continue_produce(model_step)
task_sizes = self.get_task_batch_sizes(batch_size, train_step)
local_progress = ProduceProgress.build_local(self.task_names, task_sizes, train_step)

await self._refresh_for_all_tasks(train_step, [Status.COMPLETED, Status.ABORTED])
status = await self._produce_batch_to_buffer(task_sizes, local_progress)
await self.pause_produce(use_global_progress=False, progress=local_progress)

result = await self._get_batch_from_buffer(
    batch_size=batch_size,
    task_batch_sizes=task_sizes,
    consume_progress=local_progress,
)
result.status = status
return result
```

### 10.2 非共卡 produce_loop

```python
progress = self._produce_progress
task_sizes = progress.ensure_target_upto(
    batch_size=batch_size,
    future_step=progress.producer_future_step,
    allocate_batch_sizes=self.get_task_batch_sizes,
)
status = await self._produce_batch_to_buffer(task_sizes, progress)

if status == NORMAL:
    progress.advance_future_step()
elif status == EXPIRED_BATCH:
    self._status = EXPIRED_BATCH
```

### 10.3 非共卡 get_batch

```python
progress.begin_consume(train_step)
await self._refresh_for_all_tasks(train_step, [Status.COMPLETED, Status.ABORTED])

task_sizes = self.get_task_batch_sizes(batch_size, train_step)
if await self.replay_buffer.is_ready(task_sizes):
    result = await self._get_batch_from_buffer(
        batch_size=batch_size,
        task_batch_sizes=task_sizes,
        consume_progress=progress,
    )
    progress.finish_consume(train_step)
    await self._refresh_for_all_tasks(train_step + 1, [Status.COMPLETED, Status.ABORTED])
    return result
```

## 11. 建议迁移步骤

### 步骤 1：扩展 ProduceProgress

- 新增 `build / build_local / ensure_target_upto / begin_consume / mark_consumed / finish_consume / advance_future_step / state_dict / load_state_dict`。
- 替换 manager 中直接写 progress 字段的代码。
- 保持现有测试行为不变。

### 步骤 2：扩展 ReplayBuffer

- 扩展 `refresh_staleness` 为批量接口，单 task 也用 `{task_name: stale_threshold}` 调用。
- 新增 `is_ready` 和 `take_batch`。
- 新增 `count_statuses`。
- 替换 manager 中 `_is_batch_ready` 和 `_get_batch_from_buffer` 的重复 replay 操作；`_get_batch_from_buffer` 保留为消费入口，但 `batch_size / task_batch_sizes / consume_progress` 都是必填参数，不保留默认值和跳过 progress 更新的兜底路径。

### 步骤 3：新增 ProduceContext

- 保留 `ProduceStrategy.produce_batch` / `pause_produce` 方法名，把参数收敛为 `ProduceContext`。
- 先让旧签名包一层 context 调新签名，降低一次性改动风险。
- 再逐步把 `AsyncProduceStrategy` 内部读取 progress / replay / event 的地方改成 `ctx` 方法。

### 步骤 4：新增 _PendingTasks

- 新增 `_PendingTasks`，先把 `_snapshot_pending / _pending_count / _claim_done / _claim_already_done / _claim_all` 搬进去。
- 把 `_schedule_one` 收敛成 `self._pending_tasks.schedule_one(...)`；是否循环调度到目标数量仍留在 `AsyncProduceStrategy`。
- `AsyncProduceStrategy` 保留 `_put_claimed(...)`，采样和 generate task 创建直接在调度处表达，避免再增加只用一次的 helper。
- 不改 `SyncProduceStrategy` 的局部 `pending_tasks`。

### 步骤 5：收敛 AgentLoopManager 公开状态操作

- 保留 `pause_produce(use_global_progress=True)` 作为权重同步前的显式暂停入口。
- 保留 `continue_produce(model_step)` 作为权重同步后的恢复入口。
- 新增 `shutdown()`，替换 trainer 中直接写 `_status / _finish_event` 的代码。
- 旧方法可以短期保留为兼容 wrapper。

### 步骤 6：清理旧私有 helper

可以删除或缩小：

- `_ensure_target_upto`
- `_build_local_produce_progress`
- `_is_batch_ready`
- `_get_single_task_batch_from_buffer`

保留：

- `_build_result`
- `_aggregate_status`
- `_produce_batch_to_buffer`
- `_get_batch_from_buffer` 作为唯一消费入口，内部只做参数校验、`take_batch`、`mark_consumed`、leftover 统计和结果组装
- `_pause_with_progress`

## 12. 测试建议

### ProduceProgress

- global progress 能按 future step 累计 target。
- local progress 不污染 global progress。
- `mark_consumed` 只按真实取出数量推进。
- `state_dict / load_state_dict` 原地更新 dict。

### ReplayBuffer

- `refresh_staleness` 对单个或多个 task 返回正确 expired count。
- `is_ready` 在任一 task 不足时返回 false。
- `take_batch` 返回 `batch_by_task` 和真实 `consumed_counts`。
- `count_statuses` 覆盖所有 leftover 状态。

### AgentLoopManager

- 共卡 `produce_batch` 使用 local progress。
- 非共卡 `produce_loop` 使用 global progress 并推进 future step。
- 非共卡 `get_batch` 在取出 batch 后推进 consumed 和 consumer step。
- `EXPIRED_BATCH` 返回空 batch 且不推进 consumed。
- `shutdown()` 能让后台 producer 退出。

### AsyncProduceStrategy

- strategy 通过 `ProduceContext` 读取 target / available / abort，通过 `ProduceStrategy.is_model_expired(...)` 判断 expired。
- `ctx.put_generated_group` 统一处理生成结果落库。
- `ctx.put_generated_group` 只对 completed group 执行 valid 判断，非完整样本不被误标成 filtered。
- pause drain 和 producer claim 不重复 put pending task。

### _PendingTasks

- `wait_and_claim` 对快照 wait 后只返回仍在集合中的 task。
- `claim_ready` 不重复返回已 claim 的 done task。
- `_claim_all / cancel_all` 清空集合后，后续 claim 返回空。
- `schedule_one` 在 `should_abort()` 为 true 或 pending 数已达到 `max_pending` 时不新增 task。

## 13. 设计总结

本设计不追求一次性完全拆分 `AgentLoopManager`。它优先处理最影响理解和修改的隐藏知识：

- progress 的累计口径收进 `ProduceProgress`。
- replay buffer 的通用批操作收进 `ReplayBuffer`。
- strategy 的运行时契约收进 `ProduceContext`。
- async pending task 的并发 claim / cancel / schedule 协议收进 `_PendingTasks`。

这样新增类少，接口更深，迁移路径也更短。后续如果 `_status / _update_event / _finish_event / _model_step` 的状态转移继续变复杂，再单独评估是否需要进一步抽象状态机。

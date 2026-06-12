# 共卡 / 非共卡生产代码拆分设计

## 1. 背景

当前 `AgentLoopManager` 同时承担两套生产模式：

- 共卡训练：一次 `produce_batch()` 内完成 rollout 生产、pending 收尾、从 replay buffer 取训练 batch。
- 非共卡训练：后台 **Background Producer** 持续写 replay buffer，前台 **Training Consumer** 通过 `get_batch()` 消费，并在 **Expired Produce Batch**、权重同步、评测、checkpoint 之间切换状态。

这两套模式共享同一个 `AgentLoopManager`、同一个 `ProduceProgress`、同一个 `AsyncProduceStrategy` 实现。结果是：

- 共卡路径需要理解 `_status / _update_event / _model_step / _produce_progress` 等非共卡状态。
- 非共卡路径修改容易改变共卡 `produce_batch()` 的同步行为。
- `AsyncProduceStrategy` 的 pending task 既被当作单次调用局部状态，又被当作非共卡跨调用后台状态。

本设计目标是拆开生产侧代码，让共卡生产和非共卡生产各自有独立 **Module**、独立 **Interface** 和独立状态；同时保留 `AsyncProduceStrategyConfig` 在共卡训练中的异步生产能力，并用 `DisaggAsyncProduceStrategyConfig` 显式表达非共卡后台 producer。

## 2. 目标

1. 共卡生产修改不影响非共卡生产。
2. 非共卡 **Background Producer** / **Training Consumer** 状态机修改不影响共卡 `produce_batch()`。
3. 共卡和非共卡使用不同的 strategy config：`AsyncProduceStrategyConfig` 构建共卡 `AsyncProduceStrategy`，`DisaggAsyncProduceStrategyConfig` 构建非共卡 `DisaggAsyncProduceStrategy`。
4. 共卡 async 生产保持简单：pending task 只属于单次 `AgentLoopManager.produce_batch()`，不跨 manager 调用保存。
5. 非共卡 async 生产保留后台 pending、pause/continue、Expired Produce Batch、checkpoint/resume 等能力。

## 3. 非目标

- 不兼容把 `AsyncProduceStrategyConfig(...)` 同时用于共卡和非共卡训练；非共卡训练配置必须显式使用 `DisaggAsyncProduceStrategyConfig(...)`。
- 不改变 replay buffer 的领域语义。
- 不在共卡路径引入非共卡状态机。
- 不把所有共享 helper 都拆成公开接口；共享逻辑可以作为私有 Implementation 留在 manager 包内部。

## 4. 总体方案

把现在一个宽 `AgentLoopManager` 拆成两个 manager **Module**：

- `AgentLoopManager`
- `DisaggAgentLoopManager`

把现在一个 `ProduceProgress` 拆成两个进度 **Module**：

- `ProduceProgress`
- `DisaggProduceProgress`

把现在一个完整 `AsyncProduceStrategy` 拆成两个具体 strategy **Adapter**：

- `AsyncProduceStrategy`
- `DisaggAsyncProduceStrategy`

共卡和非共卡使用不同 config 构建具体 Adapter，不在 strategy config 的 `build(...)` 里传 `mode`：

```python
AsyncProduceStrategyConfig(...).build(...)
# -> AsyncProduceStrategy

DisaggAsyncProduceStrategyConfig(...).build(...)
# -> DisaggAsyncProduceStrategy
```

也就是说，拆分的是执行模式，不是删除共卡 async。

设计约束：

- `AsyncProduceStrategy` 和 `DisaggAsyncProduceStrategy` 不继承公共父类。两者各自显式持有配置字段，少量共享算法用 module-level helper 函数表达。
- 共卡和非共卡的 strategy **Interface** 分开：`ProduceStrategy` 定义共卡 `produce_batch(ctx)`、`pause_produce(ctx)`；`DisaggProduceStrategy` 定义非共卡 `produce_batch(ctx)`、`pause_produce(ctx)`、`pending_task_count()`。共卡 `pause_produce(ctx)` 只收尾本次 manager 调用的 pending，不承载非共卡 checkpoint / update-event 语义。
- `AgentLoopManager` 和 `DisaggAgentLoopManager` 不继承公共父类。task batch 分配、staleness refresh、take batch、result 聚合等共享逻辑用 module-level helper 函数表达。
- `pause_produce` 的关键顺序和 pending drain 协议必须复用当前生产代码语义，核心 drain 协议抽成 `pause_pending_tasks(...)`，而不是藏在某个 manager 父类或 async strategy 父类里。
- 所有 Config 只暴露一个 `build(...)`；`AgentLoopManagerConfig.mode` 只选择 manager 类型，strategy 类型由 `ProduceStrategyConfig` / `DisaggProduceStrategyConfig` 的具体 config 类型决定。
- 不新增 single-task / multi-task manager，也不在本次改造里新增 single/multi 私有分支；继续复用当前 task batch allocation 和结果聚合逻辑。

## 5. Module 职责

| Module | Interface | Implementation |
| --- | --- | --- |
| `AgentLoopManagerConfig` | `build(...)` | 根据 `mode` 构建 task runner、sampler、agent loop 和 manager，并校验 strategy config 类型 |
| `AgentLoopManager` | `produce_batch(batch_size, train_step, model_step)` | 共卡单次生产、局部收尾、取训练 batch |
| `DisaggAgentLoopManager` | `produce_loop`, `get_batch`, `pause_produce`, `continue_produce`, `shutdown` | 非共卡后台生产和消费状态机 |
| `ProduceProgress` | `build`, `add_raw_rewards`, `add_produced`, `add_produce_time` | 单次共卡生产窗口，不进 checkpoint |
| `DisaggProduceProgress` | `ensure_target_upto`, `begin_consume`, `mark_consumed`, `state_dict` | 非共卡绝对累计 target/consumed 和 resume 状态 |
| `ProduceStrategy` | `produce_batch(ctx)`, `pause_produce(ctx)` | 共卡 strategy 抽象接口父类，只接收 `ProduceContext` |
| `DisaggProduceStrategy` | `produce_batch(ctx)`, `pause_produce(ctx)`, `pending_task_count()` | 非共卡 strategy 抽象接口父类，只接收 `DisaggProduceContext` |
| `AsyncProduceStrategy` | `ProduceStrategy` | 持有本次 manager 调用的局部 pending set；`produce_batch` 只生产，`pause_produce` 才 drain |
| `DisaggAsyncProduceStrategy` | `DisaggProduceStrategy` | `_PendingTasks` 跨调用保存，处理 update event 和 model expired |

建议的共享 helper：

| Helper | 用途 |
| --- | --- |
| `allocate_task_batch_sizes(...)` | 复用当前按 task weight 分配 batch 的逻辑 |
| `validate_task_batch_sizes(...)` | 复用 batch size 校验 |
| `refresh_for_all_tasks(...)` | 复用 completed / aborted staleness refresh |
| `take_train_batch(...)` | 复用 replay buffer take、consumed 记账、leftover 统计、result 聚合 |
| `pause_pending_tasks(...)` | 复用 pending task pause / drain / cancel 协议 |

这些 helper 是 Implementation 复用，不是新的业务 **Interface**。调用方仍只看到 mode-specific manager 和 strategy。

## 6. Config 构建规则

`TaskSpecConfig.produce_strategy_config` 接受两类 config：

- 共卡：`ProduceStrategyConfig`，例如 `SyncProduceStrategyConfig` / `AsyncProduceStrategyConfig`。
- 非共卡：`DisaggProduceStrategyConfig`，例如 `DisaggAsyncProduceStrategyConfig`。

所有 Config 都只保留一个 `build(...)`，不提供 `build_colocate(...)` / `build_disaggregated(...)` 这类 mode-specific wrapper，也不在 strategy config 的 `build(...)` 里传 `mode`：

```python
class AgentLoopManagerConfig:
    mode: Literal["colocate", "disaggregated"] = "colocate"

    def build(...):
        if self.mode == "colocate":
            assert isinstance(task_cfg.produce_strategy_config, ProduceStrategyConfig)
        if self.mode == "disaggregated":
            assert isinstance(task_cfg.produce_strategy_config, DisaggProduceStrategyConfig)

        strategy = task_cfg.produce_strategy_config.build(
            sync_weights_interval=sync_weights_interval,
            rollout_controller=rollout_controller,
        )
        if self.mode == "colocate":
            return AgentLoopManager(task_runners, replay_buffer, logger)
        if self.mode == "disaggregated":
            return DisaggAgentLoopManager(task_runners, replay_buffer, logger)
```

`SyncProduceStrategyConfig` 只构建普通 `SyncProduceStrategy`。非共卡训练 producer 只有非共卡 async config，因此 `AgentLoopManagerConfig(mode="disaggregated").build(...)` 下遇到 sync / 共卡 async config 应 fail fast：

```python
class SyncProduceStrategyConfig:
    def build(self, *, sync_weights_interval, rollout_controller):
        return SyncProduceStrategy(...)
```

`AsyncProduceStrategyConfig` 只构建共卡 async strategy：

```python
class AsyncProduceStrategyConfig:
    def build(self, *, sync_weights_interval, rollout_controller):
        return AsyncProduceStrategy(...)
```

`DisaggAsyncProduceStrategyConfig` 只构建非共卡后台 async strategy：

```python
class DisaggAsyncProduceStrategyConfig:
    def build(self, *, sync_weights_interval, rollout_controller):
        return DisaggAsyncProduceStrategy(...)
```

非共卡评估不是后台 producer，不构建 `DisaggProduceStrategy`：

```python
eval_agent_loop_manager_cfg = cfg.eval_agent_loop_manager_cfg.model_copy(update={"mode": "colocate"})
self.eval_agent_loop_manager = eval_agent_loop_manager_cfg.build(...)
# eval task 可以继续使用 SyncProduceStrategyConfig -> SyncProduceStrategy
```

这个构建边界的价值是：共卡 manager 拿到 `ProduceStrategy`，非共卡 manager 拿到 `DisaggProduceStrategy`，配置类型本身表达执行环境。两个 strategy **Interface** 的名字和方法签名不同，因此非共卡 pending / checkpoint 语义不会泄漏到共卡 strategy，也不会藏在 strategy config 的 `mode` 分支里。

## 6.1 Strategy Context

strategy 方法只接收 mode-specific context，不把 `Progress` 作为第二个散装参数：

```python
class BaseProduceContext:
    ...

class ProduceStrategy:
    async def produce_batch(self, ctx: ProduceContext): ...
    async def pause_produce(self, ctx: ProduceContext): ...

class DisaggProduceStrategy:
    async def produce_batch(self, ctx: DisaggProduceContext): ...
    async def pause_produce(self, ctx: DisaggProduceContext): ...
```

原因是 `Progress` 仍按当前内部字段结构由 context 持有，但不应该变成 strategy 方法签名里的通用第二参数：

- `BaseProduceContext` 保留当前 `ProduceContext` 的内部字段结构，例如 `task_batch_size`、`progress`、`stale_threshold`，以及 `sample_group()` / `generate_group()` / `put_generated_group()` 行为。
- `ProduceContext` 是共卡简化版，只去掉非共卡需要的 `update_event`、绝对 consumed/target 访问和 checkpoint 语义，不把 raw rewards / produced samples / produce time 重构成一个 `metrics` 字段。
- `DisaggProduceContext` 继承 `BaseProduceContext`，额外暴露 `update_event`、`available_count()`、`target_abs` 和 `DisaggProduceProgress`。
- 这样可以保留原来 `SyncProduceStrategy.produce_batch(ctx)` 的简单形状；不是改成 `produce_batch(ctx, progress)`。

## 7. 共卡生产流程

共卡路径只允许一个 public 入口：

```python
await manager.produce_batch(batch_size, train_step, model_step=model_step)
```

流程：

1. 根据 `train_step` 计算 task batch sizes。
2. 创建 `ProduceProgress`。
3. `continue_generation()`，切到 rollout 阶段。
4. 各 task 并发调用对应 strategy 的 `produce_batch(ctx)`，只生产到 replay buffer。
5. 等所有 active task 的 `produce_batch(ctx)` 都返回后，manager 再逐个调用 `pause_produce(ctx)`，由 strategy 内部复用 `pause_pending_tasks(...)` 收尾本次 pending。
6. 从 replay buffer 取 completed rollout groups。
7. `pause_generation()`，切回静止态。
8. 返回非空 `ProduceBatchResult`。

关键不变量：共卡 multi-task 下，先达到 target 的 task 不能在自己的 `produce_batch(ctx)` 结束时立刻调用 `pause_pending_tasks(...)`。否则它会提前向 rollout worker 发送 pause，阻塞其他还在生产的 task。pending 收尾必须由 `AgentLoopManager.produce_batch()` 在所有 task 生产结束后统一编排。

异常语义：`asyncio.gather(...)` 只有在所有 task 的 `produce_batch(ctx)` 正常返回时才会进入后续 pause/drain 和 take batch。任一 task 抛异常时，manager 不捕获、不转换成 `ProduceBatchStatus`、不做 best-effort cleanup，让原始异常直接向 trainer 传播并中断训练，避免 `finally` 里的二次异常覆盖真正的失败栈。

业务约束：同一个共卡 `AgentLoopManager` 实例不支持并发调用 `produce_batch()`。`AsyncProduceStrategy` 持有的是本次 manager 调用的局部 pending set，这个约束由 trainer 调用模型保证，不在 manager 里增加复杂防御。

共卡路径不出现：

- `_status`
- `_update_event`
- `_finish_event`
- `DisaggProduceProgress`
- `_PendingTasks`
- `produce_loop`
- `get_batch`
- `continue_produce`

## 8. 非共卡生产流程

非共卡路径由两个 public 入口协作：

```python
producer_task = create_task(manager.produce_loop(batch_size))
get_batch_task = create_task(manager.get_batch(batch_size, train_step=train_step))
done, _ = await wait({producer_task, get_batch_task}, return_when=FIRST_COMPLETED)
if producer_task in done:
    producer_task.result()
produce_result = get_batch_task.result()
```

`DisaggAgentLoopManager` 独占以下状态：

- `status`
- `update_event`
- `finish_event`
- `model_step`
- `pause_time_s`
- `DisaggProduceProgress`

核心不变量：

- **Background Producer** 只在 `NORMAL` 状态下推进 `producer_future_step`。
- **Training Consumer** 成功取出非空 batch 后推进 `consumed_samples` 和 `next_consumer_step`。
- **Expired Produce Batch** 只有在训练侧已有更新 **Model Step** 时，才允许返回空 batch 跳过训练。
- 权重同步前必须 `pause_produce()`，同步/评测后必须 `continue_produce(model_step=...)`。
- **Background Producer** 异常是终止性训练失败，不转换成 manager status；trainer 在等待 `get_batch()` 时必须同时观察 `producer_task`，用 `producer_task.result()` 暴露原始异常栈并中断训练。
- 非共卡异常路径也不做 best-effort cleanup；正常训练结束时才显式 `shutdown()` 并等待后台 producer 退出。

## 9. Async 策略拆分

旧 `AsyncProduceStrategy` 的完整实现拆成两个具体 Adapter，并由两个 config 分别构建。

### 9.1 `AsyncProduceStrategy`

职责：

- 本次 manager `produce_batch()` 期间持有局部 `pending_tasks = set()`。
- 按 `over_sample_threshold`、tail batch、partial rollout 规则调度 rollout group。
- 保留当前 async producer 的生产预算语义：normal 模式的 oversample 预算按 `ceil(over_sample_threshold * task_batch_size)` 计算；tail-batch 模式从 expired / aborted pool 采样，且不再扩大 oversample 窗口。
- 收到完成结果后过滤、写 replay buffer、更新本次统计字段。
- 达到本次 batch target 后返回；不在 `produce_batch(ctx)` 内暂停 agent loop。
- `pause_produce(ctx)` 复用 `pause_pending_tasks(...)` drain 本次 pending；只能由 manager 在所有 task 的 `produce_batch(ctx)` 都返回后调用。

它不负责：

- 跨调用保存 pending。
- 观察 `update_event`。
- 返回 `UPDATE_WEIGHT_AND_ABORT`。
- 维护 `model_step` 状态机。
- checkpoint pending task。
- 继承公共 async 父类。

### 9.2 `DisaggAsyncProduceStrategy`

职责：

- 持有 `_PendingTasks`，允许 pending task 跨多次 `produce_batch()` 调用存在。
- 观察 `ctx.should_abort()`。
- 根据 `model_step / producer_future_step` 判断 **Expired Produce Batch**。
- 保留当前 async producer 的生产预算语义：normal 模式的 oversample 预算按 `ceil(over_sample_threshold * task_batch_size)` 计算；tail-batch 模式从 expired / aborted pool 采样，且不再扩大 oversample 窗口。
- `pause_produce()` drain 或 cancel pending。
- 为 checkpoint 提供 `pending_task_count()`。

它不负责：

- 从 replay buffer 取训练 batch。
- 推进 `DisaggProduceProgress` 的 consumer step。
- 触发权重同步。
- 继承公共 async 父类。

### 9.3 pause pending helper

当前最新 `pause_produce` 有两个层次：

1. manager 层：先设置暂停信号，切换 manager 状态，再暂停 rollout controller。
2. strategy 层：如果还有 pending task，周期性发送 agent loop pause，claim 已完成任务并入库，超过 timeout 后 cancel 剩余 pending。

拆分后保留这个顺序，但把 strategy 层 pending drain 抽成全局 helper。共卡路径的 manager 层不设置非共卡 update-event/status，只负责“所有 task produce 完成后再统一收尾”的顺序；非共卡路径的 manager 层仍负责设置暂停信号和状态：

```python
async def pause_pending_tasks(
    *,
    pending_tasks: set[asyncio.Task] | _PendingTasks,
    ctx,
    put_claimed_task,
) -> float:
    if isinstance(pending_tasks, set):
        pending = _LocalPendingTasks(pending_tasks)
    else:
        pending = pending_tasks

    if pending.count() == 0:
        return 0.0

    pending_pause_tasks = {create_task(request_agent_loop_pause(ctx))}
    deadline = now() + PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S
    next_periodic_pause = now() + PERIODIC_ABORT_INTERVAL_S

    while pending.count() > 0:
        if now() > deadline:
            await pending.cancel_all()
            break

        if now() >= next_periodic_pause:
            pending_pause_tasks.add(create_task(request_agent_loop_pause(ctx)))
            next_periodic_pause += PERIODIC_ABORT_INTERVAL_S

        claimed = await pending.wait_and_claim(timeout_s=1)
        for task in claimed:
            await put_claimed_task(task)

    await cancel_and_drain(pending_pause_tasks)
    return elapsed()
```

共卡路径直接把本次调用的局部 `set[Task]` 传给 helper，helper 内部自动包成 `_LocalPendingTasks`；非共卡路径直接传 `_PendingTasks`。这样 pause 协议复用，但 pending 的生命周期仍然独立：

- 共卡：pending 生命周期等于一次 `produce_batch()`。
- 非共卡：pending 生命周期跨多次后台 `produce_batch()`。

## 10. Progress 拆分

### 10.1 `ProduceProgress`

构建入口只保留 `build(...)`：

```python
ProduceProgress.build(
    task_names=task_names,
    target_samples=task_batch_sizes,
    train_step=train_step,
)
```

字段：

- `producer_future_step`
- `target_samples`
- 本次 raw reward / produced samples / produced tokens / produce time 统计字段

特点：

- 不保存到 checkpoint。
- 不维护 `next_consumer_step / consumed_samples / target_upto_future_step`。
- 不新增 `model_step` 字段；`model_step` 仍是 manager 构建 `ProduceContext` 时传入的运行时参数。
- 不把当前 `target_samples` 改名为 `task_batch_sizes`。共卡路径里 `target_samples` 表达本次 `produce_batch()` 的局部 target，不是非共卡的绝对累计 target。
- 不维护非共卡后台 producer 推进语义；`producer_future_step` 只作为本次 staleness / future step 写入字段。
- 不暴露 `state_dict()`。

### 10.2 `DisaggProduceProgress`

字段：

- `producer_future_step`
- `next_consumer_step`
- `target_samples`
- `consumed_samples`
- `target_upto_future_step`
- 后台 producer 统计字段

特点：

- `target_samples / consumed_samples` 使用绝对累计口径。
- `state_dict / load_state_dict` 是非共卡 checkpoint 的一部分。
- producer 和 consumer 共享同一个对象引用。

## 11. ReplayBuffer 保持共享

Replay buffer 是真正共享的深 **Module**，不按共卡/非共卡拆。它提供：

- `put(...)`
- `refresh_staleness(...)`
- `is_ready(...)`
- `take_batch(...)`
- `count_statuses(...)`

共享理由：

- 共卡和非共卡都需要落库和取 completed rollout groups。
- Replay buffer 不理解 manager 状态机。
- Replay buffer 的 **Interface** 已经足够表达 storage / replay ordering 行为。

## 12. Trainer 集成

共卡 trainer：

```python
cfg.agent_loop_manager_cfg.mode = "colocate"
# task.produce_strategy_config = SyncProduceStrategyConfig(...) 或 AsyncProduceStrategyConfig(...)
self.agent_loop_manager = cfg.agent_loop_manager_cfg.build(...)
```

非共卡 trainer：

```python
cfg.agent_loop_manager_cfg.mode = "disaggregated"
# task.produce_strategy_config = DisaggAsyncProduceStrategyConfig(...)
self.agent_loop_manager = cfg.agent_loop_manager_cfg.build(...)
```

评测 manager 建议始终用共卡 manager：

```python
cfg.eval_agent_loop_manager_cfg.mode = "colocate"
self.eval_agent_loop_manager = cfg.eval_agent_loop_manager_cfg.build(...)
```

原因：evaluation 是一次性 `produce_batch()`，不是后台 **Background Producer**。

## 13. 迁移步骤

1. 用 `Literal["colocate", "disaggregated"]` 表达 `AgentLoopManagerConfig.mode`，`AgentLoopManagerConfig` 只保留 `build(...)`。
2. 新增 `AgentLoopManager`，把当前 `produce_batch()` 的共卡逻辑迁移过去。
3. 新增 `DisaggAgentLoopManager`，把 `produce_loop/get_batch/pause/continue/shutdown/save/resume` 迁移过去。
4. 拆出 `ProduceProgress` 和 `DisaggProduceProgress`，`ProduceProgress` 只保留 `build(...)` 构造入口。
5. 把当前 `AsyncProduceStrategy` 拆成 `AsyncProduceStrategy` 和 `DisaggAsyncProduceStrategy`，并新增 `DisaggAsyncProduceStrategyConfig`。
6. 把 batch allocation、refresh、take batch、pause pending drain 抽成 module-level helper。
7. trainer 通过设置 manager config `mode` 后调用同一个 `build(...)`；非共卡训练配置同步替换为 `DisaggAsyncProduceStrategyConfig`。
8. 保留必要兼容导出，但不保留“同一个 strategy config 靠 mode 切换”的兼容语义。

## 14. 测试建议

共卡 manager：

- `AgentLoopManagerConfig(mode="colocate").build(...)` 构建出的 manager 能通过 public `produce_batch(...)` 返回非空训练 batch。
- 共卡 multi-task `produce_batch(...)` 按 task 权重稳定返回训练数据。
- 共卡 multi-task async 生产中，先完成的 task 不会提前暂停 rollout worker；所有 active task 完成生产后才统一收尾 pending。
- 共卡 async `produce_batch(...)` 返回后，再次调用仍能正常生产，不受上一次 pending 收尾影响。

非共卡 manager：

- `AgentLoopManagerConfig(mode="disaggregated").build(...)` 构建出的 manager 能通过 public `produce_loop(...)` / `get_batch(...)` 协作返回训练 batch。
- `produce_loop/get_batch/pause_produce` 的 single-task 和 multi-task public 行为一致，避免复制后台状态机。
- `produce_loop/get_batch` 仍处理空/非空 **Expired Produce Batch**。
- `pause_produce/continue_produce` 顺序不变。
- checkpoint/resume 后，public `get_batch(...)` / `continue_produce(...)` 行为延续保存前的 producer progress 和 model step。

策略：

- `AsyncProduceStrategy` 通过 public `produce_batch(ctx)` / `pause_produce(ctx)` 覆盖 oversample、partial rollout、tail batch 和本次 pending 收尾结果。
- `DisaggAsyncProduceStrategy` 通过 public `produce_batch(ctx)` / `pause_produce(ctx)` / `pending_task_count()` 覆盖跨调用 pending、abort、expired 和 checkpoint 前 fail fast。

trainer：

- 共卡 trainer 只依赖 `produce_batch()`。
- 非共卡 trainer 只依赖 `produce_loop/get_batch/pause/continue/shutdown`。
- 非共卡 trainer 的 eval manager 是 colocate manager，initial evaluate 后按非共卡训练需求恢复 producer。

## 15. 关键判断

`AsyncProduceStrategy` 的领域含义不是“非共卡策略”，而是“共卡异步 rollout 生产策略”。非共卡后台 producer 需要显式的 `DisaggAsyncProduceStrategyConfig` / `DisaggAsyncProduceStrategy`。

真正需要隔离的是执行环境：

- 共卡执行环境：局部 pending，单次调用完成。
- 非共卡执行环境：后台 pending，跨调用状态机。

所以最终代码形状应是：

```python
AsyncProduceStrategyConfig
    -> AsyncProduceStrategy

DisaggAsyncProduceStrategyConfig
    -> DisaggAsyncProduceStrategy
```

而不是让 `AsyncProduceStrategyConfig` 内部继续同时认识共卡和非共卡两套执行协议。

后者会让一个 Config 继续知道两套执行协议，复杂度只是换位置，不能提供足够的 **Locality**。

# RL Trainer 高级使用

本文介绍 `AgentLoopManager`、`ProduceStrategy`、`RLColocateTrainer` 和非共卡 `RLDisaggregatedTrainer` 的用户侧用法。重点是理解它们各自负责什么、训练流程如何串起来，以及常用配置如何选择。

如果你只是跑单轮 GRPO，通常先看[基础训练教程](../tutorial/rl_grpo_trainer.md)。当你需要多任务、异步 rollout、partial rollout，或把训练和 rollout 放到不同卡组时，再关注本文。

## 整体关系

一条 RL 训练链路可以粗略理解为：

```text
Sampler
  -> AgentLoop 生成 response
  -> Judger 写入 reward
  -> ProduceStrategy 控制生产节奏
  -> AgentLoopManager 写入/读取 ReplayBuffer
  -> RLTrainer 训练、评估、保存、同步权重
```

几个模块的职责边界如下：


| 模块                     | 用户侧理解                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------- |
| `AgentLoopManager`       | 把 sampler、agent loop、judger 和生产策略组装成一个或多个 rollout 任务，并向 trainer 提供训练 batch。 |
| `ProduceStrategy`        | 决定 rollout 数据如何生产：同步生产一批，或在后台持续生产、允许超发和续跑。                           |
| `RLColocateTrainer`      | 训练 worker 和 rollout worker 使用同一组卡，rollout 和训练按 step 切换资源。                          |
| `RLDisaggregatedTrainer` | 训练 worker 和 rollout worker 使用不同卡组，rollout 在后台生产，训练在前台消费。                      |

## AgentLoopManager

`AgentLoopManagerConfig` 是 generation 侧的总入口。它本身不定义“如何生成一条样本”，而是把下面这些配置绑定成训练任务：

- `sampler_config`：从数据集中采样 prompt，并按 `prompt_repeat_k` 组成 group。
- `agent_loop_config`：定义一次 rollout 如何执行，例如单轮问答或工具调用。
- `judger_config`：给 rollout 结果打分。
- `produce_strategy_config`：控制 rollout 生产节奏。
- `weight`：多任务训练时的 batch 分配权重。

单任务配置示例：

```{code-block} python
:caption: 配置单个 rollout 任务
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)

agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=train_agent_loop_config,
        judger_config=judger_config,
        produce_strategy_config=SyncProduceStrategyConfig(),
        sampler_config=train_sampler_config,
    )
)
```

多任务训练时传入 `TaskSpecConfig` 列表即可。`weight` 控制每个训练 step 的 `train_batch_size` 如何分给不同任务：

```{code-block} python
:caption: 配置多任务 rollout
agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=[
        TaskSpecConfig(
            task_name="math",
            weight=2.0,
            agent_loop_config=math_agent_loop_config,
            judger_config=math_judger_config,
            produce_strategy_config=SyncProduceStrategyConfig(),
            sampler_config=math_sampler_config,
        ),
        TaskSpecConfig(
            task_name="code",
            weight=1.0,
            agent_loop_config=code_agent_loop_config,
            judger_config=code_judger_config,
            produce_strategy_config=SyncProduceStrategyConfig(),
            sampler_config=code_sampler_config,
        ),
    ]
)
```

上例中，若 `train_batch_size=96`，默认会尽量按 `math:code = 2:1` 分配 batch。

## ProduceStrategy

`ProduceStrategy` 决定数据生产方式。用户通常只需要选择配置类。

### SyncProduceStrategyConfig

`SyncProduceStrategyConfig` 是最简单、最接近 on-policy 的选择：当前训练 step 需要数据时，先生成足够的 rollout group，再把这批数据交给 trainer。

适用场景：

- 共卡训练的默认选择。
- 希望每个 step 尽量使用当前权重生成的数据。
- 不需要 partial rollout、超发或 stale 样本复用。

配置：

```python
produce_strategy_config = SyncProduceStrategyConfig()
```

### AsyncProduceStrategyConfig

`AsyncProduceStrategyConfig` 用于提升 rollout 吞吐。它会允许 producer 在满足当前 batch 之外继续准备后续样本，并支持 partial rollout、stale 样本复用和过期样本重试。

常用配置：

```{code-block} python
:caption: 配置异步生产策略
from xtuner.v1.rl.agent_loop_manager import AsyncProduceStrategyConfig

produce_strategy_config = AsyncProduceStrategyConfig(
    over_sample_threshold=0.2,
    enable_partial_rollout=True,
    max_staleness=1,
    tail_batch_trigger_size=64,
)
```

参数含义：

| 参数 | 说明 |
| --- | --- |
| `over_sample_threshold` | 允许额外生成的比例。值越大，rollout 侧越容易保持满载，但也可能产生更多非当前 step 的样本。 |
| `enable_partial_rollout` | 权重同步前被暂停的 rollout 是否允许在同步后续跑。工具调用或多轮任务使用前需要确认 AgentLoop 支持续跑。 |
| `max_staleness` | 允许样本相对当前训练进度滞后的同步周期数。值越大，吞吐更宽松，on-policy 程度更弱。 |
| `tail_batch_trigger_size` | 过期样本累计到一定数量后，进入 tail batch 模式，优先重试这些样本。 |

`max_staleness` 按“权重同步周期”计数。代码中实际使用的过期阈值是：

```text
stale_threshold = (max_staleness + 1) * sync_weights_interval
```

这里的 `+1` 表示当前同步周期内天然允许的滞后：`model_step` 表示 rollout 使用的是哪个 train step 同步后的模型，而训练第 `model_step + 1` 步时这批样本仍然是当前权重生成的样本。`max_staleness=0` 表示只允许当前同步周期内的样本；`max_staleness=1` 表示还允许额外跨过一个同步周期。

超发和 partial rollout 都会受到 `max_staleness` 影响：

- `over_sample_threshold>0` 会为未来 step 提前生成样本。如果这些样本跨过下一次权重同步点，只有 `max_staleness` 允许时才会继续保留为可训练样本。
- `enable_partial_rollout=True` 会让被暂停的 response 在同步后续跑。样本的 staleness 按 response 中最早的模型版本计算，因此跨同步周期续跑时也需要 `max_staleness` 留出空间。

tail batch 用于处理异步生产中已经过期的样本。当 `expired` 样本数量达到 `tail_batch_trigger_size` 时，`AsyncProduceStrategy` 会进入 tail batch 模式：本轮不再按 `over_sample_threshold` 超发，只补齐必要目标，并优先从过期样本池中取样重试。可以把它理解为一次非超发的同步补齐生产；它的目的不是提高吞吐，而是把长尾过期样本重新收集起来，避免它们长期留在 buffer 中。

注意：不建议同时设置 `max_staleness>0` 且 `enable_partial_rollout=False`。这种组合下，长尾超发样本在权重同步后可能因为不支持 partial rollout 被重置（当前在 `RolloutWorker` 中重置样本只保留 prompt 字段）；但由于每次重置过期信息归0，它们不会过期，tail batch 不会及时接管，下一轮同步窗口内仍然可能生成不完并反复重试。当前还没有支持 `tail_batch_max_tries` 机制来按重试次数触发 tail batch。因此 `max_staleness>0` 时，优先开启 `enable_partial_rollout=True`。

在非共卡训练中，不要自定义 `should_continue_fn` 做早停；当前 `RLDisaggregatedTrainer` 要求它保持默认行为，否则可能导致后台生产不足、前台训练消费不匹配。

`ReplayBufferConfig` 通常按示例选择即可：同步入门配置先用 `SyncReplayBufferConfig()`；共卡异步生产可以参考 `rl_grpo_gsm8k_async.py` 使用 `AsyncReplayBufferConfig()`，让消费侧优先处理 staleness 较高的 completed 样本。

### AsyncProduceStrategy 的两套接口

`AsyncProduceStrategy` 的核心接口分成两类，来支持共卡和非共卡：

- 共卡路径中，`AgentLoopManager.produce_batch()` 使用本地 progress 串起“生产到 buffer -> pause 收尾 -> 从 buffer 取训练 batch”。
- 非共卡路径中，`AgentLoopManager.produce_loop()` 在后台持续调用生产接口；前台 trainer 用 `get_batch()` 消费，到同步点时再调用 `pause_produce()`，同步权重后用 `continue_produce()` 恢复。

## RLColocateTrainer

`RLColocateTrainer` 对应配置类 {class}`~xtuner.v1.train.rl_trainer.RLColocateTrainerConfig`。它让训练 worker 和 rollout worker 使用同一组资源。

流程可以理解为：

```text
rollout 生成一批数据
  -> 暂停/释放 rollout 侧资源
  -> 训练 worker 消费这批数据
  -> 到同步点时把训练权重同步给 rollout worker
  -> 进入下一步 rollout
```

最小结构：

```{code-block} python
:caption: 配置共卡 Trainer
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig

resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=8,
)

trainer = RLColocateTrainerConfig(
    resources=resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=SyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_config,
    load_from=model_path,
    total_train_steps=1000,
    train_batch_size=128,
    sync_weights_interval=1,
    enable_evaluate=True,
    evaluate_step=50,
    work_dir=work_dir,
)
```

常用字段：


| 字段                                  | 说明                                                   |
| --------------------------------------- | -------------------------------------------------------- |
| `resources`                           | 共卡模式使用的一组训练/rollout 共享资源。              |
| `train_batch_size`                    | 每个训练 step 消费多少个 rollout group。               |
| `sync_weights_interval`               | 每多少个训练 step 同步一次权重到 rollout worker。      |
| `checkpoint_interval` / `hf_interval` | 保存间隔；启用时需要是 `sync_weights_interval` 的倍数。 |
| `enable_evaluate` / `evaluate_step`   | 是否评估以及评估间隔；评估只在权重同步点执行。         |

共卡常用模式：

| 模式 | 关键配置 | 说明 |
| --- | --- | --- |
| 严格 on-policy | `SyncProduceStrategyConfig()`，`sync_weights_interval=1` | 每个 step 先 rollout，再训练，再同步权重。 |
| 低频同步 | `SyncProduceStrategyConfig()`，`sync_weights_interval>1` | 减少权重同步开销，同一同步周期内的多个 step 使用同一版权重 rollout。 |
| 共卡 stale 超发 | `AsyncProduceStrategyConfig(over_sample_threshold>0, max_staleness>0)` | 允许超发样本跨额外同步周期继续训练，吞吐更高，但 on-policy 程度更弱。 |
| 共卡 partial rollout | `AsyncProduceStrategyConfig(over_sample_threshold>0, max_staleness>0, enable_partial_rollout=True)` | 适合长 response 或工具链任务 |

在共卡异步模式里，`over_sample_threshold` 生成的未来样本和 partial rollout 续跑出的样本，如果要跨过权重同步点继续用于训练，都依赖 `max_staleness>0` 放宽过期阈值。

共卡模式适合资源较少、希望配置简单的场景。

## RLDisaggregatedTrainer

非共卡训练对应配置类 {class}`~xtuner.v1.train.rl_trainer.RLDisaggregatedTrainerConfig` 和运行类 {class}`~xtuner.v1.train.rl_trainer.RLDisaggregatedTrainer`。日志中有时会看到 `RLDisaggTrainer`，指的就是这个非共卡训练器。

它把资源拆成两组：

- `train_resources`：训练 worker 使用。
- `rollout_resources`：rollout worker 使用。

运行时，rollout producer 会在后台持续把样本写入 replay buffer，trainer 前台按 step 取 batch 训练。到权重同步点时，trainer 会先让后台生产暂停，再保存、同步权重、评估，最后恢复生产。

```text
后台：rollout producer -> replay buffer -> rollout producer -> ...
前台：get batch -> train -> sync point -> pause producer -> sync weights -> continue producer
```

非共卡的训练样本生产只使用 `AsyncProduceStrategyConfig`。原因是训练 producer 和 trainer 并发运行，需要 strategy 保留 pending rollout、响应暂停信号，并在同步点由 trainer 显式收尾。非共卡评估仍然使用 `SyncProduceStrategyConfig`：评估在权重同步后、恢复后台 producer 前执行，只需要生成一批固定 eval batch，不需要后台超发、staleness 复用或 partial rollout。

```{code-block} python
:caption: 非共卡训练与评估的 produce strategy
from xtuner.v1.rl.agent_loop_manager import (
    AsyncProduceStrategyConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)

train_task = TaskSpecConfig(
    task_name="train_task",
    agent_loop_config=train_agent_loop_config,
    judger_config=judger_config,
    produce_strategy_config=AsyncProduceStrategyConfig(
        over_sample_threshold=0.2,
        max_staleness=1,
    ),
    sampler_config=train_sampler_config,
)

eval_task = TaskSpecConfig(
    task_name="eval_task",
    agent_loop_config=eval_agent_loop_config,
    judger_config=judger_config,
    produce_strategy_config=SyncProduceStrategyConfig(),
    sampler_config=eval_sampler_config,
)
```

配置示例：

```{code-block} python
:caption: 配置非共卡 Trainer
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.train.rl_trainer import RLDisaggregatedTrainerConfig

train_resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=4,
)
rollout_resources = AcceleratorResourcesConfig(
    accelerator="GPU",
    num_workers=4,
)

trainer = RLDisaggregatedTrainerConfig(
    train_resources=train_resources,
    rollout_resources=rollout_resources,
    train_worker_cfg=train_worker_cfg,
    rollout_config=rollout_config,
    tokenizer_path=model_path,
    replay_buffer_config=AsyncReplayBufferConfig(),
    agent_loop_manager_cfg=agent_loop_manager_cfg,
    eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
    evaluator_config=evaluator_config,
    load_from=model_path,
    total_train_steps=1000,
    train_batch_size=128,
    sync_weights_interval=4,
    enable_evaluate=True,
    evaluate_step=20,
    work_dir=work_dir,
)
```

常用模式：

| 模式 | 关键配置 | 说明 |
| --- | --- | --- |
| 非共卡 on-policy | `AsyncProduceStrategyConfig(over_sample_threshold=0, max_staleness=0)`，`sync_weights_interval=1` | 每步同步权重，训练和 rollout 资源分离，但两边资源等待失去非共卡优势。目前尚未支持。 |
| Stream off-policy | `AsyncProduceStrategyConfig(over_sample_threshold=0, max_staleness=0)`，`sync_weights_interval>1` | 减少同步频率；`max_staleness=0` 仍允许当前同步周期内的天然滞后。 |
| Async stale | `AsyncProduceStrategyConfig(over_sample_threshold>0, max_staleness>0)` | 允许后台 producer 适度超前，并让超发样本跨额外同步周期继续可用。 |
| Async partial rollout | `AsyncProduceStrategyConfig(over_sample_threshold>0, enable_partial_rollout=True, max_staleness>0)` | 适合长 response 或长工具链任务，利用partial rollout来降低权重同步打断后的重跑成本。 |

使用非共卡训练时注意：

- `train_resources` 和 `rollout_resources` 应该是不同资源池。
- 训练任务的 `TaskSpecConfig.produce_strategy_config` 使用 `AsyncProduceStrategyConfig`；评估任务使用 `SyncProduceStrategyConfig`。
- `evaluate_step`、`checkpoint_interval`、`hf_interval` 启用时都需要是 `sync_weights_interval` 的倍数。
- 评估会在恢复后台 producer 前执行，避免评估和后台 rollout 抢同一组 rollout 资源。
- 当前非共卡权重同步入口由 trainer 封装；用户通常只需要配置 `sync_weights_interval`。

## 如何选择


| 场景                                        | 推荐                                                             |
| --------------------------------------------- | ------------------------------------------------------------------ |
| 第一次配置 RL 训练                          | `RLColocateTrainerConfig` + `SyncProduceStrategyConfig()`。      |
| GPU 数量有限，想先跑通任务                  | 共卡模式。                                                       |
| rollout 明显慢于训练，且有额外 rollout 资源 | 非共卡模式。                                                     |
| 需要更高 rollout 吞吐                       | `AsyncProduceStrategyConfig(over_sample_threshold>0)`。          |
| 长 response 经常被同步打断                  | 开启`enable_partial_rollout=True`。 |
| 强 on-policy 要求                           | 保持`sync_weights_interval=1`，少用或不用 stale/oversample。     |

完整配置可以参考：

- `examples/v1/config/rl_grpo_gsm8k_judge.py`：共卡同步 GRPO。
- `examples/v1/config/rl_grpo_gsm8k_async.py`：共卡异步生产。
- `examples/v1/config/rl_disagg_single.py`：非共卡单任务。
- `examples/v1/config/rl_disagg_multi.py`：非共卡多任务。

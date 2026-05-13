# Judger

`Judger` 是 XTuner RL 中负责给 rollout 结果打分的组件。它接收 `RolloutState`，读取模型生成结果和标签，计算 reward，并把分数写回 `rollout_state.reward`。后续的 advantage 计算、训练数据构建和默认评测逻辑都会依赖这个字段。

在典型流程中，`AgentLoop` 先调用推理引擎生成 `response_ids`，再把 token 解码成文本形式的 `response`，最后调用 `judger.judge(rollout_state)`。

## Judger 类型

`xtuner/v1/rl/judger/native.py` 中定义了 Judger 的基础类型。可以先把它理解成两层：第一层是所有 Judger 统一实现的 `judge()` 接口；第二层是不同运行形态，决定这个 `judge()` 在本地执行、在 Ray actor 中执行，还是组合多个子 Judger 一起执行。

```text
                                  ┌─────────────────────────────┐
                                  │        Judger (ABC)          │
                                  │  async judge(RolloutState)   │
                                  └──────────────┬──────────────┘
                                                 │
                 ┌───────────────────────────────┼───────────────────────────────┐
                 │                               │                               │
     ┌───────────▼───────────┐       ┌───────────▼───────────┐       ┌───────────▼───────────┐
     │      NativeJudger      │       │      RemoteJudger      │       │       JudgerPool      │
     │                        │       │                        │       │                       │
     │ 本地执行 judge()        │       │ judge() 的远程代理       │       │ 多个 RemoteJudger 副本 │
     │ 调用 reward_handler    │       │ 调用 JudgerActor.remote │       │ 按 round-robin 分发    │
     └───────────┬───────────┘       └───────────┬───────────┘       └───────────┬───────────┘
                 │                               │                               │
                 │                               ▼                               │
                 │                   ┌───────────────────────┐                   │
                 │                   │      JudgerActor       │                   │
                 │                   │  Ray worker 内部持有   │                   │
                 │                   │      NativeJudger      │                   │
                 │                   └───────────┬───────────┘                   │
                 │                               │                               │
                 └───────────────────────────────▼───────────────────────────────┘
                                         reward_handler / HTTP endpoint


                                  ┌─────────────────────────────┐
                                  │       ComposedJudger         │
                                  │  多分支路由与结果合并         │
                                  └──────────────┬──────────────┘
                                                 │
                         ┌───────────────────────┼───────────────────────┐
                         │                       │                       │
                 ┌───────▼───────┐       ┌───────▼───────┐       ┌───────▼───────┐
                 │   branch A    │       │   branch B    │       │   branch ...  │
                 │ 任意 Judger    │       │ 任意 Judger    │       │ 任意 Judger    │
                 └───────┬───────┘       └───────┬───────┘       └───────┬───────┘
                         │                       │                       │
                         └───────────────────────┼───────────────────────┘
                                                 ▼
                         select_fn 选择 branch，merge_fn 合并 reward dict
```

`RemoteJudger` 的作用是给 Ray actor 形式的 Judger 提供一个对外统一接口。它本身不实现具体打分逻辑，而是持有 `JudgerActor` 的 actor proxy，并在自己的 `judge()` 中调用 `actor.judge.remote(...)`。这样上层代码只需要面向 `Judger.judge()` 编程，不需要关心当前 Judger 是本地对象、Ray actor，还是多副本池中的一个远程副本。

构建时有两条入口：

```text
JudgerConfig
  ├─ num_ray_actors = 0  ──► NativeJudger
  ├─ num_ray_actors = 1  ──► RemoteJudger
  │                           └─ JudgerActor
  │                                └─ NativeJudger
  └─ num_ray_actors > 1  ──► JudgerPool
                              ├─ RemoteJudger -> JudgerActor -> NativeJudger
                              ├─ RemoteJudger -> JudgerActor -> NativeJudger
                              └─ ...

ComposedJudgerConfig
  └─ ComposedJudger
       ├─ branches["a"] -> JudgerConfig 或 ComposedJudgerConfig
       ├─ branches["b"] -> JudgerConfig 或 ComposedJudgerConfig
       └─ select_fn + merge_fn 控制路由和合并
```

普通 `JudgerConfig` 根据 `num_ray_actors` 决定执行模式：

| 配置 | 构建结果 | 含义 |
| --- | --- | --- |
| `num_ray_actors = 0` | `NativeJudger` | 本地执行，不启动 Ray Judger actor |
| `num_ray_actors = 1` | `RemoteJudger -> JudgerActor -> NativeJudger` | 单个 Ray actor 执行打分 |
| `num_ray_actors > 1` | `JudgerPool` | 多个 Ray actor 副本并发打分 |

`ComposedJudgerConfig` 用于多分支场景：一个样本可以按 `select_fn` 路由到不同 Judger，也可以同时运行多个 Judger，再用 `merge_fn` 合并结果。

## 输入输出约定

使用预置 `SingleTurnAgentLoop` 和 `RLTrainer._prepare_train_data()` 时，Judger 的输入输出约定要放在整条数据链路里理解：

```text
Sampler -> RolloutState
  -> SingleTurnAgentLoop.generate_group()
  -> RolloutController.generate()
  -> Judger.judge()
  -> ReplayBuffer
  -> RLTrainer._prepare_train_data()
```

Judger 本身统一暴露异步接口：

```python
async def judge(self, rollout_state: RolloutState) -> RolloutState:
    ...
```

接口类型上也允许批量输入，但内置 `NativeJudger` 默认按单条样本调用 `reward_handler`。只有自定义 Judger 明确支持 `list[RolloutState]` 时，才应打开 `SingleTurnAgentLoopConfig(enable_batch_judge=True)`。

### Judger 输入

在预置 `SingleTurnAgentLoop` 中，进入 Judger 前的 `RolloutState` 通常已经包含：

- `prompt_ids`：prompt token，后续 `_prepare_train_data()` 会用它拼接训练输入。
- `response_ids`：模型生成的 response token，后续训练数据直接依赖它。
- `response`：模型生成的文本，`NativeJudger` 会用它计算 reward。
- `logprobs`：response token 的 rollout logprob。如果存在，长度必须和 `response_ids` 一致。
- `response_mask`：哪些 response token 参与训练。为空时，`_prepare_train_data()` 默认所有 response token 都参与训练。
- `reward_model["ground_truth"]`：标准答案或标签，预置 rule-based Judger 通常依赖这个字段。

`NativeJudger` 传给 `reward_handler` 的字段更窄：

```python
input_kwargs = {
    "response": rollout_state.response,
    "label": rollout_state.reward_model["ground_truth"],
    "extra_info": {**self.extra_info},
}
```

因此，如果只自定义 `reward_handler`，只能直接拿到 `response`、`label` 和 `extra_info`；如果需要读取 `response_ids`、`extra_fields`、工具轨迹或状态码，应继承 `Judger`。

### Judger 输出

Judger 的输出仍然是同一个 `RolloutState`，必须把分数写到 `rollout_state.reward` 这个 dict 中：

```python
rollout_state.reward = {
    "score": 1.0,
    "acc": True,
}
```

如果这批数据后续要进入 `_prepare_train_data()`，必须满足：

- `reward` 不能为 `None`。
- `reward` 必须包含数值型 `score` 字段，因为 `_prepare_train_data()` 会直接读取 `data.reward["score"]` 来计算 advantage。
- `status` 不能是 `ABORTED`、`FILTERED` 或 `FAILED`。
- `response` 和 `response_ids` 都不能为空。
- 如果提供 `response_mask`，长度必须和 `response_ids` 一致。mask 为 `0` 的 token 会在训练 label 中变成 `-100`，对应 advantage 也会置为 `0.0`。
- 如果提供 `logprobs`，长度必须和 `response_ids` 一致。

其他 reward 字段可以按任务需要扩展，例如 `acc`、`format`、`tool_ok`、`reason` 等。

如果使用 `ComposedJudger` 的默认 `merge_fn`，输出不会自动生成总分，而是按 branch 名称保存各分支分数：

```python
rollout_state.reward = {
    "correctness": 1.0,
    "format": 0.5,
}
```

如果训练或评测需要 `reward["score"]`，需要提供自定义 `merge_fn`。

## select_fn 与 merge_fn

`ComposedJudgerConfig` 的核心是 `branches`、`select_fn` 和 `merge_fn`：

```python
from xtuner.v1.rl.judger import ComposedJudgerConfig, JudgerConfig

judger_config = ComposedJudgerConfig(
    branches={
        "gsm8k": JudgerConfig(judger_name="gsm8k", reward_handler=gsm8k_reward),
        "format": JudgerConfig(judger_name="format", reward_handler=format_reward),
    },
    select_fn=lambda state, branches: ["gsm8k", "format"],
    merge_fn=merge_rewards,
)
```

`select_fn` 负责选择要运行哪些分支：

```python
def select_fn(state: RolloutState, branches: dict[str, Judger]) -> str | list[str] | None:
    ...
```

返回值含义：

- `str`：运行一个 branch。
- `list[str]`：运行多个 branch。
- `None`：走 `default_key` 或单分支 fallback。

默认 `select_fn` 会用 `rollout_state.data_source` 匹配 `branches` 的 key；匹配不到时返回 `None`。

`merge_fn` 负责把多个子 Judger 的输出合并回一个 `RolloutState`：

```python
def merge_fn(
    original: RolloutState | list[RolloutState],
    judged: dict[str, RolloutState | list[RolloutState]],
) -> RolloutState | list[RolloutState]:
    ...
```

当前实现没有独立的 `weight_fn` 配置字段。若需要加权，可以在 `merge_fn` 中实现，或者让 `merge_fn` 调用用户自己定义的 `weight_fn` 辅助函数：

```python
def weight_fn(branch_name: str) -> float:
    return {
        "correctness": 0.9,
        "format": 0.1,
    }[branch_name]

def merge_rewards(original, judged):
    merged = original.model_copy(deep=True)
    branch_scores = {
        name: state.reward["score"]
        for name, state in judged.items()
    }
    merged.reward = {
        "score": sum(weight_fn(name) * score for name, score in branch_scores.items()),
        **branch_scores,
    }
    return merged
```

批量输入时，`merge_fn` 也需要返回同样长度的 `list[RolloutState]`。默认 `merge_fn` 已支持单条和批量输入，但只会生成 `{branch_name: score}`，不会额外生成 `score` 总分。

## 自定义 Judger

自定义 Judger 有三种常见方式。

### 方式一：自定义 reward_handler

如果打分逻辑只依赖 `response`、`ground_truth` 和静态配置，推荐直接使用 `JudgerConfig(reward_handler=...)`：

```python
from xtuner.v1.rl.judger import JudgerConfig

def exact_match_reward(response, label, extra_info):
    pred = response.strip()
    gt = str(label).strip()
    score = 1.0 if pred == gt else 0.0
    return {
        "score": score,
        "acc": score > 0,
    }

judger_config = JudgerConfig(
    judger_name="exact_match",
    reward_handler=exact_match_reward,
    extra_info={},
    num_ray_actors=0,
)
```

`reward_handler` 可以是普通函数，也可以是 async 函数。返回值必须是 `dict`。

### 方式二：HTTP Judger 服务

如果打分逻辑在外部服务中，可以把 `reward_handler` 配成 URL：

```python
judger_config = JudgerConfig(
    judger_name="remote_reward",
    reward_handler="http://127.0.0.1:8000/reward",
    request_timeout=30.0,
)
```

服务端需要接收：

```json
{
  "response": "...",
  "label": "...",
  "extra_info": {}
}
```

并返回 JSON object：

```json
{
  "score": 1.0,
  "acc": true
}
```

### 方式三：继承 Judger

如果打分需要读取 `RolloutState.extra_fields`、工具调用轨迹、多轮状态、状态码或其他字段，可以直接继承 `Judger`：

```python
from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.judger import Judger

class ToolJudger(Judger):
    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        tool_ok = rollout_state.extra_fields.get("tool_ok", False)
        answer = (rollout_state.response or "").strip()
        label = rollout_state.reward_model["ground_truth"]
        rollout_state.reward = {
            "score": 1.0 if tool_ok and answer == label else 0.0,
            "tool_ok": tool_ok,
        }
        return rollout_state
```

如果希望它通过配置系统构建，并继续复用 Ray actor 扩展能力，可以继承 `JudgerConfig` 并覆盖 `build_local()`：

```python
from xtuner.v1.rl.judger import JudgerConfig

class ToolJudgerConfig(JudgerConfig):
    judger_name: str = "tool_judger"

    def build_local(self) -> ToolJudger:
        return ToolJudger()
```

之后仍可通过 `num_ray_actors` 控制本地或 Ray actor 模式：

```python
judger_config = ToolJudgerConfig(num_ray_actors=4)
judger = judger_config.build()
```

## 在训练配置中使用

在 V1 RL 配置中，Judger 通常挂到 `TaskSpecConfig.judger_config`：

```python
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import TaskSpecConfig
from xtuner.v1.rl.judger import GSM8KJudgerConfig

judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k", num_ray_actors=1)

task_config = TaskSpecConfig(
    task_name="train_task",
    agent_loop_config=SingleTurnAgentLoopConfig(
        hf_checkpoint=model_path,
        sample_params=training_sample_params,
    ),
    judger_config=judger_config,
    sampler_config=sampler_config,
)
```

`AgentLoopManagerConfig.build()` 会调用 `build_judger(task_cfg.judger_config)`。普通 `JudgerConfig` 会按 `num_ray_actors` 构建本地或 Ray 版本；`ComposedJudgerConfig` 会递归构建各个 branch。

## 预置 Judger

XTuner 预置了一些常用 Judger 配置，可直接在训练配置中使用：

- `GSM8KJudgerConfig`
- `DapoMathJudgerConfig`
- `GEO3KJudgerConfig`
- `CompassVerifierV2Config`

这些预置 Judger 都遵循上面的 `RolloutState -> reward dict` 约定；具体任务逻辑可以直接查看对应实现文件。

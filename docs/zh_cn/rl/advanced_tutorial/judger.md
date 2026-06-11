# Judger

`Judger` 是 XTuner RL 中负责给 rollout 结果打分的组件。它接收 `RolloutState`，读取模型生成结果和标签，计算 reward，并把分数写回 `rollout_state.reward`。后续的 advantage 计算、训练数据构建和默认评测逻辑都会依赖这个字段。

在典型流程中，`AgentLoop` 先调用推理引擎生成 `response_ids`，再把 token 解码成文本形式的 `response`，最后通过 `run_judger()` 调用 `judger.judge(rollout_state)` 或 `judger.batch_judge(rollout_states)`。

## Judger 类型

`xtuner/v1/rl/judger/native.py` 中定义了 Judger 的基础类型。可以先把它理解成三层：第一层是 `BaseJudger`，只定义所有 Judger 最小统一的 `judge()` 和 `batch_judge()` 接口；第二层是 `Judger`，在 `BaseJudger` 之上引入 `preprocess -> judge_payload -> postprocess` 的可组合 payload 契约；第三层是不同运行形态，决定打分逻辑在本地执行、在 Ray actor 中执行，还是组合多个子 Judger 一起执行。

```text
                                  ┌─────────────────────────────┐
                                  │        BaseJudger            │
                                  │  async judge(RolloutState)   │
                                  │  async batch_judge(list)      │
                                  └──────────────┬──────────────┘
                                                 │
                                  ┌──────────────▼──────────────┐
                                  │           Judger             │
                                  │ preprocess / judge_payload   │
                                  │        / postprocess         │
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
                 │ Judger branch  │       │ Judger branch  │       │ Judger branch  │
                 └───────┬───────┘       └───────┬───────┘       └───────┬───────┘
                         │                       │                       │
                         └───────────────────────┼───────────────────────┘
                                                 ▼
                         data_source 选择 branch，merge_fn 合并 reward dict
```

`Judger` 的 `preprocess` 和 `postprocess` 不是额外包装，而是重要的接口边界：`preprocess` 负责从 `RolloutState` 中提取评分需要的最小 payload，`judge_payload` 只处理这份轻量 payload，`postprocess` 再把输出写回原始 `RolloutState`。这样可以避免在组合打分或远程打分时对完整 `RolloutState` 做 `deepcopy`，减少大字段序列化开销，也避免由复制复杂对象带来的潜在对象生命周期风险。

`RemoteJudger` 的作用是给 Ray actor 形式的 Judger 提供一个对外统一接口。它本身不实现具体打分逻辑，而是持有 `JudgerActor` 的 actor proxy。`judge()` 或 `batch_judge()` 会先在 driver 侧把 `RolloutState` 转成轻量 payload，再调用 `actor.judge_payload.remote(...)`，因此 Ray 序列化的是 payload，而不是完整 `RolloutState`。这样上层代码只需要面向 `Judger` 接口编程，不需要关心当前 Judger 是本地对象、Ray actor，还是多副本池中的一个远程副本。

构建时有两条入口：

```text
JudgerConfig
  ├─ cpu_resources = None  ──► NativeJudger
  └─ cpu_resources = CPUResourcesConfig(...)
       ├─ num_workers = 1  ──► RemoteJudger
       │                        └─ JudgerActor
       │                             └─ NativeJudger
       └─ num_workers > 1  ──► JudgerPool
                               ├─ RemoteJudger -> JudgerActor -> NativeJudger
                               ├─ RemoteJudger -> JudgerActor -> NativeJudger
                               └─ ...

ComposedJudgerConfig
  └─ ComposedJudger
       ├─ branches["a"] -> JudgerConfig 或 ComposedJudgerConfig
       ├─ branches["b"] -> JudgerConfig 或 ComposedJudgerConfig
       └─ RolloutState.data_source + merge_fn 控制路由和合并
```

普通 `JudgerConfig` 根据 `cpu_resources` 决定执行模式。`cpu_resources` 表示 PG 外 Ray CPU worker 的资源需求，类型为 `CPUResourcesConfig`：

| 配置 | 构建结果 | 含义 |
| --- | --- | --- |
| `cpu_resources = None` | `NativeJudger` | 本地执行，不启动 Ray Judger actor |
| `cpu_resources=CPUResourcesConfig(num_workers=1, ...)` | `RemoteJudger -> JudgerActor -> NativeJudger` | 单个 Ray actor 执行打分 |
| `cpu_resources=CPUResourcesConfig(num_workers>1, ...)` | `JudgerPool` | 多个 Ray actor 副本并发打分 |

`CPUResourcesConfig.cpu_memory_per_worker` 默认是 `1024**3`，通常不需要额外配置。PG 外 CPU actor 资源会注册到全局 `CPUResourceManager`，资源不足时会在组件构建阶段报错，避免 Ray actor 长时间 pending。

`ComposedJudgerConfig` 用于多分支场景：样本通过 `RolloutState.data_source` 路由到一个或多个 Judger。`data_source` 为字符串时选择一个 branch；为字典时使用字典 key 同时选择多个 branch，并用 `merge_fn` 合并结果。`ComposedJudger` 只支持继承 `Judger` 的 branch，不支持只继承 `BaseJudger` 的 branch；因为它内部通过 `preprocess -> judge_payload -> postprocess` 组合多个分支，并且已经移除了对 `RolloutState` 的 `deepcopy` 逻辑。如果在没有 `deepcopy` 的情况下直接支持任意 `BaseJudger.judge()`，多个分支可能会同时读写同一个 `RolloutState`，存在状态互相覆盖或其他副作用风险。

## 输入输出约定

使用预置 `SingleTurnAgentLoop` 和 `RLTrainer._prepare_train_data()` 时，Judger 的输入输出约定要放在整条数据链路里理解：

```text
Sampler -> RolloutState
  -> SingleTurnAgentLoop.generate_group()
  -> RolloutController.generate()
  -> Judger.judge() / Judger.batch_judge()
  -> ReplayBuffer
  -> RLTrainer._prepare_train_data()
```

Judger 本身统一暴露异步接口：

```python
async def judge(self, rollout_state: RolloutState) -> RolloutState:
    ...

async def batch_judge(self, rollout_states: list[RolloutState]) -> list[RolloutState]:
    ...
```

`judge()` 只处理单条样本；批量打分使用 `batch_judge()`。内置 `NativeJudger` 和 `CompassVerifierV2` 默认不支持批量打分，会在 `batch_judge()` 入口直接报错。只有当前 Judger 明确实现 `batch_judge()` 时，才应打开 `SingleTurnAgentLoopConfig(enable_batch_judge=True)`。

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

如果使用 `ComposedJudger` 且 `data_source` 只选择一个 branch，输出会直接使用该 branch 的 reward。若 `data_source` 选择多个 branch，必须提供自定义 `merge_fn`，因为通用逻辑无法知道业务上应该如何聚合多个分数。例如：

```python
rollout_state.reward = {
    "correctness": 1.0,
    "format": 0.5,
}
```

如果训练或评测需要 `reward["score"]`，`merge_fn` 必须生成这个字段。

## data_source 与 merge_fn

`ComposedJudgerConfig` 的核心是 `branches` 和 `merge_fn`，路由固定读取 `RolloutState.data_source`：

```python
from xtuner.v1.rl.judger import ComposedJudgerConfig, JudgerConfig

judger_config = ComposedJudgerConfig(
    branches={
        "gsm8k": JudgerConfig(judger_name="gsm8k", reward_handler=gsm8k_reward),
        "format": JudgerConfig(judger_name="format", reward_handler=format_reward),
    },
    merge_fn=merge_rewards,
)
```

`data_source` 的含义：

- `str`：必须等于某个 branch 名称，运行这一个 branch。
- `dict`：key 必须都是 branch 名称，运行这些 branches。
- `None`、空 dict、未知 branch、非字符串 key 或其他类型都会在 `ComposedJudger` 入口报错。

`merge_fn` 负责把多个子 Judger 的输出合并回一个 `RolloutState`：

```python
from xtuner.v1.rl.judger import JudgerOutput

def merge_fn(
    original: RolloutState | list[RolloutState],
    judged: dict[str, JudgerOutput | list[JudgerOutput]],
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
    branch_scores = {
        name: output["score"]
        for name, output in judged.items()
    }
    original.reward = {
        "score": sum(weight_fn(name) * score for name, score in branch_scores.items()),
        **branch_scores,
    }
    return original
```

批量输入时，`merge_fn` 的第一个参数是 `list[RolloutState]`，`judged` 中每个 branch 的值是同样长度的 `list[JudgerOutput]`，返回值也必须是同样长度的 `list[RolloutState]`。框架不提供默认 merge；多 branch 场景必须显式传入 `merge_fn`。

## 自定义 Judger

自定义 Judger 有四种常见方式。

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

如果打分需要读取 `RolloutState.extra_fields`、工具调用轨迹、多轮状态、状态码或其他字段，推荐继承 `Judger`，并按需重载 `preprocess`、`judge_payload` 和 `postprocess`。这比直接重写 `judge()` 更适合 XTuner 的运行时：

- `preprocess` 可以只从 `RolloutState` 里提取评分需要的字段，避免把完整 `RolloutState` 传给远程 actor。
- `judge_payload` 只处理轻量 payload，方便本地、Ray actor 和多副本池复用同一套逻辑。
- `postprocess` 统一把结果写回原始 `RolloutState`，不用对 `RolloutState` 做 `deepcopy`。
- `ComposedJudger` 可以安全组合这种 Judger branch，并在多分支场景中拿到每个 branch 的 raw output 交给 `merge_fn`。

```python
from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.judger import Judger, JudgerOutput, JudgerPayload, JudgerPayloadBatch

class ToolJudger(Judger):
    def preprocess(self, rollout_state: RolloutState) -> JudgerPayload:
        return {
            "response": rollout_state.response,
            "label": rollout_state.reward_model["ground_truth"],
            "tool_ok": rollout_state.extra_fields.get("tool_ok", False),
        }

    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutput:
        assert not isinstance(payload, list)
        tool_ok = payload["tool_ok"]
        answer = (payload["response"] or "").strip()
        label = payload["label"]
        return {
            "score": 1.0 if tool_ok and answer == label else 0.0,
            "tool_ok": tool_ok,
        }

    def postprocess(self, rollout_state: RolloutState, output: JudgerOutput) -> RolloutState:
        rollout_state.reward = {
            "score": output["score"],
            "tool_ok": output["tool_ok"],
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

之后仍可通过 `cpu_resources` 控制本地或 Ray actor 模式：

```python
from xtuner.v1.rl.utils import CPUResourcesConfig

judger_config = ToolJudgerConfig(
    cpu_resources=CPUResourcesConfig(
        num_workers=4,
        num_cpus_per_worker=1,
    ),
)
judger = judger_config.build()
```

### 方式四：继承 BaseJudger

如果你的打分逻辑确实无法拆成 `preprocess -> judge_payload -> postprocess`，可以考虑直接继承 `BaseJudger` 并完整实现 `judge()` / `batch_judge()`。这通常只适合非常特殊的场景，例如打分过程必须整体接管 `RolloutState` 的状态转换，或者依赖无法序列化成轻量 payload 的本地对象。

需要注意：只继承 `BaseJudger` 的实现不能作为 `ComposedJudger` 的 branch。`ComposedJudger` 为了避免 `deepcopy(RolloutState)` 带来的潜在风险和额外序列化开销，内部只通过 `Judger` 的 payload 契约调用 branch。如果直接支持任意 `BaseJudger.judge()` 且不做 `deepcopy`，多个 branch 在同一个 `RolloutState` 上并发运行时可能互相覆盖 `reward` 或修改其他字段，导致结果不稳定。因此，如果后续需要接入 `ComposedJudger`、`RemoteJudger` 或 `JudgerPool`，应优先继承 `Judger` 并重载 `preprocess` / `judge_payload` / `postprocess`。

## 在训练配置中使用

在 V1 RL 配置中，Judger 通常挂到 `TaskSpecConfig.judger_config`：

```python
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import TaskSpecConfig
from xtuner.v1.rl.judger import GSM8KJudgerConfig
from xtuner.v1.rl.utils import CPUResourcesConfig

judger_config = GSM8KJudgerConfig(
    judger_name="openai/gsm8k",
    cpu_resources=CPUResourcesConfig(
        num_workers=1,
        num_cpus_per_worker=1,
    ),
)

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

`AgentLoopManagerConfig.build()` 会调用 `build_judger(task_cfg.judger_config)`。普通 `JudgerConfig` 会按 `cpu_resources` 构建本地或 Ray 版本；`ComposedJudgerConfig` 会递归构建各个 branch。

## 预置 Judger

XTuner 预置了一些常用 Judger 配置，可直接在训练配置中使用：

- `GSM8KJudgerConfig`
- `DapoMathJudgerConfig`
- `GEO3KJudgerConfig`
- `CompassVerifierV2Config`

这些预置 Judger 都遵循上面的 `RolloutState -> reward dict` 约定；具体任务逻辑可以直接查看对应实现文件。

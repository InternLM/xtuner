# AgentLoop

`AgentLoop` 是 XTuner RL 中最常需要按任务自定义的模块。它定义“一组样本如何完成 rollout”：如何组织推理输入、调用几次推理引擎、是否插入工具或环境反馈、哪些 token 参与训练，以及什么时候调用 Judger 写入 reward。

在默认训练链路中，AgentLoop 位于 sampler 和 replay buffer 之间：

```text
Sampler -> list[RolloutState]
  -> AgentLoop.generate_group()
  -> RolloutController.generate()
  -> Judger.judge()
  -> ReplayBuffer
  -> RLTrainer._prepare_train_data()
```

如果你的任务只是单轮问答，通常直接使用预置的 `SingleTurnAgentLoop`。如果任务包含多轮交互、工具调用、环境 step、特殊终止条件、非模型 token 插入或自定义 response mask，就应该自定义 AgentLoop。

## 类型与构建

`xtuner/v1/rl/agent_loop/agent_loop.py` 中有两个核心抽象：

- `AgentLoopConfig`：配置对象，负责构建本地或 Ray actor 形式的 AgentLoop。
- `AgentLoop`：运行时对象，负责实现 `generate_sample()` 和 `generate_group()`。

整体关系如下：

```text
	                           ┌─────────────────────────────┐
	                           │       AgentLoopConfig        │
	                           │ hf_checkpoint, sample_params │
	                           │ external_cpu                 │
	                           └──────────────┬──────────────┘
	                                          │ build(...)
	             ┌────────────────────────────┼────────────────────────────┐
	             │                            │                            │
	    external_cpu = None       external_cpu.num_workers = 1  external_cpu.num_workers > 1
             │                            │                            │
             ▼                            ▼                            ▼
   ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
   │   AgentLoop     │          │ AgentLoopActor  │          │ RouterAgentLoop │
   │ 本地执行         │          │ Ray actor        │          │ 多 actor 路由     │
   └─────────────────┘          └────────┬────────┘          └────────┬────────┘
                                          │                            │
                                          ▼                            ▼
                                  ┌─────────────────┐        ┌─────────────────┐
                                  │   AgentLoop     │        │ AgentLoopActor  │
                                  │ actor 内部实例   │        │ ...             │
                                  └─────────────────┘        └─────────────────┘
```

`AgentLoop` 统一暴露两个异步接口：

```python
async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
    ...

async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
    ...
```

`generate_sample()` 处理单条样本；默认 `generate_group()` 会并发调用多次 `generate_sample()`，并在调用前把 `self.sample_params` 写到组内每条样本上。需要组级逻辑时，例如批量打分、组内共享环境、组内排序过滤，可以覆盖 `generate_group()`。

## 输入输出约定

AgentLoop 输入和输出都是 `RolloutState`。如果后续使用预置 `RLTrainer._prepare_train_data()`，自定义 AgentLoop 必须维护好训练所需字段。

### 输入

`Sampler` 传给 AgentLoop 的 `RolloutState` 通常包含：

- `message`：原始对话消息。
- `prompt_ids`：tokenized prompt，通常由 RL tokenize function 写入。
- `reward_model`：标签信息，例如 `{"ground_truth": ...}`，供 Judger 使用。
- `sample_params`：会在 `generate_group()` 中被 AgentLoop 的默认采样参数覆盖。
- `task_name`、`uid`、`session_uid` 等调度字段。

生成前，AgentLoop 需要确保：

- `rollout_state.tokens` 是实际传给 `RolloutController.generate()` 的输入 token。单轮任务通常设为 `prompt_ids`；多轮任务通常设为历史上下文拼接后的 token。
- `rollout_state.sample_params` 是本次推理使用的参数。多轮任务里每一轮可能需要更新 `max_tokens`。

### 输出

AgentLoop 返回的 `RolloutState` 如果要进入训练，至少需要满足：

- `status == Status.COMPLETED`。预置 trainer 会跳过 `ABORTED`、`FILTERED`、`FAILED` 的样本组。
- `response_ids` 非空。`_prepare_train_data()` 用它构造训练 token。
- `response` 非空。Judger 和轨迹保存依赖文本 response。
- `reward["score"]` 存在。`_prepare_train_data()` 会直接读取它计算 advantage。

如果提供以下字段，还需要满足长度约定：

- `logprobs`：长度必须等于 `len(response_ids)`。
- `response_mask`：长度必须等于 `len(response_ids)`。mask 为 `0` 的 token 会被转成训练 label `-100`，对应 advantage 也会置为 `0.0`。

这也是自定义 AgentLoop 最容易出错的地方：工具返回、环境反馈、系统插入内容等不是模型生成的 token，应该在 `response_mask` 中置为 `0`，并给对应 `logprobs` 填 `0.0`。

## SingleTurnAgentLoop

`SingleTurnAgentLoop` 是默认单轮问答实现，适用于“给定 prompt，模型生成一次 response，然后打分”的任务。

单条样本流程如下：

```text
generate_sample(state)
  -> PartialRolloutHandler.preprocess(state, enable_partial_rollout)
  -> 如果 state.tokens 为空，则 state.tokens = state.prompt_ids
  -> await rollout_ctl.generate.remote(state)
  -> PartialRolloutHandler.postprocess(state)
  -> 如果 state.status != COMPLETED，直接返回，不触发 Judger
  -> 如果配置了 judger，调用 judger.judge(state)
```

典型配置：

```python
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig

agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=SampleParams(
        max_tokens=1024,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        min_tokens=0,
    ),
)
```

`SingleTurnAgentLoop` 还支持批量打分：

```python
agent_loop_config = SingleTurnAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=training_sample_params,
    enable_batch_judge=True,
)
```

开启后，`generate_sample()` 不会逐条调用 Judger；`generate_group()` 会在组内样本全部生成完成后调用一次 `judger.judge(group_samples)`。只有当前 Judger 明确支持 `list[RolloutState]` 输入时才应开启。

## 自定义 AgentLoop

自定义 AgentLoop 通常需要做四件事：

1. 继承 `AgentLoop`，实现 `generate_sample()`。
2. 在 `generate_sample()` 中维护 `tokens`、`sample_params`、`response_ids`、`response`、`logprobs`、`response_mask`、`status`。
3. 需要打分时，在 response 可用后调用 `self.judger.judge(...)`。
4. 继承 `AgentLoopConfig`，实现 `build_local()`，这样才能接入 `TaskSpecConfig.agent_loop_config`，并复用 Ray actor 构建逻辑。

### 最小单轮实现

下面是一个最小可用版本，行为接近 `SingleTurnAgentLoop`：

```python
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop import AgentLoop, AgentLoopConfig
from xtuner.v1.rl.judger import Judger

class CustomAgentLoop(AgentLoop):
    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        if not rollout_state.tokens:
            rollout_state.tokens = rollout_state.prompt_ids

        rollout_state.sample_params = rollout_state.sample_params or self.sample_params
        rollout_state = await self.rollout_ctl.generate.remote(rollout_state)

        if rollout_state.status != Status.COMPLETED:
            return rollout_state

        if self.judger is not None:
            rollout_state = await self.judger.judge(rollout_state)
        return rollout_state


class CustomAgentLoopConfig(AgentLoopConfig):
    def build_local(self, rollout_controller, judger: Judger | None = None, logger=None) -> CustomAgentLoop:
        return CustomAgentLoop(
            rollout_ctl=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
        )
```

这个版本适合没有工具、没有环境反馈、没有多轮上下文拼接的任务。只要 `RolloutController.generate()` 能正确写入 `response_ids`、`response`、`logprobs` 和 `status`，后续 Judger 与训练数据准备就能复用默认链路。

### 多轮或工具调用实现

多轮任务通常需要循环调用 `rollout_ctl.generate()`，每轮把上轮输出和工具或环境结果追加到下一轮输入。可以参考 `GSM8KToolAgentLoop` 的模式：

```python
class ToolAgentLoop(AgentLoop):
    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        final_response_ids: list[int] = []
        final_logprobs: list[float] = []
        final_response_mask: list[int] = []

        cur_tokens = list(rollout_state.tokens or rollout_state.prompt_ids or [])
        remaining_tokens = self.sample_params.max_tokens

        for _ in range(self.max_turns):
            rollout_state.tokens = cur_tokens
            rollout_state.sample_params = self.sample_params.model_copy(
                update={"max_tokens": remaining_tokens}
            )

            rollout_state = await self.rollout_ctl.generate.remote(rollout_state)
            if rollout_state.status != Status.COMPLETED:
                break

            response_ids = list(rollout_state.response_ids or [])
            logprobs = list(rollout_state.logprobs or [])
            assert len(response_ids) == len(logprobs)

            final_response_ids.extend(response_ids)
            final_logprobs.extend(logprobs)
            final_response_mask.extend([1] * len(response_ids))
            cur_tokens.extend(response_ids)

            tool_tokens = self._run_tool_and_encode_result(rollout_state)
            if not tool_tokens:
                break

            final_response_ids.extend(tool_tokens)
            final_logprobs.extend([0.0] * len(tool_tokens))
            final_response_mask.extend([0] * len(tool_tokens))
            cur_tokens.extend(tool_tokens)

            remaining_tokens = self.sample_params.max_tokens - len(final_response_ids)
            if remaining_tokens <= 0:
                break

        rollout_state.response_ids = final_response_ids[: self.sample_params.max_tokens]
        rollout_state.logprobs = final_logprobs[: self.sample_params.max_tokens]
        rollout_state.response_mask = final_response_mask[: self.sample_params.max_tokens]
        rollout_state.response = self.tokenizer.decode(rollout_state.response_ids)

        assert len(rollout_state.response_ids) == len(rollout_state.logprobs)
        assert len(rollout_state.response_ids) == len(rollout_state.response_mask)

        if rollout_state.status == Status.COMPLETED and self.judger is not None:
            rollout_state = await self.judger.judge(rollout_state)
        return rollout_state
```

这个例子强调两个约定：

- 模型生成 token 的 `response_mask` 为 `1`。
- 工具或环境插入 token 的 `response_mask` 为 `0`，`logprobs` 填 `0.0`。

### 覆盖 generate_group

默认 `generate_group()` 会并发处理组内样本。如果你的任务需要组级逻辑，可以覆盖它：

```python
async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
    samples = await super().generate_group(rollout_state, **kwargs)

    # 例：Judger 支持批量输入时，在组级统一打分。
    if self.judger is not None:
        samples = await self.judger.judge(samples)
    return samples
```

常见需要覆盖 `generate_group()` 的场景：

- Judger 需要一次性处理同一个 prompt 的多条 response。
- 组内样本共享同一个外部环境或缓存。
- 需要在组内做过滤、排序或重采样。
- 不希望组内样本并发执行。

### 支持 partial rollout

如果训练使用 `AsyncProduceStrategyConfig(enable_partial_rollout=True)`，producer 会把 `enable_partial_rollout` 作为运行时参数传给 `generate_group()`，再传入 `generate_sample()`。

自定义 AgentLoop 可以直接复用 `PartialRolloutHandler`：

```python
from xtuner.v1.rl.agent_loop.utils import PartialRolloutHandler

class CustomAgentLoop(AgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_rollout_handler = PartialRolloutHandler(
            max_tokens=self.sample_params.max_tokens
        )

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        enable_partial_rollout = kwargs.get("enable_partial_rollout", False)
        rollout_state = self.partial_rollout_handler.preprocess(
            rollout_state,
            enable_partial_rollout,
        )
        ...
        rollout_state = self.partial_rollout_handler.postprocess(rollout_state)
        return rollout_state
```

如果任务的多轮上下文、工具结果或环境状态不能用默认 handler 合并，需要自己定义续跑逻辑。核心原则是：续跑后的 `response_ids`、`response`、`logprobs`、`response_mask` 必须仍然表示完整 response，而不是只有本轮新增部分。

## 在训练配置中使用

训练配置中通常不手动实例化 AgentLoop，而是把 config 挂到 `TaskSpecConfig.agent_loop_config`：

```python
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)

agent_loop_config = CustomAgentLoopConfig(
    hf_checkpoint=model_path,
    sample_params=training_sample_params,
)

agent_loop_manager_cfg = AgentLoopManagerConfig(
    tasks=TaskSpecConfig(
        task_name="train_task",
        agent_loop_config=agent_loop_config,
        judger_config=judger_config,
        produce_strategy_config=SyncProduceStrategyConfig(),
        sampler_config=sampler_config,
    ),
)
```

`AgentLoopManagerConfig.build()` 会根据 `agent_loop_config` 构建 AgentLoop，根据 `judger_config` 构建 Judger，再把它们和 sampler、producer strategy 组装成 task runner。多任务训练时，每个 `TaskSpecConfig` 都可以使用不同的 AgentLoop。

## 自定义 Checklist

实现自定义 AgentLoop 时，建议逐项检查：

- `generate_sample()` 是否只处理单条 `RolloutState`。
- 推理前是否设置了 `rollout_state.tokens`。
- 每次调用 `rollout_ctl.generate.remote()` 前是否设置了本轮 `sample_params`。
- 返回训练前，`response_ids`、`response`、`logprobs`、`response_mask` 是否完整且长度一致。
- 非模型生成 token 是否在 `response_mask` 中置为 `0`。
- 需要 Judger 时，是否只对可评分的样本调用 `judger.judge()`。
- 若使用 `_prepare_train_data()`，是否保证最终有 `reward["score"]`。
- 若使用 async partial rollout，是否正确处理 `enable_partial_rollout` 和历史 response 合并。

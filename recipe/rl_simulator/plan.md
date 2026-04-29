# RL Simulator Recipe 计划

## 目标

在 `recipe/rl_simulator/` 下建设一套 RL 模拟验证 recipe，用于在不运行真实推理引擎和真实训练引擎的情况下，验证 XTuner RL 复杂控制流的正确性。

从 trainer 视角看，simulator 应该像正常 RL 训练任务一样运行：

- fake rollout engine
- fake training engine
- fake weight sync
- fake judge
- recipe 专用 `SimulationSampler`
- 继续走现有 `RLColocateTrainer`、`AgentLoopManager`、`SingleTurnAgentLoop`、replay buffer、sync/async producer 路径

正确性不在 simulator 运行时通过断言判断，而是在运行结束后由每个 case 自己的离线 log analyzer 判断。

第一版只做单机 CPU-only 模拟。Ray 可以继续作为现有 `.remote()` actor 接口的本地调度机制，但不模拟真实多机、多 rank、GPU/NPU 训练。

## 为什么放在 `recipe/`

这个 simulator 是实验性验证配方，不是稳定公共 API。

放在 `recipe/rl_simulator/` 下有几个好处：

- 可以随着代码版本快速迭代
- 允许每个 case 有自己的 `verify.py` 校验逻辑
- 对 `xtuner/v1/rl` 主路径改动最少
- 避免 fake/runtime 测试代码进入 RL 产品主接口

## 目录结构

```text
recipe/rl_simulator/
  README.md
  plan.md
  __init__.py

  core/
    __init__.py
    deterministic.py
    runner.py

  runtime/
    __init__.py
    sampler.py
    rollout.py
    trainer.py
    judger.py

  analyzer/
    __init__.py
    cli.py
    context.py
    helpers.py
    loaders.py
    result.py

  cases/
    __init__.py

    smoke_sync/
      __init__.py
      config.py
      verify.py

    async_partial/
      __init__.py
      config.py
      verify.py

    multitask_weighted/
      __init__.py
      config.py
      verify.py

    fault_filtering/
      __init__.py
      config.py
      verify.py

    tail_expired/
      __init__.py
      config.py
      verify.py

    async_update_every_4/
      __init__.py
      config.py
      verify.py

    determinism_replay/
      __init__.py
      config.py
      verify.py
```

### 文件职责说明

`core/` 放运行 simulator 所需的通用基础能力，不直接实现 fake RL 组件：

- `core/__init__.py`: 标记 Python package，并导出少量常用入口。
- `core/deterministic.py`: 提供确定性随机工具。所有 response length、token、delay、fault、reward、fake loss 都从 `seed + component + task_name + uid + attempt + repeat_index` 派生，禁止依赖全局随机状态。
- `core/runner.py`: simulator 运行入口。负责加载 case config、初始化 Ray、构建 trainer、调用 `trainer.fit()`，并可选在结束后调用 analyzer。

`runtime/` 放真正参与训练流程的 fake 组件：

- `runtime/__init__.py`: 导出 fake runtime 配置和组件，方便 case config 统一 import。
- `runtime/sampler.py`: 实现 `SimulationSamplerConfig` 和 `SimulationSampler`。它是第一版必选组件，用于稳定生成 uid、task_name、attempt 等 analyzer 需要的信息。
- `runtime/rollout.py`: 实现 `FakeRolloutConfig`、`FakeRolloutController`、`DelaySpec`、`FaultSpec`。`FakeRolloutConfig.build()` 直接返回本地 Ray actor 形式的 fake rollout controller，不实现真实 rollout worker。
- `runtime/trainer.py`: 实现 `FakeWorkerConfig`、`FakeTrainingController`、`FakeWeightSyncConfig`。`FakeWorkerConfig.build()` 直接返回本地 Ray actor 形式的 fake training controller，不再实现真实 `TrainingWorker` 或 fake model。
- `runtime/judger.py`: 实现 `FakeJudgerConfig` 和 `FakeJudger`。负责给 completed rollout 生成确定性随机 `0/1` reward。

`analyzer/` 放离线分析的公共框架，不写具体 case 的业务规则：

- `analyzer/__init__.py`: 导出 analyzer 公共 API。
- `analyzer/cli.py`: 离线分析命令入口。构造 `AnalyzerContext`，加载 case 目录下的 `verify.py`，调用 `verify(ctx)` 并写出结果。
- `analyzer/context.py`: 定义 `AnalyzerContext`，统一承载 case_dir、work_dir、从正常日志中提取出的 case 相关信息、tracker rows、train rollout 数据和原始 log 路径。
- `analyzer/helpers.py`: 提供可复用的日志分组、failure 构造和基础断言 helper。helper 不包含具体 case 策略。
- `analyzer/loaders.py`: 负责读取 `logs/rank_*.log`、`tracker.jsonl` 和 `train_rollout/*.jsonl`。其中关键 fake 日志从普通日志行里的 JSON 片段提取，使用 `json.loads()`；正则只用于非关键辅助信息。
- `analyzer/result.py`: 定义 `CheckResult`、failure 记录格式，以及把 summary/failures 写到 `logs/simulation_analysis_summary.json` 和 `logs/simulation_analysis_failures.jsonl` 的工具。

## 主代码最小改动

主代码尽量少改，主要做 duck typing 兼容。

### `xtuner/v1/train/rl_colocate_trainer.py`

放宽配置类型，让 recipe 里的 fake config 能传进来：

- `RLColocateTrainerConfig.train_worker_cfg`: 当前是 Pydantic 字段 `WorkerConfig`，需要改成 `Any` 或 `WorkerConfig | FakeWorkerConfig`。为了保持 recipe 独立，优先改成 `Any`，并在注释中说明要求实现 `build(placement_group)`。
- `RLColocateTrainerConfig.rollout_config`: 当前是 Pydantic 字段 `RolloutConfig`，需要改成 `Any` 或 `RolloutConfig | FakeRolloutConfig`。优先改成 `Any`，要求实现 `build(placement_group)`。
- `RLColocateTrainer.__init__` 的参数类型同步放宽，但运行逻辑不加 fake 分支。

注意：这里不能只改类型注解而不改 Pydantic 字段类型。`RLColocateTrainerConfig` 继承 `BaseModel`，字段类型如果仍是 `WorkerConfig` / `RolloutConfig`，Pydantic 会拒绝 recipe fake config，duck typing 不会自动生效。

当前 trainer 本身已经调用 `.build(...)`：

```python
self.train_controller = train_worker_cfg.build(self._pg)
self.rollout_controller = rollout_config.build(self._pg)
```

所以字段类型放宽后，运行逻辑可以基本不变。

### `xtuner/v1/rl/agent_loop/agent_loop_manager.py`

放宽 `TaskSpecConfig.sampler_config` 类型，让 recipe 里的 `SimulationSamplerConfig` 能通过 Pydantic 校验：

```python
sampler_config: Any
```

并要求它实现：

```python
build(tokenizer, replay_buffer)
```

不再改现有 `xtuner/v1/rl/agent_loop/sampler.py`。第一版直接使用 recipe 下的 `SimulationSampler`，避免为了 simulator 在主路径 sampler 中加入测试专用 uid 逻辑。

### Agent Loop 接入

第一版不新增 recipe 专用 agent loop，也不修改现有 `SingleTurnAgentLoop`。

继续直接使用现有 `SingleTurnAgentLoop`。`rollout_step` 仍由现有 producer/agent loop 链路内部使用，例如 partial rollout 的 `response_rollout_steps` 和 `seq_staleness` 后处理；fake rollout 不从 `RolloutState` 读取 `rollout_step`，也不把 `rollout_step` 放进确定性 key。

step 相关 correctness 通过正常日志和产物离线判断：

- aborted/completed 发生在哪个 rollout step。
- step 4 训练结束后 `weight_version` 是否变成 1。
- step 5 的 completed 样本是否使用 version 1。
- expired 样本的 `seq_staleness` 是否符合当前 step。

如果现有日志无法判断这些信息，再在 producer、trainer 或 agent loop 的事实发生位置补一条普通 logger 日志。不要为了提前传递 step 而新增 agent loop wrapper。

### Fake Judger

优先让 `FakeJudgerConfig` 继承现有 `JudgerConfig`，并 override `build_local()`，这样不需要改 `judger/factory.py`。

## Runtime 组件

### 模型与 Tokenizer 路径

不实现 `fake_hf.py`。case config 需要直接提供一个现成的 HF 模型/tokenizer 路径，例如集群上已有的 Qwen3。

这些路径只用于满足现有 trainer、tokenizer 和 dataset tokenize 初始化需求；fake rollout 不加载真实推理引擎，fake trainer 不加载真实训练模型。

### Dataset 与 Sampler

第一版所有 case 都使用 `SimulationSampler`。可以不依赖真实 dataset 内容；如果 case config 仍需要 dataset/tokenizer 占位，也可以共用同一套很小的 gsm8k/jsonl dataset config。

`SimulationSampler` 必须保持现有“被动采样”语义：producer 调用 `await sampler.sample(task_name=..., group_status=...)` 时，sampler 返回一个 `list[RolloutState]` group，而不是主动 push 到 replay buffer。

约定：

- `prompt_repeat_k` 由 `SimulationSampler` 自己实现。
- 对 fresh sample，`SimulationSampler` 直接构造最小合法 `RolloutState`，不需要真实 dataset 内容。
- 每个 fresh group 的 uid、message_uid、attempt 必须可复现；同一 group 内 `prompt_repeat_k` 个样本共享同一个 uid/message_uid，但可以用 `extra_fields["repeat_index"]` 区分。
- `SimulationSampler.sample(task_name=...)` 必须写入 `rollout_state.task_name = task_name` 和 `rollout_state.extra_fields["task_name"] = task_name`。
- 对 `group_status=Status.ABORTED` 或 `Status.EXPIRED`，sampler 继续优先从 replay buffer 对应状态池取已有 group；如果没有可复用 group，再退回 fresh sample。
- sampler 不直接判断 completed/failed 是否有效，也不决定是否进入训练；这些仍由 producer、replay buffer 和 trainer 的现有逻辑决定。

### Fake Rollout

实现 agent loop 和 trainer 需要的 rollout controller 外部方法：

- `generate`
- `pause_generation`
- `continue_generation`
- `offload`
- `onload`
- `onload_weights`
- `onload_kvcache`
- `get_rollout_metadata`
- `recover_failed_workers`
- `shutdown`
- `get_ready_status`

行为要求：

- response length 由 seed 和稳定 key 确定
- 如果用户设置 `max_tokens=512`，生成长度在 `[1, 512]`
- delay 由配置确定，并且可复现
- fault 行为由配置确定，并且可复现
- pause 可以产生带 partial response ids 的 `Status.ABORTED`
- v1 暂不支持 routed experts

`generate()` 不额外增加 `rollout_step` 参数，必须保持真实 rollout controller 的签名。

fake rollout 的确定性 key 不包含 `rollout_step`，而是使用：

```text
seed + component + task_name + uid + attempt + repeat_index
```

其中 `task_name`、`attempt`、`repeat_index` 由 `SimulationSampler` 写入 `RolloutState.task_name` 和 `extra_fields`。partial/aborted/expired 复用同一个 uid 时，uid 保持不变，attempt 递增。

Fake rollout controller 的方法签名和返回值必须与现有调用点对齐：

```python
class FakeRolloutController:
    async def generate(self, rollout_state: RolloutState) -> RolloutState: ...
    def pause_generation(self) -> None: ...
    def continue_generation(self) -> None: ...
    def offload(self) -> None: ...
    def onload(self) -> None: ...
    def onload_weights(self) -> None: ...
    def onload_kvcache(self) -> None: ...
    def recover_failed_workers(self) -> None: ...
    def shutdown(self) -> None: ...
    def get_ready_status(self) -> tuple[bool, dict[str, Any]]: ...
    def get_rollout_metadata(self) -> dict[str, Any]: ...
```

`generate()` 返回的 `RolloutState` 至少要填充这些字段：

- `status`: `Status.COMPLETED`、`Status.ABORTED` 或 `Status.FAILED`
- `response`: 非空字符串；aborted 时为 partial response
- `response_ids`: 与本次生成或 partial 结果一致的 token id list
- `logprobs`: 长度与 `response_ids` 一致
- `response_mask`: 长度与 `response_ids` 一致
- `finish_reason`: completed 时为 `stop` 或 `length`，aborted 时为 `abort`，failed 时为 `error`
- `error_msg`: failed 时必须非空

`get_rollout_metadata()` 返回的字段需要满足 `bind_train_rollout()` 和 fake trainer 消费：

```python
{
    "engine_rank_mesh_array": [[0]],
    "server_url_dict": {0: "fake://rollout/0"},
    "rollout_config": self.config,
    "worker_server_urls_status": {
        "fake://rollout/0": True,
    },
    "api_server_url": None,
}
```

`get_ready_status()` 返回：

```python
(True, {"active_workers": 1, "total_workers": 1})
```

这里保持字段名和真实 `RolloutController.get_rollout_metadata()` 一致，避免 trainer/weight sync 逻辑因为 metadata shape 不一致而失败。

`FaultSpec` 至少支持按样本精确注入故障：

```python
FaultSpec(
    task="train_task:main",
    uid=1005,
    attempt=1,
    status="failed",
    error="timeout",
)
```

匹配规则：非 `None` 字段全部相等才触发。`status` 第一版支持 `failed` 和 `aborted`，分别模拟推理失败和 partial/被暂停样本。第一版 fault injection 不按 rollout step 匹配；如果后续确实需要按 step 注入，再先确认日志和调用链能稳定提供 step。

### Fake Judge

返回确定性随机 reward：

```python
{"score": 0}
```

或：

```python
{"score": 1}
```

reward 必须能通过下面的 key 复现：

```text
seed + task_name + uid + attempt + repeat_index
```

### Fake Trainer

fake trainer 的边界是 training controller，不是 fake model。

保留 `FakeWorkerConfig` 的原因是 `RLColocateTrainer` 现有入口会调用：

```python
self.train_controller = train_worker_cfg.build(self._pg)
```

但 `FakeWorkerConfig.build()` 不创建真实 `TrainingWorker`，而是直接返回本地 Ray actor 形式的 `FakeTrainingController`。这样可以跳过真实 packing、actor logprobs、loss ctx、engine train step、optimizer step 和真实 weight update，只保留 trainer loop 需要的 controller 接口。

`FakeTrainingController` 实现 `RLColocateTrainer` 会调用的方法：

- `fit`
- `offload`
- `onload`
- `update_rollout_info`
- `update_weights`
- `save`
- `resume`
- `save_hf`
- `ready`

`fit()` 根据配置 sleep 指定时间，然后返回合法的 `list[WorkerLogItem]`，让现有 `_log_step()` 和 `_log_mini_batch_metrics()` 可以继续正常工作。

最小 `WorkerLogItem` 结构来自 `xtuner/v1/rl/trainer/worker.py`：

```python
{
    "train_entropy": 0.0,
    "train_metrics": [
        {
            "loss_log": {"loss": fake_loss},
            "rl_other_log": {
                "step_consumed_tokens": num_tokens,
                "efficient_attn_ratio": 1.0,
            },
        }
    ],
    "sft_train_metrics": {},
}
```

其中 `loss`、`num_tokens` 可以由 deterministic key 生成，或使用 case config 里的固定值。

### Fake Weight Sync

维护一个 fake `weight_version`。

对于延迟参数更新场景，支持类似这样的调度：

```python
weight_update_interval = 4
```

这样 8 step 的 case 可以验证：

- step 4 后更新一次
- step 1、2、3 不更新
- step 8 后再更新一次
- step 5、6、7 不更新
- rollout step 5 使用 version 1

### Fake Weight Sync 协调机制

`weight_version` 由 `FakeRolloutController` 持有，fake trainer 通过现有同步调用链更新它：

1. `RLColocateTrainer._sync_weights_and_save()` 调用 `bind_train_rollout(train_controller, rollout_controller)`。
2. `bind_train_rollout()` 通过 `rollout_controller.get_rollout_metadata()` 把 fake rollout metadata 传给 fake trainer。
3. fake trainer 的 `update_rollout_info(info_dict)` 保存 fake rollout actor handle 或 metadata 中的 fake endpoint 信息。
4. `train_controller.update_weights()` 被调用时，fake trainer 根据 `weight_update_interval` 判断当前 step 是否应该更新。
5. 如果应该更新，fake trainer 调用 fake rollout controller 的 `set_weight_version(version)` 或等价方法，把 rollout 侧版本推进。

对于 `async_update_every_4`：

- step 1-3: `train_update_weights` 不推进版本，rollout 仍是 version 0。
- step 4: step 4 的 rollout 先使用 version 0；step 4 train 结束后 sync，把 rollout version 推进到 1。
- step 5: rollout 读取到 version 1。
- step 8: step 8 train 结束后把 version 推进到 2。

如果为了少改 `RLColocateTrainer._sync_weights_and_save()`，也可以让 fake trainer 的 `update_weights()` 在非更新 step 直接 no-op 并打印普通日志说明 no-op；`verify.py` 负责确认只有 expected update step 出现 version 增长。

### Partial Rollout 与 Staleness 定义

v1 只按 rollout step 定义 staleness，不按 wall-clock time 定义。

语义沿用现有 `RolloutState.seq_staleness`：

- 每段 response token 都有对应的 `response_rollout_steps`。
- 当前 step 为 `cur_step` 时，`seq_staleness = cur_step - min(response_rollout_steps)`。
- 如果样本在 step 1 生成了一段 partial response 并被 abort，step 2 继续生成时，这段历史 token 的 staleness 为 `2 - 1 = 1`。
- `tail_batch_stale_threshold=N` 表示当 group 中存在 aborted/leftover sample 的 `seq_staleness >= N` 时，该 group 可以被标记为 `Status.EXPIRED`。
- expired sample 再次被采样时，partial 历史必须清空，重新从 prompt 开始生成。

`tail_expired` 的 `verify.py` 只检查 step-based staleness：

- `response_rollout_steps` 是否和生成 step 匹配。
- `seq_staleness` 是否等于 `cur_step - min(response_rollout_steps)`。
- 达到 threshold 的样本是否进入 expired/reset 路径。

## 日志与提取策略

不建设全局 event sink，也不要求 fake 组件打印特殊前缀日志。

simulator 按正常任务运行，优先复用现有日志和产物：

```text
logs/rank_*.log
logs/exp_tracking/tracker.jsonl
train_rollout/train_rollout_*.jsonl
```

每个 case 的 `verify.py` 可以按自己的需求解析正常日志。允许不同 case 使用不同提取逻辑，但关键 fake 事件第一版就使用 JSON 片段，避免字段顺序、空格或措辞变化导致校验逻辑脆弱。

如果某个 `verify.py` 无法从现有日志判断正确性，再在最接近事实发生的位置补充一条普通 logger 日志。新增日志应满足：

- 使用现有 logger 正常打印，不引入特殊前缀协议。
- 从第一版开始，关键 fake 日志使用“普通说明 + JSON 片段”的形式，verify 使用 `json.loads()` 解析 JSON 片段，不依赖字段顺序、空格或自然语言。
- 尽量包含 case 需要的 key 信息，例如 step、task、uid、attempt、status、reward、generated_tokens、weight_version。
- 只在信息不够时新增，不预先大面积埋点。

例如 fake rollout 的正常日志：

```text
Fake rollout done: {"step":1,"task":"train_task:main","uid":1001,"attempt":1,"status":"completed","generated_tokens":137,"max_tokens":512,"weight_version":0,"fault":null,"response_hash":"sha256:..."}
```

但是这仍然是普通日志，不定义 `[XTUNER_SIM]` 之类的全局特殊前缀。

第一版普通日志中的 JSON 片段建议统一包含 `event` 字段，便于公共 loader 提取成 `log_records`：

```text
Fake rollout done: {"event":"rollout_done","step":1,"task":"train_task:main","uid":1001,"attempt":1,"status":"completed","generated_tokens":137,"max_tokens":512,"weight_version":0,"fault":null,"response_hash":"sha256:..."}
```

这不是特殊前缀协议；它只是正常 logger 文本里附带的机器可读片段。

第一版预期 `verify.py` 可能需要从日志中提取的信息包括：

- sampler 实际采样的 step、task、uid/group、group_status、prompt_repeat_k
- rollout 完成时的 step、task、uid、attempt、status、generated_tokens、max_tokens、fault、weight_version
- fake judge 输出的 step、task、uid、attempt、reward
- fake trainer 完成的 step、num_groups、num_samples、loss
- fake weight sync 完成的 step、version_before、version_after

### 日志合并策略

第一版 simulator 是单机 CPU-only，但 Ray actor logger 仍可能产生多个 `logs/rank_*.log`。Analyzer 读取所有 `logs/rank_*.log`，并提取其中包含 JSON 片段的 fake runtime 日志。

约定：

- fake rollout、fake judge 可以在 actor 日志中打印，verify 必须按 `(event, step, task, uid, attempt)` 或 case 自己定义的 key 聚合。
- fake trainer 的 `train_fit_done` 和 `weight_update_done` 默认只由 `FakeTrainingController` 打印一次。
- 如果同一事件被重复打印，verify 必须显式去重，不能依赖文件读取顺序。
- 单机 fake 运行时可能只有 `rank_0.log`，也可能因为 Ray actor logger 行为产生多个 log；loader 不假设固定文件数量。
- loader 不按 wall-clock 合并日志；verify 用 step/task/uid/attempt/version 等业务字段判断。

### 确定性策略

即使第一版只做单机 CPU-only，Ray actor 调度顺序也不保证稳定，因此 determinism 不能依赖事件到达顺序。

约定：

- 每个随机结果只由稳定 key 决定，不能由“第几个 actor 执行”或全局随机序列决定。
- fake runtime 的随机 key 不包含 `rollout_step`，避免为了 simulator 修改 agent loop 或 producer。
- analyzer 比较 determinism 时使用 normalized digest，忽略日志顺序、wall-clock timestamp、pid、hostname 和 log 文件名。
- 如果某个 case 需要判断“哪个 uid 在第几步 completed / aborted”，verify 从 producer/agent loop 日志和 train rollout 产物中提取 step；如果信息不足，再补普通 logger 日志。
- 默认 verify 应比较集合或 multiset，而不是比较原始日志行顺序。

`determinism_replay` 只要求这些字段一致：

- `(task, uid, attempt, step)`
- `status`
- `generated_tokens`
- `response_hash`
- `reward`
- `fault`
- `weight_version`

不要求日志输出顺序完全一致。

## Analyzer 设计

每个 case 都有自己的 `verify.py`。期望值和校验逻辑放在同一个 Python 文件里，不再单独写 `expected.yaml`。

公共 analyzer 负责加载：

- `logs/rank_*.log`
- `logs/exp_tracking/tracker.jsonl`
- `train_rollout/train_rollout_*.jsonl`
- `logs/simulation_run_manifest.json`

公共 verify API：

```python
def verify(ctx: AnalyzerContext) -> CheckResult:
    ...
```

`AnalyzerContext`：

```python
@dataclass
class AnalyzerContext:
    case_dir: Path
    work_dir: Path
    log_records: list[dict]
    tracker_rows: list[dict]
    train_rollouts: dict[int, list[dict]]
    run_manifest: dict
    raw_log_paths: list[Path]
```

`CheckResult`：

```python
@dataclass
class CheckResult:
    case_name: str
    passed: bool
    summary: dict
    failures: list[dict]
```

Analyzer 输出：

```text
<exp_dir>/logs/simulation_analysis_summary.json
<exp_dir>/logs/simulation_analysis_failures.jsonl
```

## Case 扩展方式

这个框架的目标不是一次性写完所有 case，而是让后续新增 case 足够简单、可控、可 review。

新增一个 case 时，只需要新增一个目录：

```text
recipe/rl_simulator/cases/<case_name>/
  config.py
  verify.py
```

两份文件职责固定：

- `config.py`: 定义这个 case 怎么跑。包括 seed、rollout_steps、global_batch_size、fake rollout/fake trainer/fake judge/fake sync 配置、task 配置、producer 配置。
- `verify.py`: 定义这个 case 怎么验。文件开头直接写期望常量，下面写提取和校验逻辑。

公共 analyzer 不理解 case 的业务含义，只做三件事：

1. 加载日志和产物，构造 `AnalyzerContext`。
2. import case 目录下的 `verify.py`。
3. 调用 `verify(ctx)`，写出 summary/failures。

这样后续新增 case 时，不需要改公共 analyzer；只有当公共 loader 缺少某种通用产物读取能力时，才扩展 analyzer。

## Verify 编写约定

每个 `verify.py` 都应该保持结构清晰，建议按四段写：

```python
EXPECTED = {
    "case_name": "async_update_every_4",
    "update_steps": [4, 8],
    "forbidden_update_steps": [1, 2, 3, 5, 6, 7],
    "rollout_weight_version_by_step": {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
    },
}

def verify(ctx: AnalyzerContext) -> CheckResult:
    extracted = extract(ctx)
    failures = []

    failures.extend(check_weight_updates(extracted, EXPECTED))
    failures.extend(check_rollout_versions(extracted, EXPECTED))

    summary = build_summary(extracted, failures)
    return CheckResult(
        case_name=EXPECTED["case_name"],
        passed=not failures,
        summary=summary,
        failures=failures,
    )
```

推荐每个 `verify.py` 至少拆出这些函数：

- `extract(ctx)`: 只负责从 `ctx.log_records`、`ctx.tracker_rows`、`ctx.train_rollouts` 中提取当前 case 关心的信息，不做判断。
- `check_*(extracted, expected)`: 每个函数只检查一类规则，例如 weight sync、partial rollout、fault filtering、task allocation。这里的 `expected` 通常就是本文件顶部的 `EXPECTED`。
- `build_summary(extracted, failures)`: 输出人能读懂的结构化摘要，方便失败后 debug。

failure 记录建议统一成 dict：

```python
{
    "check": "weight_update_steps",
    "message": "unexpected weight update steps",
    "expected": [4, 8],
    "actual": [3, 8],
    "step": 3,
    "task": "train_task:main",
    "uid": 1007,
}
```

字段不要求完全固定，但建议包含：

- `check`: 失败的规则名
- `message`: 人可读说明
- `expected`: 期望值
- `actual`: 实际值
- `step/task/uid/attempt`: 如果能定位到具体样本，就尽量给出

## 公共 Verify Helper

为了避免每个 case 重复写基础逻辑，可以在 analyzer 下提供 helper，但不强制使用：

```text
recipe/rl_simulator/analyzer/helpers.py
```

初始 helper 可以包括：

- `group_by_step(records)`
- `group_by_task(records)`
- `find_log_records(records, pattern_or_predicate)`
- `load_train_rollout_items(ctx, step)`
- `assert_exact_steps(actual, expected, check_name)`
- `assert_no_forbidden_steps(actual, forbidden, check_name)`
- `build_failure(check, message, expected=None, actual=None, **loc)`

helper 只做通用数据处理和 failure 构造，不包含具体 case 策略。

## 推荐新增 Case 流程

1. 复制一个最接近的已有 case 目录。
2. 修改 `config.py`，先让 case 能跑完。
3. 修改 `verify.py` 文件顶部的 `EXPECTED`，写清楚这个 case 想证明什么。
4. 写 `verify.py` 的 `extract()`，先生成 summary，不急着加很多 check。
5. 看 summary 是否已经包含判断所需信息。
6. 如果信息不够，再在 fake 组件或少量核心路径补普通 logger 日志。
7. 补 `check_*()`，让 analyzer 能 pass/fail。
8. 故意改错一个 `EXPECTED`，确认 verify 能 fail，并且 failure 信息能定位问题。

## Verify 不应该做的事

- 不应该依赖 wall-clock 精确时间，除非 case 明确测试耗时，并设置容忍区间。
- 不应该解析高度易变的自然语言描述。
- 不应该把多个无关规则塞进一个巨大判断里。
- 不应该在 `verify.py` 里重新运行训练。
- 不应该修改 work_dir 下的原始日志和训练产物。

## 运行与 CI 集成

runner 必须优先支持单 case 运行和 debug，然后再支持批量 CI。

常用命令：

```text
# 跑单个 case，并在结束后分析
python -m recipe.rl_simulator.core.runner --case recipe/rl_simulator/cases/smoke_sync --analyze

# 跑单个 case，但不分析，方便直接看原始日志
python -m recipe.rl_simulator.core.runner --case recipe/rl_simulator/cases/async_partial

# 指定 work_dir，方便复现同一个失败 case
python -m recipe.rl_simulator.core.runner --case recipe/rl_simulator/cases/async_partial --work-dir /tmp/xtuner_sim_debug/async_partial --analyze

# 只分析已有 work_dir，不重新运行
python -m recipe.rl_simulator.core.runner --case recipe/rl_simulator/cases/async_partial --work-dir /tmp/xtuner_sim_debug/async_partial --analyze-only

# 跑所有 case，用于本地回归或 CI
python -m recipe.rl_simulator.core.runner --case-dir recipe/rl_simulator/cases --all --analyze
```

`core/runner.py` 负责：

1. 读取 case 目录下的 `config.py`。
2. 解析命令行覆盖项，例如 `--work-dir`、`--seed`、`--keep-work-dir`。
3. 写出 `logs/simulation_run_manifest.json`，记录 case、seed、work_dir、命令行参数和最终生效配置摘要。
4. 初始化 Ray。
5. 构建并运行 `trainer.fit()`。
6. 如果指定 `--analyze` 或 `--analyze-only`，调用 analyzer。
7. 根据 analyzer 的 `CheckResult.passed` 返回 exit code。

runner 参数约定：

- `--case <case_dir>`: 运行单个 case。
- `--case-dir <cases_dir> --all`: 顺序运行所有 case。
- `--work-dir <path>`: 指定输出目录；debug 时推荐固定这个路径。
- `--analyze`: 运行结束后立即分析。
- `--analyze-only`: 不运行训练，只分析已有 `--work-dir`。
- `--seed <int>`: 临时覆盖 case config 里的 seed，用于复现或缩小问题。
- `--keep-work-dir`: 如果 `work_dir` 已存在，不删除旧内容；用于保留中间产物和手动对比日志。

debug 约定：

- 每个 case 的 `config.py` 必须写默认 seed。
- runner 默认使用 case config 中的 seed，不做随机覆盖。
- 如果命令行传了 `--seed`，最终 seed 必须写入 `simulation_run_manifest.json`，analyzer summary 也要显示这个 seed。
- 单 case debug 时，推荐固定 `--work-dir`；失败后可以用同一条命令复现，也可以用 `--analyze-only` 反复调 verify。
- `--keep-work-dir` 只保证不删除旧目录；如果同名日志可能追加或混淆，runner 需要在 manifest 中记录本次 run id。

CI 可以先接入一个轻量脚本：

```text
recipe/rl_simulator/run_all.sh
```

脚本只顺序运行初始 cases，任何 case analyzer 失败都返回非零。

`determinism_replay` 需要特殊处理：

- runner 支持同一个 case 跑两次到不同 work_dir。
- runner 固定生成 `run_1/` 和 `run_2/` 两个 work_dir，并写出 manifest。
- verify 从 manifest 读取两次运行目录，不写死 `compare_work_dir`。
- verify 比较两次 normalized digest，忽略 timestamp、pid、hostname、log 文件名和日志行顺序。

manifest 路径和格式：

```text
<exp_root>/determinism_manifest.json
```

```json
{
  "case_name": "determinism_replay",
  "runs": [
    {"name": "run_1", "work_dir": ".../run_1"},
    {"name": "run_2", "work_dir": ".../run_2"}
  ]
}
```

第一阶段 CI 建议只跑 CPU-only fake cases，不依赖真实 GPU/NPU，不访问外网。

## 初始 Cases

### `smoke_sync`

目的：

- 单 task
- sync producer
- 无故障
- 每 step 都训练并同步权重

预期：

- 所有 rollout group 都完成
- 所有 group 都进入训练
- 每 step 都发生 weight update
- 每个 group 恰好包含 `prompt_repeat_k` 个样本
- reward 只可能是 `0/1`
- train rollout 中不包含 failed、aborted、expired、filtered 样本

### `async_partial`

目的：

- async producer
- oversampling
- partial rollout enabled

预期：

- 至少一个 `(task, uid)` 经历 `aborted -> completed`
- response length 从不超过 `max_tokens`
- 最终 train rollout 只包含 completed 且合法的样本

### `multitask_weighted`

目的：

- 两个 task 使用同一个 gsm8k dataset/sampler config
- task weight 控制 batch allocation

预期：

- 当 `global_batch_size=8` 且权重为 `1:3` 时，分配为 `2/6`
- 每个 group 有 `prompt_repeat_k` 个样本
- reward 只可能是 `0/1`

### `fault_filtering`

目的：

- 确定性 fault injection

预期：

- 配置指定的 `(task, uid, attempt)` 失败
- failed group 不进入 train rollout
- async 路径仍能补齐需要的 train batch

### `tail_expired`

目的：

- partial rollout staleness 和 expired 行为

预期：

- stale partial samples 会变成 expired
- expired samples 在重新生成前会清空历史 response
- 最终 response length 合法

### `async_update_every_4`

目的：

- 延迟参数更新

预期：

- 至少运行 8 step
- 只在 step 4 和 step 8 更新
- rollout step 1-4 使用 version 0
- rollout step 5-8 使用 version 1

### `determinism_replay`

目的：

- 可复现性

预期：

- 同一个 config 运行两次，产生相同的 deterministic digest
- 忽略 wall-clock timestamps
- 比较 uid、status、generated length、response hash、reward、fault、weight version

## 确定性

每个 case 必须提供 seed。

所有随机行为都由稳定 key 派生：

```text
seed + component + task_name + uid + attempt + repeat_index
```

适用于：

- response length
- response token ids/hash
- delay
- fault decision
- reward
- 如果 fake loss 使用随机，也必须适用

禁止依赖全局 `random`、`numpy` 或 `torch` 状态。

## V1 非目标

- 真实分布式训练、真实多机多 rank 行为
- MoE routed experts replay
- 真实 judge
- 真实 inference backend 分析
- 解析任意自由文本日志
- 在 `xtuner/v1/rl` 下提供稳定公共 API

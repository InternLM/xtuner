# XTuner PPO 接入设计

约定与阶段说明写在本文；接口与调用顺序见 [`PPO_design.py`](PPO_design.py) 伪代码。

## 1. 范围与目标

在现有 XTuner RL（rollout、`ReplayBuffer`、`AgentLoopManager`、`TrainingController`、
`TrainingWorker`）上接入 PPO / 多 role：状态不变量只维护一处，优先复用现有模块。

首版：text-only、同步、严格 on-policy。Critic/Frozen 与 Actor **同进程分时**（共卡与
分离均如此）。异步 PPO、独立 critic 资源池、多模态 value 不在本阶段。同套 Controller
覆盖 GRPO 与 OPD（含多 teacher / MOPD）。

文件落点：
- `head_type` / `fp32_head` → `xtuner/v1/model/base.py`、`xtuner/v1/config/fsdp.py`
- `TrainSample` → `xtuner/v1/rl/data_proto/train_sample.py`
- produce → `produce_utils.build_produce_batch_result`
- Worker/Controller → `xtuner/v1/rl/trainer/`
- advantage/loss → `xtuner/v1/rl/advantage/`、`xtuner/v1/rl/loss/`
- Trainer → `xtuner/v1/train/rl_trainer.py`（消费 `train_samples`，不再 `_prepare_train_data`）。

## 2. 数据协议

`RolloutState` 是算法无关的 rollout 协议，继续保存 `response_ids`、`response_mask`、
`logprobs`、`routed_experts`、`reward`、`status` 和 `seq_staleness`。

**GAE**（仅 PPO，`GAEAdvantageConfig` → `GAEEstimator`）：judger 的 sequence reward 映射为
逐步 \(r_t\)——写入该轨迹最后一个有效 response token，其余为 0；再由 \(V\) + GAE 回传。
`build_token_rewards` / `build_terminal_mask` 只在 `CriticWorker.fill_gae_token_fields`
（`fit` 内、scatter 之后）调用。GRPO / OPD 不构造 `token_rewards`。

**GAE 必须按原 sample/segment 分段算**：Value forward 可以 pack；`GAEEstimator.compute`
禁止作用在 pack concat 长序列上（信用会穿过 sample 边界）。实现上按 `cu_seqlens` /
pack 内 indices 把 \(V,r,m\) 切回各 sample，逐条 `compute`，再 pack 回 value loss。

同 session 多个 trainable segment **不拼成一条 MDP 轨迹**（中间常有 tool/环境洞，
后继 \(V\) 不是真实 \(s_{t+1}\)）。每段各自一条 GAE。**reward 两种都支持**
（`GAEAdvantageConfig.reward_scope`）：

* `"segment"`：每段用自己的 `sample.reward` 落到该段末 token。
* `"session"`：同 session 共用 representative 的 score，**每段仍单独 GAE**，只是末
  token 上的标量 \(r\) 相同。

与 GRPO 的 session 聚合不同：GRPO 是 session 级一个标量 adv 再广播；PPO 始终逐步、
按段回传。

训练阶段派生字段不由 produce / `build_train_sample` 写入：

| 字段 | 产生者 | 生命周期 |
| --- | --- | --- |
| `token_rewards`、`terminal_mask` | `CriticWorker.fill_gae_token_fields` | 一个 PPO batch |
| `old_values`、`bootstrap_value` | `CriticWorker.forward_only`（在 `fit` 内） | 一个 PPO batch |
| `advantages`、`returns` | `GAEEstimator.compute`（critic `fit` 内）或 GRPO 的 `train_grpo_batch` | 一个 batch |
| `old_logprobs` | `ActorWorker.forward_only` | 一个 PPO batch |
| `ref_logprobs` | `FrozenWorker.forward_only(mode="logprob")` | 一个 PPO batch |
| `current_values`、`current_logprobs` | 当前 worker | 每个 minibatch/epoch 临时值 |

GAE：

\[
\delta_t=r_t+\gamma m_tV(s_{t+1})-V(s_t),\qquad
A_t=\delta_t+\gamma\lambda m_tA_{t+1},\qquad R_t=A_t+V(s_t).
\]

正常结束时最后一个 `m_t=0`；截断轨迹使用 `bootstrap_value`。padding 和 prompt token
不参与 value/policy loss。

## 3. 角色与组件边界

进程模型（与 `PPO_design.py` 一致）：

* 三个 Worker 类：
  - `ActorWorker(SingleAcceleratorWorker)`：唯一 Ray remote；只做自己的
    `forward_only` / `fit` / `weight_update`
  - `CriticWorker` / `FrozenWorker`（普通类）：`attach` 进同进程 `_attached`
* Controller 直接调用：`self._groups["actor"].fit`、`["critic"].fit`、
  `["teacher"].forward_only`（或多 `teacher_*`）。`TrainingWorkerGroup(attached=True)`
  用 group name 路由 `call_attached`，**不加** Proxy 层。
* 各 group 共享同一批 ActorWorker Ray handles；数据先 `scatter_samples_and_meta`，
  再按 rank 发子集（不全量广播）。driver **不解包** values / logprobs tensor。
* `ActorWorker` 绑定训练侧 PlacementGroup（见第 8 节）。
* **`weight_update` / `bind_rollout` 只推 actor**；critic 不参与 rollout 权重同步。

职责：

* Actor：复用现有 clipped PG（`GRPOLossConfig` / `policy_loss_fn`）、reference/KL、
  SFT 混合（若启用）、R3 routed experts，以及向 rollout engine 同步权重。不新增
  `PPOLossConfig`；PPO 与 GRPO 的差别在 advantage（GAE），不在 actor policy loss。
  OPD 时 `fit` 消费 sample 上的 `teacher_*` 或 `teacher_targets`。
* Critic：对外统一 `fit(gae=...)`——内部
  `fill_gae_token_fields` → `forward_only` → `GAEEstimator.compute` → value clip 反传；
  返回 per-rank advantages 句柄。`ValueHead` + `loss_ctx`；不做 R3、不 sync rollout。
* Frozen：`forward_only(mode="logprob"|"sampled"|"topk")` → `FrozenWorkerResult`。
  reference KL 或 OPD teacher；Teacher 不做 R3。
* R3：Actor 复用 `TrainSample.routed_experts`（old/ref/current 同 batch 各 epoch）；
  Critic / Frozen **不做 R3**。
* Actor 不新增 `PPOLossConfig`：复用 `GRPOLossConfig` + clipped PG。Critic 用
  `CriticLossConfig` / `Kwargs` / `Context`（不挂 `policy_loss_cfg`）。
* **`TrainingController` 感知算法**：对外 **`fit`**（对齐现有 `TrainingController.fit`），
  对内转发 `train_grpo_batch` / `train_ppo_batch` / `train_opd_batch`；负责 PackMeta、
  scatter、同进程 `switch_role_modules(role)`（`actor` / `critic` / `reference` /
  `teacher[_*]` / `none`）。不实现 GAE/loss 数学细节。
  `ActorWorkerConfig.ref_load_from` 仅为过渡；PPO KL 走 `Frozen(name="reference")`。
* **`RLTrainer`**：建 PG、rollout produce、**rollout↔train 资源切换**、checkpoint、eval；
  调用 `train_controller.fit(produce_result.train_samples, rollout_idx=...)`；按 interval
  触发 `weight_update`。删除 `_prepare_train_data`。

配置：

* `TrainConfig` ≈ 现有 `train_worker_cfg`：`actor` / 可选 `critic` / `frozen` 列表 +
  `advantage_estimator_config`。
* `loss_cfg` 挂在各 WorkerConfig；`advantage_estimator_config` 注入 Controller。
* **`pack_max_length` 只在 `ActorWorkerConfig`**；Controller 持有并多 role 共用
  （对齐现有 `WorkerConfig.pack_max_length`）。

## 4. 训练入口与算法分发

`TrainingController.fit` 是 Trainer 唯一训练入口。入口统一：

1. 无效 group 已在 produce 侧过滤（对齐现有 `RLTrainer._prepare_train_data` 的有效性检查；
   删该函数后 `fit` 不再二次过滤）。
   现路径：`AgentLoop.maybe_filter_invalid_sample` 打 `FILTERED` →
   `put_generated_group` 不入 buffer → `take_batch(COMPLETED)`。
2. `build_session_meta`：按 `session_id` 建 segment→session（GRPO 用；PPO 不拼轨迹）。
3. 建 **本步唯一** `PackMeta`；GRPO/PPO/OPD 与多 role 一律复用，再 scatter。
4. 按配置转发：

```text
fit(sample_groups, rollout_idx=...)
  ├─ GAEAdvantageConfig          → train_ppo_batch
  ├─ sample 已有 teacher_logprobs
  │    或已注册 teacher[_*] group → train_opd_batch
  └─ 否则                        → train_grpo_batch
```

### 4.1 PPO（`train_ppo_batch`）

1. **Rollout**（Trainer）：生成轨迹；共卡时随后 `rollout.offload` + `train.onload`。
2. **Critic**（`switch_role_modules("critic")` → `groups["critic"].fit`）：
   `fill_gae_token_fields(samples, session_meta, reward_scope)` 写逐步 \(r_t\)
   （`segment` 用 `sample.reward`，`session` 用 `session_score`）；packed
   `forward_only` 后 **按原 sample/segment 切分**再 `GAEEstimator.compute`
   （禁止 pack concat 上直接 GAE；禁止同 session 多段拼成一条轨迹）；value clip；
   返回 advantages 句柄（driver 不解包）。
3. **Reference**（可选）→ `FrozenWorker.forward_only(mode="logprob")`（同一 PackMeta）。
4. **Actor**：`forward_only` 冻 `old_logprobs`；再 `fit`（同一 PackMeta / scatter）。
5. **同步**（Trainer）：仅 actor `weight_update`。步内不做 weight_update / offload。

`exp(old_logprobs-rollout_logprobs)` 是 rollout IS，不能替代 PPO ratio。

`train_ppo_batch` 调用顺序（driver 不解包 tensor）：

```text
switch(critic)  → critic.fit(..., gae, session_meta, reward_scope)
switch(reference)? → Frozen.forward_only(mode="logprob")
switch(actor) → actor.forward_only → actor.fit(advantages, old_logprobs, ref_logprobs)
```

### 4.2 GRPO（`train_grpo_batch`）

按 `session_meta` 在 session 级算 advantage，广播回各 segment，再 `actor.fit`
（与 PPO 同一 `PackMeta`）。

### 4.3 OPD（`train_opd_batch`）

Teacher 两种部署：

* **推理引擎**：produce 已写 `sample.teacher_*`；**不**注册 teacher WorkerGroup；
  本步 **跳过** `forward_only`，`actor.fit` 直接用 sample 字段。
* **训练引擎**：`TrainConfig.frozen` 注册 `teacher` / `teacher_0` / …；本步对每个
  name `switch_role_modules(name)` → `forward_only`，汇总 `teacher_targets`。
* sample 已预填时优先用预填。

多 teacher（MOPD）：`sample.teacher_indices` 路由；`FrozenWorkerConfig.name` 须互异。

## 5. Batch、packing 与内存

### 5.1 RolloutState 到 TrainSample

删除 `RLTrainer._prepare_train_data()`。训练侧协议为 `TrainSample`（建议文件：
`xtuner/v1/rl/data_proto/train_sample.py`）。AgentLoop / ReplayBuffer 仍产出并缓存
`RolloutState`。`build_produce_batch_result()` 是唯一转换点：将每个 group 转为
`list[list[TrainSample]]`，写入 `ProduceBatchResult.train_samples`。

最终路径：`ProduceBatchResult` **不携带** `rollout_states`；训练唯一入口是
`train_samples`。`RLTrainer` 直接：

```python
train_controller.fit(produce_result.train_samples, rollout_idx=...)
```

完整 `RolloutState` 留在 ReplayBuffer / AgentLoop；轨迹落盘、eval、debug 在 convert
前对 groups 取值，或走独立 eval API。

`TrainSample`（produce 写入）包含：`input_ids`、`labels`、`response_mask`、
`rollout_logprobs`、`routed_experts`、`group_id` / `session_id`、`extra_fields`、
`num_tokens`、`reward`，以及 OPD 可选的 `teacher_logprobs` / `target_token_ids` /
`teacher_indices`。

produce **不**写：`token_rewards` / `terminal_mask`（critic `fit` 内）、
`advantages`（GRPO 在 `train_grpo_batch` 写；PPO 由 critic GAE 产出）、
`returns` / `old_values`（仅 critic 侧临时）。

Teacher 目标来源：

* 推理引擎 Teacher：produce 写入 `RolloutState`，`build_train_sample` 对齐进
  `TrainSample`；
* 训练引擎 Teacher：`FrozenWorker.forward_only(mode="sampled"|"topk")` →
  `FrozenWorkerResult`，经 `teacher_targets` 进 `actor.fit`（worker 侧对齐）。

`FrozenWorkerResult`：`logprobs` 必有；Top-K 时另有 `token_ids`（即 `target_token_ids`），
sampled-token / reference 时 `token_ids=None`。

Packing 分层：

```text
produce_utils.build_produce_batch_result
    → build_train_sample 固化 num_tokens → ProduceBatchResult.train_samples
    （无效 group 在 produce 侧过滤，对齐 `_prepare_train_data`；fit 不再滤）
fit → build_session_meta(session_id)
    → build_pack_meta(actor.pack_max_length)   # 本步唯一；多 role / 多算法共用
    → 转发 train_*_batch → Group 内同一 scatter_samples_and_meta
worker.get_pack_train_data → 物化
GAE：packed forward OK；compute 必须按原 sample 切分
```

`pack_max_length` 只来自 `ActorWorkerConfig`，Controller 持有并多 role 共用（对齐现有
`WorkerConfig.pack_max_length`）。Critic/Frozen 不做 R3。

### 5.2 Agentic trainable segment 与 session 聚合

无效 group 在 produce 侧过滤，**不进入 `fit`**。这里是为了对齐现有
`RLTrainer._prepare_train_data` 的检查（`FILTERED` / `FAILED` / `ABORTED`、空
response 等不进训练）。删除 `_prepare_train_data` 后不再在 Trainer/Controller
里兜底：`AgentLoop.maybe_filter_invalid_sample` 整组打 `FILTERED`，
`put_generated_group` 不入 ReplayBuffer，`take_batch` 只取 `COMPLETED`。

`fit` 内 `build_session_meta`：按 `session_id`（无则回退稳定 key）建
segment→session。顺序不能颠倒——GRPO 若以裸 segment 算 group adv，同一 session
会被重复计权。

```text
RolloutState
    -> session-level trainable segments
    -> produce 丢 FILTERED/FAILED（对齐 _prepare_train_data）
    -> fit: build_session_meta
    -> GRPO: session 级 estimator → adv 广播回 segments
    -> PPO: 每 segment 独立 GAE
         reward_scope=segment → 各段自己的 score
         reward_scope=session → 共用 session score，仍不拼轨迹
    -> 本步唯一 PackMeta → 多 role 共用 scatter
```

## 6. Critic 模型输出头适配

Critic 复用现有 hidden-state 主干与输出头生命周期。`head_type` 放在
`xtuner/v1/model/base.py`（`XTunerBaseModelConfig` / TransformerConfig 继承链）；
`fp32_lm_head` 在 `xtuner/v1/config/fsdp.py` 的 `FSDPConfig` 重命名为 `fp32_head`。
rollout/lmdeploy 同名字段同步迁移。

```python
head_type: Literal["lm_head", "critic"] = "lm_head"
fp32_head: bool = False  # 原 fp32_lm_head
```

`lm_head` → `LMHead`；`critic` → `ValueHead(hidden_size, 1)`（不允许 MTP）。
`ValueHead` 对齐 `LMHead`：

```text
loss_ctx is None -> values.float()          # forward_only / GAE
loss_ctx 有值   -> CriticLossContext.forward(h, w, b)  # value clip
```

返回形与 LMHead 一致：`(loss|None, (values|None, extra))`。`CriticLossConfig` /
`CriticLossKwargs` / `CriticLossContext` 不继承 `BaseRLLossConfig`。Value loss 使用
fp32，由 `fp32_head` 同时控制 LM / value head。checkpoint 保存并校验 `head_type` 与
`fp32_head`。

## 7. 分阶段实施与验收

* **阶段 A**：role-aware controller、`TrainConfig`、PackMeta；GRPO 走
  `fit` → `train_grpo_batch`，行为回归不变。
* **阶段 B**：PPO actor 路径（GAE 配置、ratio/clip、old logprob、R3）；OPD
  `train_opd_batch`（推理/训练 Teacher、多 teacher）。
* **阶段 C**：attach critic、`switch_role_modules`、`CriticWorker.fit` 端到端、
  colocate/disagg 时序与 checkpoint。

验收：GAE 手算一致；padding/terminal 无 loss；PPO ratio ≠ rollout IS；GRPO 回归；
仅 actor 触发 `weight_update`；推理侧 OPD 无 teacher group 仍走 `train_opd_batch`；
共卡/分离 PG 符合第 8 节。

## 8. Trainer 关系与 PlacementGroup（共卡 / 分离）

### 8.1 职责

| 组件 | 负责 | 不负责 |
| --- | --- | --- |
| `RLColocateTrainer` | 建单一 PG、produce、**rollout↔train onload/offload**、ckpt、eval、`fit` / `weight_update` | GAE、步内 role 切换 |
| `RLDisaggregatedTrainer` | 建 **两个** PG、后台 produce、ckpt、eval、`fit` / `weight_update`（NCCL） | rollout↔train 腾显存；GAE、步内 role 切换 |
| `TrainingController` | `fit` 算法分发、PackMeta、`switch_role_modules`、scatter | 建 rollout PG、produce |
| Worker | `get_pack_train_data`、前向/loss/optim | 跨进程调度 |

### 8.2 PG 划分

**共卡（`RLColocateTrainer`）**

```text
resources → 单一 PG (_pg)
  ├─ TrainConfig.build(_pg)     # ActorWorker 绑 _pg
  │     └─ critic/frozen attach 同进程（不占额外 bundle）
  └─ RolloutConfig.build(_pg)
```

**分离（`RLDisaggregatedTrainer`）**

```text
train_resources   → train_pg     # self._train_pg
  └─ TrainConfig.build(train_pg)
        └─ critic/frozen 仍 attach 在 train actor 进程，不另建 critic PG

rollout_resources → rollout_pg  # self._rollout_pg；id 必须 ≠ train_pg.id
  └─ RolloutConfig.build(rollout_pg)

weight_update：仅 actor → rollout（强制 NCCL）；critic 权重只留 train 侧
```

现有实现：`__init__` 里 `train_worker_cfg.build(self._train_pg)`、
`rollout_config.build(self._rollout_pg)`；`train_pg.id == rollout_pg.id` 会直接报错。

### 8.3 一步时序（嵌套）

**共卡**（对齐现有 `_train_one_batch(..., offload_rollout_before_train=True, onload_train_before_train=True)` + `_sync_weights_and_save`）：

```text
rollout produce → ProduceBatchResult.train_samples
  → Trainer: rollout.offload
  → Trainer: train_controller.onload
  → Trainer: train_controller.fit(...)   # 内层按算法 switch role
  → Trainer: save（可先 offload optimizer）
  → Trainer: weight_update（仅 actor，按 interval）
  → Trainer: train.offload(model) → rollout onload
```

**分离**（对齐现有 `RLDisaggregatedTrainer._fit`：后台 `produce_loop`，前台 `get_batch`）：

```text
rollout 常驻 rollout_pg（与 train 并行，不 offload 给训练腾卡）
  → get_batch → train_controller.fit(...)   # 内层仍要 switch_role_modules（同 train 卡上分时）
  → 需同步时：pause_produce → save → flush_cache → bind → weight_update（仅 actor，NCCL）
  → continue_produce
```

步内 `switch_role_modules` 与外层 rollout↔train **分层**：外层只在 **共卡** Trainer 做；
分离没有这层切换。内层 Controller 两种模式都要（同 train 进程上 actor/critic 分时）。
共卡时内层嵌套在「train 已 onload、rollout 已 offload」之后。

Trainer 调用（对齐现有 `train_controller.fit`）：

```python
self.train_controller = cfg.train_worker_cfg.build(self._pg)       # colocate
# 或 build(self._train_pg)                                         # disagg
produce_result = await produce(...)  # -> train_samples
self.train_controller.fit(produce_result.train_samples, rollout_idx=...)
```

配置注入：`advantage_estimator_config` → Controller（分发 + 算 adv，PPO 时含
`reward_scope`）；`actor.pack_max_length` → Controller；`actor.loss_cfg` /
`critic.loss_cfg` → 各 worker engine。

对照：

* GRPO：actor + `GRPOAdvantageConfig` + `GRPOLossConfig`；`critic=None`
* PPO：actor + critic + `GAEAdvantageConfig(reward_scope=...)` + `GRPOLossConfig`
  （复用 PG）+ `CriticLossConfig`；常含 `Frozen(name="reference")`
* OPD：推理引擎 Teacher（sample 预填、无 teacher group）或训练引擎
  `Frozen(teacher[_*])`；`sample.teacher_indices` 做 MOPD 路由
* PG：Colocate 绑 `_pg`；Disagg 绑 `train_pg`；critic/frozen 同进程不另建 PG

# PPO

PPO 在 actor 的 policy loss 之外再训一个 value head。Critic 不是第二组 Ray actor，而是挂在 actor 的 `TrainingWorker` 上，配置写在 `WorkerConfig.critic_cfg`。两边共用训练卡，由 `switch_role_modules` 换上换下：先切到 critic，`collect_values` 取 value，`GAEEstimator.compute_gae` 算出 actor advantage 和 critic return，再 `critic.fit`，然后切回 actor 做 policy 更新。

Critic 模型配置的 `head_type` 必须是 `"value_head"`。checkpoint 里没有 value head 权重时，这个头用 `Normal(0, 1 / (hidden_size + 1))` 初始化，其余权重从 checkpoint 加载。Actor 的 `head_type` 是 `"lm_head"`。

Critic 前向不使用 rollout 记录的 expert id。`seq_ctx.rollout_routed_experts` 在进 critic 前被置成 `None`，MoE router 按 critic 自己的 logits 选 expert。这些 ObjectRef 留给后面的 actor `fit` 再取、再释放。

参数在 CPU 和 GPU 之间的切换只发生在 `onload` / `offload`，由 `switch_role_modules` 调用。`collect_values`、`forward_only` 和 `critic.fit` 不自己搬模型。

## GAEAdvantageConfig

Advantage 由 [`GAEAdvantageConfig`](xtuner.v1.rl.advantage.GAEAdvantageConfig) 配置。只做一次 GAE。Actor advantage 和 critic return 共用 `gae_gamma` 和 `gae_lambda`。Critic return 是这次 GAE 的 advantage 加回 value。

```python
from xtuner.v1.rl.advantage import GAEAdvantageConfig

advantage_estimator_config = GAEAdvantageConfig(
    gae_gamma=1.0,
    gae_lambda=1.0,
    reward_scope="segment",
    normalize_actor_advantage=True,
)
```

这个对象传给训练配置的 `advantage_estimator_config`。

```python
class GAEAdvantageConfig(BaseAdvantageConfig):
    gae_gamma: float = 1.0
    gae_lambda: float = 0.95
    reward_scope: Literal["segment", "session"] = "segment"
    normalize_actor_advantage: bool = True
```

- `gae_gamma`、`gae_lambda` 是 GAE 的折扣和 λ。取值范围是 `[0, 1]`。
- `reward_scope` 决定奖励放在哪里。`segment` 把每条样本当作一条轨迹，奖励放在该样本最后一个可训练动作上。`session` 把同一 session 的片段拼成一条轨迹，只在最后一个可训练动作上放一个奖励。`cu_seq_lens` 在轨迹边界重置，观察 token（`labels[1:] == -100`）不参与递推。
- `normalize_actor_advantage` 为 `True` 时，对整个 batch 里保留下来的 actor advantage 做标准化。Critic return 不标准化。被掩掉的位置写成 0。

`GAEEstimator.compute` 不用于 PPO。组内标量 advantage 走 GRPO 那组 estimator；PPO 在 critic value 前向之后调用 `compute_gae`。

## Actor loss

Actor 更新的是词表上的 policy loss，配置类型是 [`GRPOLossConfig`](xtuner.v1.rl.loss.GRPOLossConfig)。`compute_gae` 返回的是每个 token 一条 advantage，actor `fit` 按这个向量对齐到 `shifted_labels`。`labels == -100` 的位置 advantage 写成 0，loss 权重也是 0。切回 actor 之后、做 policy 更新之前，actor 会再算一遍当前策略的 `old_logprobs`。

`policy_loss_cfg["loss_type"]` 选 `vanilla` 或 `mask_pg`。两种都先算重要性比：

```text
ratio = exp(clamp(log_prob - old_log_prob, log_prob_diff_min, log_prob_diff_max))
clipped_ratio = clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
```

`vanilla` 是 PPO clipped surrogate，负优势再做 dual-clip：

```text
pg_losses1 = -ratio * advantages
pg_losses2 = -clipped_ratio * advantages
clip_pg_losses1 = max(pg_losses1, pg_losses2)
pg_losses3 = -clip_ratio_c * advantages
clip_pg_losses2 = min(pg_losses3, clip_pg_losses1)
pg_losses = clip_pg_losses2 if advantages < 0 else clip_pg_losses1
loss = sum(pg_losses * loss_weights)
```

`mask_pg` 用截断后的重要性比给当前 `log_prob` 加权。`ratio > clip_ratio_c` 时 `pg_mask` 为 0，该 token 不进损失：

```text
pg_mask = 1[ratio <= clip_ratio_c]
loss = sum(-loss_weights * advantages * clipped_ratio * log_prob * pg_mask)
```

`loss_weights` 先把 `ignore_idx` 置 0，再按全局可训练 token 数归一。开启 rollout importance sampling 时，再乘上对应的 `is_weights`。`use_kl_loss=True` 时，同一份 loss 上再叠加相对 reference model 的 KL。字段和两种损失的差别见[损失函数](loss.md)。

## CriticWorkerConfig

```python
class CriticWorkerConfig(BaseModel):
    model_cfg: TransformerConfig | BaseComposeConfig
    optim_cfg: OptimConfig
    loss_cfg: CriticLossConfig = CriticLossConfig()
    lr_cfg: LRConfig
    fsdp_cfg: FSDPConfig
    load_from: str | Path
    optimizer_steps: int = 1
    num_passes: int = 1
    optimizer_steps_per_pass: int | None = None
    scheduler_steps: int = 1_000_000
    sp_size: int = 1
    seed: int | None = None
```

- `optimizer_steps_per_pass` 为 `None` 时，critic 沿用 actor 的 pack 计划，并检查步数不超过 `optimizer_steps`。
- 设置了 `optimizer_steps_per_pass` 时，每个 pass 把本地 pack 摊平。第一轮保持原顺序，之后的 pass 打乱，再切成 `optimizer_steps_per_pass` 次优化器更新。`num_passes` 是重复次数。
- 一次更新里可训练 token 数为 0 时跳过。梯度范数通过 `optimizer_step_succeeds` 时才 `scheduler.step()`。

## CriticLossConfig

[`CriticLossConfig`](xtuner.v1.rl.loss.CriticLossConfig) 不继承 policy loss。它回归的是 value head，不是词表。

```python
class CriticLossConfig(BaseLossConfig):
    cliprange_value: float = 0.2
```

有 `old_values` 时，value 相对旧 value 的变化被夹到 `[-cliprange_value, cliprange_value]`。每个 token 取未裁剪平方误差和裁剪后平方误差中较大的一个，再乘 `0.5`。没有 `old_values` 时退化为未裁剪的均方误差。`labels == -100` 的位置权重为 0，其余权重按全局可训练 token 数归一。

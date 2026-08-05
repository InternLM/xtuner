# Checkpoint Engine 权重同步

## 1. 背景

XTuner RL 训练需要周期性把 train engine 中的最新权重同步到 rollout engines。现有权重同步路径包括：

| 路径 | 适用场景 | 说明 |
| --- | --- | --- |
| IPC | 共卡模式 | 训练侧直接把权重通过 IPC 更新给 rollout backend。 |
| Checkpoint Engine | 共卡模式 | 训练侧把权重注册到 in-process `ParameterServer`，再由 `ParameterServer` 推送给 rollout engines。 |
| NCCL | 分离模式 | train workers 和 rollout workers 通过 NCCL broadcast 同步权重。 |

Checkpoint Engine 路径的收益：

- 降低共卡权重更新时的显存峰值。训练侧可以先把权重注册到 `ParameterServer`，再 offload train engine、onload rollout engine，最后由 `ParameterServer` 推送权重。
- 支持恢复失败的 rollout engine。rollout engine 重启后，可以复用上一次注册成功的 checkpoint 重新推送权重。

当前实现只在共卡训练路径中启用 Checkpoint Engine。分离模式检测到 `enable_checkpoint_engine=True` 时会打印 warning，并回退到 NCCL weight transport。

## 2. 架构设计

共卡模式下，在 rollout config 中设置 `enable_checkpoint_engine=True`，可使用 Checkpoint-Engine 进行权重更新， 分离模式下即使设置 `enable_checkpoint_engine=True`，也会回退到 NCCL transport。

### 2.1 初始化

1. 创建`ParameterServer`

训练 worker 内部创建 `ParameterServer`，读取 train world size，作为 `ParameterServer` world size，不额外启动独立 PS 进程。

```python
ParameterServer(
    auto_pg=False,
    rank=self.rank,
    world_size=self.ps_world_size,
)
```
`auto_pg=False` 表示 Checkpoint Engine 复用 XTuner 已经初始化好的 `torch.distributed` 默认进程组，不在 update 后销毁该进程组。

2. 划分参数shard

根据 HF checkpoint 的 `model.safetensors.index.json` 计算当前 PS-rank 负责的参数 key 集合。每个 PS rank 只负责注册自己分到的参数 shard。这些 key 后续用于从 `WeightIterator` 中过滤当前 PS rank 需要注册的 tensor，避免每个 rank 都注册全量参数。

### 2.2 更新
`CheckpointEngineWeightTransport.update(...)` 被拆成两个可选阶段：

```python
update(weight_iterator, need_register=True, need_update=True)
```
1. Register 阶段

当 `need_register=True` 时，transport 会：

- 从 `weight_iterator.iter_batch_groups()` 收集 train engine 当前权重。
- 根据本 rank 的 local checkpoint keys 过滤 tensor。
- 注销上一轮 checkpoint 名称，限制 pinned host memory 占用。
- 调用 `ParameterServer.register_checkpoint(...)` 注册新 checkpoint：

如果 HF index 中的某些 key 没有从 train engine 收集到，transport 会记录错误日志。MTP-only key 和非 MTP key 会分开打印，便于排查模型结构差异。

2. Update 阶段

当 `need_update=True` 时，transport 会把当前 checkpoint 推送到 rollout engines：

- 从 `rollout_info.active_update_targets` 获取当前活跃 rollout targets。
- 校验每个 target 声明的 `update_ranks` 非空、不越界、不重复。
- 判断本次更新是否覆盖全部 PS ranks：
   - 覆盖全部 ranks 时，调用 `ParameterServer.update(..., ranks=None)`，使用 Checkpoint Engine broadcast 路径。
   - 只覆盖部分 ranks 时，调用 `ParameterServer.update(..., ranks=active_ranks)`，使用 p2p update 路径。

3. `need_register` 和 `need_update`

`need_register` 和 `need_update` 用于控制 Checkpoint Engine 更新的两个阶段。

| 参数 | `True` | `False` |
| --- | --- | --- |
| `need_register` | 从 train engine 注册一个新的 checkpoint。 | 复用上一次注册成功的 checkpoint 名称。 |
| `need_update` | 把当前 checkpoint 更新到 rollout engines。 | 只完成注册，不推送到 rollout engines。 |

典型使用方式：

```python
# 常规更新：注册 train 权重并同步到 rollout engines
train_controller.update_weights(need_register=True, need_update=True)

# rollout 恢复：复用上一次 checkpoint，只重新更新 rollout engines
train_controller.update_weights(need_register=False, need_update=True)

# 显存紧张：先注册 checkpoint，稍后再更新 rollout engines
train_controller.update_weights(need_register=True, need_update=False)
offload(train)
onload(rollout)
train_controller.update_weights(need_register=False, need_update=True)
```

`need_update=False` 的核心用途是把注册和 rollout 更新拆成两个时刻执行。这样训练侧可以先把权重注册到 `ParameterServer`，然后释放或切换部分资源，再让 rollout engines 加载 checkpoint，从而降低峰值显存压力。

## 3. 共卡模式 IPC / Checkpoint Engine 速度对比

使用两种更新路径的端到端权重更新时间，包括了`rollout onload_weights`、`train offload`时间。

| 模型 | 并行配置 | Backend | IPC更新时间 | Checkpoint Engine更新时间 | 
| --- | --- | --- | --- | --- |
| Qwen3-4B | TP=2 | SGLang | 0.86s | 2.3s |
| Qwen3-8B | TP=2 | SGLang | 1.1s | 2.8s |
| Qwen3-30B-A3B | TP=2 | SGLang | 3.7s | 5s |
| Qwen3.5-35B-A3B | EP=4 | SGLang | 4.3s | 6.5s |

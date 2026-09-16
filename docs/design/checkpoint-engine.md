# checkpoint-engine

## 简介和安装

[checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine) 是 Moonshot AI 开源的权重更新中间件，用于在 RL 训练中把训练侧权重高效同步到推理引擎。它的核心组件是 `ParameterServer`，支持两种更新方式：

- **Broadcast**：默认推荐路径，适合同步更新一组推理实例。
- **P2P**：适合动态新增或重启推理实例时，只向部分 rank 传输权重，依赖 `mooncake-transfer-engine` 支持 RDMA 传输。

安装方式：

```bash
# 只使用 broadcast
pip install checkpoint-engine

# 使用 P2P，会额外安装 mooncake-transfer-engine
pip install 'checkpoint-engine[p2p]'
```

## 进度

- [ x ] checkpoint-engine colocate SGLang engine 的常规权重更新
- [ ] checkpoint-engine colocate SGLang engine 的失败引擎重启
- [ ] checkpoint-engine colocate LMDeploy engine 的权重更新

## xtuner中使用方法

当前 XTuner 中 checkpoint-engine 只用于 colocate 场景下的 SGLang rollout backend。配置 rollout 时设置：

```python
rollout_config = RolloutConfig(
    weight_transport_type="checkpoint_engine",
)
```
如未设置 `weight_transport_type`，colocate-RL将默认使用 `ipc`，disaggerated-RL默认使用 `NCCL`。

权重更新流程分为两步：

```python
self.train_controller.update_weights(need_register=True, need_update=False)
self.train_controller.offload(target="model")
ray.get(self.rollout_controller.onload_weights.remote(), timeout=RL_TRAINER_RAY_GET_TIMEOUT)
self.train_controller.update_weights(need_register=False, need_update=True)
```

其中：

- `need_register=True, need_update=False`：从训练引擎收集权重并注册到 checkpoint-engine。
- `offload(target="model")`：释放训练侧模型显存，为 rollout onload/update 腾空间。
- `need_register=False, need_update=True`：复用已注册 checkpoint，把权重更新到 rollout engine。

## IPC/Checkpoint-engine显存和时间对比

IPC 路径直接把 tensor 通过 IPC 传给 rollout engine，链路较短，通常延迟更低，但在大模型和复杂并行场景下更容易受显存峰值影响。

checkpoint-engine 路径会先把训练侧权重注册到 `ParameterServer`，再由 checkpoint-engine 规划 bucket 并更新 rollout engine。它的好处是更适合大模型、分片权重和失败 engine 恢复；代价是会有一份模型 shard 常驻 CPU Memory，并且 D2H copy、bucket broadcast 会引入额外时间。

如果只需要在 `ParameterServer.register_checkpoint` 后显式等待一次 accelerator 侧异步操作完成，可以设置：

```python
rollout_config = RolloutConfig(
    weight_transport_type="checkpoint_engine",
    checkpoint_engine_sync_after_register=True,
)
```

该选项默认关闭，profile memory 发现，显式同步会释放掉 GPU 上的 tensor, 在注册时需要等 H2D 完成，而不显式同步，能更快的完成权重更新。

权重更新的显存峰值：

- IPC：trian worker weight + rollout worker weight + bucket

- checkpoint-engine（checkpoint_engine_sync_after_register=False）：max(trian worker weight + parameter server shard, rollout worker weight + buffer *2)

    - 可以优化降低成：max(trian worker weight + bucket, rollout worker weight + buffer *2)，但这会影响性能
    - parameter server shard 指将完整权重分成若干份，每一份的大小

- 这里 bucket 和 buffer 含义不同， bucket是train engine 一次导出的一个batch的权重大小， checkpoint-engine的buffer最小为单个权重的最大大小，该值默认为 8 GiB。


## debug小tips

- OOM：先看各 rank 的 checkpoint shard 是否划分均匀。日志中已打印 `[checkpoint_engine] collect matched local keys rank=xx parameter server shard total= xxx GiB `
- register 后显存不降：可将`checkpoint_engine_sync_after_register=True` 打开，注册后即可释放注册所需GPU memory。


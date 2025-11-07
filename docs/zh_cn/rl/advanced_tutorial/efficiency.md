# 如何配置 Xtuner 并发度以提升 Rollout 效率

在 Xtuner 的 Rollout 阶段，合理配置并发相关参数可以让推理引擎始终保持高负载，充分利用硬件资源，提升整体推理效率。本文介绍 Xtuner 中与并发度相关的主要配置项及其关系，并给出最佳实践建议。

## 主要并发相关配置参数

1. **RolloutConfig.rollout_max_batch_size_per_instance**
   - 控制单个推理实例（如单个模型进程）每次可处理的最大 batch size。
   - 较大的 batch size 能提升 GPU 利用率，但过大可能导致显存溢出或延迟增加。
   - 推荐根据模型的 context_length 和显卡显存实际情况进行调整。
   - Xtuner 后续会提供常见模型与上下文长度下的推荐配置。

2. **RolloutConfig.allow_over_concurrency_ratio**
   - 控制 HTTP 请求的超额并发比例，用于保证能够打满推理引擎。

3. **DataflowConfig.max_concurrent**
   - 控制 Dataflow 的最大并发任务数。Dataflow 为单一控制器，负责所有 rollout worker 的数据分发。
   - Dataflow 每次发送一组数据，实际同一时间内发送的数据条数为 `max_concurrent * prompt_repeat_k`。
   - 建议设置为略高于推理引擎实际处理能力，以保证推理队列始终有任务。

4. **RAY_MAX_CONCURRENCY**
   - Ray 后端的最大并发任务数，通过环境变量配置，默认为 1024。

5. **httpx 最大连接数**
   - 控制 HTTP 客户端（如 RolloutWorker）发起到推理服务的最大并发连接数。
   - 建议与 `rollout_max_batch_size_per_instance` 保持一致或略高。

## 配置关系与建议

- **推荐配置流程**：
  1. 根据模型和硬件资源，确定合理的 `rollout_max_batch_size_per_instance`（如 128、256、512、 1024）。该参数为可选，若用户不提供，Xtuner 会根据 `context_length` 提供预设值：`context_length` ≤ 4K 时并发度为 1024，≤ 16K 时并发度为 512，≤ 32K 时并发度为 128。
  2. 设定 `DataflowConfig.max_concurrent`，建议为 `rollout_max_batch_size_per_instance * num_of_infer_instance / prompt_repeat_k * allow_over_concurrency_ratio`。其中 `num_of_infer_instance` 为启动的推理引擎实例数量，一般为节点数 / `tensor_parallel_size`。
  3. 设置 `RAY_MAX_CONCURRENCY` 环境变量，建议与 `rollout_max_batch_size_per_instance * num_of_infer_instance` 保持一致或略高。
  4. httpx 最大连接数默认配置为 `rollout_max_batch_size_per_instance * allow_over_concurrency_ratio`。

- **动态调整**：可通过监控推理队列长度、GPU 利用率和响应延迟，动态调整上述参数，找到最优并发配置。

## 示例配置

```python
resource = AcceleratorResourcesConfig(
    num_workers=8
)
rollout = RolloutConfig(
    rollout_max_batch_size_per_instance=1024,
    tensor_parallel_size=1,
    ...
)

dataflow = DataflowConfig(
    max_concurrent=600, # int(1024 * (8 / 1) / 16 * 1.2)
    prompt_repeat_k=16,
    ...
)

# 环境变量设置
export RAY_MAX_CONCURRENCY=1024
```
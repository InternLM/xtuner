# How to Configure Xtuner Concurrency to Improve Rollout Efficiency

During the Rollout phase of Xtuner, properly configuring concurrency-related parameters ensures that the inference engine maintains high load, fully utilizes hardware resources, and improves overall inference efficiency. This document introduces the main concurrency-related configuration options in Xtuner, explains their relationships, and provides best practice recommendations.

## Main Concurrency-Related Configuration Parameters

1. **RolloutConfig.rollout_max_batch_size_per_instance**
   - Controls the maximum batch size that a single inference instance (such as a model process) can handle at one time.
   - Larger batch sizes can improve GPU utilization, but excessively large values may cause out-of-memory errors or increased latency.
   - It is recommended to adjust this parameter based on the model's context_length and actual GPU memory.
   - Xtuner will provide recommended configurations for common models and context lengths in future releases.

2. **RolloutConfig.allow_over_concurrency_ratio**
   - Controls the over-concurrency ratio for HTTP requests to ensure the inference engine is fully loaded.

3. **DataflowConfig.max_concurrent**
   - Controls the maximum number of concurrent tasks in Dataflow. Dataflow acts as a single controller, distributing data to all rollout workers.
   - Dataflow sends a batch of data each time; the actual number of data items sent at the same time is `max_concurrent * prompt_repeat_k`.
   - It is recommended to set this slightly higher than the actual processing capability of the inference engine to ensure the inference queue always has tasks.

4. **RAY_MAX_CONCURRENCY**
   - The maximum concurrency for the Ray backend, configured via environment variable. The default is 1024.

5. **httpx max connections**
   - Controls the maximum number of concurrent connections that the HTTP client (such as RolloutWorker) can initiate to the inference service.
   - It is recommended to set this equal to or slightly higher than `rollout_max_batch_size_per_instance`.

## Configuration Relationships and Recommendations

- **Recommended Configuration Process**:
  1. Determine a reasonable value for `rollout_max_batch_size_per_instance` based on the model and hardware resources (e.g., 128, 256, 512, 1024). This parameter is optional; if not provided, Xtuner will use preset values based on `context_length`: concurrency is 1024 for `context_length` ≤ 4K, 512 for ≤ 16K, and 128 for ≤ 32K.
  2. Set `DataflowConfig.max_concurrent`. It is recommended to use `rollout_max_batch_size_per_instance * num_of_infer_instance / prompt_repeat_k * allow_over_concurrency_ratio`, where `num_of_infer_instance` is the number of inference engine instances started (usually number of nodes / `tensor_parallel_size`).
  3. Set the `RAY_MAX_CONCURRENCY` environment variable. It is recommended to set this equal to or slightly higher than `rollout_max_batch_size_per_instance * num_of_infer_instance`.
  4. The default httpx max connections should be set to `rollout_max_batch_size_per_instance * allow_over_concurrency_ratio`.

- **Dynamic Adjustment**: You can dynamically adjust these parameters by monitoring the inference queue length, GPU utilization, and response latency to find the optimal concurrency configuration.

## Example Configuration

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

# Environment variable setting
export RAY_MAX_CONCURRENCY=1024
```

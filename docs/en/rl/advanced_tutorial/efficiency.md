# How to Configure Xtuner Concurrency to Improve Rollout Efficiency

During the Rollout phase of Xtuner, properly configuring concurrency-related parameters ensures that the inference engine maintains high load, fully utilizes hardware resources, and improves overall inference efficiency. This document introduces the main concurrency-related configuration options in Xtuner, explains their relationships, and provides best practice recommendations.

## Main Concurrency-Related Configuration Parameters

1. **RolloutConfig.rollout_max_batch_size_per_instance**
   - Controls the maximum batch size that a single inference instance (such as a model process) can handle at one time.
   - Larger batch sizes can improve GPU utilization, but excessively large values may cause out-of-memory errors or increased latency.
   - It is recommended to adjust this parameter based on the model's context_length and actual GPU memory.
   - Xtuner will provide recommended configurations for common models and context lengths in future releases.

2. **RolloutConfig.allow_over_concurrency_ratio**
   - Deprecated compatibility option. It no longer controls runtime rollout concurrency.
   - If this option is set to a non-default value, Xtuner emits a warning and ignores it. Ray actor concurrency and HTTP client connection caps are now fixed to large internal values so requests do not queue before reaching the inference engine.

3. **DataflowConfig.max_concurrent**
   - Controls the maximum number of concurrent tasks in Dataflow. Dataflow acts as a single controller, distributing data to all rollout workers.
   - Dataflow sends a batch of data each time; the actual number of data items sent at the same time is `max_concurrent * prompt_repeat_k`.
   - It is recommended to start from the total inference capacity divided by `prompt_repeat_k`, then tune based on GPU utilization, inference queue length, and latency.

4. **httpx max connections**
   - This is no longer a user-facing tuning parameter.
   - Xtuner sets the RolloutWorker HTTP client connection cap to a large internal value so requests do not queue in httpx; the inference engine owns request scheduling and queueing.

## Configuration Relationships and Recommendations

- **Recommended Configuration Process**:
  1. Determine a reasonable value for `rollout_max_batch_size_per_instance` based on the model and hardware resources (e.g., 128, 256, 512, 1024). This parameter is optional; if not provided, Xtuner will use preset values based on `context_length`: concurrency is 1024 for `context_length` ≤ 4K, 512 for ≤ 8K, and 128 for longer contexts.
  2. Set `DataflowConfig.max_concurrent`. A practical starting point is `rollout_max_batch_size_per_instance * num_of_infer_instance / prompt_repeat_k`, where `num_of_infer_instance` is the number of inference engine instances started. Tune this value with runtime metrics.
  3. Do not tune `allow_over_concurrency_ratio` or httpx max connections for rollout efficiency. `allow_over_concurrency_ratio` is deprecated, and httpx max connections are handled by Xtuner internals.

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
    max_concurrent=512, # int(1024 * (8 / 1) / 16)
    prompt_repeat_k=16,
    ...
)

# Ray actor concurrency and RolloutWorker httpx max connections are handled by Xtuner internals.
```

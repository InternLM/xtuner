# RL Review Guidelines

Use these rules whenever a PR changes code under `xtuner/v1/rl`. They extend the general review
standards in `.claude/CLAUDE.md`; they do not replace them.

## Required Review Output

Every review for an RL change must state the ProduceBatchResult impact near the top-level summary:

```text
ProduceBatchResult impact: <specific impact or "not affected">
```

Also include routed-experts impact when the PR touches routed-experts logic, rollout response
handling, `RolloutState.extra_fields`, object references, or memory ownership around rollout outputs:

```text
RoutedExperts impact: <specific impact or "not affected">
```

Also include Ray concurrency impact when the PR touches Ray actor methods, actor construction,
method decorators or wrappers, concurrency-group definitions, or rollout control paths:

```text
Ray concurrency impact: <effective groups and possible starvation impact, or "not affected">
```

Do not leave these impacts implicit in the finding text. If a finding is about rollout status,
pause/abort/timeout behavior, response handling, producer aggregation, routed-experts ownership, or
Ray actor concurrency, repeat the relevant impact line inside that finding so the downstream effect
is visible at the point of review.

## ProduceBatchResult Checklist

Review whether the PR can change trainer-visible batch accounting, timing, or reward semantics in
`ProduceBatchResult`.

Check this area when the PR changes any of these paths:

- `RolloutState.status`, `finish_reason`, or status conversion.
- Abort, filter, expire, retry, timeout, cancellation, or failure handling.
- Producer, agent loop, judger, replay buffer, rollout worker, rollout controller, or backend pause
  cleanup logic.
- Writers, readers, tests, or fake implementations that construct or consume `ProduceBatchResult`,
  including trainer-facing code outside `xtuner/v1/rl`.

Name the concrete field-level impact when any of these fields can change:

- Batch status: `status`.
- Returned groups: `rollout_states`.
- Generation timing: `group_gen_count`, `group_gen_mean_s`, `group_gen_p50_s`, `group_gen_p99_s`,
  `group_gen_p99_p50_ratio`, `group_gen_pause_time_s`.
- Replay-buffer leftovers: `leftover_init`, `leftover_completed`, `leftover_aborted`,
  `leftover_expired`, `leftover_failed`, `leftover_filtered`.
- Reward accounting: `raw_rewards_sum`, `raw_rewards_count`.
- Produced work counters: `produced_samples`, `produced_tokens`, `produce_time_s`.
- Multi-task aggregation: `task_batch_sizes`, `task_results`.

Common impacts to call out explicitly:

- A sample moving between `ABORTED`, `FAILED`, `FILTERED`, `EXPIRED`, and `COMPLETED` can change the
  corresponding `leftover_*` counts.
- Pause, abort, timeout, and cancellation changes can inflate or deflate `group_gen_*` timing,
  especially `group_gen_pause_time_s`.
- Reward or filter-path changes can change `raw_rewards_sum`, `raw_rewards_count`,
  `produced_samples`, and `produced_tokens`.

## Ray Actor Concurrency Checklist

Review whether high-volume rollout calls can starve pause, abort, health-check, restart, or shutdown
RPCs.

Check this area when the PR changes any of these paths:

- Ray actor methods decorated with `@ray.method(...)` or `@ray_method(...)`.
- Decorators or wrappers stacked on Ray actor methods, especially wrappers using `functools.wraps`
  or exposing `__wrapped__`.
- Actor construction through `ray.remote(...)`, including `concurrency_groups` and
  `max_concurrency`.
- Generate, pause, continue, abort, health-check, restart, or shutdown methods.

When a Ray method is combined with another decorator, keep the Ray method decorator closest to the
function definition:

```python
@trace_rollout_endpoint("rollout.worker.generate")
@ray.method(concurrency_group=ROLLOUT_CONCURRENCY_GROUP_GENERATE)
async def generate(...):
    ...
```

Do not use the reverse order:

```python
@ray.method(concurrency_group=ROLLOUT_CONCURRENCY_GROUP_GENERATE)
@trace_rollout_endpoint("rollout.worker.generate")
async def generate(...):
    ...
```

The current Ray actor-class processing unwraps decorated methods before reading Ray method metadata.
If `@ray.method(...)` is outside a wrapper, metadata such as `__ray_concurrency_group__` can remain
on the discarded wrapper, causing the method to fall back to the default concurrency group.

Review the effective dispatch behavior, not only the visible decorator order:

- After following `__wrapped__`, the callable inspected by Ray must retain the expected Ray method
  metadata.
- Every named method concurrency group must be declared by the corresponding `ray.remote(...)`
  actor construction.
- High-volume `generate` methods must not accidentally fall back to the default group.
- Pause, abort, and other control RPCs must remain schedulable while generation is saturated.
- Slow health or lifecycle methods must not occupy every slot needed by critical control RPCs.
- Blocking work inside async actor methods must not block the actor event loop or concurrency lane.

## RoutedExperts Checklist

Review whether routed-experts ownership and cleanup remain correct.

Check this area when the PR changes any of these paths:

- LMDeploy rollout response handling.
- `return_routed_experts`, `routed_experts`, `RolloutState.extra_fields`, or object-ref plumbing.
- Abort, cancellation, timeout, filter, retry, or failure paths before response handling completes.
- Background tasks, replay-buffer storage, trainer batches, metrics, tests, or fake rollout
  responses that can retain routed-experts object refs.

For LMDeploy rollout, `rollout_worker` obtains routed-experts object refs from the LMDeploy shared
store. At that point, ownership moves from LMDeploy to XTuner. A review finding should call out both
sides of the ownership boundary when relevant:

- Leak before transfer: requests whose routed experts remain in LMDeploy because XTuner never
  obtains the object refs.
- Leak after transfer: object refs that XTuner keeps alive too long through `RolloutState`, replay
  buffer, trainer batches, metrics, fake tests, or background tasks.

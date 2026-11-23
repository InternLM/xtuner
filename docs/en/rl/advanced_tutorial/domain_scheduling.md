# Multitask domain admission

The colocated manager supports an opt-in Miles-style domain scheduler. The
existing `AsyncProduceStrategy` still generates, collects and validates groups.
Its `_PendingTasks` remains the only owner of live generation tasks. Request
gates, turn keepalive, version publication and trajectory collection use the
original preview implementation.

## Configuration

The default `FixedDomainSchedulerConfig()` preserves weighted per-task batch
quotas and the replay policy configured by the recipe. To opt into adaptive
admission, change only the existing manager's scheduler field:

```python
from xtuner.v1.rl.agent_loop_manager import AdaptiveDomainSchedulerConfig

# cfg is the recipe's existing RLColocateTrainerConfig. Retain its tasks,
# AsyncProduceStrategy configurations, turn_keepalive and inference settings.
cfg.agent_loop_manager_cfg.domain_scheduler = AdaptiveDomainSchedulerConfig(
    target_groups={"tau3": 8, "automationbench": 8},
    initial_acceptance_rates={"tau3": 1.0, "automationbench": 1.0},
    selection_alpha=0.5,
    acceptance_floor=0.01,
    acceptance_ewma_alpha=0.1,
    concurrency_pbar=0.75,
    concurrency_pmin=0.0,
    concurrency_pmax=1.5,
    # Starting estimates, in seconds; subsequent normally returned groups
    # update the estimates. Choose priors appropriate to the actual workload.
    initial_duration_seconds={"tau3": 3000.0, "automationbench": 370.0},
    duration_ewma_alpha=0.1,
    max_concurrent_samples=128,
    group_start_interval=0.0,
    seed=42,
)
```

Task names must exactly match the manager's task names. Their targets must sum
to the training batch size **in groups** (16 here). With 8 siblings per group,
the independent sample ceiling above permits at most 16 live groups. Admission
and release operate on whole groups; an early-finishing sibling does not free
part of a reservation. Different sibling counts across tasks are supported.
These are example scheduler settings, not an approved full training recipe.

Adaptive mode requires the original async strategy and its default completion
predicate for every task. It does not change `turn_keepalive` or any request-gate
setting. Retain `turn_keepalive=True` from the preview recipe to keep unfinished
sessions across updates. Without keepalive, the recipe's original pause/cleanup
behavior still applies.

## Selection and budgets

For domain `i`, let `B_i` be its soft target, `r_i` its acceptance EWMA and `A_i`
the number of accepted groups from that domain in the leading shared FIFO
window of `sum(B_i)` groups:

```text
r_eff_i = max(r_i, acceptance_floor)
w_i = (alpha * B_i + (1 - alpha) * max(B_i - A_i, 0)) / r_eff_i
```

Select the next domain randomly in proportion to `w_i` among currently eligible
domains. In-flight groups and previous consumed batches are not counted in
`A_i`. If all eligible weights are zero, use `B_i / r_eff_i` as the fallback.
The scheduler chooses a task's sampler; dataset IDs, native rewards and tool
responses keep their existing meanings.

Acceptance observations come from scored, structurally valid groups returned
as `COMPLETED` or `FILTERED` when that task has a configured dynamic filter.
The former supplies 1 and the latter 0. Failed/aborted/unscored groups,
coroutine errors, cancellation and replay staleness are not filter rejections.
With filtering disabled, acceptance stays at its configured prior. The
scheduler never adds or reruns a task filter.

When `concurrency_pbar` is set, use duration estimates `t_i` to solve:

```text
m_i = B_i / r_eff_i
p_i = clip(c * t_i - 1, concurrency_pmin, concurrency_pmax)
sum(m_i * p_i) / sum(m_i) = concurrency_pbar
live_group_cap_i = floor((1 + p_i) * m_i)
```

Duration runs from generation-task admission to normal return and includes
tool work and time waiting across weight updates. Errors/cancellation release
the slot without a duration observation. A shrinking cap prevents new
admissions; it never cancels existing tasks. Omit `concurrency_pbar` and duration
priors to disable these adaptive domain caps.

The sample ceiling and the original strategy's `max_pending_tasks` remain
independent limits. The aggregate original oversampling allowance also remains:
completed groups plus owned pending futures cannot trigger new admissions at
or above `G + sum(ceil(over_sample_threshold_i * B_i))`, where `G = sum(B_i)`.
The original per-task tail-batch condition can remove that task's extra
allowance. These are admission limits, not guarantees that all capacity can be
filled. Runtime/engine semaphores retain their original behavior.

`group_start_interval` paces group admissions globally. The first group can
start immediately. A pause does not accumulate catch-up credits, and pacing
does not delay collection of completed work. It is separate from sandbox
creation controls.

## Batch composition and lifecycle

Adaptive mode consumes completed groups in **replay insertion order across all
tasks**, rather than waiting for fixed domain quotas. Targets are soft: a
16-group batch targeting 8+8 may contain 5+11. Groups completed simultaneously
can be inserted in either order by the existing collectors; this is not a new
chronological trace-merging mechanism.

The existing replay staleness refresh runs before production. The existing
per-task token masks are applied when the FIFO batch is consumed. Surplus
completed groups stay in replay for later batches, subject to those same
staleness rules. A native tool error is not a scheduler filter or mask.

Once the shared batch is ready, the original manager pauses at the existing
request barrier. With keepalive, live futures remain with their original
strategy owner. The next window resumes the original gate with the newly
published model version. Checkpoint/final quiesce still finishes live episodes
with gates open before pausing; the scheduler does not add a drain/drop path.

Checkpoints retain the existing samplers and replay entries plus the scheduler's
random-draw cursor. On a fresh process, online acceptance/duration estimates
restart from configured priors and in-flight counts start empty. Scheduler
statistics are emitted in the manager log; no dashboard schema is changed.

## CPU validation

`tests/rl/test_domain_scheduler.py` checks the weighted selection and clipped
budget solver. `tests/rl/test_domain_scheduler_async.py` checks the native async
owner and request gates across model updates, filter replacement, FIFO surplus,
token masking, cap shrinkage, cancellation, pacing and checkpoint restore.
These tests do not establish GPU or end-to-end sandbox readiness.

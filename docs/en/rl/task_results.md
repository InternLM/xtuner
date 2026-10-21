# Task results and infrastructure failures

Training and evaluation accept only completed attempts with a finite verifier
score. Infrastructure errors, adapter failures, cancellation, missing scores and
cleanup failures are unscored (`FAILED`, `reward=None`), never synthetic zeroes.
Diagnostic artifacts and verifier records remain available on failed attempts.

A task's time budget may expire with a valid verifier result. The harness must
stop the agent before grading, and record `finish_reason="timeout"` together
with `finish_info={"termination_reason": "task_budget_exhausted", "agent_stopped": true}` in the response artifact. An unqualified timeout is an
infrastructure/unknown failure. Task scoring policy belongs to the verifier.

A valid evaluation score does not imply a trainable sample: training additionally
requires recorded model tokens, logprobs and a valid trace. No trace is synthesized
by tokenizing a missing response. Retries must use a new attempt and environment;
this contract does not authorize replaying side-effectful tool calls.

Evaluation delivers unscored attempts instead of dropping them: eval producers
store failed groups with their true status and the eval batch drains them, so
`accuracy` is computed over scored attempts only and `eval_failed_ratio` reports
the unscored share. Unscored attempts count as non-passing in grouped `pass@k`
metrics (an attempt with no verifier result cannot pass, but must not become a
synthetic zero in `accuracy`). Training still discards failed groups before the
replay buffer; the two modes differ deliberately.

## Evaluation quotas and metrics

Synchronous, colocated asynchronous and disaggregated asynchronous evaluation
count both completed and failed groups toward the requested batch size. Producers
do not replace failed attempts to reach a success-only quota. Training's
`over_sample_threshold` is ignored during evaluation.
Completed and failed groups are consumed together in arrival order, without
prioritizing successful results or training staleness. A group still occupies one
batch slot even when it contains multiple rollout states; metrics count the
individual attempts delivered in those groups.

The built-in metrics use different denominators intentionally:

| Metric              | Population                                  | Meaning                                                   |
| ------------------- | ------------------------------------------- | --------------------------------------------------------- |
| `accuracy`          | Attempts with a finite verifier score       | Fraction whose score is positive                          |
| `eval_failed_ratio` | All delivered attempts                      | Fraction without a valid score                            |
| Grouped `pass@k`    | First k attempts per eligible problem group | Whether any attempt passed; unscored attempts cannot pass |

For example, one passing score, one valid zero score and one infrastructure
failure produce `accuracy=0.5` and `eval_failed_ratio=1/3`. The failure has no
reward; it is not a third task scored zero. When every attempt is unscored, the
built-in evaluator reports `accuracy=0.0` and `eval_failed_ratio=1.0`; interpret
the accuracy as an empty-population fallback, not as a measured task success rate.
Report these metrics together when comparing runs with different failure rates.

## Migrating custom metric callbacks

`EvaluatorConfig(compute_metric_func=...)` passes only scored attempts to a custom
callback by default. Existing callbacks that read `sample.reward["score"]` can
keep doing so. The evaluator computes `eval_failed_ratio` independently over the
full batch, even when a callback filters its input further. That metric name is
reserved: a value returned by a callback is replaced with the framework's value.

A callback may receive fewer attempts or fewer data sources than were requested.
If a data source has no valid scores, omit its score metric instead of raising
or assigning zero. The bundled MOPD callbacks follow this convention. If the
entire batch is unscored, the default custom callback path skips the callback and
returns only `eval_failed_ratio`; consumers must tolerate absent task metrics.
The built-in evaluator remains failure-aware and keeps failed attempts in its
grouped metrics.

Callbacks that need all attempts, for example to compute failure rates per data
source or their own grouped metrics, must opt in explicitly:

```python
evaluator_cfg = EvaluatorConfig(
    compute_metric_func=compute_failure_aware_metrics,
    include_failed_samples=True,
)
```

Such callbacks must handle `Status.FAILED`, `reward=None`, missing or non-finite
scores, and empty input. Opting in changes only the callback input. It does not
change the failure-rate denominator or remove diagnostic artifacts from failed
attempts.

## Migrating harnesses and result consumers

- A timeout with a valid verifier result is scoreable only when the harness
  records both explicit task-budget termination and confirmation that the agent
  stopped, as shown above. Do not relabel infrastructure timeouts as task-budget
  exhaustion to obtain a score.
- Infrastructure, adapter and cleanup failures remain `FAILED` even if a stale
  verifier record or diagnostic reward metadata exists. Consumers must accept
  `reward=None` and use the retained errors and artifacts for diagnosis.
- A genuine verifier zero is a valid score. Do not use zero as a fallback for
  missing, invalid or non-finite scores. The verifier result takes precedence
  over reward metadata supplied by an adapter.
- Evaluation validity and training eligibility differ: a valid score can be
  reported without manufacturing the model trace required for training.
- Dashboards and downstream exporters should preserve the failure rate and
  missing per-source score metrics. Historical success-only evaluations may have
  hidden or retried infrastructure failures, so their coverage is not directly
  comparable to a fixed-attempt evaluation.

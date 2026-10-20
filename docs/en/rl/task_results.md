# Task results and infrastructure failures

Training and evaluation accept only completed attempts with a finite verifier
score. Infrastructure errors, adapter failures, cancellation, missing scores and
cleanup failures are unscored (`FAILED`, `reward=None`), never synthetic zeroes.
Diagnostic artifacts and verifier records remain available on failed attempts.

A task's time budget may expire with a valid verifier result. The harness must
stop the agent before grading, and record `finish_reason="timeout"` together
with `finish_info={"termination_reason": "task_budget_exhausted",
"agent_stopped": true}` in the response artifact. An unqualified timeout is an
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

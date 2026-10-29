# Comparison experiment discipline

Use this file when choosing a comparison, its budget and its verdict. See [pitfalls](pitfalls.md) for
conditional failure modes and [diagnostics](diagnostics.md) for measurement methods.

## 1. Declare the question and acceptance criteria

- Separate same-input numerical alignment, faithful issue reproduction and native-stack performance
  benchmarking. A reduced diagnostic pair cannot silently replace a requested production reproduction.
- Record each arm's source/build identity, local patch, initial checkpoint/state, effective input manifest,
  update semantics, backend/topology, horizon and output path. Check existing evidence before repeating a run.
- Choose bounds before the measured pair. State the metric formula, absolute/relative tolerances, zero-norm
  handling, tensor coverage, comparison window and required statistics. Derive bounds from a trusted
  reference and/or separate target-stack repeatability probes, not another model's reported numbers.
- Preserve failures. If the contract or tolerance is wrong, explain the correction and validate the revised
  criterion on a fresh pair; never relabel an observed failure as a pass because loss happens to look close.
- State evidence limits: equal loss does not prove equal gradients; equal norms do not prove equal vectors;
  selected tensor checks do not cover the whole model; a short run does not validate a long horizon.
- Converge on symptoms, not on a fix list: localization is closed when every observed symptom has a single
  cause confirmed by a one-variable A/B and the residual is quantified or attributed. A known upstream fix
  list is a post-hoc coverage check, never the convergence target — do not run experiments just to match it.
  Fixes for conditions the current scope never triggers stay "not reproduced in this scope".
- Before concluding a residual is structural or environment-level, exhaust the inheritable fix list and your
  own decomposed factors; an exclusion argument whose premise new data has invalidated must be re-run, not
  carried forward silently.

## 2. Keep comparisons interpretable

- Freeze actual input contents and assignment before the first-step gate. Same data source, seed, label
  count or attention ratio is insufficient; compare canonical tokens/labels/context at the consumer.
- Match model/objective/update semantics and precision. If different backends or quantization paths are
  intentional, name that experimental variable and validate its forward/backward effect; do not hide it.
- Match actual lr per update, not scheduler names or displayed phases. Preserve the shared-prefix schedule
  when changing total steps. Never change one side's warmup just to align plots.
- Use one-variable contrasts for causal diagnosis. Larger environment/code changes are useful for screening
  but cannot identify one kernel or flag as the cause. Hold non-tested state fixed and check interactions.
- Close every decomposed factor. A boundary probe that splits a discrepancy into N factors owes N
  single-variable confirmations; fixing the dominant factor and watching the headline metric shrink does
  not retire the remaining factors, which resurface later at greater cost.
- For stochastic paths, inspect RNG usage and establish repeatability where needed. Do not attribute a
  discrepancy to non-determinism from a single pair, or assume matching seeds yield matching dropout masks.
- Keep pointwise checks for a supported common-input/update contract. Add long-horizon statistics under
  predeclared windows and definitions; neither divergence nor harmlessness follows from run length alone.

## 3. Bound execution and preserve evidence

- Proceed preparation → first step → short updates → agreed long horizon. Give a time estimate that includes
  startup/loading/compilation. A costly startup can dominate a one-step diagnostic run.
- Predeclare arm order, budgets and safety/failure stops. Prioritize the decision-making arm, but do not
  omit another requested arm. Stop at the agreed scope; extending repeats, pack lengths or topology needs
  budget/authorization rather than an unbounded search for either a pass or a spike.
- For intermittent failures, report repetitions and covered configurations. “Not reproduced in this scope”
  is valid; “proved absent” or “structurally harmless” is not. Bisect a reproducible anomaly, not noise
  without a defined detection criterion.
- Freeze source while workers run. For same-checkout toggle experiments, using the same source path can
  reduce cache differences; cross-checkout/framework comparisons instead control and report cold/warm cache
  state and exclude predeclared startup windows. Never edit running source to force cache equality.
- Use unique logs and the approved launcher/concurrency policy. On cancellation, stop owned relay/watchers,
  then driver and workers; preserve partial logs and confirm those resources are released.
- Do not assume access to another person's data or account. Reuse paths only after verifying authorization,
  existence and relevance. Keep restricted corpora/checkpoints in approved storage; never copy them for convenience.

## 4. Deliver a scoped verdict

Save aligned per-update data and plots for loss, pre-clip grad norm, step time and peak memory, with lr and
throughput panels where useful. Specify timing windows, token denominators, units and rank aggregation.
Compare both the declared criteria and open differences. Keep a compact attempt ledger:

| Attempt | Controlled change | Evidence and result | Next decision |
|---|---|---|---|
| First-step replay | Baseline contract | Input hashes match; gradient criterion fails | Inspect backward/scaling path; no long run |
| Focused A/B | One justified precision change | Record boundary errors and criterion result | If passed, verify short updates before long run |

These rows illustrate decisions, not experimental results. Report whether validation was a document review,
a local probe, an existing-workload replay or a new-model end-to-end run; do not conflate them.

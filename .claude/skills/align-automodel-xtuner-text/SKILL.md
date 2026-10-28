---
name: align-automodel-xtuner-text
description: Align text-only training accuracy between AutoModel (NVIDIA NeMo Automodel) and XTuner after model onboarding. Use to validate same-input loss and gradients, investigate cross-framework numerical discrepancies, or verify a fix. Read the pitfalls first, freeze effective inputs and training semantics, pass a first-step gate, then compare multi-step loss, grad norm, performance and memory. Includes symptom-based diagnostics with conditional examples, not model-specific acceptance thresholds. Not for inference or multimodal alignment.
---

# Align AutoModel ↔ XTuner Text Training Accuracy

Use this order: **freeze inputs → first-step checks → multi-step comparison**.
Decompose the failing metric before proposing a cause. Reuse diagnostic methods, not another model's
thresholds, kernel flags, or conclusions.

## Before starting

Read [pitfalls](references/pitfalls.md) in full before editing configs or launching experiments.
P1–P5 are categories, not workflow stages. Use [experiment discipline](references/experiment-discipline.md)
to plan runs and [diagnostics](references/diagnostics.md) when collecting evidence.
Use [framework anchors](references/anchors.md) to locate the shared data, loss, gradient-norm and probe
entry points before reading new model code.
Confirm the authorized machines, data, run budget and output location; this skill grants no resource or data access.
Before reserving GPUs or launching runs, post an **experiment card** and wait for the user's confirmation:
objective, predeclared pass/fail criteria, smallest sufficient scale and horizons, prerequisites
(code/data/machines), and estimated cost and duration. If the card reuses an earlier run as a comparison
baseline instead of launching a fresh arm, state the reused arm's identity and the contract-match evidence
(same checkpoint, same pack stream, same schedule and effective config).

## Preparation: freeze the comparison contract and actual inputs

**Input:** both checkouts, checkpoints, configs, data entry points and the requested validation scope.

1. Record source revisions and local patches; inspect actual model loading, data, loss, synchronization,
   optimizer and scheduler paths. Resolve missing inputs or access with the user; do not invent paths.
2. Build the checklist below. For each item, record **library default / configured / runtime-effective**
   values, evidence and status: aligned, different-with-explanation, not applicable, or unverified.
   Unverified semantics are not a pass. Match intended computations, not just option names.
3. Freeze the order and assignment of actual inputs per logical update and microbatch. Prefer replaying
   one side's consumed pack stream through a verified adapter; independently produced batches also work
   if their effective contents match. Verify token IDs, labels, masks, document boundaries, positions and
   attention context, including rank/shard mapping. Use tensor comparisons or canonical hashes, not just
   the same seed, sample IDs or label counts. See diagnostics §1.
4. Choose acceptance metrics, tolerances, short/long horizons and stopping rules **before the measured
   comparison**. Use the target dtype/backend, trusted references and separate repeatability baselines
   to justify bounds. There is no universal loss-delta or grad-norm threshold. Do not widen a failed bound
   after observing results; record the failure and validate any revised criterion on a fresh comparison.
5. Check dependencies and the smallest representative forward/backward path. Do not require a multi-update
   smoke run before the first-step gate. A reduced model must exercise the relevant layer types; its
   result does not validate omitted layers or the full topology.

| Contract area | Verify on both sides |
|---|---|
| Model and state | Checkpoint identity and loaded tensors, architecture/layer types, buffers, frozen/tied parameters, auxiliary heads/losses, optimizer state if resuming |
| Data and objective | Tokenizer/template, ordered effective batches, label shift/masking, boundaries/positions, loss terms and weights, global normalization |
| Update semantics | Global batch and accumulation, sync/unscale/clip order, optimizer groups and master/state precision, actual applied lr per update, skipped updates |
| Numerics and execution | Parameter/compute/accumulation/reduction dtypes, quantization/scales, kernels/builds, parallel mesh, checkpointing/offload, training mode and RNG behavior |

**Output / gate:** a recorded contract, CPU-verified input manifest, consumer-side probe plan, and
predeclared acceptance criteria. Device-only fields remain pending until the first-step probe checks them;
unresolved input or loss semantics block a parity verdict, not the diagnostic probe needed to resolve them.
If equivalence cannot be established, stop the numerical-parity path. A separately labeled native-pipeline
benchmark may still be useful, but is not pointwise accuracy evidence.

## Stage 1: compare the first training step

**Input:** the prepared contract and the same initial state on both sides.

- Run this gate **first in the target production configuration**, with its parallelism, kernels,
  checkpointing and offload enabled; acceptance always comes from that configuration. Only when it fails,
  localize with a reduced baseline: single rank, no EP/SP, no compile, no offload, then re-enable one
  feature at a time and repeat the gate. A discrepancy already present in the reduced baseline points away
  from the parallel stack; one that appears only after a feature is enabled points at that feature. A
  passing reduced run is a localization result, never the parity verdict.
- Run one logical update's complete forward/backward and gradient accumulation. Measure loss and gradients
  **before any optimizer update**, after required synchronization and loss-scaler unscaling, before clipping.
  “First step” denotes this computation, not a particular printed `step=0` or `Step=1`.
- Complete pending consumer-side input/context checks before judging parity. Compare finite loss values
  and loss components under the same normalization. Require finite gradients/norms, and compare global
  pre-clip grad norms using the same unique parameter population and distributed accounting.
- Treat loss and total norm as screens, not proofs of gradient equality. For a gradient-parity claim,
  also compare mapped gradient tensors (elementwise error, relative L2, cosine as appropriate); state
  coverage if only selected tensors fit the budget. A norm-only result must be labeled norm-only.
- Keep stochastic behavior equivalent where possible; equal seeds alone do not guarantee equal dropout
  masks or RNG consumption across frameworks. If this prevents a pointwise verdict, report the limitation
  and design a controlled probe rather than silently changing the production configuration.

**Output / gate:** a first-step metric table, tensor coverage and pass/fail/unverified against the declared
bounds. On failure, use the routing table below and re-run this gate after the fix. Do not launch a long
alignment run to hide an unresolved first-step discrepancy; retain failed logs as diagnostic evidence.

## Stage 2: compare multi-step training

**Input:** a passed first-step gate at the declared evidence level.

1. **Short-horizon update check:** run a predeclared short pair (for example, about 20 updates) from the
   same initial state and input sequence. Compare per-update loss and pre-clip grad norm, actual applied lr,
   and selected gradients/parameter updates where required. This catches optimizer, clipping, scheduler,
   accumulation and state differences that an update-free first step cannot expose.
2. **Long-horizon comparison:** only after the short check passes, run the agreed horizon (for example,
   a few hundred updates). Collect **loss, grad norm, performance and memory**. Keep pointwise comparisons
   where the input/update contract supports them; add predeclared statistics such as spike counts and
   envelopes. Long trajectories do not have to diverge. Do not dismiss a new gap as “chaos” without checking
   inputs, lr, states and repeatability, or replace a failed pointwise criterion with statistics after the fact.
3. **Performance accounting:** distinguish a controlled matched-stack comparison from a native-stack
   benchmark. Record backend/topology/checkpointing/offload differences in the latter; do not attribute
   the whole gap to the framework alone. Separate startup/compilation from steady-state timing and declare
   the token denominator, memory unit, allocated/reserved/device metric, and measured window.

**Output / gate:** aligned per-update records and plots, criterion results, explained differences and open
items. A short-run pass is not long-run validation; a native-pipeline trend match is not numerical parity.
If the requested scope ends at the short check, stop there and state that limit.

## Diagnose differences: decompose first, hypothesize second

This table chooses the **next observation**, not a root cause. Confirm an example's applicability in the
current implementation before testing it; no listed example is required for every model.

| Symptom | First decomposition/check | Conditional example and confirmation |
|---|---|---|
| First-step loss differs | Recheck effective inputs and loss terms; locate the earliest mismatch in embeddings → layer outputs → logits → loss | A replay adapter may shift labels twice: compare labels at the loss input |
| Loss is close, grad norm differs | Audit norm/scaling/sync accounting; decompose gradients by mapped parameter group, then inspect backward boundaries | If fused CE is used, test hidden-state-gradient accumulation against a reference; a matching head norm alone does not prove this cause |
| Norms match but gradient/update checks fail | Compare gradient elements/directions and mapped parameter updates | Equal magnitudes can hide sign, layout or parameter-mapping errors |
| First step passes, later steps drift or loss stalls | Find the earliest differing update: inputs, applied lr, clipping, optimizer state and weight deltas | Check total-step-dependent warmup or low-precision update rounding only if that path is present |
| Spikes differ or cannot be reproduced | Align input/update windows, decompose loss/gradient contributors, check repeats within budget | A parameter group dominating a spike is a propagation location, not necessarily its cause; no observed spike means “not reproduced in this scope” |
| Performance or memory differs | Split timing by compute/communication/optimizer/data/recompute; memory by state/activations/workspace/allocator | Compare backend and cache state before using traces to isolate a bottleneck; grad-norm decomposition is not a performance diagnosis |

Test a specific hypothesis with one-variable A/B comparisons at the earliest discrepant boundary.
Document supporting and contradicting evidence, exclusions and retracted hypotheses. See diagnostics
for input replay, gradient accounting, controlled experiments and worked examples.

## Reusable helpers

- `scripts/compare_step_metrics.py` pairs per-update metrics from two text logs or JSONL trackers into
  `<prefix>.csv` and `<prefix>.json` (diff = A − B), reporting unmatched updates and duplicates. The CSV
  columns are `update, a_<metric>, b_<metric>, diff_<metric>` per metric; the JSON nests the same values
  under `metrics`. Supply
  regexes/keys and offsets derived from source; the script never shifts curves until they overlap.
  `--atol/--rtol` encode the predeclared criterion and set the exit code; without them it reports
  differences only. `--plot` is optional and requires matplotlib. Run it from the repository root, and
  validate each regex against real log lines first.
- Self-test with synthetic logs, no training run needed:
  `python .claude/skills/align-automodel-xtuner-text/scripts/test_compare_step_metrics.py`.
- Deliver the result with [the report template](references/report-template.md).

## Report

Lead with the verdict level actually reached; never let a first-step screen be retold as full alignment.

| Verdict | What was demonstrated |
|---|---|
| First-step, norm-only | Same-input first-step loss and global pre-clip grad norm within declared bounds; gradient vectors not compared |
| First-step, tensor-level | The above plus mapped gradient tensors, with their coverage stated |
| Short horizon | The above plus the declared short update pair (loss, grad norm, applied lr, updates as required) |
| Long horizon | The above plus the agreed long run, with pointwise criteria and/or predeclared statistics |
| Native-stack benchmark only | Performance/memory or trend comparison without same-input parity; answers a different question, not a higher level |

Then include this checklist, marking anything not done as **not verified** instead of omitting it:

- Contract with default/configured/runtime-effective values, plus the items still unresolved.
- Actual input evidence: tensor or canonical-hash comparison and its coverage, not counts alone.
- First-step table: loss and components, global pre-clip grad norm, declared bounds, pass/fail/unverified.
- Gradient evidence type (norm-only or mapped tensors) and tensor coverage.
- Horizons actually run, per-update records and plots, with units, timing windows and rank aggregation.
- Source and run identities, an attempt ledger (change → observation → next decision), failures and open items.

Save per-step data and line plots with setup-specific names; derive step/lr mappings from source in the
parser, not by shifting curves until they overlap. Keep restricted data and private run details out of
public reports and this reusable skill.

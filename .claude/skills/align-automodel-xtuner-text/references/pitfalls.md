# Pitfalls: read before configuring or running

P1–P5 group checks by category, not workflow stage. Each entry states the relevant condition, possible
failure and evidence to collect. These are diagnostic candidates, not a model-independent list of causes.
Mark an inapplicable mechanism as not applicable instead of enabling it to follow a historical recipe.

## P1 Environment and model setup

**P1.1 Low-precision optimizer updates.** If updates are written directly into bf16/fp16 parameters,
small increments may round away. Inspect actual parameter shards, master weights, optimizer states and
weight deltas; verify adequate update precision on both sides. Do not assume a framework always has fp32
master semantics, or infer this cause solely from flat loss.

**P1.2 Scheduler names are not semantics.** Identically named schedulers may implement different schedules.
Compare applied lr sequences and warmup. Constant lr can simplify an initial diagnostic pair, but is not
mandatory when another schedule is correctly matched or required to reproduce the target workload.

**P1.3 Dependency/build mismatch.** Record commits or reproducible build identifiers and loaded library
paths, not just package names. Snapshots deployed without `.git` (tar/rsync copies) have no revision to
log; fall back to the commit hash embedded in the deployment path plus hashes of the critical files you
verified. For missing symbols or CUDA build failures, check the actual call site,
framework version requirements, CUDA/compiler compatibility and library resolution before changing the environment.

**P1.4 Wrong function inspected.** A package version or sibling function can conceal a missing argument.
Copy the actual call-site import and inspect that callable and its implementation; use a minimal execution
probe if introspection is unavailable. Confirm the option is passed and takes effect. Signature support
alone does not prove numerical equivalence, and a compatibility workaround must not silently drop a precision option.

**P1.5 Import success is not backend coverage.** Optional dependencies may fail only when a selected
backend runs. Exercise its forward/backward path in the first-step probe; do not substitute a different
backend without recording the change.

**P1.6 Config identity is not loaded-model identity.** Verify checkpoint content, load reports, mapped
tensors and buffers. Nested options, missing keys or freshly initialized modules can change the computation.
When reducing layers, update dependent layer tables and check the loader's handling of extra weights;
do not assume changing a layer count alone is sufficient. Verify the resolved model path in each run.

**P1.7 Dataset readers may filter differently.** Compare accepted record formats, filtering, truncation
and template rendering, then compare actual consumed tensors. Equal source row counts are only a preliminary check.

**P1.8 Structural switches are not interchangeable.** Wrapper layouts, tying, sharding and dispatchers
may impose different constraints even for text-only execution. Verify supported paths and preserve intended
semantics; do not toggle a structural option just because its name resembles an option on the other side.

**P1.9 Sparse paths need representative inputs.** If the target model uses sparse attention, check actual
per-document lengths, selected-token coverage and executed kernels. A long pack alone does not imply long
documents or sparse execution. Do not generalize one implementation's top-k threshold or KV layout.

## P2 Comparison contract

**P2.1 Missing effective values.** Track model/data/update/numerics items as default, configured and
runtime-effective values, with evidence. Include auxiliary losses, frozen/tied parameters and quantization
state where present. Optimizer parameter-group defaults are version-sensitive (per-group lr fields,
flatten lr-ratio special cases); diff the constructed param groups across checkouts instead of trusting
that one config name resolves identically. A config-only assumption is unverified.

**P2.2 Warmup can depend on the horizon.** Changing total steps may change warmup or decay, making later
updates incomparable. For shared-prefix comparisons, preserve the intended applied schedule on both sides.
Only the update-free first computation is unaffected by lr itself; more steps remain comparable if their
input, schedule and state histories match. Check ratio rounding and constructor constraints as well.

**P2.3 Native stacks versus controlled stacks.** Different kernels, dtypes, topology, offload or checkpointing
can legitimately affect speed/memory. Match these for controlled attribution; otherwise label a native-stack
benchmark and its differences. Neither a fast run nor matching endpoint loss proves backend gradient parity.

**P2.4 Log step/lr phases vary.** Inspect optimizer, scheduler and logger order, including resume and skipped
updates. Map records to the same logical update and actual applied lr. A zero/one-based offset is one
possible implementation, not a universal rule. Never shift values merely to make curves coincide.

**P2.5 Aggregate data statistics are weak evidence.** Equal pack counts, label totals or attention ratios
can accompany different tokens or ordering. Compare effective input/label/context contents. A small loss
gap does not establish that every remaining difference comes from a kernel.

**P2.6 Native packing is not a parity barrier.** Use a verified common stream or prove independently
constructed effective batches equivalent for pointwise checks. If not possible, report limited descriptive
or statistical results; do not claim same-input numerical parity or assume sorting has no training effect.

**P2.7 Units and windows matter.** Verify GB versus GiB rather than assuming the label, and distinguish
allocated, reserved and device-level memory. Define peak-reset timing and rank aggregation. Use matching
timing boundaries; timestamp deltas may be rounded or include logging/data waits.

**P2.8 Do not compensate one side to fit the result.** Match intended update semantics and fix real setup
errors transparently. Do not change only one warmup or tune acceptance thresholds after a failed comparison.
Record corrected configurations and run a fresh matched comparison under the revised, justified contract.

## P3 Gradients and attribution

**P3.1 Forward parity can hide backward error.** If a fused/chunked CE implementation exposes independent
hidden-state and classifier-gradient accumulation paths, inspect both. A precision flag such as
`accum_e_fp32` is relevant only where supported and actually used; verify its effect against a trusted
backward reference. Do not prescribe that flag to every CE kernel or model.

**P3.2 Norm fingerprints are clues.** Similar head norms with different upstream norms can motivate a
loss-backward investigation, but do not prove it. Equal norms can hide different gradient vectors. Group
contributions to the difference of squared norms are not the squared norm of the gradient difference.

**P3.3 Parameter grouping can lie.** Canonicalize wrapper prefixes, names, axes and fused/split layouts;
account for tied tensors once. Model-specific patterns such as `mlp.gate.` can avoid confusing a router
with `mlp.gate_proj`, but must be verified on the current model. Audit unmatched and multiply matched entries.

**P3.4 Self-check the measurement.** For a disjoint unique-parameter L2 grouping, sum of group norm squares
must reproduce the intended total norm square to justified arithmetic precision. Verify synchronization,
unscaling, population and dtype; explain any disagreement with the official metric rather than allowing an
arbitrary percentage. A matching total is necessary, not proof that every group mapping is correct.

**P3.5 First-step success is limited.** No optimizer update has yet influenced the first computation.
Follow with short-horizon gradient/norm and loss checks, and inspect parameter/state updates when claiming
update parity. Norm-only coverage does not become tensor-level gradient coverage by running more steps.

**P3.6 Distributed norm accounting.** Inspect mesh axes, Shard/Replicate/Partial placements, expert versus
shared tensors, tied parameters and reduction completion. Do not blindly all-reduce local squared norms or
change EP to match a scalar: replicas may be double-counted and partial gradients may not yet be final.

**P3.7 Dominant groups do not identify the cause.** A residual, router or attention group can amplify an
upstream discrepancy. Inspect adjacent forward/backward boundaries and test a causal change. Similar spikes
across switches, or no spikes in a limited sweep, do not prove a benign structural property.

## P4 Common input replay

**P4.1 Double label shift.** Determine whether the source dump has already shifted targets. Inspect every
receiving dataset/collator/model/loss transformation and shift exactly once in the combined pipeline.
Use an adapter that preserves labels and reconstructs context; a native collator is usable only if verified.

**P4.2 Loss normalization spans the engine.** Check the entire accumulation/reduction path, not just a
`token` reduction option. Account for valid-token counts or weights, DP averaging, CP partitioning and
auxiliary terms. Per-microbatch means are not generally a global token-weighted mean (diagnostics §2).

**P4.3 Check content before GPU.** Compare all planned replay batches on CPU where possible, including
labels and context, and cross-check the consumer entry point. Label-count equality alone cannot prove
order. For tensors created on device, add a first-step consumer probe; unresolved fields block a parity verdict.

**P4.4 Record replay identity.** Record schema, shapes/dtypes, canonical hashes, document/sample order,
logical-update/microbatch/rank assignment and allowed artifact paths, plus the step/lr-phase mapping
between the dump's step numbering and each pipeline's logs (see P2.4). A file hash identifies that file,
not the tensors a different pipeline eventually consumes. Respect data permissions; no unauthorized copying.

**P4.5 Repacking needs revalidation.** Preserve document boundaries, masking, positions, per-update token
membership and gradient weighting. Even then kernel shapes/reduction order may change. Re-run input and
first-step/short checks for the changed pair; reuse an old control only with a documented matching contract.
Do not assume repacking is numerically neutral.

**P4.6 Replay adapters and configs are pinned to a code version.** Collator/sequence-context fields and
optimizer/attention config fields get added, renamed or defaulted differently across snapshots of the same
framework; a nested `hasattr` guard does not protect a sibling field assigned inside its block. Minimum
re-validation on a new checkout: adapter import, pack-order smoke, **and executing the actual config file**
(e.g. `Config.fromfile`). An import-plus-order smoke can pass while a config `setattr` still crashes on a
missing field, or silently targets different semantics.

## P5 Execution and evidence

**P5.1 Launch and completion are separate checks.** Use the approved scheduler/launcher and unique run
paths; audit hardcoded `tee` targets before reuse and never overwrite old evidence. On SSH systems requiring
session detachment, use the approved `setsid` approach rather than `nohup`; neither defeats scheduler or
container termination. Follow concurrency limits. Completion requires expected records, successful exit
status and no remaining workers, not just process disappearance or a success string echoed by shell tracing.

**P5.2 Progress bars and duplicate logs.** Normalize carriage returns before parsing, then validate regexes
against actual lines. Select the intended rank and distinguish progress updates from completed-step records.
Check missing/duplicate steps and log precision; do not infer fine timing from rounded timestamps.

**P5.3 Hooks must execute in the workers.** Prefer existing supported diagnostics after verifying their
availability. Parent-process patches may not reach spawned workers. If temporary worker instrumentation is
needed, verify it loads once, has no semantic side effects, and is scoped/removed after the probe; do not
assume a historical environment variable or `sitecustomize.py` hook exists in another checkout.

**P5.4 Queues and caches.** Predeclare each requested arm's horizon and stop conditions. Run both named
arms unless the user changes scope or a safety/failure stop intervenes. Freeze source per run; compare
cache/compile warmup states rather than requiring two frameworks to share a directory. Before cancelling,
stop only the experiment's watcher/driver chain before its workers to prevent automatic relaunch.

**P5.5 Waiting and reporting.** Use short status checks or background notifications, not blocking sleep
loops. Estimate remaining time from observed startup and steady-state costs. Report partial, failed and
unverified runs explicitly; a skill edit or successful import is not end-to-end training validation.

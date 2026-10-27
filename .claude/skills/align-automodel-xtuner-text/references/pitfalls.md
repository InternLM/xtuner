# Pitfall list (read through before starting)

IDs P1–P5 map to the five stages in SKILL.md. Each entry = symptom → cause → avoidance/fix.
Entries marked ※ were hit in more than one model generation.

## P1 Bring-up and environment

**P1.1 ※ bf16 parameters + AdamW without an fp32 master → loss never decreases** (flat line near init).
lr×grad is truncated by bf16 rounding; weights barely move. Fix on the AutoModel side: fused AdamW with
`master_weights: true` (fp32 master). XTuner's `fully_shard` shards parameters in fp32 natively, so it does
not have this trap — the two sides reach "fp32-master semantics" by different roads. Always audit both
sides' optimizer-precision semantics (sharded dtype / master / gradient-reduce dtype) before comparing.

**P1.2 ※ lr schedule must be constant**: one framework's `cosine` may truly decay while the other's
`cosine` config actually stays constant. If unsynchronized, loss/grad-norm shapes are incomparable.
Use constant + identical warmup for every comparison run.

**P1.3 Pin kernels/dependencies by commit, not by package name**:
- A fused-attention kernel repo may need a specific commit (e.g. a commit that adds a required argument
  like `indexer_topk`); HEAD may lack it → unexpected-keyword errors.
- When verifying "commit X provides argument Y", read the call site to find the function your code
  **actually invokes**, then check that signature — sibling entry points in the same package expose
  different arguments (one forward function had `indexer_topk` while the cache-path variant did not);
  checking the wrong symbol yields a false "argument missing".
- Compiling CUDA extensions often needs matching CCCL include paths (`CPLUS_INCLUDE_PATH=$CUDA_HOME/include/cccl`).
- Multi-package stacks (e.g. transformer-engine) need all components pinned; a missing meta-package fails
  only at the sanity check.
- Runtime `LD_LIBRARY_PATH` ordering matters (an older nccl shadowing the build → undefined symbol at init).
- An older torch cannot run a newer framework main (missing symbols/classes) — check the floor version first.

**P1.4 Same version number ≠ same parameter signature**: a PyPI release and a github-main build can both
report the same version while one lacks a keyword argument (observed with a cross-entropy library whose
`accum_e_fp32` existed only in one build). For any numerics-affecting package, verify support with
`inspect.signature`, never with the version string. **Symbols also drift or collide inside a package**
(observed after an upgrade): a submodule can share a name with a function (`cce.cce` is a module), and
the call site's import path can differ from the impl module's name (top-level `linear_cross_entropy`
vs submodule `cce_linear_cross_entropy`). Correct procedure: read the import line at the call site,
**import exactly that symbol, then inspect it** — do not guess names (both a guessed function name and
a guessed submodule path failed in practice).

**P1.5 Soft-imported dependencies explode only at runtime** (try/except import prints a warning; the MoE
backend reaches the op and fails). A smoke test must cover the target backend path, not just `import`.

**P1.6 Single source of truth for weights; specs hide in nested config**: both frameworks read the same
directory; options like `num_nextn_predict_layers` may live under a nested `text_config` (setting it at the
top level silently does nothing). Reduced-layer models: change `num_hidden_layers` **and** truncate the
per-layer type lists and recompute the linear/full attention layer tables; weight files do not need
truncation (loading side takes the first N layers). **Before any rerun, verify the actual model path from
the run log (e.g. the set -x lines)** — a script template silently pointing at an old reduced model once
produced a fake "memory anomaly" of tens of GiB.

**P1.7 Dataset format asymmetry**: one side's dataset class may only accept `{"messages": [...]}` wrapped
records (bare lists raise AttributeError) while the other eats raw shards and self-filters empty samples —
reconcile row counts and empty-sample handling on the same source data first.

**P1.8 Structural switches**: with a nested (e.g. VLM-composed) model, disabling outer-model FSDP wrapping
can hit a lazy-init conflict (`FSDP state has already been lazily initialized for ...`) — structurally not
turnable-off; the other framework's flat text class has no such concept. Pick dispatchers that are actually
built and loadable on both sides.

**P1.9 Measuring sparse-attention gains needs pack > topk** plus the kernel path that absorbs the KV cache;
below topk everything is selected and an sdpa-style path expands the full cache — no sparsity benefit shows.

## P2 Comparison setup

**P2.1 Alignment checklist, item by item**: parameters/backends/bit-widths/gradient-reduce dtype — align
what can be aligned; anything not aligned must be listed explicitly (✅/🟡/❌+reason). Never leave a
difference unmentioned.

**P2.2 Warmup scales with total steps**: `warmup_ratio × total_steps` — two runs with the same config but
different total steps have different early lr (100-step run → warmup 10; 200-step → warmup 20); the
divergence can be mistaken for non-determinism. For cross-step-count comparisons: fix the absolute warmup,
or compare only s1 (before any update, lr-independent). Shrinking only `max_steps` while keeping
`lr_warmup_steps` can trip an assert; shrink proportionally or pin `lr_decay_steps`.

**P2.3 Align backend stacks before comparing memory/throughput**: linear/norm/experts/attention backends
item by item, otherwise memory gaps are backend artifacts (torch linear + fp32-norm vs fused kernels once
produced a spurious ~2 GiB gap that vanished after aligning).

**P2.4 Step numbering and lr logging differ by one phase**: framework A logs lr after the step (0-based),
framework B before the step (1-based). Under the mapping A-step k ↔ B-step k+1 the actually-applied lr
sequences are identical. Do the alignment in code (shift the logged lr by one), not in figure captions.

**P2.5 Quantify token-stream microdifferences**: chat-template differences (e.g. one side appends one
loss-counting eos per sample), data-layer shift dropping the last token, differing empty-sample filters.
Reconcile via: pack-count closure + label-token totals + first-step loss Δ on the same stream converging
to ~1e-3 (pure kernel noise).

**P2.6 When packing algorithms cannot be aligned, drop the criterion level**: organization differences
(random shuffle vs length-grouped packing) → compare trends/statistics only; any pointwise verdict must go
through Stage 4 (same pack stream). Length sorting by itself typically has no material effect (verified by
ablation) — do not chase it.

**P2.7 Memory and time accounting**: "GB" on both sides is usually GiB (÷1024³) — directly comparable;
separate torch max-allocated vs reserved; per-step time from timestamp deltas is second-quantized (~4.5 s
real shows as 4/5/6 scatter) while the other framework may log exact times — state the accounting before
comparing.

**P2.8 Short runs keep both sides in normal config; never compensate one side**: for a 20-step comparison,
run both sides with normal configs (warmup scales automatically); do not tweak one side's warmup to "fix a
phase difference". Verify the logging phase from source first. The principle: **identical configs →
identical results**, not "align until results match".

## P3 Gradients and cross entropy

**P3.1 ※ Loss parity ≠ gradient parity**: when the CE hidden-state gradient (e-grad) accumulates in bf16
atomics, loss differs only at ~1e-5 while the first-step grad norm is systematically 10–15% low. The
first-step pre-clip grad norm is the most sensitive probe. CE kernels of the chunked/fused-linear family
need explicit fp32 accumulation for e-grad (e.g. `accum_e_fp32=True`); the flag affects e-grad only, not
the weight gradient (c-grad).

**P3.2 Per-parameter ratio fingerprints**: decompose first-step grad norms per parameter. Fingerprint:
`lm_head` exactly 1.000× while everything else is off → the discrepancy lives on the CE e-grad path (head
weights receive c-grad, unaffected). Self-check: sum of per-group squares == total norm square.

**P3.3 Three regex traps when grouping parameters**: names may carry a `_checkpoint_wrapped_module.`
prefix (AC wrapper) and a mid-path module prefix — strip before anchoring `layers\.(\d+)\.`; match router
params with a trailing-dot prefix (`mlp.gate.`) or a dense `mlp.gate_proj` gets misclassified as router; if
your first grouping dumps everything into `.other`, this is why.

**P3.4 Decomposition self-check**: the per-component total must match the framework's official grad_norm
pointwise (within ~1%) before any conclusion.

**P3.5 A single aligned point can be coincidence**: step-0 probes matching while a later step is tens of
percent off (with hidden states unmatched) has happened. Gradient-parity evidence = single-step ratio +
first-few-step ratios + N-step same-schedule per-step loss Δ; never conclude from one point.

**P3.6 Grad-norm statistics under EP**: one side's total-norm computation may skip an all-reduce over the
replicate dimension at certain EP degrees (per-rank values differ); confirm both sides use the same
statistical population before comparing numbers.

**P3.7 Diagnose spikes/jumps with per-component decomposition**: residual-stream mixing parameter groups
can dominate the norm and oscillate together at optimization phase transitions — easily misattributed to
MoE routing (router norms stay orders of magnitude smaller). Cross-check by sweeping pack length / EP /
data composition: a "constant fingerprint" (same window, same height, invariant to every switchable knob)
indicates a structural property, not a config defect.

## P4 Same pack stream

**P4.1 Prevent double shift**: a dumped pack stream is usually already document-wise pre-shifted
(input=t[:-1], label=t[1:]); the receiving framework's native collator will shift again — the loss curve
becomes globally wrong. Feed via a dedicated collator that rebuilds the sequence context directly.

**P4.2 Verify loss-normalization equivalence**: one side passing a global `num_label_tokens`, the other
using token-weighted reduction (`loss_reduction="token"`) — confirm equivalence before pointwise
comparison.

**P4.3 CPU full-pass check before GPU**: verify per-step `num_label_tokens` against the source log for
every step (proves offline pack order == production batch order); then a ~10-step GPU smoke (first-step
loss Δ, no NaN/OOM) before the full run.

**P4.4 Record the dump identity**: shapes/keys/label totals/SHA256. To switch pack length later, merge
losslessly in document order (strip tail padding → concatenate → re-pad); document boundaries preserved.

**P4.5 Packing shape is numerically neutral**: re-packing the same token stream at a different length
(e.g. 8×64K → 1×512K) reproduces the loss trajectory pointwise (max|Δ| ~0.001) — switching the pack
length does not require rerunning the control arm, provided sequence-parallel splitting and attention
chunking are self-consistent at the longer pack.

## P5 Remote execution and logs

**P5.1 ※ Long runs need setsid; never nohup**: when the ssh session drops, SIGTERM kills the whole process
group and nohup does not stop it (verified by a killed multi-hour run). `tee` truncates old logs — archive
before restart. **Before reusing a historical run script, check whether its log path is hardcoded** (a
pinned tee target): parameterize it (e.g. a `LOGTAG` env var) or copy the script with a new tee target —
a new run always writes a new log name and never touches the old files. One training per machine/pod at a
time. When stdout goes to /dev/null, judge completion by "N/N step lines + process exited". Note log
timezones (often UTC).

**P5.2 Log parsing**: tqdm progress bars glue many logical steps into one physical line with `\r` — run
`tr "\r" "\n"` first; a line-start `^` anchor then matches only every ~10th step (bogus dt); derive
per-step time from adjacent timestamp deltas on frameworks that lack a time field.

**P5.3 Cross-process hooks go through sitecustomize**: the training script runs as `__main__`, so patching
classes from outside the process has no effect; inject diagnostics via `sitecustomize.py` (PYTHONPATH) or
the framework's env-gated debug switches.

**P5.4 Experiment queue discipline**: the criterion arm runs first; an arm stops once it passes its
criterion step count; paired arms named by the requester both run; A/B arms share one source directory
(a cold Dynamo cache on the first arm produces fake 87–125 s first steps); when changing direction kill the
watcher first, then the driver, then the training process.

**P5.5 Do not block while waiting**: poll once briefly (grep progress + ps) or use a background watcher;
when asked for status, give progress + remaining-time estimate.

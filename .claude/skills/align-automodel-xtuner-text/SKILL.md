---
name: align-automodel-xtuner-text
description: Align text-only training accuracy between AutoModel (NVIDIA NeMo Automodel) and XTuner after onboarding a new model. Use when validating that both frameworks produce matching loss/grad-norm on the same model and data, attributing cross-framework loss or grad-norm discrepancies, or re-verifying a numerical fix. Covers alignment checklist, five-stage workflow (smoke → trend compare → localization → same-pack-stream pointwise check → gradient-level re-verify) and a pitfall list to read before running anything.
---

# Align AutoModel ↔ XTuner Text Training Accuracy

Turn "both frameworks compute the same thing" into a falsifiable loop: same-setup comparison → layer-by-layer attribution → same-data pointwise check → gradient-level re-verification.
Reuse the methodology and the pitfalls; do not reuse model-specific defaults, switches, or benefit numbers from previous models.

## Step 0: read the pitfalls first (required, before doing anything)

Before writing any config or launching any run, read [references/pitfalls.md](references/pitfalls.md) (grouped P1–P5 by stage).
The P1 (bring-up environment) and P2 (comparison setup) pitfalls decide whether the *first* round of experiments is even valid.
Check experiment designs against [references/experiment-discipline.md](references/experiment-discipline.md); for instrumentation, decomposition and log parsing read [references/diagnostics.md](references/diagnostics.md).

## Core principles

1. **Fully align what can be aligned**: weight directory, data, chat template, global batch size, step count, lr, optimizer semantics (fp32 master), EP degree, dispatcher, activation checkpointing.
2. **Match what cannot be made bitwise-identical**: per-framework kernel paths (e.g. attention backends) — first prove equivalence inside one framework (endpoint-loss match between two backends), then let each side use its own.
3. **Keep framework-native behavior**: things like packing algorithms cannot be aligned; keep each implementation and record the difference and its impact.
4. **Do not force-align**: never change one side's normal path just to match the other; document instead.

Write every item into an ✅/🟡/❌ checklist (❌ must state the reason and expected impact — never leave a difference silently).
Track three layers of truth per knob: **library default / config value / runtime-effective value**; verify the third from logs and source, not from the config file.

## Acceptance criteria (weakest → strongest; declare the level up front)

| Level | Criterion | Typical magnitude |
|---|---|---|
| Forward parity | first-step loss Δ on identical input | ≤ 0.003 (same pack stream: 0.002) |
| Gradient parity | first-step pre-clip grad-norm ratio | within 0.5% (typical replays: 0.04–0.15%) |
| Short-horizon trajectory | per-step loss Δ over ~20 steps, same schedule | ≤ 0.003 (observed 0.0005–0.0022) |
| Trend parity | 200+ steps: curve shape / transition windows | same windows, same shape |
| Long-horizon statistics | spike counts, envelopes, statistics only | pointwise pairing diverges by design |

Magnitudes are **typical observations from past replays, not hard thresholds**: replays across
environment stacks (torch/triton/kernel versions) carry ~±0.2% noise. When a single-step ratio drifts
slightly past the typical range, first check whether the loss trajectory still matches pointwise before
calling it misalignment (observed case: one step's gn ratio 0.9985 while the 20-step loss deltas
reproduced to the 4th decimal — stack noise, not divergence).

**Loss parity ≠ gradient parity** (P3.1): a numerical-path verdict requires a gradient-level criterion; the first-step pre-clip grad norm is the most sensitive probe.

## Five-stage workflow

Proceed in order; each stage has its own gate, do not advance until it passes.

### Stage 1: bring-up and setup freeze

- Bring one side up on a reduced-layer/small model, then the other; both frameworks read the **same single weight directory** (including any MTP-stripped variant).
- Freeze the environment: conda env, kernel commits, dependency builds and their **parameter signatures** (not version numbers, P1.4), `LD_LIBRARY_PATH`.
- Fill the ✅/🟡/❌ checklist; pre-clear the recurring traps: fp32 master (P1.1), constant lr (P1.2), pinned kernel commits (P1.3).
- Output: one ~20-step smoke log per side + the alignment checklist.

### Stage 2: same-setup trend comparison

- ~200 steps; extract four metrics per step: loss, grad_norm, s/step, peak memory.
- Align backend stacks before comparing memory/throughput (P2.3), otherwise conclusions are artifacts.
- Classify every gap as: accounting difference / implementation fingerprint / real bug — then enter Stage 3.

### Stage 3: discrepancy localization (decompose first, hypothesize second)

- **Per-parameter-group grad-norm decomposition before any guessing**: whichever group dominates the squared difference identifies the path to inspect.
- Single-variable A/B to eliminate switchable items (backends, EP, pack length, precision stack); one variable per arm.
- Discrimination matrix: one entry script, same data, flip exactly one suspect variable (see references/diagnostics.md).
- Record excluded candidates and retracted hypotheses — they prevent repeated work later.

### Stage 4: same pack stream, pointwise verdict

- Dump the pack sequence one side actually consumed (keys/shapes/label totals/SHA), write an adapter, feed the other side.
- **Prevent double shift** (P4.1): pre-shifted labels must bypass the native collator; verify loss normalization equivalence (P4.2).
- CPU full-pass check: per-step `num_label_tokens` equal to the source log → offline pack order = production batch order; then a ~10-step GPU smoke.
- After this stage, first-step loss Δ is pure kernel-level numeric difference; per-step Δ no longer contains data-organization noise.

### Stage 5: gradient-level re-verify

- First-step pre-clip grad norm, per-parameter decomposition, with the self-check "sum of group squares == total norm square".
- Fix acceptance triple: single-step ratio back in range; first-few-step ratios in range; ~20-step per-step loss Δ under a same warmup schedule.
- Step numbering and lr logging differ by one phase across frameworks (P2.4); align in code, not in figure captions.

## Reporting

- Conclusion first; every gap carries attribution and an evidence chain (list of exclusions); open items labeled as open.
- Tabulate failed attempts (change → failure point → progress → attribution).
- Figure names carry the setup (pack length, step count, arm names); after in-code step alignment, do not explain numbering in the figure.

## Resources

| What you are about to do | Read |
|---|---|
| Pitfall list (required before starting) | [references/pitfalls.md](references/pitfalls.md) |
| Rules for designing/executing comparison experiments | [references/experiment-discipline.md](references/experiment-discipline.md) |
| Instrumentation, decomposition, pack-stream tooling, log parsing | [references/diagnostics.md](references/diagnostics.md) |

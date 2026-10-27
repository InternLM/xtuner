# Diagnostics toolbox: instrumentation, decomposition, pack streams, log parsing

File/function anchors drift — re-locate the equivalents in both codebases before reusing the methods.

## 1. Per-component / per-parameter grad-norm decomposition

- Principle: at the trainer's clip-grad call site, compute pre-clip norms per parameter group with
  `max_norm=inf` (gradient untouched); you get the total plus each group's contribution.
- Instrumentation: prefer the framework's existing env-gated debug switches (grep the trainer for
  grad-norm logging / clip hooks); if absent, add a temporary env-gated dump via `sitecustomize.py`
  (P5.3) — a windowed hook (active only over target steps) keeps long runs cheap.
- Grouping regexes (P3.3): strip `_checkpoint_wrapped_module.` and module prefixes first; anchor
  `layers\.(\d+)\.`; router via trailing-dot prefix (`mlp.gate.`). Group names are asymmetric across
  frameworks (e.g. one side's "dense" bucket includes shared experts, the other files them under
  "other") — unify bucket definitions before cross-framework comparison.
- Self-check (P3.4): sum of group squares == total norm square; component total matches the official
  grad_norm pointwise within ~1%.

## 2. Discrimination matrix (experiment design for root-cause)

When multiple environments/code versions intersect, run one entry script on the same data and flip
exactly one suspect variable:

| Config | Env A | Env B | Reading |
|---|---|---|---|
| code X + env α | baseline | — | |
| code X + env β | same | flips | environment problem |
| code Y + env α | flips | — | code problem |

Design rules: minimum arms; one variable apart per arm; re-check the step index of any borrowed number
(a step-0 match can be coincidence, P3.5).

## 3. Same pack-stream toolchain (Stage 4)

- Dump: at the source side's collate/data entry, persist the packs each step actually consumed (keys:
  `input_ids/labels/seq_lens/steps/micro_batches_per_step`; int32; record SHA256 and label totals).
- Adapter: dedicated Dataset + collator on the target side — never through the native tokenize/shift path;
  use pre-shifted labels as-is (P4.1); rebuild the sequence context from `seq_lens` (num_tokens /
  cu_seq_lens / padding accounting, tail padding per the target framework's chunk rules).
- Engine: confirm the target treats N packs per step as N micro-batches; `loss_reduction="token"`
  equivalent to the source's global `num_label_tokens` (P4.2).
- Order of checks: CPU full-pass per-step `num_label_tokens` vs source log → ~10-step GPU smoke → full run.
- Re-packing: strip per-pack tail padding → concatenate in document order → re-pad to the target length;
  document boundaries preserved; numerically neutral (P4.5).

## 4. Log parsing and step alignment

- `tr "\r" "\n"` before extracting step lines; avoid `^`-anchored regexes (tqdm eats lines); per-step time
  from adjacent timestamp deltas on frameworks without a time field (second-quantized; use windowed
  mean/median).
- Step mapping (P2.4): framework-A step k (0-based) ↔ framework-B step k+1 (1-based); when plotting the
  applied lr, shift A's logged (post-step) lr forward by one so both curves coincide; keep the mapping in
  code.
- Accounting: memory "GB" == GiB; allocated vs reserved stated explicitly.
- Cross-process instrumentation via `sitecustomize.py` on PYTHONPATH (P5.3).

## 5. Plotting conventions

- Panels: loss / grad_norm / s-per-step / peak memory (+ tokens-per-second, lr), linear axes; x-axis
  unified 1-based after in-code alignment — no numbering explanations on the figure.
- Figure names carry the setup: pack length, step count, arm names; mixed setups label both sides.
- The per-step lr panel is the first tool against fake non-determinism (discipline 12); split committed vs
  uncommitted arms into separate figures.

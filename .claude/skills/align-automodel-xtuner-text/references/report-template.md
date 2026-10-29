# Text alignment report — <model>, <date>

Fill every section. Write **not verified** where something was not measured; do not delete the row.

## 1. Verdict

- Horizon: first step / short <N> updates / long <N> updates; outcome: pass / fail / not verified.
- Gradient coverage: norm-only / mapped tensors (<population and coverage>); update checks actually performed: <items>.
- Benchmark scope, if any: controlled matched stack / native stack; separate from the parity verdict.
- One-sentence result, including the direction of any remaining difference (A − B convention).
- Explicitly out of scope: <items>; keep failed gates failed even if later diagnostic steps were collected.

## 2. Comparison contract

| Item | A: default / configured / runtime-effective | B: default / configured / runtime-effective | Status | Evidence |
|---|---|---|---|---|
| Checkpoint and loaded tensors | | | | |
| Ordered effective batches, labels, context | | | | |
| Loss terms and global normalization | | | | |
| Accumulation, sync/unscale/clip, optimizer and lr | | | | |
| Dtypes, kernels, parallel mesh, checkpointing/offload | | | | |
| RNG / training mode | | | | |

## 3. Actual input evidence

- Method (tensor compare / canonical hashes), coverage (updates × microbatches × fields), result.
- Unverified fields and why (for example device-generated context checked only at the consumer).
- Input manifest identity: file/content hashes, schema, source revisions.

## 4. Acceptance criteria declared before measurement

| Metric | Formula and population | Bound | Horizon | Basis for the bound |
|---|---|---|---|---|

## 5. First-step result

| Metric | A | B | A − B | Bound | Pass / fail / not verified |
|---|---|---|---|---|---|

- Measurement point: before any optimizer update, after sync/unscale, before clip.
- Gradient evidence: norm-only or mapped tensors; tensor coverage.
- Consumer-side input/context probe result.

## 6. Multi-step result

- Short horizon: <N> updates; per-update loss / pre-clip grad norm / applied lr outcome; artifact links.
- Long horizon: <N> updates; pointwise criteria and predeclared statistics; artifact links.
- Comparison data: `<prefix>.csv`, `<prefix>.json`, plots; the exact script command lines used.

## 7. Performance and memory (if in scope)

| Metric | A | B | Unit, window, token denominator, rank aggregation | Controlled matched stack or native benchmark |
|---|---|---|---|---|

## 8. Attempts

| Attempt | Controlled change | Observation | Next decision |
|---|---|---|---|

## 9. Identities and open items

- Source revisions, local patches, environments, run/log paths for every arm.
- Failed criteria kept as failures; any revised criterion and the fresh pair validating it.
- Open items, retracted hypotheses, and what a reviewer should check next.
- Cause provenance for each claimed cause: independently found / externally informed / mixed / unknown
  (N/A while no candidate cause); state whether independent localization succeeded before any external
  answer appeared.

# Diagnostics: collect evidence before assigning a cause

Locate the current implementations in both checkouts. Names, configs and debug hooks from an older model
are search hints, not APIs guaranteed by this skill. Keep probes read-only with respect to training state
and limited to the comparison window; validate their accounting before trusting their output.

## 1. Freeze and verify effective inputs

1. Locate each data → collator → model → loss path. Record where tokenization, shifting, truncation,
   padding, sequence-context construction and loss weighting occur.
2. Capture what the source actually consumes. A replay manifest should associate each logical update,
   microbatch and data/sequence shard with ordered document IDs and the following semantic fields:

   | Field | Content to preserve or map |
   |---|---|
   | Inputs and targets | Token IDs, labels, valid/loss masks or weights, shapes and dtypes |
   | Sequence context | Document lengths/boundaries, positions, attention masks or an equivalent boundary representation, padding |
   | Assignment | Update/microbatch order, rank/shard mapping, accumulation membership |
   | Identity | Source/config/checkpoint identifiers, schema version, file and canonical content hashes |

3. Implement only the missing adapter. Do not retokenize or re-shift already prepared targets. Reconstruct
   the receiving framework's context with preserved document boundaries and position resets. Trace later
   transformations too: matching adapter output is insufficient if the model/loss shifts again.
4. Compare every planned replay batch on CPU where possible. Canonicalize supported layout/shard differences
   into logical sequences, preserving order and semantics; do not sort tokens or erase masks to get a match.
   Hash ordered field names, shapes/dtypes and tensor bytes under a specified canonical representation.
   Cross-check consumer-side values, including device-generated fields, in the first-step probe.
5. Fail on the first content/context mismatch and show its update, microbatch, field and offset. Report
   any unverified fields. Counts are useful checksums, but not identity proofs. Then run the first-step
   gate, followed by the declared short pair; no preliminary multi-update training is required.

**Toy counterexample:** `input_ids=[11,12]` and `[21,22]` can both have two supervised labels. Equal label
counts do not identify the stream. For document `[11,12,13]`, a pre-shifted replay is input `[11,12]`,
target `[12,13]`; shifting again makes the first prediction target `13` instead of `12`.

Load only trusted replay artifacts with the project's approved serialization policy. If replay storage is
not permitted, use authorized in-place comparisons; do not copy restricted training data to another machine.

## 2. Check loss and forward computation

First compare the input contract, loaded model state, training mode, and loss components. If those match,
compare embeddings → layer/block outputs → final normalization → logits → unreduced loss. Map layouts
before measuring absolute/relative errors. Narrow to the first discrepant boundary rather than assuming
that a large final loss difference starts in CE.

For a global token-mean objective, write down the intended equation:

`L = (sum of valid-token losses over the logical global update) / (number of valid tokens in that update)`.

Audit how each engine realizes it through microbatch accumulation, loss scaling and distributed reduction.
If the target objective uses sample weights or auxiliary terms, derive those terms separately; do not
force token mean on a different intended objective.

**Toy example:** one microbatch has two valid tokens with loss sum 4, another has six with sum 18.
The global token mean is `(4+18)/(2+6) = 2.75`, not `(4/2 + 18/6)/2 = 2.5`.
Under plain DDP averaging over D ranks, summing local loss contributions divided by the global token count
would introduce an extra factor `1/D`; a compensating factor is needed only if the engine has not already
provided it. Derive the actual DP/CP/EP behavior before adding any multiplier. Handle empty-supervision
batches explicitly and preserve collective participation.

## 3. Compare gradients and norm accounting

- Capture gradients after the logical update's accumulation, required reductions and unscaling, before
  clip/update/zeroing. Prefer an existing measurement API; do not call clipping repeatedly with infinity
  and assume it is side-effect-free. Audit hooks, collectives and DTensor dispatch if reusing an API.
- Map corresponding parameter names and tensor axes, including fused/split/transposed layouts. Identify
  tied/shared parameters and unused/frozen parameters. Distinguish `None` from a present zero gradient.
  Audit unmatched names and confirm disjoint groups on both sides.
- Construct the intended global gradient population once: count unique shards, do not duplicate replicas,
  and resolve partial reductions first. Respect the actual mesh and rank-symmetric collective order;
  do not all-reduce every local norm over the world group. Gather only selected mapped tensors when full
  materialization is too expensive, and report that coverage.
- For each disjoint group j, compute `S_j = sum(g_i²)` in a declared diagnostic accumulation dtype.
  Check `sum(S_j) = ||g||₂²` against an independently verified total to expected arithmetic precision.
  If the framework's logged metric differs, investigate population/timing/dtype before blaming a kernel.
- Rank groups by their signed contributions `S_j(A) - S_j(B)` to the difference of squared total norms.
  Also compare gradient vectors: these contributions can cancel, and are **not** `||g_A - g_B||₂²`.
  Measure max absolute error, relative L2 and/or cosine under the declared bounds. Near zero norms, use
  absolute bounds and report relative/cosine metrics as undefined where appropriate rather than dividing by zero.

**Toy counterexample:** `g_A=(3,4)` and `g_B=(-3,4)` both have norm 5, but their difference has norm 6
and cosine is `7/25`. Matching total or head norms cannot certify matching gradients.

If loss is close but upstream gradients differ, inspect backward boundaries starting at the loss input.
For a fused CE candidate, hold hidden states, classifier weights, targets and upstream loss scaling fixed;
compare both hidden-state and classifier gradients to a trusted reference. A low-precision accumulation
hypothesis requires this boundary evidence plus a controlled change, not merely a matching head norm.
For later-step drift, inspect the earliest differing applied lr, clip coefficient, optimizer state and
parameter delta (including master weights and quantization state where relevant).

## 4. Test a hypothesis with controlled contrasts

Use the smallest representative input that still triggers the discrepant path. Record all non-tested state.
If code and environment are both suspects, use this design only when needed and within the authorized budget:

| Arm | Code | Environment | Valid contrast |
|---|---|---|---|
| A | X | alpha | Reference arm |
| B | X | beta | A/B changes environment only |
| C | Y | alpha | A/C changes code only |
| D | Y | beta | C/D and B/D test whether the effect depends on the other factor |

An environment effect localizes to a bundle, not a specific dependency. Narrow further to one package,
flag or operation; compare its forward/backward outputs and repeat the first-step and short-horizon gates
when proposing a fix. Record null results and competing explanations; do not declare a cause from one scalar.

## 5. Decompose performance and memory separately

For speed, compare time windows for data preparation, forward/backward compute, communication, recomputation
and optimizer work. Use synchronized measurements or correctly interpreted traces; overlapping phases
cannot simply be summed. Separate cold compilation/loading from the predeclared steady-state window.
For memory, distinguish parameters/optimizer states, saved activations, temporary workspaces and allocator
reservation, using consistent rank aggregation and peak-reset windows. A hook that gathers gradients or
synchronizes each layer can distort both metrics; measure final performance without such instrumentation.

## 6. Parse logs and report the right updates

Normalize `\r` into logical lines, then test extraction against real examples. Select completed records
from the intended rank, check unique steps, missing values and restart boundaries. Preserve original
indices and derive a logical-update index from trainer order; verify resume, skipped updates and lr phases.
Do not assume an offset of one or infer applied lr solely from a post-scheduler log.

Plot aligned loss, pre-clip grad norm, step time and memory; add applied lr/throughput when useful. Declare
GB/GiB, allocated/reserved/device memory, timing boundaries, token denominator and rank aggregation.
Use linear axes by default and setup-specific filenames. Keep raw observations and criterion failures;
when logs are rounded, state that limit rather than claiming elementwise or bitwise reproducibility.

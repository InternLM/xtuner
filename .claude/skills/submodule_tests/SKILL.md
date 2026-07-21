---
name: submodule_tests
description: >
  Decide what tests a single XTuner submodule needs before it is trusted in a
  model. Use this when adding or reviewing a new operator, module, attention
  helper, weight-mapping, or multi-backend kernel under `xtuner/v1` — especially
  while porting a model (see `add_hf_model`), where each new piece must be
  validated in isolation before whole-model parity. Maps each kind of submodule
  to the specific tests it owns (forward/backward vs a readable reference, packed
  correctness, pad safety, backend agreement, distributed equivalence, weight
  loading), points at the canonical benchmark test for each, and says when a
  piece should be left to parity instead of unit-tested. Defines the three test
  intents and their base classes (`ParityTest` / `CorrectnessTest` / `SmokeTest`
  in `xtuner._testing`), the directory/file/class/test naming hierarchy, and the
  device/smoke pytest markers.
---

# What tests does this submodule need?

A submodule is any piece below the full model: an op/kernel (`xtuner/v1/ops`), a
module (`xtuner/v1/module`, e.g. a router, attention, indexer, compressor), an
attention/packing helper, a weight key-mapping, or a backend of a
multi-backend interface. This skill decides **which tests that piece owns**.

Decoder-layer bitwise parity against HuggingFace is a *different* layer and lives
in `add_hf_model` (§6–7). Parity runs the model down an **eager reference path**
(the `hf_parity` switch: plain aten ops, no fused kernels) and proves it matches
HF op-for-op — **forward and backward both**, because autograd differentiates that
eager forward into the same graph HF uses, so the gradients match bit-for-bit too.

What parity never touches is the **production path**: the fused/flash/triton/cutlass
kernels that actually run in training, the packed-varlen boundaries a single parity
sequence never straddles, the static-pad buffers, sharded execution. **A submodule
test earns its place by covering exactly that — what parity, running eager and
unpacked, cannot reach.** That is the test for whether to write one, not "would it
help localize a break". What parity does not reach:

- **the production kernel's forward AND backward** — parity exercises the eager
  anchor, not the fused/flash/triton/cutlass implementation that runs in training;
- **cross-sample isolation** in packed input — a parity batch may never straddle
  a `cu_seqlens` boundary in the revealing way;
- **padded / static-shape safety** — pad positions must not change values or leak
  gradient;
- **pieces with no HF counterpart** — custom routers, EP dispatch;
- **a module's own new numeric or masking logic** — validated where it lives.

The converse is the delete rule (last section): a thin composition of
already-tested parts that adds none of the above, and whose eager forward+backward
parity covers it end-to-end, needs no dedicated test. Being able to localize a
parity failure is a *benefit* of having these tests — never a reason, on its own,
to keep one.

### The three layers this splits into

- **op / kernel** (§1–6): full correctness — the math is right, right when packed,
  right at the padding boundary — **plus backward**, each production backend agreeing
  with the eager anchor. Exception: an op whose output is a discrete choice
  (top-k indices, argmax) has no differentiable backward to test; its correctness
  *is* "picks the right indices".
- **integrated assembly** (e.g. DSA as a whole): a **smoke** test only — forward and
  backward run across a few input shapes/backends without error. Correctness is
  already owned by the ops (§1–6) and by parity; the smoke guards wiring/shape
  regressions, and is *not* a stand-in for a missing op-level test.
- **parity** (decoder layer, `add_hf_model`): eager forward+backward == HF.

These three layers *are* the three test **intents**, and each has a base class in
`xtuner._testing` — read the next section before writing a class.

### Intent is the base class you extend

A reviewer must be able to tell what a test guarantees **without reading its body**.
That is what these base classes are for — the intent lives in the class you extend
and the check method you call, not in a comment. Pick exactly one per test class:

| Intent | Base class | Check method | Standard |
|---|---|---|---|
| **parity** — reproduce HF | `ParityTest` | `assert_parity` | strict `torch.equal`; **no tolerance** |
| **correctness** — match an oracle | `CorrectnessTest` | `assert_correct` | within class-level `atol`/`rtol` (fp32 default `1e-3`) |
| **smoke** — runs & stays finite | `SmokeTest` | `assert_smoke` | outputs finite; no oracle |

The standards are **deliberately not shared**: `assert_parity` is bitwise,
`assert_correct` reads the class tolerance, `assert_smoke` only checks finiteness.
Do not fold them into one parametrised helper — the separate types *are* the signal.
They are pure mixins, so a distributed test composes intent with the process base:
`class TestFooCorrectness(CorrectnessTest, DeterministicDDPTestCase)`. Determinism
(fixed seeds, deterministic collectives) still comes from `DeterministicDDPTestCase`;
plain single-process tests seed themselves as their benchmark files already do.

**"parity" means bitwise-against-HF, and nothing else.** A comparison that cannot
be bitwise is *not* parity, even when people call it that loosely:

- **backend agreement** (Triton vs. native, flash vs. native — §4), **padded-vs-native**
  equivalence (§3), and **distributed** equivalence (§5) all match an *equivalent*
  computation, not HF, and differ in the last bits (reduction order, rounding). They
  are **`CorrectnessTest`** with a tolerance — name such a class `…Correctness` or
  `…Equivalence`, never `…Parity`. If you inherited a `…Parity` class doing a
  tolerance compare between two backends, reclassify it.
- if a fused kernel only matches HF within bf16 rounding, it is correctness against
  the eager reference (§1), not parity.

**Where each axis lives** (so the `pytest` collection tree reads as a hierarchy on
its own):

- **directory** = subsystem — `tests/model` / `tests/module` / `tests/ops` / `tests/data` …
- **file** = the submodule under test — `test_indexer.py`.
- **class** = the intent — `TestIndexerCorrectness`, `TestIndexerSmoke`.
- **test name** = the contract — `test_topk_indices_in_range`. Name it for what it
  guarantees, never for what changed (`test_softmax_matches_formula`, not
  `test_softmax_now_works`).

Give every intent class a one-line docstring stating **what is under test**, **what
the oracle is**, and **why that check is enough** — the "why enough" line is where a
circular oracle (comparing an implementation against itself) becomes visible:

```python
class TestIndexerCorrectness(CorrectnessTest):
    """Under test: Indexer scores. Oracle: hand-written eager scoring. Enough
    because: the oracle is an independent formula, not the production kernel."""
```

**Markers** cover the *orthogonal* axes you actually select on — registered in
`pyproject.toml` under `[tool.pytest.ini_options]`:

- `@pytest.mark.gpu` / `@pytest.mark.npu` — device requirement; CPU-runnable tests
  carry no device marker. A CPU box runs `-m "not gpu"`.
- `@pytest.mark.smoke` — the one intent that is *also* a marker, because "run the
  fast subset" is a real selection (`-m smoke`); put it on the class.

Do **not** add `parity`/`correctness` markers — those axes are already the class, and
a marker that duplicates the class name only drifts.

## Pick the rows that apply

A submodule usually matches more than one row. Apply every row that fits; skip
the rest. The last row is the only license to write *no* unit test.

| The submodule is… | Owns these tests | Canonical benchmark to mirror |
|---|---|---|
| an **op / kernel / fused module** with its own numeric logic | forward **and** backward vs a simple reference | `tests/ops/test_hc_post.py` |
| **attention-related** (attention, indexer, kv-compressor, any cu_seqlens consumer) | packed correctness + **no cross-sample contamination** + causal constraint | `tests/module/dsa/test_indexer.py` |
| a consumer of a **static/padded shape** (alignment pad, `-1` fill, fixed buffer) | pad **does not change values or leak gradient** | `tests/module/dsa/test_dsa.py` (`test_hca_*`) |
| a **multi-backend** interface (e.g. native / flash_mla / cudnn) | every backend's forward+backward agrees, anchored on the reference backend | `tests/module/dsa/test_dsa.py` |
| **sharded / parallel** (EP dispatch, TP/FSDP-aware weight or grad path) | single-rank result == gathered multi-rank result | `tests/model/test_qwen3_moe.py` (`ep_size` params, `test_fsdp_accuracy`) |
| a **weight key-mapping** (`to_hf_key_list`, `from_hf`/`save_hf`) | round-trip + **no silent skip** + full key coverage | `tests/model/test_qwen3_moe.py::test_save_hf` |
| an **integrated assembly** of already-tested ops (e.g. DSA as a whole) | a **smoke** test — forward+backward run across a few shapes/backends without error (correctness stays with the ops + parity) | `tests/module/dsa/test_dsa.py` (`TestDeepSeekSparseAttention`) |
| a **thin composition** of already-unit-tested pieces, no new numeric logic | *none* — leave to parity (see last section) | — |

---

## 1. Op-level correctness — forward AND backward vs a readable reference

An op or fused kernel must match a reference that a reader can trust at a glance.
The reference is the *spec*: keep it a plain, eager, unfused expression of the
math (`tests/ops/test_hc_post.py::_hc_post_ref` is the model — the reference is
literally the un-fused body of the eager path). Do not reuse the production code
as its own reference.

- **Forward and backward are separate obligations.** A kernel can be bitwise-correct
  forward and still have a wrong backward — test both. For backward, feed a grad
  and compare input `.grad` (see `test_backward_matches_reference`).
- **For a custom `autograd.Function`, run `torch.autograd.gradcheck`** on fp64
  inputs — it catches backward-formula errors the reference comparison can miss.

### Tolerance policy (this is where bf16 bites)

An op-vs-eager-reference test is `CorrectnessTest`; set its `atol`/`rtol` on the
class so the tolerance is visible in the header, and widen it in a bf16 subclass
rather than at call sites.

- **fp32**: compare near-exact (tight `rtol`/`atol`). Reserve strict `torch.equal`
  (`ParityTest`) for HF parity; two fp32 kernels usually differ in the last bits.
- **bf16**: never `allclose` element-wise. Either compare a **relative error /
  norm ratio**, or — the stronger form — assert the fused kernel is **no worse
  than the eager reference measured against an fp32 ground truth**
  (`test_forward_no_worse_than_reference_vs_fp32`). This tolerates unavoidable
  bf16 rounding without letting a real regression through.

## 2. Attention & packed samples — correctness and no cross-contamination

Anything that consumes `cu_seqlens` (attention, indexer, compressor) owns two
distinct guarantees, tested separately (`tests/module/dsa/test_indexer.py`):

- **Packed == per-sample.** A sample must produce the same output whether run
  alone or packed after another sample. Test: build packed input as
  `cat([A, B])` with the matching `cu_seqlens`, and assert B's slice of the
  packed output equals B computed alone (`test_two_samples_no_cross_contamination`,
  `test_two_samples_parity`, `test_single_sample_parity`).
- **Causal / masking constraint holds within each document** — no attention or
  top-k selection crosses a `cu_seqlens` boundary (`test_causal_constraint`).

Cross-sample leakage is the most common packed-attention bug and parity will not
reliably surface it (packed batches in parity may not straddle a boundary in the
revealing way), so it must be a dedicated submodule test.

## 3. Special / padded shapes must not pollute

When a submodule pads to a static shape (FlashMLA 128-alignment, a `-1`-filled
top-k buffer, a fixed pre-allocated buffer), "doesn't crash" is **too weak**. The
real risk is the padded positions silently changing the answer. Assert:

- padded positions **do not affect the valid output**, and
- **no gradient flows into the pad**.

Also cover the branch where the pad path is *not* taken (e.g. native backend
needs no alignment), so a shape assumption doesn't regress — see
`test_hca_pad_buffer_static_shape` and
`test_hca_pack_max_length_without_flash_backend_falls_back`.

## 4. Multi-backend interfaces — every backend agrees

For an interface with several backends (native / flash_mla / cudnn, triton /
cutlass, …):

This is `CorrectnessTest`, not parity: backends differ in the last bits, so the
anchor comparison carries a tolerance (name the class `…Correctness`/`…Equivalence`).

- Treat the simplest backend (usually `native`) as the **anchor reference** and
  assert every other backend's **forward and backward** agree with it within the
  bf16 tolerance policy of §1 — not just pairwise, all against the anchor.
- Backends whose backward differs structurally from their forward (e.g. a
  forward kernel paired with a recompute/native backward) need their backward
  tested explicitly; a matching forward does not imply a matching backward.
- `skip` cleanly when a backend is unavailable in the environment rather than
  failing (`pytest.skip`), so the suite still runs where only native exists.

Reference layout: `tests/module/dsa/test_dsa.py` parametrizes over `backend`.

## 5. Distributed / parallel equivalence

Sharding (EP dispatch, TP/FSDP-aware weight and gradient paths) is the highest-risk
area — its bugs are rank-specific and parity may hide them by averaging. The core
invariant: **the sharded computation equals the unsharded one after gathering.**

- Parametrize over `ep_size` (and dispatcher) and assert the result matches the
  `ep_size=1` / single-process baseline — see `test_qwen3_moe_run`'s `ep_size`
  parametrization and `test_fsdp_accuracy`.
- All ranks must enter every collective in the same order; a test that hangs on
  one rank is reporting a real ordering bug, not a flaky test.
- Represent empty/zero-sized shards correctly — a rank owning no experts/tokens
  is legal and must not be special-cased by name.

## 6. Weight-loading correctness

One test covers the whole load/save contract: **round-trip + no silent skip +
full key coverage**. Benchmark: `tests/model/test_qwen3_moe.py::test_save_hf`.

It `from_hf`-loads then `save_hf`-writes, then on rank 0:

- for **every** key in the original HF index, asserts the saved tensor
  `torch.equal`s the original (round-trip is loss-less), and
- asserts the **set of keys written == the set of keys in the model index** — this
  is what catches a `to_hf_key_list` that silently drops layers (a wrong prefix
  mapping made whole layers vanish without error before; the key-set assertion is
  the guard that would have failed loudly).

A key-mapping change is not covered until both halves hold: values match *and* no
key is missing on either side.

---

## When to write no unit test

Skip a dedicated unit test **only** when both are true:

1. the submodule is a **thin composition** of pieces that are already unit-tested,
   introducing **no new numeric, masking, shape, or sharding logic** of its own; and
2. whole-model **parity covers it** end-to-end.

This "none" is for a *thin* composition. An **integrated assembly** (DSA as a
whole) is the middle layer, not "none": its **correctness** is owned by its parts
(indexer, compressor, backends, pad path — each with their §1–6 tests) and by
parity, so it writes no correctness test — but it still owns a **smoke** test
(forward+backward across a few shapes/backends, just "runs without error") to
catch wiring and shape regressions. A smoke is not a stand-in for a part-test: a
missing part-test (e.g. no backend-backward agreement) is fixed by **adding that
part-test**, never by promoting the smoke to a correctness claim it does not make.
And the moment a module adds its own numeric or masking logic, that logic owns a
real test even deep inside an integrated block — it must fail in its own test, not
only as a whole-model mismatch.

> Regression tests for specific fixed bugs are **not** listed here on purpose —
> that coverage is driven by CI feedback, not by this matrix.

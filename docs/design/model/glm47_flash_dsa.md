# GLM-4.7-Flash DSA and IndexShare integration design

## 1. Objective and invariants

Convert the pretrained `zai-org/GLM-4.7-Flash` MLA model into a deployable
DeepSeek Sparse Attention model without changing the pretrained dense function
before training.  Training is split into two separate jobs:

1. `dense_indexer_warmup`: keep the original dense MLA path, freeze every
   pretrained parameter, and train only newly added indexers against full
   causal-attention teachers.
2. `sparse_train`: reload the warm-up checkpoint, run SparseMLA with shared
   Top-K indices, and train the complete model.  Language-model losses update
   the backbone; the indexer KL path updates indexers only.

The implementation must preserve these invariants:

- Loading a GLM-4.7-Flash checkpoint may miss only explicitly enumerated new
  `indexer.*` parameters and may not ignore any unexpected key.
- `dense_baseline` and `dense_indexer_warmup` produce the same attention output
  and logits as the original Hugging Face model before an optimizer update.
- Teacher tensors and indexer inputs are detached from the KL autograd path.
- Top-K selection is always non-differentiable.
- A source indexer is supervised by every attention layer that consumes its
  indices, not only by the source layer.
- Dense warm-up never retains an `O(S^2)` tensor for backward.
- The main-attention and indexer RoPE layouts are independently configurable.

## 2. Model topology

GLM-4.7-Flash has 47 main decoder layers and one physical MTP layer.  The main
stack uses uniform one-in-four IndexShare:

```text
0F 1S 2S 3S | 4F 5S 6S 7S | ... | 44F 45S 46S
```

`F` owns an indexer and `S` reuses the nearest preceding `F` result.  This gives
12 main-stack indexers.  The physical MTP layer owns and trains a separate
indexer; it must not freeze an uninitialised random indexer.

The initial indexer shape follows GLM-5.2:

```text
index_n_heads = 32
index_head_dim = 128
index_topk = 2048
index_topk_freq = 4
index_skip_topk_offset = 1
```

For source layer `r`, indexer projections are

```math
q^I_t = W^I_q q^{resid}_t,
\qquad
k^I_s = \operatorname{Norm}(W^I_k h_s),
\qquad
w_t = W^I_w h_t,
```

and the score is

```math
I_{t,s} = \frac{1}{\sqrt{H_I D_I}}
\sum_{j=1}^{H_I} w_{t,j}
\operatorname{ReLU}\left((q^I_{t,j})^\top k^I_s\right).
```

Selection and training must implement the same effective scale even if their
kernel APIs divide the head and head-dimension factors at different points.

## 3. Training modes

`DSAMLAConfig` exposes an architecture/runtime mode independent of whether the
module is in `train()` or `eval()`:

```python
attention_mode: Literal[
    "dense_baseline",
    "dense_indexer_warmup",
    "sparse_train",
]
```

The two trainable modes are intentionally run as separate Trainer jobs.  The
warm-up job builds an optimizer from indexer parameters only.  The sparse job
reloads the hand-off checkpoint and builds a new optimizer from all trainable
parameters.  No optimizer or FSDP membership changes in the middle of a job.

### 3.1 Dense frozen-teacher warm-up

For attention layer `l`, the full causal teacher is the head-aggregated dense
attention distribution

```math
p^{(l)}_{t,s} = \frac{1}{H_A}\sum_h
\operatorname{softmax}(A^{(l,h)}_{t,:})_s.
```

For source indexer `r`, the full causal student is

```math
q^{(r)}_{t,s} = \operatorname{softmax}(I^{(r)}_{t,:})_s.
```

Let `G_r` be the main layers served by source `r`.  Its loss is

```math
L^{(r)}_{dense} = \frac{1}{|G_r|N}
\sum_{l\in G_r}\sum_t
D_{KL}\left(p^{(l)}_{t,:}\|q^{(r)}_{t,:}\right).
```

All pretrained parameters have `requires_grad=False`.  Hidden states and
Q-LoRA residuals entering the indexer are detached, so only indexer projection
parameters receive gradients.

### 3.2 Sparse full-model training

The source selects `T_t = TopK(I_{t,:}, K)`.  Every layer in `G_r` consumes the
same `T_t` through SparseMLA and contributes a selected-support teacher:

```math
L^{(r)}_{sparse} = \frac{1}{|G_r|N}
\sum_{l\in G_r}\sum_t
D_{KL}\left(p^{(l)}_{t,T_t}\|q^{(r)}_{t,T_t}\right).
```

The total training loss is

```math
L = L_{LM} + \lambda_{indexer}L_{indexer} + L_{MTP}.
```

The sparse teacher is stop-gradient but moves with the fully trainable model.
A second frozen dense model is not retained during sparse training.

## 4. Dense-loss memory design

A single FP32 score matrix at `S=200000` occupies about 160 GB, so dense score
recompute is query-blocked.  `dense_query_block_size` starts at 256 and remains
configurable.

For each local query block:

1. Recompute dense attention raw scores and denominator.
2. Recompute dense indexer raw scores and denominator.
3. Accumulate the KL value.
4. Run cuDNN DenseIndexerBackward and accumulate feature gradients.
5. Release both block score tensors immediately.

The full score tensors are never saved in an autograd context.  Instead, a
first-order custom autograd node attaches the accumulated `dQ`, `dK`, and `dW`
buffers to the original indexer features.  Its backward multiplies them by the
upstream scalar and returns ordinary PyTorch gradients, allowing FSDP and the
optimizer to handle indexer parameters normally.  Higher-order derivatives are
unsupported by design.

Under sequence parallelism, queries remain sharded, K is gathered as required,
and every query block passes its global `q_causal_offsets`.  Loss normalisation
uses the global number of valid query rows.  Tests compare one-rank and
multi-rank loss and gradients.

## 5. Layering and ownership

- `xtuner/v1/model/moe/glm47_flash.py`
  - HF config conversion, model defaults, strict base-to-DSA conversion, MTP
    policy, HF key mapping and export.
- `xtuner/v1/module/attention/dsa_mla.py`
  - dense/sparse runtime modes, feature production, teacher capture and
    per-layer calls into the loss interface.
- `xtuner/v1/module/attention/dsa_topk_sharing.py`
  - source resolution, Top-K sharing, group lifetime and checkpoint replay.
- `xtuner/v1/data_proto/sequence_context.py`
  - per-call source features, Top-K entries and group-loss accumulator state.
- `xtuner/v1/loss/dsa_indexer_loss.py`
  - backend-neutral dense/sparse KL definitions and reduction semantics.
- `xtuner/v1/ops/sparse_mla/`
  - cuDNN dense/sparse score recompute and manual first-order gradient ops.

The math/reduction API belongs under `v1/loss`; cuDNN-specific implementation
details remain under `v1/ops`.

## 6. Checkpoint and export contract

The original `glm4_moe_lite` model is first supported as an exact dense XTuner
model.  DSA conversion reuses every pretrained tensor and initialises only the
new indexers.  Conversion asserts the exact missing-key set and rejects every
unexpected key.

The warm-up hand-off is an XTuner distributed checkpoint to avoid an extra full
HF save.  Final export writes a Hugging Face DSA checkpoint only after its RoPE
layout and weight names are proven compatible with the target Transformers
runtime.  If upstream `GlmMoeDsaForCausalLM` cannot express the preserved
GLM-4.7 main-attention RoPE layout, export uses a dedicated
`glm4_moe_lite_dsa` config/runtime rather than silently changing numerics.

## 7. Verification gates

1. Dense baseline: per-layer forward, loss and input-gradient parity with HF;
   full-model forward parity where hardware permits; byte-equal save/load.
2. Dense loss: PyTorch reference parity for loss and `dQ/dK/dW`; invariant
   across query block sizes; causal-offset tests.
3. Frozen warm-up: only indexers are in the optimizer, every teacher gradient is
   `None`, teacher parameter checksums do not change, and logits equal the dense
   baseline before/after indexer-only updates.
4. IndexShare: each source receives every served layer exactly once; group loss
   and gradients match an explicit reference.
5. Sparse path: one Top-K computation per group, shared indices are identical,
   sparse loss updates only indexers, and LM loss updates the full model.
6. Parallel path: SP 1/2/8 parity, checkpoint original/replay equivalence,
   compiled/eager comparison, activation-offload smoke test.
7. Integration: 4K fixed-sample overfit, 16K/32K smoke training, then 128K and
   200K forward/backward runs on a GPU node.

For a single teacher, the fixed-sample KL must approach zero.  For multi-layer
supervision, the optimum is the mean teacher distribution and the irreducible
minimum is the corresponding Jensen-Shannon divergence, so the acceptance gate
compares against that oracle floor rather than requiring zero.

## 8. Training defaults

The initial reproduction schedule is:

```text
dense warm-up: 1000 steps, indexers only, LR 1e-3 starting point
sparse train:  4000 steps, full model, LR 7.3e-6 starting point
```

Training ramps through 4K correctness, 16K/32K smoke tests, and only then
128K/200K.  Exact batch size and gradient accumulation are selected from the
available node memory without changing the loss normalisation.

The existing sparse-backward `-1` padding change is outside this integration's
scope unless it blocks a required correctness test; any such failure is reported
separately rather than silently broadening the patch.

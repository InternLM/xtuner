# GLM-5.3 KPool with LMDeploy FP8 forward

The existing `indexer_backend="tilelang"` setting and `kpool_topk_indices`
function name are preserved. For GLM-5.3 production geometry, their default
implementation reuses LMDeploy KPool compression, query rotation, FP8 scoring,
deterministic selection and token expansion. No new backend name or flag is
introduced. Other geometries, explicit alternative selectors/scorers and SP
query balancing retain the previous TileLang implementation.

This is separate from the model's linear-layer precision. A MoE-only recipe
still uses:

```python
Float8Config(
    scaling_granularity_gemm=None,
    scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
)
```

The rollout configuration uses `enable_float8=True` and the environment variable
`LMDEPLOY_FP8_MOE_ONLY=True`. The serving GLM-5.3 builder must actually set the
FP8 quantization scope to `moe_only`; check the effective serving config.

## Forward contract

The adapter requires an importable LMDeploy tree with GLM-5.3 KPool support and
DeepGEMM's `fp8_mqa_logits` API. It is tested against LMDeploy commit
`3228395546dfb2f6427faf39ec0c0c285f4c378a` on H200. Imports are lazy; other
indexer backends do not acquire an LMDeploy dependency.

- Indexer head dimension: 128; query heads: 32, 64 or 128.
- Pool size: 4; token budget: 2048 or 8192.
- Query chunking bounds score memory. The default is 1024 queries per chunk.
- Packed pools restart at document boundaries. Keys and compression gates
  gather across sequence parallelism before pools are built; query outputs
  retain their original shard ownership.
- The operator adapter rejects query redistribution. The existing public path
  retains its previous TileLang implementation when query balancing is enabled.
- Head-gate projection and key normalization follow LMDeploy's FP32 arithmetic
  boundaries. Both keys and queries follow its Hadamard/FP8 scale contract.
- Selected pool order and compact tail placement follow LMDeploy. Re-sorting
  IDs or moving valid tails behind padding changes sparse-attention reduction
  order and is deliberately avoided.
- Pool compression uses decode's two-pass order, matching generated history.
  Serving constructs initial prompt pools using extend's online order. This
  difference is explicit; full forward equality for every prefill/decode
  schedule is not claimed.

## Compression backward

`lmdeploy_kpool_compress(keys, scores, ape, mode="decode")` returns a
`KPoolCompression` with three fields:

| Field | Contract |
| --- | --- |
| `dequantized` | FP32 values with a custom first-order backward |
| `values` | Nondifferentiable E4M3FN storage from LMDeploy |
| `scales` | Nondifferentiable FP32 quantization scales from LMDeploy |

CUDA forward calls LMDeploy's fused `compress_kpool` directly. CPU forward uses
its reference implementation. Empty groups return correctly shaped empty
outputs without invoking the reference implementation's ambiguous reshape.

Backward uses an explicit straight-through estimator for BF16/FP8 rounding.
For `p = softmax(scores + ape)` and `z = sum(p * keys)` over pool slots:

```text
dz = normalized_hadamard(doutput)
dkeys = p * dz
dscores = p * (keys - z) * dz
dape = sum(dscores over groups)
```

Arithmetic is FP32 and returned gradients use the corresponding input dtypes.
This is a surrogate derivative, not the mathematical derivative of the
discrete quantizer. Higher-order derivatives are unsupported. Tests compare
the first-order gradients with independent FP64 autograd using an explicit
Hadamard matrix, and compare CUDA FP8 values/scales bitwise with LMDeploy.

Top-K IDs remain discrete and nondifferentiable. GLM-5.3's DSA indexer remains
frozen, and `freeze_dsa_indexer=False` remains unsupported. Native GRPO runs
exercise the new selection forward and the existing attention backward;
compression backward is validated separately. A lower mismatch metric alone
does not establish nonzero learning or model quality.

## Tests

With compatible XTuner/LMDeploy paths in `PYTHONPATH`:

```bash
python -m pytest tests/ops/test_lmdeploy_kpool.py -m 'not gpu' -q
python -m pytest tests/ops/test_lmdeploy_kpool.py -m gpu -q
```

Use the same checkpoint, commits, seed, batch size, sampling parameters,
MoE-only scope and rollout settings for both native RL runs. Compare the
per-step `mismatch/mismatch_k3_kl` and record generated lengths and timing.
Native runs generate their own trajectories; this is not a fixed-token causal
ablation or a complete-model quality benchmark.

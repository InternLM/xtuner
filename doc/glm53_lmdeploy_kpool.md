# GLM-5.3 KPool with LMDeploy FP8 forward

The existing `indexer_backend="tilelang"` setting and `kpool_topk_indices`
function name are preserved. For GLM-5.3 production geometry, their default
implementation can reuse LMDeploy KPool compression, query rotation, FP8 scoring,
deterministic selection and token expansion. No new backend name or flag is
introduced. Runtime selection requires SM90+ and compatible LMDeploy/DeepGEMM
APIs; missing optional dependencies or unsupported GPUs retain the original
TileLang implementation, including its projection arithmetic. Availability is
cached per device and checked at forward time, allowing CPU/meta construction.
Other geometries, explicit alternative selectors/scorers and SP query balancing
also retain the previous TileLang implementation. Operator failures after
dispatch propagate rather than silently changing the computation.

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
DeepGEMM's compatible MQA APIs. It is tested against LMDeploy commit
`3228395546dfb2f6427faf39ec0c0c285f4c378a` on H200. Imports are lazy; other
indexer backends do not acquire an LMDeploy dependency. This adapter imports
internal LMDeploy helpers; the commit above identifies the verified interface.

- Indexer head dimension: 128; query heads: 32, 64 or 128.
- Pool size: 4; token budget: 2048 or 8192.
- Query chunks are capped at 1024 rows by default (or the configured row cap),
  then limited by `LMDEPLOY_DSA_INDEXER_MAX_LOGITS_MB` (LMDeploy default: 512 MiB).
  LMDeploy's helper accounts for DeepGEMM's query/head and 256-column alignment;
  a budget smaller than one aligned block raises an error. This budget bounds
  logits, not total operator memory. Query rotation/quantization is chunked too.
- Packed pools restart at document boundaries. Keys and compression gates
  gather across sequence parallelism before pools are built; query outputs
  retain their original shard ownership.
- The operator adapter rejects query redistribution. The existing public path
  retains its previous TileLang implementation when query balancing is enabled.
- Head-gate projection explicitly disables autocast via `fp32_linear`; key normalization follows LMDeploy's FP32 arithmetic
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

CUDA compression uses LMDeploy's own SM90+ fused/reference dispatcher. CPU
compression uses its reference implementation. Empty groups return correctly shaped empty
outputs without invoking the reference implementation's ambiguous reshape.
The no-grad Top-K path requests only FP8 storage/scales and does not allocate the
FP32 dequantized output; the differentiable compression API remains unchanged.

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
Hadamard matrix. Tests check bitwise wrapper parity with the serving operator
and independently compare fused compression against LMDeploy's PyTorch reference
for both modes and scale-rounding settings. The latter allows rounding-boundary
differences; mathematical equivalence does not imply bitwise reference parity.

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
CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/ops/test_lmdeploy_kpool_sp.py -q
```

The two-GPU test covers pools crossing a shard seam, packed document boundaries,
global token IDs, causal tails and local query ownership, comparing against an
unsharded run. Short contexts also use an independent token-level causal oracle.

On `10.102.236.51` (H200), a fixed-input selection check with seed 123, packed
lengths `(8195, 8189)`, 32 heads, dimension 128 and Top-K 2048 found zero changed
token IDs relative to PR commit `0a08a402`, for query chunks 1024 and 257. The
incremental PyTorch allocator peak fell from 910.79 MiB to 472.21 MiB (chunk
1024), or 461.03 MiB (chunk 257). These exclude input tensors and are not total
device memory. GPUs had concurrent workloads, so no latency claim is made.

Use the same checkpoint, commits, seed, batch size, sampling parameters,
MoE-only scope and rollout settings for both native RL runs. Compare the
per-step `mismatch/mismatch_k3_kl` and record generated lengths and timing.
Native runs generate their own trajectories; this is not a fixed-token causal
ablation or a complete-model quality benchmark.

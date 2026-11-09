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
  The allocation limit follows the active DeepGEMM API: compact columns use
  `align(N, 256)`, while legacy absolute columns use `align(N + 256, 64)`.
  Both pad query rows to `128 / heads`; a budget smaller than one aligned block
  raises an error. This budget bounds
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

## Frozen indexer and forward-only scope

`_compress_storage()` returns only E4M3FN values and FP32 scales. CUDA
compression uses LMDeploy's SM90+ fused/reference dispatcher; CPU compression
uses its reference implementation. Empty groups return correctly shaped empty
outputs without invoking the reference implementation's ambiguous reshape.

Top-K selection runs under `torch.no_grad()`. The DSA indexer stays frozen and
`freeze_dsa_indexer=False` remains unsupported. There is no differentiable
compression API or custom backward in this adapter. GRPO still uses the
existing attention/MoE backward, and a lower mismatch metric does not establish
nonzero learning or model quality.

## Tests

With compatible XTuner/LMDeploy paths in `PYTHONPATH`:

```bash
python -m pytest tests/ops/test_lmdeploy_kpool.py -m 'not gpu' -q
python -m pytest tests/ops/test_lmdeploy_kpool.py -m gpu -q
CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/ops/test_lmdeploy_kpool_sp.py -q
```

Forward tests compare pool FP8 values/scales bitwise with the serving operator
and independently with its PyTorch reference for both compression modes and
scale-rounding settings. Query FP8 payloads/scaled weights are checked at the
actual MQA call, and valid causal logits are compared against Torch FP8 scoring.
Storage-budget tests inspect `logits.untyped_storage().nbytes()` for compact and
legacy APIs, including the 4096-token/32-head/1-MiB regression and short tails.

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

A three-step MoE-only GRPO check of commit `75aeaec1` on 8 H200 GPUs compared
against the pre-adapter baseline `c93eb15d` using the same five-layer checkpoint,
seed 123, EP4/SP1 training and TP1/EP4 rollout. Mean
`mismatch/mismatch_k3_kl` was `3.7143783866e-4` for the baseline and
`3.4190760300e-4` with the adapter (7.95% lower). Each step had 64 responses.
The runs generated independent trajectories and had zero rewards/losses/gradient
norms. This measures forward mismatch; it does not validate indexer learning.

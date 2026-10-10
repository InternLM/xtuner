# GLM53 KPool indexer optimizations

This change builds on `feat/glm53flash-f5-nope-dsa` and optimizes its pooled indexer.
The optimized indexer retains the original backend as an implementation choice:

```python
attention_cfg.indexer_backend = "tilelang_cooperative"
attention_cfg.indexer_topk_query_chunk_size = 1024
# Reference production choice: attention_cfg.indexer_backend = "tilelang"
```

Here `attention_cfg` is the existing `NoPEDSAMLAConfig`; its build method forwards these options.
Cooperative scoring requires Hopper/SM90, BF16 queries/keys with 32 heads of dimension 128,
FP32 head weights, and an installed DeepSelect module. Pool selection budgets must be
multiples of eight. Packed-document boundaries and causal pool/tail semantics are preserved.
The scorer accumulates and publishes FP32 scores and reuses aligned scratch across query chunks.
The wrapper supports torch.compile through a custom op; host boundary reads and the zero-start
check currently prevent CUDA graph capture.

## Scoring measurements

Previously measured on H200 with BF16 Q/K, 32 heads x128, pool size 4 and top512 pools.
For the final 1024 causal queries at 128K, median CUDA-event time over 15 samples:

| Component | Original | Optimized |
|---|---:|---:|
| Scoring including original score cleanup | 0.919424 ms | 0.425504 ms |
| Top-k on identical logits | 0.513184 ms | 0.091872 ms |

Across all 128K queries on one GPU, scoring plus selection takes 124.379 ms with the
original chunk1024 implementation versus 36.451 ms with cooperative scoring plus DeepSelect.
A full indexer including projections takes 174.623 versus 86.552 ms. These are synthetic
measurements including allocation/launch gaps, with compilation excluded. Short 1K/4K
full-indexer cases do not improve. The 128K aligned score scratch is 128 MiB; overall memory
also includes queries, pooling, selected IDs and token expansion.

The score comparison measured maximum absolute error 4.768e-7 and relative L2 error 6.561e-8.
Selection is not bitwise identical near the cutoff: 131065/131072 full-indexer rows were exact,
with selected-ID overlap 99.9999895% at 128K.

Reproduce using an existing environment; the launcher does not install packages:

```bash
CUDA_HOME=/path/to/cuda PYTHON=/path/to/python bash tools/run_glm53_indexer_comparison.sh
```

The cuDNN DSA backward adapter stably compacts valid indices into a prefix. This preserves
valid tokens following -1 holes, including appended tails, without changing FlashMLA forward
index ordering. Tests cover holes, duplicates, strided inputs and empty query sets.

## Indexer sequence-parallel work balancing

```python
attention_cfg.indexer_balance_sp = True
attention_cfg.indexer_sp_full_pool_work_ratio = 0.5
```

Attention retains uniform contiguous sequence shards. Only scoring queries and FP32 head
weights move through neighbor P2P into intervals balanced for causal work. Selected pool IDs
return to their original owners before canonical sorting, token expansion and tail insertion.
The fixed work ratio estimates per-query overhead; it does not mean DeepSelect scans every pool.
Collective eligibility checks fall back uniformly for unsupported layouts or no estimated gain.

Previously measured on eight H200s at 128K: full cooperative indexer 14.992 ms unbalanced
versus 13.688 ms balanced, an 8.70% latency reduction. The complete block change in that
experiment was small relative to overlapping sample variation. Regression cases preserve exact
selected-ID ordering when toggling balancing, including packed/poolless documents and forwarding
across multiple ranks. Reproduce with `tools/run_glm53_sp_indexer_balance.sh`.

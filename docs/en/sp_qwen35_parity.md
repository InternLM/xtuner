# Qwen3.5 Dense sequence-parallel parity

This regression reproduces SP-dependent training divergence with a small Qwen3.5
Dense backbone and Muon. It requires two CUDA GPUs and the normal XTuner GDN
dependencies; no checkpoint, tokenizer download, or external dataset is needed.

## What changes

- `BaseModel.scale_and_reduce_grad` averages gradients of FP32 parameters that
  `_fully_shard` explicitly excluded from FSDP. These parameters are replicated
  DTensors, so FSDP does not synchronize their gradients. Without this reduction,
  Qwen3.5's `A_log` and GDN norm weights diverge between ranks after updates.
- Qwen3.5 Dense keeps convolution parameters and their gradients in FP32 through
  replica reduction. The convolution wrapper casts weights to the activation
  dtype for computation and returns gradients in the original parameter dtype.
  This retains the existing BF16 forward values while avoiding BF16 rounding of
  local weight gradients before synchronization.
- With PyTorch deterministic algorithms enabled, the packed convolution backward
  runs the existing CUDA kernel separately for each document and accumulates its
  FP32 weight/bias gradients in document order. Otherwise it retains the existing
  packed kernel call. SP changes document packing and channel partitioning; using
  document boundaries avoids the observed change in the kernel's reduction
  grouping. Each document's input gradient is computed by that same kernel call.

Muon arithmetic, forward activation precision, and loss normalization are unchanged.
The reproduction calls XTuner's complete `set_deterministic()` entry point before
CUDA setup, as the standard Trainer does. Setting only an environment variable or
only `torch.use_deterministic_algorithms` is insufficient for this comparison.

## Minimal experiment

From this checkout in a CUDA environment with XTuner and GDN dependencies:

```bash
export XTUNER_DETERMINISTIC=true
unset XTUNER_HF_IMPL
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  torchrun --standalone --nproc-per-node=2 tests/model/repro_qwen35_sp.py \
  --sp 1 --steps 30 --out /tmp/qwen35-fixed-sp1
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  torchrun --standalone --nproc-per-node=2 tests/model/repro_qwen35_sp.py \
  --sp 2 --steps 30 --out /tmp/qwen35-fixed-sp2
python -m pytest -q tests/ops/test_causal_conv1d_sp.py
python -m pytest -q tests/model/test_fsdp_ignored_grad.py
```

Output directories must be new. Each run saves per-step loss, rank spread before
and after gradient reduction, initialization/data hashes, dependency versions,
determinism settings, first-step gradients/updated weights, and final full-model
weights. `COMPLETE` is written only after all optimizer updates finish.

To reproduce the unfixed baseline, run the **same harness from this branch** with
the parent source first on `PYTHONPATH`:

```bash
git worktree add --detach /tmp/xtuner-sp-original fb51baebdf91b03bd39255c4427edac9aca82865
for sp in 1 2; do
  PYTHONPATH="/tmp/xtuner-sp-original${PYTHONPATH:+:$PYTHONPATH}" \
    torchrun --standalone --nproc-per-node=2 tests/model/repro_qwen35_sp.py \
    --sp "$sp" --steps 30 --out "/tmp/qwen35-original-sp$sp"
done
```

The model has 5,145,200 parameters: four layers (three GDN and one eager MHA),
hidden size 256, MLP size 768, eight attention heads, and convolution width four.
The global batch is two 512-token documents from embedded text, tokenized as UTF-8
bytes. SP1 gives one document to each rank; SP2 packs both documents and partitions
channels across ranks. Both use identical initial weights, batches, Muon
(`lr=1e-3`, `weight_decay=0`, `enable_all2all=True`,
`clip_grad_mode="adamw_only"`, and upstream-default MuonSplit), BF16 computation, and FP32
gradient reduction. This isolates SP without attention-kernel or data-loader noise.

Compare the two completed runs:

```python
import json
from pathlib import Path
import torch

a, b = Path('/tmp/qwen35-fixed-sp1'), Path('/tmp/qwen35-fixed-sp2')
assert (a / 'COMPLETE').exists() and (b / 'COMPLETE').exists()
rows = [[json.loads(line) for line in (p / 'metrics.jsonl').read_text().splitlines()]
        for p in (a, b)]
assert len(rows[0]) == len(rows[1]) == 30
assert [x['batch_sha256'] for x in rows[0]] == [x['batch_sha256'] for x in rows[1]]
loss_diff = max(abs(x['loss'] - y['loss']) for x, y in zip(*rows))
wa, wb = [torch.load(p / 'final.pt', map_location='cpu', weights_only=True) for p in (a, b)]
assert wa.keys() == wb.keys()
param_diff = max((wa[k] - wb[k]).abs().max().item() for k in wa)
print({'max_loss_abs': loss_diff, 'final_parameter_max_abs': param_diff})
assert loss_diff < 1e-5
assert param_diff < 1e-5
```

## Measured results

Based on upstream `fb51baebdf91b03bd39255c4427edac9aca82865`.
Verified on 2026-09-24 with two NVIDIA H200 GPUs, CUDA 12.8, PyTorch 2.10.0,
Transformers 5.14.1, flash-linear-attention 0.4.2, and causal-conv1d 1.6.2.post1.
The unfixed parent ran once for each SP size. The final patch ran independently
three times for each SP size. All runs completed 30 optimizer updates and matched
initialization, corpus, per-step batch, and harness hashes.

| Check | Unfixed parent, SP1 vs SP2 | Fixed, SP1 vs SP2 |
| --- | ---: | ---: |
| Maximum absolute loss difference, 30 steps | 0.0009149312973022461 | 0 |
| First step with different loss (zero-based) | 1 | None |
| Maximum first-step gradient difference, after reduction/clipping | 0.000244140625 | 2.3283064365386963e-10 |
| Maximum parameter difference after the first update | 3.0994415283203125e-06 | 0 |
| Maximum final parameter difference | 0.002135753631591797 | 0 |

All 15 pairs among the six fixed runs have zero maximum loss difference and zero
final parameter difference: six same-SP repeat pairs and nine cross-SP pairs.
The fixed runs' explicitly synchronized gradients and updated replicated
parameters have zero rank spread on every step. The remaining first-step
cross-SP gradient difference is in `A_log`; it produces no first-update or final
parameter difference in this experiment. Unfixed parameter comparisons use the
saved full-model snapshot from rank 0; its replicated parameters also diverge
within each run (maximum rank spread 0.009059607982635498).

The 16 CUDA convolution cases pass, including unequal document lengths and optional
bias. Two distributed regression cases also pass on two GPUs, covering FP32 ignored
gradient averaging, unchanged FSDP-managed gradients, unused parameters, and an SGD
update on 1D/2D meshes. Ruff lint/format checks, Python compilation, and `git diff --check` pass.
The entire repository test suite was not run. GitHub Actions runs the pre-commit
checks for `xtuner/v1`.

In the earlier controlled diagnosis on `393a0273` (before the upstream Muon
changes), gradient synchronization plus FP32
convolution gradients alone left a maximum loss difference of
7.462501525878906e-5. The native FP32 reduction differed by about 1e-9, crossed a
Muon BF16 rounding boundary, and eventually changed BF16 forward weights. This
motivates the document-wise reduction in addition to the synchronization fix.

PJLab job for both the unfixed parent and final patch, including all repeats and
16 CUDA tests: `xtuner-sp-upstream-20260924-97500998`.
Follow-up job for all 18 regression cases and the documented comparison with both
loss and parameter assertions: `xtuner-sp-review-20260924-37136298`.

Initialization SHA-256: `65b3f557d2d8c67bbfd0d10d192e9f4c4c511d44e33c19ec974eb9fc641747f7`.
Corpus SHA-256: `31e39d875fed6d6998587bf8e810b103452497c94d8ef91cd18f08dd8ea60b01`.

## Scope and cost

The loss target is **absolute error below 1e-5**. The validation covers the specific
two-GPU SP1/SP2 setup and 30 updates above. It is not a bitwise-parity guarantee
for all gradients, SP4/SP8, different batch packing, hardware, or long training.
With more documents or ranks, floating-point collective reduction grouping can
still change. Earlier diagnosis found tiny residual `A_log` gradient differences
even when losses and updated parameters were identical.

The deterministic path introduces a CPU boundary scan and one CUDA backward call
per document. It avoids the diagnosis prototype's extra whole-sequence backward,
but throughput has not been benchmarked. FP32 replicated convolution weights also
use additional memory. This path targets reproducibility; disabling deterministic
mode retains packed backward and does not carry the measured parity guarantee.
The end-to-end reproduction disables model compilation and uses eager MHA; it
does not validate compiled training, MoE, GLM, or the HF diagnostic adapter.

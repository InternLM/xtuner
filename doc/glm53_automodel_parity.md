# GLM-5.3 parity with a trainer-free AutoModel backend

The existing `TestGlm53TextMoEAccuracy.test_fsdp_accuracy` compares native HF
and XTuner. An optional AutoModel result adds a third model implementation to
the same checks. AutoModel loads through
`NeMoAutoModelForImageTextToText.from_pretrained(..., force_hf=False)`; no
AutoModel trainer, recipe, dataset or collator is involved.

## Inputs and metrics

`tests/model/glm53_parity_cases.py` builds the original four text cases and two
image cases on every run. Images come from
`tests/resource/mscoco_twocat_000000039769.jpg`: one RGB image resized to
224 × 224, then a two-image case containing that image and its horizontal flip.
Both model backends use the same tokenizer, processor and chat-template code.
`--processor-python` can select a separate interpreter for that preprocessing:
each run invokes the shared case helper on the raw inputs and transfers its CPU
tensors through an in-memory stdout pipe using `torch.save`/`torch.load`.
There are no pre-tokenized input files to generate, persist or reuse. Without
this option, preprocessing uses the model interpreter and requires a compatible
processor there. No AutoModel recipe processor is substituted.

The result manifest hashes the processor tensors before model dtype conversion,
including tokens, labels, pixels, image grids and sampled positions. The XTuner
test rejects mismatched manifests before comparing predictions. Different
Transformers versions may produce different preprocessing; such a mismatch is
an input-contract failure, not evidence of model numerical error.

These are the existing next-token test labels: visual tokens are ignored, but
text supervision is not restricted to assistant answers. The last eight valid
next-token positions retain the entire vocabulary for logits comparison.

The existing HF `output.loss`, XTuner `CELossConfig` path, and their assertions
remain unchanged. The standalone HF/AutoModel runner computes FP32 cross entropy
outside the model after one explicit input/label shift. Its `mean_ce` is compared
with the existing losses. Logits retain the separate per-case relative-L2 < 5%
and cosine > 0.998 checks; text and image loss curves retain their existing
checks. Runs use eval/no-grad, without updates, backward, packing or MTP.

## Run in separate environments

Run the reference entry point directly with Python. Running it through the
XTuner pytest suite would load XTuner's `conftest.py` and dependencies into the
AutoModel environment. The direct entry point and shared cases do not import
XTuner. Use the same checkpoint and checkout for every command, and a fresh
output directory for each run.

The following shell variables stand for caller-supplied absolute paths:

```bash
XT_ROOT=/absolute/path/to/xtuner
MODEL=/absolute/path/to/GLM-5.3-Flash-25B
AM_PYTHON=/absolute/path/to/automodel/bin/python
XT_PYTHON=/absolute/path/to/xtuner-environment/bin/python
RESULTS=/absolute/path/to/new-parity-results
```

If AutoModel is used from a source checkout rather than installed, expose that
checkout through `PYTHONPATH` for the AutoModel commands. No environment
activation or package installation is performed by the test.
The commands below explicitly use the HF/XT interpreter for preprocessing and
the AutoModel interpreter only for model execution. This keeps preprocessing
fixed while swapping the model backend.

First check raw-input preprocessing without loading a model or using a GPU:

```bash
"$AM_PYTHON" "$XT_ROOT/tests/model/run_glm53_reference.py" \
  --backend automodel --checkpoint "$MODEL" --prepare-only \
  --processor-python "$XT_PYTHON" \
  --output "$RESULTS/am-input-check"
```

Run AutoModel single-process, EP1/CP1:

```bash
"$AM_PYTHON" "$XT_ROOT/tests/model/run_glm53_reference.py" \
  --backend automodel --checkpoint "$MODEL" --ep-size 1 --cp-size 1 \
  --processor-python "$XT_PYTHON" \
  --output "$RESULTS/am-ep1-cp1"
```

Run AutoModel FSDP2 with EP8, first CP1 and then CP2:

```bash
"$AM_PYTHON" -m torch.distributed.run --standalone --nproc-per-node 8 \
  "$XT_ROOT/tests/model/run_glm53_reference.py" \
  --backend automodel --checkpoint "$MODEL" --ep-size 8 --cp-size 1 \
  --processor-python "$XT_PYTHON" \
  --output "$RESULTS/am-ep8-cp1"

"$AM_PYTHON" -m torch.distributed.run --standalone --nproc-per-node 8 \
  "$XT_ROOT/tests/model/run_glm53_reference.py" \
  --backend automodel --checkpoint "$MODEL" --ep-size 8 --cp-size 2 \
  --processor-python "$XT_PYTHON" \
  --output "$RESULTS/am-ep8-cp2"
```

`DistributedSetup` is passed to `from_pretrained`. The native GLM CP sharder
handles sequence slicing and ignored padding; NLL/count and sampled logits are
reduced within the CP group. DP replicas deliberately process the same case,
and their NLL range is recorded instead of averaging it away.

Consume one completed AutoModel result in the existing HF/XTuner test:

```bash
cd "$XT_ROOT"
GLM_5_3_FLASH_PATH="$MODEL" \
GLM53_AUTOMODEL_REFERENCE_DIR="$RESULTS/am-ep1-cp1" \
XTUNER_TEST_WORLD_SIZE=8 \
"$XT_PYTHON" -m pytest -v \
  tests/model/test_glm53_text_moe.py::TestGlm53TextMoEAccuracy
```

Change `GLM53_AUTOMODEL_REFERENCE_DIR` to compare the other AutoModel runs.
Unset it to retain the original HF/XTuner-only test. The existing test launches
its own workers and retains EP1/4/8 parameterization; do not wrap this pytest
command in `torchrun`. Select only the EP1 case for a one-GPU run:

```bash
GLM_5_3_FLASH_PATH="$MODEL" \
GLM53_AUTOMODEL_REFERENCE_DIR="$RESULTS/am-ep1-cp1" \
XTUNER_TEST_WORLD_SIZE=1 \
"$XT_PYTHON" -m pytest -v \
  'tests/model/test_glm53_text_moe.py::TestGlm53TextMoEAccuracy::test_fsdp_accuracy[None-1]'
```

The optional `XTUNER_TEST_SP_SIZE=2` enables sequence parallelism; its default
remains 1. Compare EP8/SP2 against the AutoModel EP8/CP2 result with:

```bash
GLM_5_3_FLASH_PATH="$MODEL" \
GLM53_AUTOMODEL_REFERENCE_DIR="$RESULTS/am-ep8-cp2" \
XTUNER_TEST_WORLD_SIZE=8 XTUNER_TEST_SP_SIZE=2 \
"$XT_PYTHON" -m pytest -v \
  'tests/model/test_glm53_text_moe.py::TestGlm53TextMoEAccuracy::test_fsdp_accuracy[all2all-8]'
```

XTuner uses `SequenceContext.split` and its existing sequence-parallel loss
interface. Only the eight sampled logits rows are reconstructed in each SP
group, with exactly one owner required per position. Media tensors remain
complete. AutoModel's CP and XTuner's SP refer to their respective model paths;
the comparison does not assume identical internal parallel algorithms. The validation below covers these configurations on the local reduced checkpoint.

For an optional standalone HF result, use the direct runner with `--backend hf`
in the HF/XT environment, EP1/CP1, and a new output directory. The existing pytest
test still computes its own HF reference using the original loss path.

## Attention backends and output

The original XTuner test retains Torch DSA and indexer implementations, with
FlashAttention for vision. HF uses eager attention. AutoModel defaults to
`--attn cudnn` for sparse MLA, with native Torch KPool, FLA KDA and SDPA vision;
SDPA kernel selection depends on the runtime. `--attn sdpa` changes the
AutoModel sparse MLA selection explicitly. These configurations do not claim
identical kernels across frameworks. Production XTuner TileLang/FlashMLA
experiments are separate from this existing test.

The output contains `manifest.json`, `results.json`, sampled logits in
Safetensors, and per-rank metadata describing the interpreter, imported source,
versions, model class, backend and topology. Distributed metadata includes mesh
and expert-shard information. Existing output directories are rejected.

## Local reproduction environment

The following shared-machine environments were used for this entry point's
validation; they are not requirements embedded in the scripts. Completed
AutoModel execution alone does not establish the three-framework comparison.

| Component | Path / version |
|---|---|
| HF / XTuner Python | `/mnt/shared-storage-user/llmrazor-share/comm_env/pt29_glm2/bin/python` |
| HF / XTuner packages | PyTorch `2.9.1+cu128`, Transformers `5.17.0` |
| AutoModel Python | `/mnt/shared-storage-user/llmrazor-share/zhangxinsen/dev_envs/automodel/bin/python` |
| AutoModel packages | PyTorch `2.10.0+cu130`, Transformers `5.15.1`, FLA `0.4.2` |
| Shared preprocessing interpreter | `/mnt/shared-storage-user/llmrazor-share/comm_env/pt29_glm2/bin/python` (Transformers `5.17.0`) |
| AutoModel source | `/mnt/shared-storage-user/llmrazor-share/zhangxinsen/Automodel` |
| AutoModel recorded commit | `fda1bf266c78e3723599b9441f1c93b94f26380f` |
| Test checkout | `/mnt/shared-storage-user/llmrazor-share/zhangxinsen/xtuner-glm53-automodel-parity` |
| Checkpoint | `/mnt/shared-storage-user/llmrazor-share/model/GLM-5.3-Flash-25B` |

The shared HF/XTuner environment should remain unchanged. Record the actual
source revisions and dirty state for new runs; a path alone does not pin code.

In the local AutoModel environment, Transformers `5.15.1`'s `AutoProcessor`
did not provide the required `image_grid_thw` for these raw image cases. Local
AutoModel runs therefore explicitly pass
`--processor-python /mnt/shared-storage-user/llmrazor-share/comm_env/pt29_glm2/bin/python`.
This selects the existing compatible HF preprocessing environment without
changing either installation. Model and processor environments are distinct
parts of the reproduction setup; input-contract checks still apply.

This self-contained local example sets the CUDA libraries and source import
path as well as the interpreter. Run it as a separate Bash script/subshell so
its CUDA 13 library settings do not leak into later HF/XTuner commands. It does
not source another script or modify either environment. Choose an unused GPU
and a new output directory before running.

```bash
(
set -euo pipefail
AM_PYTHON=/mnt/shared-storage-user/llmrazor-share/zhangxinsen/dev_envs/automodel/bin/python
PROCESSOR_PYTHON=/mnt/shared-storage-user/llmrazor-share/comm_env/pt29_glm2/bin/python
RUN_PY=/mnt/shared-storage-user/llmrazor-share/zhangxinsen/xtuner-glm53-automodel-parity/tests/model/run_glm53_reference.py
MODEL=/mnt/shared-storage-user/llmrazor-share/model/GLM-5.3-Flash-25B
OUTPUT=/mnt/shared-storage-user/llmrazor-share/zhangxinsen/data/xtuner-automodel-loss-compare-data/forward-parity/am-local-rerun-ep1-cp1
export CUDA_HOME=/mnt/shared-storage-user/llmrazor-share/zhangxinsen/dev_envs/automodel/lib/python3.12/site-packages/nvidia/cu13
export CUDA_PATH="$CUDA_HOME"
export CUDNN_HOME=/mnt/shared-storage-user/llmrazor-share/zhangxinsen/dev_envs/automodel/lib/python3.12/site-packages/nvidia/cudnn
export NVTE_CUDA_INCLUDE_DIR="$CUDA_HOME/include"
export PATH="/mnt/shared-storage-user/llmrazor-share/zhangxinsen/dev_envs/automodel/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$CUDA_HOME/lib:/mnt/shared-storage-user/llmrazor-share/zhangxinsen/dev_envs/automodel/lib/python3.12/site-packages/nvidia/nvshmem/lib:/mnt/shared-storage-user/llmrazor-share/zhangxinsen/dev_envs/automodel/lib/python3.12/site-packages/nvidia/nccl/lib:/mnt/shared-storage-user/llmrazor-share/zhangxinsen/dev_envs/automodel/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH=/mnt/shared-storage-user/llmrazor-share/zhangxinsen/Automodel
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export TRITON_CACHE_DIR=/mnt/shared-storage-user/llmrazor-share/zhangxinsen/data/xtuner-automodel-loss-compare-data/forward-parity/am-local-rerun-cache
CUDA_VISIBLE_DEVICES=1 "$AM_PYTHON" "$RUN_PY" \
  --backend automodel --checkpoint "$MODEL" --processor-python "$PROCESSOR_PYTHON" \
  --ep-size 1 --cp-size 1 --output "$OUTPUT"
)
```

For the eight-GPU configurations, keep the same environment block, choose a
different absolute `OUTPUT`, and replace its final command with the corresponding
`torch.distributed.run --standalone --nproc-per-node 8` invocation above. Set
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` only when those GPUs are available.

## Validation (2026-10-10)

On the local reduced checkpoint and 8×H200, the new runner completed all six raw
cases for AM EP1/CP1, EP8/CP1 and EP8/CP2. The original pytest entry point then
passed the HF/XT/AM loss and logits checks at each corresponding topology.
CPU contract tests: 8 passed. Raw preprocessing contracts match across the two
entry points. Ruff and diff whitespace checks passed.

Maximum sampled logits relative L2 across the six cases (and participating XT
ranks); arrows mean reference → candidate:

| XT / AM topology | HF → XT | HF → AM | AM → XT | Loss checks |
|---|---:|---:|---:|---|
| EP1/SP1 / EP1/CP1 | 2.5726% | 3.5015% | 2.4247% | Passed |
| EP8/SP1 / EP8/CP1 | 2.5726% | 3.2442% | 3.1426% | Passed |
| EP8/SP2 / EP8/CP2 | 1.5077% | 3.0806% | 3.1397% | Passed |

The smallest sampled-logits cosine was 0.9993868. No thresholds were relaxed.
AM CP2 recorded a maximum DP-replica NLL span of 0.2164154 (different replicas
process the same case); results use global rank0's CP group, not a replica
average. XT loss continues to use its existing globally reduced public loss
path; sampled logits are checked on every rank. These observations do not
claim bitwise determinism, backward parity or training convergence.
Existing optional-extension/deprecation, deterministic-histogram and collective
shutdown warnings remain visible in the logs. Both dependency environments and
all model implementations were left unchanged. No MedPix/tokenized fixtures
are required by this PR.

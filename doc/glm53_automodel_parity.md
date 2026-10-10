# Add AutoModel to the existing GLM-5.3 HF/XTuner accuracy test

Set `GLM53_AUTOMODEL_PYTHON` and run the original pytest entry point. The
class-scoped fixture runs a single-GPU AutoModel reference once, waits for that
process to release its GPU, then lets `DeterministicDDPTestCase` launch the
existing XTuner workers. No manual reference-generation command or external
`torchrun` is needed. Unset the variable for the original HF/XTuner comparison.

```bash
GLM_5_3_FLASH_PATH=/absolute/path/to/GLM-5.3-Flash-25B \
GLM53_AUTOMODEL_PYTHON=/absolute/path/to/automodel/bin/python \
XTUNER_TEST_WORLD_SIZE=8 \
/absolute/path/to/xtuner-env/bin/python -m pytest -s -v \
  tests/model/test_glm53_text_moe.py::TestGlm53TextMoEAccuracy
```

This runs the original EP1/4/8 parameterization against HF and the same AM
EP1/CP1 reference. It does **not** enable AutoModel EP/CP or add XTuner SP.
For one GPU, use `XTUNER_TEST_WORLD_SIZE=1` and select only:

```text
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEAccuracy::test_fsdp_accuracy[None-1]
```

## Model-only substitution

AutoModel loads through
`NeMoAutoModelForImageTextToText.from_pretrained(..., force_hf=False)`.
The helper checks that the native NeMo GLM class was loaded. No AutoModel
trainer, recipe, dataset, collator or optimizer is involved. Its dependencies
remain in its own interpreter; neither environment is changed by the test.

The original four text cases and two image cases are extracted into one shared
helper. Images still come from `tests/resource/mscoco_twocat_000000039769.jpg`
(single image and image plus horizontal flip). Every test invocation starts
from these raw cases. AutoModel preprocessing runs in the pytest interpreter
through a CPU tensor pipe, preserving HF's tokenizer, processor and chat
template even when the model environment has a different Transformers version.
No pre-tokenized input fixtures are required or persisted.

Temporary reference results live under pytest's temporary directory and are
rebuilt on every invocation. Before comparing results, workers verify input
hashes (including pixels/grid), labels, sample order, positions and token counts.
The reference includes interpreter/package/backend metadata and sampled logits.

HF's original `output.loss` and XTuner's public `CELossConfig` path remain.
AM computes external FP32 CE with exactly one next-token shift. Visual-token
labels are ignored; these are the existing whole-text labels, not assistant-only
SFT supervision. All three pairs use the original loss checks (mean relative
error <3%, cosine >0.97) and sampled-logits checks (per-case relative L2 <5%,
cosine >0.998, full vocabulary at the last eight supervised positions).
Only eval/no-grad forward is tested: no backward, updates, packing or MTP.

Attention choices remain explicit: HF eager; XTuner Torch DSA/indexer and
FlashAttention vision; AutoModel cuDNN/FlashMLA sparse MLA, native Torch KPool,
FLA KDA and SDPA vision. These are not claims of identical kernels. Production
XTuner TileLang/FlashMLA experiments are outside this existing accuracy test.

## Shared-machine environment

| Component | Validated path / version |
|---|---|
| HF/XT Python | `/mnt/shared-storage-user/llmrazor-share/comm_env/pt29_glm2/bin/python` |
| HF/XT packages | PyTorch `2.9.1+cu128`, Transformers `5.17.0` |
| AM Python | `/mnt/shared-storage-user/llmrazor-share/zhangxinsen/dev_envs/automodel/bin/python` |
| AM packages | PyTorch `2.10.0+cu130`, Transformers `5.15.1`, FLA `0.4.2` |
| AM source | `/mnt/shared-storage-user/llmrazor-share/zhangxinsen/Automodel` |
| AM commit | `fda1bf266c78e3723599b9441f1c93b94f26380f` |
| Checkpoint | `/mnt/shared-storage-user/llmrazor-share/model/GLM-5.3-Flash-25B` |

`GLM53_AUTOMODEL_PYTHON` may also point to an executable wrapper ending in
`exec /absolute/path/to/automodel/bin/python "$@"` when that environment needs
its own CUDA library paths or source `PYTHONPATH`. Keep these exports inside the
wrapper so CUDA 13 settings do not leak into the HF/XT CUDA 12.8 process.
The shared machine's self-contained wrapper is:
`/mnt/shared-storage-user/llmrazor-share/zhangxinsen/data/xtuner-automodel-loss-compare-data/forward-parity/am-pr-simplified-20261010/am-python`.
It configures the existing AM environment without sourcing or changing it.

Local Transformers 5.15.1 preprocessing did not supply `image_grid_thw` for
these cases. Using the pytest interpreter for preprocessing avoids that version
mismatch without installing packages or substituting an AM recipe template.

## Validation (2026-10-10, reduced checkpoint, H200)

- CPU input/result contract regressions: 7 passed.
- One pytest command on eight GPUs: original EP1/4/8 cases, 3 passed.
  AM ran the six reference cases exactly once before workers started.
- One-GPU invocation selecting `[None-1]`: 1 passed, including automatic AM execution.
- All three pairwise loss/logits checks passed at the existing thresholds.
  Across the eight-GPU suite, maximum sampled relative L2 was 2.5741% HF→XT,
  3.5018% HF→AM and 2.4247% AM→XT; minimum cosine was 0.9993867.
- Ruff, formatting and diff whitespace checks passed. Existing optional-extension,
  deprecation and collective-shutdown warnings were retained.

These results validate the automatic pytest entry point, not AM EP/CP or
training/backward. Earlier distributed experiment artifacts remain separate.

# GLM 5.3 Flash forward loss and logits parity

A scalar cross entropy can hide changes in full-vocabulary logits. The existing real-checkpoint text test now retains its public loss-path check and also checks the last eight next-token positions against native Transformers. `tests/model/glm53_forward_parity.py` adds image/text replay with FSDP, EP and SP using the same preprocessed inputs on both implementations.

This is eval/no-grad validation: no packing, backward, optimizer step, or MTP. Use a reduced GLM checkpoint that fits a single GPU for the HF reference. The checkpoint is supplied by the caller; it is not downloaded by the test.

## Backend and precision contract

XT uses FlashAttention for the vision tower and torch reference DSA/indexer for the language tower. HF explicitly uses eager: Transformers 5.17 rejects `flash_attention_2` for `Glm5NextVisionModel`. This is an implementation/backend comparison, not a claim that both frameworks use identical kernels. BF16 model compute and FP32 CE are used; TF32 is disabled by the replay runner. CPU metric reductions use FP64.

For each sample, select the final eight supervised positions (or all if fewer than eight), retaining the entire vocabulary at each position. Labels are shifted exactly once. Check shape, finiteness, fixture SHA256, supervised-token count, global positions and target labels before comparing. Sequence padding is excluded. With SP, gather only sampled logits and sum NLL/count within the SP group; EP/world are not loss reduction groups. DP groups deliberately replay identical inputs.

The logits check requires **each sample** to have relative L2 below 0.05 and cosine above 0.998, independently of the existing loss-vector cosine >0.97 and mean relative loss error <0.03. These are BF16 regression bounds, not bitwise-parity guarantees. The thresholds are fixed in the helper, not adjusted by the replay runner to fit results. Reports include per-sample maximum absolute error as a diagnostic. A CPU regression verifies that a common logits offset, which leaves CE unchanged, fails the logits check.

## Prepare portable image fixtures

Run from the repository root in the XTuner environment. Supply absolute paths for your model, XTuner-format image JSONL, media root and a new output directory:

```bash
python tests/model/prepare_glm53_parity_inputs.py \
  --model /models/GLM-5.3-Flash-25B \
  --jsonl /data/medpix/train.jsonl --media-root /data/medpix \
  --limit 20 --output /results/glm53/inputs
```

The preparer uses the real GLM tokenizer/processor once (max pixels 1048576, max length 16384). It writes raw, unshifted `input_ids`, `labels`, `mm_token_type_ids`, `pixel_values`, `image_grid_thw`, and shifted-input `logit_positions` as safetensors. The manifest stores relative filenames, hashes, sample IDs and supervised counts. Dataset/images/weights are external fixtures and are not added to Git. This replay entry point currently supports image inputs, not video/audio.

## Run sequentially

Use the same checkpoint for all runs. All output directories must be new; the runner refuses overwrites. Both entry points run from the XTuner checkout with its test dependencies installed; separate processes release model memory between implementations.

```bash
# Native HF reference on one GPU.
torchrun --standalone --nproc-per-node=1 tests/model/glm53_forward_parity.py \
  --backend hf --model /models/GLM-5.3-Flash-25B \
  --inputs /results/glm53/inputs --output /results/glm53/hf

# XT FSDP EP1/SP1 baseline.
torchrun --standalone --nproc-per-node=1 tests/model/glm53_forward_parity.py \
  --backend xt --model /models/GLM-5.3-Flash-25B \
  --inputs /results/glm53/inputs --reference /results/glm53/hf \
  --ep-size 1 --sp-size 1 --output /results/glm53/xt-ep1-sp1

# Run both SP1 and SP2 on eight GPUs, with distinct output directories.
torchrun --standalone --nproc-per-node=8 tests/model/glm53_forward_parity.py \
  --backend xt --model /models/GLM-5.3-Flash-25B \
  --inputs /results/glm53/inputs --reference /results/glm53/hf \
  --ep-size 8 --sp-size 2 --output /results/glm53/xt-ep8-sp2
```

Set `--sp-size 1` and another output directory for EP8/SP1. The program exits nonzero on input-contract, logits, or loss-check failure. Each run records metadata, sampled tensors and `loss.jsonl`; comparisons also write `comparison.json`. This real-checkpoint replay is opt-in and is not advertised as an automatically provisioned CI test.

Run the inexpensive metric regressions with `pytest tests/model/test_logits_metrics.py`. The existing real-checkpoint text regression remains in `TestGlm53TextMoEAccuracy.test_fsdp_accuracy`, with `GLM_5_3_FLASH_PATH` and `XTUNER_TEST_WORLD_SIZE` selecting the local checkpoint and GPU count.

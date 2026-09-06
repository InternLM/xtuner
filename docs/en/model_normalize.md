# HF model normalization

XTuner provides a standalone conversion entry point for turning training
checkpoints that already have HF tensor names into standard HF shard layouts.
It does not upload models or run inference validation.

## BF16/FP16 repack

```bash
bash xtuner/tools/model_normalize/run_model_normalize.sh repack \
  --source /path/to/source \
  --output /path/to/output \
  --shard-size-gb 4
```

All tensors, including MTP tensors, are retained. Non-weight files such as
`config.json`, tokenizer files, chat templates, and an existing
`generation_config.json` are copied. A model-team supplied generation config
can be explicitly added with `--generation-config`.

## FP8 conversion

For a reference-guided conversion:

```bash
bash xtuner/tools/model_normalize/run_model_normalize.sh to-fp8 \
  --source /path/to/bf16 \
  --output /path/to/fp8 \
  --reference /path/to/reference-fp8 \
  --max-save-workers 4
```

Without a reference, the heuristic policy must be explicit:

```bash
python -m xtuner.tools.model_normalize to-fp8 \
  --source /path/to/bf16 --output /path/to/fp8 --policy heuristic
```

FP8 conversion requires CUDA. Save workers are bounded so conversion and disk
writes can overlap without submitting an unbounded number of shard writes.

## MTP and output safety

The tool has no option that drops MTP. Repack and FP8 conversion preserve the
complete tensor key set and the source MTP configuration. Source and output
directories must differ, and a non-empty output is not overwritten unless
`--overwrite` is passed.

The tool intentionally does not run a full-model validation or SHA256 scan.
Those checks remain optional release-side operations.

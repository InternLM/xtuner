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
`generation_config.json` are copied from the source.

## Base-model assets (LICENSE + generation_config)

Both `repack` and `to-fp8` accept `--base-model-dir <dir>`, a directory of
user-supplied release assets applied *after* conversion completes:

- `generation_config.json` (if present) is copied verbatim into the output,
  overwriting any existing file (the overwrite is logged).
- `LICENSE` (if present) is copied into the output with its Copyright line
  rewritten to the fixed string `Copyright 2025-2026 Shanghai AI Laboratory`;
  the rest of the license text (e.g. the MIT permission grant) is left
  untouched. An existing output `LICENSE` is overwritten (logged). If no
  Copyright line is found, the file is written unchanged with a warning.

When `--base-model-dir` is omitted, conversion behavior is unchanged. A
source-supplied `generation_config.json` remains in the output as-is.

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

## One-command GLM-5.2 example

`examples/run_glm52.sh` is a small wrapper around the generic CLI. It writes
the two release variants below one level under `OUTPUT_ROOT`.

For the BF16, standard-shard variant:

```bash
SOURCE_DIR=/path/to/glm52-hf-source \
OUTPUT_ROOT=/path/to/release \
bash xtuner/tools/model_normalize/examples/run_glm52.sh bf16
```

For the reference-guided FP8 variant:

```bash
SOURCE_DIR=/path/to/glm52-hf-source \
OUTPUT_ROOT=/path/to/release \
REFERENCE_DIR=/path/to/glm52-fp8-reference \
SHARD_SIZE_GB=4 \
MAX_SAVE_WORKERS=4 \
bash xtuner/tools/model_normalize/examples/run_glm52.sh fp8
```

To stamp a user-supplied `LICENSE` and `generation_config.json` into the
product, set `BASE_MODEL_DIR` (optional; applies to both variants):

```bash
SOURCE_DIR=/path/to/glm52-hf-source \
OUTPUT_ROOT=/path/to/release \
BASE_MODEL_DIR=/path/to/base-model/glm5-2 \
bash xtuner/tools/model_normalize/examples/run_glm52.sh bf16
```

`run_glm52.sh` environment variables:

| Variable | BF16 | FP8 | Default | Description |
|---|---|---|---|---|
| `SOURCE_DIR` | required | required | — | Input HF model directory |
| `OUTPUT_ROOT` | required | required | — | Output root directory |
| `SHARD_SIZE_GB` | optional | optional | `4` | Target shard size in GiB |
| `REFERENCE_DIR` | not used | required | — | FP8 reference model directory |
| `MAX_SAVE_WORKERS` | not used | optional | `4` | Parallel FP8 shard-save workers |
| `BASE_MODEL_DIR` | optional | optional | unset | Directory with user-supplied `LICENSE` and `generation_config.json` |

The wrapper produces this shape (the actual shard count depends on the input
and `SHARD_SIZE_GB`):

```text
<OUTPUT_ROOT>/
├── 20_hf_bf16_mtp/
│   ├── model-00001-of-00NNN.safetensors
│   ├── ...
│   ├── model.safetensors.index.json
│   ├── config.json
│   ├── tokenizer.json / tokenizer_config.json
│   ├── chat_template.jinja        # when supplied by the source
│   ├── generation_config.json     # from source or base-model-dir
│   └── LICENSE                    # when supplied via --base-model-dir (Copyright rewritten)
└── 20_hf_fp8_mtp/
    ├── model-00001-of-00NNN.safetensors
    ├── ...
    ├── model.safetensors.index.json
    ├── config.json                 # includes the FP8 quantization metadata
    ├── tokenizer/chat-template files
    ├── generation_config.json      # from source or base-model-dir
    └── LICENSE                     # when supplied via --base-model-dir (Copyright rewritten)
```

The BF16 output keeps the original tensor keys, including MTP keys, and only
repackages them into standard HF shards. The FP8 output also keeps the complete
key set, while converting selected weights to FP8 and writing their matching
`*_scale_inv` tensors. Both variants regenerate a consistent
`model.safetensors.index.json`; neither variant uploads to the Hub or performs a
full-model validation/SHA256 scan.

## MTP and output safety

The tool has no option that drops MTP. Repack and FP8 conversion preserve the
complete tensor key set and the source MTP configuration. Source and output
directories must differ, and a non-empty output is not overwritten unless
`--overwrite` is passed.

The tool intentionally does not run a full-model validation or SHA256 scan.
Those checks remain optional release-side operations.

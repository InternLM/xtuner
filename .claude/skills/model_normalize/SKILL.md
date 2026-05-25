---
name: model_normalize
description: Normalize an HF model checkpoint — quantize bf16/fp16 weights to per-block FP8, and/or repack engine-named safetensors into HF-standard ~4GB shards.
---

# model_normalize

Two complementary capabilities for getting a training-engine checkpoint into a
format that HF / vLLM / SGLang can consume:

1. **FP8 quantization** of large linears, in two modes:
   - **Reference-guided** — point at an existing FP8 checkpoint of the same
     architecture; the tool reads its index and reproduces the exact same
     quantization decisions on the source.
   - **Heuristic** — apply a built-in rule set that quantizes only the large
     matrix multiplications (attention, MLP, MoE experts) and keeps norms,
     routers, biases, embeddings, and vision tower in their original dtype.
2. **Standard repack** — re-shard arbitrary safetensors directories into the
   HF-standard ``model-{i:05d}-of-{n:05d}.safetensors`` layout with ~4GB
   shards and a fresh ``model.safetensors.index.json``.

When ``to-fp8`` detects non-standard source shard names, it automatically
runs the repack step on the quantized output, so the user-visible result is
always a clean HF-conformant directory.

## Layout

```
.claude/skills/model_normalize/
├── SKILL.md                       # this file
├── model_normalize.py             # top-level CLI (to-fp8 / repack)
├── hf_to_fp8.py                   # FP8 conversion library + standalone CLI
├── heuristics.py                  # codified default quantization rule set
├── repack_hf.py                   # 4GB-shard repacker library + CLI
└── convert_interns2_to_fp8.sh     # example invocation for InternS2
```

## Usage

### Mode 1 — Reference-guided FP8 conversion

```bash
python model_normalize.py to-fp8 \
    --source    /path/to/bf16-model \
    --output    /path/to/fp8-model \
    --reference /path/to/fp8-reference
```

Behavior:
- Loads the reference's ``model.safetensors.index.json``.
- For each ``<name>_scale_inv`` entry in the reference index, marks ``<name>``
  as an FP8-quantized tensor.
- Applies that exact set to the source as a literal name match (no regex
  surprise; works even when names don't follow any pattern).

### Mode 2 — Heuristic FP8 conversion (no reference)

```bash
python model_normalize.py to-fp8 \
    --source /path/to/bf16-model \
    --output /path/to/fp8-model
```

The rule set is defined in ``heuristics.py``. Concretely, **quantize iff** the
tensor name matches one of:

| Group                            | Pattern (anchored)                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Self-attention projections       | ``[…]layers.<i>.self_attn.{q,k,v,o}_proj.weight``                                                                   |
| Dense MLP                        | ``[…]layers.<i>.mlp.{gate,up,down}_proj.weight``                                                                    |
| MoE per-expert (unfused)         | ``[…]layers.<i>.mlp.experts.<j>.{gate,up,down}_proj.weight``                                                        |
| MoE fused-expert (3D tensors)    | ``[…]layers.<i>.mlp.experts.{gate_up_proj, down_proj}``                                                             |
| MoE shared expert                | ``[…]layers.<i>.mlp.shared_expert.{gate,up,down}_proj.weight``                                                      |
| Linear-attn wide projections     | ``[…]layers.<i>.linear_attn.{in_proj_qkv, in_proj_z, out_proj}.weight``                                             |

The ``[…]`` prefix allows ``model.`` and ``model.language_model.``; the same
patterns repeat under ``mtp.layers.<i>.`` for multi-token-prediction blocks.

**Everything else is kept in the original dtype**, in particular:
- all norms (``*norm*``, ``*_layernorm.*``)
- MoE routers (``mlp.gate.weight``, ``mlp.shared_expert_gate.weight``)
- embeddings (``embed_tokens``, ``lm_head``, ``pos_embed``, ``patch_embed``, ``mtp.fc``)
- vision tower (``model.visual.*``)
- all ``*.bias``
- ``linear_attn`` control-flow tensors (``A_log``, ``conv1d.weight``, ``dt_bias``, ``in_proj_a/b.weight``, ``norm.weight``)

This rule set was validated against the InternS2 / Qwen3-MoE FP8 references:
it reproduces their quantization decisions on every tensor (no false
positives or false negatives).

### Mode 3 — Standalone repack (no quantization)

```bash
python model_normalize.py repack \
    --source /path/to/messy-shards \
    --output /path/to/standard-shards \
    --shard-size-gb 4
```

Greedy bin-packs tensors into ``model-{i:05d}-of-{n:05d}.safetensors`` shards
of at most ``--shard-size-gb`` GiB, writes a fresh
``model.safetensors.index.json`` (including ``metadata.total_size``), and
copies all non-safetensors files (config, tokenizer, etc.) through.

## When to use which mode

| Situation                                                                    | Mode               |
| ---------------------------------------------------------------------------- | ------------------ |
| You already have an FP8 reference for this architecture.                     | Reference-guided   |
| New architecture / first-time conversion; standard naming.                   | Heuristic          |
| Quantized model has engine-specific shard names; need HF-conformant output.  | (auto, see below)  |
| You only need to standardize shard naming; weights are already FP8 / bf16.   | ``repack``         |

The ``to-fp8`` subcommand auto-detects non-standard source shard names (anything
that isn't ``model.safetensors`` or ``model-NNNNN-of-NNNNN.safetensors``) and
chains the repack step. Pass ``--no-repack`` to keep the source naming.

## Quantization details

- Block size **128 × 128**, dtype ``float8_e4m3fn`` with saturating cast.
- Per-block scale stored as ``<name>_scale_inv`` (``float32``) in the same
  shard as the parent weight. Layout matches what vLLM / SGLang load for
  per-block FP8.
- 3D MoE-expert tensors (shape ``(E, D0, D1)``) are quantized per-expert; the
  per-expert scales are stacked along a leading dim.
- ``config.json`` receives a ``quantization_config`` block with
  ``quant_method=fp8``, ``weight_block_size=[128, 128]``,
  ``scale_fmt=ue8m0``, and an explicit ``modules_to_not_convert`` list
  derived from the actual quantization decisions.

## Post-conversion handoff (mandatory for the agent)

After **any** successful run of `to-fp8` or `repack`, the agent must ask the
user — via `AskUserQuestion` — where to **place** the produced
`*.safetensors` and `model.safetensors.index.json` files. The conversion
output is treated as a staging area; the user decides the final home.

Offer at least these options:
1. **Keep in place** — leave the files at the `--output` path as-is.
2. **Copy to a new directory** — let the user specify a target; copy the
   shards + index there. Always **copy** (`cp`), never move — the
   `--output` staging directory must remain intact unless the user
   explicitly asks to delete it as a separate step.
3. **Copy into an existing model directory** — copy shards/index into a
   user-specified directory while preserving its other files (tokenizer,
   config, etc.).

If the user picks (2) or (3), execute the copy with `cp` (confirming the
destination path back to the user before any overwrite). Do not infer the
destination from earlier turns — the user may have a fresh target in mind
each time. Cleaning up the staging directory afterwards is a separate,
user-initiated step — never delete it as part of the handoff.

## Example: InternS2 Preview

```bash
python model_normalize.py to-fp8 \
    --source    /mnt/shared-storage-user/puyudelivery/user/yehaochen/models/InternS2PreviewCandidate5 \
    --output    /tmp/InternS2-FP8 \
    --reference /mnt/shared-storage-user/puyudelivery/user/yehaochen/models/InternS2PreviewCandidate5-FP8
```

The source uses engine-specific names like
``model-language-0001-fused-save_rank0.safetensors``; the output will be
repacked into the standard ``model-00001-of-NNNNN.safetensors`` layout.

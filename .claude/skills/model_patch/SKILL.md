---
name: model_patch
description: Merge a base HF model dir with an extra HF model dir by index diff — take tensors only present in extra, append them to a new output dir in HF-standard shard layout. Base wins on overlap.
---

# model_patch

Produce ``out = base ∪ (extra \ base)`` at the **tensor level**, by diffing the
two ``model.safetensors.index.json`` files. Useful when ``extra`` is a sibling
checkpoint (different training run, fine-tune, or fp8 conversion) that holds a
few tensors the base is missing — e.g. an updated head, a newly-added MTP
block, or auxiliary projections — and you want a self-contained merged dir
without re-running the whole conversion pipeline.

Overlap rule: **base wins**. Any tensor present in both indexes is taken from
``base``; the corresponding entry in ``extra`` is ignored.

## Layout

```
.claude/skills/model_patch/
├── SKILL.md          # this file
└── model_patch.py    # CLI + library
```

## Usage

```bash
python model_patch.py \
    --base  /path/to/base-model \
    --extra /path/to/extra-model \
    --out   /path/to/merged-model
```

## What it writes

- ``model-{i:05d}-of-{N:05d}.safetensors`` shards (total ``N`` = base shards +
  newly-packed shards holding the extra-only tensors)
- a fresh ``model.safetensors.index.json``
- **every non-tensor file from ``base``** — ``config.json``, tokenizer files,
  modeling/configuration ``.py``, ``generation_config.json``, README, etc. —
  copied verbatim so ``out`` is loadable on its own. Subdirectories are
  copied recursively. Existing entries in ``out`` are overwritten.

``extra``'s aux files are deliberately **not** copied. The skill's overlap
rule is "base wins" at the tensor level, and that extends to configs: extra
often carries engine-specific edits or modality keys that don't match the
tensor topology we keep. If you need fields from ``extra``'s ``config.json``
(e.g. a new modality), merge them into ``out/config.json`` as a separate,
explicit step.

## How shards are produced

- **Base shards**: re-emitted under the new ``-of-N`` total. By default the
  bytes are fully copied so ``out`` is independent of ``base``. Pass
  ``--hardlink`` to use ``os.link`` instead (fast, but the two trees share
  inodes — editing one mutates the other; deleting one keeps the other
  intact since hardlinks are symmetric).
- **Extra-only tensors**: greedily bin-packed into new shards of at most
  ``--shard-size-gb`` GiB (default 4), appended after the base shards.

So a base with 12 shards plus 5 GB of extra-only tensors yields:
``model-00001-of-00014.safetensors`` … ``model-00012-of-00014.safetensors``
(copies of the base shards) and
``model-00013-of-00014.safetensors`` / ``model-00014-of-00014.safetensors``
(freshly written, containing only the extra tensors).

## When to use this skill

| Situation                                                                              | Use this? |
| -------------------------------------------------------------------------------------- | --------- |
| ``extra`` contains a few tensors ``base`` is missing; everything else is identical.    | Yes       |
| You need to merge two checkpoints whose overlapping tensors **differ** in value.       | No — this skill silently keeps base; you probably want a manual decision per key. |
| ``extra`` is a full quantization of ``base`` and you want the quantized weights.       | No — use ``model_normalize`` with ``--reference extra`` instead. |
| You want a single ``model.safetensors`` instead of standard shards.                    | No — repack downstream. |

## Required usage protocol (mandatory for the agent)

Before invoking ``model_patch.py``:

1. Confirm the three paths (``--base``, ``--extra``, ``--out``) with the user
   via ``AskUserQuestion``. Never infer them from earlier conversation.
2. Show the user the **diff summary** before writing, by reading the two
   ``model.safetensors.index.json`` files and reporting:
   - number of tensors in base
   - number of tensors in extra
   - number of extra-only keys that will actually be added
   - number of overlapping keys that will be dropped from extra
3. Ask the user to confirm those numbers before launching the run.

After a successful run, **do not** ask the user where the output should go —
``--out`` is already the final destination the user named. Just report what
landed there (shard count, index path, total size, and that base's aux files
were mirrored over). Unlike ``model_normalize``, ``model_patch`` has no
staging-area concept: there is no temp dir, and the caller already made the
placement decision when they passed ``--out``.

If the user wanted any of ``extra``'s aux files (e.g. a ``config.json`` that
documents a new modality), call that out — those are not copied by the
skill and must be merged in by hand.

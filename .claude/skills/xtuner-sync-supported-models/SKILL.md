---
name: xtuner-sync-supported-models
description: Synchronize xtuner's supported model documentation (docs/en/pretrain_sft/advanced_tutorial/model.md and docs/zh_cn/pretrain_sft/advanced_tutorial/model.md) with the actual Config classes defined under xtuner/v1/model/. Use when (1) new TransformerConfig, MoEConfig, or BaseComposeConfig subclasses are added, removed, or renamed in xtuner/v1/model/, (2) existing model configs change their inheritance hierarchy, scale, or HuggingFace counterpart, or (3) a code review or user request points out that model.md is out of sync with the codebase.
---

# Update XTuner Supported Model Docs

Keep the English and Chinese `model.md` files synchronized with the actual Config classes in `xtuner/v1/model/`.

## Scan the Codebase

Run the bundled scan script from the xtuner project root to discover all Config classes and their inheritance:

```bash
python3 .agents/skills/xtuner-sync-supported-models/scripts/scan_model_configs.py
```

The script outputs JSON with two keys:
- `configs`: list of every `*Config` class under `xtuner/v1/model/` with its parent classes and file path
- `children`: parent-to-children mapping for the hierarchy tree

## What to Update

Compare the script output against the two files:
- `docs/en/pretrain_sft/advanced_tutorial/model.md`
- `docs/zh_cn/pretrain_sft/advanced_tutorial/model.md`

Both files share the same structure and must stay in sync:

1. **Base Config Classes** — configs that directly inherit from `TransformerConfig` (or `MoEConfig`) and provide a `from_hf` classmethod for loading HuggingFace weights
2. **Concrete Model Configs** — fixed-scale subclasses of the base configs above
3. **Compose Models** — multimodal configs that inherit from `BaseComposeConfig`
4. **Inheritance Hierarchy** — a text tree showing the full `XTunerBaseModelConfig` hierarchy

### Rules for the Base Config table

Include these direct descendants of `TransformerConfig`/`MoEConfig`:
- `Qwen2DenseConfig`
- `Qwen3DenseConfig`
- `DeepSeekV3Config`
- `GptOssConfig`
- `Qwen3MoEConfig`

Exclude from the base table:
- `MoEConfig` — it is an intermediate base class, not a usable model family
- `Qwen3_5_VLTextMoEConfig` — it is an intermediate base with only one concrete child; its child `Qwen3_5_VLTextMoE35BA3BConfig` belongs under the MoE concrete table

### Rules for the Concrete Model table

Include every concrete subclass that has fixed parameter defaults. For each row note:
- `Config Class`
- `Base Class / Family`
- `Architecture Type`: `Dense`, `MoE`, `Dense (VL backbone)`, `MoE (VL backbone)`
- `Scale / Notes`: parameter count or total/activated size; for VL backbones note "for multimodal"

`DeepSeekV3Config` appears here even though it has no separate base entry (it is both base and concrete).

### Rules for the Compose Models section

Include three sub-tables:
1. **Compose Base Config Classes** — `Qwen3VLBaseConfig`, `InternVLBaseConfig`, `InternS1BaseConfig`
   - `Qwen3VLBaseConfig`: VL model based on Qwen3 text backbone
   - `InternVLBaseConfig`: VL model based on InternViT + Qwen3
   - `InternS1BaseConfig`: Science multimodal model based on InternViT + Qwen3
2. **Concrete Compose Model Configs** — every subclass of the above bases; for each row note the wrapped `Text Config` and scale

### Rules for the Inheritance Hierarchy tree

Rebuild the tree from `XTunerBaseModelConfig` with two top-level branches:

```text
XTunerBaseModelConfig
├── TransformerConfig
│   ├── Dense Models
│   │   ├── Qwen2DenseConfig
│   │   │   └── Qwen2Dense7BConfig
│   │   └── Qwen3DenseConfig
│   │       ├── Qwen3Dense8BConfig
│   │       ├── Qwen3Dense4BConfig
│   │       ├── Qwen3Dense0P6BConfig
│   │       ├── Qwen3VLTextDense4BConfig
│   │       └── Qwen3VLTextDense8BConfig
│   └── MoE Models (via MoEConfig)
│       ├── DeepSeekV3Config
│       ├── GptOssConfig
│       │   ├── GptOss21BA3P6Config
│       │   └── GptOss117BA5P8Config
│       ├── Qwen3MoEConfig
│       │   ├── Qwen3MoE30BA3Config
│       │   ├── Qwen3MoE235BA22Config
│       │   ├── Qwen3MoEFoPEConfig
│       │   ├── Qwen3VLTextMoE30BA3Config
│       │   └── Qwen3VLTextMoE235BA22Config
│       └── Qwen3_5_VLTextMoEConfig
│           └── Qwen3_5_VLTextMoE35BA3BConfig
└── BaseComposeConfig
    ├── Qwen3VLBaseConfig
    │   ├── Qwen3VLMoE30BA3Config
    │   ├── Qwen3VLMoE235BA22Config
    │   ├── Qwen3VLDense4BConfig
    │   ├── Qwen3VLDense8BConfig
    │   └── Qwen3_5_BaseConfig
    │       └── Qwen3_5_VLMoE35BA3Config
    ├── InternVLBaseConfig
    │   ├── InternVL3P5Dense8BConfig
    │   ├── InternVL3P5MoE30BA3Config
    │   └── InternVL3P5Dense1BConfig
    └── InternS1BaseConfig
        ├── InternS1Config
        └── InternS1MiniConfig
```

When new configs are added, insert them into the appropriate branch following the same indentation style.

## Translation Notes

Keep the Chinese `model.md` (`docs/zh_cn/...`) structurally identical to the English one. Translate:
- Section headings
- Table header cells
- Description cells (e.g., "Image / Video + Text" → "图像/视频 + 文本")
- Scale descriptions (e.g., "~7B parameters" → "约 7B 参数", "FoPE variant" → "FoPE 变体")

Do **not** translate Config class names, file paths, or code identifiers.

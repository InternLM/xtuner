# Model

XTuner v1's `TrainEngine` supports a variety of Transformer architectures through different `TransformerConfig` subclasses. The documentation below summarizes the currently supported models (RL-related configs are excluded).

## Base Config Classes

The following table lists the **base config classes** that define each model family. They provide the `from_hf` interface for loading pretrained weights from HuggingFace.

| Base Config Class | Model Family | Architecture Type | HuggingFace Counterpart |
|---|---|---|---|
| `Qwen2DenseConfig` | Qwen2 Dense | Dense | `Qwen2ForCausalLM` |
| `Qwen3DenseConfig` | Qwen3 Dense | Dense | `Qwen3ForCausalLM` |
| `DeepSeekV3Config` | DeepSeek-V3 | MoE | `DeepseekV3ForCausalLM` |
| `GptOssConfig` | GPT-OSS | MoE | `GptOssForCausalLM` |
| `Qwen3MoEConfig` | Qwen3 MoE | MoE | `Qwen3MoeForCausalLM` |

## Concrete Model Configs

The following table lists the **concrete model configs** that inherit from the base classes above. Each config corresponds to a specific model scale or variant.

| Config Class | Base Class / Family | Architecture Type | Scale / Notes |
|---|---|---|---|
| `Qwen2Dense7BConfig` | `Qwen2DenseConfig` | Dense | ~7B parameters |
| `Qwen3Dense8BConfig` | `Qwen3DenseConfig` | Dense | ~8B parameters |
| `Qwen3Dense4BConfig` | `Qwen3DenseConfig` | Dense | ~4B parameters |
| `Qwen3Dense0P6BConfig` | `Qwen3DenseConfig` | Dense | ~0.6B parameters |
| `Qwen3VLTextDense4BConfig` | `Qwen3DenseConfig` | Dense (VL backbone) | ~4B parameters, for multimodal |
| `Qwen3VLTextDense8BConfig` | `Qwen3DenseConfig` | Dense (VL backbone) | ~8B parameters, for multimodal |
| `DeepSeekV3Config` | — | MoE | ~671B total / ~37B activated |
| `GptOss21BA3P6Config` | `GptOssConfig` | MoE | ~21B total / ~3.6B activated |
| `GptOss117BA5P8Config` | `GptOssConfig` | MoE | ~117B total / ~5.8B activated |
| `Qwen3MoE30BA3Config` | `Qwen3MoEConfig` | MoE | ~30B total / ~3B activated |
| `Qwen3MoE235BA22Config` | `Qwen3MoEConfig` | MoE | ~235B total / ~22B activated |
| `Qwen3MoEFoPEConfig` | `Qwen3MoEConfig` | MoE | FoPE (Frequency-based Position Embedding) variant |
| `Qwen3VLTextMoE30BA3Config` | `Qwen3MoEConfig` | MoE (VL backbone) | ~30B total, for multimodal |
| `Qwen3VLTextMoE235BA22Config` | `Qwen3MoEConfig` | MoE (VL backbone) | ~235B total, for multimodal |
| `Qwen3_5_VLTextMoE35BA3BConfig` | `Qwen3_5_VLTextMoEConfig` | MoE (VL backbone) | ~35B total / ~3B activated, for multimodal |

## Compose Models

In addition to pure text models, XTuner also supports **multimodal compose models** that combine a vision encoder, a projector, and a language model. These configs inherit from `BaseComposeConfig` rather than `TransformerConfig` directly, but they wrap the text configs listed above.

### Compose Base Config Classes

| Base Config Class | Model Family | Modality | Description |
|---|---|---|---|
| `Qwen3VLBaseConfig` | Qwen3-VL | Image / Video + Text | VL model based on Qwen3 text backbone |
| `InternVLBaseConfig` | InternVL | Image + Text | VL model based on InternViT + Qwen3 |
| `InternS1BaseConfig` | InternS1 | Image + Text | Science multimodal model based on InternViT + Qwen3 |

### Concrete Compose Model Configs

| Config Class | Compose Base / Family | Text Config | Scale / Notes |
|---|---|---|---|
| `Qwen3VLMoE30BA3Config` | `Qwen3VLBaseConfig` | `Qwen3VLTextMoE30BA3Config` | ~30B total, MoE VL |
| `Qwen3VLMoE235BA22Config` | `Qwen3VLBaseConfig` | `Qwen3VLTextMoE235BA22Config` | ~235B total, MoE VL |
| `Qwen3VLDense4BConfig` | `Qwen3VLBaseConfig` | `Qwen3VLTextDense4BConfig` | ~4B parameters, Dense VL |
| `Qwen3VLDense8BConfig` | `Qwen3VLBaseConfig` | `Qwen3VLTextDense8BConfig` | ~8B parameters, Dense VL |
| `Qwen3_5_VLMoE35BA3Config` | `Qwen3_5_BaseConfig` | `Qwen3_5_VLTextMoE35BA3BConfig` | ~35B total / ~3B activated, MoE VL |
| `InternVL3P5Dense8BConfig` | `InternVLBaseConfig` | `Qwen3Dense8BConfig` | ~8B parameters, Dense VL |
| `InternVL3P5MoE30BA3Config` | `InternVLBaseConfig` | `Qwen3MoE30BA3Config` | ~30B total, MoE VL |
| `InternVL3P5Dense1BConfig` | `InternVLBaseConfig` | `Qwen3Dense0P6BConfig` | ~1B parameters, Dense VL |
| `InternS1Config` | `InternS1BaseConfig` | `Qwen3MoE235BA22Config` | ~235B total, MoE multimodal |
| `InternS1MiniConfig` | `InternS1BaseConfig` | `Qwen3Dense8BConfig` | ~8B parameters, Dense multimodal |

## Inheritance Hierarchy

The following diagram shows the complete inheritance hierarchy of all config classes supported by `TrainEngine`, including both `TransformerConfig` and `BaseComposeConfig` branches.

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

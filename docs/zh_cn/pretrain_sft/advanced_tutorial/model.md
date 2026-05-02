# 模型

XTuner v1 的 `TrainEngine` 通过不同的 `TransformerConfig` 子类支持多种 Transformer 架构。下文总结了当前支持的模型（不包含 RL 相关配置）。

## 基类配置

下表列出**基类配置**，它们定义了各个模型系列，并提供了从 HuggingFace 加载预训练权重的 `from_hf` 接口。

| 基类配置 | 模型系列 | 架构类型 | 对应的 HuggingFace 模型 |
|---|---|---|---|
| `Qwen2DenseConfig` | Qwen2 Dense | Dense | `Qwen2ForCausalLM` |
| `Qwen3DenseConfig` | Qwen3 Dense | Dense | `Qwen3ForCausalLM` |
| `DeepSeekV3Config` | DeepSeek-V3 | MoE | `DeepseekV3ForCausalLM` |
| `GptOssConfig` | GPT-OSS | MoE | `GptOssForCausalLM` |
| `Qwen3MoEConfig` | Qwen3 MoE | MoE | `Qwen3MoeForCausalLM` |

## 具体模型配置

下表列出**具体模型配置**，它们继承自上述基类，每个配置对应特定的模型规模或变体。

| 配置类名 | 基类 / 所属系列 | 架构类型 | 规模 / 说明 |
|---|---|---|---|
| `Qwen2Dense7BConfig` | `Qwen2DenseConfig` | Dense | 约 7B 参数 |
| `Qwen3Dense8BConfig` | `Qwen3DenseConfig` | Dense | 约 8B 参数 |
| `Qwen3Dense4BConfig` | `Qwen3DenseConfig` | Dense | 约 4B 参数 |
| `Qwen3Dense0P6BConfig` | `Qwen3DenseConfig` | Dense | 约 0.6B 参数 |
| `Qwen3VLTextDense4BConfig` | `Qwen3DenseConfig` | Dense（VL 文本主干） | 约 4B 参数，用于多模态 |
| `Qwen3VLTextDense8BConfig` | `Qwen3DenseConfig` | Dense（VL 文本主干） | 约 8B 参数，用于多模态 |
| `DeepSeekV3Config` | — | MoE | 约 671B 总参 / 约 37B 激活 |
| `GptOss21BA3P6Config` | `GptOssConfig` | MoE | 约 21B 总参 / 约 3.6B 激活 |
| `GptOss117BA5P8Config` | `GptOssConfig` | MoE | 约 117B 总参 / 约 5.8B 激活 |
| `Qwen3MoE30BA3Config` | `Qwen3MoEConfig` | MoE | 约 30B 总参 / 约 3B 激活 |
| `Qwen3MoE235BA22Config` | `Qwen3MoEConfig` | MoE | 约 235B 总参 / 约 22B 激活 |
| `Qwen3MoEFoPEConfig` | `Qwen3MoEConfig` | MoE | FoPE（基于频率的位置编码）变体 |
| `Qwen3VLTextMoE30BA3Config` | `Qwen3MoEConfig` | MoE（VL 文本主干） | 约 30B 总参，用于多模态 |
| `Qwen3VLTextMoE235BA22Config` | `Qwen3MoEConfig` | MoE（VL 文本主干） | 约 235B 总参，用于多模态 |
| `Qwen3_5_VLTextMoE35BA3BConfig` | `Qwen3_5_VLTextMoEConfig` | MoE（VL 文本主干） | 约 35B 总参 / 约 3B 激活，用于多模态 |

## Compose 多模态模型

除了纯文本模型外，XTuner 还支持**多模态 Compose 模型**，它们将视觉编码器（vision encoder）、投影层（projector）和语言模型组合在一起。这些配置直接继承自 `BaseComposeConfig` 而非 `TransformerConfig`，但其内部封装了上文列出的文本模型配置。

### Compose 基类配置

| 基类配置 | 模型系列 | 模态 | 说明 |
|---|---|---|---|
| `Qwen3VLBaseConfig` | Qwen3-VL | 图像/视频 + 文本 | 基于 Qwen3 文本主干的 VL 模型 |
| `InternVLBaseConfig` | InternVL | 图像 + 文本 | 基于 InternViT + Qwen3 的 VL 模型 |
| `InternS1BaseConfig` | InternS1 | 图像 + 文本 | 基于 InternViT + Qwen3 的科学多模态模型 |

### 具体 Compose 模型配置

| 配置类名 | Compose 基类 / 系列 | 文本模型配置 | 规模 / 说明 |
|---|---|---|---|
| `Qwen3VLMoE30BA3Config` | `Qwen3VLBaseConfig` | `Qwen3VLTextMoE30BA3Config` | 约 30B 总参，MoE VL |
| `Qwen3VLMoE235BA22Config` | `Qwen3VLBaseConfig` | `Qwen3VLTextMoE235BA22Config` | 约 235B 总参，MoE VL |
| `Qwen3VLDense4BConfig` | `Qwen3VLBaseConfig` | `Qwen3VLTextDense4BConfig` | 约 4B 参数，Dense VL |
| `Qwen3VLDense8BConfig` | `Qwen3VLBaseConfig` | `Qwen3VLTextDense8BConfig` | 约 8B 参数，Dense VL |
| `Qwen3_5_VLMoE35BA3Config` | `Qwen3_5_BaseConfig` | `Qwen3_5_VLTextMoE35BA3BConfig` | 约 35B 总参 / 约 3B 激活，MoE VL |
| `InternVL3P5Dense8BConfig` | `InternVLBaseConfig` | `Qwen3Dense8BConfig` | 约 8B 参数，Dense VL |
| `InternVL3P5MoE30BA3Config` | `InternVLBaseConfig` | `Qwen3MoE30BA3Config` | 约 30B 总参，MoE VL |
| `InternVL3P5Dense1BConfig` | `InternVLBaseConfig` | `Qwen3Dense0P6BConfig` | 约 1B 参数，Dense VL |
| `InternS1Config` | `InternS1BaseConfig` | `Qwen3MoE235BA22Config` | 约 235B 总参，MoE 多模态 |
| `InternS1MiniConfig` | `InternS1BaseConfig` | `Qwen3Dense8BConfig` | 约 8B 参数，Dense 多模态 |

## 继承关系

下图展示了 `TrainEngine` 支持的所有配置类的完整继承层级，包括 `TransformerConfig` 和 `BaseComposeConfig` 两大分支。

```text
XTunerBaseModelConfig
├── TransformerConfig
│   ├── Dense 模型
│   │   ├── Qwen2DenseConfig
│   │   │   └── Qwen2Dense7BConfig
│   │   └── Qwen3DenseConfig
│   │       ├── Qwen3Dense8BConfig
│   │       ├── Qwen3Dense4BConfig
│   │       ├── Qwen3Dense0P6BConfig
│   │       ├── Qwen3VLTextDense4BConfig
│   │       └── Qwen3VLTextDense8BConfig
│   └── MoE 模型（经由 MoEConfig）
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

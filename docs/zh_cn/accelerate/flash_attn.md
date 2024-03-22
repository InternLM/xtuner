# 使用 Flash Attention 加速训练

Flash Attention (Flash Attention 2) 是一种用于加速 Transformer 模型中 Attention 计算，并减少其显存消耗的算法。XTuner 中 Flash Attention (Flash Attention 2) 的支持情况如下表所示：

|     模型     |  Flash Attention   |
| :----------: | :----------------: |
| baichuan 1/2 |        :x:         |
| chatglm 2/3  |        :x:         |
|   deepseek   | :white_check_mark: |
|    gemma     |        :x:         |
| internlm 1/2 | :white_check_mark: |
|   llama 2    | :white_check_mark: |
|   mistral    | :white_check_mark: |
|  qwen 1/1.5  | :white_check_mark: |
|  starcoder   | :white_check_mark: |
|      yi      | :white_check_mark: |
|    zephyr    | :white_check_mark: |

**XTuner 会根据运行环境自动控制 Flash Attention 的使用情况：**

| 环境                                                                                                 | Flash Attention 使用情况 |
| ---------------------------------------------------------------------------------------------------- | ------------------------ |
| 安装 [flash attn](https://github.com/Dao-AILab/flash-attention)                                      | Flash Attention 2        |
| 未安装 [flash attn](https://github.com/Dao-AILab/flash-attention) 且 PyTorch Version \<= 1.13        | No Flash Attention       |
| 未安装 [flash attn](https://github.com/Dao-AILab/flash-attention) 且 2.0 \<= PyTorch Version \<= 2.1 | Flash Attention 1        |
| 未安装 [flash attn](https://github.com/Dao-AILab/flash-attention) 且 PyTorch Version >= 2.2          | Flash Attention 2        |

注：使用 XTuner 训练 QWen1/1.5 时若想使用 Flash Attention 加速，需要先安装 [flash attn](https://github.com/Dao-AILab/flash-attention)（参考 [flash attn 安装](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features)，需要 cuda ）

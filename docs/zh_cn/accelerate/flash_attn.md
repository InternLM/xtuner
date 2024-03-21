# 使用 Flash Attention 加速训练

Flash Attention (Flash Attention 2) 是一种用于加速 Transformer 模型中 Attention 计算，并减少其显存消耗的算法。XTuner 中 Flash Attention (Flash Attention 2) 的支持情况如下表所示：

|   模型    |        Flash Attention        |
| :-------: | :---------------------------: |
| baichuan  | :negative_squared_cross_mark: |
|  chatglm  | :negative_squared_cross_mark: |
| deepseek  |      :white_check_mark:       |
|   gemma   | :negative_squared_cross_mark: |
| internlm  |      :white_check_mark:       |
|   llama   |      :white_check_mark:       |
|  mistral  |      :white_check_mark:       |
|   qwen    |      :white_check_mark:       |
| starcoder |      :white_check_mark:       |
|    yi     |      :white_check_mark:       |
|  zephyr   |      :white_check_mark:       |

**XTuner 会根据运行环境自动控制 Flash Attention 的使用情况：**

| 环境                                                                                                 | Flash Attention 使用情况 |
| ---------------------------------------------------------------------------------------------------- | ------------------------ |
| 安装 [flash attn](https://github.com/Dao-AILab/flash-attention)                                      | Flash Attention 2        |
| 未安装 [flash attn](https://github.com/Dao-AILab/flash-attention) 且 PyTorch Version \<= 1.13        | No Flash Attention       |
| 未安装 [flash attn](https://github.com/Dao-AILab/flash-attention) 且 2.0 \<= PyTorch Version \<= 2.1 | Flash Attention 1        |
| 未安装 [flash attn](https://github.com/Dao-AILab/flash-attention) 且 PyTorch Version >= 2.2          | Flash Attention 2        |

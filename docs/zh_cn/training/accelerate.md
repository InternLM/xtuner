# 加速训练

## 数据集拼接

### 简介

对于大型语言模型（LLM）的输入而言，“数据集拼接” 这一概念指的是将多个 token 序列拼接成一个单独的输入。大量的数据集都存在一个特点，即其长度分布严重偏向较短的序列，而 Transformers 模型接收固定长度的输入。因此，在模型训练过程中，通常需要将每条数据 "Pad" 至当前 batch 最长序列的长度，而 "Pad Token" 往往是某个特定的无意义的 token。

将多条数据打包在一起可以不再需要使用 "Pad Token" 进行无意义的填充，减少计算资源的浪费，同时还可以保持模型作为具有固定大小输入的静态图表示的优点。

下表展示了 InternLM2 7B 模型在 Alpaca 数据集上使用不同数据集拼接策略进行训练的速度对比，如表所示，“数据集拼接”会大幅度提升训练效率：

| 拼接策略   | 每秒处理 token 数 | 加速比 |
| ---------- | ----------------- | ------ |
| 不使用     | 362.9             | -      |
| 拼接至 2k  | 2677.1            | 7.38x  |
| 拼接至 4k  | 3124.3            | 8.61x  |
| 拼接至 8k  | 3173.9            | 8.76x  |
| 拼接至 16k | 2864.4            | 7.89x  |
| 拼接至 32k | 2965.4            | 8.17x  |

### 在 XTuner 中使用数据拼接

XTuner 中提供的 config 文件中默认使用了“数据集拼接”这一功能，可以通过设置 `max_length` 字段来调整数据拼接长度。例如可通过以下方式将拼接长度调整为 32k ：

```diff
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- max_length = 2048
+ max_length = 32768
pack_to_max_length = True

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    max_length=max_length,
    pack_to_max_length=pack_to_max_length,
    ...)
```

## 使用 DeepSpeed 加速训练

[DeepSpeed](https://github.com/microsoft/DeepSpeed) 是一个开源的深度学习优化库，旨在简化并加速大规模模型的训练。

XTuner 支持一键启动 DeepSpeed 进行训练，只需在启动命令后插入 `--deepspeed deepspeed_zero2(deepspeed_zero1 or deepspeed_zero3)` 即可：

```shell
xtuner train xxx --deepspeed deepspeed_zero2
```

例如若想使用 DeepSpeed Zero3 显存优化算法运行 QLoRA 算法在 oasst1 数据集上微调 InternLM2-Chat-7B，可使用以下命令：

```shell
# 单卡
xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero3
# 多卡
  (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero3
  (SLURM) srun ${SRUN_ARGS} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero3
```

## 使用 Flash Attention 加速训练

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

## 变长注意力 (Variable Length Flash Attention)

### 简介

在[第一节](#数据集拼接)中，我们讨论了“数据集拼接”策略对模型训练效率的显著提升。理论上，数据集拼接可能会对注意力（Attention）机制的计算过程产生影响。这是因为，在未采用数据拼接策略的情况下，每条数据在计算注意力时仅与自身相关联。然而，当采用数据拼接策略后，由多条短数据拼接成的长数据在计算注意力时会相互关联。以一个由若干短数据拼接成长度为 4096 的数据为例，如果不采用变长注意力机制，在注意力计算阶段，每个 token 将会关注全部 4096 个 tokens ，如图左侧所示。

相反，在使用变长注意力机制的情况下，每个 token 在注意力计算阶段仅会关注其所在短数据中的所有 tokens ，如图右侧所示。因此，**变长注意力机制确保了无论是否采用“数据集拼接”策略，模型训练的行为保持一致性**。

<div align="center">
  <img src="https://github.com/InternLM/InternLM/assets/41630003/7e0c6a02-a970-4bd3-a10b-8341720bf654" width="600"/>
  <br /><br />
</div>

### XTuner 变长注意力支持情况

> \[!IMPORTANT\]
> 使用变长注意力需要首先安装 [flash attn](https://github.com/Dao-AILab/flash-attention) （参考 [flash attn 安装](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) ）

|   模型    | Variable Length Flash Attention |
| :-------: | :-----------------------------: |
| baichuan  |  :negative_squared_cross_mark:  |
|  chatglm  |  :negative_squared_cross_mark:  |
| deepseek  |       :white_check_mark:        |
|   gemma   |  :negative_squared_cross_mark:  |
| internlm  |       :white_check_mark:        |
|   llama   |       :white_check_mark:        |
|  mistral  |       :white_check_mark:        |
|   qwen    |       :white_check_mark:        |
| starcoder |  :negative_squared_cross_mark:  |
|    yi     |       :white_check_mark:        |
|  zephyr   |       :white_check_mark:        |

### 在 XTuner 中使用变长注意力机制

#### Step 1, 安装 flash_attn

XTuner 中实现的变长注意力需要依赖 Flash Attention 2，可通过以下命令安装：

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

详细安装步骤请参考 [flash attn 安装](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features)

#### Step 2, 列出候选模型名字

XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：

```bash
xtuner list-cfg -p internlm
```

`-p` 为模糊查找，若想训练其他模型，可以修改 `internlm` 为 XTuner 支持的其他模型名称。

#### Step 3, 复制 config 文件

导出需要使用的 config ：

```bash
xtuner copy-cfg ${CONFIG_NAME} ${SAVE_DIR}
```

例如通过下列命令将名为 `internlm_7b_full_oasst1_e3` 的 config 导出至当前目录下：

```bash
xtuner copy-cfg internlm_7b_full_oasst1_e3 .
```

#### Step 4, 修改 config 文件

将 Step 3 复制得到的 config 文件中的 `use_varlen_attn` 属性由 False 改为 True 即可激活变长注意力训练机制：

```diff
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm-7b'
- use_varlen_attn = False
+ use_varlen_attn = True
...
```

> \[!IMPORTANT\]
> 需要注意，当设置 `use_varlen_attn = True` 后，请确保 `batch_size` 被设置为 1，且 `pack_to_max_length` 被设置为 True。

#### Step 5, 开始训练

```
xtuner train ${CONFIG_NAME_OR_PATH}
```

例如，我们可以基于 Step 4 中修改得到的 `internlm_7b_full_oasst1_e3_copy.py` 进行训练：

```bash
# On a single GPU
xtuner train internlm_7b_full_oasst1_e3_copy.py --deepspeed deepspeed_zero1
# On multiple GPUs
(DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm_7b_full_oasst1_e3_copy.py --deepspeed deepspeed_zero1
(SLURM) srun ${SRUN_ARGS} xtuner train internlm_7b_full_oasst1_e3_copy.py --launcher slurm --deepspeed deepspeed_zero1
```

- `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。若未安装 DeepSpeed ，可通过 `pip install deepspeed>=0.12.3` 进行安装。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。如果用户期望关闭此功能，请直接移除此参数。

#### Step 6, 模型转换

将保存的 PTH 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 HuggingFace 模型：

```
xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}
```

对应上面的例子，模型转换脚本为：

```
xtuner convert pth_to_hf internlm_7b_full_oasst1_e3_copy.py ${PTH} ${SAVE_PATH}
```

其中 `${PTH}` 为训练权重保存的路径，若未指定，默认保存在 `./work_dirs/internlm_7b_full_oasst1_e3_copy` 路径下。

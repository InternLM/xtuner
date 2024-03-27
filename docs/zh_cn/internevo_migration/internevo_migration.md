# 总览

XTuner 可以复现 InternEvo (train_internlm) 仓库训练得到的开源模型 internlm/internlm2-chat-7b 的训练精度。

下面是 XTuner 和 InternEvo (train_internlm) 在相同数据集上训练相同基座模型的训练结果对比：

|        能力类别        | xtuner | internevo |
| :--------------------: | :----: | :-------: |
| 全数据集平均(无智能体) | 56.44  |   55.26   |
|  全维度平均(无智能体)  | 49.58  |   48.96   |
|     语言 Language      | 64.77  |   62.41   |
|     知识 Knowledge     | 52.24  |   52.52   |
|     推理 Reasoning     |  65.5  |   63.91   |
|    数学 Mathematics    | 30.95  |   30.26   |
|      代码 Coding       | 38.91  |   41.06   |
|    长文本 LongEval     | 45.09  |   43.62   |
|      智能体 Agent      | 44.85  |   43.97   |
|      数学题智能体      |   37   |   37.19   |
|        CIBench         | 79.07  |   69.78   |
|       PluginEval       | 65.57  |   65.62   |

64 * A100 的训练时间对比如下：

|   xtuner    | internevo  |
| :---------: | :--------: |
| 15 h 55 min | 16h 09 min |

注：使用 XTuner 提供的序列并行算法可以进一步提升训练速度，使用方式请参考 [序列并行文档](../training/training_extreme_long_sequence.md) 。

在从 InternEvo (train_internlm) 向 XTuner 迁移的过程中，我们需要关注模型、数据以及训练策略这三个方面的适配问题。后续内容将详细阐述如何进行适配。

# 适配

## 模型

InternEvo 在训练时读取和保存的模型权重满足以下目录结构（以 tp2pp2 为例）：

```
|-- root
    |-- model_config.pt
    |-- model_tp0_pp0.pt
    |-- model_tp0_pp1.pt
    |-- model_tp1_pp0.pt
    |-- model_tp1_pp1.pt
```

其中，`model_config.pt` 保存模型权重的一些 meta 信息，其余 4 个 checkpoint 则分别保存 4 组 GPUs 上的模型权重。因此，InternEvo 训练过程中要求读取预训练权重的 tp、pp 策略与训练使用的 tp、pp 策略一致才能正常读取预训练权重进行训练。

XTuner 支持基于 Huggingface Hub 上的模型进行训练，如下修改 config 内容即可将基座模型从 internlm2-7b 切换为 internlm2-20b：

```diff
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
- pretrained_model_name_or_path = 'internlm/internlm2-7b'
+ pretrained_model_name_or_path = 'internlm/internlm2-20b'

```

## 数据

InternEvo 在训练过程中通常会把多条数据拼接为一个特定的最大长度，随后输入模型训练。其配置往往满足以下形式：

```python
data = dict(
    seq_len=SEQ_LEN,
    pack_sample_into_one=False,
    min_length=MIN_LENGTH,
    train_folder=TRAIN_FOLDER,
    dataset_weights=DATASET_WEIGHTS,
    ...)
```

其中，数据配比 (`dataset_weights=DATASET_WEIGHTS`) 功能 XTuner 尚未支持。`TRAIN_FOLDER` 中的训练数据需要满足 ftdp tokenized 数据集格式：

```
|-- TRAIN_FOLDER
    |-- cn
    |   |-- dataset1
    |   |   |-- data1.bin
    |   |   |-- data1.bin.meta
    |   |-- dataset2
    |   |   |-- data2.bin
    |   |   |-- data2.bin.meta
```

在 XTuner 中实现在线数据集拼接策略需要参考 `xtuner/configs/internlm/internlm2_7b/internlm2_7b_w_internevo_dataset.py` 文件中的配置：

```diff
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Data
- dataset_folder = '/path/to/sft/data/folder'
+ dataset_folder = TRAIN_FOLDER
- max_length = 32768
+ max_length = SEQ_LEN

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=build_packed_dataset,
    dataset_cfg=dict(
        type=load_intern_repo_tokenized_dataset,
        data_order_path=None,
        folder=dataset_folder,
-       min_length=0,
+       min_length=MIN_LENGTH,
        file_type='.bin'),
    packed_length=max_length,
    seed=1024)
```

> \[!IMPORTANT\]
> 需要注意，由于训练数据喂给模型的先后顺序可能对训练结果造成影响，因此建议不要轻易修改上述配置中的 `seed` 选项。同时，可参考[文档todo](./ftdp_dataset/Case4.md#step-3-获取数据顺序-可选)进一步固定数据顺序。

## 训练策略

### 变长注意力 (Variable Length Flash Attention)

InternEvo 通过设置 [数据配置](https://github.com/InternLM/InternEvo/blob/77c3b46bfe51f6bc245c4aba98639221b8618372/doc/usage.md#%E6%95%B0%E6%8D%AE%E9%85%8D%E7%BD%AE) 中的 `pack_sample_into_one` 参数为 False 来使用“变长注意力机制”（见下图右侧）。

```python
data = dict(
    pack_sample_into_one=False,
    ...)
```

<div align="center">
  <img src="https://github.com/InternLM/InternEvo/blob/develop/doc/imgs/pack_into_one.png?raw=true" width="800"/>
  <br /><br />
</div>

在 XTuner 中使用这一功能需要设置 config 中的 `use_varlen_attn` 配置为 True，即可保证训练行为与 InternEvo 一致：

```diff
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm2-7b'
- use_varlen_attn = False
+ use_varlen_attn = True
...
```

> \[!IMPORTANT\]
> 需要注意，当设置 `use_varlen_attn = True` 后，请确保 `batch_size` 被设置为 1，且 `pack_to_max_length` 被设置为 True。

### batch_size 与 accumulative_counts

在 InternEvo 的配置中，与 batch_size 和 accumulative_counts 相关的配置有如下几个：

```python
data = dict(
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=MICRO_NUM,
    # MICRO_BATCH_SIZE * SEQ_LEN = PACKED_LENGTH
    micro_bsz=MICRO_BATCH_SIZE,
    total_steps=TOTAL_STEP,
    # 梯度累计，默认等于MICRO_NUM（BS）
    gradient_accumulation=GRADIENT_ACCUMULATION,
    ...)
```

其中：

1. `micro_num` 与 `gradient_accumulation` 通常具有相同含义，其数值默认相等。
2. `total_steps` 在 XTuner 中可以不手动指定，可通过 `max_epochs` 指定。
3. XTuner 目前只支持 `micro_bsz = 1` 。

为对齐以上配置，可参考 XTuner 中 `xtuner/configs/internlm/internlm2_7b/internlm2_7b_w_internevo_dataset.py` 文件中的配置，并进行如下修改：

```diff
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Scheduler & Optimizer
- accumulative_counts = 1
+ accumulative_counts = MICRO_NUM # or GRADIENT_ACCUMULATION
- max_epochs = 1
+ max_epochs = MAX_EPOCHS
```

### 并行训练

#### ZeRO 系列显存优化

XTuner 支持使用 ZeRO 系列显存优化降低训练过程中的显存消耗：

```shell
  # 单卡
  xtuner train ${CONFIG_NAME_OR_PATH} --deepspeed deepspeed_zero2
  # 多卡
  (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train ${CONFIG_NAME_OR_PATH} --deepspeed deepspeed_zero2
  (SLURM) srun ${SRUN_ARGS} xtuner train ${CONFIG_NAME_OR_PATH} --launcher slurm --deepspeed deepspeed_zero2
```

- `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 。

#### 序列并行

InternEvo 中支持了 Data Parallel、Tensor Parallel、Pipeline Parallel 和 Sequence Parallel 四种并行策略。XTuner 目前支持了 Data Parallel 和 Sequence Parallel 两种并行策略，可满足基本全部的训练需求（搭配 zero3 显存优化策略可支持 70B 模型 256K 上下文训练）。

假定 InternEvo 训练过程中：tp_world_size = TP, pp_world_size = PP, sequence_parallel = True。则训练的 global_batch_size 满足以下计算公式:

```
# 多除的一个 TP 是因为启用了 sequence parallel
global_batch_size = num_gpus * batch_size_per_device * gradient_accumulate / TP / PP / TP
```

需要注意的是，internlm2-chat 的训练过程中通常启用了 [“变长注意力”](#变长注意力-variable-length-flash-attention) 策略，此时 `单卡 batch size 等于 2，拼接数据集至最大长度 2k` 的配置与 `单卡 batch size 等于 1，拼接数据集至最大长度 4k` 的配置训练行为是近似的，因此 XTuner 目前只支持了 `batch_size_per_device = 1` 的情况。因此，若想使用 XTuner 训练时保证 global_batch_size 与 InternEvo 一致，需要在配置文件中综合调整 `gradient_accumulate` 和 `sequence_parallel_size` 两项的数值：

```diff
+ from xtuner.parallel.sequence import SequenceParallelSampler

+ sequence_parallel_size = SP
- accumulative_counts = 1  # 1bs * 1acc * 64gpu = 64 batchsize
+ accumulative_counts = TP * PP * TP / SP

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataloader = dict(
-   sampler=dict(type=DefaultSampler, shuffle=True),
+   sampler=dict(type=SequenceParallelSampler, shuffle=True),
    ...)
```

XTuner 序列并行的详细用法请参考 [序列并行文档](../training/training_extreme_long_sequence.md)。

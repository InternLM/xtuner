<div align="center">

# 序列并行：训练极长序列大模型的系统优化

</div>

XTuner 中的序列并行设计思路参考了 DeepSpeed 的工作 [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)，并加以优化，以达到直接基于 transformers 算法库或 Huggingface Hub 上的开源模型训练 1M 以上超长序列的目标。

## 简介

从生成性AI到科研模型，长序列训练正在变得非常重要。

在生成性AI领域，会话式AI、长文档摘要、代码库理解和例如 Sora 这种视频生成任务都需要在空间和时间层面对长上下文进行推理。

对于科学AI来说，长序列同样至关重要，它为更好地理解结构生物学、医疗保健、气候和天气预测以及大分子模拟打开了大门。

然而，尽管序列长度的重要性不断增长，XTuner 现有的显存优化策略（如 zero 系列），却不足以解决大模型、长序列训练问题。

同时，受限于通信效率，现有的许多序列并行方法也不够高效。

另外，现有的序列并行方法普遍存在较多的代码侵入式修改，易用性和维护性都要大打折扣。同时也不满足 XTuner 基于 transformers 算法库或 Huggingface Hub 上的开源模型直接进行训练的要求。

<div align="center">
  <img src="https://github.com/InternLM/xtuner/assets/41630003/0b791458-40bd-4dc6-aaf5-ff891fcc112a" width="1000"/>
  <br /><br />
</div>

为了解决上述长序列训练带来的问题，XTuner 采用了一种简单、易用且高效的序列并行算法。由于 Transformer 结构较为规整，除 attention 计算外，其他计算过程中 token 之间不会互相影响（即每个 token 的计算是独立的），这一条件为序列并行提供了有利条件。上图展示了序列并行的核心设计。设由 P 个 GPUs 共同计算一个长度为 N 的长序列，在 Attention 计算的第一阶段，长度为 N / P 的子序列会通过线性层投影为 Query、Key、Value。接下来， QKV Tensor 会在参与序列并行计算的多个 GPUs 之间通过高度优化的 all-to-all 通信算子汇聚，得到序列长度为 N ，但更少注意力头的子序列。注意力计算后，通过另一个 all-to-all 通信算子将其转换为长度为 N / P 的子序列，进行后续计算。

总体而言，XTuner 的序列并行算法具有以下关键特性：

* 支持全量训练**超过百万个token**的序列
* 支持百 B 级模型训练：XTuner 的序列并行不仅支持长序列训练，还可结合 zero3 显存优化策略训练大尺寸模型
* 完全通用的序列并行 **API 抽象**

## 使用 XTuner 进行序列并行训练

### Step 1 修改 config 文件

1. 在 config 中修改 `sequence_parallel_size` 字段即可调整 $sequence\\_parallel\\_world\\_size$ 。
2. 同时若想保证与不使用序列并行的训练效果类似，需要同步增大梯度累积的数值为原来的 $sequence\\_parallel\\_world\\_size$ 倍，因为在使用序列并行训练时， $data\\_parallel\\_world\\_size$ 降为了原来的 $\frac{1}{sequence\\_parallel\\_world\\_size}$。
3. 替换 DefaultSampler 为支持序列并行的 SequenceParallelSampler。

**注：需要保证所使用的 GPU 总数可以被 `sequence_parallel_size` 整除。**

```diff
+ from xtuner.parallel.sequence import SequenceParallelSampler

- sequence_parallel_size = 1
+ sequence_parallel_size = 4  # take `sequence_parallel_size = 4`` as an example

- accumulative_counts = 1
+ accumulative_counts = 4  # accumulative_counts = accumulative_counts * sequence_parallel_size

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataloader = dict(
-   sampler=dict(type=DefaultSampler, shuffle=True),
+   sampler=dict(type=SequenceParallelSampler, seed=1024, shuffle=True),
    ...)
```

另外，若需要进一步拓展模型的长文本处理能力，需要进一步修改 config 中的 `max_position_embeddings` 字段。例如需要将模型的上下文长度拓展为 64K 时，可进行如下修改：

```diff
+ max_position_embeddings = 65536

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
model = dict(
    type=SupervisedFinetune,
+   max_position_embeddings = max_position_embeddings,
    ...)
```

### Step 2 开始训练

需要使用 DeepSpeed 进行训练：

```bash
(DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train ${CONFIG_PATH} --deepspeed deepspeed_zero2
(SLURM) srun ${SRUN_ARGS} xtuner train ${CONFIG_PATH} --launcher slurm --deepspeed deepspeed_zero2
```

- ${CONFIG_PATH} 为 Step 1 中修改得到的 config 文件路径
- 可根据实际情况选择使用不同的 zero 策略

## 序列并行 API 抽象

为了提升算法的可迁移性，XTuner 中抽象出了序列并行所必须的五个 API 接口：
- 序列并行分布式环境初始化 (init_sequence_parallel)
- 适配序列并行的 Data Sampler (SequenceParallelSampler)
- 数据 Pad (pad_for_sequence_parallel)
- 数据切分 (split_for_sequence_parallel)
- 适配序列并行的 Attention (dispatch_modules)
- reduce loss 以正确打印训练损失 (reduce_sequence_parallel_loss)

### 序列并行分布式环境初始化

由于序列并行算法会将长序列切分为 $sequence\\_parallel\\_world\\_size$ 块，并将每个子序列分发给对应的 GPU 独立进行计算。因此需要在训练开始前初始化序列并行分布式环境，以指定哪几块 GPU 共同负责一个长序列输入的计算。

一个 $sequence\\_parallel\\_world\\_size = 4$ 的示例如下：

```python
# We have to initialize the distributed training environment first.
# Here is an example when training on slurm scheduler
# from xtuner.parallel.sequence import init_dist
# init_dist('slurm', 'nccl', init_backend='deepspeed')
from xtuner.parallel.sequence import init_sequence_parallel
sequence_parallel_world_size = 4
init_sequence_parallel(sequence_parallel_world_size)
```

上述过程在 xtuner/engine/_strategy/deepspeed.py 中实现。

### Data Sampler 适配序列并行

在使用序列并行后，Dataloader 的采样策略需要进一步调整。例如当 $sequence\\_parallel\\_world\\_size = 4$ 时，4 块 GPU 从 Dataloader 拿到的数据需要是完全一样的。

在构建 Dataloader 时搭配 XTuner 中提供的 SequenceParallelSampler 使用即可：

```python
from xtuner.parallel.sequence import SequenceParallelSampler
dataloader = DataLoader(
    train_dataset, sampler=SequenceParallelSampler(train_dataset),
    **other_dataloader_params)
```

### 数据 Pad

由于每条训练数据的长度可能不尽相同，我们需要将数据进行 Pad 以使得序列长度可以被 $sequence\\_parallel\\_world\\_size$ 整除，这样一条长数据才能被均等地分发给不同的 GPU 上。

训练过程中需要被 Pad 的 Tensor 往往有 input_ids, labels, position_ids, attention_mask 四个，pad 的过程可以通过以下方式实现：

```python
from xtuner.parallel.sequence import pad_for_sequence_parallel

input_ids = pad_for_sequence_parallel(input_ids, padding_value=0)
labels = pad_for_sequence_parallel(labels, padding_value=-100)
position_ids = pad_for_sequence_parallel(position_ids, padding_value=0)
attention_mask = pad_for_sequence_parallel(attention_mask, padding_value=0)
```

以上过程在 `xtuner/dataset/collate_fns/default_collate_fn.py` 中实现。

### 数据切分

在传入给 Transformer 模型前，我们需要对长序列均等切分：

```python
from xtuner.parallel.sequence import split_for_sequence_parallel
# attention mask should not be split
# `dim` is 1 as the shape of tensor is (bs, seq_len, ...)
input_ids = split_for_sequence_parallel(input_ids, dim=1)
labels = split_for_sequence_parallel(labels, dim=1)
position_ids = split_for_sequence_parallel(position_ids, dim=1)
```

以上过程在 `xtuner/model/sft.py` 中实现。

### Attention 适配序列并行

在 Attention 的计算过程中，序列中的不同 token 是不能独立运算的，但不同的 attention head 之间的计算却是独立的。因此，如[第一节](#简介)所述，需要在计算 Attention 前后（即 qkv_proj 后和 o_proj 前）分别插入一个 *all-to-all* 操作。

XTuner 提供了 dispatch_modules 接口以支持修改模型 Attention 的计算方式：

```python
from xtuner.model.modules import dispatch_modules
model: AutoModelForCausalLM
dispatch_modules(model)
```

上述过程在 xtuner/model/sft.py 中实现。

### Reduce Loss 以正确打印训练损失

这个 API 对于保证训练的正确性不是必须的，但对于观测模型训练状态，打印训练 loss 是非常有用的。

```python
from xtuner.parallel.sequence import reduce_sequence_parallel_loss
outputs = llm(input_ids=input_ids, labels=labels, **kwargs)
num_tokens_per_rank = (labels != -100).sum()
# Suppose sequence parallel world size equals to 4,
# losses on rank0, rank1, rank2, rank3 are different.
loss = reduce_sequence_parallel_loss(outputs.loss, num_tokens_per_rank)
# After loss reduction, losses on rank0, rank1, rank2, rank3 are the same.
```

上述过程在 xtuner/model/sft.py 中实现。

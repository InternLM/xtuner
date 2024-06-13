## DPO 快速上手

在本章节中，我们将介绍如何使用 XTuner 训练 1.8B 的 DPO（Direct Preference Optimization）模型，以帮助您快速上手。

### 准备预训练模型权重

我们使用经过 SFT 的语言模型[InternLM2-chat-1.8b-sft](https://huggingface.co/internlm/internlm2-chat-1_8b-sft)作为 DPO 模型的初始化模型来进行偏好对齐。

在训练配置文件中设置`pretrained_model_name_or_path = 'internlm/internlm2-chat-1_8b-sft'`，则会在启动训练时自动下载模型文件。若您需要手动下载模型权重，那么请参考[准备预训练模型权重](https://xtuner.readthedocs.io/zh-cn/latest/preparation/pretrained_model.html)章节，其中详细说明了如何从 Huggingface 或者是 Modelscope 下载模型权重的方法。这里我们附上模型的 HuggingFace 链接与 ModelScope 链接：

- HuggingFace 链接位于：https://huggingface.co/internlm/internlm2-chat-1_8b-sft
- ModelScope 链接位于：https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft/summary

### 准备训练数据

在本教程中使用 Huggingface 上的[mlabonne/orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)数据集作为演示，

```python
train_dataset = dict(
    type=build_preference_dataset,
    dataset=dict(
        type=load_dataset,
        path='mlabonne/orpo-dpo-mix-40k'),
    dataset_map_fn=orpo_dpo_mix_40k_map_fn,
    is_dpo=True,
    is_reward=False,
)
```

在配置文件中使用以上配置，即可自动下载并处理该数据集。如果您希望使用其他 Huggingface 上的开源数据集或是使用自定义的数据集，请参阅[偏好数据集](../reward_model/preference_data.md)章节。

### 准备配置文件

XTuner 提供了多个开箱即用的配置文件，可以通过 `xtuner list-cfg` 查看。我们执行如下指令，以复制一个配置文件到当前目录。

```bash
xtuner copy-cfg internlm2_chat_1_8b_dpo_full .
```

打开复制后的配置文件，如果您选择自动下载模型和数据集，则无需修改配置。若您希望填入您预先下载的模型路径和数据集路径，请修改配置中的`pretrained_model_name_or_path`以及`train_dataset`中`dataset`的`path`参数。

更多的训练参数配置，请参阅[修改DPO训练配置](./modify_settings.md)章节。

### 启动训练

在完成上述操作后，便可以使用下面的指令启动训练任务了。

```bash
# 单机单卡
xtuner train ./internlm2_chat_1_8b_dpo_full_copy.py
# 单机多卡
NPROC_PER_NODE=${GPU_NUM} xtuner train ./internlm2_chat_1_8b_dpo_full_copy.py
# slurm 集群
srun ${SRUN_ARGS} xtuner train ./internlm2_chat_1_8b_dpo_full_copy.py --launcher slurm
```

### 模型转换

XTuner 已经集成好了将模型转换为 HuggingFace 格式的工具，我们只需要执行

```bash
# 创建存放 hf 格式参数的目录
mkdir work_dirs/internlm2_chat_1_8b_dpo_full_copy/iter_15230_hf

# 转换格式
xtuner convert pth_to_hf internlm2_chat_1_8b_dpo_full_copy.py \
                            work_dirs/internlm2_chat_1_8b_dpo_full_copy.py/iter_15230.pth \
                            work_dirs/internlm2_chat_1_8b_dpo_full_copy.py/iter_15230_hf
```

便能够将 XTuner 的 ckpt 转换为 Huggingface 格式的模型。

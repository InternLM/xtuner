## Reward Model 快速上手

在本章节中，我们将介绍如何使用 XTuner 训练 1.8B 的 Reward Model，以帮助您快速上手。

### 准备预训练模型权重

依据 [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) 论文中的描述，我们使用进过 SFT 的语言模型作为 Reward Model 的初始化模型。这里我们使用[InternLM2-chat-1.8b-sft](https://huggingface.co/internlm/internlm2-chat-1_8b-sft)作为初始化模型。

在训练配置文件中设置`pretrained_model_name_or_path = 'internlm/internlm2-chat-1_8b-sft'`，则会在启动训练时自动下载模型文件。若您需要手动下载模型权重，那么请参考[准备预训练模型权重](https://xtuner.readthedocs.io/zh-cn/latest/preparation/pretrained_model.html)章节，其中详细说明了如何从 Huggingface 或者是 Modelscope 下载模型权重的方法。这里我们附上模型的 HuggingFace 链接与 ModelScope 链接：

- HuggingFace 链接位于：https://huggingface.co/internlm/internlm2-chat-1_8b-sft

- ModelScope 链接位于：https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft/summary

### 准备训练数据

在本教程中使用 [UltraFeedback](https://arxiv.org/abs/2310.01377) 数据集作为演示，为了方便起见，我们使用 huggingface 上已经预处理过的 [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) 数据集，

```python
train_dataset = dict(
    type=build_preference_dataset,
    dataset=dict(
        type=load_dataset,
        path='argilla/ultrafeedback-binarized-preferences-cleaned'),
    dataset_map_fn=orpo_dpo_mix_40k_map_fn,
    is_dpo=False,
    is_reward=True,
)
```

在配置文件中使用以上配置，即可自动下载并处理该数据集。如果您希望使用其他 huggingface 上的开源数据集或是使用自定义的数据集，请参阅[偏好数据集](./preference_data.md)章节。

### 准备配置文件

XTuner 提供了多个开箱即用的配置文件，可以通过 `xtuner list-cfg` 查看。我们执行如下指令，以复制一个配置文件到当前目录。

```bash
xtuner copy-cfg internlm2_chat_1_8b_reward_full_ultrafeedback .
```

打开复制后的配置文件，如果您选择自动下载模型和数据集，则无需修改配置。若您希望填入您预先下载的模型路径和数据集路径，请修改配置中的 `pretrained_model_name_or_path` 以及 `train_dataset` 中 `dataset` 的 `path` 参数。

更多的训练参数配置，请参阅[修改Reward训练配置](./modify_settings.md)章节。

### 启动训练

在完成上述操作后，便可以使用下面的指令启动训练任务了。

```bash
# 单机单卡
xtuner train ./internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py
# 单机多卡
NPROC_PER_NODE=${GPU_NUM} xtuner train ./internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py
# slurm 集群
srun ${SRUN_ARGS} xtuner train ./internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py --launcher slurm
```

正确的训练日志应当如下所示（在单卡 A800 上运行）：

```
06/06 16:12:11 - mmengine - INFO - Iter(train) [   10/15230]  lr: 3.9580e-07  eta: 2:59:41  time: 0.7084  data_time: 0.0044  memory: 18021  loss: 0.6270  acc: 0.0000  chosen_score_mean: 0.0000  rejected_score_mean: 0.0000  num_samples: 4.0000  num_tokens: 969.0000
06/06 16:12:17 - mmengine - INFO - Iter(train) [   20/15230]  lr: 8.3536e-07  eta: 2:45:25  time: 0.5968  data_time: 0.0034  memory: 42180  loss: 0.6270  acc: 0.5000  chosen_score_mean: 0.0013  rejected_score_mean: 0.0010  num_samples: 4.0000  num_tokens: 1405.0000
06/06 16:12:22 - mmengine - INFO - Iter(train) [   30/15230]  lr: 1.2749e-06  eta: 2:37:18  time: 0.5578  data_time: 0.0024  memory: 32121  loss: 0.6270  acc: 0.7500  chosen_score_mean: 0.0016  rejected_score_mean: 0.0011  num_samples: 4.0000  num_tokens: 932.0000
06/06 16:12:28 - mmengine - INFO - Iter(train) [   40/15230]  lr: 1.7145e-06  eta: 2:36:05  time: 0.6033  data_time: 0.0025  memory: 42186  loss: 0.6270  acc: 0.7500  chosen_score_mean: 0.0027  rejected_score_mean: 0.0016  num_samples: 4.0000  num_tokens: 994.0000
06/06 16:12:35 - mmengine - INFO - Iter(train) [   50/15230]  lr: 2.1540e-06  eta: 2:41:03  time: 0.7166  data_time: 0.0027  memory: 42186  loss: 0.6278  acc: 0.5000  chosen_score_mean: 0.0031  rejected_score_mean: 0.0032  num_samples: 4.0000  num_tokens: 2049.0000
06/06 16:12:40 - mmengine - INFO - Iter(train) [   60/15230]  lr: 2.5936e-06  eta: 2:33:37  time: 0.4627  data_time: 0.0023  memory: 30238  loss: 0.6262  acc: 1.0000  chosen_score_mean: 0.0057  rejected_score_mean: 0.0030  num_samples: 4.0000  num_tokens: 992.0000
06/06 16:12:46 - mmengine - INFO - Iter(train) [   70/15230]  lr: 3.0331e-06  eta: 2:33:18  time: 0.6018  data_time: 0.0025  memory: 42186  loss: 0.6247  acc: 0.7500  chosen_score_mean: 0.0117  rejected_score_mean: 0.0055  num_samples: 4.0000  num_tokens: 815.0000
```

### 模型转换

XTuner 已经集成好了将模型转换为 HuggingFace 格式的工具，我们只需要执行

```bash
# 创建存放 hf 格式参数的目录
mkdir work_dirs/internlm2_chat_1_8b_reward_full_ultrafeedback_copy/iter_15230_hf

# 转换格式
xtuner convert pth_to_hf internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py \
                            work_dirs/internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py/iter_15230.pth \
                            work_dirs/internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py/iter_15230_hf
```

便能够将 XTuner 的 ckpt 转换为 Huggingface 格式的模型。

需要注意的是，由于 Reward Model 的类型并未在 transformers 官方库中集成，因此目前只有InternLM2模型训练得到的 Reward Model 会被转换为 InternLM2ForRewardModel 类型，而其他模型则会默认转换为 SequenceClassification 类型（例如 LLaMa3 会被转换为 LlamaForSequenceClassification 类型），但这并不影响其在 XTuner PPO 训练中的使用。

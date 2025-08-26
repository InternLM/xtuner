## 修改 DPO 训练配置

本章节仅介绍与 DPO（Direct Preference Optimization）训练相关的配置参数，更多 XTuner 配置文件的细节，请参考[修改训练配置](https://xtuner.readthedocs.io/zh-cn/latest/training/modify_settings.html)

### 损失函数

在 DPO 训练中，你可以根据需求选择不同的损失函数类型。XTuner 提供了多种损失函数选项，如 `sigmoid`、`hinge`、`ipo` 等。可以通过设置 `dpo_loss_type` 参数来选择使用的损失函数类型。

此外，你还可以通过调整 `loss_beta` 参数来控制损失函数中的温度系数。同时，`label_smoothing` 参数可以用于平滑标签。

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
dpo_loss_type = 'sigmoid'  # One of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'sppo_hard', 'nca_pair', 'robust']
loss_beta = 0.1
label_smoothing = 0.0
```

### 修改模型

用户可以修改 `pretrained_model_name_or_path` 对预训练模型进行修改。

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm2-chat-1_8b-sft'
```

### 训练数据

在 DPO 训练中，你可以通过 `max_length` 来指定单个样本序列的最大 token 数，XTuner 会自动对数据进行截断或是填充。

```python
# Data
max_length = 2048
```

在配置文件中，我们通过 `train_dataset` 字段来指定训练数据集，你可以通过 `dataset` 字段指定数据集的加载方式，通过 `dataset_map_fn` 字段指定数据集的映射函数。

```python
#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
sampler = SequenceParallelSampler \
    if sequence_parallel_size > 1 else DefaultSampler

train_dataset = dict(
    type=build_preference_dataset,
    dataset=dict(type=load_dataset, path='mlabonne/orpo-dpo-mix-40k'),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=orpo_dpo_mix_40k_map_fn,
    is_dpo=True,
    is_reward=False,
    reward_token_id=-1,
    num_proc=32,
    use_varlen_attn=use_varlen_attn,
    max_packed_length=max_packed_length,
    shuffle_before_pack=True,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(
        type=preference_collate_fn, use_varlen_attn=use_varlen_attn))
```

上述配置中，我们使用了 `load_dataset` 来加载 huggingface 上的 `mlabonne/orpo-dpo-mix-40k` 数据集，使用 `orpo_dpo_mix_40k_map_fn` 作为数据集映射函数。

关于如何处理数据集以及如何编写数据集映射函数，请参考[偏好数据集章节](../reward_model/preference_data.md)。

### 加速训练

在使用偏好数据训练时，我们推荐您开启[变长注意力机制](https://xtuner.readthedocs.io/zh-cn/latest/acceleration/varlen_flash_attn.html)， 以避免单个偏好内的 chosen 和 rejected 的样本长度差异造成的显存浪费。你可以通过 `use_varlen_attn=True` 来开启变长注意力机制。

XTuner 中还支持了大量的训练加速方法，关于它们的使用方法，请参考[加速策略章节](https://xtuner.readthedocs.io/zh-cn/latest/acceleration/hyper_parameters.html)。

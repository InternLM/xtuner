## 修改 Reward Model 训练配置

本章节仅介绍与 Reward Model 训练相关的配置参数，更多 XTuner 配置文件的细节，请参考[修改训练配置](https://xtuner.readthedocs.io/zh-cn/latest/training/modify_settings.html)

### 损失函数

XTuner 使用了 [Bradley–Terry 模型](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) 作为 Reward Model 的偏好建模方式，你可以指定 `loss_type="ranking"` 来使用 ranking loss。XTuner 中也实现了 InternLM2 中提出的 focal 损失函数，它通过调整难易样本的权重来避免过拟合，可以设置 `loss_type="focal"` 来使用该损失函数。对于该损失函数的详细说明，请参考 [InternLM2 技术报告](https://arxiv.org/abs/2403.17297)。

另外，为了使 reward model 输出的 score 数值保持稳定，我们还在 loss 中额外增加了一个约束项，你可以指定 `penalty_type='log_barrier'` 或是 `penalty_type='L2'` 以启用对数约束或是L2约束。

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
loss_type = 'focal'  # 'ranking' or 'focal'
penalty_type = 'log_barrier'  # 'log_barrier' or 'L2'
```

### 修改模型

用户可以修改 `pretrained_model_name_or_path` 对预训练模型进行修改。

需要注意的是，由于 XTuner 通过对数据的末尾添加 `<|reward|>` 特殊 token 的方式计算 reward 得分，因此当切换模型的词表发生变化时，该特殊 token 的 id 也需要进行相应的修改，我们通常会使用词表末尾未使用的 token 作为 reward token。

例如，在 InternLM2 中我们使用 `[UNUSED_TOKEN_130]` 作为 reward token:

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm2-chat-1_8b-sft'
reward_token_id = 92527  # use [UNUSED_TOKEN_130] as reward token
```

如果用户将模型切换为llama3,我们则可以使用 `<|reserved_special_token_0|>` 作为 reward token:

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
reward_token_id = 128002  # use <|reserved_special_token_0|> as reward token
```

### 训练数据

在 Reward Model 训练中，你可以通过 `max_length` 来指定单个样本序列的最大 token 数，XTuner 会自动对数据进行截断或是填充。

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
    dataset=dict(
        type=load_dataset,
        path='argilla/ultrafeedback-binarized-preferences-cleaned'),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=orpo_dpo_mix_40k_map_fn,
    is_dpo=False,
    is_reward=True,
    reward_token_id=reward_token_id,
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

上述配置中，我们使用了 `load_dataset` 来加载 huggingface 上的 `argilla/ultrafeedback-binarized-preferences-cleaned` 数据集，使用 `orpo_dpo_mix_40k_map_fn` 作为数据集映射函数（这是因为 `orpo_dpo_mix_40k` 与 `ultrafeedback-binarized-preferences-cleaned` 的格式相同，因此这里共用了同一个映射函数）。

关于如何处理数据集以及如何编写数据集映射函数，请参考[偏好数据集章节](./preference_data.md)。

### 加速训练

在使用偏好数据训练时，我们推荐您开启[变长注意力机制](https://xtuner.readthedocs.io/zh-cn/latest/acceleration/varlen_flash_attn.html)， 以避免单个偏好内的 chosen 和 rejected 的样本长度差异造成的显存浪费。你可以通过 `use_varlen_attn=True` 来开启变长注意力机制。

XTuner 中还支持了大量的训练加速方法，关于它们的使用方法，请参考[加速策略章节](https://xtuner.readthedocs.io/zh-cn/latest/acceleration/hyper_parameters.html)。

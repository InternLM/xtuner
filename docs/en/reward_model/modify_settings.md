## Modify Reward Model Training Configuration

This section introduces the config related to Reward Model training. For more details on XTuner config files, please refer to [Modify Settings](https://xtuner.readthedocs.io/zh-cn/latest/training/modify_settings.html).

### Loss Function

XTuner uses the [Bradley–Terry Model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) for preference modeling in the Reward Model. You can specify `loss_type="ranking"` to use ranking loss. XTuner also implements the focal loss function proposed in InternLM2, which adjusts the weights of difficult and easy samples to avoid overfitting. You can set `loss_type="focal"` to use this loss function. For a detailed explanation of this loss function, please refer to the [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297).

Additionally, to maintain stable reward model output scores, we have added a constraint term in the loss. You can specify `penalty_type='log_barrier'` or `penalty_type='L2'` to enable log barrier or L2 constraints, respectively.

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
loss_type = 'focal'  # 'ranking' or 'focal'
penalty_type = 'log_barrier'  # 'log_barrier' or 'L2'
```

### Modifying the Model

Users can modify `pretrained_model_name_or_path` to change the pretrained model.

Note that XTuner calculates reward scores by appending a special token at the end of the data. Therefore, when switching models with different vocabularies, the ID of this special token also needs to be modified accordingly. We usually use an unused token at the end of the vocabulary as the reward token.

For example, in InternLM2, we use `[UNUSED_TOKEN_130]` as the reward token:

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm2-chat-1_8b-sft'
reward_token_id = 92527  # use [UNUSED_TOKEN_130] as reward token
```

If the user switches to the llama3 model, we can use `<|reserved_special_token_0|>` as the reward token:

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
reward_token_id = 128002  # use <|reserved_special_token_0|> as reward token
```

### Training Data

In Reward Model training, you can specify the maximum number of tokens for a single sample sequence using `max_length`. XTuner will automatically truncate or pad the data.

```python
# Data
max_length = 2048
```

In the configuration file, we use the `train_dataset` field to specify the training dataset. You can specify the dataset loading method using the `dataset` field and the dataset mapping function using the `dataset_map_fn` field.

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

In the above configuration, we use `load_dataset` to load the `argilla/ultrafeedback-binarized-preferences-cleaned` dataset from Hugging Face, using `orpo_dpo_mix_40k_map_fn` as the dataset mapping function (this is because `orpo_dpo_mix_40k` and `ultrafeedback-binarized-preferences-cleaned` have the same format, so the same mapping function is used).

For more information on handling datasets and writing dataset mapping functions, please refer to the [Preference Data Section](./preference_data.md).

### Accelerating Training

When training with preference data, we recommend enabling the [Variable-Length Attention Mechanism](https://xtuner.readthedocs.io/zh-cn/latest/acceleration/varlen_flash_attn.html) to avoid memory waste caused by length differences between chosen and rejected samples within a single preference. You can enable the variable-length attention mechanism by setting `use_varlen_attn=True`.

XTuner also supports many training acceleration methods. For details on how to use them, please refer to the [Acceleration Strategies Section](https://xtuner.readthedocs.io/zh-cn/latest/acceleration/hyper_parameters.html).

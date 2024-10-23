## Modify DPO Training Configuration

This section introduces config parameters related to DPO (Direct Preference Optimization) training. For more details on XTuner config files, please refer to [Modifying Training Configuration](https://xtuner.readthedocs.io/zh-cn/latest/training/modify_settings.html).

### Loss Function

In DPO training, you can choose different types of loss functions according to your needs. XTuner provides various loss function options, such as `sigmoid`, `hinge`, `ipo`, etc. You can select the desired loss function type by setting the `dpo_loss_type` parameter.

Additionally, you can control the temperature coefficient in the loss function by adjusting the `loss_beta` parameter. The `label_smoothing` parameter can be used for smoothing labels.

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
dpo_loss_type = 'sigmoid'  # One of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'sppo_hard', 'nca_pair', 'robust']
loss_beta = 0.1
label_smoothing = 0.0
```

### Modifying the Model

Users can modify `pretrained_model_name_or_path` to change the pretrained model.

```python
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm2-chat-1_8b-sft'
```

### Training Data

In DPO training, you can specify the maximum number of tokens for a single sample sequence using the `max_length` parameter. XTuner will automatically truncate or pad the data.

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

In the above configuration, we use `load_dataset` to load the `mlabonne/orpo-dpo-mix-40k` dataset from Hugging Face and use `orpo_dpo_mix_40k_map_fn` as the dataset mapping function.

For more information on handling datasets and writing dataset mapping functions, please refer to the [Preference Dataset Section](../reward_model/preference_data.md).

### Accelerating Training

When training with preference data, we recommend enabling the [Variable-Length Attention Mechanism](https://xtuner.readthedocs.io/zh-cn/latest/acceleration/varlen_flash_attn.html) to avoid memory waste caused by length differences between chosen and rejected samples within a single preference. You can enable the variable-length attention mechanism by setting `use_varlen_attn=True`.

XTuner also supports many training acceleration methods. For details on how to use them, please refer to the [Acceleration Strategies Section](https://xtuner.readthedocs.io/zh-cn/latest/acceleration/hyper_parameters.html).

# 基于数据长度分组的 Sampler

生成式大模型（例如LLM）的训练数据往往是不定长的，这就导致同一批次（batch）内的数据长短不一。为实现并行化训练，一种常见的做法是将同一批次的数据填充到最长长度。例如，一个批次数据内各样本的长度分别为 10、2、5、3，那么数据处理阶段会将每个样本都填充到最长长度 10，这就导致了计算资源的浪费：有效训练长度为 10 + 2 + 5 + 3 = 20，实际训练长度为 10 + 10 + 10 + 10 = 40，效率仅为 50%。

现阶段有两种技术方案可以解决 / 缓解这一问题（两者选其一即可，优先考虑 **数据拼接技术**）：

1. 利用 **数据拼接技术**，将多条数据拼接至训练支持的最大长度。这一做法可以确保同一批次内的数据长度完全一致，进而避免了填充数据所导致的训练效率降低。具体可参考 [这里](https://github.com/InternLM/xtuner/blob/main/docs/zh_cn/accelerate/pack_to_max_length.md)。
2. （本文）利用 **基于数据长度分组的 Sampler**，在构建批次数据时，基于实际长度进行排序，确保同一批次内的数据长度尽可能相近，进而尽可能减少填充的长度。

## 在 XTuner 中使用 基于数据长度分组的 Sampler

XTuner 中基于数据长度分组的 Sampler 的实现在 [这里](https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/samplers/length_grouped.py)。用户可以通过在配置文件中修改 `train_dataloader` 的 `sampler` 参数进行配置。以 [internlm2_chat_7b_qlora_oasst1_512_e3](https://github.com/InternLM/xtuner/blob/main/xtuner/configs/internlm/internlm2_chat_7b/internlm2_chat_7b_qlora_oasst1_512_e3.py) 配置文件为例，其默认是使用随机的 Sampler，我们可以通过下列修改使其使用 基于数据长度分组的 Sampler：

```diff
- from mmengine.dataset import DefaultSampler
+ from xtuner.dataset.samplers import LengthGroupedSampler

batch_size = 16  # per_device
accumulative_counts = 1

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
-   sampler=dict(type=DefaultSampler, shuffle=True),
+   sampler=dict(
+       type=LengthGroupedSampler,
+       length_property='length',
+       per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))
```

其中，`length_property` 需要传入获取数据集长度的“属性”，这一数值在通过 `process_hf_dataset` 构建数据集时会自动设置为 `'length'`（因此，如果使用自定义的数据类，请确保这一属性的正确设置）。

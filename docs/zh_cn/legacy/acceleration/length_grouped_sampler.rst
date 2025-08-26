.. _length_grouped_sampler:

数据分组
========================

.. raw:: html

   <html xmlns="http://www.w3.org/1999/xhtml"><head></head><body><div align="center">
   <img src="https://github.com/InternLM/xtuner/assets/36994684/779c5429-1f3c-4158-8261-289ba16c347a" width="728" data-src="https://github.com/InternLM/xtuner/assets/36994684/779c5429-1f3c-4158-8261-289ba16c347a" onerror="this.style.display = 'none';" />
   </div></body></html>

生成式大模型（例如LLM）的训练数据往往是不定长的，这就导致同一批次（batch）内的数据长短不一。为实现并行化训练，一种常见的做法是将同一批次的数据填充到最长长度。然而，这一填充（Pad）操作会导致训练的低效。如上图，假设数据内各样本的长度分别为
2、3、7、9，期望分为2个批次进行训练，那么如果使用默认的随机采样器（左侧），数据处理阶段会引入过多的填充数据，实际效率只有65.6%。

现阶段有两种技术方案可以解决 / 缓解这一问题（两者选其一即可，优先考虑
**数据拼接技术**\ ）：

1. 利用
   **数据拼接技术**\ ，将多条数据拼接至训练支持的最大长度。这一做法可以确保同一批次内的数据长度完全一致，进而避免了填充数据所导致的训练效率降低。具体可参考
   \ :ref:`数据拼接文档 <pack_to_max_length>` \ 。

   :优点: 可以合并多个数据样本，显著降低训练 iter 数，加速效果好。

   :缺点: 随机合并的多个数据样本间会互相影响，进而影响训练效果（实际影响程度未知）；数据进行了合并，丢失了一定数据随机性。

2. （本文）利用
   **基于数据长度分组的采样器**\ ，在构建批次数据时，基于实际长度进行排序，确保同一批次内的数据长度尽可能相近，进而尽可能减少填充的长度。如上图右侧，利用该采样器后，同样的数据效率将提升至87.5%。

   :优点: 每条数据依然独立存在（独立计算
      attention），避免数据拼接技术导致的数据样本间的互相影响；数据进行了分组，丢失了一定数据随机性。

   :缺点: 在数据样本长度比较一致的情况下，加速效果一般。

使用 ``LengthGroupedSampler``
-----------------------------------------

XTuner 中基于数据长度分组的采样器 的实现在
`这里 <https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/samplers/length_grouped.py>`__\ 。用户可以通过在配置文件中修改
``train_dataloader`` 的 ``sampler`` 参数进行配置。以
`internlm2_chat_7b_qlora_oasst1_512_e3 <https://github.com/InternLM/xtuner/blob/main/xtuner/configs/internlm/internlm2_chat_7b/internlm2_chat_7b_qlora_oasst1_512_e3.py>`__
配置文件为例，其默认是使用随机的采样器，我们可以通过下列修改使其使用
基于数据长度分组的采样器：

.. code:: diff

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

.. note::
   其中，\ ``length_property``
   需要传入获取数据集长度的“属性”，这一数值在通过 ``process_hf_dataset``
   构建数据集时会自动设置为
   ``'length'``\ （因此，如果使用自定义的数据类，请确保这一属性的正确设置）。

=====================
调整加速策略
=====================

本节将会列举 XTuner 中会影响训练速度的配置项。


max_length
-------------------

``max_length`` 表示在数据预处理过程中，单条数据长度超过 ``max_length`` 的部分会被截断，基本所有实验都会设置该项。

pack_to_max_length
---------------------------

``pack_to_max_length`` 用于配置是否进行\ :ref:`数据集拼接 <pack_to_max_length>` \ 。

``pack_to_max_length = True`` 表示在数据预处理过程中将多条短数据拼接为一条长度为 ``max_length`` 的长数据，该配置可以大幅提升训练速度。

若 ``pack_to_max_length = False``，则推荐将 ``batch_size`` 适度调大以保证训练的稳定性。

use_varlen_attn
---------------------------

``use_varlen_attn`` 用于配置是否在训练过程中使用\ :ref:`Varlen Flash Attention <varlen_flash_attn>` \  。

当 ``use_varlen_attn = True`` 时，要求 ``pack_to_max_length`` 也要设置为 True。在此情况下，每个 token 在注意力计算阶段仅会关注其所在短数据中的所有 tokens （而非整个序列）。

当 ``use_varlen_attn = False`` 时，每个 token 在注意力计算阶段会关注整个序列。

max_position_embeddings
---------------------------------

当需要扩展模型上下文窗口的大小时，需要将 ``max_position_embeddings`` 设置为期望的上下文长度。 **需要保证 max_position_embeddings 不大于 max_length。**\

假设需要将 Llama2-7B 模型支持的上下文长度自 4k 拓展为 32k：

1. 若训练数据集中存在较多长度接近 32k 的数据，则推荐 ``max_length = 32k, pack_to_max_length = False, use_varlen_attn = False, max_position_embeddings = 32k`` 这一配置
2. 若训练数据集中长度接近 32k 的数据量较少甚至没有时，则推荐 ``max_length = 32k, pack_to_max_length = True, use_varlen_attn = False, max_position_embeddings = 32k`` 这一配置

sequence_parallel_size
-------------------------------------------

在使用序列并行策略训练超长序列时， ``sequence_parallel_size`` 个 GPUs 会共同计算一条长序列。而 ``accumulative_counts`` 则用于控制模型参数更新的频率。


accumulative_counts
----------------------------------------------
用于控制模型参数更新的频率；假设需要在 N 块 GPUs 上执行 ``batch_size_per_device = 1, max_length = 128k`` 的训练策略。当设置序列并行维度为 ``sequence_parallel_size`` 后，为了保证训练的等价性， ``accumulative_counts`` 需要设置为原来的 ``sequence_parallel_size`` 倍，因为 128k 长度的序列会被切分为 ``sequence_parallel_size`` 份后分发给 ``sequence_parallel_size`` 个 GPUs 进行训练， ``data_parallel_world_size`` 会变为原来的 :math:`\frac{1}{sequence\_parallel\_size}`。

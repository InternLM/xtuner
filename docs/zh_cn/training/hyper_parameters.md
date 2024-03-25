# 超参数设置

本节将会列举一些在 XTuner 训练过程往往需要搭配设置的超参数，并列举他们搭配使用时各个超参数的含义。

## `max_length`, `pack_to_max_length`, `use_varlen_attn` 和 `max_position_embeddings`

#### max_length

`max_length` 表示在数据预处理过程中，单条数据长度超过 `max_length` 的部分会被截断，基本所有实验都会设置该项。

#### pack_to_max_length

`pack_to_max_length` 用于配置是否进行[数据集拼接](../accelerate/pack_to_max_length.md)。

`pack_to_max_length = True` 表示在数据预处理过程中将多条短数据拼接为一条长度为 `max_length` 的长数据，该配置可以大幅提升训练速度。

若 `pack_to_max_length = False`，则推荐将 `batch_size` 适度调大以保证训练的稳定性。

#### use_varlen_attn

`use_varlen_attn` 用于配置是否在训练过程中使用[变长注意力机制](../accelerate/varlen_flash_attn.md)。

当 `use_varlen_attn = True` 时，要求 `pack_to_max_length` 也要设置为 True。在此情况下，每个 token 在注意力计算阶段仅会关注其所在短数据中的所有 tokens （而非整个序列）。

当 `use_varlen_attn = False` 时，每个 token 在注意力计算阶段会关注整个序列。

#### max_position_embeddings

当需要扩展模型上下文窗口的大小时，需要将 `max_position_embeddings` 设置为期望的上下文长度。**需要保证 `max_position_embeddings` 不大于 `max_length`。**

假设需要将 Llama2-7B 模型支持的上下文长度自 4k 拓展为 32k：

1. 若训练数据集中存在较多长度接近 32k 的数据，则推荐 `max_length = 32k, pack_to_max_length = False, use_varlen_attn = False, max_position_embeddings = 32k` 这一配置
2. 若训练数据集中长度接近 32k 的数据量较少甚至没有时，则推荐 `max_length = 32k, pack_to_max_length = True, use_varlen_attn = False, max_position_embeddings = 32k` 这一配置

- `max_length` 表示在数据预处理过程中，单条数据长度超过 `max_length` 的部分会被截断，基本所有实验都会设置该项。
- `pack_to_max_length` 用于配置是否进行[数据集拼接](../accelerate/pack_to_max_length.md)，`pack_to_max_length = True`表示在数据预处理过程中将多条短数据拼接为一条长度为 `max_length` 的长数据，该配置可以大幅提升训练速度。
- `use_varlen_attn` 用于配置是否在训练过程中使用[变长注意力机制](../accelerate/varlen_flash_attn.md)。当 `use_varlen_attn = True` 时，要求 `pack_to_max_length` 也要设置为 True，每个 token 在注意力计算阶段仅会关注其所在短数据中的所有 tokens （而非整个序列）。
- 当需要扩展模型上下文窗口的大小时，需要将 `max_position_embeddings` 设置为期望的上下文长度。**设置 `max_position_embeddings` 后，需要保证训练数据集中存在一定量的长度接近 `max_position_embeddings` 的数据。**

本节以 Llama2-7B 模型（最大上下文长度 4k）为例介绍上述四个超参数在不同设置时的含义。

| max_length | pack_to_max_length | use_varlen_attn | max_position_embeddings |      Description      |
| :--------: | :----------------: | :-------------: | :---------------------: | :-------------------: |
|    32k     |        True        |      False      |          None           | [Case1 Forbidden](<>) |
|    32k     |        True        |      False      |           32k           |      [Case2](<>)      |
|    32k     |        True        |      False      |           64k           | [Case3 Forbidden](<>) |
|    32k     |        True        |      True       |          None           |      [Case4](<>)      |
|    32k     |        True        |      True       |           32k           |      [Case5](<>)      |
|     4k     |       False        |      False      |          None           |      [Case6](<>)      |
|    32k     |       False        |      False      |          None           | [Case6 Forbidden](<>) |
|    32k     |       False        |      False      |           32k           |      [Case7](<>)      |
|    32k     |       False        |      True       |            -            | [Case8 Forbidden](<>) |

### Case 1

| max_length | pack_to_max_length | use_varlen_attn | max_position_embeddings |
| :--------: | :----------------: | :-------------: | :---------------------: |
|    32k     |        True        |      False      |          None           |

以上配置表示：

- 在数据预处理阶段，单条数据长度超过 32k 的会被截断
- 在后续预处理阶段，会将多条数据拼接为一条 32k 长度的长数据
- 在注意力模块计算过程中，32k 长序列中的每个 token 都会与其他 tokens 建立联系
- 不需要扩展 Llama2-7B 模型的上下文长度。**由于 Llama2-7B 模型支持的最大上下文长度为 4k，远小于计算注意力时的上下文长度 32k，因此模型训练效果会受到影响。**

### Case 2

| max_length | pack_to_max_length | use_varlen_attn | max_position_embeddings |
| :--------: | :----------------: | :-------------: | :---------------------: |
|    32k     |        True        |      False      |           32k           |

与 Case 1 相比 Case 2 将 `max_position_embeddings` 设为了 32k。XTuner 会根据 `max_position_embeddings` 的配置对模型的 RotaryEmbedding 层做线性差值，以拓展模型的上下文能力。

### Case 3

| max_length | pack_to_max_length | use_varlen_attn | max_position_embeddings |
| :--------: | :----------------: | :-------------: | :---------------------: |
|    32k     |        True        |      False      |           64k           |

与 Case 2 相比，Case 3 将 `max_position_embeddings` 进一步扩大为 64k。**这是不推荐的，因为训练数据的最长长度为 32k。**

### Case 4

| max_length | pack_to_max_length | use_varlen_attn | max_position_embeddings |
| :--------: | :----------------: | :-------------: | :---------------------: |
|    32k     |        True        |      True       |          None           |

以上配置表示：

- 在数据预处理阶段，单条数据长度超过 32k 的会被截断。
- 在后续预处理阶段，会将多条数据拼接为一条 32k 长度的长数据。
- 每个 token 在注意力计算阶段仅会关注其所在短数据中的所有 tokens （而非整个序列）。
- 不需要扩展 Llama2-7B 模型的上下文长度。**由于仅对超过 32k 的单条数据做截断，而 Llama2-7B 模型支持的最长上下文是 4k，因此需要保证数据集中长度超过 4k 的数据量较少，否则会影响训练。**

### Case 5

| max_length | pack_to_max_length | use_varlen_attn | max_position_embeddings |
| :--------: | :----------------: | :-------------: | :---------------------: |
|    32k     |        True        |      True       |           32k           |

与 Case 1 相比 Case 2 将 `max_position_embeddings` 设为了 32k。XTuner 会根据 `max_position_embeddings` 的配置对模型的 RotaryEmbedding 层做线性差值，以拓展模型的上下文能力。**类似的，如果数据集中长度接近 32k 的数据量较少，这一配置可能会影响训练。**

### Case 6

- 在数据预处理阶段，单条数据长度超过 4k 的会被截断。
- 在后续预处理阶段，不会使用数据拼接策略。
- 由于没有做数据集拼接，自然也不需要在计算注意力时使用变长注意力机制。
- 不需要扩展 Llama2-7B 模型的上下文长度。

### Case 7

与 Case 6 相比，Case 7 将 `max_length` 设为了 32k。**由于没有设置 `max_position_embeddings` ，因此当数据集中存在较多长度大于 4k 的数据时，模型训练会受到影响。**

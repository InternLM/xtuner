# Xtuner Dataset 设计方案

## 背景与问题
- 对于极长文本样本，直接整行送入 tokenizer 时容易出现极端慢节点（例如单行约 100 万 token，单次 tokenize 耗时可达数分钟），会严重拖慢训练。
- 当某个数据配方训练到中途，需要更换各数据源或任务之间的配方比例：
  - 核心难点在于：新配方需要基于「已经训练过的样本比例」来设计，同时训练流程仍需支持 resume。
- 其他问题：在较大 `seq_len` 下，长短文本在 pack 时容易出现不平衡，影响不同 rank 间的数据均匀性。

## 总体设计
1. 提供 LongTextTokenizeFn，解决长文本导致的 tokenizer 慢节点问题。
2. 通过自定义 Pack 与 Sampler，支持「离线生成数据配方 + 运行时严格消费顺序」，从而在中途可以平滑更换数据配方。
3. 补充讨论在大 `seq_len` 场景下长短文本 pack 不平衡问题及与自定义 Sampler 的配合方案。

## LongTextTokenizeFn 解决长文本慢节点问题

### 背景与思路
在分布式训练中，每个 GPU 一次通常会读取一整行样本。如果直接把一整行（例如包含约 100 万 token 的长文）交给 tokenizer，极端情况下单次 tokenize 可能耗时约 5 分钟，导致所有进程被这一慢节点「卡死」。  
为此，需要一种「按固定窗口动态分块 tokenize」的方案，但又不能简单按字节粗暴切分，否则会把同一个单词拆到不同 chunk 中，破坏 tokenizer 的分词边界。  
LongTextTokenizeFn 的核心思路是：对原始字符序列按长度 `tokenizer_chunk_chars` 切分，并在相邻块之间保留长度为 `overlap_chars` 的重叠区域；对每一块单独调用 tokenizer，然后在相邻两次 tokenize 结果之间通过寻找最长公共子串（LCS）来对齐重叠区域，只保留「不会被截断」的那部分 token，从而在保证 tokenizer 语义一致性的前提下，实现可控窗口、避免超长文本导致的单次极慢 tokenize。

### 迁移到 Xtuner 的方案

**目标（Purpose）**  
通过「字符级分块 + 重叠窗口 + LCS 边界对齐与合并」的方式，将超长文本拆分成多个中等长度 chunk，避免单个 1M-token 文本导致 5 分钟级别的 tokenizer 阻塞。

**关键参数（Parameters）**

| 参数名 | 默认值 | 含义 |
|--------|--------|------|
| `chunk_size` | 4096 | 每个输出 chunk 期望的 token 数（逻辑上的目标长度） |
| `tokenizer_chunk_chars` | 4096 | 每次送入 tokenizer 的字符窗口大小 |
| `overlap_chars` | 512 | 相邻窗口的字符级重叠长度，用于保证边界 token 的正确性 |
| `min_chunk_tokens` | 0 | 若最后一个 chunk 的长度低于该阈值，则可以选择丢弃该尾部短 chunk |

**两阶段行为（Two-phase behavior）**

| 阶段（`state`） | `__call__` 输入 | `__call__` 输出 |
|-----------------|-----------------|-----------------|
| `"cache"` | `{"text": str, "state": "cache"}` | `{"num_tokens": [int], "chunks": [{"char_start": xx, "char_end": xx, "token_start_offset": xx}]}` —— 总是返回（即便是短文本只形成 1 个 chunk），用于离线统计与切分规划；其中 `token_start_offset` 表示该 chunk 在整体 tokenize 结果中的起始 token 偏移，用于后续按 token 维度切片 |
| `"runtime"` | `{"text": str, "state": "runtime", "char_start": int, "char_end": int, "token_start_offset": int}` | `{"input_ids", "labels", "num_tokens", "char_start", "char_end", "token_start_offset"}` —— 只对 `text[char_start:char_end]` 进行真正 tokenize，并从 `token_start_offset` 开始截取最多 `self.max_length` 长度的输出（如 `input_ids[token_start_offset:token_start_offset+max_length]`），用于在线训练 |

**对其他模块的影响**

对 `JsonlDataset`：
- `__len__()`：由「样本数」变为「chunk 数」，即一个超长样本可能产生多个逻辑样本。
- `__getitem__()`：由「返回单个样本的 token id」变为「返回单个 chunk 的 token 序列」。

对 `JsonlDataset` 之后的模块（如 `PackDataset`、`Sampler` 等）接口保持不变，只是其看到的「样本粒度」从原先的行级样本变为 chunk 级样本。

## 自定义 Pack 和 Sampler 解决中途换配方问题

### 自定义 Pack

1. **Pack Config 文件（JSONL 或 NPY）**

每个 pack 由若干 **sample slice** 组成，每个 slice 描述「取某个样本的哪一段 token（或全部 token）」：

以 **JSONL 格式** 为例（每行一个 pack）：

```jsonl
{"samples": [[path0, 42, 0, 512, 0], [path0, 43, 0, 256, 256], [path1, 7, 128, 384, 0]]}
{"samples": [[path1, 100, 384, 512, 0], [path2, 5, 0, 1024, 0], [path2, 6, 0, 512, 512]]}
```

- `samples[i] = [dataset_path, sample_idx, char_start, char_end, token_start_offset]`
- `dataset_path`：对应 Jsonl 文件路径，即 `JsonlDataset.anno_path`。
- `sample_idx`：该 dataset 内的逻辑下标（对应 post-filter-sampleratio 之后的 `sampled[]` 下标）。
  - **文件与索引一一对应原则**：开启自定义 Pack 时，在 `JsonlDataset` 中强制关闭 filter，并设置 `sampleratio = 1`，以确保 `sample_idx` 与真实文件行号一一对应。这样可以使 JSONL 文件中的样本行与 Pack Config 中的 `sample_idx` 明确对应，降低配置出错概率。
  - 如需 filter、sampleratio 等逻辑，应通过生成 Pack Config 时离线实现，而不是在 `JsonlDataset` 内部动态实现。
- `char_start`：从该样本的 `text[char_start:]` 开始取（含起始，inclusive）。用于运行时检查和上一节 cache 内容是否对应。
- `char_end`：取到 `text[:char_end]`（不含结尾，exclusive）。用于运行时检查。
- 对于 `LongTextTokenizeFn`，同一个样本可以在多个不同 pack 中出现（长文被截断后分配到多个 pack），运行时会检查 `char_start` 与 `char_end` 的合法性。
- `token_start_offset`：表示此次输出 token 结果在该样本整体 tokenize 结果中的起始 token 偏移，从该位置开始最多截取 `max_len` 长度的输出（即 `input_ids[token_start_offset:token_start_offset+max_len]`），用于精确描述每个 pack slice 在 token 维度上的位置。
- 对于普通 TokenizeFn（即非长文本场景），`char_start` 和 `char_end` 均可设置为 `-1` 表示「不基于 token 范围切片」，此时 `token_start_offset` 也可以固定为 `0`。

2. **运行时严格校验逻辑**

在初始化 PackDataset 时，需要对每一个 pack 做强校验：
- `dataset_path` 不存在 → 抛出错误。
- `sample_idx` 越界（超过 `len(dataset.sampled)`）→ 抛出错误。
- `char_start < 0` 或 `char_end > num_tokens` 或 `char_start >= char_end`（非 0 情况）→ 抛出错误（`char_start` 和 `char_end` 均为 -1 的特例除外，对应普通 TokenizeFn 的 JsonlDataset）。
- `token_start_offset < 0` 或 `token_start_offset >= num_tokens`（在 LongTextTokenizeFn 场景下）→ 抛出错误，保证 token 级切片位置合法。
- 记每个 slice 的长度为 `len_i = char_end - char_start`，若
  - `sum(len_i) < pack_max_length`：由 `short_pack_strategy` 控制：
    - `"error"`：直接 `raise ValueError`；
    - `"padding"`：在末尾补齐 pad token，同时在 labels 中对应位置填 -100。
  - `sum(len_i) > pack_max_length`：由 `long_pack_strategy` 控制：
    - `"error"`：直接 `raise ValueError`；
    - `"truncate"`：截断最后一个 slice，使总长度恰好等于 `pack_max_length`。
- 默认策略均为 `"error"`。
- 出于「索引与配置一一对应」的考虑，这里不支持 `"skip"` 策略（与前文 JsonlDataset 不支持 filter/sampleratio 的原因一致），以确保 PackConfig 中的样本索引可以稳定地与后续 SamplerConfig 中的消费顺序对应。

在实际取样本时，再次做一致性检查：  
对于 `samples[i] = [dataset_path, sample_idx, char_start, char_end, token_start_offset]`：
- 若 `char_start` 和 `char_end` 均为 -1，则检查 tokenize 结果中不含 `char_start` 等字段，表示这是普通 TokenizeFn 情况（此时 `token_start_offset` 一般为 `0` 且不参与校验）。
- 否则，检查 tokenize 结果中记录的 `char_start`、`char_end` 与 `token_start_offset` 与配置一致，确保 LongTextTokenizeFn 的切片语义在运行时没有被破坏。

3. **resume 行为**

自定义 Pack 本身不维护运行时状态（它只是一个「离线规划好的切片表」），因此无需单独做 resume。训练过程的断点续训只需依赖上层的 Sampler 进度与 checkpoint。

4. **兼容性**

上述设计同时支持：
- 基于 LongTextTokenizeFn 的 `JsonlDataset`；
- 基于普通 TokenizeFn 的 `JsonlDataset`。  
二者可以在同一个 PackConfig 中混合出现，通过 `char_start`/`char_end` 是否为 -1 来区分。

### 自定义 Sampler

1. **Sampler Config 文件**

SamplerConfig 描述了一个有序的「pack 全局消费序列」（1D 整数数组）：
- JSONL：单行 JSON 数组，如 `[3, 1, 7, 2, 0, 5, 4, 6]`；
- NPY：如 `sampler_order.npy`，shape 为 `(num_steps,)`。

2. **运行时检查**

加载 SamplerConfig 时，需要验证：
- 所有值均在 `[0, len(dataset))` 范围内，否则 `raise ValueError`。

3. **resume 行为**

Sampler 在运行时维护一个当前 step 指针。  
在保存 checkpoint 时记录该指针；在恢复时只需恢复该指针，即可从上次中断的位置继续按同一消费顺序读取 pack。

### 如何支持中途更换数据配方

在训练过程中，如果某个数据配方已经训练到中途，需要调整不同数据源或任务的比例，可以按以下方式实现：
- 通过当前保存的 step 指针以及既有的 PackConfig、SamplerConfig，可以精确统计「哪些 pack / sample slice 已经被消费」，从而推算「每个数据源已经训练的样本量或 token 量」。
- 基于上述统计信息，离线生成新的 PackConfig 与 SamplerConfig，使得新的配方在「已训练样本」之上继续按新的比例分配后续样本。
- 重新启动训练时，加载模型 checkpoint，但不加载旧的 dataset 状态，通过新的 PackConfig + SamplerConfig 来驱动后续训练，从而完成「中途换配方，但总体训练进度可控且可复现」。

### 离线脚本

围绕自定义 Pack 与 Sampler，建议提供以下配套离线脚本：
- 生成脚本：根据原始 JsonlDataset、配方比例与 LongTextTokenizeFn 的统计信息生成 PackConfig 与 SamplerConfig。
- 可视化脚本：对不同数据源、样本长度分布、token 占比等进行可视化，帮助检查配方是否合理。
- 多 rank 离线数据迭代脚本：在不启动完整训练的前提下，复现真实多进程数据迭代过程，用于 debug 与问题重现。

## 其他问题

当使用较大的 `seq_len` 时，长短文本在 pack 时的不平衡问题会变得更加明显：  
- 已有的 `LengthGroupSampler` 可以在默认策略下，以长度分桶的方式在不同 rank 之间做 pack 均衡。  
- 当启用自定义 Sampler 时，需要在生成 SamplerConfig 时显式考虑该问题，`SamplerConfig` 就要代替`LengthGroupSampler` 承担「负载均衡」的职责。


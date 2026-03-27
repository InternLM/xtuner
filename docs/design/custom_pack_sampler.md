# Xtuner Dataset 设计方案

## 背景与问题

- 对于极长文本样本，直接整行送入 tokenizer 时容易出现极端慢节点（例如单行约 100 万 token，单次 tokenize 耗时可达数分钟），会严重拖慢训练。
- 当某个数据配方训练到中途，需要更换各数据源或任务之间的配方比例：
  - 核心难点在于：新配方需要基于「已经训练过的样本比例」来设计，同时训练流程仍需支持 resume。
- 其他问题：在较大 `seq_len` 下，长短文本在 pack 时容易出现不平衡，影响不同 rank 间的数据均匀性。

## 总体设计

1. 提供 LongTextTokenizeFn，解决长文本导致的 tokenizer 慢节点问题。
2. 通过预定义 Pack 与 Sampler，支持「离线生成数据配方 + 运行时严格消费顺序」，从而在中途可以平滑更换数据配方。
3. 补充讨论在大 `seq_len` 场景下长短文本 pack 不平衡问题及与预定义 Sampler 的配合方案。

## LongTextTokenizeFn 解决长文本慢节点问题

### 背景与思路

在分布式训练中，每个 GPU 一次通常会读取一整行样本。如果直接把一整行（例如包含约 100 万 token 的长文）交给 tokenizer，极端情况下单次 tokenize 可能耗时约 5 分钟，导致所有进程被这一慢节点「卡死」。
为此，需要一种「按固定窗口动态分块 tokenize」的方案，但又不能简单按字节粗暴切分，否则会把同一个单词拆到不同 chunk 中，破坏 tokenizer 的分词边界。
LongTextTokenizeFn 的核心思路是：对原始字符序列按长度 `tokenizer_chunk_chars` 切分，并在相邻块之间保留长度为 `overlap_chars` 的重叠区域；对每一块单独调用 tokenizer，然后在相邻两次 tokenize 结果之间通过寻找最长公共子串（LCS）来对齐重叠区域，只保留「不会被截断」的那部分 token，从而在保证 tokenizer 语义一致性的前提下，实现可控窗口、避免超长文本导致的单次极慢 tokenize。

### 迁移到 Xtuner 的方案

**目标（Purpose）**
通过「字符级分块 + 重叠窗口 + LCS 边界对齐与合并」的方式，将超长文本拆分成多个中等长度 chunk，避免单个 1M-token 文本导致 5 分钟级别的 tokenizer 阻塞。

**关键参数（Parameters）**


| 参数名                  | 默认值 | 含义                                                            |
| ------------------------- | -------- | ----------------------------------------------------------------- |
| `chunk_size`            | 4096   | 每个输出 chunk 期望的 token 数（逻辑上的目标长度）              |
| `tokenizer_chunk_chars` | 4096   | 每次送入 tokenizer 的字符窗口大小                               |
| `overlap_chars`         | 512    | 相邻窗口的字符级重叠长度，用于保证边界 token 的正确性           |
| `min_chunk_tokens`      | 0      | 若最后一个 chunk 的长度低于该阈值，则可以选择丢弃该尾部短 chunk |

**两阶段行为（Two-phase behavior）**


| 阶段（`state`） | `__call__` 输入                                                                                    | `__call__` 输出                                                                                                                                                                                                                                                                                        |
| ----------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"cache"`       | `{"text": str, "state": "cache"}`                                                                  | `{"num_tokens": [int], "chunks": [{"char_start": xx, "char_end": xx, "token_start_offset": xx}]}` —— 总是返回（即便是短文本只形成 1 个 chunk），用于离线统计与切分规划；其中 `token_start_offset` 表示该 chunk 在整体 tokenize 结果中的起始 token 偏移，用于后续按 token 维度切片                    |
| `"runtime"`     | `{"text": str, "state": "runtime", "char_start": int, "char_end": int, "token_start_offset": int}` | `{"input_ids", "labels", "num_tokens", "char_start", "char_end", "token_start_offset"}` —— 只对 `text[char_start:char_end]` 进行真正 tokenize，并从 `token_start_offset` 开始截取最多 `self.max_length` 长度的输出（如 `input_ids[token_start_offset:token_start_offset+max_length]`），用于在线训练 |

**对其他模块的影响**

对 `JsonlDataset`：

- `__len__()`：由「样本数」变为「chunk 数」，即一个超长样本可能产生多个逻辑样本。
- `__getitem__()`：由「返回单个样本的 token id」变为「返回单个 chunk 的 token 序列」。
  - 使用 `LongTextPretrainTokenizeFunction` 时，返回 `LongTextDataItem`（含 `char_start`、`char_end`、`token_start_offset` 字段）。
  - 使用普通 `TokenizeFunction` 时，返回 `DataItem`（仅含 `input_ids`、`labels`、`num_tokens`）。

对 `JsonlDataset` 之后的模块（如 `PackDataset`、`Sampler` 等）接口保持不变，只是其看到的「样本粒度」从原先的行级样本变为 chunk 级样本。

## 预定义 Pack 和 Sampler 解决中途换配方问题

### 预定义 Pack

1. **Pack Config 文件（NPY 目录格式）**

每个 pack 由若干 **sample slice** 组成，每个 slice 描述「取某个样本的哪一段（或全部）token」。Pack Config 存储为一个目录，目录下包含以下文件：

| 文件名 | 格式 | 含义 |
| --- | --- | --- |
| `boundaries.npy` | `int64` ndarray，shape `(num_packs + 1,)` | CSR 边界数组，`boundaries[i]:boundaries[i+1]` 是第 i 个 pack 的 slice 行范围 |
| `samples.npy` | `int64` ndarray，shape `(total_slices, 6)` | 所有 slice 的数值字段，列顺序见下表 |
| `paths.json` | JSON 数组，`list[str]` | `path_id → dataset_path` 映射，避免使用 `allow_pickle` |

`samples.npy` 的列定义（列索引 0–5）：

| 列索引 | 字段名 | 含义 |
| --- | --- | --- |
| 0 | `path_id` | `paths.json` 中的整数下标，对应实际的 `dataset_path` |
| 1 | `sample_idx` | 该 dataset 内的逻辑下标 |
| 2 | `char_start` | 字符级起始（inclusive）；普通 TokenizeFn 时固定为 `-1` |
| 3 | `char_end` | 字符级结束（exclusive）；普通 TokenizeFn 时固定为 `-1` |
| 4 | `token_start_offset` | token 起始偏移（inclusive） |
| 5 | `token_end_offset` | token 结束偏移（exclusive） |

字段语义补充说明：

- **文件与索引一一对应原则**：开启预定义 Pack 时，在 `JsonlDataset` 中强制关闭 filter（`disable_filter=True`），并设置 `sample_ratio=1`、`enable_sequential_sampler=True`，以确保 `sample_idx` 与真实文件行号（或 chunk 索引）一一对应。如需 filter、sample_ratio 等逻辑，应通过生成 Pack Config 时离线实现。
- `char_start`/`char_end`：均为 `-1` 时表示普通 TokenizeFn（全量样本），否则为 `LongTextTokenizeFn` 的字符切片范围，运行时用于一致性校验。
- `token_start_offset`/`token_end_offset`（左闭右开）：取 `input_ids[token_start_offset:token_end_offset]`。
  - 基于 `LongTextTokenizeFn` 时：`JsonlDataset.__getitem__` 返回的 `LongTextDataItem` 已按此范围截断，`PresetPackDataset.__getitem__` 只做校验。
  - 基于普通 `TokenizeFn` 时：`DataItem` 为全量 token，`PresetPackDataset.__getitem__` 负责执行截取。
- 普通 TokenizeFn 场景：`char_start = char_end = -1`，`token_start_offset = 0`，`token_end_offset` = 该样本实际 token 数。

**`load_config` 加载函数：**

```python
def load_config(path: str, mmap: bool = True) -> dict:
    """从 NPY 目录加载 pack config。

    Args:
        path:  存放配置文件的目录路径。
        mmap:  若为 True，使用 mmap_mode='r' 以只读内存映射方式加载 int64 数组，
               避免将大数组完整读入 RAM；paths.json 始终完整加载（数据量小）。

    Returns:
        dict：
            'boundaries': np.ndarray, shape (num_packs+1,), dtype int64
            'samples':    np.ndarray, shape (total_slices, 6), dtype int64
            'paths':      list[str]，path_id -> dataset_path 映射
    """
```

写入示例（离线生成工具使用）：

```python
import json, numpy as np, os

np.save(os.path.join(pack_dir, "boundaries.npy"),
        np.array([0, 1, 3, ...], dtype=np.int64))
np.save(os.path.join(pack_dir, "samples.npy"),
        np.array([[path_id, s_idx, c_start, c_end, tok_start, tok_end], ...], dtype=np.int64))
with open(os.path.join(pack_dir, "paths.json"), "w") as f:
    json.dump(["path/to/ds0.jsonl", "path/to/ds1.jsonl", ...], f)
```

2. **运行时严格校验逻辑**

在初始化 `PresetPackDataset` 时，通过 `load_config` 加载 mmap 只读数组后，使用 **numpy 向量化操作**对全量数组做结构性校验，避免 Python 逐行循环（对于亿级 slice 而言逐行校验开销过高）：

- **结构校验**：`boundaries[-1] == len(samples)`，`samples.shape[1] == 6`。
- **字段取值校验**（向量化，无需全量加载进 RAM，numpy 会按页读取 mmap）：
  - `samples[:, 0]`（`path_id`）均在 `[0, len(paths))` 范围内。
  - `samples[:, 4]`（`token_start_offset`）均 `>= 0`。
  - `samples[:, 5]`（`token_end_offset`）均 `> samples[:, 4]`。
  - 对 `char_start != -1` 的行：`char_start >= 0` 且 `char_end > char_start`（用 mask 筛出后校验）。
- **per-pack token 总量校验**（`short_pack_strategy == "error"` 或 `long_pack_strategy == "error"` 时）：利用 `np.add.reduceat` 按 `boundaries[:-1]` 对 `token_end_offset - token_start_offset` 做分段求和，一次向量化操作得到每个 pack 的 token 总量，与 `pack_max_length` 比较。
- **`long_pack_strategy == "truncate"` 情况**：不在 `__init__` 中修改 samples 数据（mmap 为只读），截断逻辑推迟到 `__getitem__` 中按需执行。
- 不支持 `"skip"` 策略（与「索引与配置一一对应」原则冲突）。

在实际取样本时（`PresetPackDataset.__getitem__(i)`），通过 **段式读取** `_samples[_boundaries[i]:_boundaries[i+1]]` 获取该 pack 的 slice 行，再逐行处理：

- 对每一行 `[path_id, s_idx, char_start, char_end, tok_start, tok_end]`：
  - 调用 `datasets[path_to_ds_idx[paths[path_id]]][s_idx]` 得到 `item`。
  - 若 `char_start == -1`（普通 TokenizeFn）：校验 `item` 不含 `char_start` 字段；执行 `input_ids[tok_start:tok_end]` 截取。
  - 否则（LongTextTokenizeFn）：校验 `item` 为 `LongTextDataItem`，且 `item.char_start == char_start`、`item.char_end == char_end`、`item.token_start_offset == tok_start`；不再做额外截取。
  - 若 `long_pack_strategy == "truncate"` 且当前累计 token 数加上本 slice 会超出 `pack_max_length`，则对本 slice 截取 `input_ids[:remaining]` 后停止（不再处理后续 slice）。
- `short_pack_strategy == "padding"` 时，在所有 slice 处理完毕后在末尾追加 pad DataItem。

3. **resume 行为**

预定义 Pack 本身不维护运行时状态（它只是一个「离线规划好的切片表」），因此无需单独做 resume。训练过程的断点续训只需依赖上层的 Sampler 进度与 checkpoint。

4. **兼容性**

上述设计同时支持：

- 基于 `LongTextTokenizeFn` 的 `JsonlDataset`：`char_start`/`char_end` 为实际字符范围，`token_start_offset`/`token_end_offset` 为 token 起止偏移；tokenize 阶段已完成截取，运行时只做校验。
- 基于普通 `TokenizeFn` 的 `JsonlDataset`：`char_start`/`char_end` 均为 `-1`，`token_start_offset` 为 `0`，`token_end_offset` 为该样本实际 token 数；运行时在 `PresetPackDataset.__getitem__` 中执行截取。

二者可以在同一个 PackConfig 中混合出现，通过 `char_start`/`char_end` 是否为 `-1` 来区分。

**内存占用分析：**

使用 `mmap=True` 时，`boundaries.npy` 和 `samples.npy` 通过 OS mmap 以只读方式映射，操作系统按需将文件页加载进物理内存，未访问的页不占用 RAM。以 20 亿条 slice（`samples.npy` 约 96 GB）为例，若每次训练 step 仅访问其中极小一部分，常驻 RAM 可控制在操作系统的 page cache 范围内，远低于完整加载。`paths.json` 数据量极小，始终完整加载。

5. **DataloaderConfig 集成**

当 `DataloaderConfig.pack_level = "custom"` 时，自动创建 `PresetPackDataset`，并强制每个 `DatasetConfig` 的以下设置（无需用户手动指定），以保证 `sample_idx` 与文件行号一一对应：

- `sample_ratio = 1.0`
- `enable_sequential_sampler = True`
- `disable_filter = True`（新增参数，跳过 `JsonlDataset` 中所有 filter 逻辑，包括 num_tokens==0 过滤和 max_length 过滤）

`JsonlDataset.__init__` 新增 `disable_filter: bool = False` 参数；`DatasetConfig` 同步新增该字段。

### 预定义 Sampler（PresetSampler）

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

围绕预定义 Pack 与 Sampler，建议提供以下配套离线脚本：

- 生成脚本：根据原始 JsonlDataset、配方比例与 LongTextTokenizeFn 的统计信息生成 PackConfig 与 SamplerConfig。
- 可视化脚本：对不同数据源、样本长度分布、token 占比等进行可视化，帮助检查配方是否合理。
- 多 rank 离线数据迭代脚本：在不启动完整训练的前提下，复现真实多进程数据迭代过程，用于 debug 与问题重现。

## 其他问题

当使用较大的 `seq_len` 时，长短文本在 pack 时的不平衡问题会变得更加明显：

- 已有的 `LengthGroupSampler` 可以在默认策略下，以长度分桶的方式在不同 rank 之间做 pack 均衡。
- 当启用预定义 Sampler 时，需要在生成 SamplerConfig 时显式考虑该问题，`PresetSampler` 就要代替 `LengthGroupSampler` 承担「负载均衡」的职责。

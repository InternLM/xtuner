# Rule Summary: xtuner/v1/datasets Pretrain Dataloader Pipeline

> Status: Updated 2026-03-10 — added Rule 2b (LongTextPretrainTokenizeFunction), updated Rules 3 & 4 for chunked tokenization.

---

## Paths

| Role | File |
|------|------|
| Pipeline orchestration | `xtuner/v1/datasets/config.py` — `DataloaderConfig.build()` |
| Tokenize fn (pretrain) | `xtuner/v1/datasets/pt_tokenize_fn/text.py` — `PretrainTokenizeFunction` |
| Tokenize fn (long text) | `xtuner/v1/datasets/pt_tokenize_fn/long_text.py` — `LongTextPretrainTokenizeFunction` |
| Single-file dataset | `xtuner/v1/datasets/jsonl.py` — `JsonlDataset` |
| Packing | `xtuner/v1/datasets/packing.py` — `HardPackDataset`, `ExpandSoftPackDataset` |
| Sampler | `xtuner/v1/datasets/sampler.py` — `LengthGroupedSampler`, `ParallelSampler` |
| Resume helpers | `xtuner/v1/datasets/resume.py` — `get_dataloader_state / load_dataloader_state` |
| Dataloader wrapper | `xtuner/v1/datasets/dataloader.py` — `Dataloader` |
| Data item schema | `xtuner/v1/datasets/data_item.py` — `DataItem` |

---

## Rule 1 — Data Item Schema (Protocol Contract)

Every layer must produce / consume `DataItem`:
```python
{
    "input_ids":  list[int],   # tokenized token IDs
    "labels":     list[int],   # same length; first token masked to -100
    "num_tokens": int,         # len(input_ids); used for sampler and packing
}
```
- `labels[0] = -100` always (pretrain: first token not predicted)
- `HardPackDataset.__getitem__` returns `list[DataItem]` (one per source sample in pack)
- All other datasets return a single `DataItem`

---

## Rule 2 — PretrainTokenizeFunction

**Input JSONL line formats (either is accepted):**
```json
{"content": "raw text"}
{"messages": [{"role": "pretrain", "content": "raw text"}]}
```

**Hash** = `"{tokenizer_hash}_{source_hash}_{add_bos_token}_{add_eos_token}"`
- Source hash covers `__call__` + `__init__` source code → cache auto-invalidates on code change

---

## Rule 2b — LongTextPretrainTokenizeFunction (chunked tokenization)

**Purpose:** Avoid 5-minute tokenizer blocks on 1M-token documents by splitting at the character level with overlapping windows and LCS-based boundary merging.

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `chunk_size` | 4096 | Target tokens per output chunk |
| `tokenizer_chunk_chars` | 4096 | Characters fed to the tokenizer per call |
| `overlap_chars` | 512 | Overlap window for boundary accuracy |
| `min_chunk_tokens` | 0 | Drop trailing chunk if below this size |

**Two-phase behavior:**

| Phase (`state`) | `__call__` output |
|-----------------|-------------------|
| `"cache"` | `{"num_tokens": int, "chunks": [{"char_start", "char_end", "num_tokens"}]}` — always returned, even for short texts (1 chunk) |
| `"runtime"` | `{"input_ids", "labels", "num_tokens"}` — tokenizes only `text[char_start:char_end]` |

**`shard_char_boundaries(text)` algorithm:**
1. Tokenize `text[start : start + tokenizer_chunk_chars]` (fast tokenizer: with `return_offsets_mapping`; slow tokenizer: approximate char offsets + warning).
2. For non-first chunks: compute overlap token count, reverse-LCS match against previous chunk tail → roll back buffer by `match.a`, skip `overlap_len - match.b` tokens from current chunk head.
3. Accumulate into `buffer_tokens` / `buffer_abs_chars`.
4. Flush one chunk whenever `len(buffer) >= 2 * chunk_size`.
5. Advance pointer by `tokenizer_chunk_chars - overlap_chars`.
6. After document end: append EOS and flush remaining buffer.

**BOS / EOS placement:**
- BOS prepended only when `char_start == 0`.
- EOS appended only when `char_end >= len(text)` (or `char_end is None`).

**Hash** = `"{tokenizer_hash}_{source_hash}_{add_bos_token}_{add_eos_token}_{chunk_size}_{tokenizer_chunk_chars}_{overlap_chars}_{min_chunk_tokens}"`
- Any parameter change → new cache subdirectory.

**Config:**
```python
LongTextPretrainTokenizeFunctionConfig(
    chunk_size=4096,
    tokenizer_chunk_chars=4096,
    overlap_chars=512,
)
```
Drop-in replacement for `PretrainTokenizeFunctionConfig` — no other pipeline changes required.

---

## Rule 3 — JsonlDataset: Shared Memory + Offset Table

- Entire file is loaded into **shared memory** once at init; freed afterwards.
- `offsets: np.ndarray` — byte offsets, shape `(num_raw_lines + 1,)` → O(1) seek to any line.
- `num_tokens: np.ndarray` — token count per raw line; computed by tokenizing (cached or live).
- `sampled: list[int]` — final list of indices after filtering (0-token removed, max_length filter) and `sample_ratio` repetition.
- `__len__` = `len(sampled)`, `__getitem__(i)` → seek to `offsets[sampled[i]]`, read JSON, call `tokenize_fn`.

**Chunked mode (activated when `chunks.npy` exists in the tokenize cache dir):**

After `sample_ratio` is applied, `sampled` is expanded into per-chunk entries:

| Array | Shape | Meaning |
|-------|-------|---------|
| `offsets` | `(total_chunks,)` | Byte offset of the owning line for each chunk entry |
| `num_tokens` | `(total_chunks,)` | Token count of each chunk |
| `sampled` | `list[range(total_chunks)]` | Identity mapping after expansion |
| `_chunk_char_starts` | `(total_chunks,)` | `char_start` for each chunk |
| `_chunk_char_ends` | `(total_chunks,)` | `char_end` for each chunk; `-1` means end-of-line |

`__getitem__(i)` calls `tokenize_fn(raw_data, char_start=cs, char_end=ce)` instead of `tokenize_fn(raw_data)`.

Short lines (1 chunk, `cs=0, ce=-1`) follow the same code path as long lines — no special casing needed.

**`max_length` filter is skipped** when `chunks.npy` exists: filtering by total doc length would incorrectly discard valid long-text samples that have been split.

**`__len__` semantics change:** returns chunk count (not line count) when in chunked mode. Downstream consumers (`HardPackDataset`, `LengthGroupedSampler`, etc.) treat each chunk as an independent `DataItem` — correct behavior since `num_tokens` per entry reflects the chunk size.

---

## Rule 4 — Cache Directory Layout

```
cache_dir/
  .xpuyu-cache-meta.json          # registry: maps file_hash + tokenize_hash → paths
  {file_xxhash}/
    offsets.npy                   # line byte offsets
    {tokenize_fn_hash}/
      num_tokens.npy              # one int per raw line (always present)
      chunks.npy                  # (total_chunks, 4): [line_idx, char_start, char_end, num_tokens]
                                  # only present when tokenize_fn returns "chunks" key (LongText mode)
```

**Cache key triple:** `(file xxhash, tokenizer hash, tokenize_fn source hash)`

Cache is **tag-based**: `cache_tag` on `DatasetConfig` allows look-up by name instead of recomputing the file hash. Rank 0 writes; all ranks sync at barriers.

**`chunks.npy` format:** shape `(total_chunks, 4)`, dtype `int64`.

| Column | Field | Notes |
|--------|-------|-------|
| 0 | `line_idx` | 0-based index into the raw JSONL file |
| 1 | `char_start` | Start character offset within the line's text |
| 2 | `char_end` | End character offset; `-1` means end-of-line (short/single-chunk lines) |
| 3 | `num_tokens` | Token count of this chunk (including BOS/EOS as applicable) |

Short lines (< `chunk_size` tokens) also appear in `chunks.npy` with a single row (`char_start=0, char_end=-1`), making `JsonlDataset` expansion logic uniform.

---

## Rule 5 — HardPackDataset: Static Pack Infos

**Packing algorithm** (run once at init, deterministic from seed):
1. Shuffle all source sample indices.
2. Build cumulative token length array.
3. For each packed slot `[i*L, (i+1)*L)` (L = `pack_max_length`): use `searchsorted` to find which source samples fall in this range → record `(dataset_id, indices[], start_offset, end_offset)`.

**`pack_infos`** is a HuggingFace `Dataset` object:
```python
{
    "dataset_id":    int,        # source dataset index
    "indices":       list[int],  # source sample indices
    "start_offset":  int,        # token offset within first sample
    "end_offset":    int,        # token offset within last sample
    "longest":       int,        # max sequence length in this pack (for sampler)
}
```

**`__getitem__(i)`** → fetches each source `DataItem`, slices `input_ids`/`labels`, asserts `sum(num_tokens) == pack_max_length`. Returns `list[DataItem]`.

**No state** — pack_infos are immutable; resume does not need to checkpoint packing.

---

## Rule 6 — LengthGroupedSampler: Epoch + Step Tracking

**Purpose:** Group samples of similar length into the same mini-batch to reduce padding waste.

**Megabatch logic:**
```python
group_batch_size = mega_batch_mult * global_batch_size   # default ~4x global_batch
# Within each megabatch: sort by max_length DESC → chunk by world_size → shuffle chunks
```

**`__iter__`:**
1. `torch.randperm` all indices (seeded with `seed + epoch`).
2. Form megabatches → sort each by `max_length` DESC → shuffle chunk order.
3. Round up to multiple of `global_batch_size`.
4. Slice for this rank: `indices[step::world_size]` (skip already-consumed `step` items).
5. Reset `self.step = 0` only after full epoch consumed.

**State dict keys:** `{epoch, step, world_size, round_up, num_samples, total_size, group_batch_size, group_size}`
- `step` saved as `consumed_samples % total_size` (wraps within epoch).
- Mismatch of `group_batch_size` / `group_size` on resume logs a warning but continues.

---

## Rule 7 — Resume Contract

```python
# Save
state = dataloader.get_state_dict(consumed_samples)
# → {"sampler": {...}, "dataset": {}}

# Restore
dataloader.load_state_dict(state)
# → sampler.load_state_dict restores epoch + step
# → dataset.load_state_dict is a no-op (packing is deterministic)
```

`consumed_samples` = total **globally-seen samples** (across all ranks). Sampler converts to a per-epoch `step` by modulo.

---

## Rule 8 — Collation: sft_llm_collator

Input: `batch: list[list[DataItem]]` (outer = micro-batch, inner = source items per packed sample)

Processing:
1. Concatenate `input_ids` and `labels` within each packed sample.
2. Shift labels by 1 (auto-regressive: predict next token).
3. Pad `input_ids` → `pack_max_length`; pad `labels` → `-100`.
4. Compute `cu_seq_lens_q / cu_seq_lens_k` for flash-attention variable-length support.

Output:
```python
{
    "seq_ctx": SequenceContext(...),   # contains input_ids + cu_seq_lens
    "shifted_labels": Tensor,          # shape [batch, pack_max_length]
}
```

---

## Rule 9 — Multi-Dataset Support

**`DatasetConfigList`** = `list[DatasetCombine]`，每个 entry = `{dataset: DatasetConfig, tokenize_fn: BaseTokenizeFnConfig}`。

**`build_datasets()` 返回 `list[JsonlDataset]`，映射关系是 1-file → 1-JsonlDataset：**
- 每个 `DatasetConfig.anno_path` 如果是文件 → 1 个 JsonlDataset
- 如果是目录 → 递归找所有 `.jsonl` → 每个文件各一个 JsonlDataset
- 最终返回的 list 长度 = 所有 config entries 中 `.jsonl` 文件总数

**sample_ratio 作用域**：per-DatasetConfig，在 JsonlDataset 内部通过重复 `sampled[]` 实现权重采样。

---

## Rule 10 — HardPackDataset.global_pack 行为差异

| | `global_pack=False`（默认）| `global_pack=True` |
|-|--------------------------|-------------------|
| 打包粒度 | 每个 JsonlDataset 单独打包 | 所有 dataset 先 concat 再统一打包 |
| pack 内样本来源 | 只来自同一个 source dataset | 可跨 dataset 混合 |
| `dataset_id` in pack_infos | 对应原始 dataset list 的 index | 始终为 0 |
| `_LegacySoftPackDataset.datasets` | 保留原始 list | 变为 `[ConcatDataset(all)]` |
| `num_tokens` 传入 | 每个 dataset 的 num_tokens 数组 | 所有 dataset concat 后的单一数组 |

**实现细节**（`_LegacySoftPackDataset.__init__`）：
```python
if global_pack:
    num_tokens = [np.concatenate([dset.num_tokens for dset in datasets])]
    datasets = [ConcatDataset(datasets)]
else:
    num_tokens = [dset.num_tokens for dset in datasets]
```

`HardPackDataset` 与 `ExpandSoftPackDataset` 均继承此逻辑，`get_pack_infos_by_hard_split()` 每次调用处理一个 (dataset_id, num_tokens_array) 对。

---

---

# Implementation Plan: CustomPackDataset + CustomSampler

## Context

现有的 `HardPackDataset` + `LengthGroupedSampler` 自动计算打包方案和采样顺序。新需求是允许用户自行提供 pack 配置（哪些样本打包在一起）和 sampler 配置（pack 以何种顺序消费），以实现对训练数据顺序的完全控制。`PretrainTokenizeFunction` 和 `JsonlDataset` 不变，缓存和 resume 逻辑须保持一致。

---

## 1. 外部配置文件格式

### Pack Config 文件（JSONL 或 NPY）

每个 pack 由若干 **sample slice** 组成，每个 slice 描述取某个样本的哪段 token：

**JSONL 格式**（每行一个 pack）：
```jsonl
{"samples": [[0, 42, 0, 512], [0, 43, 0, 256], [1, 7, 128, 384]]}
{"samples": [[1, 100, 384, 512], [2, 5, 0, 1024], [2, 6, 0, 512]]}
```
- `samples[i] = [dataset_id, sample_idx, token_start, token_end]`
- `dataset_id`：对应 `build_datasets()` 返回的 `list[JsonlDataset]` 的下标
- `sample_idx`：该 dataset 内的逻辑下标（post-filter 的 `sampled[]` 下标）
- `token_start`：从该样本 `input_ids[token_start:]` 开始取（inclusive）
- `token_end`：取到 `input_ids[:token_end]`（exclusive，0 表示取到末尾）
- 同一个样本可以出现在多个不同的 pack 中（截断后分配）

**NPY 格式**（大规模高效）：两个文件：
- `pack_boundaries.npy`：shape `(num_packs+1,)`，CSR 式边界
- `pack_samples.npy`：shape `(total_slices, 4)`，flat list of `[dataset_id, sample_idx, token_start, token_end]`

按文件后缀自动识别格式（`.jsonl` → JSONL；否则尝试 NPY 双文件）。

### Sampler Config 文件

一个有序的 pack 全局消费序列（1D 整数数组）：
- JSONL：`[3, 1, 7, 2, 0, 5, 4, 6]`（单行 JSON 数组）
- NPY：`sampler_order.npy`，shape `(num_steps,)`

`num_steps` 可以 > `num_packs`（允许重复消费某些 pack，例如过采样）；也可以 < `num_packs`（只消费部分）。

---

## 2. 新增文件

| 文件 | 内容 |
|------|------|
| `xtuner/v1/datasets/custom_pack.py` | `CustomPackDataset` + `CustomPackDatasetConfig` |
| `xtuner/v1/datasets/custom_sampler.py` | `CustomSampler` + `CustomSamplerConfig` |

---

## 3. CustomPackDataset

**文件**：`xtuner/v1/datasets/custom_pack.py`

```python
class CustomPackDataset(Dataset):
    def __init__(
        self,
        datasets: list[JsonlDataset],
        pack_config_path: str,
        pack_max_length: int,
        short_pack_strategy: Literal["error", "skip", "padding"] = "error",
        long_pack_strategy: Literal["error", "skip", "truncate"] = "error",
    )
```

**初始化逻辑**：
1. 加载 pack config 文件（JSONL 或 NPY）
2. 构建各 dataset 的 `sample_num_tokens`：
   ```python
   sample_num_tokens[ds_id][s_idx] = datasets[ds_id].num_tokens[datasets[ds_id].sampled[s_idx]]
   ```
3. **验证每个 pack**：
   - `dataset_id` 越界 → 错误
   - `sample_idx` 越界（超过 `len(dataset.sampled)`）→ 错误
   - `token_start < 0` 或 `token_end > num_tokens` 或 `token_start >= token_end`（非 0 情况）→ 错误
   - `sum(token_end - token_start for each slice) < pack_max_length` → 由 `short_pack_strategy` 控制：
     - `"error"`：raise ValueError
     - `"skip"`：跳过该 pack（logger.warning）
     - `"padding"`：在末尾补 pad token，labels 对应位置填 -100
   - `sum(token_end - token_start for each slice) > pack_max_length` → 由 `long_pack_strategy` 控制：
     - `"error"`：raise ValueError
     - `"skip"`：跳过该 pack（logger.warning）
     - `"truncate"`：截断最后一个 slice 使总长度恰好等于 pack_max_length
   - 其余越界类错误（dataset_id、sample_idx、token range 非法）始终 raise ValueError，不可跳过
4. 过滤后的有效 pack_infos 保存为 `self.pack_infos: list[list[tuple[int,int,int,int]]]`

**初始化完成后报告**（`logger.info`）：
```
CustomPackDataset: loaded {num_valid_packs} packs ({num_skipped} skipped).
Total sample coverage: {used_samples}/{total_samples} samples ({pct:.1f}%) across all datasets.
  dataset[0] (name): {used_i}/{total_i} samples ({pct_i:.1f}%)
  dataset[1] (name): ...
```
- `used_samples`：所有有效 pack 中被引用的 `(dataset_id, sample_idx)` 去重集合大小
- `total_samples`：各 dataset 的 `len(sampled)` 总和
- 同一样本被多个 pack 引用时，只计一次（按样本粒度去重，不按 token slice 去重）

**`__len__`**：`len(self.pack_infos)`

**`__getitem__(i)`**：
```python
pack = self.pack_infos[i]   # list of (dataset_id, sample_idx, token_start, token_end)
items = []
for ds_id, s_idx, t_start, t_end in pack:
    item = self.datasets[ds_id][s_idx]   # DataItem: {input_ids, labels, num_tokens}
    t_end = t_end if t_end != 0 else item["num_tokens"]
    sliced = {
        "input_ids":  item["input_ids"][t_start:t_end],
        "labels":     item["labels"][t_start:t_end],
        "num_tokens": t_end - t_start,
    }
    items.append(sliced)
return items   # list[DataItem]，与 HardPackDataset 接口一致
```

**State dict**：`get_state_dict()` → `{}`；`load_state_dict()` → no-op（pack_infos 由文件确定，deterministic）。

---

## 4. CustomSampler

**文件**：`xtuner/v1/datasets/custom_sampler.py`

```python
class CustomSampler(Sampler):
    def __init__(
        self,
        dataset: CustomPackDataset,
        global_order: list[int],      # 加载自文件的全局 pack 消费顺序
        global_batch_size: int,
        dp_mesh: DeviceMesh | None = None,
        seed: int | None = None,
    )
```

**初始化**：
- 加载 sampler config 文件 → `global_order: list[int]`
- 验证：所有值在 `[0, len(dataset))` 范围内；否则 `raise ValueError`
- `world_size = dp_mesh.size()` 或 1
- `local_rank = dp_mesh.get_local_rank()` 或 0
- 对 `global_order` 做 round-up 使其长度为 `global_batch_size * world_size` 的倍数（循环重复末尾元素，与 `ParallelSampler` 一致）
- 每个 rank 的样本：`global_order[local_rank::world_size]`
- `self.epoch = 0`，`self.step = 0`

**初始化完成后报告**（`logger.info`）：
```
CustomSampler: global_order covers {num_used_packs} / {num_total_packs} packs ({pct:.1f}%).
  ({num_repeated} packs referenced more than once)
```
- `num_total_packs` = `len(dataset)`
- `num_used_packs` = `len(set(global_order))`（去重后实际覆盖的 pack 数）
- `num_repeated` = 被引用超过一次的 pack 数量（反映过采样情况）

**`__iter__`**：
```python
local_indices = self._get_local_indices()   # 已 round-up 后切分给本 rank
yield from local_indices[self.step:]        # 跳过已消费的部分
self.step = 0                                # epoch 结束后重置
```

**`__len__`**：`len(self._get_local_indices())`

**State dict**（与现有 sampler 对齐）：
```python
# get_state_dict(consumed_samples: int)
{
    "epoch": self.epoch,
    "step": consumed_samples % total_size,   # 本 epoch 内偏移
    "world_size": world_size,
    "num_samples": len(self),
    "total_size": total_size,
}

# load_state_dict(state)
# 恢复 epoch 和 step；world_size 不匹配时 logger.warning 继续
```

---

## 5. DataloaderConfig 修改

**文件**：`xtuner/v1/datasets/config.py`

新增字段：
```python
class DataloaderConfig(BaseModel):
    ...
    custom_pack_config_path: str | None = None    # 新增
    custom_sampler_config_path: str | None = None  # 新增
```

在 `build()` 中，当 `pack_level == "custom"` 时：
1. 断言 `custom_pack_config_path` 和 `custom_sampler_config_path` 均不为 None
2. 创建 `CustomPackDataset(datasets, custom_pack_config_path, pack_max_length)`
3. 加载 sampler config 文件 → `global_order`
4. 创建 `CustomSampler(packed_dataset, global_order, global_batch_size, dp_mesh)`
5. 其余 collator、dataloader 逻辑不变（`sft_llm_collator` 接口兼容）

---

## 6. Cache 逻辑

`JsonlDataset` 的 cache（`num_tokens.npy`、`offsets.npy`）**不需要改动**。`CustomPackDataset` 自身不做额外 cache：
- pack_infos 直接从用户文件加载，无需计算
- 验证结果不缓存（每次启动重新验证，开销小）

---

## 7. Resume 逻辑

与现有逻辑完全对齐（`xtuner/v1/datasets/resume.py` 无需修改）：

```python
# 保存
state = dataloader.get_state_dict(consumed_samples)
# → {"sampler": {epoch, step, ...}, "dataset": {}}

# 恢复
dataloader.load_state_dict(state)
# → CustomSampler.load_state_dict 恢复 epoch+step
# → CustomPackDataset.load_state_dict 是 no-op
```

---

## 8. 修改文件清单

| 文件 | 类型 | 改动 |
|------|------|------|
| `xtuner/v1/datasets/custom_pack.py` | **新建** | `CustomPackDataset` + `CustomPackDatasetConfig` |
| `xtuner/v1/datasets/custom_sampler.py` | **新建** | `CustomSampler` + `CustomSamplerConfig` |
| `xtuner/v1/datasets/config.py` | **修改** | `DataloaderConfig` 新增 2 字段；`build()` 支持 `pack_level="custom"` |
| `xtuner/v1/datasets/__init__.py` | **修改** | 导出新类 |

---

## 9. 验证方案

1. 准备一个小 JSONL 数据集（`tests/resource/pretrain_example_data.jsonl`）
2. 手动构造 pack config 文件（token 数精确等于 `pack_max_length`）和 sampler config 文件
3. 运行 `torchrun xtuner/v1/train/cli/sft.py <config>` 完成至少 10 步
4. 保存 checkpoint，恢复后验证 `consumed_samples` 和 `step` 正确
5. 构造一个含无效 pack（token 数不匹配）的 config，验证 warn+skip 行为

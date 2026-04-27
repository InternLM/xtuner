---
paths:
  - "xtuner/v1/datasets/**/*"
---

# XTuner v1 Datasets Architecture Rules

## Core Data Contract (DataItem Protocol)

Every dataset layer must produce/consume `DataItem`:
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

## Tokenization Functions

**Standard pretrain**: `PretrainTokenizeFunction`
- Accepts: `{"content": "text"}` or `{"messages": [{"role": "pretrain", "content": "text"}]}`
- Hash: `"{tokenizer_hash}_{source_hash}_{add_bos_token}_{add_eos_token}"`

**Long text pretrain**: `LongTextPretrainTokenizeFunction`
- Two-phase: cache phase returns chunk metadata, runtime phase tokenizes specific char ranges
- Parameters: `chunk_size`, `tokenizer_chunk_chars`, `overlap_chars`, `min_chunk_tokens`
- BOS only at `char_start==0`, EOS only at document end
- Hash includes all chunking parameters

## JsonlDataset Architecture

- **Shared memory**: entire file loaded once at init
- **Offset table**: `np.ndarray` for O(1) line access
- **Chunked mode**: activated when `chunks.npy` exists in cache
  - Expands `sampled` to per-chunk entries
  - `__getitem__` calls `tokenize_fn(data, char_start=cs, char_end=ce)`
  - `max_length` filter skipped (would incorrectly discard valid long-text samples)
  - `__len__` returns chunk count (not line count)

## Cache Directory Layout

```
cache_dir/
  .xpuyu-cache-meta.json
  {file_xxhash}/
    offsets.npy
    {tokenize_fn_hash}/
      num_tokens.npy              # always present
      chunks.npy                  # only for LongText mode: (total_chunks, 4)
                                  # [line_idx, char_start, char_end, num_tokens]
```

Cache key: `(file xxhash, tokenizer hash, tokenize_fn source hash)`
- Tag-based lookup via `cache_tag` on `DatasetConfig`
- Source hash auto-invalidates on code change

## Packing Strategies

**HardPackDataset**:
- Static pack_infos computed at init (deterministic from seed)
- Returns `list[DataItem]` per pack
- `global_pack=False`: per-dataset packing (default)
- `global_pack=True`: concat all datasets then pack

**PresetPackDataset**:
- User-provided pack config (NPY directory + `paths.json`)
- Validates structure and token ranges via vectorized numpy
- Strategies for length mismatch: error/padding (short), error/truncate (long)
- No state (deterministic from config file)

## Samplers

**LengthGroupedSampler**:
- Megabatch logic: group similar lengths, sort DESC, shuffle chunks
- State: `{epoch, step, world_size, num_samples, total_size}`
- `step` = `consumed_samples % total_size`

**PresetSampler**:
- User-provided order via `sampler_config_path` (`.npy` path only, mmap read)
- Round-down to `global_batch_size * world_size` multiple (truncate tail; error if length too short)
- Per-rank slice: `effective[local_rank::world_size]` on the truncated mmap-backed view (`self.global_order`)
- `pack_level="preset"` in `DataloaderConfig`; requires `pack_config_path` and `sampler_config_path`
- State dict compatible with existing resume logic

## Resume Contract

```python
# Save
state = dataloader.get_state_dict(consumed_samples)
# → {"sampler": {...}, "dataset": {}}

# Restore
dataloader.load_state_dict(state)
```

- `consumed_samples` = total globally-seen samples (across all ranks)
- Sampler converts to per-epoch `step` by modulo
- Packing datasets have no state (deterministic)

## Multi-Dataset Support

- `DatasetConfigList` = list of `{dataset, tokenize_fn}` pairs
- `build_datasets()` returns `list[JsonlDataset]` (1 file → 1 dataset)
- `sample_ratio` applied per-DatasetConfig via `sampled[]` repetition

## Key Implementation Files

| File | Purpose |
|------|---------|
| `config.py` | Pipeline orchestration via `DataloaderConfig.build()` |
| `pt_tokenize_fn/text.py` | `PretrainTokenizeFunction` |
| `pt_tokenize_fn/long_text.py` | `LongTextPretrainTokenizeFunction` |
| `jsonl.py` | `JsonlDataset` with shared memory + offset table |
| `packing.py` | `HardPackDataset`, `ExpandSoftPackDataset` |
| `sampler.py` | `LengthGroupedSampler`, `ParallelSampler` |
| `preset_pack.py` | `PresetPackDataset` |
| `preset_sampler.py` | `PresetSampler` |
| `resume.py` | `get_dataloader_state / load_dataloader_state` |
| `dataloader.py` | `Dataloader` wrapper |
| `data_item.py` | `DataItem` schema |

## Design Principles

1. **Protocol-based**: All layers communicate via `DataItem` contract
2. **Cache-aware**: Tokenization cached with automatic invalidation
3. **Deterministic**: Packing and sampling reproducible from seed/config
4. **Resumable**: State dict pattern for checkpoint/resume
5. **Composable**: Tokenize → Dataset → Pack → Sample → Collate pipeline

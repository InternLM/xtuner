# 设计文档：MTP 序列并行（SP）支持

## 背景

`roll_sequence_context`（`xtuner/v1/module/mtp/utils.py`）目前对 SP 直接 assert 不支持：

```python
assert seq_ctx.sequence_parallel_mesh is None, "Sequence parallel is not yet supported"
```

`MTPBlock.forward()` 被调用时，`seq_ctx` 已经是 SP-split 后的状态（在 trainer 的 `_prepare_model_input` 里做的），因此需要处理 SP 下 `roll_sequence_context` 的正确性。

---

## 关键前提

通过阅读 `SequenceContext.split()` 确认：

| 字段 | SP split 后的状态 |
|---|---|
| `input_ids` | **已 split**，每个 rank 持有 local shard |
| `cu_seq_lens_q/k` | **未 split**，始终是全局 sequence boundaries |
| `inputs_embeds` | SP-split 的 local shard（由 `embed_tokens(sp_split_input_ids)` 产生）|
| `position_ids` | 已 split |

两条路径的核心问题一致：local shard 末尾若处于某条 sequence 中间，roll 后该位置应为下一个 token 的值，而非 fill_value。

---

## 统一方案：全量 roll + 取 slice，通信各做一次

两条路径都采用相同的思路：**先获得完整 tensor，用全局 `cu_seq_lens` 做 roll，再取本 rank 对应的 slice**。

### Case 1：`input_ids` 路径（零通信）

split 前将完整 `input_ids` 存为 `raw_input_ids`，是 int64，内存代价小。

`roll_sequence_context` 时：
1. 对 `raw_input_ids` + 全局 `cu_seq_lens_q` 做 `roll_packed_tensor(shifts=cumulative_shift)`
2. 取本 rank 的 slice：`rolled[..., shard_start : shard_start + shard_size]`

无需任何跨 rank 通信。

### Case 2：`inputs_embeds` 路径（一次 allgather）

`inputs_embeds` 是 SP-split 的 local shard（`float16/bfloat16`），内存代价大，不适合在 split 时存全量。

在 `MTPBlock.forward()` 入口做**一次 allgather**，得到完整 `inputs_embeds`：
1. `full_embeds = allgather(local_inputs_embeds, sp_group)`（沿 seq 维度拼接）
2. 之后所有 D 层复用 `full_embeds`，各层用 `shifts=-(layer_idx+1)` 从全量 roll，取 slice

**通信次数**：1 次 allgather，与 MTP 层数 D 无关。相比 P2P 方案（D 次通信，latency 线性累积）更优。

---

## 多层 MTP：保持 relative shift(-1)

两条路径在入口处都已经拿到全量 tensor（`raw_input_ids` / allgather 后的 `full_embeds`），之后每层 roll(-1) 作用在上一层已 rolled 的全量结果上，与非 SP 情况完全等价，**`MTPBlock.forward()` 的循环逻辑不需要改动**。

累积 shift 的动机是"在 local shard 上叠加 roll 会出错"，但有了全量 tensor 这个前提就不成立了，relative shift 更自然。

---

## 变更清单

| 文件 | 操作 |
|---|---|
| `xtuner/v1/data_proto/sequence_context.py` | `split()` 新增 `raw_input_ids`（split 前赋值）、`shard_start`、`shard_size` 字段；`__init__` 及 `copy()` 透传 |
| `xtuner/v1/module/mtp/utils.py` | `roll_sequence_context()` 移除 SP assert；`input_ids` 路径用 `raw_input_ids` 全量 roll 取 slice；`inputs_embeds` 路径接收 allgather 后的全量 embedding 做同样处理 |
| `xtuner/v1/module/mtp/mtp_block.py` | `forward()` 在入口对 `inputs_embeds` 做一次 allgather（SP 模式下）；循环逻辑不变 |

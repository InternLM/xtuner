以下是 EP + TP 同时开启时，`MoELayer.forward` 调用 `MoEAlltoAllTokenDispatcher` 的完整流程。

---

## 前置形状约定

| 符号             | 含义                                               |
| ---------------- | -------------------------------------------------- |
| `S/TP * B`       | 每个设备持有的 local tokens（SP 下序列按 TP 切分） |
| `H`              | hidden size（专家计算不按 TP 切分 H 维）           |
| `E`              | 总专家数                                           |
| `E_local = E/EP` | 每个 EP rank 持有的本地专家数                      |

输入：`hidden_states [S/TP, B, H]`，每个设备只持有序列的 `1/TP` 片段。

---

## token_permutation 流程

### 1. `preprocess(routing_map)`

在 `tp_ep_group`（TP × EP 域）上做一次 AllGather，收集全局的 `num_tokens → expert` 分布，计算：

- `input_splits [EP]`：本 rank 要向各 EP rank 发送多少 token
- `output_splits [EP]`：本 rank 将从各 EP rank 收到多少 token（仅计我的 TP 切片）
- `output_splits_tp [TP]`：EP A2A 后，各 TP rank 各持有多少 token（用于后续 AllGather 的不等分）
- `num_global_tokens_per_local_expert_cpu`：每个本地专家将处理多少 token（用于 sort_chunks）

---

### 2. Permutation 1：按专家排序（本地）

```
hidden_states [N_local, H]
    → permute(routing_map)
    → permutated_local_input_tokens [num_out_tokens, H]
```

将本地 token 按 **目标 EP rank → 目标专家** 的顺序排列，为 EP A2A 的连续内存布局做准备。同时保存逆映射 `reversed_local_input_permutation_mapping`。

---

### 3. EP AlltoAll（第一次 A2A）

```
all_to_all(ep_group,
           send=permutated_local_input_tokens,
           output_splits=output_splits,   # 我将收到多少
           input_splits=input_splits)     # 我将发出多少
→ global_input_tokens [M_ep_recv, H]
```

每个 EP rank 收到来自所有 EP rank 的、目标是本 rank 本地专家的 token，但**仍只是每个 EP rank 的 TP 切片**（即来自同一 EP rank 不同 TP rank 的 token 还未合并）。

---

### 4. TP AllGather（补全序列切片）

```python
if self.tp_size > 1:
    global_input_tokens = gather_from_sequence_parallel_region(
        global_input_tokens, group=tp_group,
        output_split_sizes=output_splits_tp.tolist()
    )
→ global_input_tokens [M_total, H]
```

在 TP 组内 AllGather，把同一 EP rank 下不同 TP rank 持有的 token 片段拼合。之后每个设备（同一 EP rank 内的所有 TP rank）都持有完整的、需要送入本地专家的 token 集合。

---

### 5. Permutation 2：按本地专家排序（为 Grouped GEMM）

```python
if self.num_local_experts > 1:
    global_input_tokens = sort_chunks_by_idxs(
        global_input_tokens,
        num_global_tokens_per_local_expert_cpu.ravel(),
        sort_input_by_local_experts
    )
→ dispatched_input [M_total, H]，按 local expert 连续分组
```

AllGather 后的顺序是 `[TP rank 0 的 block | TP rank 1 的 block | ...]`，每块内部已按本地专家排序，但整体不连续。这里做一次 sort_chunks 让同一专家的 token 在内存中连续，满足 Grouped GEMM 的输入要求。

---

## 专家计算

```
experts(dispatched_input, tokens_per_expert)
→ expert_output [M_total, H]
```

每个 EP rank 用 Grouped GEMM 计算本地 `E_local` 个专家，各 TP rank 计算相同的数据（专家权重本身不按 TP 切分 H 维，是完整权重的副本）。

---

## token_unpermutation 流程（逆序）

### 6. Unpermutation 2：逆 sort_chunks

```python
if self.num_local_experts > 1:
    hidden_states = sort_chunks_by_idxs(
        hidden_states,
        num_global_tokens_per_local_expert_cpu.T.ravel(),
        restore_output_by_local_experts
    )
→ [M_total, H]，恢复为 [TP rank 0 block | TP rank 1 block | ...] 顺序
```

---

### 7. TP ReduceScatter

```python
if self.tp_size > 1:
    hidden_states = reduce_scatter_to_sequence_parallel_region(
        hidden_states, group=tp_group,
        input_split_sizes=output_splits_tp.tolist()
    )
→ [M_ep_recv, H]
```

对专家输出在 TP 组内做 ReduceScatter：各 TP rank 持有相同的专家输出，reduce（求和）后 scatter，每个 TP rank 只保留属于自己的 token 片段。

---

### 8. EP AlltoAll（第二次 A2A，逆向）

```python
all_to_all(ep_group,
           send=hidden_states,
           output_splits=input_splits,    # 逆向：原来发多少现在收多少
           input_splits=output_splits)
→ permutated_local_input_tokens [num_out_tokens, H]
```

将专家输出发回各 source EP rank，每个 rank 收回自己原来发出的 token 的专家输出。

---

### 9. Unpermutation 1：还原 token 顺序 + topK 加权求和

```python
output = unpermute(
    permutated_local_input_tokens,
    reversed_local_input_permutation_mapping,
    restore_shape=hidden_shape_before_permute,
    probs=self.probs,
    routing_map=self.routing_map
)
→ output [N_local, H]
```

用 Permutation 1 保存的逆映射，将 token 还原到原始顺序，并对 topK 个专家的输出按 `probs` 加权求和。

最终 `reshape` 回 `[S/TP, B, H]`。

---

## 整体数据流一览

```
[S/TP, B, H]
  │
  ▼ Permutation 1（按 EP rank/expert 排序）
[num_out_tokens, H]
  │
  ▼ EP AlltoAll → 各 EP rank 收到目标 token（仍是 TP 切片）
[M_ep_recv, H]
  │
  ▼ TP AllGather → 补全序列切片，每 TP rank 数据一致
[M_total, H]
  │
  ▼ Permutation 2（按 local expert 连续分组）
[M_total, H]
  │
  ▼ Grouped GEMM（E_local 个专家）
[M_total, H]
  │
  ▼ Unpermutation 2（逆 sort_chunks）
[M_total, H]
  │
  ▼ TP ReduceScatter → 各 TP rank 只保留自己的片段
[M_ep_recv, H]
  │
  ▼ EP AlltoAll（逆向）→ token 回到 source rank
[num_out_tokens, H]
  │
  ▼ Unpermutation 1 + topK 加权求和
[S/TP, B, H]
```

---

## 关键设计要点

| 通信             | 作用                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| EP A2A（正向）   | 将 token 路由到持有目标专家的 EP rank                                                                   |
| TP AllGather     | 每个 EP rank 内合并 TP 切片，得到完整待计算 token 集；各 TP rank 计算完全相同的专家输出（**冗余计算**） |
| TP ReduceScatter | 对冗余输出 reduce，并按 SP 切分还给各 TP rank                                                           |
| EP A2A（逆向）   | 将专家输出归还 source rank                                                                              |

TP 维度的 AllGather + ReduceScatter 是对称的，引入冗余计算但避免了对专家权重做 TP 切分，保持专家计算的完整性。EP 维度的两次 A2A 实现了 token 到专家的路由与归还。
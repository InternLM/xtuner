# MoEDecoderLayer._forward 中 TorchAll2AllDispatcher 的 EP 流程

下面用一个缩小版一致例子，把 `MoEDecoderLayer._forward` 里的 EP all2all 流程从头串起来。真实 Qwen3MoE30BA3 是 `E=128, K=8, EP=4`；示例改成：

```text
EP = 2
E_local = 3
E = 6
K = 2
每个 EP rank 本地 N = B*S = 4 个 token
```

专家归属：

```text
ep0 owns global expert 0,1,2  -> local expert 0,1,2
ep1 owns global expert 3,4,5  -> local expert 0,1,2
```

示例 token：

```text
ep0 source tokens: A0 A1 A2 A3
ep1 source tokens: B0 B1 B2 B3
```

为方便阅读，下面主要跟踪 activation 行的来源，不展开 `D_h` 维。

## 0. `_pre_moe_forward` 后

对任意一个 EP rank，本地输入：

```text
hidden_states: [N, D_h] = [4, D_h]
logits:        [N, E]   = [4, 6]
topk_ids:      [N, K]   = [4, 2]
topk_weights:  [N, K]   = [4, 2]
```

设两个 source rank 的 routing 结果如下：

```text
ep0 topk_ids:
A0 -> [0, 4]
A1 -> [3, 1]
A2 -> [2, 5]
A3 -> [4, 0]

ep1 topk_ids:
B0 -> [1, 3]
B1 -> [4, 2]
B2 -> [5, 0]
B3 -> [3, 1]
```

## 1. `dispatch_preprocess`: 本地 token 按 global expert 排序

先把每个 token 复制 `K=2` 份，所以每个 source rank 都从 `[4, D_h]` 变成 `[8, D_h]`。

`grouped_gemm.backend.permute` 内部使用 **topk-slot-first** 展开：先列出所有 N 个 token 的
第 0 号 topk copy，再列出第 1 号 topk copy，依此类推。`row_id_map[i] = j` 表示源 flat 空间
（topk-slot-first）第 `i` 个位置的 token copy 排序后落在第 `j` 个位置（scatter 语义）；
同 expert 时按 token index 升序排列。

对 `ep0`，flatten 后的 copy 是：

```text
flat pos:          0   1   2   3   4   5   6   7
token copy:        A0  A1  A2  A3  A0  A1  A2  A3
global expert id:  0   3   2   4   4   1   5   0
topk slot:         0   0   0   0   1   1   1   1
```

按 `(expert, token index)` 排序后：

```text
pre row:           0   1   2   3   4   5   6   7
token copy:        A0  A3  A1  A2  A1  A0  A3  A2
global expert id:  0   0   1   2   3   4   4   5
row_id_map:        0   4   3   6   5   2   7   1
```

将上面两组放到一起看`row_id_map`映射关系

```text
flat pos:          0   1   2   3   4   5   6   7
token copy:        A0  A1  A2  A3  A0  A1  A2  A3
row_id_map:        0   4   3   6   5   2   7   1

pre row:           0   1   2   3   4   5   6   7
token copy:        A0  A3  A1  A2  A1  A0  A3  A2
global expert id:  0   0   1   2   3   4   4   5
```



所以：

```text
pre_dispatched[“hidden_states”]: [N*K, D_h] = [8, D_h]
pre_dispatched[“row_id_map”]:    [N*K]      = [8]
```

`backend.unpermute(combined, row_id_map, probs)` 对应的逆操作是 gather：
`output[i] = combined[row_id_map[i]]`，输出按 topk-slot-first 排布后乘以 `probs` 再沿 K 方向求和。

对 `ep1` 同理：

```text
flat pos:          0   1   2   3   4   5   6   7
token copy:        B0  B1  B2  B3  B0  B1  B2  B3
global expert id:  1   4   5   3   3   2   0   1
topk slot:         0   0   0   0   1   1   1   1

pre row:           0   1   2   3   4   5   6   7
token copy:        B2  B0  B3  B1  B0  B3  B1  B2
global expert id:  0   1   1   2   3   3   4   5
row_id_map:        1   6   7   5   4   3   0   2
```

## 2. `dispatch`: 第一次 all2all

每个 source rank 根据 global expert 所属 EP rank 切分。

`ep0` 的 pre rows：

```text
pre row:           0  1  2  3 | 4  5  6  7
token copy:        A0 A3 A1 A2| A1 A0 A3 A2
global expert id:  0  0  1  2 | 3  4  4  5
target ep rank:    0  0  0  0 | 1  1  1  1
```

所以：

```text
ep0 input_splits = [4, 4]
```

`ep1` 的 pre rows：

```text
pre row:           0  1  2  3 | 4  5  6  7
token copy:        B2 B0 B3 B1| B0 B3 B1 B2
global expert id:  0  1  1  2 | 3  3  4  5
target ep rank:    0  0  0  0 | 1  1  1  1
```

所以：

```text
ep1 input_splits = [4, 4]
```

all2all 后，`ep0` 收到所有发给 experts `0,1,2` 的 token copy：

```text
dispatched row:    0  1  2  3 | 4  5  6  7
source ep rank:    0  0  0  0 | 1  1  1  1
token copy:        A0 A3 A1 A2| B2 B0 B3 B1
global expert id:  0  0  1  2 | 0  1  1  2
local expert id:   0  0  1  2 | 0  1  1  2
```

`ep1` 收到所有发给 experts `3,4,5` 的 token copy：

```text
dispatched row:    0  1  2  3 | 4  5  6  7
source ep rank:    0  0  0  0 | 1  1  1  1
token copy:        A1 A0 A3 A2| B0 B3 B1 B2
global expert id:  3  4  4  5 | 3  3  4  5
local expert id:   0  1  1  2 | 0  0  1  2
```

形状：

```text
dispatched["hidden_states"]:              [M_recv, D_h]
dispatched["tokens_per_expert_group"]:    [EP, E_local] = [2, 3]
```

在这个例子里两个 rank 都是 `M_recv=8`，但真实训练里不保证均匀。

### 2.1 变长 all2all 的 host metadata 同步

上面的 `input_splits` / `output_splits` 在真实 `TorchAll2AllDispatcher` 中不是纯 GPU metadata。
当前实现会先在 GPU 上统计和交换每个 expert 的 token 数，然后把 split sizes 拉回 CPU：

```python
tokens_per_expert = torch.histc(topk_ids, bins=n_routed_experts, min=0, max=n_routed_experts)
dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=process_group)

input_splits = (
    tokens_per_expert.reshape(ep_size, num_experts_per_rank)
    .to(device=torch.device("cpu"))
    .sum(dim=1)
    .tolist()
)
output_splits = tokens_per_expert_group.to(device=torch.device("cpu")).sum(dim=-1).tolist()
```

这一步会形成 CPU/host 同步点，因为 PyTorch 变长 `all_to_all_single` 需要 Python `list[int]` 形式的
`input_split_sizes` / `output_split_sizes`。也就是说，EP-only 的 `async_op=True` 并不是“完全无 host 同步”：

- 大块 hidden 的 EP all2all 会被放到 dispatcher 的通信流中，并由 CUDA event 串依赖。
- 但在真正发起大块 hidden all2all 之前，host 需要等 token count 交换完成并拿到 split list。
- `combine` 会复用 dispatch 阶段保存的 `input_splits` / `output_splits`，通常不会再新增同类 split-size 同步。

这个细节对 Domino EP 的计算通信重叠很重要。host 等 split list 时，已经 enqueue 到 GPU 的另一个 micro batch
计算仍然可以继续执行；但 host 不能继续 enqueue 后续的 `dispatch_postprocess -> expert -> combine_preprocess`
或下一个 dispatch。如果 split-size 同步能被另一个 micro batch 的 attention/gate/pre-dispatch 覆盖，7.3 中的
流水基本成立；如果同步时间更长，就会吃掉一部分甚至全部重叠窗口。

## 3. `dispatch_postprocess`: destination rank 内按 local expert 再排序

all2all 后的顺序是：

```text
source ep0 block | source ep1 block
```

并且每个 source 块内部已经按当前 destination rank 的 local expert id 排好。但 grouped GEMM 要的是整个 `M_recv` 范围内按 local expert 连续分组，所以还要再 permute 一次。

对 `ep0`：

```text
dispatch 后:
dispatched row:    0  1  2  3 | 4  5  6  7
source ep rank:    0  0  0  0 | 1  1  1  1
token copy:        A0 A3 A1 A2| B2 B0 B3 B1
local expert id:   0  0  1  2 | 0  1  1  2
```

按 local expert id 全局排序后：

```text
post row:          0  1  2 | 3  4  5 | 6  7
token copy:        A0 A3 B2| A1 B0 B3| A2 B1
local expert id:   0  0  0 | 1  1  1 | 2  2
row_ids_map:       0  1  3 | 6  2  4 | 5  7
```

所以：

```text
post_dispatched["hidden_states"]:     [8, D_h]
post_dispatched["row_ids_map"]:       [8]
post_dispatched["tokens_per_expert"]: [3] = [3, 3, 2]
```

对 `ep1`：

```text
dispatch 后:
dispatched row:    0  1  2  3 | 4  5  6  7
source ep rank:    0  0  0  0 | 1  1  1  1
token copy:        A1 A0 A3 A2| B0 B3 B1 B2
local expert id:   0  1  1  2 | 0  0  1  2
```

按 local expert id 全局排序后：

```text
post row:          0  1  2 | 3  4  5 | 6  7
token copy:        A1 B0 B3| A0 A3 B1| A2 B2
local expert id:   0  0  0 | 1  1  1 | 2  2
row_ids_map:       0  3  4 | 6  1  2 | 5  7
```

形状仍然：

```text
post_dispatched["hidden_states"]:     [8, D_h]
post_dispatched["tokens_per_expert"]: [3] = [3, 3, 2]
```

## 4. local experts grouped GEMM

每个 EP rank 只计算自己本地 3 个 experts。

对 `ep0`，grouped GEMM 分段是：

```text
post row:          0  1  2 | 3  4  5 | 6  7
token copy:        A0 A3 B2| A1 B0 B3| A2 B1
local expert id:   0  0  0 | 1  1  1 | 2  2
tokens_per_expert: 3        | 3        | 2
```

输出：

```text
experts_out: [M_recv, D_h] = [8, D_h]
```

`ep1` 也是同理：

```text
post row:          0  1  2 | 3  4  5 | 6  7
token copy:        A1 B0 B3| A0 A3 B1| A2 B2
local expert id:   0  0  0 | 1  1  1 | 2  2
tokens_per_expert: 3        | 3        | 2
```

## 5. `combine_preprocess`: 恢复 all2all receive 顺序

专家输出现在是 local expert grouped 顺序，必须先恢复成 dispatch 后的 source-block 顺序，才能反向 all2all。

对 `ep0`，用：

```text
row_ids_map = [0, 1, 3, 6, 2, 4, 5, 7]
```

做 `unpermute(experts_out, row_ids_map)` 后：

```text
pre_combined row:  0  1  2  3 | 4  5  6  7
source ep rank:    0  0  0  0 | 1  1  1  1
token copy:        A0 A3 A1 A2| B2 B0 B3 B1
local expert id:   0  0  1  2 | 0  1  1  2
```

形状：

```text
pre_combined["hidden_states"]: [M_recv, D_h] = [8, D_h]
```

## 6. `combine`: 第二次 all2all，把 expert 输出送回 source rank

`combine` 用的是第一次 dispatch 的反向 split：

```text
input_split_sizes  = dispatched["output_splits"]
output_split_sizes = dispatched["input_splits"]
```

这里没有重新统计 token，也不会再把新的 split tensor 拉回 CPU；它依赖第一次 dispatch 已经确定的
source/destination 分片关系。因此对于 `TorchAll2AllDispatcher`，前向中最主要的 host metadata 同步点在第一次
dispatch，而不是 combine。

对 source `ep0` 来说，它会收回自己原来发出去的 8 个 token copy 输出：

```text
combined row on source ep0: 0  1  2  3 | 4  5  6  7
from dest ep rank:          0  0  0  0 | 1  1  1  1
token copy:                 A0 A3 A1 A2| A1 A0 A3 A2
global expert id:           0  0  1  2 | 3  4  4  5
```

这个顺序正好对应 `ep0 dispatch_preprocess` 后的 sorted order。

形状：

```text
combined["hidden_states"]: [N*K, D_h] = [8, D_h]
```

## 7. `combine_postprocess`: 用第一次 `row_id_map` 加权合并 topK

回到 source `ep0` 后，用最开始的：

```text
pre_dispatched["row_id_map"] = [0, 4, 3, 6, 5, 2, 7, 1]
topk_weights:                [N, K] = [4, 2]
```

把 sorted expert output 加权合并回原始 token 空间。概念上等价于先按原始 topK copy 分组：

```text
combined row:      0  1  2  3  4  5  6  7
token copy:        A0 A3 A1 A2 A1 A0 A3 A2
conceptual group:  A0 A0 | A1 A1 | A2 A2 | A3 A3
topk slot:         0  1  | 0  1  | 0  1  | 0  1
```

然后 reshape：

```text
[N*K, D_h] -> [N, K, D_h] = [4, 2, D_h]
```

乘 `topk_weights [4, 2]` 并对 `K` 求和：

```text
A0 final = out(A0,e0) * w(A0,e0) + out(A0,e4) * w(A0,e4)
A1 final = out(A1,e3) * w(A1,e3) + out(A1,e1) * w(A1,e1)
A2 final = out(A2,e2) * w(A2,e2) + out(A2,e5) * w(A2,e5)
A3 final = out(A3,e4) * w(A3,e4) + out(A3,e0) * w(A3,e0)
```

形状：

```text
post_combined["hidden_states"]: [N, D_h] = [4, D_h]
```

最后恢复原始 batch/seq：

```text
combined_hidden_states: [B, S, D_h]
```

## 8. `_post_moe_forward`

前提是 `n_shared_experts=0`，所以没有 shared expert 分支：

```text
hidden_states = combined_hidden_states * hidden_factor + residual
```

输出：

```text
hidden_states:  [B, S, D_h]
router_logits:  [N, E]
router_weights: [N, E]
```

## 核心总结

第一次 `row_id_map [N*K]` 是 source rank 上 `permute` 产生、最后由 `unpermute(..., probs=topk_weights)`
消费的还原 map，负责加权合并回 `[N, D_h]`。其精确语义：

- **scatter**：`row_id_map[i] = j` 表示 topk-slot-first 源 flat 空间第 `i` 个位置的 token copy
  排序后落在 sorted 空间第 `j` 个位置。
- **unpermute 逆操作**：gather，`output[i] = combined[row_id_map[i]]`，输出按 topk-slot-first
  排布后乘 `probs` 再沿 K 求和，得到 `[N, D_h]`。
- `grouped_gemm.backend.permute` 内部使用 topk-slot-first 展开，同 expert 时按 token index 升序；
  手动从 token-first flat 展开推导会得到不同的值，两者不可混用。

第二次 `post_dispatched["row_ids_map"] [M_recv]` 是 destination EP rank 上第二次 `permute` 产生的还原 map，
语义相同（scatter，1D indices 无 topk 展开），只负责 expert 计算后恢复 source-block 顺序，方便反向 all2all。

## DeepEP dispatcher 的对应差异

`DeepEPDispatcher` 使用 DeepEP 的 `Buffer.get_dispatch_layout()` / `Buffer.dispatch()` / `Buffer.combine()` 来管理
layout、通信 handle 和事件。它不像 `TorchAll2AllDispatcher` 那样显式执行：

```python
to(device=torch.device("cpu")).tolist()
```

但它仍然存在 host 可见的 metadata 准备点。`xtuner/v1/ops/comm/deepep_op.py::dispatch_forward()` 中已经注明：

```python
# NOTES: the CPU will wait for GPU's signal to arrive,
# so this is not compatible with CUDA graph
```

DeepEP dispatch 会返回：

```python
num_recv_tokens_per_expert_list, handle, event
```

其中 `num_recv_tokens_per_expert_list` 是 Python list，`dispatch_postprocess` 需要用它计算 `num_out_tokens` 和
`tokens_per_expert`。因此 DeepEP 也不是完全没有 host 同步；只是同步被 DeepEP 的 layout/dispatch handle 机制封装
在库内部，不是 PyTorch split-size list 的 `.tolist()` 同步。

对 Domino EP 来说，两者的影响边界一致：

- 已经 enqueue 到 GPU 的另一个 micro batch 计算不会被 host 同步打断。
- host 等 metadata 时无法继续 enqueue 后续本地算子和通信。
- 如果 metadata 等待短于可覆盖的另一个 micro batch 计算，重叠效果基本保留。
- 如果 metadata 等待更长，`xtuner_ep_domino.md` 7.3 中的理想时间线会被压缩，真实重叠比例下降。

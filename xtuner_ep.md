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

对 `ep0`，flatten 后的 copy 是：

```text
flat row:          0   1   2   3   4   5   6   7
token copy:        A0  A0  A1  A1  A2  A2  A3  A3
global expert id:  0   4   3   1   2   5   4   0
```

按 global expert id 稳定排序后：

```text
pre row:           0   1   2   3   4   5   6   7
token copy:        A0  A3  A1  A2  A1  A0  A3  A2
global expert id:  0   0   1   2   3   4   4   5
row_id_map:        0   4   3   6   5   2   7   1
```

所以：

```text
pre_dispatched["hidden_states"]: [N*K, D_h] = [8, D_h]
pre_dispatched["row_id_map"]:    [N*K]      = [8]
```

这里的 `row_id_map` 是 `permute` 返回、后续 `unpermute` 消费的还原 map。当前 `grouped_gemm`
backend 下它不是简单的 “pre row j 对应原始 topK flatten 空间里的哪个位置”，不要把它当成普通
`index_put` 的下标表来手算。

对 `ep1` 同理：

```text
flat row:          0   1   2   3   4   5   6   7
token copy:        B0  B0  B1  B1  B2  B2  B3  B3
global expert id:  1   3   4   2   5   0   3   1

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
消费的还原 map，负责加权合并回 `[N, D_h]`。

第二次 `post_dispatched["row_ids_map"] [M_recv]` 是 destination EP rank 上第二次 `permute` 产生的还原 map，
只负责 expert 计算后恢复 source-block 顺序，方便反向 all2all。两个 map 都应当按 backend opaque map 理解，
不要按普通排序下标手算。

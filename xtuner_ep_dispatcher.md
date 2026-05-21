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

## TP+EP 中 ReduceScatterRowsSum 与 padding/capacity 取舍

`TorchAll2AllTPEPDispatcher` 在 EP dispatch 之后会额外做 TP AllGather，在 combine 阶段会做 TP
ReduceScatterRowsSum。这里的 **TP ReduceScatterRowsSum** 是语义名：对同一 TP group 中完整 token 批的 hidden 做
SUM 归约，并只保留当前 TP rank 负责的 token slice。它同时出现在两个方向：

- combine forward：row-parallel expert output 先做 TP ReduceScatterRowsSum，再进入 EP combine all2all。
- TP AllGather backward：AllGather 的反向也是 TP ReduceScatterRowsSum。

TP+EP MoE routing 后，同一个 EP rank 上的不同 TP rank 不一定收到相同数量的 token。以 `tp_size=2` 为例：

```text
EP dispatch 后：
  TP rank0 hidden: [3, H]
  TP rank1 hidden: [5, H]

TP rank row counts:
  tp_rank_row_counts = [3, 5]

TP AllGather 后每个 TP rank 都看到：
  gathered hidden: [8, H] = rank0 rows [0:3] | rank1 rows [3:8]
```

expert 的 row-parallel down projection 后，两个 TP rank 都有 `[8, H]` 的 partial hidden。TP ReduceScatterRowsSum 需要
对这两个 `[8, H]` 做 SUM，并按同一个 `tp_rank_row_counts` 切回：

```text
TP rank0 output: rows [0:3] -> [3, H]
TP rank1 output: rows [3:8] -> [5, H]
```

因此当前设计选择是：**优先实现真正的变长 `reduce_scatter`，不引入 padding/capacity**。dispatcher 已经有
`tp_rank_row_counts` 正好可以作为变长 reduce scatter 的 split 边界：

```python
input_tensor_list = list(torch.split(hidden.contiguous(), tp_rank_row_counts, dim=0))
output = torch.empty_like(input_tensor_list[tp_rank])
dist.reduce_scatter(output, input_tensor_list, op=dist.ReduceOp.SUM, group=tp_group)
```

当 `tp_rank_row_counts` 全部相等时，可以在共享核心函数内部走等长 fast path：

```python
dist.reduce_scatter_tensor(output, hidden.contiguous(), op=dist.ReduceOp.SUM, group=tp_group)
```

但这只是实现优化，不改变 dispatcher 对外的 `tp_rank_row_counts` 语义。真正的 ReduceScatterRowsSum 实现应集中在一个共享核心
函数中，避免 combine forward 和 TP AllGather backward 分叉。

### 为什么不先做 padding/capacity

padding 和 capacity 带来的收益不同，需要分开看：

- **padding 的收益** 是把一次变长 collective 包装成等长 collective。通信前把每个 TP rank 的真实 slice pad 到同一
  长度，通信时就可以使用 `reduce_scatter_tensor` / `all_gather_into_tensor` 这类 tensor fast path。若 capacity
  仍由本 step 的 `max(tp_rank_row_counts)` 动态决定，padding 只减少大块 hidden collective 的 variable-list
  split 开销，不能消除 `tp_rank_row_counts` 的 CPU 同步。
- **固定 capacity 的收益** 是让这个等长长度跨 step 稳定下来。只有 capacity 是配置值或静态上界时，shape 才稳定，
  大块通信 shape 才能从本 step 的 Python split list 中解耦，后续也才更容易做 CUDA graph、buffer 复用或通信
  buffer 预分配。
- **对 Domino 的影响** 主要来自 host CPU split metadata 同步。只做动态 padding 时，host 仍要拿到
  `tp_rank_row_counts` 来决定 pad/unpad 边界和本步 capacity，因此这个同步点仍然存在；固定 capacity 才可能减少
  运行时 shape 决策，并把大块通信从 split-list 发起路径中移出。这和前面 EP All2All 的 host metadata 同步问题
  类似：host 等 split list 时，已经 enqueue 到 GPU 的另一个 micro batch 计算仍可继续，但 host 不能继续
  enqueue 后续本地算子和通信；如果等待时间超过可覆盖窗口，会压缩 Domino 的真实 overlap。

因此，如果只是每步动态取 `capacity = max(tp_rank_row_counts)`，它仍然需要 `tp_rank_row_counts` 的 CPU 同步，只能减少
variable collective 的 split-list 开销，不能获得固定 shape / CUDA graph，也不能消除 `tp_rank_row_counts` 对 Domino
host enqueue 的影响。

但它会把问题从通信层扩散到 layout 层。至少有两种做法：

1. **通信内部 padding，通信后立刻 unpad。**

   例如 `tp_rank_row_counts` 是 `[3, 5]`，capacity 取 `5`。AllGather 前把 rank0 的 `[3, H]` pad 到 `[5, H]`，
   rank1 保持 `[5, H]`；等长 AllGather 得到 `[10, H]` 后再按真实 sizes compact 回 `[8, H]`。ReduceScatter
   则需要先按 `[3, 5]` 切分、分别 pad 到 `[5, H]`，concat 成 `[10, H]` 后走 `reduce_scatter_tensor`，
   最后再 unpad 成当前 rank 的真实 `[3, H]` 或 `[5, H]`。

   这个方案不改变 expert 看到的 token 数，但增加 pad/unpad copy，并且仍然需要 `tp_rank_row_counts`。收益要靠 benchmark
   证明。

2. **端到端 capacity，让 padding token 进入 expert layout。**

   这种方案会让 `[tp_size * capacity, H]` 直接进入 `dispatch_postprocess` 和 grouped GEMM。它需要定义 padding
   token 的 expert 归属、`tokens_per_expert` 是否包含 padding、grouped GEMM 是否计算 padding、combine 如何剔除
   padding，以及 `row_ids_map` / `topk_weights` 如何保证 padding 不影响真实 token。

   这会把改动扩散到 routing、expert layout、postprocess/combine，不适合作为替换 `all_reduce + slice` 的第一步。

因此当前阶段的目标是局部替换：用真正的 TP ReduceScatterRowsSum 取代 `all_reduce + slice`，输出 shape 严格按照
`tp_rank_row_counts[tp_rank]` 分配，允许 0 行，不做 padding/capacity。
# torch.compile 与 dispatcher 边界

`FSDPConfig.torch_compile=True` 目前只是一个兼容入口，真正决定 compile 行为的是
`XTunerBaseModelConfig.compile_cfg`：

- `compile_cfg=None` 或 `True`：使用模型自己的 `default_compile_cfg`。
- `compile_cfg=False`：关闭 compile。
- `compile_cfg=dict[...]`：用户显式指定 compile target。
- `FSDPConfig.torch_compile=False` 会在 trainer 配置解析阶段把 `model_cfg.compile_cfg` 强制设成 `False`；反过来
  `FSDPConfig.torch_compile=True` 不会强制覆盖用户自定义的 `compile_cfg`。

对 MoE 来说，默认 compile target 会根据 dispatcher 是否包含跨 rank 通信编排分两类：

- `ep_size == 1` 且 `expert_tp_size == 1`：使用 `MOE_NON_EP_COMPILE_CFG`。普通 MoE 会把
  `MoEDecoderLayer.forward` 作为 compile target，同时也 compile `MoEBlock.forward`、
  `_pre_moe_forward`、`_shared_experts_forward`、`_post_moe_forward`、dense layer 和 float8 相关函数。
- `ep_size > 1` 或 `expert_tp_size > 1`：使用 `MOE_EP_COMPILE_CFG`。它从 non-EP 配置复制而来，但显式删除
  `MoEDecoderLayer.forward`，保留局部计算函数的 compile。

`qwen3_5_text` 的 non-EP 配置也包含 `MoEDecoderLayer.forward`，但该 target 使用 `fullgraph=False`；EP 开启后同样会从
默认配置中删除顶层 `MoEDecoderLayer.forward`。

这个差异是 dispatcher 边界的核心：EP 或 ExpertTP 开启后，`MoEDecoderLayer.forward` 顶层会承载
`dispatch_preprocess -> dispatch -> dispatch_postprocess -> expert -> combine_preprocess -> combine -> combine_postprocess`
的变长通信编排，以及 Domino micro batch 的多输入分支、CUDA event、autograd hook、DeepEP handle 等动态对象。
这些部分不适合作为稳定的 fullgraph compile 边界，因此当前设计让 dispatcher 编排保持 eager Python，只把相对稳定的本地计算块交给
`torch.compile`。

这也意味着 compile 不会消除前面描述的 dispatcher host metadata 同步：

- `TorchAll2AllDispatcher` 仍需要在 dispatch 阶段拿到 Python `input_splits` / `output_splits`。
- `DeepEPDispatcher` 仍可能在库内部等待 receive count，并把 `num_recv_tokens_per_expert_list` 暴露给 Python。
- TP+EP 路径仍需要 `tp_rank_row_counts` 来发起变长 TP AllGather / ReduceScatterRowsSum。

因此，对 Domino EP 来说，compile 的收益主要是缩短 `_pre_moe_forward`、expert block、`_post_moe_forward` 等本地计算段；
它不能把 dispatcher 的 host 等待变成 GPU-only 异步，也不能改变 2.1 和 DeepEP “Host metadata 同步”小节里的重叠约束。
如果 host metadata 等待超过另一个 micro batch 能覆盖的计算窗口，真实 overlap 仍会下降。

# DeepEPDispatcher
## DeepEPDispatcher: DeepEP Buffer dispatch/combine 原理

`DeepEPDispatcher` 仍然暴露和其他 dispatcher 一样的六阶段接口，但它把 EP all2all 的 routing layout、通信 handle
和 event 管理交给 DeepSeek 开源 DeepEP 库的 `Buffer` API。DeepEP 的核心接口是：

- `Buffer.get_dispatch_layout(topk_idx, num_experts, ...)`：根据 topK expert 选择计算 dispatch layout。
- `Buffer.dispatch(...)`：把 token、`topk_idx`、`topk_weights` 发到拥有选中 expert 的 EP rank。
- `Buffer.combine(...)`：使用 dispatch 返回的 handle，把 expert 输出或 dispatch backward 的梯度送回 source rank。
- `EventOverlap`：DeepEP 对 CUDA event 的包装，支持 `current_stream_wait()` 让当前 compute stream 等通信完成。

XTuner 的包装在 `xtuner/v1/ops/comm/deepep_op.py` 中：

```python
num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
    buffer.get_dispatch_layout(topk_idx, num_experts, previous_event=previous_event, async_finish=True)

recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
    buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
```

### DeepEP dispatch

`DeepEPDispatcher.dispatch_preprocess` 不像 `TorchAll2AllDispatcher` 那样先本地 `permute`。它只保留原始 source token
hidden，并把 `topk_ids` 转成 DeepEP 需要的 `int64`：

```text
hidden_states: [N, H]
topk_ids:      [N, K]
topk_weights:  [N, K]
```

跨 EP rank 搬运由 DeepEP dispatch kernel 完成；真正的 route-copy 展开仍在本 rank 的
`dispatch_postprocess -> permute(recv_topk_idx)` 中完成。`Buffer.dispatch` 返回：

```text
recv_x                         # 本 EP rank 收到的 source token hidden
recv_topk_idx                  # 与 recv_x 对齐的 [M_recv, K] expert ids；非本 rank expert 位置为 -1
recv_topk_weights              # 与 recv_topk_idx 对齐的 topK weights
num_recv_tokens_per_expert_list # 本 rank 每个 local expert 收到的 token 数
handle                         # combine/backward 复用的通信 handle
event                          # dispatch 完成事件
```

`handle` 是 DeepEP 的关键抽象。XTuner 注释里列出的 intranode handle 包括：

```text
rank_prefix_matrix
channel_prefix_matrix
recv_channel_prefix_matrix
recv_src_idx
is_token_in_rank
send_head
```

这些张量记录了 dispatch 的源/目的映射、channel 前缀和接收源索引。后续 combine 不再重新根据 routing 计算布局，而是
复用这个 handle 把 token 送回原 source rank；dispatch backward 和 combine backward 也复用同一个 handle。

### DeepEP dispatch_postprocess

DeepEP dispatch 已经把 token 发到拥有相关 local expert 的 EP rank，但输出还不是 grouped GEMM 需要的 local expert 连续分组。
`dispatch_postprocess` 会先等待 dispatch event，然后用 `recv_topk_idx` 再做一次本地 `permute`：

```text
recv_x
  --permute(recv_topk_idx, num_out_tokens=sum(num_recv_tokens_per_expert_list))-->
local expert grouped hidden
```

`num_recv_tokens_per_expert_list` 被转换成 `tokens_per_expert`，供 grouped GEMM 使用。

### DeepEP combine_preprocess / combine

DeepEP 当前方案和 `TorchAll2AllDispatcher` 的一个重要差异是 `topk_weights` 的位置：

- `TorchAll2AllDispatcher` 把 `topk_weights` 留在 source rank，最后 `combine_postprocess` 本地加权合并。
- `DeepEPDispatcher` 在 dispatch 时把 `topk_weights` 一起发到拥有选中 expert 的 EP rank，并在
  `combine_preprocess` 先加权合并：

```python
hidden_states = unpermute(
    hidden_states,
    post_dispatched["row_ids_map"],
    probs=dispatched["topk_weights"],
)
```

因此 DeepEP 的 forward combine 调用不再传 `topk_weights`：

```python
combined_x, _, event = buffer.combine(x, handle, async_finish=True, previous_event=previous_event)
```

进入 combine 的 hidden 已经是按 `recv_topk_weights` fold 过的 source-token partial output。DeepEP combine 只负责使用
dispatch handle 把这些 hidden 送回 source rank 并做 SUM reduce。

### DeepEP backward

DeepEP 的反向复用相反方向的通信原语：

- `DeepEPCombine.backward` 调用 `Buffer.dispatch(..., handle=handle)`：combine forward 的反向是 dispatch。
- `DeepEPDispatch.backward` 调用 `Buffer.combine(grad_recv_x, handle, topk_weights=grad_recv_topk_weights)`：
  dispatch forward 的反向是 combine，并且同时把 `grad_recv_topk_weights` 送回 source 侧，得到
  `combined_grad_recv_topk_weights`。

这解释了为什么 DeepEP dispatch 是一个 composite autograd op：它的 forward 同时产生 `recv_x` 和
`recv_topk_weights`，backward 也同时返回 `x` 和 `topk_weights` 的梯度。

## DeepEPDispatcher 前向示例

继续使用前面 All2All 示例里的配置和 routing：

```text
EP = 2
E_local = 3
E = 6
K = 2
每个 EP rank 本地 N = 4 个 token

ep0 owns global expert 0,1,2
ep1 owns global expert 3,4,5

ep0 source tokens: A0 A1 A2 A3
ep1 source tokens: B0 B1 B2 B3
```

routing 仍然是：

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

为了把 weighted combine 写成具体数字，取验证脚本里的 `topk_weights`：

```text
ep0 weights:
A0 -> [0.25, 0.75]
A1 -> [0.40, 0.60]
A2 -> [0.70, 0.30]
A3 -> [0.80, 0.20]

ep1 weights:
B0 -> [0.20, 0.80]
B1 -> [0.50, 0.50]
B2 -> [0.90, 0.10]
B3 -> [0.35, 0.65]
```

### 1. `dispatch_preprocess`: 不做本地 route-copy 展开

DeepEP 不像 `TorchAll2AllDispatcher` 那样先在 source rank 本地把 token 展开成 `[N*K, H]` 并按 global expert 排序。
`dispatch_preprocess` 只保留原始 token，并把 `topk_ids` 转成 `int64`：

```text
pre_dispatched["hidden_states"]: [N, H] = [4, H]
pre_dispatched["topk_ids"]:      [N, K] = [4, 2]
```

### 2. `dispatch`: 每个目标 EP rank 收一份 source token

DeepEP 的 layout 先判断每个 token 是否需要发送到某个 EP rank：只要 token 的任意 topK expert 在该 rank，本 token 就向该
rank 发送一行 hidden。也就是说，通信粒度是 **token 到 rank**，不是一开始就按 expert 展开成 route-copy。

本例中每个 token 都正好有一个 expert 在 `ep0`、一个 expert 在 `ep1`，所以两个目标 rank 都收到 8 行 source token：

```text
dispatched row: 0  1  2  3 | 4  5  6  7
source token:   A0 A1 A2 A3| B0 B1 B2 B3
```

DeepEP 同时把 global expert id 转成当前 receiver rank 的 local expert id；不属于当前 rank 的 topK slot 写成 `-1`，
对应 weight 写成 `0`。

`ep0` 收到：

```text
recv_topk_idx row:  0      1      2      3    | 4      5      6      7
source token:       A0     A1     A2     A3   | B0     B1     B2     B3
recv_topk_idx:      [0,-1] [-1,1] [2,-1] [-1,0] [1,-1] [-1,2] [-1,0] [-1,1]
recv_topk_weights:  [.25,0] [0,.60] [.70,0] [0,.20] [.20,0] [0,.50] [0,.10] [0,.65]
```

`ep1` 收到：

```text
recv_topk_idx row:  0      1      2      3    | 4      5      6      7
source token:       A0     A1     A2     A3   | B0     B1     B2     B3
recv_topk_idx:      [-1,1] [0,-1] [-1,2] [1,-1] [-1,0] [1,-1] [2,-1] [0,-1]
recv_topk_weights:  [0,.75] [.40,0] [0,.30] [.80,0] [0,.80] [.50,0] [.90,0] [.35,0]
```

两边的 local expert token 数都是：

```text
num_recv_tokens_per_expert_list = [3, 3, 2]
```

### 3. `dispatch_postprocess`: receiver rank 内展开并按 local expert 分组

`dispatch_postprocess` 对 `recv_topk_idx` 做本地 `permute`。这一步才真正把收到的 token 展开成 local expert 的
route-copy，并丢掉 `-1` slot。

对 `ep0`：

```text
post row:        0  1  2 | 3  4  5 | 6  7
token copy:      A0 A3 B2| A1 B0 B3| A2 B1
local expert id: 0  0  0 | 1  1  1 | 2  2
row_ids_map:     [0,-1,6,-1,4,-1,-1,-1,-1,3,-1,1,-1,7,2,5]
```

对 `ep1`：

```text
post row:        0  1  2 | 3  4  5 | 6  7
token copy:      A1 B0 B3| A0 A3 B1| A2 B2
local expert id: 0  0  0 | 1  1  1 | 2  2
row_ids_map:     [-1,0,-1,4,-1,5,7,2,3,-1,6,-1,1,-1,-1,-1]
```

这里的 `row_ids_map` 长度是 `M_recv*K`，因为它对应的是带 `-1` 的 `recv_topk_idx` flat 空间；`-1` slot 在
`row_ids_map` 里也保持为 `-1`。这和 All2All 例子中 destination rank 第二次 `permute` 的 `[M_recv]` map 不同。

### 4. local experts grouped GEMM

假设为了便于观察，每个 expert 输出第一列为：

```text
out(token, global_expert_id) = token_value + global_expert_id * 100
```

那么 `ep0` grouped GEMM 输出：

```text
post row:        0  1  2 | 3   4   5  | 6   7
token copy:      A0 A3 B2| A1  B0  B3 | A2  B1
global expert:   0  0  0 | 1   1   1  | 2   2
experts_out:     10 13 22| 111 120 123| 212 221
```

`ep1` grouped GEMM 输出：

```text
post row:        0   1   2  | 3   4   5  | 6   7
token copy:      A1  B0  B3 | A0  A3  B1 | A2  B2
global expert:   3   3   3  | 4   4   4  | 5   5
experts_out:     311 320 323| 410 413 421| 512 522
```

### 5. `combine_preprocess`: expert rank 上先做 topK 加权折叠

DeepEP 已经把 `topk_weights` 发送到了 expert rank，所以 `combine_preprocess` 会在 receiver rank 本地执行：

```python
hidden_states = unpermute(experts_out, row_ids_map, probs=recv_topk_weights)
```

输出回到 `dispatch` 后的 source-token 顺序 `[A0 A1 A2 A3 | B0 B1 B2 B3]`，但每行已经只包含当前 EP rank 负责的
expert 加权结果。

`ep0`：

```text
pre_combined row: 0    1     2     3   | 4   5     6   7
source token:     A0   A1    A2    A3  | B0  B1    B2  B3
weighted output:  2.5  66.6  148.4 2.6 | 24  110.5 2.2 79.95
```

`ep1`：

```text
pre_combined row: 0     1      2     3    | 4   5     6     7
source token:     A0    A1     A2    A3   | B0  B1    B2    B3
weighted output:  307.5 124.4  153.6 330.4| 256 210.5 469.8 113.05
```

### 6. `combine`: 使用 DeepEP handle 送回 source rank 并 SUM

DeepEP combine 复用 dispatch 返回的 `handle`，把这些已经加权的 partial output 送回原 source rank，并对同一个 source
token 来自不同 EP rank 的 partial output 做 SUM。

source `ep0` 收回：

```text
A0 final = 2.5   + 307.5 = 310
A1 final = 66.6  + 124.4 = 191
A2 final = 148.4 + 153.6 = 302
A3 final = 2.6   + 330.4 = 333
```

source `ep1` 收回：

```text
B0 final = 24    + 256    = 280
B1 final = 110.5 + 210.5  = 321
B2 final = 2.2   + 469.8  = 472
B3 final = 79.95 + 113.05 = 193
```

因此 DeepEP 的：

```text
combined["hidden_states"]: [N, H] = [4, H]
post_combined["hidden_states"]: [N, H] = [4, H]
```

`combine_postprocess` 不再像 All2All 那样使用 source rank 的 `row_id_map` 和 `topk_weights` 做本地 topK 加权合并；DeepEP 的
topK 加权已经在 `combine_preprocess` 完成，`combine_postprocess` 主要负责 event 等待和返回 hidden。

## Host metadata 同步

DeepEP 不像 `TorchAll2AllDispatcher` 那样在 XTuner 代码里显式执行：

```python
to(device=torch.device("cpu")).tolist()
```

但它仍然存在 host 可见的 metadata 准备点。DeepEP 的 legacy Buffer API 文档和 XTuner 包装都注明：dispatch 内部不知道
当前 rank 会收到多少 token，因此 CPU 会等待 GPU signal，拿到 receive count 后才能继续。XTuner 代码中的表现是
`Buffer.dispatch` 返回 Python list：

```python
num_recv_tokens_per_expert_list, handle, event
```

`dispatch_postprocess` 必须用这个 list 计算 `num_out_tokens` 和 `tokens_per_expert`。因此 DeepEP 也不是完全无 host
同步；只是同步被 DeepEP 的 layout/dispatch handle 机制封装在库内部，不是 PyTorch split-size list 的
`.tolist()` 同步。

对 Domino EP 来说，两者的影响边界一致：

- 已经 enqueue 到 GPU 的另一个 micro batch 计算不会被 host 同步打断。
- host 等 metadata 时无法继续 enqueue 后续本地算子和通信。
- 如果 metadata 等待短于可覆盖的另一个 micro batch 计算，重叠效果基本保留。
- 如果 metadata 等待更长，`xtuner_ep_domino.md` 7.3 中的理想时间线会被压缩，真实重叠比例下降。

## 当前支持边界

当前 `build_dispatcher(dispatcher="deepep", tp_group=...)` 会直接构造 `DeepEPDispatcher`，`tp_group` 没有接入
DeepEP dispatcher。也就是说，XTuner 当前的 DeepEP 路径是 EP dispatcher，不包含 `TorchAll2AllTPEPDispatcher`
那套 TP AllGather / TP ReduceScatterRowsSum 通信段。DeepEP + ExpertTP 如果要成为 Domino-compatible ExpertTP，需要
额外设计 DeepEP dispatch 后的 TP AllGather、combine 前的 TP ReduceScatterRowsSum，以及相应的 `topk_weights`
event 语义；这部分见 `xtuner_etp.md`。

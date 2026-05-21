# XTuner MoE Dispatch

This context describes the communication language used by XTuner MoE dispatchers when routed experts use Expert Parallelism or Expert Tensor Parallelism.

## Language

**TP ReduceScatterRowsSum**:
对同一 TP group 中完整 token 批的 hidden 做 SUM 归约，并只保留当前 TP rank 负责的 token slice 的通信语义。
_Avoid_: all_reduce + slice

**Variable TP ReduceScatterRowsSum**:
使用 **TP rank row counts** 描述不等长 token slice 的 **TP ReduceScatterRowsSum**。
_Avoid_: equal-only reduce scatter

**TP rank row counts**:
每个 expert TP rank 在 TP AllGather 前、当前 dispatcher token 空间中拥有的 token 行数列表。代码中叫
`tp_rank_row_counts`，用来描述变长 TP token slice 的拼接和切分边界。
_Avoid_: shape hack, split list

**Token-sliced Expert TP**:
expert MLP 权重按 TP 切分，并让每个 expert TP rank 只保留自己的 token slice；expert 前用 **TP AllGather** 得到完整 token 批，expert 后用 **TP ReduceScatterRowsSum** 回到本 rank 的 token slice。
_Also called_: ExpertTP in dispatcher code
_Avoid_: replicated-token expert TP

**Domino-compatible ExpertTP**:
让 **Token-sliced Expert TP** 的 **TP AllGather** 属于 dispatcher dispatch 通信段，让 **TP ReduceScatterRowsSum** 属于 dispatcher combine 通信段，从而能被 Domino micro-batch 流水隐藏的 MoE expert TP 语义。
_Avoid_: attention TP, dense MLP TP

**Expert-side topK folding**:
在拥有 routed expert 的 rank 上，使用收到的 topK weights 将同一 source token 的多个 expert output 加权合并成一行 partial output。
_Avoid_: source-side DeepEP folding

## Relationships

- **TP AllGather** 的反向通信是 **TP ReduceScatterRowsSum**。
- **TP ReduceScatterRowsSum** 的反向通信是 **TP AllGather**。
- **TP rank row counts** 定义 **TP ReduceScatterRowsSum** 输出给每个 TP rank 的 token slice 边界。
- **Token-sliced Expert TP** 是 `expert_tp_size > 1` 的默认语义；`ep_size=1` 时 EP AllToAll 退化为空，但 TP AllGather / TP ReduceScatterRowsSum 仍然保留。
- **Variable TP ReduceScatterRowsSum** 是 routed MoE token-sliced expert TP 下的默认语义；等长 fast path 只是实现优化。
- **TP ReduceScatterRowsSum** 的实现策略应集中在一个共享核心函数中，避免 combine forward 和 TP AllGather backward 分叉。
- **TP ReduceScatterRowsSum** 的输出 shape 严格由当前 TP rank 的 **TP rank row counts** 决定，允许 0 行，不引入 padding 或 capacity。
- 当 `ep_size=1` 且 `expert_tp_size>1` 时，expert ownership 维度仍然存在，只是大小为 1；所有 routed experts 都属于这个唯一 EP rank。
- 在 Naive routing + **Token-sliced Expert TP** 下，**TP rank row counts** 记录 source token rows。
- 在 All2All routing + **Token-sliced Expert TP** 下，**TP rank row counts** 记录 EP AllToAll 后的 route-copy rows。
- 在 DeepEP routing + **Token-sliced Expert TP** 下，**TP rank row counts** 记录 DeepEP dispatch 后的 received source-token rows；local expert route-copy rows 由 DeepEP 的 received topK ids 展开得到。
- **Token-sliced Expert TP** 的异步边界由 TP AllGather 和 **TP ReduceScatterRowsSum** 定义；这个边界不依赖 EP 是否开启。
- 当前支持范围是 Naive routing、All2All routing、DeepEP routing 与 **Token-sliced Expert TP** 的组合。
- DeepEP routing + **Token-sliced Expert TP** 保留 **Expert-side topK folding**：DeepEP dispatch 后 TP AllGather hidden、topK ids 和 topK weights；expert output 先按 gathered topK weights 折叠，再做 **TP ReduceScatterRowsSum** 和 DeepEP combine。
- DeepEP routing + **Token-sliced Expert TP** 的 dispatch TP 段必须使用同一份 **TP rank row counts** 对齐 AllGather hidden、received topK ids 和 received topK weights。
- DeepEP routing + **Token-sliced Expert TP** 中，received topK ids 是无梯度 row metadata；它参与 TP AllGather 只为保持与 hidden/topK weights 的行顺序一致。
- DeepEP routing + **Token-sliced Expert TP** 的 TP AllGather 属于 dispatcher `dispatch` 阶段；`dispatch_postprocess` 只消费 gathered 数据并构造 local expert layout。
- DeepEP routing + **Token-sliced Expert TP** 的 **TP ReduceScatterRowsSum** 属于 dispatcher `combine` 阶段；`combine_preprocess` 只做 **Expert-side topK folding**。
- DeepEP routing + **Token-sliced Expert TP** 的 grouped GEMM `tokens_per_expert` 来自各 TP rank 的 DeepEP `num_recv_tokens_per_expert_list` 聚合求和；重新扫描 gathered topK ids 只适合作为校验。
- DeepEP routing + **Token-sliced Expert TP** 中，DeepEP 原始 `num_recv_tokens_per_expert_list` 字段不随 ExpertTP 开启而改变；TP 聚合后的计数只作为 grouped GEMM 的 `tokens_per_expert`。
- DeepEP routing + **Token-sliced Expert TP** 中，同一 EP rank 内每个 expert TP rank 都对完整 gathered expert input 运行自己的 expert weight shard，输出在 `combine` 阶段通过 **TP ReduceScatterRowsSum** 求和并切回本 TP rank token slice。
- DeepEP routing + **Token-sliced Expert TP** 必须保留 `async_op=True` 语义；hidden 和 topK weights 的反向通信完成前，不能让上游 backward 消费对应梯度。
- DeepEP routing + **Token-sliced Expert TP** 的 async backward 边界必须同时覆盖 hidden 分支和 topK weights 分支；`topk_weights.grad_fn` 需要等待 TP weights ReduceScatterRowsSum 与 DeepEP dispatch backward 完成后再继续上游 router backward。
- DeepEP routing + **Token-sliced Expert TP** 必须支持 topK weights 有梯度和无梯度两种输入；有梯度路径是验证重点。
- DeepEP routing + **Token-sliced Expert TP** 的 DeepEP `EventOverlap` 与 TP `torch.cuda.Event` 衔接只属于 `DeepEPDispatcher` 内部适配；共享 **Token-sliced Expert TP** helper 不依赖 DeepEP 类型。
- `dispatcher="deepep"` 在 `expert_tp_size>1` 时仍表示 `DeepEPDispatcher`，由同一个 dispatcher 根据 `tp_group` 接入 **Token-sliced Expert TP**，不引入新的 dispatcher 名称。
- DeepEP routing + **Token-sliced Expert TP** 的验证应覆盖 dispatcher 六阶段 public API 的真实 forward/backward 路径、模型级 MoE 接线路径和 Domino micro-batch async staging，而不是只验证内部 helper。
- **Token-sliced Expert TP** 的 TP group 必须位于同一个 expert ownership 内；在 `(fsdp, ep, etp)` mesh 中，同一 TP group 的 ranks 共享相同 EP rank，只在 expert TP rank 上不同。
- DeepEP routing + **Token-sliced Expert TP** 的首个支持目标是训练 forward/backward；`decoding=True` 仍不属于支持范围。
- DeepEP routing + **Token-sliced Expert TP** 的首个支持目标只要求 BF16 训练通信路径；FP8 DeepEP 通信 dtype 不属于该目标。
- `tp_group=None` 时，DeepEP routing 不启用 **Token-sliced Expert TP**，行为必须保持原有 DeepEP-only 语义。
- **Domino-compatible ExpertTP** 只覆盖 MoE routed experts 的 **Token-sliced Expert TP** 通信隐藏，不表示 attention 或 dense MLP 的普通 TP。
- 进入 routed experts 前，每个 expert TP rank 已经持有不重复的 source token slice；这些 slice 可以来自不同样本，也可以来自同一样本的不同序列片段。

## Example dialogue

> **Dev:** "combine forward 和 TP AllGather backward 都能叫 **TP ReduceScatterRowsSum** 吗？"
> **Domain expert:** "可以。它们都是先跨 TP rank 做 SUM，再只保留当前 rank 的 token slice。具体用 reduce_scatter 还是 all_reduce + slice 是实现细节。"

> **Dev:** "只支持等长 reduce scatter 够吗？"
> **Domain expert:** "不够。EP routing 后每个 TP rank 的 token 数可能不同，默认要按 **TP rank row counts** 做 **Variable TP ReduceScatterRowsSum**。"

> **Dev:** "等长和变长 reduce scatter 要不要分别写在不同调用点？"
> **Domain expert:** "不要。调用点只表达 **TP ReduceScatterRowsSum**，共享核心函数内部选择等长 fast path 或变长路径。"

> **Dev:** "如果某个 TP rank 没有 token，要不要 pad 到 1 行或固定容量？"
> **Domain expert:** "不要。**TP ReduceScatterRowsSum** 输出真实 token slice，0 行就是合法输出。"

> **Dev:** "不开 EP 只开 expert TP 时，是不是可以让每个 TP rank 都持有完整 token 批，最后做 all-reduce？"
> **Domain expert:** "不采用这个语义。无 EP expert TP 仍然是 **Token-sliced Expert TP**：前向按 TP token slice 进入 dispatcher，expert 前 all-gather，expert 后 reduce-scatter。"

> **Dev:** "Naive routing + expert TP 时，TP AllGather 是 gather source tokens，还是 gather topK 展开后的 route-copy tokens？"
> **Domain expert:** "gather source tokens。topK route-copy 展开仍然发生在 expert layout 阶段；expert 输出先 fold 回 source token partial output，再做 **TP ReduceScatterRowsSum**。"

> **Dev:** "Naive routing + expert TP 的异步路径要不要和 EP routing + expert TP 使用同一套分段语义？"
> **Domain expert:** "要。Naive routing 没有 EP AllToAll，但 **TP AllGather** 和 **TP ReduceScatterRowsSum** 仍然是 dispatcher 通信段，异步依赖边界应保持一致。"

> **Dev:** "DeepEP + expert TP 的 **TP rank row counts** 是 route-copy 行数吗？"
> **Domain expert:** "不是。DeepEP dispatch 收到的是 source-token rows；route-copy/local expert 展开发生在 `dispatch_postprocess`，所以 **TP rank row counts** 记录 received source-token rows。"

## Flagged ambiguities

- "reduce scatter" 在本上下文中特指 **TP ReduceScatterRowsSum**；不是只做 scatter，也不是不带 SUM 的切分。

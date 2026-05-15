# XTuner MoE Dispatch

This context describes the communication language used by XTuner MoE dispatchers when routed experts use Expert Parallelism or Expert Tensor Parallelism.

## Language

**TP ReduceScatterSum**:
对同一 TP group 中完整 token 批的 hidden 做 SUM 归约，并只保留当前 TP rank 负责的 token slice 的通信语义。
_Avoid_: all_reduce + slice

**Variable TP ReduceScatterSum**:
使用 **TP size meta** 描述不等长 token slice 的 **TP ReduceScatterSum**。
_Avoid_: equal-only reduce scatter

**TP size meta**:
每个 expert TP rank 在 TP AllGather 前、当前 dispatcher token 空间中拥有的 token 行数列表，用来描述变长 TP token slice 的拼接和切分边界。
_Avoid_: shape hack, split list

**Token-sliced Expert TP**:
expert MLP 权重按 TP 切分，并让每个 expert TP rank 只保留自己的 token slice；expert 前用 **TP AllGather** 得到完整 token 批，expert 后用 **TP ReduceScatterSum** 回到本 rank 的 token slice。
_Also called_: ExpertTP in dispatcher code
_Avoid_: replicated-token expert TP

**Domino-compatible ExpertTP**:
让 **Token-sliced Expert TP** 的 **TP AllGather** 属于 dispatcher dispatch 通信段，让 **TP ReduceScatterSum** 属于 dispatcher combine 通信段，从而能被 Domino micro-batch 流水隐藏的 MoE expert TP 语义。
_Avoid_: attention TP, dense MLP TP

## Relationships

- **TP AllGather** 的反向通信是 **TP ReduceScatterSum**。
- **TP ReduceScatterSum** 的反向通信是 **TP AllGather**。
- **TP size meta** 定义 **TP ReduceScatterSum** 输出给每个 TP rank 的 token slice 边界。
- **Token-sliced Expert TP** 是 `expert_tp_size > 1` 的默认语义；`ep_size=1` 时 EP AllToAll 退化为空，但 TP AllGather / TP ReduceScatterSum 仍然保留。
- **Variable TP ReduceScatterSum** 是 routed MoE token-sliced expert TP 下的默认语义；等长 fast path 只是实现优化。
- **TP ReduceScatterSum** 的实现策略应集中在一个共享核心函数中，避免 combine forward 和 TP AllGather backward 分叉。
- **TP ReduceScatterSum** 的输出 shape 严格由当前 TP rank 的 **TP size meta** 决定，允许 0 行，不引入 padding 或 capacity。
- 当 `ep_size=1` 且 `expert_tp_size>1` 时，expert ownership 维度仍然存在，只是大小为 1；所有 routed experts 都属于这个唯一 EP rank。
- 在 Naive routing + **Token-sliced Expert TP** 下，**TP size meta** 记录 source token rows；在 EP routing + **Token-sliced Expert TP** 下，**TP size meta** 记录 EP routing 后的 route-copy rows。
- **Token-sliced Expert TP** 的异步边界由 TP AllGather 和 **TP ReduceScatterSum** 定义；这个边界不依赖 EP 是否开启。
- 当前支持范围是 Naive routing + **Token-sliced Expert TP** 和 All2All routing + **Token-sliced Expert TP**；DeepEP routing + **Token-sliced Expert TP** 暂不作为目标语义。
- **Domino-compatible ExpertTP** 只覆盖 MoE routed experts 的 **Token-sliced Expert TP** 通信隐藏，不表示 attention 或 dense MLP 的普通 TP。
- 进入 routed experts 前，每个 expert TP rank 已经持有不重复的 source token slice；这些 slice 可以来自不同样本，也可以来自同一样本的不同序列片段。

## Example dialogue

> **Dev:** "combine forward 和 TP AllGather backward 都能叫 **TP ReduceScatterSum** 吗？"
> **Domain expert:** "可以。它们都是先跨 TP rank 做 SUM，再只保留当前 rank 的 token slice。具体用 reduce_scatter 还是 all_reduce + slice 是实现细节。"

> **Dev:** "只支持等长 reduce scatter 够吗？"
> **Domain expert:** "不够。EP routing 后每个 TP rank 的 token 数可能不同，默认要按 **TP size meta** 做 **Variable TP ReduceScatterSum**。"

> **Dev:** "等长和变长 reduce scatter 要不要分别写在不同调用点？"
> **Domain expert:** "不要。调用点只表达 **TP ReduceScatterSum**，共享核心函数内部选择等长 fast path 或变长路径。"

> **Dev:** "如果某个 TP rank 没有 token，要不要 pad 到 1 行或固定容量？"
> **Domain expert:** "不要。**TP ReduceScatterSum** 输出真实 token slice，0 行就是合法输出。"

> **Dev:** "不开 EP 只开 expert TP 时，是不是可以让每个 TP rank 都持有完整 token 批，最后做 all-reduce？"
> **Domain expert:** "不采用这个语义。无 EP expert TP 仍然是 **Token-sliced Expert TP**：前向按 TP token slice 进入 dispatcher，expert 前 all-gather，expert 后 reduce-scatter。"

> **Dev:** "Naive routing + expert TP 时，TP AllGather 是 gather source tokens，还是 gather topK 展开后的 route-copy tokens？"
> **Domain expert:** "gather source tokens。topK route-copy 展开仍然发生在 expert layout 阶段；expert 输出先 fold 回 source token partial output，再做 **TP ReduceScatterSum**。"

> **Dev:** "Naive routing + expert TP 的异步路径要不要和 EP routing + expert TP 使用同一套分段语义？"
> **Domain expert:** "要。Naive routing 没有 EP AllToAll，但 **TP AllGather** 和 **TP ReduceScatterSum** 仍然是 dispatcher 通信段，异步依赖边界应保持一致。"

## Flagged ambiguities

- "reduce scatter" 在本上下文中特指 **TP ReduceScatterSum**；不是只做 scatter，也不是不带 SUM 的切分。

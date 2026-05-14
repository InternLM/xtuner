# XTuner MoE Dispatch

This context describes the communication language used by XTuner MoE dispatchers when Expert Parallelism and Tensor Parallelism are enabled together.

## Language

**TP ReduceScatterSum**:
对同一 TP group 中完整 token 批的 hidden 做 SUM 归约，并只保留当前 TP rank 负责的 token slice 的通信语义。
_Avoid_: all_reduce + slice

**Variable TP ReduceScatterSum**:
使用 **TP size meta** 描述不等长 token slice 的 **TP ReduceScatterSum**。
_Avoid_: equal-only reduce scatter

**TP size meta**:
每个 TP rank 在 EP dispatch 后拥有的 token 行数列表，用来描述变长 TP token slice 的拼接和切分边界。
_Avoid_: shape hack, split list

## Relationships

- **TP AllGather** 的反向通信是 **TP ReduceScatterSum**。
- **TP ReduceScatterSum** 的反向通信是 **TP AllGather**。
- **TP size meta** 定义 **TP ReduceScatterSum** 输出给每个 TP rank 的 token slice 边界。
- **Variable TP ReduceScatterSum** 是 TP+EP MoE routing 下的默认语义；等长 fast path 只是实现优化。
- **TP ReduceScatterSum** 的实现策略应集中在一个共享核心函数中，避免 combine forward 和 TP AllGather backward 分叉。
- **TP ReduceScatterSum** 的输出 shape 严格由当前 TP rank 的 **TP size meta** 决定，允许 0 行，不引入 padding 或 capacity。

## Example dialogue

> **Dev:** "combine forward 和 TP AllGather backward 都能叫 **TP ReduceScatterSum** 吗？"
> **Domain expert:** "可以。它们都是先跨 TP rank 做 SUM，再只保留当前 rank 的 token slice。具体用 reduce_scatter 还是 all_reduce + slice 是实现细节。"

> **Dev:** "只支持等长 reduce scatter 够吗？"
> **Domain expert:** "不够。EP routing 后每个 TP rank 的 token 数可能不同，默认要按 **TP size meta** 做 **Variable TP ReduceScatterSum**。"

> **Dev:** "等长和变长 reduce scatter 要不要分别写在不同调用点？"
> **Domain expert:** "不要。调用点只表达 **TP ReduceScatterSum**，共享核心函数内部选择等长 fast path 或变长路径。"

> **Dev:** "如果某个 TP rank 没有 token，要不要 pad 到 1 行或固定容量？"
> **Domain expert:** "不要。**TP ReduceScatterSum** 输出真实 token slice，0 行就是合法输出。"

## Flagged ambiguities

- "reduce scatter" 在本上下文中特指 **TP ReduceScatterSum**；不是只做 scatter，也不是不带 SUM 的切分。

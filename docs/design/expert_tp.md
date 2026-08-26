# MoE Expert Tensor Parallelism

Expert Tensor Parallelism（ETP）在每个 expert 内切分 grouped-linear 权重，并与 Expert Parallelism（EP）正交组合。配置入口是 `MoEConfig.expert_tp_size`；它与用于模型其余部分的 `FSDPConfig.tp_size` 含义不同。

```mermaid
flowchart LR
    A["Router 输出物理 expert id"] --> B{"Dispatcher"}
    B -->|"Naive / All2All"| C["ETP rank 间复制 token"]
    B -->|"DeepEP"| D["映射为 EP × ETP 虚拟 expert"]
    C --> E["Column-parallel w1/w3"]
    D --> E
    E --> F["Row-parallel w2"]
    F --> G["合并 ETP partial output"]
    G --> H["返回原 token 顺序"]
```

## Mesh 与参数布局

训练 mesh 使用 `(fsdp, ep, etp)` 维度，其中 `fsdp = world_size / (ep_size * expert_tp_size)`。Expert 权重位于 `(ep, etp)` 子 mesh：

- `fused_w1w3` 同时沿 expert 维和每个 expert 的输出维切分，placement 为 `(Shard(0), InterleavedShard(0))`。融合 gate/up projection 时，stripe 数是 `local_experts * 2`。
- `fused_w2` 沿 expert 维和输入维切分，placement 为 `(Shard(0), Shard(1))`。
- 非 expert 参数在 EP、ETP 两个维度均为 `Replicate()`；保留二维子 mesh，使其与 FSDP mesh 共享同一个 parent。

`InterleavedShard` 让运行时切片保留完整全局语义，因此 HF save/load 和 DCP 可以从 DTensor placement 直接规划，无需为 MoE 参数维护另一套 checkpoint 特例。

## Dispatcher

无 EP 时使用 `NaiveDispatcher`，All2All 场景使用统一的 `TorchAll2AllDispatcher`。两者在 dispatch 后执行 ETP row all-gather，在 combine 前执行 row reduce-scatter sum，并共享同步、异步实现。

DeepEP 使用扁平的 `(ep, etp)` process group。每个物理 expert 映射为 `expert_tp_size` 个虚拟 expert；`topk_ids` 和 `topk_weights` 在 dispatch 前同步扩展。这样 DeepEP 一次 collective 即可完成 EP 路由和 ETP token 复制，且同步事件覆盖扩展 kernel，支持多 micro-batch 重叠。

## 梯度与 FP8

Expert 梯度按 `ep_size * expert_tp_size` 缩放；非 expert DTensor 根据 `Replicate` placement 归约。梯度范数对所有 expert shard 求和，确保 clipping 与单模型基线一致。

BF16 与 tile-wise FP8 grouped GEMM 使用相同权重布局。FP8 padding、量化和输出 reshape 均基于本地 shard shape，checkpoint 仍保留原始全局 shape。

综上，ETP 只改变 expert 内部的计算与通信布局；路由语义、模型输出及 HF/DCP checkpoint 的全局表示保持不变。

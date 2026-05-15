# XTuner ExpertTP Event Notes

本文记录 XTuner MoE dispatcher 中 Expert Tensor Parallelism（ExpertTP）的异步 event 语义。

## 几种 dispatcher 语义

ExpertTP 相关路径在 XTuner 里有几种常见组合：

1. Naive routing + ExpertTP：没有 EP AllToAll，TP rank 持有不重复的 source token slice。dispatch 阶段用
   TP AllGather 把各 TP rank 的 source token slice 拼成完整 source-token batch，然后本地展开 topK route-copy。
2. TorchAll2All EP + TP：先由 EP AllToAll 把 route-copy hidden 发到 expert 所在 EP rank，再由 TP AllGather
   把同一 EP rank 内各 TP rank 的 route-copy token slice 拼成 expert 输入。
3. DeepEP dispatcher：由 DeepEP `Buffer.dispatch` 同时通信 hidden、`topk_idx`、`topk_weights`，再用 DeepEP
   `Buffer.combine` 送回 source rank。当前 XTuner 的 DeepEP dispatcher 尚未接入 ExpertTP 的 TP AllGather /
   TP ReduceScatterSum。

这几种方式最大的差异是 `topk_weights` 在哪里参与 topK folding。

### Naive routing + ExpertTP

Naive + ExpertTP 的 dispatch TP AllGather 发生在 source-token 空间：

```text
local source tokens [N_local, H]
  --TP AllGather-->
full source tokens  [N_total, H]
  --dispatch_postprocess / permute(topk_ids)-->
route-copy tokens   [N_total * K, H]
```

因此，`topk_ids` 和 `topk_weights` 也必须和 gathered hidden 对齐到 `N_total` 个 source token。否则
`dispatch_postprocess` 无法基于完整 token batch 做 route-copy 展开，`combine_preprocess` 也无法在完整 source-token
空间中按本 token 的 topK weight fold 回 `[N_total, H]`。

所以 Naive + ExpertTP 的 dispatch 通信段需要：

```text
hidden_states TP AllGather
topk_ids      TP AllGather
topk_weights  TP AllGather
```

`topk_ids` 只是路由元数据，不需要 autograd。`topk_weights` 参与 `unpermute(..., probs=topk_weights)`，需要梯度，
因此它的 TP AllGather backward 会执行 TP ReduceScatterSum，把完整 token 空间里的 `dtopk_weights` 切回本 TP rank
的 source-token slice。

### TorchAll2All EP + TP

TorchAll2All EP + TP 的 dispatch 首先已经在 route-copy 空间中通信 hidden：

```text
source route-copy hidden
  --EP AllToAll-->
expert-rank route-copy hidden
  --TP AllGather-->
expert-rank full route-copy hidden
```

当前 XTuner 的 `TorchAll2AllTPEPDispatcher` 设计选择 **不通信 `topk_weights`**：专家侧只计算每个 route-copy
的 expert output，combine 通信把 route-copy output 送回 source 侧，最后由 `combine_postprocess` 在 source 侧使用
本地保留的 `topk_weights` 做 topK folding：

```text
expert output route-copy
  --TP ReduceScatterSum + EP combine-->
source route-copy output
  --unpermute(..., probs=local topk_weights)-->
source hidden [N_local, H]
```

这种设计下，`topk_weights` 一直留在 source rank / source TP slice 上，不需要 EP AllToAll，也不需要 TP AllGather。
因此当前 `TorchAll2AllTPEPDispatcher` 不存在 Naive + ExpertTP 中那条 `topk_weights` TP AllGather backward 的额外
event 问题。

### DeepEP dispatcher

DeepEP 的默认处理方式不同：`Buffer.dispatch` 会把 `topk_weights` 和 hidden、`topk_idx` 一起发到拥有选中
expert 的 EP rank。
XTuner 的 `DeepEPDispatcher.combine_preprocess` 随后在 expert rank 上执行：

```python
unpermute(expert_out, row_ids_map, probs=dispatched["topk_weights"])
```

也就是说，DeepEP 路径是在 expert 侧先按 `recv_topk_weights` 做 topK folding，再调用 `Buffer.combine` 把已经加权
合并后的 hidden 送回 source rank。它不是 `TorchAll2AllTPEPDispatcher` 那种“`topk_weights` 留在 source 侧，
最后再加权”的设计。

DeepEP dispatch 本身是一个 composite autograd op：

```text
forward : Buffer.dispatch(x, topk_idx, topk_weights) -> recv_x, recv_topk_idx, recv_topk_weights, handle
backward: Buffer.combine(grad_recv_x, handle, topk_weights=grad_recv_topk_weights)
          -> grad_x, grad_topk_weights
```

因此 DeepEP 的 `topk_weights` 梯度会沿 dispatch handle 反向通信回 source rank。异步情况下，`grad_x` 和
`grad_topk_weights` 都来自同一个 DeepEP backward communication event；如果 `topk_weights` 是非叶子张量并且上游
router backward 会继续消费 `grad_topk_weights`，也必须等待这个 event。当前代码显式给 `hidden_states.grad_fn`
挂了 dispatch backward pre-hook；从 event 语义上看，`topk_weights.grad_fn` 也应等待同一个 dispatch backward
完成事件，除非实现改成在 composite op 内部统一保证两个返回梯度被消费前已经同步。

### DeepEP + ExpertTP 的方案

当前 XTuner 的 `DeepEPDispatcher` 没有接入 `tp_group`；`dispatcher="deepep"` 时不会自动获得
`TorchAll2AllTPEPDispatcher` 那套 TP AllGather / TP ReduceScatterSum。因此 DeepEP + ExpertTP 还需要单独设计。

如果保留 DeepEP 的“`topk_weights` 发到 expert 侧并在 combine 前加权”的语义，那么混合 ExpertTP 后 dispatch
阶段应当这样对齐：

```text
DeepEP dispatch:
  recv_x, recv_topk_idx, recv_topk_weights

TP dispatch segment:
  recv_x            TP AllGather
  recv_topk_idx     TP AllGather
  recv_topk_weights TP AllGather

dispatch_postprocess:
  基于 TP-gather 后的 recv_topk_idx 做 local expert layout

combine_preprocess:
  基于 TP-gather 后的 recv_topk_weights 做 topK folding
```

这会让 DeepEP + ExpertTP 出现和 Naive + ExpertTP 相同的 `topk_weights` TP AllGather backward 问题：
`recv_topk_weights` 的 TP AllGather backward 需要 TP ReduceScatterSum，得到本 TP rank 的
`grad_recv_topk_weights` 后，DeepEP dispatch backward 再用 `Buffer.combine(..., topk_weights=grad_recv_topk_weights)`
把梯度送回 source rank。

推荐的 event 方案是把 DeepEP dispatch 和后续 TP AllGather 封装成一个 dispatch-level composite autograd stage：

1. 前向在同一个 dispatch 通信段中排队 DeepEP dispatch、TP hidden AllGather、TP metadata AllGather 和
   TP `topk_weights` AllGather，只在最后记录一个 dispatch `forward_finished_event`。
2. 反向先完成 TP `topk_weights` / hidden 的 ReduceScatterSum，再调用 DeepEP dispatch backward，把
   `grad_x` 和 `grad_topk_weights` 都送回 source rank。
3. 只有当 hidden 和 `topk_weights` 两条反向通信都完成后，才记录同一个 dispatch `backward_finished_event`。
4. 如果实现上仍拆成多个 autograd Function，则必须给 `topk_weights` 分支保留独立完成 event，并让
   `topk_weights.grad_fn` 的 pre-hook 等待它；否则 router backward 可能在 TP/DeepEP 通信仍在写
   `grad_topk_weights` 时提前读取。

## 前向 event 边界

ExpertTP 的通信阶段应和 All2All dispatcher 保持同一套六阶段边界：

1. `dispatch_preprocess`：本地准备 dispatch 输入，并在 compute stream 上记录 `forward_finished_event`。
2. `dispatch`：在 dispatcher 的通信 stream 上发起 TP AllGather。
3. `dispatch_postprocess`：compute stream 等待 dispatch 的 `forward_finished_event`，再做本地 expert layout。
4. `combine_preprocess`：本地 topK folding，并在 compute stream 上记录 `forward_finished_event`。
5. `combine`：在 dispatcher 的通信 stream 上发起 TP ReduceScatterSum。
6. `combine_postprocess`：compute stream 等待 combine 的 `forward_finished_event`，再返回本 rank 的 source token slice。

同一个通信阶段里的多个 NCCL collective 如果都排在同一条 communication stream 上，阶段内部不需要额外 event 串行化。
例如 Naive + ExpertTP 的 dispatch 会依次发起：

```text
hidden_states TP AllGather
topk_ids      TP AllGather
topk_weights  TP AllGather
```

它们都 enqueue 到同一条 communication stream，CUDA stream FIFO 已经保证顺序。因此前向只需要：

- 阶段开始：communication stream 等待上一阶段的 `forward_finished_event`。
- 阶段结束：最后一个 collective 后记录本阶段的 `forward_finished_event`。
- 本地 postprocess：compute stream 等待本阶段的 `forward_finished_event`。

## 反向 `topk_weights` event

反向也应尽量保持“一阶段一组 event”的模型：

- `backward_previous_event`：下游本地 backward 已经产出这个通信阶段需要的梯度。
- `backward_finished_event`：该通信阶段的 backward collective 已完成，上游可以继续消费梯度。

但 Naive + ExpertTP 的 dispatch 有一个细节：`hidden_states` 和 `topk_weights` 都经过 TP AllGather，且二者都是带梯度的输入。
如果实现上把它们拆成两个独立 autograd Function，那么反向会形成两条独立分支：

```text
dP = TPReduceScatterSum.backward(dO)

dE, dW_full = combine_preprocess.backward(dP)

dH_full = dispatch_postprocess.backward(dE)

dhidden = TPAllGather(hidden_states).backward(dH_full)
dweight = TPAllGather(topk_weights).backward(dW_full)
```

其中 `topk_weights` 的本地梯度 `dweight` 不是纯本地计算得到的，而是由
`TPAllGather(topk_weights).backward()` 在 communication stream 上执行 TP ReduceScatterSum 后得到。

如果没有给 `topk_weights` 上游 autograd 节点单独挂一个等待通信完成的 event，可能出现：

```text
compute stream: topk_weights 上游 backward 读取 dweight
comm stream:    TP ReduceScatterSum 仍在写 dweight
```

这就是跨 stream 读写 race。`hidden_states` 分支的 dispatch backward event 不能证明 `topk_weights` 分支的
TP ReduceScatterSum 已完成，因为两者是独立 autograd Function，完成顺序由 autograd 调度和 CUDA 队列共同决定。

因此，在当前“每个 TP collective 一个 autograd Function”的实现下：

- 前向 dispatch 内部的中间 event 可以省掉，依靠同一条 communication stream 的 FIFO 顺序。
- `topk_weights` backward 仍需要自己的完成 event，并让 `topk_weights.grad_fn` 的 pre-hook 等待该 event 后再继续上游 backward。

如果未来把 Naive + ExpertTP dispatch 封装成一个 dispatch-level composite autograd Function，由它同时管理
`hidden_states` / `topk_ids` / `topk_weights` 的通信和反向，那么可以在这个 composite op 内部统一使用一组 stage-level
backward event：只有在 hidden 和 topk_weights 两条反向 collective 都已正确排队并完成后，才记录同一个
`backward_finished_event`。

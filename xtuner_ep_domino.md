# XTuner 中 Domino EP 的原理和实现

本文只梳理当前 XTuner 已有实现，重点解释 `intra_layer_micro_batch=2` 时，
`MoE._micro_batch_forward` 和 `MoEDecoderLayer._micro_batch_forward` 如何把 MoE 层中的 EP 通信拆出来，
并用异步通信和 autograd hook 在前向/反向中尝试做计算通信重叠。

相关背景：

- EP 单个 micro batch 的 dispatch/combine 数据流见 `xtuner_ep_dispatcher.md`。
- TP 中专家权重切分和 TP collectives 的背景见 `TP.md`。
- Domino 论文（https://arxiv.org/html/2409.15241v1）的核心思想是把一个 batch 沿无依赖维度切成多个独立片段，
  再把这些片段的通信和计算流水起来，从而隐藏通信开销。XTuner 这里采用的是面向 MoE EP 的变种：
  切的是 layer 内的 micro batch，通信对象从 TP AllReduce 变成 EP dispatch/combine。

## 1. 原版 Domino 论文中的 TP 流程和实现

原版 Domino 主要针对 dense Transformer 的 TP AllReduce。论文把 self-attention 和 MLP 都抽象成两段线性计算：

```text
X -> A -> B -> AllReduce
```

在 Megatron-LM 风格 TP 中，`A` 做 column parallel，`B` 做 row parallel。每个 TP rank 持有一份
`A_i` 和 `B_i`，本地计算得到一份 partial output，最后通过 AllReduce 恢复完整输出。每个 transformer block
里 self-attention 和 MLP 在前向各有一次 AllReduce，反向也各有一次 AllReduce，所以 TP 通信天然在关键路径上。

Domino 的做法不是改变 TP 的数学等价性，而是在原 TP 切分之上再切出更小的、彼此无依赖的计算单元，
然后把这些计算单元和 AllReduce 流水起来。

### 1.1 输入 batch 维 row split

第一种切法是在输入 `X` 的 batch 维切分。假设切成两块：

```text
X = [X0; X1]
```

因为 batch 维之间没有数据依赖，MLP 的 GeMM、element-wise 激活/dropout，以及 attention 中按 sequence 维做的
softmax，都可以分别在 `X0` 和 `X1` 上独立计算。前向可以调度成：

```text
compute stream:
  attn(X0)  launch AllReduce(attn0)  attn(X1)  launch AllReduce(attn1)
  LN/dropout/residual(X0, X1)
  mlp(X0)   launch AllReduce(mlp0)   mlp(X1)   launch AllReduce(mlp1)

comm stream:
            AllReduce(attn0) ----->            AllReduce(attn1) ----->
                                           AllReduce(mlp0) ----->      AllReduce(mlp1) ----->
```

这里的重点是：

- `AllReduce(attn0)` 可以和 `attn(X1)` 重叠。
- `AllReduce(attn1)` 可以和后面的 layernorm、dropout、residual 等本地算子重叠。
- `AllReduce(mlp0)` 可以和 `mlp(X1)` 重叠。
- `AllReduce(mlp1)` 可以和下一层中 `X0` 的计算重叠，因此 row split 同时提供 intra-layer 和 inter-layer 重叠。

论文中提到，batch 维 row split 的通信隐藏比例可以接近 100%。但切得太细会让单个 GeMM 变窄，影响 kernel
效率，所以实际 partition 数需要通过 benchmark/grid search 选。

### 1.2 权重 `B` 的 column split

第二种切法是在第二段权重 `B` 的输出列维切分。假设 `B` 切成两块：

```text
B = [B0, B1]
```

本地可以先算第一半输出，再异步启动这半输出的 AllReduce，同时计算第二半输出：

```text
compute stream:
  Y0 = hidden @ B0  launch AllReduce(Y0)  Y1 = hidden @ B1  launch AllReduce(Y1)  concat(Y0, Y1)

comm stream:
                    AllReduce(Y0) ----->                  AllReduce(Y1) ----->
```

这种切法的总通信量和原始 TP 一样，因为只是把同一个输出 hidden 维拆成多个 piece 后分别 AllReduce。
但它有一个同步边界：下一层或后续算子需要完整 hidden 维，所以必须等所有 piece 都完成并拼回完整输出。
因此 weight column split 主要提供 intra-layer 重叠，不像 input row split 那样自然跨层流水。

实现上，论文没有直接依赖 `torch.cat()` 频繁拼接；它预分配大 buffer，把各个 piece 写到对应位置，以减少额外
GPU 内存分配和 OOM 风险。论文报告这种切法通常隐藏 50% 到 70% 的通信。

### 1.3 hybrid split

第三种是 hybrid split：同时在输入 batch 维切 `X`，并在第二段权重输出列维切 `B`。这样能得到更细粒度的
计算通信流水，同时保持总通信量不变。

hybrid 的依赖继承自 `B` 的 column split：row 维上仍然没有跨 chunk 同步，但 hidden 维 piece 最终必须 concat，
所以整体更偏向 intra-layer 重叠。论文把它作为大模型上的实用方案，因为只切 batch 或只切 hidden 都可能让
kernel shape 太窄。

### 1.4 反向和工程实现

反向大体按前向的相反顺序执行，但 Domino 额外利用两个重叠窗口：

1. 跨 batch chunk 的重叠：例如一个 chunk 的梯度 AllReduce 和另一个 chunk 的本地反向计算重叠。
2. 同一个 chunk 内的 sub-module 重叠：把输入梯度 matmul 和权重梯度 matmul 分开，先启动输入梯度相关通信，
   同时继续计算权重梯度。

论文没有手写完整 backward，因为绕开 PyTorch autograd 会损失高效 kernel。它使用一个 no-op module 保存前向
阶段的异步通信 handle，并在反向图中控制通信何时等待完成。这样既保留 autograd 生成的 kernel，又能把等待点放到
真正消费梯度之前。

此外，Domino 还用固定数量的全局 CUDA streams 承载独立计算单元，避免从 stream pool 反复取 stream 的开销。
配合 `torch.compile()`、CUDA Graph 等优化，可以减少切成小 kernel 后的 launch bubble。

## 2. 原始 EP MoE 的关键路径

对单个 micro batch，一个 MoE decoder layer 的主路径是：

```text
attention + gate
  -> dispatch_preprocess      # 本地按 expert 排序
  -> dispatch                 # EP all2all，把 token copy 发到 expert 所在 rank
  -> dispatch_postprocess     # 接收端再按 local expert 排序
  -> experts grouped GEMM
  -> combine_preprocess       # 恢复 all2all receive 顺序
  -> combine                  # EP all2all，把 expert 输出送回 source rank
  -> combine_postprocess      # 按 topK weight 合并回 token
  -> residual / shared expert
```

如果完全同步执行，两个 EP all2all 都在本层关键路径上：

```text
pre_moe -> dispatch_comm -> expert_compute -> combine_comm -> post_moe
```

这里的 `dispatch_comm` 必须先完成，接收端才能跑本地专家；`combine_comm` 必须完成，source rank 才能得到
本层 MoE 输出。所以单个 micro batch 内部很难把这两段通信藏在自己的后续计算后面。

## 3. XTuner 的 Domino EP 切分单位

训练引擎在 `intra_layer_micro_batch > 1` 时，每次从 `data_batches` 中取出多个 `seq_ctx`：

```text
seq_ctx_list = [seq_ctx0, seq_ctx1]
loss_ctx_list = [loss_ctx0, loss_ctx1]
output = model(seq_ctx=seq_ctx_list, loss_ctx=loss_ctx_list)
loss.backward()
```

模型侧 `xtuner/v1/model/moe/moe.py::MoE._micro_batch_forward` 做两件事：

1. MoE 层之前的 dense 层仍然在 concat 后的大 batch 上执行。
2. 进入第一层 MoE 后，把 hidden states 沿 batch/sequence 维切回两个 micro batch：

```text
hidden_states_list = [hidden0, hidden1]
```

后续每一层 MoE decoder layer 都以这两个独立 hidden state 为输入：

```text
layer_results = decoder_layer(
    hidden0,
    hidden1,
    position_embeddings=[pos0, pos1],
    seq_ctx=[seq_ctx0, seq_ctx1],
)
```

这就是 XTuner 里 Domino EP 的基本独立性来源：`seq_ctx0` 和 `seq_ctx1` 在同一层的 attention、gate、EP dispatch、
expert、combine 都是数学上互不依赖的。实现上不改变路由结果和专家计算，只改变两个 micro batch 的调度顺序。

## 4. 单层内的前向调度

核心代码在 `xtuner/v1/module/decoder_layer/moe_decoder_layer.py::MoEDecoderLayer._micro_batch_forward`。
设 `mb0 = seq_ctx_list[0]`，`mb1 = seq_ctx_list[1]`，当前实现的前向调度可以分成 5 段。

### 4.1 先完成两个 micro batch 的 pre-MoE

第一段循环依次处理 `mb0` 和 `mb1`：

```text
mb0: attention + residual + post_attention_layernorm + gate
mb0: dispatch_preprocess(async_op=True)

mb1: attention + residual + post_attention_layernorm + gate
mb1: dispatch_preprocess(async_op=True)
```

`dispatch_preprocess` 仍是本地操作，主要是按 expert 对 token copy 做 `permute`，生成：

```text
pre_dispatched["hidden_states"]
pre_dispatched["row_id_map"]
pre_dispatched["topk_ids"]
```

当 `async_op=True` 时，它额外记录两个事件：

- `forward_finished_event`：在当前 compute stream 上记录，表示本地 pre-dispatch 已经完成。
- `backward_previous_event`：留给反向使用，表示 dispatch backward 的通信完成点。

注意：当前代码没有在 `mb0` pre-dispatch 后立刻启动 `mb0` 的 dispatch all2all，而是先继续做 `mb1` 的
attention/gate/pre-dispatch。因此这一步主要完成输入切片和前向事件准备。

### 4.2 再依次做 dispatch、expert、combine_preprocess

第二段循环依次处理两个 micro batch：

```text
mb0: dispatch(async_op=True)            # 在 dispatcher 的 comm stream 上发起 EP all2all
mb0: dispatch_postprocess(async_op=True) # compute stream 等 dispatch 完成，再本地重排
mb0: experts grouped GEMM
mb0: combine_preprocess(async_op=True)  # 本地 unpermute，准备 combine all2all

mb1: dispatch(async_op=True)
mb1: dispatch_postprocess(async_op=True)
mb1: experts grouped GEMM
mb1: combine_preprocess(async_op=True)
```

对 `TorchAll2AllDispatcher`，`dispatch(async_op=True)` 会调用 `_AsyncDispatch`：

```text
comm_stream.wait_event(pre_dispatched.forward_finished_event)
EP all2all
forward_finished_event.record(comm_stream)
```

随后 `dispatch_postprocess(async_op=True)` 会在当前 compute stream 等待这个 `forward_finished_event`。
也就是说，当前实现保证同一个 micro batch 的 expert 计算一定在 dispatch all2all 完成后开始。

`combine_preprocess(async_op=True)` 是本地重排：

```text
experts_out --unpermute(row_ids_map)--> pre_combined["hidden_states"]
```

并记录一个新的 `forward_finished_event`，表示 combine 的输入已经准备好。

### 4.3 批量发起两个 combine all2all

第三段循环只负责发起通信，不立刻做最终 postprocess：

```text
mb0: combine(async_op=True)  # 在 comm stream 上发起回程 EP all2all
mb1: combine(async_op=True)
```

对 `TorchAll2AllDispatcher`，`combine(async_op=True)` 会调用 `_AsyncCombine`：

```text
comm_stream.wait_event(pre_combined.forward_finished_event)
EP all2all
forward_finished_event.record(comm_stream)
```

这里是前向中最明确的流水点：两个 `combine` 都先被挂到独立 comm stream 上，当前 compute stream 可以继续往下执行。

### 4.4 combine 通信期间计算 shared experts

如果配置了 shared experts，代码会在 `combine` 已经发起后，计算两个 micro batch 的 shared expert：

```text
mb0: shared_experts(pre_moe_forward_out0)
mb1: shared_experts(pre_moe_forward_out1)
```

因此前向中可见的主要重叠是：

```text
comm stream   : combine(mb0) -> combine(mb1)
compute stream: shared_expert(mb0) -> shared_expert(mb1)
```

如果 `n_shared_experts=0`，这一段为空，`combine` 之后会很快进入 `combine_postprocess` 的等待，前向可隐藏的
通信就会少很多。

### 4.5 等 combine 完成并做 post-MoE

最后一段依次完成两个 micro batch：

```text
mb0: combine_postprocess(async_op=True)
mb0: _post_moe_forward(...)

mb1: combine_postprocess(async_op=True)
mb1: _post_moe_forward(...)
```

`combine_postprocess(async_op=True)` 会先让 compute stream 等待 `combine.forward_finished_event`，再做：

```text
combined["hidden_states"]
  --unpermute(pre_dispatched["row_id_map"], probs=topk_weights)-->
post_combined["hidden_states"]
```

这一步把 `[N * topK, hidden]` 的 expert 输出按最初的 topK token copy 顺序 gather 回来，乘以
`topk_weights` 后对 topK 求和，恢复成 `[N, hidden]`。随后 `_post_moe_forward` 加上 shared expert 输出和
residual，得到本层输出。

## 5. `intra_layer_micro_batch=2` 的前向时间线

这一节不能简单理解成“CPU 先调用什么，GPU 就一定先执行什么”。CUDA kernel/collective 的 launch 只是把操作放进
某个 stream 的队列：

- 同一个 stream 内部保持 FIFO 顺序。
- 不同 stream 之间没有天然先后关系。
- 跨 stream 的先后只由 `cudaEventRecord` / `cudaStreamWaitEvent` 这类 event 操作建立。

因此，下面更准确地分成两层：CPU 侧调用顺序，以及 CUDA stream 上由 event 建立的偏序。
表中的 `wait x` 表示 CPU 在对应 CUDA stream 上插入 `cudaStreamWaitEvent(x)`，不是 CPU 阻塞等待
这个 event 完成。

### 5.1 图一：CPU/host 侧顺序

`MoEDecoderLayer._micro_batch_forward` 在 host 侧大致按下面顺序调用：
表中加粗的 `A/D/E/C/S` 是相对耗时大的主算子，后续时间线主要围绕它们观察重叠。


| CPU/host 操作                                                                                               |
| ------------------------------------------------------------------------------------------------------------- |
| **`A0`** -> `Dpre0` -> `record Fa0`                                                                         |
| **`A1`** -> `Dpre1` -> `record Fa1`                                                                         |
| `wait Fa0` -> **`D0`** -> `record Fb0`; `wait Fb0` -> `Dpost0` -> **`E0`** -> `Cpre0` -> `record Fc0`     |
| `wait Fa1` -> **`D1`** -> `record Fb1`; `wait Fb1` -> `Dpost1` -> **`E1`** -> `Cpre1` -> `record Fc1`     |
| `wait Fc0` -> **`C0`** -> `record Fd0`                                                                      |
| `wait Fc1` -> **`C1`** -> `record Fd1`                                                                      |
| **`S0`** -> **`S1`**                                                                                        |
| `wait Fd0` -> `Cpost0`                                                                                      |
| `wait Fd1` -> `Cpost1`                                                                                      |

其中：

- `A{i}`：第 `i` 个 micro batch 的 attention + gate，即 `_pre_moe_forward`。
- `Dpre{i}`：`dispatch_preprocess`，本地 permute。
- `D{i}`：`dispatch`，EP all2all。
- `Dpost{i}`：`dispatch_postprocess`，接收端本地按 local expert 重排。
- `E{i}`：本地 experts grouped GEMM。
- `Cpre{i}`：`combine_preprocess`，本地 unpermute。
- `C{i}`：`combine`，EP all2all。
- `S{i}`：shared experts；如果 `n_shared_experts=0`，这一段不存在。
- `Cpost{i}`：`combine_postprocess + _post_moe_forward`。
- `Fa{i}`：`Dpre{i}` 在 compute stream 上完成后记录，`D{i}` 在 comm stream 上等待它。
- `Fb{i}`：`D{i}` 在 comm stream 上完成后记录，`Dpost{i}` 在 compute stream 上等待它。
- `Fc{i}`：`Cpre{i}` 在 compute stream 上完成后记录，`C{i}` 在 comm stream 上等待它。
- `Fd{i}`：`C{i}` 在 comm stream 上完成后记录，`Cpost{i}` 在 compute stream 上等待它。

这里的 `wait Fa0 -> D0 -> record Fb0; wait Fb0 -> Dpost0 -> E0 -> Cpre0 -> record Fc0` 是 CPU 连续调用；
`Dpost0` 内部会先在 compute stream 上发起
`wait Fb0`，所以 GPU 上的 `Dpost0/E0/Cpre0` 仍必须等 comm stream 上的 `D0` 完成。`D1` 同理。

但这个 host 顺序不能直接当作 GPU 执行顺序。例如 CPU 上先在 compute stream 上发起 `A1/Dpre1`，再在
comm stream 上发起 `D0`，并不意味着 `D0` 一定在 `A1/Dpre1` 之后执行。`D0` 只需要等待 `Dpre0` 后记录的
event；如果 `Dpre0` 已完成，而 `A1/Dpre1` 还在 compute stream 中排队或执行，`D0` 就可能和
`A1/Dpre1` 重叠。

### 5.2 图二：CUDA stream 上的实际依赖顺序

对 `TorchAll2AllDispatcher`，CUDA 侧更接近下面这张图。这里画的是 event 约束下的一种典型执行偏序，
不是一个所有机器都完全相同的绝对时间轴。

`record Fa0` 表示在 compute stream 上记录 `mb0` 的 `dispatch_preprocess.forward_finished_event`，
`wait Fa0` 表示 comm stream 等这个 event。其他 event 同理。


| 计算 stream                                                                       | 通信 stream                                  |
| ----------------------------------------------------------------------------------- | ---------------------------------------------- |
| **`A0`**                                                                          |                                              |
| `Dpre0` -> `record Fa0`                                                           |                                              |
| **`A1`**                                                                          | `wait Fa0` -> **`D0`** -> `record Fb0`      |
| `Dpre1` -> `record Fa1`                                                           |                                              |
| `wait Fb0` -> `Dpost0`                                                            |                                              |
| **`E0`** -> `Cpre0` -> `record Fc0`                                               | `wait Fa1` -> **`D1`** -> `record Fb1`      |
| `wait Fb1` -> `Dpost1` -> **`E1`** -> `Cpre1` -> `record Fc1`                     | `wait Fc0` -> **`C0`** -> `record Fd0`      |
| **`S0`** -> **`S1`**                                                              | `wait Fc1` -> **`C1`** -> `record Fd1`      |
| `wait Fd0` -> `Cpost0`                                                            |                                              |
| `wait Fd1` -> `Cpost1`                                                            |                                              |

同一行两列表示这两个 stream 上的操作可以重叠；长通信可能延续到后面的行。每一行到下一行的顺序只表达同一
stream FIFO 或 event 约束能保证的偏序。为避免表格过长，主算子和紧邻的 event `record/wait` 写在同一个
单元格里，单元格内部按左到右顺序执行。

如果没有 shared experts，则 compute stream 中的 **`S0`** -> **`S1`** 为空，`record Fc1` 后会直接进入 `wait Fd0`。

从这个依赖图可以看出：

- `D0` 只依赖 `Fa0`，不依赖 `Fa1`。所以即使 CPU 是在 `A1/Dpre1` launch 之后才调用 `dispatch(mb0)`，
  CUDA 上 `D0` 仍然可以在 `A1/Dpre1` 完成前开始。
- `D1` 依赖 `Fa1`，并且因为和 `D0` 在同一个 comm stream 上，所以不能越过 `D0`。一旦 `D0` 完成且 `Fa1`
  已记录，`D1` 可以和 compute stream 上的 `E0/Cpre0` 重叠。
- `C0` 只依赖 `Fc0`，不依赖 `Fc1`。虽然 CPU 是在两个 micro batch 的 `Cpre` 都调用完以后才进入
  `combine` 循环，但 CUDA 上 `C0` 可以在 `Dpost1/E1/Cpre1` 完成前执行，因为 `Fc0` 早在 `Cpre0` 后就记录了。
- `C1` 依赖 `Fc1`，并且在同一个 comm stream 上排在 `C0` 后面。它可以和 **`S0`**/**`S1`**、甚至 `Cpost0` 的一部分重叠；
  `Cpost1` 必须等 `Fd1`。

因此，前向的重叠不应理解成一条严格线性的时间轴，而应理解成 event 约束下的跨 stream 流水：

- `dispatch` 的 `D0` 可以覆盖 `A1/Dpre1`，`D1` 可以覆盖 `E0/Cpre0`。
- `combine` 的 `C0` 可以覆盖 `Dpost1/E1/Cpre1`，`C1` 还可以覆盖 shared expert 和后续 postprocess 的一部分。
- 当前代码仍会在 `dispatch_postprocess` / `combine_postprocess` 处插入 compute stream 对对应通信完成 event 的等待，
  所以每个 micro batch 真正消费通信结果前仍有明确同步点。
- 这种实现仍保留了 Domino 的关键前提：两个 micro batch 沿 batch/sequence 维独立，通信和计算可以用事件显式串依赖。

### 5.3 图三：CPU 与 CUDA stream 合并表

下表第一列是严格 CPU 时间轴，行内容和 5.1 的单列表一致。第二、三列展示这一 CPU 步之后，
compute/comm stream 上已经允许出现的操作。某个 GPU 操作可以出现在其 CPU 行之后的后续行；
这样才能表达 CUDA 异步执行导致的计算通信重叠。


| CPU/host 严格时间轴                                                                                         | 计算 stream                                                                       | 通信 stream                                  |
| ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------- |
| **`A0`** -> `Dpre0` -> `record Fa0`                                                                         |                                                                                   |                                              |
| **`A1`** -> `Dpre1` -> `record Fa1`                                                                         | **`A0`** -> `Dpre0` -> `record Fa0`                                               |                                              |
| `wait Fa0` -> **`D0`** -> `record Fb0`; `wait Fb0` -> `Dpost0` -> **`E0`** -> `Cpre0` -> `record Fc0`     | **`A1`** -> `Dpre1` -> `record Fa1`                                               | `wait Fa0` -> **`D0`** -> `record Fb0`      |
| `wait Fa1` -> **`D1`** -> `record Fb1`; `wait Fb1` -> `Dpost1` -> **`E1`** -> `Cpre1` -> `record Fc1`     | `wait Fb0` -> `Dpost0` -> **`E0`** -> `Cpre0` -> `record Fc0`                    | `wait Fa1` -> **`D1`** -> `record Fb1`      |
| `wait Fc0` -> **`C0`** -> `record Fd0`                                                                      | `wait Fb1` -> `Dpost1` -> **`E1`** -> `Cpre1` -> `record Fc1`                    | `wait Fc0` -> **`C0`** -> `record Fd0`      |
| `wait Fc1` -> **`C1`** -> `record Fd1`                                                                      |                                                                                   |                                              |
| **`S0`** -> **`S1`**                                                                                        | **`S0`** -> **`S1`**                                                              | `wait Fc1` -> **`C1`** -> `record Fd1`      |
| `wait Fd0` -> `Cpost0`                                                                                      | `wait Fd0` -> `Cpost0`                                                            |                                              |
| `wait Fd1` -> `Cpost1`                                                                                      | `wait Fd1` -> `Cpost1`                                                            |                                              |

## 6. 反向中的事件链

反向不在 `MoEDecoderLayer._micro_batch_forward` 里手写循环，而是通过 dispatcher 的 autograd `Function` 和
hook 串起依赖。以 `TorchAll2AllDispatcher` 为例，前向 `async_op=True` 会布置四类事件：

```text
dispatch_preprocess.forward_finished_event
dispatch.backward_previous_event
combine_preprocess.forward_finished_event
combine.backward_previous_event
```

它们在反向中的含义和前向相反：

1. `combine_postprocess` 的 backward hook 在当前 compute stream 上记录 `combine.backward_previous_event`，
   表示 `combine` 的反向通信输入梯度已经准备好。
2. `_AsyncCombine.backward` 在 comm stream 上等待 `combine.backward_previous_event`，
   然后执行 forward combine 的反向 all2all；完成后记录 `combine_preprocess.backward_previous_event`。
3. `combine_preprocess` 的 backward pre-hook 让当前 compute stream 等
   `combine_preprocess.backward_previous_event`，确保 expert 输出梯度已经从 comm stream 回来，然后才继续专家反向。
4. `dispatch_postprocess` 的 backward hook 在 expert 反向结束后记录 `dispatch.backward_previous_event`。
5. `_AsyncDispatch.backward` 在 comm stream 上等待这个事件，执行 forward dispatch 的反向 all2all；
   完成后记录 `dispatch_preprocess.backward_previous_event`。
6. `dispatch_preprocess` 的 backward pre-hook 等 `dispatch_preprocess.backward_previous_event`，
   然后才把梯度传回 pre-MoE 的 attention/gate 部分。

反向单个 micro batch 的依赖关系可以写成：

```text
grad Cpost
  -> combine_postprocess backward
  -> [comm stream] combine backward all2all
  -> combine_preprocess backward
  -> experts backward
  -> dispatch_postprocess backward
  -> [comm stream] dispatch backward all2all
  -> dispatch_preprocess backward
  -> pre_moe backward
```

## 7. `intra_layer_micro_batch=2` 的反向重叠

反向同样不能只看 CPU/autograd 的调用顺序。autograd engine 在 host 上访问到某个 backward node 时，只是向当前
compute stream 或 dispatcher 的 comm stream 继续写入待执行操作。真正的 GPU 先后关系仍然由同 stream FIFO 和
event 决定。
本节表格里的 `wait Ba*` / `wait Bb*` / `wait Bc*` / `wait Bd*` 也表示向 CUDA stream 插入 event wait，
不表示 host 线程同步等待。

下面用一个例子画图：假设 autograd 先处理 `mb1` 的 combine 反向，再处理 `mb0` 的 combine 反向。
如果 autograd 实际遍历顺序相反，comm stream 上同类通信的排队顺序也会相反。

### 7.1 图一：CPU/autograd 侧顺序

CPU/autograd 侧看到的是 backward node 的遍历顺序：
表中加粗的 `A/D/E/C/S` 同样表示反向中相对耗时大的主算子。


| CPU/autograd 操作示例                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------- |
| `Cpost1_bwd` -> `record Bd1`; `wait Bd1` -> **`C1_bwd`** -> `record Bc1`                                                   |
| `Cpost0_bwd` -> `record Bd0`; `wait Bd0` -> **`C0_bwd`** -> `record Bc0`                                                   |
| `wait Bc1` -> `Cpre1_bwd` -> **`E1_bwd`** -> `Dpost1_bwd` -> `record Bb1`; `wait Bb1` -> **`D1_bwd`** -> `record Ba1`     |
| `wait Bc0` -> `Cpre0_bwd` -> **`E0_bwd`** -> `Dpost0_bwd` -> `record Bb0`; `wait Bb0` -> **`D0_bwd`** -> `record Ba0`     |
| `wait Ba1` -> `Dpre1_bwd` -> **`A1_bwd`**                                                                                  |
| `wait Ba0` -> `Dpre0_bwd` -> **`A0_bwd`**                                                                                  |

其中：

- `Ba{i}` 和前向 `Fa{i}` 对应：`D{i}_bwd` 在 comm stream 上完成后记录，`Dpre{i}_bwd` 在 compute stream 上等待它。
- `Bb{i}` 和前向 `Fb{i}` 对应：`Dpost{i}_bwd` 在 compute stream 上完成后记录，`D{i}_bwd` 在 comm stream 上等待它。
- `Bc{i}` 和前向 `Fc{i}` 对应：`C{i}_bwd` 在 comm stream 上完成后记录，`Cpre{i}_bwd` 在 compute stream 上等待它。
- `Bd{i}` 和前向 `Fd{i}` 对应：`Cpost{i}_bwd` 在 compute stream 上完成后记录，`C{i}_bwd` 在 comm stream 上等待它。

这张图仍然只是 CPU 发起顺序，不等价于 CUDA 实际执行顺序。比如 CPU 先发起 `C1_bwd`，后发起某些
compute stream 上的 `Cpost0_bwd`，只要 `Bd1` 已经被记录，`C1_bwd` 就可以在 `Cpost0_bwd` 还没完成时开始。

### 7.2 图二：CUDA stream 上的实际依赖顺序

在上述 autograd 发起顺序下，CUDA 侧更接近下面这张 event 依赖图：


| 计算 stream                                                                                         | 通信 stream                                  |
| ------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `Cpost1_bwd` -> `record Bd1`                                                                          |                                              |
| `Cpost0_bwd` -> `record Bd0`                                                                          | `wait Bd1` -> **`C1_bwd`** -> `record Bc1`  |
| `wait Bc1` -> `Cpre1_bwd` -> **`E1_bwd`** -> `Dpost1_bwd` -> `record Bb1`                              | `wait Bd0` -> **`C0_bwd`** -> `record Bc0`  |
| `wait Bc0` -> `Cpre0_bwd` -> **`E0_bwd`** -> `Dpost0_bwd` -> `record Bb0`                              | `wait Bb1` -> **`D1_bwd`** -> `record Ba1`  |
| `wait Ba1` -> `Dpre1_bwd` -> **`A1_bwd`**                                                            | `wait Bb0` -> **`D0_bwd`** -> `record Ba0`  |
| `wait Ba0` -> `Dpre0_bwd` -> **`A0_bwd`**                                                            |                                              |

同一行两列表示可重叠窗口；长通信可能延续到后面的行。每个 `wait Ba*` / `wait Bc*` 都位于对应
`record Ba*` / `record Bc*` 同一行或之后，每个 `wait Bb*` / `wait Bd*` 都位于对应
`record Bb*` / `record Bd*` 同一行或之后。为避免表格过长，主算子和紧邻
的 event `record/wait` 写在同一个单元格里，单元格内部按左到右顺序执行。

上图只表达 event 约束下的一种可能执行。两个 micro batch 之间没有额外的显式 event 依赖，除了共享同一条
`comm_stream`，因此通信在 comm stream 上按发起顺序串行执行。这个发起顺序由 autograd 实际遍历到
backward node 的顺序决定，不能仅凭 `hidden0, hidden1` 的返回顺序推断。若 autograd 先发起 `mb0` 的
`C0_bwd`，再发起 `mb1` 的 `C1_bwd`，则 comm stream 上会变成 `C0_bwd -> C1_bwd`。

### 7.3 图三：前向/反向六列对齐视图

下表把 5.3 的前向三列表和 7.2 的反向 stream 表放在一起。前三列按前向实际时间正序排列；
后三列把反向 GPU 时间线按实际执行的相反方向排列，并尽量让第 2/3 列和第 5/6 列的主算子在同一行：
**`A`** 对 **`A_bwd`**，**`D`** 对 **`D_bwd`**，**`E`** 对 **`E_bwd`**，**`C`** 对 **`C_bwd`**。
第 4 列是反向 CPU/autograd 的对应阶段，它相对第 1 列整体滞后一行；第 4 列内部仍保持“对应前向阶段”的顺序。

注意：第 5/6 列是反向实际执行顺序的反向视图，所以其中 event 的 `wait/record` 在视觉上可能和 7.2 的正向
反向时间线相反；严格 event 约束以 7.2 为准。


| 前向 CPU/host 严格时间轴                                                                                      | 前向计算 stream                                                                  | 前向通信 stream                                 | 反向 CPU/autograd 对应阶段（滞后）                                                                                                               | 反向计算 stream（逆序，对齐前向 GPU）                                                         | 反向通信 stream（逆序，对齐前向 GPU）             |
| -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **`A0`** -> `Dpre0` -> `record Fa0`                                                                         |                                                                                  |                                                 |                                                                                                                                                   |                                                                                               |                                                   |
| **`A1`** -> `Dpre1` -> `record Fa1`                                                                         | **`A0`** -> `Dpre0` -> `record Fa0`                                             |                                                 | `wait Ba0` -> `Dpre0_bwd` -> **`A0_bwd`**                                                                                                       | `wait Ba0` -> `Dpre0_bwd` -> **`A0_bwd`**                                                   |                                                   |
| `wait Fa0` -> **`D0`** -> `record Fb0`; `wait Fb0` -> `Dpost0` -> **`E0`** -> `Cpre0` -> `record Fc0`      | **`A1`** -> `Dpre1` -> `record Fa1`                                             | `wait Fa0` -> **`D0`** -> `record Fb0`         | `wait Ba1` -> `Dpre1_bwd` -> **`A1_bwd`**                                                                                                       | `wait Ba1` -> `Dpre1_bwd` -> **`A1_bwd`**                                                   | `wait Bb0` -> **`D0_bwd`** -> `record Ba0`       |
| `wait Fa1` -> **`D1`** -> `record Fb1`; `wait Fb1` -> `Dpost1` -> **`E1`** -> `Cpre1` -> `record Fc1`      | `wait Fb0` -> `Dpost0` -> **`E0`** -> `Cpre0` -> `record Fc0`                  | `wait Fa1` -> **`D1`** -> `record Fb1`         | `wait Bc0` -> `Cpre0_bwd` -> **`E0_bwd`** -> `Dpost0_bwd` -> `record Bb0`; `wait Bb0` -> **`D0_bwd`** -> `record Ba0`                            | `wait Bc0` -> `Cpre0_bwd` -> **`E0_bwd`** -> `Dpost0_bwd` -> `record Bb0`                | `wait Bb1` -> **`D1_bwd`** -> `record Ba1`       |
| `wait Fc0` -> **`C0`** -> `record Fd0`                                                                      | `wait Fb1` -> `Dpost1` -> **`E1`** -> `Cpre1` -> `record Fc1`                  | `wait Fc0` -> **`C0`** -> `record Fd0`         | `wait Bc1` -> `Cpre1_bwd` -> **`E1_bwd`** -> `Dpost1_bwd` -> `record Bb1`; `wait Bb1` -> **`D1_bwd`** -> `record Ba1`                            | `wait Bc1` -> `Cpre1_bwd` -> **`E1_bwd`** -> `Dpost1_bwd` -> `record Bb1`                | `wait Bd0` -> **`C0_bwd`** -> `record Bc0`       |
| `wait Fc1` -> **`C1`** -> `record Fd1`                                                                      |                                                                                  |                                                 | `wait Bd0` -> **`C0_bwd`** -> `record Bc0`                                                                                                     |                                                                                               |                                                   |
| **`S0`** -> **`S1`**                                                                                        | **`S0`** -> **`S1`**                                                            | `wait Fc1` -> **`C1`** -> `record Fd1`         | `wait Bd1` -> **`C1_bwd`** -> `record Bc1`                                                                                                     | **`S*_bwd`**                                                                                | `wait Bd1` -> **`C1_bwd`** -> `record Bc1`       |
| `wait Fd0` -> `Cpost0`                                                                                      | `wait Fd0` -> `Cpost0`                                                          |                                                 | `S*_bwd`，如果开启 shared experts                                                                                                                | `Cpost0_bwd` -> `record Bd0`                                                               |                                                   |
| `wait Fd1` -> `Cpost1`                                                                                      | `wait Fd1` -> `Cpost1`                                                          |                                                 | `Cpost0_bwd` -> `record Bd0`                                                                                                                    | `Cpost1_bwd` -> `record Bd1`                                                               |                                                   |
|                                                                                                                |                                                                                  |                                                 | `Cpost1_bwd` -> `record Bd1`                                                                                                                    |                                                                                               |                                                   |

shared experts 的反向本地计算没有在上面的 EP dispatcher event 链中单独展开；如果开启 `n_shared_experts`，
`S*_bwd` 也是 compute stream 上的耗时计算，能否覆盖某段 EP 通信取决于 autograd 对 shared 分支和 MoE 分支的实际调度。

重叠的关键也在 event：

- 如果 `Bd1` 已经在 compute stream 上记录，而 compute stream 后面还排着 `Cpost0_bwd` 或其他本地反向计算，
  那么 comm stream 上的 `C1_bwd` 可以在这些后续 compute 操作完成前开始。
- compute stream 只有走到 `Cpre1_bwd` 前的 pre-hook 时，才会等待 `Bc1`。因此 `C1_bwd` 的等待点靠近
  梯度真正被消费的位置，而不是通信发起位置。
- `D{i}_bwd` 同理：它等待 `Bb{i}`，但 pre-MoE 的反向只在 `dispatch_preprocess` 的 backward pre-hook 处等待 `Ba{i}`。
- 由于 `C0_bwd/C1_bwd/D0_bwd/D1_bwd` 都在同一条 comm stream 上，较早排队且尚未满足 event 的通信会挡住
  后面通信，后面的通信不能绕过它。这也是判断实际重叠时必须看 event 和 stream 队列的原因。

这里的重叠来自两点：

- comm stream 上的反向 EP all2all 不阻塞 CPU 继续构建/执行其他 autograd 节点。
- compute stream 只在 `combine_preprocess` / `dispatch_preprocess` 的 backward pre-hook 处等待对应事件，
  等待位置尽量靠近梯度真正被消费的地方。

因此，反向比前向更依赖 autograd 图的调度，但事件链的目标很明确：把 `combine` 和 `dispatch` 的反向通信从
compute stream 中剥离出来，让它们尽可能和另一个 micro batch 的本地反向计算重叠。

## 8. TP+EP 情况下的差异

当同时打开 TP 和 EP 时，`build_dispatcher` 会选择 `TorchAll2AllTPEPDispatcher`。它继承 EP-only 的
`dispatch_preprocess`、`dispatch`、`combine`、`combine_postprocess`，只改两处：

1. `dispatch_postprocess`：EP all2all 后先做 TP AllGather，把同一 EP rank 上不同 TP rank 的 token slice 拼成
   `[M_total, hidden]`，再按 local expert 排序给 grouped GEMM。
2. `combine_preprocess`：expert 输出先按 local expert 的 `row_ids_map` unpermute 回 TP AllGather 顺序，再做
   TP ReduceScatterSum，恢复每个 TP rank 自己的 `[M_ep_recv, hidden]`，最后进入 EP combine all2all。

专家权重本身由 `GroupedLinear` 按 TP 切分：

- `fused_w1w3` 是 column parallel。
- `fused_w2` 是 row parallel。

需要注意的是，当前 TPEP dispatcher 的 TP AllGather / ReduceScatterSum 仍是同步实现；`async_op=True` 只复用
EP all2all 的事件链。也就是说，Domino EP 的异步重叠主要作用在 EP dispatch/combine 上，TP collectives 还没有
被同样地放到独立通信 stream 中流水。

## 9. 小结

XTuner 当前 Domino EP 实现可以概括为：

- 用 `intra_layer_micro_batch` 把一个 layer 的输入沿 batch/sequence 维切成多个独立 micro batch。
- 模型级 `MoE._micro_batch_forward` 负责在进入 MoE 层后维护 `hidden_states_list`，逐层调用 decoder layer 的
  micro-batch forward。
- 层级 `MoEDecoderLayer._micro_batch_forward` 负责重新排列单层内两个 micro batch 的 attention/gate、EP
  dispatch、expert、combine、shared expert、postprocess。
- dispatcher 的 `async_op=True` 负责把 EP all2all 放到独立 comm stream，并用 CUDA event 和 autograd hook
  维持正确依赖。
- 前向重叠需要按 event 判断：`D0` 可覆盖 `A1/Dpre1`，`D1` 可覆盖 `E0/Cpre0`，`C0/C1` 可覆盖后续
  compute；但每个 micro batch 在 `dispatch_postprocess` / `combine_postprocess` 消费通信结果前仍会等待。
- 反向通过 `_AsyncDispatch.backward`、`_AsyncCombine.backward` 和 backward hook，把 dispatch/combine 的反向
  all2all 延后到梯度准备好后异步发起，并只在梯度消费点等待，从而给两个 micro batch 之间的反向计算通信重叠留下空间。

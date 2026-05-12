# XTuner FSDP + EP 机制说明

本文说明 XTuner v1 MoE 模型中 FSDP 和 EP 如何配合。EP dispatcher 内部的 token
排序、all2all、combine 细节已经在 `xtuner_ep_dispatcher.md` 中展开，本文只说明这些 dispatcher
步骤在 FSDP 并行体系中的位置和边界。

主要代码入口：

- `xtuner/v1/model/moe/moe.py`
- `xtuner/v1/module/decoder_layer/moe_decoder_layer.py`
- `xtuner/v1/module/grouped_linear/moe_group_linear.py`
- `xtuner/v1/module/dispatcher/torch_all2all.py`

## 1. 并行维度

记：

```text
world_size = 全部训练 rank 数
EP         = ep_size
FSDP       = world_size / EP
E          = n_routed_experts
E_local    = E / EP
```

FSDP + EP 的核心约定是：

- EP 维负责专家归属，不同 EP rank 拥有不同 routed experts。
- FSDP 维负责数据并行和参数切分，同一个 EP rank 列上的 FSDP ranks 拥有同一批专家的不同 FSDP shard。
- 非 expert 参数在 EP 维是 replicated，在 FSDP 维由 FSDP shard。
- expert 参数在 EP 维是 sharded，在 FSDP 维继续被 FSDP shard。

例如 `world_size=8, EP=4` 时，`FSDP=2`，FSDP 模式下的 root mesh 逻辑上是：

```text
mesh shape = (FSDP=2, EP=4)

          ep0   ep1   ep2   ep3
fsdp0      0     1     2     3
fsdp1      4     5     6     7
```

对应的通信组：

```text
EP group:
  fsdp0 行: [0, 1, 2, 3]
  fsdp1 行: [4, 5, 6, 7]

FSDP group:
  ep0 列: [0, 4]
  ep1 列: [1, 5]
  ep2 列: [2, 6]
  ep3 列: [3, 7]
```

也就是说，dispatcher 的 all2all 只发生在同一 FSDP 数据副本内部的 EP group
里；FSDP 的参数 all-gather / reduce-scatter 只发生在同一 EP rank 对应的 FSDP
group 里。

## 2. mesh 建立

### 2.1 `MoE.__init__` 先建立 EP mesh

`MoE.__init__` 在 `config.ep_size > 1` 时先建立一个用于 MoE 模块构造的 mesh：

```python
fsdp_size = world_size // config.ep_size
init_device_mesh(DEVICE, (fsdp_size, config.ep_size), mesh_dim_names=("*.dp", "*.ep"))
```

这一阶段虽然变量名叫 `fsdp_size`，但 mesh 维度名是 `*.dp`。它的作用主要是让模型在
FSDP 之前也能拿到 EP group：

- `GroupedLinear` 构造 expert 参数时要知道 `ep_mesh`。
- `MoEDecoderLayer` 构造 dispatcher 时要传入 `ep_mesh.get_group()`。
- 推理或非 FSDP 运行也可以直接使用这个 EP mesh。

### 2.2 `fully_shard()` 重新建立 FSDP root mesh

训练引擎会在 meta device 上构造模型，然后调用：

```python
model = model.fully_shard(fsdp_cfg)
```

`MoE.fully_shard()` 首先要求：

```python
fsdp_config.ep_size == model.config.ep_size
```

然后 `_init_device_mesh()` 建立真正的 FSDP root mesh：

```python
model_mesh = init_device_mesh(
    DEVICE,
    (FSDP, EP),
    mesh_dim_names=("*.fsdp", "*.ep"),
)
self.fsdp_mesh = model_mesh["*.fsdp"]
self.ep_mesh = model_mesh["*.ep"]
```

这里有一个关键细节：模型在 `__init__` 中已经创建过旧的 `ep_mesh`，而 FSDP 要求参与
组合的 submesh 来自同一个 root mesh。`_init_device_mesh()` 会从新的 `model_mesh`
中访问同名 EP submesh，并检查它和旧 `ep_mesh` 的 rank layout 完全一致，然后把
`self.ep_mesh` 绑定到新的 submesh。这样 FSDP 看到的是同一个 root mesh 下的
`fsdp` 和 `ep` 维。

当前代码中 HSDP 与 EP 不同时支持：

```python
assert fsdp_config.ep_size == 1, "Currently, HSDP requires expert parallel size to be 1"
```

## 3. 参数切分

参数可以分为 expert 参数和非 expert 参数。

### 3.1 expert 参数：EP shard 后再 FSDP shard

routed experts 位于 `MoEBlock`：

```text
MoEBlock.experts.fused_w1w3
MoEBlock.experts.fused_w2
```

它们由 `build_grouped_linear()` 创建。`GroupedLinear.__init__` 先构造全局排布的融合权重：

```text
fused_w1w3.weight: [E * 2 * moe_intermediate_size, hidden_size]
fused_w2.weight:   [E * hidden_size, moe_intermediate_size]
```

如果 `ep_mesh.size() > 1`，权重会被：

```python
distribute_tensor(weight, ep_mesh, [Shard(0)])
```

因为 dim0 按 expert 连续排布，`Shard(0)` 等价于按专家范围切分。每个 EP rank 只保留：

```text
E_local = E / EP
local_expert_start = ep_rank * E_local
local_expert_end   = local_expert_start + E_local
```

本地 shape 变成：

```text
fused_w1w3.weight local: [E_local * 2 * moe_intermediate_size, hidden_size]
fused_w2.weight local:   [E_local * hidden_size, moe_intermediate_size]
```

随后 `MoE.fully_shard()` 对每个 decoder layer 调用 FSDP `fully_shard()`。因此 expert
参数的逻辑布局是：

```text
EP 维:   Shard(0), 不同 EP rank 拥有不同专家
FSDP 维: Shard(0), 同一批本地专家的参数继续被 FSDP 切分
```

前向时，FSDP 在 FSDP group 内 all-gather 当前 layer 的本地专家参数；`GroupedLinear.forward()`
再通过：

```python
weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
weight = weight.view(-1, self.local_out_features, self.local_in_features)
```

把当前 rank 可见的本地 expert 权重交给 grouped GEMM。

### 3.2 非 expert 参数：EP replicated 后再 FSDP shard

非 expert 参数包括：

- embedding、final norm、lm head
- attention、layer norm
- router gate
- shared experts, 如果 `n_shared_experts > 0`

这些参数不是按 expert 归属切开的。开启 EP 时，`MoE.fully_shard()` 会先调用：

```python
self._replicate_other_params(self)
```

该函数递归遍历模型，但遇到 `MoEBlock` 会直接返回，因为 routed expert 参数已经由
`GroupedLinear` 负责 EP 切分。其余参数会被替换为：

```python
distribute_tensor(param, self.ep_mesh, [Replicate()])
```

然后再由 FSDP 在 `fsdp_mesh` 上切分。逻辑布局是：

```text
EP 维:   Replicate(), 每个 EP rank 都有同一份逻辑参数
FSDP 维: Shard(0), FSDP 负责参数分片和梯度同步
```

router gate 也属于这一类。每个 EP rank 都要用完整 gate 权重计算对全部 `E` 个专家的
logits，这样 `topk_ids` 才是全局 expert id，dispatcher 才能按 global expert id 把
token 发到正确的 EP rank。

### 3.3 FSDP 包裹顺序

`MoE.fully_shard()` 的大致顺序是：

1. 初始化 FSDP/EP mesh。
2. 必要时把可训练参数转成 fp32 参数。
3. EP 开启时复制非 expert 参数到 EP 维。
4. 按 layer 逐个调用 `_fully_shard()`，可按 `recompute_ratio` 加 checkpoint wrapper。
5. 对相邻 layer 设置 FSDP forward prefetch。
6. 分别 shard `embed_tokens`、`norm`、`lm_head`。
7. 最后对 root model 调用一次 `_fully_shard()`。
8. 对 embedding patch forward，让 DTensor weight 先 `to_local()` 再进入 `F.embedding()`。
9. `_to_empty_meta()` 只物化本 rank 需要的 local shard。

这种顺序的目标是：构造阶段可以在 meta device 上完成，真正占显存的是 FSDP/EP 切分后的
本地 shard。

## 4. HF 权重加载与保存

`BaseModel._init_load_spec()` 在模型初始化末尾执行。对 MoE 来说，这发生在 EP 参数已经
由 `GroupedLinear` 切好之后、FSDP 切分之前。

因此 load spec 表达的是“EP 切分后、FSDP 切分前”的参数布局。后续 FSDP 再根据
`self.fsdp_mesh` 做第二次 slicing。

### 4.1 fused expert 权重

Qwen3 MoE 的 HF 权重是逐 expert 保存的：

```text
experts.{i}.gate_proj.weight
experts.{i}.up_proj.weight
experts.{i}.down_proj.weight
```

XTuner 内部为了 grouped GEMM 使用融合参数：

```text
fused_w1w3.weight
fused_w2.weight
```

`Qwen3MoE.to_hf_key_list()` 会把一个融合参数映射到多个 HF key。开启 EP 后，
`_init_load_spec()` 看到 expert 参数是 `Shard(0)` DTensor，会根据当前 EP rank 的
global offset 只保留本地专家对应的 HF keys。

开启 FSDP 后，`_load_fused_hf_param()` 再根据：

```python
compute_local_shape_and_global_offset(load_spec.shape, self.fsdp_mesh, [Shard(0)])
```

计算本 FSDP rank 在 EP-local 参数中的 dim0 范围，只加载和拷贝这一段。代码里明确要求：

```python
assert load_spec.dim == self.FSDP_SHARD_DIM
```

也就是当前只支持 FSDP 和专家并行都沿同一个维度切 fused expert 参数。

### 4.2 非 expert 权重

非 expert 参数通常只有一个 HF key。EP 维是 `Replicate()`，所以每个 EP rank 逻辑上加载同一份
参数；FSDP 再按本 rank 的 local offset 取 dim0 shard。

保存 HF 时，fused 参数和普通参数分开处理。fused expert 参数可以由多个 rank 分摊保存，
普通 replicated 参数只需要避免重复写同一个 HF key。

## 5. 前向流程

下面只描述 FSDP 与 EP 的交界，不展开 dispatcher 内部 token 排列。dispatcher 细节见
`xtuner_ep_dispatcher.md`。

### 5.1 layer 进入前

每个 FSDP-wrapped module 前向时，FSDP 会在对应 `fsdp_mesh` group 内 all-gather 当前
module 的参数。对一个 MoE decoder layer 来说：

- attention、norm、gate 等非 expert 参数是 EP replicated + FSDP sharded。
- routed expert 参数是 EP sharded + FSDP sharded。

所以当前 layer 前向开始时，本 rank 可以使用：

- 本 EP rank 对应的完整 local experts 参数。
- 本 EP rank 上 replicated 的非 expert 参数。

这里的“完整”只是在当前 FSDP group 内 all-gather 后完整，不表示跨 EP 收集了所有专家。

### 5.2 `_pre_moe_forward`

`MoEDecoderLayer._pre_moe_forward()` 做三件事：

1. input layernorm。
2. self attention。
3. post attention layernorm + gate。

gate 在每个 EP rank 上都会计算完整的 `[N, E]` router logits，并输出：

```text
topk_ids:     [N, K], global expert id
topk_weights: [N, K]
```

因为 gate 参数是 EP replicated，所以同一个输入 token 在不同 EP rank 上看到的是同一套
router 参数。

### 5.3 dispatcher

之后进入 dispatcher：

```python
pre_dispatched = dispatcher.dispatch_preprocess(...)
dispatched = dispatcher.dispatch(...)
post_dispatched = dispatcher.dispatch_postprocess(...)
```

FSDP + EP 下需要注意两点：

- dispatcher 使用的是 `ep_mesh.get_group()`，只在同一 FSDP 行内做 EP 通信。
- dispatcher 只搬 activation 和 routing 信息，不搬 expert 参数。

经过 `dispatch_postprocess()` 后，每个 EP rank 得到的 hidden states 都已经按本地
experts 排好，并提供：

```text
post_dispatched["hidden_states"]:     [M_local, hidden_size]
post_dispatched["tokens_per_expert"]: [E_local]
```

这里的 `E_local` 正好和当前 EP rank 持有的 local experts 数一致。

### 5.4 local experts grouped GEMM

`MoEBlock.forward()` 只计算本 EP rank 的 local experts：

```python
gate_up_out = self.fused_w1w3(x, tokens_per_expert, decoding)
out = self.moe_act(gate_up_out, split_dim=-1)
res = self.fused_w2(out, tokens_per_expert, decoding)
```

`GroupedLinear.forward()` 取本地权重：

```python
weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
```

然后按：

```text
weight:            [E_local, out_features, in_features]
tokens_per_expert: [E_local]
```

调用 grouped GEMM。由于 dispatcher 已经保证输入按 local expert 连续分组，grouped GEMM
不需要再跨 EP 通信。

### 5.5 combine 和 layer 输出

expert 输出再经过：

```python
pre_combined = dispatcher.combine_preprocess(...)
combined = dispatcher.combine(...)
post_combined = dispatcher.combine_postprocess(...)
```

被送回 token 的 source EP rank，并按 `topk_weights` 合并回 `[N, hidden_size]`。这部分
的行号映射和 all2all 反向 split 见 `xtuner_ep_dispatcher.md`。

如果有 shared experts，它们是非 routed dense MLP，属于 EP replicated + FSDP sharded 参数，
在本 rank 本地计算，不经过 dispatcher。最后：

```python
hidden_states = (routed_out + shared_out) * hidden_factor + residual
```

## 6. 反向流程

反向可以看成前向的逆序。

### 6.1 activation 梯度

`combine_postprocess -> combine -> combine_preprocess` 的 autograd 会把 source token 上的
梯度送回 expert 输出所在的 EP rank。随后 grouped GEMM 计算：

- 对输入 activation 的梯度。
- 对当前 EP rank local expert 参数的梯度。

接着 `dispatch_postprocess -> dispatch -> dispatch_preprocess` 的 autograd 再把 activation
梯度送回原 token 所在 rank。

dispatcher 内部 all2all 的反向通信仍然只在 EP group 内发生，具体顺序见 `xtuner_ep_dispatcher.md`。

### 6.2 expert 参数梯度

expert 参数不是 EP replicated 参数。每个 EP rank 只拥有自己那段 experts，所以不能对
expert 参数在 EP 维 all-reduce。

FSDP 会在同一 EP rank 列对应的 FSDP group 内对 expert 参数梯度做 reduce-scatter。
这会聚合不同 FSDP 数据副本上同一批 local experts 的梯度。

在 `TrainEngine.clip_grad_norm()` 开始时会调用：

```python
self.model.scale_and_reduce_grad()
```

`MoE.scale_and_reduce_grad()` 对 expert 参数有特殊逻辑：

```python
if ep_enabled and ".experts" in name:
    param.grad.div_(self.ep_mesh.size())
    continue
```

它只除以 `EP`，不做 EP all-reduce。原因是 expert 参数在 EP 维不是同一个参数的多个副本；
不同 EP rank 上是不同专家。这里的除法用于抵消全局 loss/backward 在 EP 维带来的重复缩放，
而不是同步专家参数。

### 6.3 非 expert 参数梯度

非 expert 参数在 EP 维是 replicated。FSDP backward 已经处理了 FSDP 维的梯度同步，但
EP 维上的多个 replicas 仍然需要得到一致梯度。

`scale_and_reduce_grad()` 会检查 DTensor placement 中的 `Replicate()` 维度，并在这些维度
上执行平均 all-reduce：

```python
grad.div_(replicate_world_size)
dist.all_reduce(grad, ReduceOp.SUM, group=replicate_group)
```

因此：

- router、attention、norm、embedding、lm head 等 replicated 参数在 EP ranks 上保持一致更新。
- expert 参数不经过这个分支，因为前面已经按 `".experts"` 单独处理。

### 6.4 grad norm 和 clip

所有 micro-batch 都 backward 完之后，训练流程才进入：

```python
grad_norm = engine.clip_grad_norm()
engine.step_optimizer(grad_norm)
```

`clip_grad_norm()` 的顺序是：

1. `model.scale_and_reduce_grad()` 处理 EP expert 缩放和 replicated 参数同步。
2. 收集所有 trainable 参数的 `.grad`。
3. `cal_grad_norm()` 按 DTensor placement 计算全局 grad norm。
4. 如需 clip，对各组梯度乘同一个 clip 系数。

所以 optimizer step 看到的是已经完成 FSDP 同步、EP replicated 参数同步、expert 梯度缩放后的
梯度。

## 7. 关键约束

- `model.config.ep_size` 必须和 `FSDPConfig.ep_size` 一致。Trainer 会在其中一个为 1 时做一次
  自动对齐，`MoE.fully_shard()` 内部仍然会 assert。
- `n_routed_experts % ep_size == 0`，否则 `GroupedLinear` 无法按 EP 均分 experts。
- HSDP 当前要求 `ep_size == 1`，所以不能和 EP 同时使用。
- routed expert 参数的 EP shard 和 FSDP shard 当前都沿 dim0，`BaseModel.FSDP_SHARD_DIM = 0`。
- dispatcher 只处理 activation，不处理参数。参数归属由 `GroupedLinear` 和 FSDP 决定。
- 非 expert 参数必须在 EP 维 replicated，否则不同 EP rank 的 router/attention 等参数会分叉。
- expert 参数不能在 EP 维 all-reduce，因为不同 EP rank 上不是同一批 experts。

## 8. 一句话总结

XTuner 的 FSDP + EP 可以理解为二维并行：

```text
EP 维决定“这个 rank 负责哪些 experts”
FSDP 维决定“这些参数在数据并行副本之间如何切片、all-gather 和 reduce-scatter”
```

前向时 dispatcher 在 EP 维移动 token，FSDP 在 FSDP 维移动参数；反向时 dispatcher 把
activation 梯度送回 token/source 和 expert/destination，FSDP 聚合同一专家 shard 的数据并行
梯度，`scale_and_reduce_grad()` 再补齐 EP 维上 expert 梯度缩放和 replicated 参数同步。

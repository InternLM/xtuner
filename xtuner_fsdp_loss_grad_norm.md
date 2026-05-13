# XTuner FSDP Loss 校准与 Grad Norm 机制

## 背景

XTuner 的 loss 校准目标是：在相同 global batch 下，不论使用多少张卡、是否使用 FSDP、是否使用 SP、以及一个 optimizer step 内拆成多少个 micro-batch，最终用于 optimizer update 的梯度都应等价于单卡一次性计算同一批数据的梯度。

这里有一个关键前提：FSDP 反向阶段对参数梯度做 `ReduceScatter` 时采用的是 reduce mean。也就是说，如果上游 loss 只按普通的全局平均来构造，FSDP 的梯度同步会额外除以 FSDP world size，导致梯度比期望值小。

## 相关代码入口

- 训练前准备 loss ctx：`xtuner/v1/train/trainer.py::_prepare_model_input`
- 模型批量构建并校准 loss ctx：`xtuner/v1/model/base.py::build_loss_ctx_batch`
- CE loss 校准核心：`xtuner/v1/loss/ce_loss.py::LMHeadLossContext.build_batches`
- CE loss 前向与 autograd all-reduce：`xtuner/v1/loss/ce_loss.py::LMHeadLossContext.forward`
- 逐 micro-batch backward：`xtuner/v1/engine/train_engine.py::train_step`
- grad norm/clip：`xtuner/v1/engine/train_engine.py::clip_grad_norm`
- MoE FSDP + EP mesh：`xtuner/v1/model/moe/moe.py::MoE._init_device_mesh`
- MoE EP 参数复制与梯度修正：`xtuner/v1/model/moe/moe.py::_replicate_other_params`、`xtuner/v1/model/moe/moe.py::scale_and_reduce_grad`

## Step 内一次性构建 loss ctx

Trainer 在拿到一个 optimizer step 对应的 `data_batch` 后，会先把每个样本的 `seq_ctx` 移到设备上，并在 SP 开启时切分序列：

```python
if self.sp_mesh.size() > 1:
    seq_ctx = seq_ctx.split(sequence_parallel_mesh=self.sp_mesh)
```

随后调用：

```python
loss_ctx_dict_list = self._engine.model.build_loss_ctx_batch(data_batch, sp_mesh=self.sp_mesh)
```

这里的重点是：loss ctx 不是在每个 micro-batch forward 时临时独立构建，而是对当前 step 的所有 micro-batch 一次性构建并校准。这样梯度累积的分母天然覆盖整个 optimizer step。

## loss weight 的构造

CE loss 使用 `shifted_labels` 和 `loss_weight`。`CELossConfig.loss_reduction` 支持三种模式：

- `token`：每个有效 token 的原始权重为 1。
- `sample`：每个样本内有效 token 的原始权重为 `1 / num_grad_tokens`。
- `square`：每个样本内有效 token 的原始权重为 `1 / sqrt(num_grad_tokens)`。

所有 `label == ignore_idx`，默认 `-100`，的位置都会被置为 0：

```python
loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0
```

SP 下需要注意 `sample` 和 `square`：因为样本被按 sequence 维切到不同 SP rank 上，代码会先 gather 出完整 `shifted_labels` 来统计每个样本真实有效 token 数，再把算好的 `loss_weight` split 回各个 SP rank。

## 全局分母

构造完当前 rank 上、当前 step 内所有 micro-batch 的原始 `loss_weight` 后，XTuner 计算：

```python
rank_denominator = sum(loss_weight.sum() for loss_weight in loss_weight_list)
global_denominator = rank_denominator
if dist.is_initialized():
    dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
```

然后对每个 loss ctx 的权重做归一化：

```python
loss_ctx.loss_kwargs.loss_weight /= global_denominator + 1e-12
```

因此：

- `token` 模式下，`global_denominator` 等价于当前 step 内所有 rank、所有 micro-batch 的有效 token 数。
- `sample/square` 模式下，`global_denominator` 是当前 step 内所有 rank、所有 micro-batch 的原始 loss weight 总和，而不是简单 token 数。

## 本地 loss 计算

前向时，CE loss 先以 `reduction="none"` 算出逐 token loss：

```python
loss = F.cross_entropy(
    logits,
    shifted_labels,
    reduction="none",
    ignore_index=self.loss_cfg.ignore_idx,
)
loss = (loss * loss_weight).sum()
```

由于 `loss_weight` 已经除过 `global_denominator`，这个 `local_loss` 表示当前 rank 当前 micro-batch 对全局 loss 的局部贡献。

`eager`、`chunk`、`liger` 的差异主要在实现方式：

- `eager`：直接算 logits 和 CE。
- `chunk`：按 sequence chunk 计算，降低 lm_head logits 和 CE backward 的显存峰值。
- `liger`：用 fused linear CE，只支持 `token` reduction。

这三种模式的校准目标是一致的。

## autograd all-reduce 与 FSDP reduce mean 的抵消

本地 loss 算完后，XTuner 会在返回前做 autograd 版 all-reduce sum：

```python
if dist.is_initialized():
    loss = all_reduce(loss, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
```

这是 FSDP loss 校准里最关键的一步。

先看没有 EP 的普通 FSDP 情况。

记：

- `N` 为 FSDP mesh size。
- `L_r` 为 rank `r` 上已经除过 `global_denominator` 的本地 loss。
- 期望的全局 loss 为 `L = sum_r L_r`。

如果不做 autograd all-reduce，而是直接对 `L_r` backward，FSDP 反向的 `ReduceScatter(mean)` 会让最终梯度变成期望值的 `1 / N`。

XTuner 做了：

```text
forward:  L = all_reduce_sum(L_r)
backward: dL/dL_r = N
```

于是每个 rank 本地 loss 的反向梯度先被放大 `N` 倍。随后 FSDP `ReduceScatter(mean)` 再除以 `N`。两者抵消后，参数梯度等价于：

```text
sum_over_all_ranks_and_micro_batches grad(ce * raw_loss_weight / global_denominator)
```

也就是单卡一次性在同一 global batch 上计算校准后 loss 的梯度。

开启 EP 后，loss 仍然对 `dist.group.WORLD` 做 autograd all-reduce，但 FSDP 的 `ReduceScatter(mean)` 只发生在同一 EP rank 对应的 FSDP group 内。因此 EP 情况下不能只看这一处抵消，剩余的 EP 缩放会在 `MoE.scale_and_reduce_grad()` 中处理，详见后文。

## 梯度累积

训练循环中，XTuner 对每个 micro-batch 直接执行：

```python
loss.backward()
```

没有再除以 `grad_accumulation_steps`。原因是 `build_batches` 的 `global_denominator` 已经覆盖了当前 optimizer step 内的所有 micro-batch。

因此一个 step 内多个 micro-batch 的 backward 累积结果为：

```text
step_grad =
  sum_over_micro_batches_and_ranks grad(ce * raw_loss_weight / global_denominator)
```

这正是全局 batch 一次性 backward 的结果。

## FSDP + EP 下的 Loss 校准

MoE 开启 EP 后，训练 mesh 可以简化成二维：

```text
F = fsdp_mesh.size()
E = ep_mesh.size()
world_size = F * E
```

忽略 TP 时，逻辑布局类似：

```text
          ep0   ep1   ...   ep(E-1)
fsdp0      *     *            *
fsdp1      *     *            *
...
fsdp(F-1)  *     *            *
```

EP 维负责 expert 归属，FSDP 维负责同一批参数在数据并行副本之间的 shard、all-gather 和 reduce-scatter。

参数分两类：

- routed expert 参数：EP 维 `Shard(0)`，每个 EP rank 只拥有一部分 experts；FSDP 维继续 shard。
- 非 expert 参数：EP 维 `Replicate()`，包括 embedding、attention、norm、router、lm head、shared experts；FSDP 维 shard。

Loss 分母仍然按全局 rank 统计。`LMHeadLossContext.build_batches()` 对当前 step 内所有 micro-batch 构造 raw `loss_weight` 后，直接用默认分布式组做：

```python
dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
```

这意味着 `global_denominator` 覆盖所有 FSDP rank、EP rank 和 micro-batch。EP dispatcher 后续会移动 activation，但 label/loss ctx 仍按 source token 所在 rank 构造；每个 token 在分母中只贡献一次。

前向返回前的 loss 也仍然做 `WORLD` 范围的 autograd all-reduce：

```text
L = sum_{f=0}^{F-1} sum_{e=0}^{E-1} L_{f,e}
```

因此 backward 时，每个本地 `L_{f,e}` 收到的上游缩放是：

```text
world_size = F * E
```

而 FSDP 对参数梯度的 reduce mean 只在 FSDP 维发生，会除以 `F`。所以 FSDP 反向后还会剩下一个 `E` 倍缩放。这个剩余缩放不能在 loss 里统一处理，因为 expert 参数和非 expert 参数在 EP 维的语义不同。

## FSDP + EP 下的 Expert 梯度

routed expert 参数在 EP 维不是副本。不同 EP rank 上是不同专家，所以不能在 EP 维 all-reduce。

前向时，dispatcher 在同一 FSDP 行内把 token 发送到 owning EP rank。本地 expert grouped GEMM 计算当前 EP rank 持有的 `E_local` 个 experts。反向时，dispatcher 的 autograd 会把来自所有 source EP rank 的 token 梯度送回对应 expert owner。因此某个 expert 参数在一个 FSDP 行上已经收到了这一行内所有 EP source token 对它的贡献。

但 loss 的 autograd all-reduce 是 `WORLD` 范围，给每个本地 loss 带来 `F * E` 的 backward 缩放；FSDP reduce mean 只除以 `F`。所以 expert 参数梯度还多了 `E` 倍。

`MoE.scale_and_reduce_grad()` 对 expert 参数的处理是：

```python
if ep_enabled and ".experts" in name:
    param.grad.div_(self.ep_mesh.size())
    continue
```

这里的语义是：

- `div_(E)`：消掉 loss `WORLD` all-reduce 相对 FSDP mean 多出来的 EP 倍数。
- `continue`：不做 EP all-reduce，因为 EP 维上不是同一个参数的多个副本，而是不同 experts。

修正后，expert 参数梯度等价于：

```text
sum_over_fsdp_rows_and_source_ep_ranks grad(local_expert_loss / global_denominator)
```

即该 expert 在整个 global batch 中实际接收到的 token 对它的梯度。

## FSDP + EP 下的 Replicated 参数梯度

非 expert 参数在 EP 维是 replicated，例如 router、attention、norm、embedding、lm head。每个 EP rank 上是同一个逻辑参数的副本，但它们处理的 source token 不同，所以反向后各 EP replica 的梯度先是各自数据切片上的贡献。

对某个 EP rank `e` 的非 expert 参数副本，FSDP reduce mean 后梯度形如：

```text
E * sum_f grad(L_{f,e})
```

还多了一个 `E`。但和 expert 不同，replicated 参数需要聚合所有 EP rank 的数据贡献，并让每个 replica 得到一致梯度。

`MoE.scale_and_reduce_grad()` 会检查 DTensor placement 中的 `Replicate()` 维度，并在 replicate mesh 上做平均 all-reduce：

```python
grad.div_(replicate_world_size)
dist.all_reduce(grad, ReduceOp.SUM, group=replicate_group)
```

对单个 EP replicate 维来说，这等价于：

```text
sum_e (E * sum_f grad(L_{f,e}) / E)
= sum_e sum_f grad(L_{f,e})
```

因此它同时完成两件事：

- 消掉 EP 维多出来的 `E` 倍缩放。
- 聚合所有 EP rank 的数据贡献，使 replicated 参数的各个副本保持一致。

如果一个参数有多个 `Replicate()` 维，代码会 flatten 对应 submesh 后做同样的平均 all-reduce。

## Grad Norm 与 Clip

一个 train step 内所有 micro-batch 都 backward 完后，Trainer 调用：

```python
grad_norm = self._engine.clip_grad_norm(do_clip=self._do_clip, dtype=self._grad_norm_dtype)
self._engine.step_optimizer(grad_norm)
```

`clip_grad_norm` 里会先调用：

```python
self.model.scale_and_reduce_grad()
```

随后收集所有可训练参数的 `.grad`，调用 `cal_grad_norm` 计算全局 grad norm。

对 Dense FSDP 模型，`BaseModel.scale_and_reduce_grad()` 默认是空操作。常规 FSDP 参数的梯度同步已经由 FSDP backward 完成，且 loss 校准已经处理了 reduce mean 的缩放问题。

对 MoE 模型，`MoE.scale_and_reduce_grad()` 会额外处理 EP/replicated 参数：

- expert 参数在 EP 下只除以 `ep_mesh.size()`，不做 EP all-reduce。
- replicated DTensor 参数会在 replicate mesh 上做平均 all-reduce，使这些未按普通 FSDP shard 语义同步的参数也得到一致梯度。

通用 `cal_grad_norm` 会按 DTensor 的 mesh 和 placement 分组计算 norm。对于 sharded placement，会对局部 norm square 做 all-reduce sum，再开方得到全局 norm。这样 clip 使用的是全局参数梯度范数，而不是单 rank 的局部范数。

在 FSDP + EP 下，这个顺序很重要：grad norm 是在 expert 梯度除 EP、replicated 参数 EP 平均 all-reduce 之后计算的。`cal_grad_norm()` 对 `Shard()` 维度累加 norm square，对 `Replicate()` 维度不重复计数。因此：

- expert 参数的 norm 会覆盖所有 EP shard 上的 experts。
- replicated 参数的 norm 只按一份逻辑参数计数，不会因为 EP replica 数量而重复放大。
- clip 系数作用在已经完成 FSDP/EP 校准后的梯度上，optimizer step 看到的是校准后的全局梯度。

## FSDP + EP + expert TP 相对 FSDP + EP 的差异

新增的 TP 指 `MoEConfig.expert_tp_size`，这里称为 `T`。它是 expert tensor parallel，用来切分 routed expert 的 column/row 权重 shard；它和 `FSDPConfig.tp_size` 不是同一个概念。当前语境下，不同 expert TP rank 拿到的是不同数据。

相对 FSDP + EP，mesh 从二维变为三维：

```text
F = fsdp_mesh.size()
E = ep_mesh.size()
T = expert_tp_size
world_size = F * E * T
```

核心差异只有三类。

### 参数布局多了一维 expert TP

FSDP + EP 下，routed expert 参数只在 EP 维切 expert；开启 expert TP 后，同一个 expert 的权重还会在 expert TP 维继续切 shard：

```text
expert weight: EP 切 expert, expert TP 切 column/row, FSDP 继续 shard
```

非 expert 参数在 EP 和 expert TP 维都是 replicated。实现上会把 `EP x expert TP` 子网格 flatten 成一维 replicate mesh，避免 PyTorch FSDP 不支持二维 `Replicate(), Replicate()` TP 布局。

### loss 分母和 autograd all-reduce 覆盖更大的 world

FSDP + EP 下：

```text
L = sum_{f,e} L_{f,e}
backward scale = F * E
FSDP reduce mean 除以 F
剩余缩放 = E
```

FSDP + EP + expert TP 下：

```text
L = sum_{f,e,t} L_{f,e,t}
backward scale = F * E * T
FSDP reduce mean 除以 F
剩余缩放 = E * T
```

因此，所有 EP-only 里出现的剩余 `E`，在 expert TP 开启后都变成 `E * T`。loss 分母仍然按默认分布式组统计，覆盖所有 FSDP rank、EP rank、expert TP rank 和 micro-batch；每个 token 仍只按 source rank 贡献一次。

### expert 与 replicated 参数的梯度修正多乘一个 T

expert 参数在 expert TP 维不是副本，而是同一个 expert 权重的不同 shard。因此它和 EP 维一样，不能 all-reduce 成一份完整梯度，只能消掉多出来的缩放：

```python
if ep_enabled and ".experts" in name:
    param.grad.div_(self.ep_mesh.size() * self.config.expert_tp_size)
    continue
```

相对 EP-only 的 `div_(E)`，这里变成 `div_(E * T)`。

非 expert 参数在 `EP x expert TP` 上是 replica，需要聚合所有 source 数据贡献，并让每个 replica 得到一致梯度。EP-only 是在 EP replicate mesh 上平均 all-reduce；开启 expert TP 后是在 flatten 后的 `EP x expert TP` replicate mesh 上平均 all-reduce：

```text
sum_{e,t} (E * T * sum_f grad(L_{f,e,t}) / (E * T))
= sum_{e,t} sum_f grad(L_{f,e,t})
```

这同时完成两件事：

- 消掉 `E * T` 倍缩放。
- 聚合所有 EP / expert TP rank 的数据贡献。

### grad norm 需要额外覆盖 expert TP shard

FSDP + EP 下，通用 `cal_grad_norm()` 能根据 DTensor placement 汇总 `Shard()` 维的 norm square，并对 `Replicate()` 维不重复计数。

开启 expert TP 后，grouped expert 权重的 EP / expert TP shard 是本地 tensor 布局，并没有编码成 DTensor 的 EP / TP `Shard()` placement。如果继续只用通用逻辑，expert 参数的 global norm 会漏掉跨 `expert_tp_size` 的 norm square 汇总，clip 系数也会偏小或偏大。

因此 MoE 覆盖模型级 `cal_grad_norm()`：在普通 DTensor shard 汇总之外，对 expert 参数的 local norm square 额外沿 `ep_mesh` 和 `tp_mesh` 做 `SUM all_reduce`：

```python
if expert_tp_size > 1 and ".experts" in name:
    dist.all_reduce(local_norm_squared, op=ReduceOp.SUM, group=ep_mesh.get_group())
    dist.all_reduce(local_norm_squared, op=ReduceOp.SUM, group=tp_mesh.get_group())
```

这样 clip 使用的是覆盖所有 EP / expert TP shard 的 expert norm，同时 replicated 参数仍只按一份逻辑参数计数。

## 总结

XTuner FSDP loss 校准可以概括为三步：

1. 在当前 optimizer step 的所有 micro-batch 上构造 raw `loss_weight`，并跨 rank 求 `global_denominator`。
2. 每个 rank/micro-batch 计算 `sum(ce_per_token * raw_loss_weight / global_denominator)`。
3. 对 loss 做 autograd `all_reduce(SUM)`，用其 backward 放大效应抵消 FSDP `ReduceScatter(mean)`。

FSDP + EP 时还要再区分两类参数：

- expert 参数：FSDP mean 后剩余的 EP 倍数通过 `grad.div_(ep_size)` 消掉，不能 EP all-reduce。
- EP replicated 参数：通过 replicate mesh 上的平均 all-reduce 同时消掉 EP 倍数并聚合所有 EP rank 的数据贡献。

FSDP + EP + expert TP 不改变上述主线，只是在 EP 之外多了一维 expert TP：

- expert 参数：剩余缩放从 `E` 变为 `E * T`，通过 `grad.div_(ep_size * expert_tp_size)` 消掉。
- replicated 参数：replicate mesh 从 EP 扩展为 flatten 后的 `EP x expert TP`。
- grad norm：expert shard 没有用 DTensor placement 表达 expert TP shard，因此 MoE 需要额外跨 EP 和 expert TP 汇总 expert norm square。

最终效果是：FSDP、EP、SP、梯度累积和不同卡数不应改变同一 global batch 对参数更新的数学含义；grad norm/clip 发生在所有 micro-batch backward 完成之后，基于已经校准和同步后的全局梯度计算。

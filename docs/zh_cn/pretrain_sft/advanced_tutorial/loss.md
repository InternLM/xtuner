# 损失函数

## 动机

预训练和微调任务的损失函数往往使用的是 CE Loss 。CE Loss 大家想必都不陌生，可为什么 XTuner 还要设计自己的 CE Loss 呢？

1. 节约显存

当今大语言模型的词表普遍较大，同时，我们希望增加输入序列的长度来充分利用算力，导致 lm_head 计算 logits 再计算 loss 进而 backward 这一过程将会耗费大量显存。如下所示，使用 XTuner 提供的 chunk loss 可以节约 4/5 左右的显存：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem, CELossContext
import time


hidden_states = torch.randn(32768, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
lm_head = nn.Linear(4096, 151936, bias=False).to(device="cuda", dtype=torch.bfloat16)
torch.cuda.reset_peak_memory_stats()
t1 = time.time()
logits = lm_head(hidden_states)
shifted_labels = torch.randint(0, 151936, (32768, ), device="cuda")
loss = F.cross_entropy(logits, shifted_labels)
loss.backward()
max_memory = torch.cuda.max_memory_allocated()
reserved_memory = torch.cuda.max_memory_reserved()
print(f"Eager mode Loss: {loss.item()}")
print(f"Eager mode hidden_states grad norm: {hidden_states.grad.norm().item()}")
print(f"Eager mode lm_head weight grad norm: {lm_head.weight.grad.norm().item()}")
print(f"Eager mode Max memory allocated: {max_memory / 1024**3:.2f} GB")
print(f"Eager mode Max memory reserved: {reserved_memory / 1024**3:.2f} GB")
print(f"Eager mode Time taken: {time.time() - t1:.2f} seconds")

del logits
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

shifted_labels = shifted_labels.unsqueeze(0)
hidden_states = hidden_states.unsqueeze(0)
hidden_states = hidden_states.clone().detach().requires_grad_(True)
lm_head.weight.grad = None
t1 = time.time()
loss_ctx_input_list = [CELossContextInputItem(shifted_labels=shifted_labels)]
loss_cfg = CELossConfig(mode='chunk', chunk_size=1024, loss_reduction="token")
batches_loss_kwargs = CELossContext.build_batches_loss_kwargs(loss_ctx_input_list, loss_cfg)
loss_ctx = CELossContext(loss_cfg, batches_loss_kwargs[0])
loss, _ = loss_ctx.forward(hidden_states, lm_head.weight)
loss.backward()
max_memory = torch.cuda.max_memory_allocated()
reserved_memory = torch.cuda.max_memory_reserved()
print(f"Chunk mode Loss: {loss.item()}")
print(f"Chunk mode hidden_states grad norm: {hidden_states.grad.norm().item()}")
print(f"Chunk mode lm_head weight grad norm: {lm_head.weight.grad.norm().item()}")
print(f"Chunk mode Max memory allocated: {max_memory / 1024**3:.2f} GB")
print(f"Chunk mode Max memory reserved: {reserved_memory / 1024**3:.2f} GB")
print(f"Chunk mode Time taken: {time.time() - t1:.2f} seconds")

```

```shell
Eager mode Loss: 12.125
Eager mode hidden_states grad norm: 0.0031890869140625
Eager mode lm_head weight grad norm: 0.353515625
Eager mode Max memory allocated: 38.54 GB
Eager mode Max memory reserved: 38.54 GB
Eager mode Time taken: 0.38 seconds
Chunk mode Loss: 12.099200248718262
Chunk mode hidden_states grad norm: 0.0031890869140625
Chunk mode lm_head weight grad norm: 0.353515625
Chunk mode Max memory allocated: 6.87 GB
Chunk mode Max memory reserved: 10.72 GB
Chunk mode Time taken: 2.67 seconds
```

(global-average)=
2. 实现 loss 的全局校准

**什么是 loss 全局校准？**

loss 全局校准是指，无论使用多少张显卡，无论使用什么并行策略和梯度累积策略，其训练的效果都等价于在一张显卡上不使用任何并行策略时的效果（不考虑是否会 OOM）。

**为什么要做 loss 全局校准？**

我们希望模型的训练不受显卡数量、并行策略、梯度累积策略的变化而变化。

如果不进行 loss 全局校准，那么对于同样一批数据，使用 8 卡梯度累积 2 和使用 16 卡梯度累积 1 （global batch size 相同）的训练行为是不同的。换言之，当显卡数量、并行策略、梯度累积策略的变化时，如果不进行 loss 全局校准，则训练行为是不可复现的，如下所示。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContext
from mmengine.dist import infer_launcher, init_dist
import torch.distributed as dist


dist_launcher = infer_launcher()
init_dist(dist_launcher)
rank = dist.get_rank()
world_size = dist.get_world_size()

torch.manual_seed(0)
lm_head = nn.Linear(2, 10, bias=False).to(device="cuda", dtype=torch.bfloat16)
hidden_states_gt = torch.randn(8, 2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
shifted_labels_gt = torch.tensor([-100, 0, 1, -100, 0, 1, 2, 3], device="cuda")

# 1 gpu
logits = lm_head(hidden_states_gt)
loss = F.cross_entropy(logits, shifted_labels_gt)
loss.backward()
grad_1_gpu = lm_head.weight.grad.clone()

# 2 gpu without global average
hidden_states = hidden_states_gt.clone().detach().requires_grad_(True)
lm_head.weight.grad = None
hidden_states = torch.chunk(hidden_states, world_size, dim=0)[rank]
shifted_labels = torch.chunk(shifted_labels_gt, world_size, dim=0)[rank]
logits = lm_head(hidden_states)
loss = F.cross_entropy(logits, shifted_labels)
loss.backward()
dist.all_reduce(lm_head.weight.grad, op=dist.ReduceOp.AVG)
grad_2_gpu = lm_head.weight.grad.clone()
print(f'Without global average, torch.allclose(grad_1_gpu, grad_2_gpu, atol=1e-2, rtol=1e-2) = {torch.allclose(grad_1_gpu, grad_2_gpu, atol=1e-2, rtol=1e-2)}')

# 2 gpu without global average
hidden_states = hidden_states_gt.clone().detach().requires_grad_(True)
lm_head.weight.grad = None
hidden_states = torch.chunk(hidden_states, world_size, dim=0)[rank]
shifted_labels = torch.chunk(shifted_labels_gt, world_size, dim=0)[rank]
hidden_states = hidden_states.unsqueeze(0)
shifted_labels = shifted_labels.unsqueeze(0)
loss_cfg = CELossConfig(mode='chunk', chunk_size=1024, loss_reduction="token")
loss_ctx = loss_cfg.build(shifted_labels)
loss_ctx_list = CELossContext.build_batches([loss_ctx])
loss_ctx = loss_ctx_list[0]
loss, _ = loss_ctx.forward(hidden_states, lm_head.weight)
loss.backward()

dist.all_reduce(lm_head.weight.grad, op=dist.ReduceOp.AVG)
grad_2_gpu = lm_head.weight.grad.clone()
print(f'With global average, torch.allclose(grad_1_gpu, grad_2_gpu, atol=1e-2, rtol=1e-2) = {torch.allclose(grad_1_gpu, grad_2_gpu, atol=1e-2, rtol=1e-2)}')
```

```shell
Without global average, torch.allclose(grad_1_gpu, grad_2_gpu, atol=1e-2, rtol=1e-2) = False
Without global average, torch.allclose(grad_1_gpu, grad_2_gpu, atol=1e-2, rtol=1e-2) = False
With global average, torch.allclose(grad_1_gpu, grad_2_gpu, atol=1e-2, rtol=1e-2) = True
With global average, torch.allclose(grad_1_gpu, grad_2_gpu, atol=1e-2, rtol=1e-2) = True
```

**如何做 loss 全局校准？**

假设我们有两张显卡，序列并行度为 2，梯度累积 2 次。

```text
                            rank0         rank1
iter0 loss                 l00, l01      l02, l03
      loss weight          w00, w01      w02, w03
      loss mask (0 or 1)   m00, m01      m02, m03
iter1 loss                 l10, l11      l12, l13
      loss weight          w10, w11      w12, w13
      loss mask (0 or 1)   m10, m11      m12, m13
```


那么，loss 校准的方式如下：
1. 计算所有显卡在梯度累积范围内的 loss mask 的和：

```python
global_loss_mask_sum = all_reduce(sum([loss_mask.sum() for loss_mask in loss_masks_grad_acc]), op=dist.ReduceOp.SUM, group=world)
                     = (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
```

2. 计算当前 iter 的 loss，以 rank0 iter0 为例：

```python
loss_rank0iter0 = (l00 * w00 * m00 + l01 * w01 * m01)
loss_rank0iter0 = loss_rank0iter0 / global_loss_mask_sum
                = (l00 * w00 * m00 + l01 * w01 * m01) / (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
loss_rank0iter0 = all_reduce_autograd(loss_rank0iter0, op=dist.ReduceOp.SUM, group=world)
                = (l00 * w00 * m00 + l01 * w01 * m01 + l02 * w02 * m02 + l03 * w03 * m03) / (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
```

3. 计算梯度累积范围内的 step_loss，与一张显卡不使用梯度累积时的效果相同:

```python
step_loss = loss_rank0iter0 + loss_rank0iter1
          = (l00 * w00 * m00 + l01 * w01 * m01 + l02 * w02 * m02 + l03 * w03 * m03 + l10 * w10 * m10 + l11 * w11 * m11 + l12 * w12 * m12 + l13 * w13 * m13) / (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
```

(ce-loss)=
## CE Loss

XTuner 中所有的 loss 计算均涉及两个核心组件 `LossConfig` 和 `LossContext` 。CE Loss 则对应 [`CELossConfig`](xtuner.v1.loss.ce_loss.CELossConfig) 和 [`CELossContext`](xtuner.v1.loss.ce_loss.CELossContext)。下面是一个简单的 CE Loss 的使用示例：

```python
import torch
import torch.nn as nn
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContext

emb = nn.Embedding(4, 2)
head = nn.Linear(2, 4, bias=False)

input_ids = torch.randint(0, 10, (1, 5))
shifted_labels = input_ids[:, 1:]
input_ids = input_ids[:, :-1]
hidden_states = emb(input_ids)

loss_ctx_input_list = [CELossContextInputItem(shifted_labels=shifted_labels)]
loss_cfg = CELossConfig(mode='chunk', chunk_size=1024, loss_reduction="token")
loss_ctx = loss_cfg.build(shifted_labels=data["shifted_labels"])
loss_ctx_list = CELossContext.build_batches([loss_ctx])
loss_ctx = loss_ctx_list[0]
loss, _ = loss_ctx.forward(hidden_states, head.weight)
loss.backward()
```

### CELossConfig

`CELossConfig` 包含了 CE Loss 计算所需的所有可配置项。由三个通用配置项：`ignore_idx`, `mode` 和 `chunk_size`，以及一个 CE Loss 特有的 `loss_reduction` 组成。

```python
class CELossConfig:
    ignore_idx: Annotated[int, Parameter(help="ignore index for loss calculation")] = -100
    mode: Annotated[Literal["eager", "chunk"], Parameter(help="loss calculation mode")] = "eager"
    chunk_size: Annotated[int | None, Parameter(help="chunk size when mode is chunk")] = 1024
    loss_reduction: Annotated[Literal["token", "sample", "square"], Parameter(help="loss reduction mode")] = "token"
```

- `ignore_idx` 表示在 loss 计算中被忽略的 label ids ，通常为 `-100` ，用户无需额外设置。
- `mode` 共有 "eager" 和 "chunk" 两种可选，推荐设置为 "chunk" 模式来节省显存。
- `chunk_size` 只有 `mode` 是 "chunk" 是才会生效。
- `loss_reduction` 有 "token", "sample", "square" 三种可选，我们通常选择 "token" 模式，即 token 之间的 CE Loss 计算互不影响。

### CELossContext

在 `CELossContext` 中我们引入了额外的一个数据结构：[`CELossKwargs`](xtuner.v1.loss.ce_loss.CELossKwargs)。

- `CELossKwargs` 表示 CE Loss 实际计算的时候需要用到哪些参数，即：`shifted_labels` 和 `loss_weight` 两项，注意此时的 `loss_weight` 已经经历过全局校准的处理了，详细实现请参考 `xtuner/v1/loss/ce_loss.py`。

我们在 `CELossContext` 中只需要实现两个接口：

1. 为了做 loss 全局校准，staticmethod `build_batches` 计算全局校准对应的loss weight。
2. `loss_fn` 根据 `CELossKwargs` 计算出当前 iter 的 loss。

对于其他功能（如：chunk loss），不同 loss 都是通用的，我们统一放到 `BaseLossContext` 里实现。

## Custom Loss

如需自定义 loss 形式，需要重新实现 `CustomLossConfig` 和 `CustomLossContext` 两个数据结构。

### CustomLossConfig

继承 `BaseLossConfig` 并拓展所需字段：

```python
from xtuner.v1.loss import BaseLossConfig

class CustomLossConfig(BaseLossConfig):
    arg1: Any
    ...

    @property
    def loss_ctx_cls(self) -> type[CustomLossContext]:
        return CustomLossContext
```

### CustomLossContext

第一步，定义 custom loss 实际计算的时候需要用到哪些参数：

```python
from xtuner.v1.loss import BaseLossContext, BaseLossKwargs

class CustomLossKwargs(BaseLossKwargs):
    shifted_labels: torch.Tensor
    loss_weight: torch.Tensor
    arg1: Any
    ...
```

第二步，继承 `BaseLossContext` 并实现 `CustomLossContext` 中的 classmethod `build_batches` 和 `loss_fn`：

```python
from xtuner.v1.loss import BaseLossContext, BaseLossKwargs

class CustomLossContext(BaseLossContext):
    loss_cfg: CustomLossConfig
    loss_kwargs: CustomLossKwargs

    @staticmethod
    def build_batches(
        loss_ctx_list: list["CELossContext"],
        # 为了提高计算效率，XTuner 会将多条短数据 pack 成一条长数据进行训练
        # 若在计算 CustomLossKwargs 的过程中需要解 pack 成若干短数据，则需要传入 cu_seq_lens_list
        # 默认为 None 即可。
        cu_seq_lens_list: Sequence[torch.IntTensor] | None = None,
        # 若开启了序列并行 (sp) 且计算 CustomLossKwargs 的过程中需要 sp 切分前的数据，则需要传入 cu_seq_lens_list
        # 默认为 None 即可。
        sp_mesh: DeviceMesh | None = None,
    ) -> list[CustomLossKwargs]:
        ...
    
    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CustomLossKwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ...
```

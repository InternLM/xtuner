# Loss Function

## Motivation

Pre-training and fine-tuning tasks often use CE Loss. CE Loss is certainly familiar to everyone, so why does XTuner design its own CE Loss?

1. Save GPU Memory

Today's large language models generally have large vocabularies, and we want to increase the input sequence length to fully utilize computing power, resulting in the process of lm_head calculating logits, then calculating loss, and then backward propagation consuming a lot of GPU memory. As shown below, using XTuner's chunk loss can save about 4/5 of GPU memory:

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
Eager mode Max memory allocated: 38.53 GB
Eager mode Max memory reserved: 38.54 GB
Eager mode Time taken: 0.57 seconds
Chunk mode Loss: 12.096513748168945
Chunk mode hidden_states grad norm: 0.0031890869140625
Chunk mode lm_head weight grad norm: 0.353515625
Chunk mode Max memory allocated: 8.32 GB
Chunk mode Max memory reserved: 8.40 GB
Chunk mode Time taken: 0.40 seconds
```

(global-average)=
2. Implement Global Loss Calibration

**What is Global Loss Calibration?**

Global loss calibration means that no matter how many GPUs are used and no matter what parallel strategy and gradient accumulation strategy are used, the training effect is equivalent to the effect when using one GPU without any parallel strategy (regardless of whether OOM will occur).

**Why do Global Loss Calibration?**

We want model training to be unaffected by changes in the number of GPUs, parallel strategy, and gradient accumulation strategy.

Without global loss calibration, for the same batch of data, using 8 GPUs with gradient accumulation 2 and using 16 GPUs with gradient accumulation 1 (same global batch size) will have different training behaviors. In other words, when the number of GPUs, parallel strategy, and gradient accumulation strategy change, if global loss calibration is not performed, the training behavior is not reproducible, as shown below.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem, CELossContext
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
loss_ctx_input_list = [CELossContextInputItem(shifted_labels=shifted_labels)]
loss_cfg = CELossConfig(mode='chunk', chunk_size=1024, loss_reduction="token")
batches_loss_kwargs = CELossContext.build_batches_loss_kwargs(loss_ctx_input_list, loss_cfg)
loss_ctx = CELossContext(loss_cfg, batches_loss_kwargs[0])
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

**How to do Global Loss Calibration?**

Suppose we have two GPUs, sequence parallelism degree is 2, gradient accumulation is 2 times.

```text
                            rank0         rank1
iter0 loss                 l00, l01      l02, l03
      loss weight          w00, w01      w02, w03
      loss mask (0 or 1)   m00, m01      m02, m03
iter1 loss                 l10, l11      l12, l13
      loss weight          w10, w11      w12, w13
      loss mask (0 or 1)   m10, m11      m12, m13
```


Then, the loss calibration method is as follows:
1. Calculate the sum of loss masks within the gradient accumulation range for all GPUs:

```python
global_loss_mask_sum = all_reduce(sum([loss_mask.sum() for loss_mask in loss_masks_grad_acc]), op=dist.ReduceOp.SUM, group=world)
                     = (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
```

2. Calculate the loss of the current iter, taking rank0 iter0 as an example:

```python
loss_rank0iter0 = (l00 * w00 * m00 + l01 * w01 * m01)
loss_rank0iter0 = loss_rank0iter0 / global_loss_mask_sum
                = (l00 * w00 * m00 + l01 * w01 * m01) / (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
loss_rank0iter0 = all_reduce_autograd(loss_rank0iter0, op=dist.ReduceOp.SUM, group=world)
                = (l00 * w00 * m00 + l01 * w01 * m01 + l02 * w02 * m02 + l03 * w03 * m03) / (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
```

3. Calculate the step_loss within the gradient accumulation range, which has the same effect as using one GPU without gradient accumulation:

```python
step_loss = loss_rank0iter0 + loss_rank0iter1
          = (l00 * w00 * m00 + l01 * w01 * m01 + l02 * w02 * m02 + l03 * w03 * m03 + l10 * w10 * m10 + l11 * w11 * m11 + l12 * w12 * m12 + l13 * w13 * m13) / (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
```

(ce-loss)=
## CE Loss

All loss calculations in XTuner involve two core components: `LossConfig` and `LossContext`. CE Loss corresponds to [`CELossConfig`](xtuner.v1.loss.ce_loss.CELossConfig) and [`CELossContext`](xtuner.v1.loss.ce_loss.CELossContext). Below is a simple usage example of CE Loss:

```python
import torch
import torch.nn as nn
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem, CELossContext

emb = nn.Embedding(4, 2)
head = nn.Linear(2, 4, bias=False)

input_ids = torch.randint(0, 10, (1, 5))
shifted_labels = input_ids[:, 1:]
input_ids = input_ids[:, :-1]
hidden_states = emb(input_ids)

loss_ctx_input_list = [CELossContextInputItem(shifted_labels=shifted_labels)]
loss_cfg = CELossConfig(mode='chunk', chunk_size=1024, loss_reduction="token")
batches_loss_kwargs = CELossContext.build_batches_loss_kwargs(loss_ctx_input_list, loss_cfg)
loss_ctx = CELossContext(loss_cfg, batches_loss_kwargs[0])
loss, _ = loss_ctx.forward(hidden_states, head.weight)
loss.backward()
```

### CELossConfig

`CELossConfig` contains all configurable items needed for CE Loss calculation. It consists of three general configuration items: `ignore_idx`, `mode` and `chunk_size`, and one CE Loss specific `loss_reduction`.

```python
class CELossConfig:
    ignore_idx: Annotated[int, Parameter(help="ignore index for loss calculation")] = -100
    mode: Annotated[Literal["eager", "chunk"], Parameter(help="loss calculation mode")] = "eager"
    chunk_size: Annotated[int | None, Parameter(help="chunk size when mode is chunk")] = 1024
    loss_reduction: Annotated[Literal["token", "sample", "square"], Parameter(help="loss reduction mode")] = "token"
```

- `ignore_idx` represents the label ids ignored in loss calculation, usually `-100`, users don't need to set it additionally.
- `mode` has two options: "eager" and "chunk", it is recommended to set "chunk" mode to save GPU memory.
- `chunk_size` only takes effect when `mode` is "chunk".
- `loss_reduction` has three options: "token", "sample", "square", we usually choose "token" mode, that is, CE Loss calculation between tokens does not affect each other.

### CELossContext

In `CELossContext`, we introduce two additional data structures: [`CELossKwargs`](xtuner.v1.loss.ce_loss.CELossKwargs) and [`CELossContextInputItem`](xtuner.v1.loss.ce_loss.CELossContextInputItem).

- `CELossKwargs` represents what parameters are needed for actual CE Loss calculation, namely: `shifted_labels` and `loss_weight`. Note that `loss_weight` at this time has already been processed by global calibration. For detailed implementation, please refer to `xtuner/v1/loss/ce_loss.py`.
- `CELossContextInputItem` represents what information is needed to calculate `CELossKwargs`, namely: `shifted_labels`

We only need to implement two interfaces in `CELossContext`:

1. To do global loss calibration, the classmethod `build_batches_loss_kwargs` inputs `CELossContextInputItem` corresponding to each data within the gradient accumulation range, and calculates `CELossKwargs` for each iter.
2. `loss_fn` calculates the loss of the current iter based on `CELossKwargs`.

For other functions (such as: chunk loss), different losses are universal, and we put them all in `BaseLossContext` for implementation.

## Custom Loss

To customize the loss form, you need to re-implement two data structures: `CustomLossConfig` and `CustomLossContext`.

### CustomLossConfig

Inherit `BaseLossConfig` and expand the required fields:

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

Step 1, define what parameters are needed for actual custom loss calculation:

```python
from xtuner.v1.loss import BaseLossContext, BaseLossKwargs

class CustomLossKwargs(BaseLossKwargs):
    shifted_labels: torch.Tensor
    loss_weight: torch.Tensor
    arg1: Any
    ...
```

Step 2, inherit `BaseLossContext` and implement the classmethod `build_batches_loss_kwargs` and `loss_fn` in `CustomLossContext`:

```python
from xtuner.v1.loss import BaseLossContext, BaseLossKwargs
from xtuner.v1.loss.ce_loss import CELossContextInputItem

class CustomLossContext(BaseLossContext[CELossContextInputItem]):
    loss_cfg: CustomLossConfig
    loss_kwargs: CustomLossKwargs

    @classmethod
    def build_batches_loss_kwargs(
        cls,
        data_batches: list[RLLossContextInputItem],
        loss_cfg: CustomLossConfig,
        # To improve calculation efficiency, XTuner will pack multiple short data into one long data for training
        # If you need to unpack into several short data during the calculation of CustomLossKwargs, you need to pass in cu_seq_lens_list
        # The default is None.
        cu_seq_lens_list: list[torch.Tensor] | None = None,
        # If sequence parallelism (sp) is enabled and sp pre-split data is needed during the calculation of CustomLossKwargs
        # The default is None.
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
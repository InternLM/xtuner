# 损失函数

强化学习的损失函数往往包含 policy loss 用来优化当前策略， kl loss 用来防止当前策略过度偏离原始策略，以及其他的一些自定义 loss。这里我们以非常经典的 GRPO Loss 为例介绍 XTuner RL Loss 相关的机制。

## GRPOLoss

XTuner 中所有的 loss 计算均涉及两个核心组件 `LossConfig` 和 `LossContext` 。GRPO Loss 则对应 [`GRPOLossConfig`](xtuner.v1.rl.grpo.loss.GRPOLossConfig) 和 [`GRPOLossContext`](xtuner.v1.rl.grpo.loss.GRPOLossContext)。下面是一个简单的 GRPO Loss 的使用示例：

```python
import torch
import torch.nn as nn
from xtuner.v1.rl.grpo import GRPOLossConfig, GRPOLossContext
from xtuner.v1.rl.base import RLLossContextInputItem
from xtuner.v1.data_proto import SequenceContext

def gather_logprobs(logits, shifted_labels):
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs = logprobs.gather(dim=-1, index=shifted_labels.clip(min=0).unsqueeze(-1)).squeeze(-1)
    return logprobs

emb = nn.Embedding(4, 2)
head = nn.Linear(2, 4, bias=False)

input_ids = torch.randint(0, 10, (1, 5))
shifted_labels = input_ids[:, 1:]
input_ids = input_ids[:, :-1]
advantages = torch.tensor([0.5, 0.5, -0.5, -0.5])
hidden_states = emb(input_ids)

loss_ctx_input = RLLossContextInputItem(shifted_labels=shifted_labels, advantages=advantages)

with torch.no_grad():
    logits = lm_head(emb(input_ids))
    old_logprobs = gather_logprobs(logits, loss_ctx_input.shifted_labels)
    loss_ctx_input.old_logprobs = old_logprobs

loss_ctx_input_list = [loss_ctx_input]
loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.2,
        cliprange_low=0.2,
        loss_type='vanilla',
    ),
    use_kl_loss=False,
    mode='chunk', 
    chunk_size=1024
)
batches_loss_kwargs = GRPOLossContext.build_batches_loss_kwargs(loss_ctx_input_list, loss_cfg)
loss_ctx = GRPOLossContext(loss_cfg, batches_loss_kwargs[0])
loss, _ = loss_ctx.forward(hidden_states, head.weight)
loss.backward()
```

### GRPOLossConfig

`GRPOLossConfig` 包含了 GRPO Loss 计算所需的所有可配置项。

```python
class GRPOLossConfig:
    policy_loss_cfg: dict[str, Any]
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None = None
    ignore_idx: Annotated[int, Parameter(help="ignore index for loss calculation")] = -100
    mode: Annotated[Literal["eager", "chunk"], Parameter(help="loss calculation mode")] = "eager"
    chunk_size: Annotated[int | None, Parameter(help="chunk size when mode is chunk")] = 1024
    
```

其中 `policy_loss_cfg` 是 policy loss 的相关配置，`xtuner/v1/rl/loss_fn.py` 支持了不同的 rl policy loss fn。

### GRPOLossContext

与 [`CELossContext`](ce-loss) 相似，`GRPOLossContext` 同样需要考虑 loss 的全局校准。在 `GRPOLossContext` 中我们引入了 [`GRPOLossKwargs`](xtuner.v1.rl.grpo.loss.GRPOLossKwargs) 和 [`RLLossContextInputItem`](xtuner.v1.rl.base.loss.RLLossContextInputItem) 两个数据结构：

- `GRPOLossKwargs` 表示 GRPO Loss 实际计算的时候需要用到哪些参数，详细实现请参考 `xtuner/v1/rl/grpo/loss.py`。
- `RLLossContextInputItem` 是 RL 算法中的一个通用数据结构，基本囊括了当前 RL 算法在计算 `LossKwargs` 时所需要的所有物料。

与 `CELossContext` 类似，我们在 `GRPOLossContext` 中只需要实现两个接口：classmethod `build_batches_loss_kwargs` 和 `loss_fn`。

```{hint}
什么是全局校准？不妨来看看这个教程: [全局校准](global-average)
```

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

第二步，继承 `BaseLossContext` 并实现 `CustomLossContext` 中的 classmethod `build_batches_loss_kwargs` 和 `loss_fn`：

```python
from xtuner.v1.loss import BaseLossContext, BaseLossKwargs
from xtuner.v1.rl.base import RLLossContextInputItem

class CustomLossContext(BaseLossContext[RLLossContextInputItem]):
    loss_cfg: CustomLossConfig
    loss_kwargs: CustomLossKwargs

    @classmethod
    def build_batches_loss_kwargs(
        cls,
        data_batches: list[RLLossContextInputItem],
        loss_cfg: CustomLossConfig,
        # 为了提高计算效率，XTuner 会将多条短数据 pack 成一条长数据进行训练
        # 若在计算 CustomLossKwargs 的过程中需要解 pack 成若干短数据，则需要传入 cu_seq_lens_list
        # 默认为 None 即可。
        cu_seq_lens_list: list[torch.Tensor] | None = None,
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

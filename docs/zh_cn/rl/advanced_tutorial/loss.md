# 损失函数

强化学习的损失函数通常包含 policy loss，用来优化当前策略；也可以包含 KL loss，用来限制当前策略过度偏离参考策略。这里以 GRPO Loss 为例介绍 XTuner 当前的 RL Loss 机制。

## GRPOLoss

XTuner 中的 RL loss 仍然由 `LossConfig` 和 `LossContext` 两个核心组件组成。GRPO Loss 对应 [`GRPOLossConfig`](xtuner.v1.rl.loss.GRPOLossConfig) 和 [`GRPOLossContext`](xtuner.v1.rl.loss.GRPOLossContext)。当前接口中，训练样本先由 `GRPOLossConfig.build()` 转成 `GRPOLossContext`，再由 `GRPOLossContext.build_batches()` 在一个梯度累积 batch 内完成全局校准。

下面是一个最小示例：

```python
import torch
import torch.nn as nn

from xtuner.v1.rl.loss import GRPOLossConfig, GRPOLossContext
from xtuner.v1.rl.utils import gather_logprobs

emb = nn.Embedding(10, 4)
head = nn.Linear(4, 10, bias=False)

input_ids = torch.randint(0, 10, (1, 5))
shifted_labels = input_ids[:, 1:]
hidden_states = emb(input_ids[:, :-1])
advantages = torch.tensor([[0.5, 0.5, -0.5, -0.5]], dtype=torch.float32)

with torch.no_grad():
    logits = head(hidden_states)
    old_logprobs = gather_logprobs(logits, shifted_labels)

loss_cfg = GRPOLossConfig(
    policy_loss_cfg=dict(
        cliprange_high=0.2,
        cliprange_low=0.2,
        loss_type="vanilla",
    ),
    use_kl_loss=False,
    mode="chunk",
    chunk_size=1024,
)
loss_ctx = loss_cfg.build(
    {
        "shifted_labels": shifted_labels,
        "advantages": advantages,
        "old_logprobs": old_logprobs,
    }
)
assert loss_ctx is not None
loss_ctx = GRPOLossContext.build_batches([loss_ctx])[0]

loss, _ = loss_ctx.forward(hidden_states, head.weight)
loss.backward()
```

### GRPOLossConfig

`GRPOLossConfig` 继承自 [`BaseRLLossConfig`](xtuner.v1.rl.loss.BaseRLLossConfig)，而 `BaseRLLossConfig` 又继承自 `CELossConfig`。因此 GRPO Loss 的有效配置项由 RL loss 配置和基础 CE loss 配置共同组成：

```python
class GRPOLossConfig(BaseRLLossConfig):
    policy_loss_cfg: dict[str, Any]
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None = None
    rollout_is: RolloutImportanceSampling = RolloutImportanceSampling()
    ignore_idx: int = -100
    mode: Literal["eager", "chunk", "liger"] = "eager"
    chunk_size: int | None = 1024
    loss_reduction: Literal["token", "sample", "square"] = "token"
```

- `policy_loss_cfg` 配置 policy loss。当前内置 `loss_type="vanilla"`，对应 `xtuner/v1/rl/loss/loss_fn.py` 中注册的 policy gradient loss；也可以通过完整导入路径注册自定义 loss 函数。
- `use_kl_loss`、`kl_loss_coef` 和 `kl_loss_type` 控制是否叠加 KL loss。开启 KL loss 时，训练 worker 需要提供参考模型产生的 `ref_logprobs`。
- `rollout_is` 是 rollout importance sampling 配置，用于根据 rollout 阶段和训练阶段的 logprob 差异过滤或重加权样本。
- `ignore_idx` 表示不参与 loss 的 label id，默认是 `-100`。
- `mode` 和 `chunk_size` 控制 loss 的计算方式。RL 训练中常用 `mode="chunk"` 来降低显存占用。

### GRPOLossContext

`GRPOLossContext` 中实际参与计算的数据保存在 [`GRPOLossKwargs`](xtuner.v1.rl.loss.GRPOLossKwargs) 中。它继承自 [`BaseRLLossKwargs`](xtuner.v1.rl.loss.BaseRLLossKwargs)，主要字段包括：

- `shifted_labels`：右移后的 label，形状通常为 `[batch, seq_len]`。
- `advantages`：每个 token 对应的 advantage，形状需要与 `shifted_labels` 对齐。
- `old_logprobs`：旧策略在训练 token 上的 logprob，GRPO policy loss 必需。
- `rollout_logprobs` 和 `is_weights`：rollout importance sampling 相关字段。
- `ref_logprobs` 和 `kl_loss_weight`：KL loss 相关字段，仅在 `use_kl_loss=True` 时需要。
- `policy_loss_weight` 和 `global_grad_tokens`：由 `GRPOLossContext.build_batches()` 根据当前 batch 的有效 token 数计算。

当前接口中不再使用旧版输入项结构，也不再实现旧版批处理 kwargs 构造接口。对应关系如下：

```python
loss_ctx = loss_cfg.build(data)
loss_ctx = GRPOLossContext.build_batches([loss_ctx])[0]
loss, extra_info = loss_ctx.forward(hidden_states, head_weight)
```

```{hint}
什么是全局校准？可以参考 [预训练/SFT loss 文档](../../pretrain_sft/advanced_tutorial/loss.md) 中的全局校准说明。
```

## Custom Loss

如需自定义 RL loss，建议基于当前的 RL loss 基类实现：

- [`BaseRLLossConfig`](xtuner.v1.rl.loss.BaseRLLossConfig)：负责从训练数据字典构造 loss context。
- [`BaseRLLossKwargs`](xtuner.v1.rl.loss.BaseRLLossKwargs)：保存实际 loss 计算所需的 tensor 和配置。
- [`BaseRLLossContext`](xtuner.v1.rl.loss.BaseRLLossContext)：负责 batch 内全局校准和实际 loss 计算。

### CustomLossKwargs

第一步，定义自定义 loss 实际需要的参数。若仍然使用 RL 训练数据中的 `shifted_labels`、`advantages`、`old_logprobs` 等字段，可以继承 `BaseRLLossKwargs` 并追加自己的字段：

```python
from typing import Any

import torch

from xtuner.v1.rl.loss import BaseRLLossKwargs


class CustomLossKwargs(BaseRLLossKwargs):
    arg1: Any | None = None
```

### CustomLossConfig

第二步，继承 `BaseRLLossConfig`，声明对应的 context 和 kwargs 类型。若自定义字段需要从 `data` 中读取，可以覆盖 `build()`：

```python
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.rl.loss import BaseRLLossConfig


class CustomLossConfig(BaseRLLossConfig):
    arg1: Any

    @property
    def loss_ctx_cls(self) -> type["CustomLossContext"]:
        return CustomLossContext

    @property
    def _loss_kwargs_cls(self) -> type["CustomLossKwargs"]:
        return CustomLossKwargs

    def build(
        self,
        data: dict,
        sp_mesh: DeviceMesh | None = None,
    ) -> "CustomLossContext | None":
        # 从 data 中读取 shifted_labels、advantages、old_logprobs 等字段，
        # 构造 CustomLossKwargs 后返回 CustomLossContext(self, loss_kwargs)。
        ...
```

### CustomLossContext

第三步，继承 `BaseRLLossContext`，实现 `build_batches()` 和 `loss_fn()`：

```python
from typing import Any

import torch

from xtuner.v1.rl.loss import BaseRLLossContext


class CustomLossContext(BaseRLLossContext):
    loss_cfg: CustomLossConfig
    loss_kwargs: CustomLossKwargs

    @staticmethod
    def build_batches(loss_ctx_list: list["CustomLossContext"]) -> list["CustomLossContext"]:
        # 在这里统计当前梯度累积 batch 内的有效 token，
        # 并为每个 loss_ctx.loss_kwargs 写入 loss weight。
        ...

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CustomLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        # 计算当前 iter 的 loss，并返回 loss 与额外日志信息。
        ...
```

如果自定义算法仍然遵循 GRPO 的数据流，只是替换 policy loss 形式，优先使用 `register_policy_loss()` 注册新的 policy loss 函数，并在 `GRPOLossConfig.policy_loss_cfg["loss_type"]` 中指定它；只有当所需字段、全局校准方式或 loss 组合方式发生变化时，才需要自定义完整的 `LossConfig/LossContext`。

# Loss Function

Reinforcement learning loss functions often include policy loss for optimizing the current policy, kl loss for preventing the current policy from deviating too much from the original policy, and other custom losses. Here we take the very classic GRPO Loss as an example to introduce XTuner RL Loss related mechanisms.

## GRPOLoss

All loss calculations in XTuner involve two core components: `LossConfig` and `LossContext`. GRPO Loss corresponds to [`GRPOLossConfig`](xtuner.v1.rl.loss.GRPOLossConfig) and [`GRPOLossContext`](xtuner.v1.rl.loss.GRPOLossContext). Below is a simple usage example of GRPO Loss:

```python
import torch
import torch.nn as nn
from xtuner.v1.rl.loss import GRPOLossConfig, GRPOLossContext, RLLossContextInputItem
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

`GRPOLossConfig` contains all configurable items needed for GRPO Loss calculation.

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

`policy_loss_cfg` selects the policy loss. `loss_type` is one of the two built-in functions below, or a `module.function` import path for a custom function. GRPO, OREAL, and distillation share this implementation. The default is `vanilla`.

### Policy Loss

Both functions live in `xtuner/v1/rl/loss/loss_fn.py`. Each takes the current-policy `log_prob`, the old-policy `old_log_prob`, the token `advantages`, and the token `loss_weights`, and returns one weighted scalar. The weights already include the global calibration computed by the loss context.

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `cliprange_low` | yes |  | Lower clip on the importance ratio |
| `cliprange_high` | yes |  | Upper clip on the importance ratio |
| `clip_ratio_c` | no | `3.0` | Negative-advantage cap for `vanilla`, and the `pg_mask` threshold |
| `log_prob_diff_min` | no | `-20.0` | Lower bound on the log importance ratio |
| `log_prob_diff_max` | no | `20.0` | Upper bound on the log importance ratio |

Both losses first form the importance ratio and then clip it:

```text
ratio = exp(clamp(log_prob - old_log_prob, log_prob_diff_min, log_prob_diff_max))
clipped_ratio = clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
```

#### `vanilla`

`vanilla` is the PPO clipped surrogate with dual-clip on negative advantages.

```text
pg_losses1 = -ratio * advantages
pg_losses2 = -clipped_ratio * advantages
clip_pg_losses1 = max(pg_losses1, pg_losses2)
pg_losses3 = -clip_ratio_c * advantages
clip_pg_losses2 = min(pg_losses3, clip_pg_losses1)
pg_losses = clip_pg_losses2 if advantages < 0 else clip_pg_losses1
loss = sum(pg_losses * loss_weights)
```

For `advantages >= 0`, the loss is `-min(ratio, clipped_ratio) * advantages`. Once the ratio passes `1 + cliprange_high`, a larger ratio does not increase the update. For `advantages < 0` and a very large `ratio`, plain PPO would let the loss grow with `ratio`; dual-clip caps that token at `clip_ratio_c * abs(advantages)`.

#### `mask_pg`

`mask_pg` weights the current log probability by the clipped importance ratio. `pg_mask` drops tokens with `ratio > clip_ratio_c` from the loss.

```text
pg_mask = 1[ratio <= clip_ratio_c]
loss = sum(-loss_weights * advantages * clipped_ratio * log_prob * pg_mask)
```

### GRPOLossContext

Similar to [`CELossContext`](ce-loss), `GRPOLossContext` also needs to consider global calibration of loss. In `GRPOLossContext`, we introduce the [`GRPOLossKwargs`](xtuner.v1.rl.loss.GRPOLossKwargs) data structure:

- `GRPOLossKwargs` represents what parameters are needed for actual GRPO Loss calculation. For detailed implementation, please refer to `xtuner/v1/rl/loss/grpo_loss.py`.
- `RLLossContextInputItem` is a general data structure in RL algorithms, basically including all materials needed for calculating `LossKwargs` in current RL algorithms.

Similar to `CELossContext`, we only need to implement two interfaces in `GRPOLossContext`: classmethod `build_batches_loss_kwargs` and `loss_fn`.

```{hint}
What is global calibration? You might want to check out this tutorial: [Global Calibration](global-average)
```

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
from xtuner.v1.rl.base import RLLossContextInputItem

class CustomLossContext(BaseLossContext[RLLossContextInputItem]):
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

import torch
import torch.nn as nn


class BaseChunkLoss(nn.Module):
    def __init__(self, ctx, chunk_loss_class, chunk_loss_fn) -> None:
        super().__init__()
        self.ctx = ctx
        self.chunk_loss_class = chunk_loss_class
        self.chunk_loss_fn = chunk_loss_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        forward_item,
        head_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # 注意： 不能用 kwargs 传参数
        return self.chunk_loss_class.apply(hidden_states, head_weight, forward_item, head_bias, self.chunk_loss_fn)

import torch
import torch.nn as nn

from ..data_proto.loss_context import CELossForwardItem
from .chunk_ce_loss import ChunkCELoss


class BaseChunkLoss(nn.Module):
    def __init__(self, loss_cfg, chunk_loss_class: type[ChunkCELoss], chunk_loss_fn) -> None:
        super().__init__()
        self.loss_cfg = loss_cfg
        self.chunk_loss_class = chunk_loss_class
        self.chunk_loss_fn = chunk_loss_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        loss_forward_item: CELossForwardItem,
        head_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # 注意： 不能用 kwargs 传参数
        return self.chunk_loss_class.apply(
            hidden_states, head_weight, loss_forward_item, head_bias, self.chunk_loss_fn
        )

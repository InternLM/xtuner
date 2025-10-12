from xtuner.v1.ops.act_fn import get_act_fn
import torch.nn as nn
import torch
from .qwen3_vl_config import Qwen3VLProjectorConfig


class Qwen3VLProjector(nn.Module):
    def __init__(self, config: Qwen3VLProjectorConfig) -> None:
        super().__init__()
        self.hidden_size = config.vision_hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = config.use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if self.use_postshuffle_norm else config.vision_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.text_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

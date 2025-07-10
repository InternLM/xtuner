from abc import abstractmethod
from typing import Literal, Tuple

import torch
from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated


class OptimConfig(BaseModel):
    lr: Annotated[float, Parameter(help="Learning rate for optimization")] = 1e-5
    max_grad_norm: Annotated[float, Parameter(help="Maximum gradient norm for gradient clipping")] = 1.0

    @abstractmethod
    def build(self, params):
        pass


class AdamWConfig(OptimConfig):
    weight_decay: Annotated[float, Parameter(help="Weight decay coefficient for L2 regularization")] = 0.01
    betas: Annotated[Tuple[float, float], Parameter(help="Beta coefficients for Adam optimizer")] = (0.9, 0.95)
    eps: Annotated[float, Parameter(help="Epsilon value for numerical stability in Adam optimizer")] = 1e-8

    def build(self, params):
        return torch.optim.AdamW(params, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)


class LRConfig(BaseModel):
    lr_type: Annotated[Literal["cosine", "linear", "constant"], Parameter(help="Type of learning rate schedule")] = (
        "constant"
    )
    warmup_ratio: Annotated[float, Parameter(help="Ratio of warmup steps to total training steps")] = 0.03
    lr_min: Annotated[float, Parameter(help="Minimum learning rate for optimization")] = 1e-6
    # todo: total_steps 如何传参给 engine
    total_steps: Annotated[int, Parameter(help="Total number of training steps, -1 for automatic determination")] = -1

import torch.nn as nn
from torch import Tensor


def native_dropout(input: Tensor, p: float, training: bool, inplace: bool) -> Tensor:
    return nn.functional.dropout(input, p, training, inplace)


def npu_dropout(input: Tensor, p: float, training: bool, inplace: bool) -> Tensor:
    import torch_npu

    return torch_npu._npu_dropout(input, p)[0]  # _npu_dropout returns a tuple


def get_dropout():
    from xtuner.v1.utils import get_device

    device = get_device()
    if device == "npu":
        return npu_dropout
    else:
        return native_dropout


dropout_fn = get_dropout()


class Dropout(nn.Dropout):
    def forward(self, x: Tensor) -> Tensor:
        return dropout_fn(x, self.p, self.training, self.inplace)

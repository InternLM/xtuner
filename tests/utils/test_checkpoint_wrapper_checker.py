from torch._prims_common import check
from xtuner.v1.model.utils import checkpoint_wrapper
import torch.nn as nn
import torch
import pytest


# Missing typehints
class ErrorDecoderLayer1(nn.Module):
    def forward(self, x):
        return x


# Inputs args missing raw tensor
class ErrorDecoderLayer2(nn.Module):
    def forward(self, x: list[torch.Tensor], y: tuple[torch.Tensor], z: dict[str, torch.Tensor]) -> torch.Tensor:
        ...


# Missing return type
class ErrorDecoderLayer3(nn.Module):
    def forward(self, x: torch.Tensor, y: tuple[torch.Tensor], z: dict[str, torch.Tensor]):
        ...

# Missing raw tensor in return type
class ErrorDecoderLayer4(nn.Module):
    def forward(self, x: torch.Tensor, y: tuple[torch.Tensor], z: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], int]:
        ...


# return type must be a tuple
class ErrorDecoderLayer5(nn.Module):
    def forward(self, x: torch.Tensor, y: tuple[torch.Tensor], z: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        ...


class DecoderLayer1(nn.Module):
    def forward(self, x: torch.Tensor, y: tuple[torch.Tensor], z: dict[str, torch.Tensor]) -> torch.Tensor:
        ...


class DecoderLayer2(nn.Module):
    def forward(self, x: torch.Tensor, y: tuple[torch.Tensor], z: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        ...


class DecoderLayer3(nn.Module):
    def forward(
            self, x: torch.Tensor, y: tuple[torch.Tensor], z: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, int] | torch.Tensor:
        ...


def test_checkpoint_wrapper_checker():
    with pytest.raises(TypeError):
        checkpoint_wrapper(ErrorDecoderLayer1())

    with pytest.raises(TypeError):
        checkpoint_wrapper(ErrorDecoderLayer2())

    with pytest.raises(TypeError):
        checkpoint_wrapper(ErrorDecoderLayer3())

    with pytest.raises(TypeError):
        checkpoint_wrapper(ErrorDecoderLayer4())

    with pytest.raises(TypeError):
        checkpoint_wrapper(ErrorDecoderLayer5())

    # Correct cases
    checkpoint_wrapper(DecoderLayer1())
    checkpoint_wrapper(DecoderLayer2())
    checkpoint_wrapper(DecoderLayer3())


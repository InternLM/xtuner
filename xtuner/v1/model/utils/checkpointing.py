import inspect
from types import UnionType
from typing import Union, get_args, get_origin

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from xtuner.v1.utils import copy_signature


# TODO: Currently xtuner uses the internal, outdated `torch.distributed.algorithms._checkpoint.checkpoint_wrapper` interface
# We should look for opportunities to use the public, updated interface in the future

# NOTE:
# PyTorch's `torch.distributed.algorithms._checkpoint.checkpoint_wrapper` has some limitations. Modules decorated with `checkpoint_wrapper`
# must have forward interfaces that conform to the specifications of `torch.autograd.function.Function`.
# Specifically, for input parameters, the `forward` interface must explicitly accept parameters of type `torch.Tensor` to ensure proper gradient backpropagation.
# For return values, the `forward` interface must return either `torch.Tensor` or tuple[torch.Tensor, ...].
# For example If the forward interface is declared as:
# def forward(self, x: tuple[torch.Tensor], y: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, ...]:
# This interface will break the gradient graph because the inputs don't meet the requirements. For instance, x and y are not of type `torch.Tensor`
# `_check_signature_of_forward` will check (not exhaustively) whether the signature of the `forward` interface meets the requirements to identify issues early.


def _check_signature_of_forward(module: nn.Module):
    def _is_tensor_or_tuple_tensor(arg_type: type):
        if arg_type is torch.Tensor:
            return True

        origin_type = get_origin(arg_type)

        if not origin_type or origin_type not in (tuple, UnionType, Union):
            return False

        if origin_type in [UnionType, Union]:
            type_list = get_args(arg_type)
            return any(_is_tensor_or_tuple_tensor(t) for t in type_list)

        else:
            type_list = get_args(arg_type)
            return any(t is torch.Tensor for t in type_list)

    def _has_missing_type(arg_type: type):
        if arg_type is inspect._empty:
            return True
        origin_arg = get_origin(arg_type)
        return any(_has_missing_type(t) for t in get_args(origin_arg))

    input_type = inspect.signature(module.forward).parameters
    ret_type = inspect.signature(module.forward).return_annotation

    for name, arg_type in input_type.items():
        if _has_missing_type(arg_type.annotation):
            raise TypeError(
                f"The type of argument '{name}' of {module.__class__.__name__}.forward must be annotated, but got "
                f"{name} unannotated."
            )

    if _has_missing_type(ret_type):
        raise TypeError(
            f"The return type of {module.__class__.__name__}.forward must be annotated, but got {ret_type}"
        )

    for arg_type in input_type.values():
        origin_arg = get_origin(arg_type.annotation)
        # Union[Tensor, None] or Optional[Tensor] is legal
        if origin_arg:
            if torch.Tensor in origin_arg:
                break
        else:
            if arg_type.annotation is torch.Tensor:
                break
    else:
        raise TypeError(
            f"The type of all arguments of the {module.__class__.__name__}.forward must be torch.Tensor, but got "
            f"{input_type}"
        )

    if not _is_tensor_or_tuple_tensor(ret_type):
        raise TypeError(
            f"The return type of {module.__class__.__name__}.forward must be torch.Tensor or tuple of torch.Tensor, "
            f"but got {ret_type}"
        )


@copy_signature(ptd_checkpoint_wrapper)
def checkpoint_wrapper(module: nn.Module, *args, **kwargs):
    _check_signature_of_forward(module)
    return ptd_checkpoint_wrapper(module, *args, **kwargs)

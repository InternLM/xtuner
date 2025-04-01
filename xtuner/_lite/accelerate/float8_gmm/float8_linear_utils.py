# Copyright (c) OpenMMLab. All rights reserved.
# Copied from https://github.com/pytorch/ao/blob/v0.8.0/torchao/float8/float8_linear_utils.py
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor, all_reduce

from xtuner._lite import get_logger
from xtuner._lite.accelerate.float8_gmm.config import Float8LinearConfig, ScalingType
from xtuner._lite.accelerate.float8_gmm.float8_linear import Float8Linear
from xtuner._lite.accelerate.float8_gmm.float8_utils import amax_history_to_scale_stack

logger = get_logger()


def linear_requires_sync(config: Float8LinearConfig):
    """Returns whether the given linear_type requires sync before forward."""
    return any(
        [
            config.cast_config_input.scaling_type is ScalingType.DELAYED,
            config.cast_config_weight.scaling_type is ScalingType.DELAYED,
            config.cast_config_grad_output.scaling_type is ScalingType.DELAYED,
        ]
    )


def _update_history_stack(
    new_amax: torch.Tensor, amax_history_stack: torch.Tensor
) -> torch.Tensor:
    """Updates `amax_history` (the last N cur_amax values) inplace with the
    value of `new_amax`.

    Args:
        new_amax (torch.Tensor): The new amax value to add to the history. (n_amaxes, 1)
        amax_history_stack (torch.Tensor): The history of amax values. (n_amaxes, history_length)
    """
    assert (
        amax_history_stack.dim() == 2
    ), f"Expected amat_history_stack to be 2D, got {amax_history_stack.shape()}"
    assert new_amax.size(0) == amax_history_stack.size(0), (
        "Expected new_amax to have the same size as the first dimension of amax_history_stack, "
        f"got {new_amax.size(0)} and {amax_history_stack.size(0)}"
    )
    new_amax_history_stack = torch.roll(amax_history_stack, 1, dims=1)
    new_amax_history_stack[:, 0] = new_amax.squeeze(-1)
    amax_history_stack.copy_(new_amax_history_stack)


def swap_linear_layers(
    module: nn.Module,
    from_float_func: Callable[[nn.Linear], nn.Linear],
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> nn.Module:
    """Generic function to swap linear layers in a module with a new type of
    linear layer.

    Note:
        If applied to a root-level nn.Linear, the module will not be modified in place
        and returned instead

    Args:
        module: Module to modify.
        from_float_func: Function that accepts a linear layer and returns a new type of linear layer.
        module_filter_fn: If specified, only the `torch.nn.Linear` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance, and the FQN.

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    if isinstance(module, nn.Linear) and (
        module_filter_fn is None or module_filter_fn(module, "")
    ):
        if len(list(module.children())) > 0:
            raise AssertionError(
                f"Does not support a root nn.Linear with children: {module}"
            )
        return from_float_func(
            module,
        )

    root_module = module

    def post_order_traversal(
        module: nn.Module,
        cur_fqn: Optional[str] = None,
        parent_module: Optional[nn.Module] = None,
    ):
        if cur_fqn is None:
            cur_fqn = ""

        for child_module_name, child_module in module.named_children():
            if cur_fqn == "":
                new_fqn = child_module_name
            else:
                new_fqn = f"{cur_fqn}.{child_module_name}"

            post_order_traversal(child_module, new_fqn, module)

        if isinstance(module, nn.Linear) and (
            module_filter_fn is None or module_filter_fn(module, cur_fqn)
        ):
            assert (
                parent_module is not None
            ), f"Linear root module should return early: {module}"
            new_linear_module = from_float_func(module)
            cur_module_name = cur_fqn.split(".")[-1]
            setattr(parent_module, cur_module_name, new_linear_module)

    post_order_traversal(root_module)
    return root_module


def convert_to_float8_training(
    module: nn.Module,
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
    config: Float8LinearConfig = None,
) -> nn.Module:
    """Swaps `torch.nn.Linear` in `module` with `Float8Linear`.

    Args:
        module: Module to modify.
        module_filter_fn: If specified, only the `torch.nn.Linear` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance and the FQN.
        config (Float8LinearConfig): configuration for conversion to float8

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    if config is None:
        config = Float8LinearConfig()

    from_float = lambda m: Float8Linear.from_float(
        m,
        config=config,
    )

    return swap_linear_layers(
        module,
        from_float,
        module_filter_fn=module_filter_fn,
    )


def get_float8_layers(model: torch.nn.Module):
    """Iterates through the model and returns all the Float8Linear layers.

    Args:
        model (torch.nn.Module): The model to look for Float8Linear layers in.
    """

    # Get all fp8 layers and tensors
    fp8_layers = [child for child in model.modules() if isinstance(child, Float8Linear)]
    if not torch.compiler.is_compiling():
        for layer in fp8_layers:
            for buf in layer.buffers():
                torch._dynamo.mark_static_address(buf, guard=True)
    return fp8_layers


@torch.no_grad()
def sync_float8_amax_and_scale_history(model: torch.nn.Module, fp8_layers=None) -> None:
    """
    Manages the float8 amax and scale bookkeeping. In detail, it does the
    following:
    1. in distributed contexts, syncs amax values across workers for activations and gradients
    2. adds the `amax` values to history
    3. calculates the scales to be used for next iteration
    4. sets the `amax_and_scale_synced` flag on the Float8Linear modules
       to signal that they have been synced

    TODO(future): design the UX for this (context manager, etc)

    PERFORMANCE NOTE:
        When you can, it is much more efficient to call get_float8_layers once at
        the beginning of the training loop and pass the result to this function.
        Because of how this interacts with torch.compile

    Args:
        model (torch.nn.Module): The model to track amaxes for
        fp8_layers (optional): If fp8_layers are provided, fp8_classes are ignored,
            and we loop over all fp8_layers to sync and update amax scale histories.
            Users can use get_float8_layers to get all fp8 layers.
    """
    # TODO(future): consider adding a flag to control setting the `is_amax_initialized`
    # flag only on the first iteration.

    if fp8_layers is None:
        fp8_layers = get_float8_layers(model)

    if len(fp8_layers) == 0:
        logger.warning(
            "Calling sync_float8_amax_and_scale_history on a module with no Float8Linear layers"
        )
        return

    def inner_func():
        """Why do we have this inner_function?

        There are two portions of the outer sync_function that cause graph_breaks:
            1. The `get_float8_layers` call can cause graph breaks if the user did not pass
                in the fp8_layers.
            2. At the end of syncing all the amaxes and scales we set the attr on the module
                signaling that we have synced the amaxes and scales and the next forward can be run.
                # TODO Maybe we should remove this safety check to remove the graph break?

        By having this inner function, we can ensure that although the outer function may cause graph breaks
        the inner function will not.
        """
        # Loop over all fp8 layers and grab the needed tensors
        fp8_amax_input_tensor_list = [None] * len(fp8_layers)
        fp8_amax_weight_tensor_list = [None] * len(fp8_layers)
        fp8_amax_grad_output_tensor_list = [None] * len(fp8_layers)

        fp8_input_amax_history_stack = [None] * len(fp8_layers)
        fp8_weight_amax_history_stack = [None] * len(fp8_layers)
        fp8_grad_output_amax_history_stack = [None] * len(fp8_layers)

        input_dtypes = set()
        weight_dtypes = set()
        grad_output_dtypes = set()
        scale_fn_recipes = set()

        for idx, child in enumerate(fp8_layers):
            fp8_amax_input_tensor_list[idx] = child.fp8_amax_input
            fp8_amax_weight_tensor_list[idx] = child.fp8_amax_weight
            fp8_amax_grad_output_tensor_list[idx] = child.fp8_amax_grad_output

            fp8_input_amax_history_stack[idx] = child.fp8_amax_history_input
            fp8_weight_amax_history_stack[idx] = child.fp8_amax_history_weight
            fp8_grad_output_amax_history_stack[idx] = child.fp8_amax_history_grad_output

            input_dtypes.add(child.config.cast_config_input.target_dtype)
            weight_dtypes.add(child.config.cast_config_weight.target_dtype)
            grad_output_dtypes.add(child.config.cast_config_grad_output.target_dtype)
            scale_fn_recipes.add(child.config.delayed_scaling_config.scale_fn_name)

        (input_dtype,) = input_dtypes
        (weight_dtype,) = weight_dtypes
        (grad_output_dtype,) = grad_output_dtypes

        if len(scale_fn_recipes) != 1:
            raise ValueError(
                f"All layers must have the same scale_fn recipe, got {scale_fn_recipes}"
            )
        scale_fn_recipe = next(iter(scale_fn_recipes))

        assert (
            len(fp8_amax_input_tensor_list)
            == len(fp8_amax_weight_tensor_list)
            == len(fp8_amax_grad_output_tensor_list)
        ), "Mismatched lengths of amax tensors."

        if dist.is_initialized():
            all_amax_tensors = torch.cat(
                fp8_amax_input_tensor_list
                + fp8_amax_weight_tensor_list
                + fp8_amax_grad_output_tensor_list
            )
            all_reduced_amax_tensor = all_reduce(
                all_amax_tensors, "MAX", list(range(dist.get_world_size()))
            )
            if isinstance(all_reduced_amax_tensor, AsyncCollectiveTensor):
                all_reduced_amax_tensor = all_reduced_amax_tensor.wait()

            (
                reduced_fp8_amax_input_tensor,
                reduced_fp8_amax_weight_tensor,
                reduced_fp8_amax_grad_output_tensor,
            ) = torch.split(all_reduced_amax_tensor, len(fp8_amax_input_tensor_list))

            for idx, child in enumerate(fp8_layers):
                child.fp8_amax_input.copy_(reduced_fp8_amax_input_tensor[idx])
                child.fp8_amax_weight.copy_(reduced_fp8_amax_weight_tensor[idx])
                child.fp8_amax_grad_output.copy_(
                    reduced_fp8_amax_grad_output_tensor[idx]
                )

        # We create two stacked tensor groups, one for the amax history and one for the current scales
        fp8_amax_input_tensors = torch.vstack(fp8_amax_input_tensor_list)
        fp8_amax_weight_tensors = torch.vstack(fp8_amax_weight_tensor_list)
        fp8_amax_grad_output_tensors = torch.vstack(fp8_amax_grad_output_tensor_list)

        fp8_input_amax_history_stack = torch.vstack(fp8_input_amax_history_stack)
        fp8_weight_amax_history_stack = torch.vstack(fp8_weight_amax_history_stack)
        fp8_grad_output_amax_history_stack = torch.vstack(
            fp8_grad_output_amax_history_stack
        )

        # Update the history stacks with the new amax values
        _update_history_stack(fp8_amax_input_tensors, fp8_input_amax_history_stack)
        _update_history_stack(fp8_amax_weight_tensors, fp8_weight_amax_history_stack)
        _update_history_stack(
            fp8_amax_grad_output_tensors, fp8_grad_output_amax_history_stack
        )

        # Calculate the new scales from the updated history stacks
        new_input_scales = amax_history_to_scale_stack(
            fp8_input_amax_history_stack, input_dtype, scale_fn_recipe
        )
        new_weight_scales = amax_history_to_scale_stack(
            fp8_weight_amax_history_stack, weight_dtype, scale_fn_recipe
        )
        new_grad_output_scales = amax_history_to_scale_stack(
            fp8_grad_output_amax_history_stack, grad_output_dtype, scale_fn_recipe
        )

        # Iterate through the layers and update the scales
        for idx, child in enumerate(fp8_layers):
            child.fp8_scale_input.copy_(new_input_scales[idx])
            child.fp8_scale_weight.copy_(new_weight_scales[idx])
            child.fp8_scale_grad_output.copy_(new_grad_output_scales[idx])

    # This allows for the compile to succeed on the inner func and fail on the graph breaks
    # at the beginning and and of syncing
    inner_func()

    for child in fp8_layers:
        # Set a flag to signal that initialization is done
        child.is_amax_initialized = True

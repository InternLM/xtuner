"""Activation checkpointing entry points."""

from typing import Any, Callable

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint


__all__ = ["apply_activation_checkpointing"]


def apply_activation_checkpointing(
    module: nn.Module,
    *,
    preserve_rng_state: bool = True,
) -> nn.Module:
    """Wrap ``module`` in fixed reentrant activation checkpointing.

    The PyTree bridge keeps nested positional and keyword inputs visible to autograd and saved-tensor hooks, and
    restores structured model outputs.
    """

    def checkpoint_fn(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return _checkpoint_pytree(
            function,
            call_args=args,
            call_kwargs=kwargs,
            preserve_rng_state=preserve_rng_state,
        )

    # FSDP must wrap this checkpoint boundary. Its output hook then unshards
    # parameters before replay; wrapping FSDP itself would replay an FSDP forward.
    return checkpoint_wrapper(
        module,
        checkpoint_impl=CheckpointImpl.REENTRANT,
        checkpoint_fn=checkpoint_fn,
    )


@torch.compiler.disable(recursive=False)
def _checkpoint_pytree(
    function: Callable[..., Any],
    /,
    *,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    preserve_rng_state: bool,
) -> Any:
    """Adapt a structured module call to reentrant checkpoint's flat
    boundary."""
    flat_inputs, input_spec = tree_flatten((call_args, call_kwargs))
    has_grad_input = any(isinstance(value, torch.Tensor) and value.requires_grad for value in flat_inputs)
    checkpoint_inputs = flat_inputs
    if not has_grad_input:
        # Reentrant checkpoint only replays when at least one input requires grad. MTP can detach
        # every model input, so this empty leaf preserves the replay entry without affecting data.
        first_tensor = next(value for value in flat_inputs if isinstance(value, torch.Tensor))
        checkpoint_entry = torch.empty(0, device=first_tensor.device, requires_grad=True)
        checkpoint_inputs = [checkpoint_entry, *flat_inputs]
    output_spec: TreeSpec | None = None

    def call_with_original_signature(*replayed_inputs: Any) -> tuple[Any, ...]:
        nonlocal output_spec
        if not has_grad_input:
            replayed_inputs = replayed_inputs[1:]
        replayed_args, replayed_kwargs = tree_unflatten(list(replayed_inputs), input_spec)
        flat_outputs, current_output_spec = tree_flatten(function(*replayed_args, **replayed_kwargs))
        if output_spec is None:
            output_spec = current_output_spec
        elif current_output_spec != output_spec:
            raise RuntimeError("Checkpoint replay returned a different output PyTree structure")
        return tuple(flat_outputs)

    flat_outputs = checkpoint(
        call_with_original_signature,
        *checkpoint_inputs,
        use_reentrant=True,
        preserve_rng_state=preserve_rng_state,
    )
    assert output_spec is not None, "XTuner internal error: checkpoint did not run the function"
    if not isinstance(flat_outputs, tuple):
        flat_outputs = (flat_outputs,)
    return tree_unflatten(list(flat_outputs), output_spec)

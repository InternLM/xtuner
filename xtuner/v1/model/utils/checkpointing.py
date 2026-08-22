"""Activation checkpointing entry points."""

from collections import deque
from contextvars import ContextVar
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint


__all__ = ["apply_activation_checkpointing", "reuse_during_recompute"]


class _ActivationCheckpointFrame:
    """Reusable outputs owned by one checkpoint invocation."""

    def __init__(self) -> None:
        self.outputs: dict[Callable[..., Any], deque[Any]] = {}

    def save(self, function: Callable[..., Any], output: Any) -> None:
        self.outputs.setdefault(function, deque()).append(output)

    def replay(self, function: Callable[..., Any]) -> Any:
        queue = self.outputs.get(function)
        if not queue:
            raise RuntimeError("Checkpoint replay has no matching reusable output from the original forward")
        return queue.popleft()

    def assert_consumed(self) -> None:
        if any(self.outputs.values()):
            raise RuntimeError("Checkpoint replay did not consume every reusable output from the original forward")


# The bool is true only while a reentrant checkpoint invocation is replaying.
_CURRENT_ACTIVATION_CHECKPOINT: ContextVar[tuple[_ActivationCheckpointFrame, bool] | None] = ContextVar(
    "xtuner_current_activation_checkpoint",
    default=None,
)


@torch.compiler.disable(recursive=False)
def reuse_during_recompute(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run a no-grad callable once and reuse its output during checkpoint
    replay.

    Outside :func:`apply_activation_checkpointing`, this is a direct call. Inside a
    checkpoint invocation, calls to the same stable callable are matched in FIFO order.
    """
    state = _CURRENT_ACTIVATION_CHECKPOINT.get()
    if state is None:
        return function(*args, **kwargs)

    frame, is_replay = state
    if is_replay:
        return frame.replay(function)

    output = function(*args, **kwargs)
    flat_output, _ = tree_flatten(output)
    if any(isinstance(leaf, torch.Tensor) and leaf.requires_grad for leaf in flat_output):
        raise RuntimeError("reuse_during_recompute only supports Tensor outputs that do not require gradients")
    frame.save(function, output)
    return output


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
    output_spec: TreeSpec | None = None
    frame = _ActivationCheckpointFrame()

    @torch.compiler.disable(recursive=False)
    def call_with_original_signature(*replayed_inputs: Any) -> tuple[Any, ...]:
        nonlocal output_spec
        replayed_args, replayed_kwargs = tree_unflatten(list(replayed_inputs), input_spec)
        is_replay = torch.is_grad_enabled()
        token = _CURRENT_ACTIVATION_CHECKPOINT.set((frame, is_replay))
        try:
            output = function(*replayed_args, **replayed_kwargs)
        finally:
            _CURRENT_ACTIVATION_CHECKPOINT.reset(token)

        flat_outputs, current_output_spec = tree_flatten(output)
        if output_spec is None:
            output_spec = current_output_spec
        elif current_output_spec != output_spec:
            raise RuntimeError("Checkpoint replay returned a different output PyTree structure")
        if is_replay:
            frame.assert_consumed()
        return tuple(flat_outputs)

    flat_outputs = checkpoint(
        call_with_original_signature,
        *flat_inputs,
        use_reentrant=True,
        preserve_rng_state=preserve_rng_state,
    )
    assert output_spec is not None, "XTuner internal error: checkpoint did not run the function"
    if not isinstance(flat_outputs, tuple):
        flat_outputs = (flat_outputs,)
    return tree_unflatten(list(flat_outputs), output_spec)

"""Gradient checkpointing (activation recomputation) entry points."""

from contextlib import AbstractContextManager
from functools import partial
from typing import Any, Callable

import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint


__all__ = ["apply_gradient_checkpointing", "checkpoint_flattened"]


ContextFn = Callable[[], tuple[AbstractContextManager, AbstractContextManager]]


def apply_gradient_checkpointing(
    module: nn.Module,
    *,
    preserve_rng_state: bool = True,
    use_reentrant: bool = True,
    context_fn: ContextFn | None = None,
) -> nn.Module:
    """Make ``module``'s forward recomputed during backward instead of kept in
    memory.

    Inputs and outputs are flattened around reentrant checkpointing so gradients flow through
    nested containers, ``TypedDict`` returns, and keyword-only arguments. Non-reentrant
    checkpointing remains available for selective activation checkpointing.

    Args:
        module (nn.Module): Module whose forward should be recomputed during backward.
        preserve_rng_state (bool): Restore the RNG state before recomputing, so dropout and other
            stochastic ops replay identically. Defaults to True.
        use_reentrant (bool): Use reentrant checkpointing. Defaults to True.
        context_fn (Callable | None): Factory returning the ``(forward_context, recompute_context)``
            pair that ``torch.utils.checkpoint.checkpoint`` enters around the two passes. This is
            the seam for selective checkpointing: passing the contexts built by
            ``create_selective_checkpoint_contexts`` turns whole-module recompute into a per-op
            decision. Requires ``use_reentrant=False``. Defaults to None, i.e. recompute everything.

    Returns:
        nn.Module: An outer checkpoint wrapper containing ``module``.
    """
    # FSDP must wrap the checkpoint boundary. Its output hook then runs before
    # checkpoint replay and unshards the parameters for backward. Putting the
    # checkpoint around the FSDP module itself makes reentrant replay look like
    # another forward and corrupts FSDP's default backward-prefetch order.
    extra: dict[str, Any] = {} if context_fn is None else {"context_fn": context_fn}
    checkpoint_fn = partial(
        checkpoint_flattened,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=use_reentrant,
        **extra,
    )
    checkpoint_impl = CheckpointImpl.REENTRANT if use_reentrant else CheckpointImpl.NO_REENTRANT
    return checkpoint_wrapper(module, checkpoint_impl=checkpoint_impl, checkpoint_fn=checkpoint_fn)


def checkpoint_flattened(
    function: Callable[..., Any],
    *args: Any,
    preserve_rng_state: bool = True,
    use_reentrant: bool = True,
    **kwargs: Any,
) -> Any:
    """Run ``function`` under checkpointing with flattened inputs and outputs.

    Reentrant checkpointing needs both sides flattened so autograd sees tensors nested in container
    inputs and outputs. Non-reentrant checkpointing handles those structures itself, but still
    needs flattened inputs so the caller's saved-tensor hooks can see every input tensor.
    ``_CheckpointFrame.save_inputs`` wraps only *top-level* tensor arguments into a
    ``SavedVariable``, and constructing one is what fires the ambient ``saved_tensors_hooks``. It
    runs just before the checkpoint installs its own hooks, so those ambient hooks are still the
    caller's -- which is how activation offloading gets hold of a layer's inputs. A tensor nested in
    a list, or passed by keyword, is stored as a plain reference instead, reaches no hook, and is
    silently never offloaded.

    Args:
        function (Callable): The callable to run inside the checkpointed region.
        preserve_rng_state (bool): Restore the RNG state before recomputing. Defaults to True.
        use_reentrant (bool): Use reentrant checkpointing. Defaults to True.
        **kwargs (Any): Forwarded to ``function``, except ``context_fn`` which goes to
            ``torch.utils.checkpoint.checkpoint``.

    Returns:
        Any: Whatever ``function`` returns.
    """
    context_fn = kwargs.pop("context_fn", None)
    checkpoint_kwargs: dict[str, Any] = {
        "use_reentrant": use_reentrant,
        "preserve_rng_state": preserve_rng_state,
    }
    if context_fn is not None:
        checkpoint_kwargs["context_fn"] = context_fn

    flat_inputs, input_spec = tree_flatten((args, kwargs))
    output_spec: TreeSpec | None = None

    def call_with_original_signature(*replayed: Any) -> tuple[Any, ...]:
        nonlocal output_spec
        replayed_args, replayed_kwargs = tree_unflatten(list(replayed), input_spec)
        flat_outputs, output_spec = tree_flatten(function(*replayed_args, **replayed_kwargs))
        return tuple(flat_outputs)

    flat_outputs = checkpoint(call_with_original_signature, *flat_inputs, **checkpoint_kwargs)
    assert output_spec is not None, "XTuner Internal Error: checkpoint did not run the function"
    if not isinstance(flat_outputs, tuple):
        flat_outputs = (flat_outputs,)
    return tree_unflatten(list(flat_outputs), output_spec)

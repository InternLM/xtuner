"""Gradient checkpointing (activation recomputation) entry points."""

from contextlib import AbstractContextManager
from typing import Any, Callable

import torch.nn as nn
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint


__all__ = ["apply_gradient_checkpointing", "checkpoint_flattened", "install_checkpointing", "CheckpointModule"]


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
        nn.Module: ``module`` itself, now checkpointed.
    """
    # A real `context_fn` compiles fine, but forwarding torch's own `noop_context_fn` as the
    # "no policy" default does not: Dynamo's checkpoint higher-order op rejects it with
    # `NotImplementedError: ... LazyVariableTracker context_fn`. Omit the kwarg entirely instead,
    # which is also what leaves the default recompute-everything behaviour to torch.
    extra: dict[str, Any] = {} if context_fn is None else {"context_fn": context_fn}

    def checkpointed_call(original_call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return checkpoint_flattened(
            original_call,
            *args,
            preserve_rng_state=preserve_rng_state,
            use_reentrant=use_reentrant,
            **extra,
            **kwargs,
        )

    return install_checkpointing(module, checkpointed_call)


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


def install_checkpointing(module: nn.Module, checkpointed_call: Callable[..., Any]) -> nn.Module:
    """Give ``module`` a checkpointed ``__call__`` by extending its own class.

    Following ``fully_shard``, this inserts :class:`CheckpointModule` leftmost in the module's MRO
    rather than wrapping the module in a new one. The module therefore stays itself: ``isinstance``
    still holds, attributes and container protocols resolve natively, and parameter names and
    ``state_dict`` keys are untouched -- so a checkpointed model loads from, and saves into, a
    checkpoint produced without checkpointing.

    Args:
        module (nn.Module): The module to checkpoint.
        checkpointed_call (Callable): Called as ``checkpointed_call(original_call, *args, **kwargs)``;
            it is responsible for invoking ``original_call`` inside a checkpoint region.

    Returns:
        nn.Module: ``module`` itself.
    """
    cls = type(module)
    checkpoint_cls = _CHECKPOINT_CLASSES.get(cls)
    if checkpoint_cls is None:
        checkpoint_cls = type(f"Checkpoint{cls.__name__}", (CheckpointModule, cls), {})
        _CHECKPOINT_CLASSES[cls] = checkpoint_cls

    module.__class__ = checkpoint_cls
    module._checkpointed_call = checkpointed_call  # type: ignore[assignment]
    return module


class CheckpointModule:
    """Mixin that routes a module's call through a checkpoint function.

    Installed by :func:`install_checkpointing`; do not inherit from it directly.

    It overrides ``__call__`` rather than ``forward`` on purpose. Hooks registered on the module run
    inside ``nn.Module.__call__``, so replacing ``forward`` would leave them *outside* the
    checkpointed region -- and hooks that maintain per-forward state, such as the DSA top-k cache
    lifecycle, must see the recompute pass too.
    """

    _checkpointed_call: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        def original_call(*replayed_args: Any, **replayed_kwargs: Any) -> Any:
            return super(CheckpointModule, self).__call__(*replayed_args, **replayed_kwargs)  # type: ignore[misc]

        return self._checkpointed_call(original_call, *args, **kwargs)


# One checkpoint class per original class: checkpointing N layers of the same type must not build N
# classes, and two checkpointed layers of the same type should stay `isinstance`-comparable.
_CHECKPOINT_CLASSES: dict[type, type] = {}

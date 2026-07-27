"""Gradient checkpointing (activation recomputation) entry points."""

from contextlib import AbstractContextManager
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint


__all__ = [
    "apply_gradient_checkpointing",
    "apply_legacy_reentrant_checkpointing",
    "checkpoint_flattened",
    "install_checkpointing",
    "CheckpointModule",
]


ContextFn = Callable[[], tuple[AbstractContextManager, AbstractContextManager]]


def apply_gradient_checkpointing(
    module: nn.Module,
    *,
    preserve_rng_state: bool = True,
    context_fn: ContextFn | None = None,
) -> nn.Module:
    """Make ``module``'s forward recomputed during backward instead of kept in
    memory.

    Recomputation uses ``use_reentrant=False``, which is implemented with saved-tensor hooks rather
    than an ``autograd.Function``. Gradients therefore flow correctly through arbitrary forward
    signatures -- nested containers, ``TypedDict`` returns, keyword-only arguments -- none of which
    the reentrant implementation supported.

    Args:
        module (nn.Module): Module whose forward should be recomputed during backward.
        preserve_rng_state (bool): Restore the RNG state before recomputing, so dropout and other
            stochastic ops replay identically. Defaults to True.
        context_fn (Callable | None): Factory returning the ``(forward_context, recompute_context)``
            pair that ``torch.utils.checkpoint.checkpoint`` enters around the two passes. This is
            the seam for selective checkpointing: passing the contexts built by
            ``create_selective_checkpoint_contexts`` turns whole-module recompute into a per-op
            decision. Defaults to None, i.e. recompute everything.

    Returns:
        nn.Module: ``module`` itself, now checkpointed.
    """
    # A real `context_fn` compiles fine, but forwarding torch's own `noop_context_fn` as the
    # "no policy" default does not: Dynamo's checkpoint higher-order op rejects it with
    # `NotImplementedError: ... LazyVariableTracker context_fn`. Omit the kwarg entirely instead,
    # which is also what leaves the default recompute-everything behaviour to torch.
    extra: dict[str, Any] = {} if context_fn is None else {"context_fn": context_fn}

    def checkpointed_call(original_call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return checkpoint_flattened(original_call, *args, preserve_rng_state=preserve_rng_state, **extra, **kwargs)

    return install_checkpointing(module, checkpointed_call)


def apply_legacy_reentrant_checkpointing(module: nn.Module, *, preserve_rng_state: bool = True) -> nn.Module:
    """Wrap ``module`` with the legacy reentrant checkpoint implementation.

    Prefer :func:`apply_gradient_checkpointing`. Reentrant checkpointing runs the original pass
    under ``no_grad`` and only the replay with grad enabled, which is required by the MTP layers of
    DSA-based models: several logical MTP depths share one top-k cache, and only a grad-free
    original pass lets the cache advance consistently across the two passes. This function can be
    removed once the DSA top-k cache tracks the original/replay phase explicitly and the
    ``mtp_checkpoint_use_reentrant`` config switch goes away.

    Args:
        module (nn.Module): Module whose forward should be recomputed during backward.
        preserve_rng_state (bool): Restore the RNG state before recomputing. Defaults to True.

    Returns:
        nn.Module: A module that runs ``module`` under reentrant gradient checkpointing.
    """

    def checkpointed_call(original_call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return _pytree_reentrant_checkpoint(original_call, *args, preserve_rng_state=preserve_rng_state, **kwargs)

    return install_checkpointing(module, checkpointed_call)


def checkpoint_flattened(
    function: Callable[..., Any],
    *args: Any,
    preserve_rng_state: bool = True,
    **kwargs: Any,
) -> Any:
    """Run ``function`` under a non-reentrant checkpoint, with its inputs
    flattened first.

    Flattening is not about gradient correctness -- non-reentrant checkpointing handles nested
    inputs either way -- but about **who else gets to see the input tensors**.
    ``_CheckpointFrame.save_inputs`` wraps only *top-level* tensor arguments into a
    ``SavedVariable``, and constructing one is what fires the ambient ``saved_tensors_hooks``. It
    runs just before the checkpoint installs its own hooks, so those ambient hooks are still the
    caller's -- which is how activation offloading gets hold of a layer's inputs. A tensor nested in
    a list, or passed by keyword, is stored as a plain reference instead, reaches no hook, and is
    silently never offloaded.

    Args:
        function (Callable): The callable to run inside the checkpointed region.
        preserve_rng_state (bool): Restore the RNG state before recomputing. Defaults to True.
        **kwargs (Any): Forwarded to ``function``, except ``context_fn`` which goes to
            ``torch.utils.checkpoint.checkpoint``.

    Returns:
        Any: Whatever ``function`` returns.
    """
    context_fn = kwargs.pop("context_fn", None)
    checkpoint_kwargs: dict[str, Any] = {"use_reentrant": False, "preserve_rng_state": preserve_rng_state}
    if context_fn is not None:
        checkpoint_kwargs["context_fn"] = context_fn

    flat_inputs, input_spec = tree_flatten((args, kwargs))

    def call_with_original_signature(*replayed: Any) -> Any:
        replayed_args, replayed_kwargs = tree_unflatten(list(replayed), input_spec)
        return function(*replayed_args, **replayed_kwargs)

    return checkpoint(call_with_original_signature, *flat_inputs, **checkpoint_kwargs)


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


def _pytree_reentrant_checkpoint(
    function: Callable[..., Any],
    *args: Any,
    preserve_rng_state: bool = True,
    **kwargs: Any,
) -> Any:
    """让嵌套 Tensor 也成为 reentrant checkpoint 的 autograd 输入。"""
    # 原生 CheckpointFunction 只看得到顶层位置参数。例如：
    #   future_embeddings=[embedding_0, embedding_1]
    # 对它来说只是“一个 list 参数”，它看不到 list 里的两个 Tensor，也就不会 detach
    # 它们。这可能会造成反向传播的错误，因为这两个 Tensor 的梯度应该交由
    # CheckpointFunction.backward 的返回值交回原始 Tensor，而不是由他们自己来传递梯度。
    # tree_flatten 会把输入变成近似：
    #   hidden, embedding_0, embedding_1
    # 这样 checkpoint 能逐个 detach；tree_unflatten 再在 replay 前把 list 还原。
    flat_inputs, input_spec = tree_flatten((args, kwargs))

    def run_function(*replayed_flat_inputs: Any) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # 这里只还原参数结构，不会把 detached Tensor 重新连接到旧 graph；梯度由
        # CheckpointFunction.backward 的返回值交回原始 Tensor。
        replayed_args, replayed_kwargs = tree_unflatten(list(replayed_flat_inputs), input_spec)
        return function(*replayed_args, **replayed_kwargs)

    return checkpoint(run_function, *flat_inputs, use_reentrant=True, preserve_rng_state=preserve_rng_state)

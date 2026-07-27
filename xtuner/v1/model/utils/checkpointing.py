"""Gradient checkpointing (activation recomputation) entry points."""

from contextlib import AbstractContextManager
from typing import Any, Callable

import torch.nn as nn
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint


__all__ = ["apply_gradient_checkpointing", "apply_legacy_reentrant_checkpointing"]


ContextFn = Callable[[], tuple[AbstractContextManager, AbstractContextManager]]

# Name of the attribute holding the wrapped module. Downstream parameter-name normalization
# (`xtuner.v1.utils.misc.clean_param_name` and friends) strips this prefix, so it is part of the
# contract rather than an implementation detail.
_CHECKPOINT_WRAPPED_MODULE = "_checkpoint_wrapped_module"
_CHECKPOINT_PREFIX = f"{_CHECKPOINT_WRAPPED_MODULE}."


def apply_gradient_checkpointing(
    module: nn.Module,
    *,
    preserve_rng_state: bool = True,
    context_fn: ContextFn | None = None,
) -> nn.Module:
    """Wrap ``module`` so its forward is recomputed during backward instead of
    kept in memory.

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
        nn.Module: A module that runs ``module`` under gradient checkpointing.
    """
    # Dynamo lowers `checkpoint` to a higher-order op that rejects any explicitly passed
    # `context_fn`, so a compiled module must not receive one -- not even the default no-op.
    checkpoint_kwargs: dict[str, Any] = {"use_reentrant": False, "preserve_rng_state": preserve_rng_state}
    if context_fn is not None:
        checkpoint_kwargs["context_fn"] = context_fn

    def checkpointed_call(wrapped: nn.Module, *args: Any, **kwargs: Any) -> Any:
        return checkpoint(wrapped, *args, **checkpoint_kwargs, **kwargs)

    return CheckpointWrapper(module, checkpointed_call)


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

    def checkpointed_call(wrapped: nn.Module, *args: Any, **kwargs: Any) -> Any:
        return _pytree_reentrant_checkpoint(wrapped, *args, preserve_rng_state=preserve_rng_state, **kwargs)

    return CheckpointWrapper(module, checkpointed_call)


class CheckpointWrapper(nn.Module):
    """Runs a wrapped module through a checkpoint function, staying transparent
    to callers.

    Prefer the :func:`apply_gradient_checkpointing` / :func:`apply_legacy_reentrant_checkpointing`
    entry points over instantiating this directly.

    The wrapper is deliberately transparent: parameter names, ``state_dict`` keys and attribute
    lookups all skip the wrapper level, so a checkpointed module can be saved into, and loaded
    from, a checkpoint produced without checkpointing. The wrapped module's ``__call__`` -- and
    therefore any forward hook registered on it -- runs *inside* the checkpointed region, so hooks
    that maintain per-forward state (such as the DSA top-k cache lifecycle) see the recompute pass
    as well.

    Args:
        module (nn.Module): The module to run under checkpointing.
        checkpointed_call (Callable): Called as ``checkpointed_call(module, *args, **kwargs)``; it
            is responsible for invoking ``module`` inside a checkpoint region.
    """

    def __init__(self, module: nn.Module, checkpointed_call: Callable[..., Any]) -> None:
        super().__init__()
        self._checkpoint_wrapped_module = module
        self._checkpointed_call = checkpointed_call
        self._register_state_dict_hook(_strip_checkpoint_prefix)
        self.register_load_state_dict_pre_hook(_add_checkpoint_prefix)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._checkpointed_call(self._checkpoint_wrapped_module, *args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any):
        for name, param in super().named_parameters(*args, **kwargs):
            yield name.replace(_CHECKPOINT_PREFIX, ""), param

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == _CHECKPOINT_WRAPPED_MODULE:
                raise
            return getattr(super().__getattr__(_CHECKPOINT_WRAPPED_MODULE), name)

    def __getitem__(self, key: int) -> Any:
        return self._checkpoint_wrapped_module[key]  # type: ignore[index]


def _strip_checkpoint_prefix(
    module: nn.Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
) -> dict[str, Any]:
    # Saved checkpoints must be loadable by an unwrapped model, so the wrapper level is erased.
    _rename_by_prefix(state_dict, f"{prefix}{_CHECKPOINT_PREFIX}", prefix)
    return state_dict


def _add_checkpoint_prefix(
    module: nn.Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
) -> None:
    # Mirror of the save hook: a state dict written without checkpointing has to be re-prefixed
    # before it can be loaded into the wrapped module.
    _rename_by_prefix(state_dict, prefix, f"{prefix}{_CHECKPOINT_PREFIX}")


def _rename_by_prefix(state_dict: dict[str, Any], old_prefix: str, new_prefix: str) -> None:
    if old_prefix == new_prefix:
        raise ValueError("old_prefix and new_prefix must be distinct")
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        state_dict[new_prefix + key[len(old_prefix) :]] = state_dict.pop(key)


def _pytree_reentrant_checkpoint(
    function: Callable[..., Any],
    *args: Any,
    preserve_rng_state: bool = True,
    **kwargs: Any,
) -> Any:
    """让嵌套 Tensor 在 reentrant checkpoint 的出入口都成为 autograd 的一等公民。"""
    # 原生 CheckpointFunction 只看得到顶层位置参数与 tensor 返回值，嵌套容器两头都会出问题。
    #
    # 入参侧：future_embeddings=[embedding_0, embedding_1] 对它来说只是“一个 list 参数”，
    # 它看不到 list 里的两个 Tensor，也就不会 detach 它们。这可能会造成反向传播的错误，因为
    # 这两个 Tensor 的梯度应该交由 CheckpointFunction.backward 的返回值交回原始 Tensor，
    # 而不是由他们自己来传递梯度。tree_flatten 把输入摊平成
    #   hidden, embedding_0, embedding_1
    # 后 checkpoint 才能逐个 detach；tree_unflatten 再在 replay 前把 list 还原。
    #
    # 返回值侧：original 那趟在 no_grad 下执行，dict / TypedDict 里的 Tensor 因此不会被
    # autograd.Function 注册为输出，回来时既没有 grad_fn 也不 require_grad——梯度会在此处
    # 静默断掉（MTP 返回 TypedDict 时整个 mtp_block 的 grad 都是 None）。所以出口同样要摊平：
    # 交给 checkpoint 的是纯 Tensor 元组，拿回来后再按 spec 还原成原本的结构。
    flat_inputs, input_spec = tree_flatten((args, kwargs))
    output_spec: TreeSpec | None = None

    def run_function(*replayed_flat_inputs: Any) -> tuple[Any, ...]:
        # 这里只还原参数结构，不会把 detached Tensor 重新连接到旧 graph；梯度由
        # CheckpointFunction.backward 的返回值交回原始 Tensor。
        nonlocal output_spec
        replayed_args, replayed_kwargs = tree_unflatten(list(replayed_flat_inputs), input_spec)
        flat_outputs, output_spec = tree_flatten(function(*replayed_args, **replayed_kwargs))
        return tuple(flat_outputs)

    flat_outputs = checkpoint(run_function, *flat_inputs, use_reentrant=True, preserve_rng_state=preserve_rng_state)
    # original 与 replay 写入的 spec 相同，取任意一趟的结果即可。
    assert output_spec is not None, "XTuner Internal Error: reentrant checkpoint did not run the function"
    if not isinstance(flat_outputs, tuple):
        flat_outputs = (flat_outputs,)
    return tree_unflatten(list(flat_outputs), output_spec)

"""The selective activation checkpointing engine.

Drives a per-op checkpoint policy for the units a user selected. The vocabulary those units are
written in lives in :mod:`xtuner.v1.utils.selective_checkpointing`; the config resolution that turns
a user's ``recompute_cfg`` into resolved targets lives with the model configs.

A unit reaches the policy by one of two routes, and the policy answers the same question either way
-- "does this op belong to a kept unit":

- :class:`~xtuner.v1.utils.selective_checkpointing.KeptOps` targets are recognised from the op
  itself, so they need nothing installed and work identically inside and outside compiled code.
- :class:`~xtuner.v1.utils.selective_checkpointing.KeptCallables` targets are recognised from a
  ``ContextVar`` the engine's wrapper sets around the callable. That only works because the model
  layer has also taken those callables out of the compiled set: a ``ContextVar`` set inside compiled
  code is neither written nor readable.
"""

from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from xtuner.v1.utils import log_rank0
from xtuner.v1.utils.selective_checkpointing import RecomputeUnit, active_recompute_unit, recompute_unit

from .checkpointing import CheckpointWrapper, apply_gradient_checkpointing


__all__ = ["apply_selective_checkpointing", "in_recompute_unit", "resolve_kept_ops"]


def apply_selective_checkpointing(
    module: nn.Module,
    kept_ops: frozenset = frozenset(),
    *,
    keeps_any_unit: bool = False,
    preserve_rng_state: bool = True,
) -> nn.Module:
    """Wrap ``module`` so its forward is recomputed during backward, keeping
    the selected units.

    An op is kept when it is one of ``kept_ops``, or when it runs inside a callable the model layer
    wrapped with :func:`in_recompute_unit`. Everything else in the layer is recomputed. With nothing
    selected this is plain full recompute -- the degenerate case of the same mechanism -- so a
    sharding path can call this for every layer ``recompute_ratio`` selects.

    Args:
        module (nn.Module): The layer to checkpoint.
        kept_ops (frozenset): Op overloads to keep wherever they run, as resolved from the user's
            ``recompute_cfg``. Defaults to keeping none.
        keeps_any_unit (bool): Whether any unit at all was selected, including ones installed as
            callable wrappers rather than op names. Defaults to False, which skips the per-op policy
            entirely -- torch's own default already recomputes everything, so running a policy to
            reach the same answer would only put a dispatch mode in the way of every op.
        preserve_rng_state (bool): Restore the RNG state before recomputing, so dropout and other
            stochastic ops replay identically. Defaults to True.

    Returns:
        nn.Module: The checkpoint-wrapped layer, transparent to parameter names and ``state_dict``.
    """
    if not keeps_any_unit:
        return apply_gradient_checkpointing(module, preserve_rng_state=preserve_rng_state)

    return CheckpointWrapper(module, partial(_run_checkpointed_region, kept_ops, preserve_rng_state))


def in_recompute_unit(unit: RecomputeUnit, func: Any) -> Any:
    """Wrap ``func`` so everything it dispatches belongs to ``unit``.

    Installed by the model layer on each callable a
    :class:`~xtuner.v1.utils.selective_checkpointing.KeptCallables` target names, together with the
    exclusion from compilation that makes the marker readable at all.

    Args:
        unit (RecomputeUnit): The unit this callable implements.
        func (Any): The callable to wrap.

    Returns:
        Any: A callable with the same signature that marks its own execution.
    """

    # No `functools.wraps`: it sets `__wrapped__`, which Dynamo resolves through to the unbound
    # function and then loses `self` ("Missing required positional argument: self").
    def in_unit(*args: Any, **kwargs: Any) -> Any:
        with recompute_unit(unit):
            return func(*args, **kwargs)

    in_unit.__name__ = getattr(func, "__name__", "in_unit")
    in_unit.__qualname__ = getattr(func, "__qualname__", "in_unit")
    in_unit.__module__ = getattr(func, "__module__", __name__)
    return in_unit


def resolve_kept_ops(names: tuple[str, ...]) -> frozenset:
    """Resolve qualified op names to overloads, skipping ones this build does
    not register.

    A model may list several backends' spellings of the same kernel -- flash-attention v2 and v3, a
    vendor kernel and its fallback -- and only one of them exists at runtime.

    Args:
        names (tuple[str, ...]): Qualified op names such as ``"flash_attn::_flash_attn_varlen_forward_v3"``.

    Returns:
        frozenset: The overloads that resolve in this build.
    """
    resolved = set()
    for name in names:
        namespace, _, opname = name.partition("::")
        packet = getattr(getattr(torch.ops, namespace, None), opname, None)
        if packet is None:
            log_rank0.debug(f"Selective checkpointing: {name} is not registered in this build, skipping it.")
            continue
        resolved.add(packet.default)
    return frozenset(resolved)


# Ops from these namespaces are never kept, whatever unit they fall in. See `_checkpoint_policy`.
_NEVER_KEPT_NAMESPACES = ("c10d", "_c10d_functional")

# Mutating ops that leave tensor *values* alone, so a kept unit may contain them. Anything else with
# a mutable schema goes through `_reject_non_replayable_op_in_kept_unit`.
_VALUE_PRESERVING_MUTATING_OPS = frozenset({torch.ops.aten.record_stream.default})


def _run_checkpointed_region(
    kept_ops: frozenset,
    preserve_rng_state: bool,
    module: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Any:
    # `context_fn` must be a module-level function or a `functools.partial` of one: Dynamo's
    # checkpoint higher-order op rejects anything else (lambdas, closures, bound methods) with
    # `NotImplementedError: ... LazyVariableTracker context_fn`. Keep it that way.
    return checkpoint(
        module,
        *args,
        use_reentrant=False,
        preserve_rng_state=preserve_rng_state,
        context_fn=partial(_selective_checkpoint_contexts, kept_ops),
        **kwargs,
    )


def _selective_checkpoint_contexts(kept_ops: frozenset) -> tuple[Any, Any]:
    return create_selective_checkpoint_contexts(partial(_checkpoint_policy, kept_ops))


def _checkpoint_policy(kept_ops: frozenset, ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
    if op not in kept_ops and active_recompute_unit() is None:
        return CheckpointPolicy.MUST_RECOMPUTE

    # Keeping a collective would elide it from the recompute pass. That is only correct while the op
    # that allocated its destination buffer is kept too, and a unit boundary falling between the
    # allocation and the collective would leave the recompute reading an uninitialised buffer --
    # silently, and differently on each rank.
    if op.namespace in _NEVER_KEPT_NAMESPACES:
        return CheckpointPolicy.MUST_RECOMPUTE

    if op._schema.is_mutable:
        _warn_non_replayable_op_in_kept_unit(op)
        # Never kept: what the recompute pass would get back from the cache is whatever the last
        # writer left behind.
        return CheckpointPolicy.MUST_RECOMPUTE

    return CheckpointPolicy.MUST_SAVE


def _warn_non_replayable_op_in_kept_unit(op: Any) -> None:
    """Name an in-place op inside a kept unit, once, as a lead for a later
    failure.

    A kept unit has to survive being replayed on top of its own results: the recompute pass gets the *forward's*
    tensors back from the cache, so a read-modify-write op such as ``add_`` would apply its update a second time to a
    value that already includes it -- finite loss, no error, wrong gradients.

    Whether that actually happens is not decidable from the op's schema, only from whether the tensor it writes to
    was cached. Torch decides exactly that, by version counter, and raises "Tensor cached during selective activation
    checkpoint has been mutated" when it does (``torch/utils/checkpoint.py``, ``_VersionWrapper``). Refusing here on
    the schema alone would reject units that are perfectly safe -- a library mutating its own buffers, which nothing
    in the unit ever cached.

    So this only reports. It runs in the forward pass and names the op, which torch's check -- raised in backward,
    naming only the tensor -- cannot.

    Writing through ``out`` is not reported at all: those overwrite the destination with a value that does not depend
    on what was there, which is what inductor's extern kernels do, and replaying them is idempotent.
    """
    if op in _VALUE_PRESERVING_MUTATING_OPS or _writes_only_through_out(op):
        return
    if op in _REPORTED_MUTATING_OPS:
        return
    _REPORTED_MUTATING_OPS.add(op)
    log_rank0.warning(
        f"Selective checkpointing: the in-place op {op} runs inside a kept unit. It is recomputed rather than kept, "
        f"which is safe unless it writes to a tensor the unit did keep -- torch reports that case as "
        f'"Tensor cached during selective activation checkpoint has been mutated".'
    )


# Reported once per op, not once per layer or per step.
_REPORTED_MUTATING_OPS: set = set()


def _writes_only_through_out(op: Any) -> bool:
    return all(
        argument.name == "out"
        for argument in op._schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    )

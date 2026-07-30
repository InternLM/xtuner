"""The selective activation checkpointing engine.

Applies a recompute strategy to one decoder layer: the whole layer goes under a single
``torch.utils.checkpoint`` region whose per-op policy keeps the marker intervals the config layer
selected and recomputes everything else. The vocabulary and the marker session it drives live in
:mod:`xtuner.v1.utils.selective_checkpointing`, below the model layer, because the marker calls
themselves sit in ``xtuner.v1.module`` forwards.

Granularity note, because it decides which units are worth declaring for a given model: a region is
addressable only where its **contents** run in eager python, not merely its endpoints. A
``torch.compile``d region executes as fused kernels whose ops never reach the per-op policy, so an
interval enclosing one keeps nothing even when both its markers fire -- ``SAVE_MLP`` under EP is the
case to remember, its markers sitting in the uncompiled layer body while the region encloses the
compiled ``_shared_experts_forward``. In eager, every region is addressable.
"""

from collections.abc import Sequence
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from xtuner.v1.utils import log_rank0
from xtuner.v1.utils.selective_checkpointing import (
    MarkerInterval,
    active_marker_session,
    declare_selective_regions,
    marker_session,
)

from .checkpointing import CheckpointWrapper, apply_gradient_checkpointing


__all__ = ["apply_selective_checkpointing"]


def apply_selective_checkpointing(
    module: nn.Module,
    intervals: Sequence[MarkerInterval] = (),
    *,
    owner: nn.Module | None = None,
    preserve_rng_state: bool = True,
    layer_compiled_as_one_region: bool = False,
) -> nn.Module:
    """Apply to ``module`` the recompute strategy it supports, keeping
    ``intervals`` resident.

    Ops executed while one of ``intervals`` is open are kept; every other op in the layer is
    recomputed during backward. An empty ``intervals`` is the degenerate case of the same mechanism
    -- nothing kept, everything recomputed -- so a sharding path can call this for every layer
    ``recompute_ratio`` selects, whether or not the model declares any recompute regions.

    Intervals need not be balanced: an ``end`` marker that never runs only widens the kept region.
    Both the kept and the recomputed path are numerically exact, so marker bookkeeping can cost
    memory but can never change gradients.

    Args:
        module (nn.Module): The layer to checkpoint.
        intervals (Sequence[MarkerInterval]): Half-open ``[start, end)`` marker intervals to keep
            resident, as resolved from the user's ``recompute_cfg``. Defaults to keeping nothing.
        owner (nn.Module | None): The model these layers belong to. Diagnostics are aggregated over
            it, so that an interval which keeps nothing in one layer type but works in another is
            not reported. Defaults to None, which diagnoses each layer on its own.
        preserve_rng_state (bool): Restore the RNG state before recomputing, so dropout and other
            stochastic ops replay identically. Defaults to True.
        layer_compiled_as_one_region (bool): Whether the caller compiles this whole layer as a
            single ``torch.compile`` region. Such a layer never runs its forward in eager python, so
            no marker ever fires and the intervals silently keep nothing; passing True lets that be
            reported instead of leaving the user to wonder why memory did not move. Defaults to
            False.

    Returns:
        nn.Module: The checkpoint-wrapped layer, transparent to parameter names and ``state_dict``.
    """
    kept_intervals = tuple(intervals)

    if not kept_intervals:
        return apply_gradient_checkpointing(module, preserve_rng_state=preserve_rng_state)

    if layer_compiled_as_one_region:
        _warn_intervals_unsupported(module, kept_intervals, "it is compiled as a single region")

    # Not routed through `apply_gradient_checkpointing` because the marker session has to open
    # *inside* the checkpointed region: `use_reentrant=False` re-runs this forward to recompute, and
    # a session opened around the checkpoint call would be long gone by then.
    declare_selective_regions(owner, kept_intervals)
    checkpointed_call = partial(_run_checkpointed_region, kept_intervals, preserve_rng_state, owner)
    return CheckpointWrapper(module, checkpointed_call)


# Ops from these namespaces are never kept, whatever the markers say. See `_checkpoint_policy`.
_NEVER_KEPT_NAMESPACES = ("c10d", "_c10d_functional")

# Mutating ops that leave tensor *values* alone, so a kept region may contain them. Anything else
# with a mutable schema goes through `_reject_non_replayable_op_in_kept_region`.
_VALUE_PRESERVING_MUTATING_OPS = frozenset({torch.ops.aten.record_stream.default})

# Reported once per distinct diagnosis, not once per layer. See `_warn_intervals_unsupported`.
_REPORTED_UNSUPPORTED_DIAGNOSES: set[tuple[type, tuple[MarkerInterval, ...], str]] = set()


def _run_checkpointed_region(
    intervals: tuple[MarkerInterval, ...],
    preserve_rng_state: bool,
    owner: nn.Module | None,
    module: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Any:
    # `context_fn` must be a module-level function or a `functools.partial` of one: Dynamo's
    # checkpoint higher-order op rejects anything else (lambdas, closures, bound methods) with
    # `NotImplementedError: ... LazyVariableTracker context_fn`. Keep it that way.
    return checkpoint(
        partial(_forward_with_marker_session, module, intervals, owner),
        *args,
        use_reentrant=False,
        preserve_rng_state=preserve_rng_state,
        context_fn=_selective_checkpoint_contexts,
        **kwargs,
    )


def _forward_with_marker_session(
    module: nn.Module,
    intervals: tuple[MarkerInterval, ...],
    owner: nn.Module | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    # Calling `module` rather than `module.forward` keeps the module's hooks inside the checkpointed
    # region, so a hook that maintains per-forward state sees the recompute pass too.
    #
    # Skipped under compile for the same reason `checkpoint_record` is: Dynamo cannot trace
    # ContextVar. Nothing is lost by it -- the markers are erased from the traced graph, so the
    # policy would see an empty session anyway and recompute the whole region.
    if torch.compiler.is_compiling():
        return module(*args, **kwargs)

    with marker_session(intervals, owner=owner) as session:
        output = module(*args, **kwargs)

    # Only a pass that ran to the end counts: with `set_checkpoint_early_stop` on, the recompute
    # pass is cut short by an exception once the last needed tensor is repacked, and folding a
    # truncated pass into the diagnostics would blame regions that simply had not come up yet.
    session.finish()
    return output


def _selective_checkpoint_contexts() -> tuple[Any, Any]:
    return create_selective_checkpoint_contexts(_checkpoint_policy)


def _checkpoint_policy(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
    session = active_marker_session()
    if session is None or not session.keeping:
        return CheckpointPolicy.MUST_RECOMPUTE

    # Keeping a collective would elide it from the recompute pass. That is only correct while the op
    # that allocated its destination buffer is kept too, and an interval boundary falling between
    # the allocation and the collective would leave the recompute reading an uninitialised buffer --
    # silently, and differently on each rank.
    if op.namespace in _NEVER_KEPT_NAMESPACES:
        return CheckpointPolicy.MUST_RECOMPUTE

    if op._schema.is_mutable:
        _reject_non_replayable_op_in_kept_region(op)
        # Keeping a mutating op is unsound from the other side too: it writes into an argument, so
        # what the recompute pass gets back from the cache is whatever the last writer left behind.
        # Under torch.compile inductor's `out=` extern kernels reuse buffers and torch catches this
        # as "Tensor cached during selective activation checkpoint has been mutated".
        return CheckpointPolicy.MUST_RECOMPUTE

    session.note_kept()
    return CheckpointPolicy.MUST_SAVE


def _reject_non_replayable_op_in_kept_region(op: Any) -> None:
    # A kept region must survive being replayed on top of its own results. The recompute pass gets
    # the *forward's* tensors back from the cache, so a read-modify-write op such as `add_` applies
    # its update a second time to a value that already includes it: gradients then differ from the
    # recomputed path with a finite loss and no error anywhere. Refuse rather than compute silently
    # wrong gradients.
    #
    # Writing through the `out` argument is the exception, and it is what the policy sees most of
    # the time: inductor's extern kernels are `out=` variants. Those overwrite the destination with
    # a value that does not depend on what was there, so replaying them is idempotent. An op that
    # writes through any other argument is rejected even when it happens to be idempotent
    # (`copy_`), because the schema cannot tell the two apart.
    if op in _VALUE_PRESERVING_MUTATING_OPS or _writes_only_through_out(op):
        return
    raise RuntimeError(
        f"Selective checkpointing cannot keep a region containing the in-place op {op}. Move the "
        f"`checkpoint_record` boundary so that the in-place write falls outside the kept interval, or "
        f"rewrite it out of place."
    )


def _writes_only_through_out(op: Any) -> bool:
    return all(
        argument.name == "out"
        for argument in op._schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    )


def _warn_intervals_unsupported(module: nn.Module, intervals: tuple[MarkerInterval, ...], reason: str) -> None:
    # Reported here rather than left to `_report_pass`, which never speaks for such a layer: its
    # forward never runs in eager python, so no session ever opens and no pass ever completes.
    #
    # Deduplicated on the whole diagnosis, not on the layer alone: every layer of a model produces
    # the same one, but a second model in the same process with different intervals -- an RL
    # reference model, a compose model's other tower -- has a diagnosis of its own to report.
    layer = type(module)
    diagnosis = (layer, intervals, reason)
    if diagnosis in _REPORTED_UNSUPPORTED_DIAGNOSES:
        return
    _REPORTED_UNSUPPORTED_DIAGNOSES.add(diagnosis)
    log_rank0.warning(
        f"Selective checkpointing: {layer.__name__} cannot keep recompute regions resident because {reason}, so "
        f"the intervals {list(intervals)} have no effect and every selected layer is recomputed whole."
    )

"""Region-level selective activation checkpointing (SAC).

The three SAC layers meet here:

- Model authors call :func:`checkpoint_record` inside ``forward`` to name addressable semantic
  boundaries, and declare a :data:`RecomputeIntervalMap` mapping each :class:`RecomputeUnit` they
  support to the marker intervals that implement it for their architecture.
- Users select :class:`RecomputeUnit` members in the model config; they never see marker strings.
- The sharding paths hand the resolved intervals to :func:`apply_selective_checkpointing`, which
  wraps one layer in a single checkpoint whose per-op policy keeps those intervals resident.

Granularity note, because it decides which units are worth declaring for a given model: markers
delimit regions only where the marked code runs in eager python. A ``torch.compile``d region
executes as one unit and its markers are folded away while it is traced, so an interval whose
endpoints fall inside a compiled region has no effect and that region is recomputed whole.
Intervals whose endpoints fall *between* compiled regions work in both modes; in eager, all do.
"""

import contextvars
from collections.abc import Sequence
from functools import partial
from typing import Any, TypeAlias

import torch
import torch.nn as nn
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from xtuner.v1.module.attention.dsa_topk_sharing import uses_dsa_topk_lifecycle
from xtuner.v1.utils import log_rank0
from xtuner.v1.utils.enum_helper import StrEnum

from .checkpointing import CheckpointWrapper, apply_gradient_checkpointing, apply_legacy_reentrant_checkpointing


__all__ = [
    "RecomputeUnit",
    "MarkerInterval",
    "RecomputeIntervalMap",
    "apply_selective_checkpointing",
    "checkpoint_record",
]


class RecomputeUnit(StrEnum):
    """Semantic units of activation that may be kept resident instead of
    recomputed.

    Each member names a *class of sub-structure* whose activations are worth keeping in memory because recomputing it
    is expensive relative to what it costs to store. A unit selected by the user means "do not recompute this part";
    everything not selected is recomputed. Which marker intervals a unit resolves to is architecture-specific and is
    declared per model, so the same unit can cover different regions in different models.

    Members are strings so pydantic round-trips them to readable names in serialized configs.
    """

    SAVE_ATTN = "save_attn"
    """Keep the attention core: the flash-attention / SDPA call and its immediate surroundings."""

    SAVE_MOE_GATE = "save_moe_gate"
    """Keep the MoE router: gating projection, top-k selection, and routing weights."""

    SAVE_MOE_DISPATCH = "save_moe_dispatch"
    """Keep the expert dispatch / combine communication region."""

    SAVE_MLP = "save_mlp"
    """Keep the dense MLP / shared-expert region."""


MarkerInterval: TypeAlias = tuple[str, str]
"""A half-open ``[start, end)`` interval between two :func:`checkpoint_record`
marker names.

Ops executed at or after the ``start`` marker and strictly before the ``end`` marker belong to the
interval. ``end`` is the name of the marker that begins the *next* region, which is why the
interval is half-open: regions are delimited by points, not by explicit closing markers.

Intervals are resolved by program order at runtime, so an interval may span module boundaries (its
``start`` in one module's ``forward`` and its ``end`` in a sibling module's). Unbalanced intervals
-- for instance an ``end`` marker that sits on a branch the forward did not take -- are not an
error: they only widen the region that stays resident. Both the kept and the recomputed paths are
numerically exact, so marker bookkeeping can never corrupt gradients.
"""


RecomputeIntervalMap: TypeAlias = dict[RecomputeUnit, list[MarkerInterval]]
"""Per-model declaration of how each supported :class:`RecomputeUnit` maps to
marker intervals.

Models expose this as a ``default_recompute_cfg`` property, mirroring ``default_compile_cfg``. A
unit absent from the mapping is simply not supported by that architecture and contributes no
intervals.
"""


def apply_selective_checkpointing(
    module: nn.Module,
    intervals: Sequence[MarkerInterval] = (),
    *,
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
    intervals = tuple(intervals)

    if uses_dsa_topk_lifecycle(module):
        # DSA cross-layer top-k sharing recognises a checkpoint's original pass by grad being
        # disabled, which only the reentrant implementation provides: under the non-reentrant one
        # both passes run with grad enabled, so the cache is never marked active and the shared
        # top-k is never released. Such layers get whole-layer reentrant recompute and no regions.
        # Remove this branch once the cache takes its phase from the checkpoint itself rather than
        # from `torch.is_grad_enabled()`.
        if intervals:
            _warn_intervals_unsupported(module, intervals, "it uses DSA cross-layer top-k sharing")
        return apply_legacy_reentrant_checkpointing(module, preserve_rng_state=preserve_rng_state)

    if not intervals:
        return apply_gradient_checkpointing(module, preserve_rng_state=preserve_rng_state)

    if layer_compiled_as_one_region:
        _warn_intervals_unsupported(module, intervals, "it is compiled as a single region")

    # Not routed through `apply_gradient_checkpointing` because the marker session has to open
    # *inside* the checkpointed region: `use_reentrant=False` re-runs this forward to recompute, and
    # a session opened around the checkpoint call would be long gone by then.
    checkpointed_call = partial(_run_checkpointed_region, intervals, preserve_rng_state)
    return CheckpointWrapper(module, checkpointed_call)


def checkpoint_record(name: str) -> None:
    """Mark an addressable semantic boundary in the current forward pass.

    This is a point marker, not a region: it records "execution reached ``name`` here". Regions are
    formed by pairing markers into :data:`MarkerInterval` s, which is why a marker can open a region
    in one module and have it closed by a marker in another -- ordering is by runtime program order,
    not lexical scope.

    Outside an active SAC session the call is a no-op, so models may be instrumented independently
    of whether selective checkpointing is enabled. It carries no autograd semantics and is safe to
    call during recomputation.

    Args:
        name (str): Marker name, unique within a layer's forward. Use a dotted, structural name
            such as ``"attn.core"`` or ``"moe.dispatch"`` so that ``default_recompute_cfg`` entries
            read as coordinates into the architecture.
    """
    # The marker session is backed by contextvars, which Dynamo cannot trace: reading a ContextVar
    # inside a `fullgraph=True` region is a hard compile error rather than a graph break, and xtuner
    # compiles several MoE forward methods that way. Dynamo constant-folds this check, so the body
    # never enters the graph and instrumented models stay compilable.
    #
    # This branch must stay free of side effects for the same reason -- a global mutation or a log
    # call here would break `fullgraph=True`. Markers that turn out to be inert are reported from
    # `_MarkerSession.report_unreached`, which runs in eager python.
    if torch.compiler.is_compiling():
        return

    session = _MARKER_SESSION.get()
    if session is not None:
        session.record(name)


_MARKER_SESSION: contextvars.ContextVar["_MarkerSession | None"] = contextvars.ContextVar(
    "xtuner_sac_marker_session", default=None
)

# Ops from these namespaces are never kept, whatever the markers say. See `_checkpoint_policy`.
_NEVER_KEPT_NAMESPACES = ("c10d", "_c10d_functional")

# Mutating ops that leave tensor *values* alone, so a kept region may contain them. Anything else
# with a mutable schema is rejected by `_reject_in_place_op_in_kept_region`.
_VALUE_PRESERVING_MUTATING_OPS = frozenset({torch.ops.aten.record_stream.default})

# Diagnostics fire once per distinct cause, not once per layer or per step.
_REPORTED_UNREACHED_INTERVALS: set["MarkerInterval"] = set()
_REPORTED_UNSUPPORTED_LAYERS: set[str] = set()


class _MarkerSession:
    """Tracks which marker intervals are open at the current point of one pass
    through a checkpointed layer."""

    def __init__(self, intervals: tuple[MarkerInterval, ...]) -> None:
        self._intervals = intervals
        self._open: set[MarkerInterval] = set()
        self._recorded: set[str] = set()

    @property
    def keeping(self) -> bool:
        # Overlapping and nested intervals are defined as "any interval open => keep", which is why
        # this is set emptiness rather than a paired counter: no pairing is asserted anywhere.
        return bool(self._open)

    def record(self, name: str) -> None:
        self._recorded.add(name)
        for interval in self._intervals:
            start, end = interval
            if name == start:
                self._open.add(interval)
            elif name == end:
                self._open.discard(interval)

    def report_unreached(self) -> None:
        # An `end` that never runs is legal -- it only widens the kept region. A `start` that never
        # runs means the interval kept nothing, and the user sees no memory change with no way to
        # find out why. Two causes produce this and cannot be told apart from here: a marker name
        # that this architecture does not have, and a marker that exists but sits inside a compiled
        # region, where markers are folded away.
        for interval in self._intervals:
            if interval[0] in self._recorded or interval in _REPORTED_UNREACHED_INTERVALS:
                continue
            _REPORTED_UNREACHED_INTERVALS.add(interval)
            log_rank0.warning(
                f"Selective checkpointing: marker {interval[0]!r} of interval {interval} never ran, so this "
                f"interval keeps nothing resident and its region is recomputed. Either the model does not record "
                f"that marker, or it records it inside a torch.compile'd region, where markers have no effect."
            )


def _run_checkpointed_region(
    intervals: tuple[MarkerInterval, ...],
    preserve_rng_state: bool,
    module: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Any:
    # `context_fn` must be a module-level function or a `functools.partial` of one: Dynamo's
    # checkpoint higher-order op rejects anything else (lambdas, closures, bound methods) with
    # `NotImplementedError: ... LazyVariableTracker context_fn`. Keep it that way.
    return checkpoint(
        partial(_forward_with_marker_session, module, intervals),
        *args,
        use_reentrant=False,
        preserve_rng_state=preserve_rng_state,
        context_fn=_selective_checkpoint_contexts,
        **kwargs,
    )


def _forward_with_marker_session(
    module: nn.Module,
    intervals: tuple[MarkerInterval, ...],
    *args: Any,
    **kwargs: Any,
) -> Any:
    # Calling `module` rather than `module.forward` keeps its hooks inside the checkpointed region,
    # which the DSA top-k cache lifecycle depends on.
    #
    # Skipped under compile for the same reason `checkpoint_record` is: Dynamo cannot trace
    # ContextVar. Nothing is lost by it -- the markers are erased from the traced graph, so the
    # policy would see an empty session anyway and recompute the whole region.
    if torch.compiler.is_compiling():
        return module(*args, **kwargs)

    session = _MarkerSession(intervals)
    token = _MARKER_SESSION.set(session)
    try:
        output = module(*args, **kwargs)
    finally:
        _MARKER_SESSION.reset(token)
    # Only a pass that ran to the end can say a marker never ran: with `set_checkpoint_early_stop`
    # on, the recompute pass is cut short by an exception once the last needed tensor is repacked,
    # and reporting from there would blame markers that simply had not come up yet.
    session.report_unreached()
    return output


def _selective_checkpoint_contexts() -> tuple[Any, Any]:
    return create_selective_checkpoint_contexts(_checkpoint_policy)


def _checkpoint_policy(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
    session = _MARKER_SESSION.get()
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
    # Reported here rather than left to `_MarkerSession.report_unreached`, which never speaks for
    # these layers: neither a compiled-as-one-region layer nor a reentrant one ever opens a session.
    layer = type(module).__name__
    if layer in _REPORTED_UNSUPPORTED_LAYERS:
        return
    _REPORTED_UNSUPPORTED_LAYERS.add(layer)
    log_rank0.warning(
        f"Selective checkpointing: {layer} cannot keep recompute regions resident because {reason}, so the "
        f"intervals {list(intervals)} have no effect and every selected layer is recomputed whole."
    )

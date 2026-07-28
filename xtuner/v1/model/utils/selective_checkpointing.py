"""Region-level selective activation checkpointing (SAC).

The three SAC layers meet here:

- Model authors call :func:`checkpoint_record` inside ``forward`` to name addressable semantic
  boundaries, and declare a :data:`RecomputeIntervalMap` mapping each :class:`RecomputeUnit` they
  support to the marker intervals that implement it for their architecture.
- Users select :class:`RecomputeUnit` members in the model config; they never see marker strings.
- The sharding paths hand the resolved intervals to :func:`apply_selective_checkpointing`, which
  wraps one layer in a single checkpoint whose per-op policy keeps those intervals resident.

Granularity note, because it decides which units are worth declaring for a given model: a region is
addressable only where its **contents** run in eager python, not merely its endpoints. A
``torch.compile``d region executes as fused kernels whose ops never reach the per-op policy, so an
interval enclosing one keeps nothing even when both its markers fire -- ``SAVE_MLP`` under EP is the
case to remember, its markers sitting in the uncompiled layer body while the region encloses the
compiled ``_shared_experts_forward``. In eager, every region is addressable.
"""

import contextvars
import weakref
from collections.abc import Sequence
from functools import partial
from typing import Any, TypeAlias

import torch
import torch.nn as nn
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from xtuner.v1.utils import log_rank0
from xtuner.v1.utils.enum_helper import StrEnum

from .checkpointing import CheckpointWrapper, apply_gradient_checkpointing


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
    intervals = tuple(intervals)

    if not intervals:
        return apply_gradient_checkpointing(module, preserve_rng_state=preserve_rng_state)

    if layer_compiled_as_one_region:
        _warn_intervals_unsupported(module, intervals, "it is compiled as a single region")

    # Not routed through `apply_gradient_checkpointing` because the marker session has to open
    # *inside* the checkpointed region: `use_reentrant=False` re-runs this forward to recompute, and
    # a session opened around the checkpoint call would be long gone by then.
    _declare_selective_regions(owner, intervals)
    checkpointed_call = partial(_run_checkpointed_region, intervals, preserve_rng_state, owner)
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

# Diagnostics accumulate per model and are held weakly, so they neither keep models alive nor let
# one model's conclusions silence another's.
_DIAGNOSTICS: "weakref.WeakKeyDictionary[nn.Module, _OwnerDiagnostics]" = weakref.WeakKeyDictionary()
_REPORTED_UNSUPPORTED_DIAGNOSES: set[tuple[type, tuple["MarkerInterval", ...], str]] = set()


class _MarkerSession:
    """Tracks which marker intervals are open at the current point of one pass
    through a checkpointed layer."""

    def __init__(self, intervals: tuple[MarkerInterval, ...], owner: nn.Module | None = None) -> None:
        self._intervals = intervals
        self._owner = owner
        self._open: set[MarkerInterval] = set()
        self._recorded: set[str] = set()
        self._kept: set[MarkerInterval] = set()

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

    def note_kept(self) -> None:
        # The only evidence that a region is addressable at all. Markers merely delimit; an interval
        # whose contents run inside a compiled region has both endpoints fire and still keeps
        # nothing, because those ops execute as fused kernels and never reach the policy.
        self._kept |= self._open

    def finish(self) -> None:
        _report_pass(self._owner, self._intervals, self._recorded, self._kept)


class _OwnerDiagnostics:
    def __init__(self) -> None:
        self.expected_layers = 0
        self.completed_layers = 0
        self.recorded_markers: set[str] = set()
        self.kept_intervals: set[MarkerInterval] = set()
        self.declared_intervals: set[MarkerInterval] = set()
        self.reported: set[MarkerInterval] = set()


def _declare_selective_regions(owner: nn.Module | None, intervals: tuple[MarkerInterval, ...]) -> None:
    # Diagnostics are only trustworthy once every layer has run: a marker missing from one layer
    # type is normal when the intervals span several -- `mlp.begin` never runs in a MoE layer and
    # `moe.gate.begin` never runs in a dense one -- and warning per layer fires on models that are
    # configured correctly. This is how the diagnostics know how many layers to wait for.
    if owner is None or not intervals:
        return
    state = _DIAGNOSTICS.get(owner)
    if state is None:
        state = _OwnerDiagnostics()
        _DIAGNOSTICS[owner] = state
    state.expected_layers += 1


def _report_pass(
    owner: nn.Module | None,
    intervals: tuple[MarkerInterval, ...],
    recorded: set[str],
    kept: set[MarkerInterval],
) -> None:
    state = _DIAGNOSTICS.get(owner) if owner is not None else None
    if state is None:
        # No owner declared these regions -- a direct caller, or a test. Diagnose the region alone.
        state = _OwnerDiagnostics()
        state.expected_layers = 1

    state.completed_layers += 1
    state.recorded_markers |= recorded
    state.kept_intervals |= kept
    state.declared_intervals |= set(intervals)

    # Wait for a full pass over the owner's layers before concluding anything: only then has every
    # layer type had its chance to record a marker and keep an op.
    if state.completed_layers < state.expected_layers:
        return

    for interval in sorted(state.declared_intervals):
        if interval in state.kept_intervals or interval in state.reported:
            continue
        state.reported.add(interval)
        if interval[0] not in state.recorded_markers:
            log_rank0.warning(
                f"Selective checkpointing: marker {interval[0]!r} of interval {interval} never ran, so this "
                f"interval keeps nothing resident and its region is recomputed. Either the model does not record "
                f"that marker, or it records it inside a torch.compile'd region, where markers have no effect."
            )
        else:
            log_rank0.warning(
                f"Selective checkpointing: interval {interval} opened but kept nothing resident, so its region is "
                f"recomputed. Its contents run inside a torch.compile'd region or consist only of ops that are "
                f"never kept, such as collectives; markers delimiting a compiled region do not make it addressable."
            )


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

    session = _MarkerSession(intervals, owner)
    token = _MARKER_SESSION.set(session)
    try:
        output = module(*args, **kwargs)
    finally:
        _MARKER_SESSION.reset(token)
    # Only a pass that ran to the end counts: with `set_checkpoint_early_stop` on, the recompute
    # pass is cut short by an exception once the last needed tensor is repacked, and folding a
    # truncated pass into the diagnostics would blame regions that simply had not come up yet.
    session.finish()
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
    # Reported here rather than left to `_MarkerSession.report_unreached`, which never speaks for
    # such a layer: its forward never runs in eager python, so no session ever opens.
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

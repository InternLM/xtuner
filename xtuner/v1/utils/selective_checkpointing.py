"""Shared contract for region-level selective activation checkpointing (SAC).

This module holds the vocabulary that the three SAC layers agree on and nothing else:

- Model authors call :func:`checkpoint_record` inside ``forward`` to name addressable semantic
  boundaries, and declare a :data:`RecomputeIntervalMap` mapping each :class:`RecomputeUnit` they
  support to the marker intervals that implement it for their architecture.
- Users select :class:`RecomputeUnit` members in the model config; they never see marker strings.
- The SAC engine turns the selected units into intervals and drives a per-op checkpoint policy.

It lives under ``xtuner.v1.utils`` rather than next to the models because the marker calls sit in
``xtuner.v1.module`` forwards while the unit vocabulary is consumed by ``xtuner.v1.model`` configs;
a home inside either package would make the two import each other.

The vocabulary and the marker session live here -- the session because :func:`checkpoint_record`
is what reads it, and that call sits in ``xtuner.v1.module``. The checkpoint policy and the wrapping
that drives a session live in ``xtuner.v1.model.utils.selective_checkpointing``, which is free to
import from the model and module layers; the config resolution that turns user selections into
intervals lives with the model configs.
"""

import contextvars
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeAlias

import torch

from .enum_helper import StrEnum
from .logger import log_rank0


__all__ = [
    "RecomputeUnit",
    "MarkerInterval",
    "RecomputeIntervalMap",
    "MarkerSession",
    "active_marker_session",
    "declare_selective_regions",
    "marker_session",
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
interval.

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
            such as ``"attn.begin"`` or ``"moe.dispatch.end"`` so that ``default_recompute_cfg``
            entries read as coordinates into the architecture.
    """
    # The marker session is backed by contextvars, which Dynamo cannot trace: reading a ContextVar
    # inside a `fullgraph=True` region is a hard compile error rather than a graph break, and xtuner
    # compiles several MoE forward methods that way. Dynamo constant-folds this check, so the body
    # never enters the graph and instrumented models stay compilable.
    #
    # This branch must stay free of side effects for the same reason -- a global mutation or a log
    # call here would break `fullgraph=True`. Markers that turn out to be inert are reported from
    # `MarkerSession.report_unreached`, which runs in eager python.
    if torch.compiler.is_compiling():
        return

    session = _MARKER_SESSION.get()
    if session is not None:
        session.record(name)


def active_marker_session() -> "MarkerSession | None":
    """Return the marker session of the region currently executing, if any.

    Returns:
        MarkerSession | None: The active session, or None outside a selectively checkpointed region.
    """
    return _MARKER_SESSION.get()


@contextmanager
def marker_session(intervals: Sequence[MarkerInterval], owner: Any = None) -> Iterator["MarkerSession"]:
    """Open a marker session for one pass through a checkpointed region.

    The session must be opened *inside* the checkpointed callable: ``use_reentrant=False`` re-runs
    that callable to recompute, and both passes have to build their marker state from scratch or the
    two sets of kept ops drift apart.

    Args:
        intervals (Sequence[MarkerInterval]): Intervals this region keeps resident.
        owner (Any): The model these layers belong to, used only to aggregate diagnostics across
            its layers. Pass the same object that was given to :func:`declare_selective_regions`.
            Defaults to None, which diagnoses this region on its own.

    Returns:
        Iterator[MarkerSession]: The session that :func:`checkpoint_record` will drive.
    """
    session = MarkerSession(tuple(intervals), owner)
    token = _MARKER_SESSION.set(session)
    try:
        yield session
    finally:
        _MARKER_SESSION.reset(token)


def declare_selective_regions(owner: Any, intervals: Sequence[MarkerInterval]) -> None:
    """Declare that one more layer of ``owner`` will run with ``intervals``.

    Diagnostics are only trustworthy once every such layer has run: a marker missing from one layer
    type is normal when the intervals span several -- ``mlp.begin`` never runs in a MoE layer and
    ``moe.gate.begin`` never runs in a dense one -- and warning per layer would fire on models that
    are configured correctly. This tells the diagnostics how many layers to wait for.

    Args:
        owner (Any): The model that owns the layer, held weakly.
        intervals (Sequence[MarkerInterval]): The intervals that layer will keep resident.
    """
    if owner is None or not intervals:
        return
    state = _DIAGNOSTICS.get(owner)
    if state is None:
        state = _OwnerDiagnostics()
        _DIAGNOSTICS[owner] = state
    state.expected_layers += 1


class MarkerSession:
    """Tracks which marker intervals are open at the current point of one pass
    through a region.

    Driven by :func:`checkpoint_record` and read by the checkpoint policy; the SAC engine owns its
    lifetime through :func:`marker_session`.
    """

    def __init__(self, intervals: tuple[MarkerInterval, ...], owner: Any = None) -> None:
        self._intervals = intervals
        self._owner = owner
        self._open: set[MarkerInterval] = set()
        self._recorded: set[str] = set()
        self._kept: set[MarkerInterval] = set()

    @property
    def keeping(self) -> bool:
        """Whether execution is currently inside at least one interval.

        Returns:
            bool: True while any interval is open.
        """
        # Overlapping and nested intervals are defined as "any interval open => keep", which is why
        # this is set emptiness rather than a paired counter: no pairing is asserted anywhere.
        return bool(self._open)

    def record(self, name: str) -> None:
        """Register that execution reached the marker ``name``.

        Args:
            name (str): The marker name passed to :func:`checkpoint_record`.
        """
        self._recorded.add(name)
        for interval in self._intervals:
            start, end = interval
            if name == start:
                self._open.add(interval)
            elif name == end:
                self._open.discard(interval)

    def note_kept(self) -> None:
        """Register that an op was kept resident while the open intervals were
        open."""
        # This is the only evidence that a region is addressable at all. Markers merely delimit; an
        # interval whose contents run inside a compiled region has both endpoints fire and still
        # keeps nothing, because those ops execute as fused kernels and never reach the policy.
        self._kept |= self._open

    def finish(self) -> None:
        """Fold this pass into the owner's diagnostics, warning once the model
        has run in full."""
        _report_pass(self._owner, self._intervals, self._recorded, self._kept)


class _OwnerDiagnostics:
    def __init__(self) -> None:
        self.expected_layers = 0
        self.completed_layers = 0
        self.recorded_markers: set[str] = set()
        self.kept_intervals: set[MarkerInterval] = set()
        self.declared_intervals: set[MarkerInterval] = set()
        self.reported: set[MarkerInterval] = set()


def _report_pass(
    owner: Any,
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


_MARKER_SESSION: contextvars.ContextVar["MarkerSession | None"] = contextvars.ContextVar(
    "xtuner_sac_marker_session", default=None
)

# Diagnostics accumulate per model and are held weakly, so they neither keep models alive nor let
# one model's conclusions silence another's.
_DIAGNOSTICS: "weakref.WeakKeyDictionary[Any, _OwnerDiagnostics]" = weakref.WeakKeyDictionary()

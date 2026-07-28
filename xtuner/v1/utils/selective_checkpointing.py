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

Only the vocabulary lives here. The contextvars session behind :func:`checkpoint_record`, the
policy function, and the config resolution that turns user selections into intervals are owned by
the SAC engine and the config layer respectively.
"""

from typing import TypeAlias

import torch

from .enum_helper import StrEnum


__all__ = [
    "RecomputeUnit",
    "MarkerInterval",
    "RecomputeIntervalMap",
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
    if torch.compiler.is_compiling():
        return

    # The rest is intentionally empty: the SAC engine implements the session behind this marker.
    # Remove this stub only together with that implementation.

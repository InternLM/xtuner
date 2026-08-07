"""Region-level selective activation checkpointing (SAC): contract and engine.

This module holds the vocabulary that the SAC layers agree on and nothing else:

- Users select :class:`RecomputeUnit` members in the model config.
- Model authors declare a :data:`RecomputeTargetMap` saying what each unit they support resolves to
  for their architecture.
- The SAC engine turns the selection into a per-op checkpoint policy.

**A unit resolves to one of two things, and which one is not a style choice.** The checkpoint policy
is asked about every op that reaches the dispatcher, and it has to answer "is this op part of a kept
unit". There are exactly two ways to know:

- :class:`KeptOps` names the ops directly. This is the cheapest possible form -- it changes nothing
  about compilation -- but it only works when the op is *specific enough to identify the unit on its
  own*. Attention qualifies: `flash_attn::_flash_attn_varlen_forward` appears nowhere else. A gate's
  ``addmm`` does not.
- :class:`KeptCallables` names callables whose whole body belongs to the unit. Everything they
  dispatch is kept. This can express any region, at the price of taking those callables out of the
  compiled set -- a marker is ordinary python state and the ops it would cover run as fused kernels,
  so a region inside compiled code is invisible either way.

Prefer :class:`KeptOps`. Reach for :class:`KeptCallables` when no op identifies the unit, and then
name the *smallest* callable that covers it: the compilation given up is the whole callable's, not
the region's.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
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
    "KeptOps",
    "KeptCallables",
    "RecomputeTarget",
    "RecomputeTargetMap",
    "active_recompute_unit",
    "recompute_unit",
    "apply_selective_checkpointing",
    "in_recompute_unit",
    "resolve_kept_ops",
]


class RecomputeUnit(StrEnum):
    """Semantic units of activation that may be kept resident instead of
    recomputed.

    Each member names a *class of sub-structure* whose activations are worth keeping in memory because recomputing it
    is expensive relative to what it costs to store. A unit selected by the user means "do not recompute this part";
    everything not selected is recomputed. What a unit resolves to is architecture-specific and is declared per model,
    so the same unit can cover different code in different models.

    Members are strings so pydantic round-trips them to readable names in serialized configs.
    """

    SAVE_ATTN = "save_attn"
    """Keep the attention kernel's output -- the flash-attention call itself, not the projections
    around it.

    This is the narrowest unit and the only one that costs no compilation, because the attention
    kernel is a custom op: inductor cannot fuse it, so it is always called as a fallback kernel and
    is always visible to the checkpoint policy, compiled or not.
    """

    SAVE_MOE_GATE = "save_moe_gate"
    """Keep the MoE router: gating projection, top-k selection, and routing weights."""

    SAVE_MOE_DISPATCH = "save_moe_dispatch"
    """Keep the tensors produced around expert dispatch and combine: the permutation, padding and
    unpermutation buffers on either side of the all-to-all.

    The collective itself is always recomputed, never kept. Keeping a collective would elide it from
    the recompute pass, which is only sound if nothing it communicates with is replayed; the safe
    rule is to replay it. So this unit trades memory for the surrounding tensor work, not for the
    communication.
    """


@dataclass(frozen=True)
class KeptOps:
    """Resolve a unit by op identity: keep these ops wherever they run.

    Costs nothing -- compilation is untouched and no graph breaks are introduced -- because the ops
    worth naming this way are exactly the ones inductor cannot fuse, which are the ones that still
    reach the dispatcher from inside a compiled region.

    Args:
        names (tuple[str, ...]): Qualified op names, e.g. ``"flash_attn::_flash_attn_varlen_forward_v3"``.
            A name that is not registered in this build is skipped, so a model may list several
            backends' spellings of the same kernel.
    """

    names: tuple[str, ...]

    def __init__(self, *names: str) -> None:
        object.__setattr__(self, "names", tuple(names))


@dataclass(frozen=True)
class KeptCallables:
    """Resolve a unit by callable: keep everything these callables dispatch.

    The callables are taken out of the compiled set, because a region inside compiled code cannot be
    addressed at all -- so the compilation given up is the whole callable's. Name the smallest
    callable that covers the unit.

    Args:
        names (tuple[str, ...]): Qualified callable names, in the same form ``compile_cfg`` uses.
    """

    names: tuple[str, ...]

    def __init__(self, *names: str) -> None:
        object.__setattr__(self, "names", tuple(names))


RecomputeTarget: TypeAlias = KeptOps | KeptCallables
"""What a single :class:`RecomputeUnit` resolves to for one architecture."""

RecomputeTargetMap: TypeAlias = dict[RecomputeUnit, RecomputeTarget]
"""Per-model declaration of what each supported :class:`RecomputeUnit` resolves
to.

Models expose this as a ``default_recompute_cfg`` property, mirroring ``default_compile_cfg``. A unit absent from the
mapping is not supported by that architecture.
"""


def active_recompute_unit() -> RecomputeUnit | None:
    """Return the unit whose callable is currently executing, if any.

    Meaningful only while a checkpoint policy is running, which is the only caller. Units resolved by
    :class:`KeptOps` never set this -- they are recognised from the op itself.

    Returns:
        RecomputeUnit | None: The innermost open unit, or None outside one.
    """
    return _ACTIVE_UNIT.get()


@contextmanager
def recompute_unit(unit: RecomputeUnit) -> Iterator[None]:
    """Mark the enclosed call as belonging to ``unit``.

    Entered by the wrapper the engine installs on a :class:`KeptCallables` target, never by model
    code. It is a plain ``ContextVar``, which is only readable because the wrapped callable has been
    taken out of the compiled set; inside compiled code it would neither be set nor read.

    Args:
        unit (RecomputeUnit): The unit the enclosed call belongs to.
    """
    token = _ACTIVE_UNIT.set(unit)
    try:
        yield
    finally:
        _ACTIVE_UNIT.reset(token)


_ACTIVE_UNIT: ContextVar[RecomputeUnit | None] = ContextVar("xtuner_recompute_unit", default=None)


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
    :class:`KeptCallables` target names, together with the
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

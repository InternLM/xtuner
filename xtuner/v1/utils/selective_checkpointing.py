"""Shared contract for region-level selective activation checkpointing (SAC).

This module holds the vocabulary that the SAC layers agree on and nothing else:

- Users select :class:`SaveUnit` members in the model config.
- Model authors declare a :data:`RecomputeTargetMap` saying what each unit they support resolves to
  for their architecture.
- The SAC engine turns the selection into a per-op checkpoint policy.

It lives under ``xtuner.v1.utils`` rather than next to the models because the targets name callables
in ``xtuner.v1.module`` while the unit vocabulary is consumed by ``xtuner.v1.model`` configs; a home
inside either package would make the two import each other.

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
from typing import TypeAlias

from cyclopts import Parameter
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from typing_extensions import Annotated

from .enum_helper import StrEnum


__all__ = [
    "SaveUnit",
    "RecomputeConfig",
    "KeptOps",
    "KeptCallables",
    "RecomputeTarget",
    "RecomputeTargetMap",
    "active_recompute_unit",
    "recompute_unit",
]


class SaveUnit(StrEnum):
    """Semantic units of activation that are *saved* -- kept resident instead
    of recomputed.

    Naming is deliberate: selecting a member means "save this", not "recompute this". Everything not
    selected is recomputed, which is the default and costs no memory.

    Each member names a *class of sub-structure* whose activations are worth keeping because
    recomputing it is expensive relative to what it costs to store. What a unit resolves to is
    architecture-specific and is declared per model, so the same unit can cover different code in
    different models.

    Members are strings so pydantic round-trips them to readable names in serialized configs.
    """

    ATTN = "attn"
    """Keep the attention kernel's output -- the flash-attention call itself,
    not the projections around it.

    This is the narrowest unit and the only one that costs no compilation, because the attention
    kernel is a custom op: inductor cannot fuse it, so it is always called as a fallback kernel and
    is always visible to the checkpoint policy, compiled or not.
    """

    DSA_INDEXER = "dsa_indexer"
    """Keep the no-grad DSA top-k selection result, without keeping its projections."""

    MOE_GATE = "moe_gate"
    """Keep the MoE router: gating projection, top-k selection, and routing weights."""

    MOE_DISPATCH = "moe_dispatch"
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
"""What a single :class:`SaveUnit` resolves to for one architecture."""


class RecomputeConfig(PydanticBaseModel):
    """All recompute settings in one place: which layers, and what stays saved
    inside them.

    ``ratio`` and ``vision_ratio`` decide *which layers* are recomputed at all; ``save`` decides
    *what inside a recomputed layer* is kept resident anyway. They used to live in two different
    configs -- the ratios on ``FSDPConfig``, the unit selection on the model config -- which made
    the pair hard to reason about, since neither means much without the other.

    Args:
        ratio (float): Fraction of decoder layers to recompute, counted from the end. Defaults to
            1.0, i.e. recompute every layer.
        vision_ratio (float): The same, for a compose model's vision tower. Defaults to 1.0.
        save (list[SaveUnit] | bool | None): What stays resident inside the recomputed layers.
            ``None`` and ``False`` save nothing, ``True`` saves every unit the model declares, and a
            list saves exactly those. Defaults to None.

            ``None`` deliberately does not mean "model default", unlike ``compile_cfg``: saving a
            unit trades memory for speed, so an unset config must not silently change an existing
            run's peak memory.
    """

    model_config = ConfigDict(extra="forbid")

    ratio: Annotated[float, Parameter(help="Fraction of decoder layers to recompute, from the end")] = 1.0
    vision_ratio: Annotated[float, Parameter(help="The same, for a compose model's vision tower")] = 1.0
    save: Annotated[
        list[SaveUnit] | bool | None,
        Parameter(help="What stays resident inside recomputed layers: None/False none, True all declared, or a list"),
    ] = None


RecomputeTargetMap: TypeAlias = dict[SaveUnit, RecomputeTarget]
"""Per-model declaration of what each supported :class:`SaveUnit` resolves to.

Models expose this as a ``default_recompute_cfg`` property, mirroring ``default_compile_cfg``. A unit absent from the
mapping is not supported by that architecture.
"""


def active_recompute_unit() -> SaveUnit | None:
    """Return the unit whose callable is currently executing, if any.

    Meaningful only while a checkpoint policy is running, which is the only caller. Units resolved by
    :class:`KeptOps` never set this -- they are recognised from the op itself.

    Returns:
        SaveUnit | None: The innermost open unit, or None outside one.
    """
    return _ACTIVE_UNIT.get()


@contextmanager
def recompute_unit(unit: SaveUnit) -> Iterator[None]:
    """Mark the enclosed call as belonging to ``unit``.

    Entered by the wrapper the engine installs on a :class:`KeptCallables` target, never by model
    code. It is a plain ``ContextVar``, which is only readable because the wrapped callable has been
    taken out of the compiled set; inside compiled code it would neither be set nor read.

    Args:
        unit (SaveUnit): The unit the enclosed call belongs to.
    """
    token = _ACTIVE_UNIT.set(unit)
    try:
        yield
    finally:
        _ACTIVE_UNIT.reset(token)


_ACTIVE_UNIT: ContextVar[SaveUnit | None] = ContextVar("xtuner_recompute_unit", default=None)

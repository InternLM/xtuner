"""Judger dataclass — a named, weighted :class:`SandboxStage`.

Bench projects build :class:`Judger` instances inline with all their
fields (pre-hooks, env vars, entry, post-hooks) visible; the validator
just orchestrates them.  See ``claw_bench/pipeline.py`` for an example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sandbox import Hook, SandboxStage
from schemas import SandboxSpec


@dataclass
class Judger:
    """A named, weighted verifier stage.

    Attributes:
        name (str): Identifier used by aggregation + as ``JUDGER_NAME``.
        stage (SandboxStage): The actual work (upload + entry + parse).
        weight (float): Weight for weighted aggregation.
        sandbox (Literal["shared"] | SandboxSpec): ``"shared"`` reuses the
            infer sandbox; a :class:`SandboxSpec` spins up a fresh one.
        on_isolated_pre (list[Hook]): Extra hooks run once an isolated
            sandbox is acquired (e.g. seed workspace).
    """

    name: str
    stage: SandboxStage
    weight: float = 1.0
    sandbox: Literal["shared"] | SandboxSpec = "shared"
    on_isolated_pre: list[Hook] = field(default_factory=list)

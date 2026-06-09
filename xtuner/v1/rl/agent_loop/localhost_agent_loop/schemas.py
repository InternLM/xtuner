"""Schemas specific to the localhost_agent_loop runner.

Shared types (``AgentRolloutItem``, ``RolloutError``, ``StageStatus`` ...) are
re-exported from :mod:`sandbox_agent_loop.schemas` so localhost and sandbox
runners agree on item shape; the only local addition is
:class:`LocalhostAgentSpec`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class LocalhostAgentSpec(BaseModel):
    """One local agent variant.

    ``config`` holds the agent definition.  Two forms accepted:

      - **dotted-path string** (preferred, mirrors sandbox ``AgentSpec.config="config.py"``):
        ``"recipe.math_ci_eval.infer.agents.python.config"`` resolves at run
        time to ``<module>.agent_config`` via ``importlib``.  Use
        ``"<module>:<attr>"`` to point at a different attribute name.
      - **inline dict**: the lagent ``FunctionCallAgent`` config dict
        directly.  Convenient for tests; verbose for production pipelines.

    The runner resolves ``config`` to a dict and ``create_object``s it once
    per rollout (so module-level side effects in the agent module fire only
    at first import).  Nothing about the variant leaks outside the
    resolved dict.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    config: str | dict
    weight: float = 1.0


__all__ = ["LocalhostAgentSpec"]

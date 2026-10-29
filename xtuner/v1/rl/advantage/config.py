import math
from typing import Annotated, Literal

from cyclopts import Group, Parameter
from pydantic import BaseModel, ConfigDict, model_validator

from xtuner.v1.rl.advantage.base import AdvantageEstimator


advantage_group = Group("Advantage Estimation", sort_key=2, help="Advantage estimation configuration.")


class BaseAdvantageConfig(BaseModel):
    """Intermediate base for discriminated union."""

    model_config = ConfigDict(extra="forbid")

    def build(self) -> AdvantageEstimator:
        raise NotImplementedError("Subclasses must implement this method.")


class GAEAdvantageConfig(BaseAdvantageConfig):
    """Configuration for generalized-advantage estimation."""

    gae_gamma: Annotated[
        float,
        Parameter(group=advantage_group, help="Discount used by GAE."),
    ] = 1.0
    gae_lambda: Annotated[
        float,
        Parameter(group=advantage_group, help="GAE lambda used for actor advantage and critic return."),
    ] = 0.95
    reward_scope: Annotated[
        Literal["segment", "session"],
        Parameter(
            group=advantage_group,
            help="Place one reward per sample, or one reward on the last action of a session.",
        ),
    ] = "segment"
    normalize_actor_advantage: Annotated[
        bool,
        Parameter(
            group=advantage_group,
            help="Standardize actor advantages over kept tokens. Critic returns stay unnormalized.",
        ),
    ] = True

    @model_validator(mode="after")
    def _resolve_discounts(self) -> "GAEAdvantageConfig":
        for name in ("gae_gamma", "gae_lambda"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if self.reward_scope not in ("segment", "session"):
            raise ValueError(f"reward_scope must be 'segment' or 'session', got {self.reward_scope!r}.")
        return self

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.gae import GAEEstimator

        return GAEEstimator(
            gae_gamma=self.gae_gamma,
            gae_lambda=self.gae_lambda,
            reward_scope=self.reward_scope,
            normalize_actor_advantage=self.normalize_actor_advantage,
        )


class DecoupledGAEAdvantageConfig(BaseAdvantageConfig):
    """Configuration for decoupled GAE with independent actor/critic discounts.

    Decoupled GAE as proposed in `VAPO: Efficient and Reliable Reinforcement
    Learning for Advanced Reasoning Tasks <https://arxiv.org/abs/2504.05118>`_.

    ``actor_*`` discounts drive the actor advantage recursion and ``critic_*``
    discounts drive the critic TD(lambda) return; defaults follow the paper
    (actor lambda 0.95, critic lambda 1.0).
    """

    reward_scope: Annotated[
        Literal["segment"],
        Parameter(
            group=advantage_group,
            help="Place one reward per sample. Session scope is not supported for decoupled GAE currently.",
        ),
    ] = "segment"
    normalize_actor_advantage: Annotated[
        bool,
        Parameter(
            group=advantage_group,
            help="Standardize actor advantages over kept tokens. Critic returns stay unnormalized.",
        ),
    ] = True
    actor_gamma: Annotated[
        float,
        Parameter(group=advantage_group, help="Actor-advantage discount."),
    ] = 1.0
    actor_lambda: Annotated[
        float,
        Parameter(group=advantage_group, help="Actor-advantage GAE lambda."),
    ] = 0.95
    critic_gamma: Annotated[
        float,
        Parameter(group=advantage_group, help="Critic-return discount."),
    ] = 1.0
    critic_lambda: Annotated[
        float,
        Parameter(group=advantage_group, help="Critic-return GAE lambda."),
    ] = 1.0
    actor_length_adaptive_alpha: Annotated[
        float | None,
        Parameter(
            group=advantage_group,
            help="Replace the actor lambda with clamp(1 - 1 / (alpha * action_length), 0, 1) when set.",
        ),
    ] = None
    actor_positive_scale: Annotated[
        float,
        Parameter(group=advantage_group, help="Scale applied to positive actor advantages before normalization."),
    ] = 1.0
    actor_negative_scale: Annotated[
        float,
        Parameter(group=advantage_group, help="Scale applied to non-positive actor advantages before normalization."),
    ] = 1.0

    @model_validator(mode="after")
    def _resolve_discounts(self) -> "DecoupledGAEAdvantageConfig":
        for name in ("actor_gamma", "actor_lambda", "critic_gamma", "critic_lambda"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        alpha = self.actor_length_adaptive_alpha
        if alpha is not None and not (math.isfinite(alpha) and alpha > 0):
            raise ValueError(f"actor_length_adaptive_alpha must be finite and positive when set, got {alpha}.")
        return self

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.gae import DecoupledGAEEstimator

        return DecoupledGAEEstimator(
            normalize_actor_advantage=self.normalize_actor_advantage,
            actor_gamma=self.actor_gamma,
            actor_lambda=self.actor_lambda,
            critic_gamma=self.critic_gamma,
            critic_lambda=self.critic_lambda,
            actor_length_adaptive_alpha=self.actor_length_adaptive_alpha,
            actor_positive_scale=self.actor_positive_scale,
            actor_negative_scale=self.actor_negative_scale,
        )


class GRPOAdvantageConfig(BaseAdvantageConfig):
    """Configuration for :class:`~xtuner.v1.rl.advantage.grpo.GRPOEstimator`.

    Attributes:
        eps (float): Small constant for numerical stability. Default 1e-8.
    """

    eps: Annotated[
        float,
        Parameter(group=advantage_group, help="Small constant for numerical stability."),
    ] = 1e-8

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.grpo import GRPOEstimator

        return GRPOEstimator(eps=self.eps)


class DrGRPOAdvantageConfig(BaseAdvantageConfig):
    """Configuration for :class:`~xtuner.v1.rl.advantage.grpo.DrGRPOEstimator`.

    Attributes:
        max_length (float): Max response length for duration scaling.
            Default 32768.
        eps (float): Small constant for numerical stability. Default 1e-8.
    """

    max_length: Annotated[
        float,
        Parameter(group=advantage_group, help="Max response length for duration scaling."),
    ] = 32768
    eps: Annotated[
        float,
        Parameter(group=advantage_group, help="Small constant for numerical stability."),
    ] = 1e-8

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.grpo import DrGRPOEstimator

        return DrGRPOEstimator(max_length=self.max_length, eps=self.eps)


class RLOOAdvantageConfig(BaseAdvantageConfig):
    """Configuration for
    :class:`~xtuner.v1.rl.advantage.rloo.RLOOEstimator`."""

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.rloo import RLOOEstimator

        return RLOOEstimator()


class OPOAdvantageConfig(BaseAdvantageConfig):
    """Configuration for :class:`~xtuner.v1.rl.advantage.opo.OPOEstimator`.

    Attributes:
        eps (float): Small constant for numerical stability. Default 1e-8.
    """

    eps: Annotated[
        float,
        Parameter(group=advantage_group, help="Small constant for numerical stability."),
    ] = 1e-8

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.opo import OPOEstimator

        return OPOEstimator(eps=self.eps)


class PassKAdvantageConfig(BaseAdvantageConfig):
    """Configuration for :class:`~xtuner.v1.rl.advantage.passk.PassKEstimator`.

    Attributes:
        k (int): The K in pass@k. Default 4.
        eps (float): Small constant for numerical stability. Default 1e-6.
    """

    k: Annotated[
        int,
        Parameter(group=advantage_group, help="The K in pass@k."),
    ] = 4
    eps: Annotated[
        float,
        Parameter(group=advantage_group, help="Small constant for numerical stability."),
    ] = 1e-6

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.passk import PassKEstimator

        return PassKEstimator(k=self.k, eps=self.eps)

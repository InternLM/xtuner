"""Configuration for the PPO critic co-resident with the actor."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.config.optim import LRConfig, OptimConfig
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.model.value import wants_scalar_value_head
from xtuner.v1.rl.loss.value_loss import ValueLossConfig


class KLRewardConfig(BaseModel):
    """Per-token KL penalty folded into the reward.

    Classic PPO for RLHF subtracts ``beta * KL(policy || reference)`` from the
    token reward, so the penalty flows through the GAE recursion and shapes the
    value targets. That is not equivalent to the loss-side KL used by the
    group-baseline algorithms here, which leaves the critic unaware of it.

    Args:
        coef (float): The KL coefficient ``beta``.
        kl_type (str): KL estimator; see
            :func:`~xtuner.v1.rl.loss.kl_divergence_per_token`. ``low_var_kl``
            is clamped and non-negative, which keeps a single token from
            dominating the return.
        behavior_logprobs (Literal["old", "rollout"]): Which log probabilities
            to measure divergence from. ``old`` recomputes them with the current
            policy, matching the loss-side convention and costing one extra
            actor forward. ``rollout`` reuses the inference engine's values,
            which is free but reflects the sampling policy, including any
            numerical mismatch between the training and inference stacks.
    """

    model_config = ConfigDict(extra="forbid")

    coef: float = 0.001
    kl_type: Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] = "low_var_kl"
    behavior_logprobs: Literal["old", "rollout"] = "old"

    @model_validator(mode="after")
    def _validate(self) -> "KLRewardConfig":
        if self.coef < 0:
            raise ValueError(f"coef must be non-negative, got {self.coef}")
        return self


class CriticWorkerConfig(BaseModel):
    """Configuration for the PPO value model.

    The critic lives in the same worker process as the actor and shares its
    placement group, but owns a separate model, optimizer and learning-rate
    schedule. Only one of the two is resident on the accelerator at a time; see
    the phase machine in :mod:`xtuner.v1.rl.trainer.worker`.

    Args:
        model_cfg (TransformerConfig | BaseComposeConfig): Value-model
            configuration, normally produced by
            :func:`~xtuner.v1.model.value.as_value_config`.
        optim_cfg (OptimConfig): Critic optimizer. Typically a larger learning
            rate than the actor's, since the value head starts untrained.
        loss_cfg (ValueLossConfig): Value regression objective.
        lr_cfg (LRConfig): Critic learning-rate schedule.
        fsdp_cfg (FSDPConfig): Critic sharding configuration.
        load_from (str | Path | None): Checkpoint to initialize from. When
            ``None`` with ``load_mode="init_from_actor"`` the actor's
            ``load_from`` is reused.
        load_mode (Literal["init_from_actor", "load_weights"]): Whether to grow
            a fresh value head on an actor backbone, or load a trained critic.
        num_passes (int): Passes over each rollout batch. The critic can afford
            more reuse than the actor because value regression has no trust
            region to violate.
        optimizer_steps_per_pass (int): Optimizer updates per pass.
        scheduler_steps (int): Total learning-rate scheduler steps.
        warmup_steps (int): Training steps during which only the critic is
            updated. A freshly initialized value head predicts noise, so early
            advantages are noise too; letting the critic catch up first avoids
            driving the policy with them.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model_cfg: TransformerConfig | BaseComposeConfig
    optim_cfg: OptimConfig
    loss_cfg: ValueLossConfig
    lr_cfg: LRConfig
    fsdp_cfg: FSDPConfig
    load_from: str | Path | None = None
    load_mode: Literal["init_from_actor", "load_weights"] = "init_from_actor"
    num_passes: int = 1
    optimizer_steps_per_pass: int = 1
    scheduler_steps: int = 1_000_000
    warmup_steps: int = 0

    @model_validator(mode="after")
    def _validate(self) -> "CriticWorkerConfig":
        if self.num_passes <= 0:
            raise ValueError(f"num_passes must be positive, got {self.num_passes}")
        if self.optimizer_steps_per_pass <= 0:
            raise ValueError(f"optimizer_steps_per_pass must be positive, got {self.optimizer_steps_per_pass}")
        if self.scheduler_steps <= 0:
            raise ValueError(f"scheduler_steps must be positive, got {self.scheduler_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.load_mode == "load_weights" and self.load_from is None:
            raise ValueError("critic load_from is required when load_mode='load_weights'.")

        # Catch an actor config passed here by mistake: it would build a
        # vocabulary head and fail much later, inside the value loss.
        text_cfg = self.model_cfg.text_config if isinstance(self.model_cfg, BaseComposeConfig) else self.model_cfg
        if not wants_scalar_value_head(text_cfg):
            raise ValueError(
                "critic model_cfg must be a value-model configuration. "
                "Derive it with `xtuner.v1.model.value.as_value_config(actor_model_cfg)`."
            )
        return self

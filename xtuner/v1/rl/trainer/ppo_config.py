import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.config.optim import LRConfig, OptimConfig
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.loss.value_loss import ValueLossConfig


class CriticWorkerConfig(BaseModel):
    """Configuration for the independent PPO value model."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model_cfg: TransformerConfig | BaseComposeConfig
    optim_cfg: OptimConfig
    loss_cfg: ValueLossConfig
    lr_cfg: LRConfig
    fsdp_cfg: FSDPConfig
    load_from: str | Path | None = None
    # Full resume is intentionally handled only by trainer LoadCheckpointConfig,
    # which also restores both schedulers, counters, and RNG state.
    load_mode: Literal["init_from_actor", "load_weights"] = "init_from_actor"
    num_passes: int = 2
    optimizer_steps_per_pass: int = 8
    scheduler_steps: int = 1_000_000
    sp_size: int = 1

    @model_validator(mode="after")
    def _validate_first_version(self):
        if self.sp_size != 1:
            raise ValueError("The first PPO/Critic implementation supports critic sp_size=1 only.")
        if self.num_passes <= 0:
            raise ValueError("num_passes must be positive.")
        if self.optimizer_steps_per_pass <= 0:
            raise ValueError("optimizer_steps_per_pass must be positive.")
        if self.scheduler_steps <= 0:
            raise ValueError("scheduler_steps must be positive.")
        if self.load_mode != "init_from_actor" and self.load_from is None:
            raise ValueError(f"critic load_from is required for load_mode={self.load_mode!r}.")
        return self


class PPOConfig(BaseModel):
    """Token-level PPO target, mask, and pass configuration."""

    model_config = ConfigDict(extra="forbid")

    actor_gamma: float = 1.0
    actor_lambda: float = 0.95
    actor_length_adaptive_alpha: float | None = None
    critic_gamma: float = 1.0
    critic_lambda: float = 1.0
    normalize_actor_advantage: bool = True
    normalize_critic_return: bool = False
    normalize_reward: bool = False
    actor_num_passes: int = 1
    selection_seed: int = 0
    keep_uniform_groups: bool = True
    train_actor_on_uniform_groups: bool = False
    max_truncated_per_group: int | None = None

    # Keep the current sampled-token surprisal heuristic for the Actor only.
    enable_actor_surprisal_scaling: bool = False
    surprisal_upper_bound: float = 0.65
    surprisal_lower_bound: float = 0.4
    tau_upper: float = 0.0
    tau_lower: float = 0.0
    coeff_min_upper: float = 0.2
    coeff_min_lower: float = 0.5

    @model_validator(mode="after")
    def _validate_policy(self):
        if self.actor_num_passes != 1:
            raise ValueError("The first PPO implementation requires actor_num_passes=1.")
        if self.normalize_critic_return:
            raise ValueError("Critic return normalization is intentionally disabled.")
        if self.normalize_reward:
            raise ValueError("Reward normalization is intentionally disabled.")
        if self.max_truncated_per_group is not None and self.max_truncated_per_group < 0:
            raise ValueError("max_truncated_per_group must be non-negative or None.")
        if self.actor_length_adaptive_alpha is not None and (
            not math.isfinite(self.actor_length_adaptive_alpha) or self.actor_length_adaptive_alpha <= 0
        ):
            raise ValueError("actor_length_adaptive_alpha must be finite and positive when enabled.")
        for name in ("actor_gamma", "actor_lambda", "critic_gamma", "critic_lambda"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        return self

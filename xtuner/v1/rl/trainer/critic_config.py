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

    @model_validator(mode="after")
    def _validate(self) -> "CriticWorkerConfig":
        if self.num_passes <= 0:
            raise ValueError(f"num_passes must be positive, got {self.num_passes}")
        if self.optimizer_steps_per_pass <= 0:
            raise ValueError(f"optimizer_steps_per_pass must be positive, got {self.optimizer_steps_per_pass}")
        if self.scheduler_steps <= 0:
            raise ValueError(f"scheduler_steps must be positive, got {self.scheduler_steps}")
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

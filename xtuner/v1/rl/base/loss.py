from typing import Any, Literal, TypeVar

import torch
from pydantic import BaseModel, ConfigDict
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import Self

from xtuner.v1.loss import BaseLossConfig
from xtuner.v1.loss.base_loss_ctx import BaseLossContext
from .rollout_is import RolloutImportanceSampling
from ..utils import sp_split



T = TypeVar("T")


class BaseRLLossConfig(BaseLossConfig):
    """Base configuration for reinforcement learning loss functions in XTuner
    RL.

    Configuration base class for RL loss computations, providing a framework for
    policy optimization objectives with optional KL divergence regularization.
    Serves as the foundation for various RL algorithms including PPO, GRPO, and custom implementations.

    Args:
        policy_loss_cfg (dict[str, Any]): Configuration parameters for the main policy loss.
            Contains algorithm-specific parameters for policy optimization.
        use_kl_loss (bool): Whether to include KL divergence penalty in the loss.
            When True, requires a reference model for KL computation. Defaults to False.
        kl_loss_coef (float): Coefficient for weighting the KL divergence penalty.
            Controls the strength of regularization against the reference policy. Defaults to 0.001.
        kl_loss_type (Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None):
            Type of KL penalty computation method. Different types provide various
            regularization behaviors and numerical stability properties. Defaults to None.
        rollout_is (RolloutImportanceSampling): Configuration parameters for the rollout importance sampling.
            Contains algorithm-specific parameters for rollout importance sampling.
            Defaults to RolloutImportanceSampling().
    **Abstract Method:**
        loss_ctx_cls: Must be implemented by subclasses to return the appropriate
        loss context class for the specific RL algorithm.

    **Examples:**

    Example configuration for basic RL loss ::

        config = GRPOLossConfig(
            policy_loss_cfg=dict(
                cliprange_high=0.2,
                cliprange_low=0.2,
                loss_type='vanilla',
            ),
            use_kl_loss=False
        )

    Example configuration RL loss with KL regularization::

        config = GRPOLossConfig(
            policy_loss_cfg=dict(
                cliprange_high=0.2,
                cliprange_low=0.2,
                loss_type='vanilla',
            ),
            use_kl_loss=True,
            kl_loss_coef=0.001,
            kl_loss_type="low_var_kl"
        )

    .. note::
       When ``use_kl_loss=True``, ensure that the training worker is configured
       with a reference model for KL divergence computation.
    """

    policy_loss_cfg: dict[str, Any]
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None = None
    rollout_is: RolloutImportanceSampling = RolloutImportanceSampling()

    @property
    def loss_ctx_cls(self) -> type[BaseLossContext]:
        raise NotImplementedError


class RLLossContextInputItem(BaseModel):
    """Input item for reinforcement learning loss context.

    Args:
        shifted_labels (torch.Tensor): The shifted labels for the input sequences.
        advantages (torch.Tensor): Advantage estimates for the actions taken.
        old_logprobs (torch.Tensor | None): Log probabilities from the old policy.
        ref_logprobs (torch.Tensor | None): Reference log probabilities for KL penalty, if used.
        rollout_logprobs (torch.Tensor | None): Rollout log probabilities from inference engine, used for importance sampling.
        is_weights (torch.Tensor | None): Importance sampling weights. If None, importance sampling is not used.
    """

    model_config = ConfigDict(title="RLLossContextInputItem", extra="forbid", arbitrary_types_allowed=True)
    shifted_labels: torch.Tensor
    advantages: torch.Tensor
    old_logprobs: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    rollout_logprobs: torch.Tensor | None = None
    is_weights: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> Self:
        shifted_labels = sp_split(self.shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100)
        advantages = sp_split(self.advantages, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        if self.rollout_logprobs is not None:
            rollout_logprobs = sp_split(self.rollout_logprobs, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        else:
            rollout_logprobs = None
        if self.is_weights is not None:
            is_weights = sp_split(self.is_weights, sp_mesh=sp_mesh, split_dim=1, padding_value=1.0)
        else:
            is_weights = None
        # 这里不用对old_logprobs和ref_logprobs进行sp_split，因为他是模型 fwd 生成的
        # 模型 fwd 前一定会对 seq_ctx 进行 sp_split
        return type(self)(
            shifted_labels=shifted_labels,
            advantages=advantages,
            old_logprobs=self.old_logprobs,
            ref_logprobs=self.ref_logprobs,
            rollout_logprobs=rollout_logprobs,
            is_weights=is_weights,
        )

    def to(self, device: torch.device | str) -> Self:
        self.shifted_labels = self.shifted_labels.to(device)
        self.advantages = self.advantages.to(device)
        if self.old_logprobs is not None:
            self.old_logprobs = self.old_logprobs.to(device)
        if self.ref_logprobs is not None:
            self.ref_logprobs = self.ref_logprobs.to(device)
        if self.rollout_logprobs is not None:
            self.rollout_logprobs = self.rollout_logprobs.to(device)
        if self.is_weights is not None:
            self.is_weights = self.is_weights.to(device)
        return self

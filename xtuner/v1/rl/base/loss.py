from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import Self

from xtuner.v1.loss import BaseLossConfig, BaseLossKwargs
from xtuner.v1.loss.base_loss_ctx import BaseLossContext
from xtuner.v1.loss.utils import sp_gather, sp_split
from xtuner.v1.utils.device import get_device

# from ..utils import sp_split
from .rollout_is import RolloutImportanceSampling


DEVICE = get_device()


def compute_kl_loss_weight(
    shifted_labels: torch.Tensor, global_grad_tokens: torch.Tensor, kl_loss_coef: float, ignore_idx: int = -100
) -> torch.Tensor:
    kl_loss_weight = torch.ones_like(shifted_labels, dtype=torch.float32) / global_grad_tokens * kl_loss_coef
    kl_loss_weight[shifted_labels == ignore_idx] = 0.0
    return kl_loss_weight


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
    def loss_ctx_cls(self) -> type["BaseRLLossContext"]:
        raise NotImplementedError

    @property
    def _loss_kwargs_cls(self) -> type["BaseRLLossKwargs"]:
        raise NotImplementedError

    def build(
        self,
        sp_mesh: DeviceMesh | None,
        shifted_labels: torch.Tensor,
        advantages: torch.Tensor,
        rollout_logprobs: torch.Tensor | None = None,
        old_logprobs: torch.Tensor | None = None,
        rollout_is_weights: torch.Tensor | None = None,
        ref_logprobs: torch.Tensor | None = None,
    ) -> "BaseRLLossContext":
        # loss_ctx_input = RLLossContextInputItem(
        #     shifted_labels=shifted_labels,
        #     advantages=advantages,
        #     rollout_logprobs=rollout_logprobs,
        #     old_logprobs=old_logprobs,
        #     is_weights=rollout_is_weights,
        #     ref_logprobs=ref_logprobs,
        # ).to(DEVICE)
        # if sp_mesh.size() > 1:
        #     loss_ctx_input = loss_ctx_input.sp_split(sp_mesh)

        LossKwargs = self._loss_kwargs_cls
        loss_kwargs = LossKwargs(
            shifted_labels=shifted_labels,
            old_logprobs=old_logprobs,
            advantages=advantages,
            rollout_logprobs=rollout_logprobs,
            is_weights=rollout_is_weights,
            ref_logprobs=ref_logprobs,
        ).to(DEVICE)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)

        LossContext = self.loss_ctx_cls
        return LossContext(self, loss_kwargs)


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


class BaseRLLossKwargs(BaseLossKwargs):
    """Keyword arguments for reinforcement learning loss computation.

    Args:
        shifted_labels (torch.Tensor): The shifted labels for the input sequences.
        old_logprobs (torch.Tensor): Log probabilities from the old policy.
        advantages (torch.Tensor): Advantage estimates for the actions taken.
        policy_loss_weight (torch.Tensor): Weights for each token in the policy loss computation.
        ref_logprobs (torch.Tensor | None): Reference log probabilities for KL penalty, if used.
        kl_loss_weight (torch.Tensor | None): Weights for each token in the KL loss computation, if used.
        rollout_logprobs (torch.Tensor | None): Rollout log probabilities from inference engine, used for importance sampling.
        is_weights (torch.Tensor | None): Importance sampling weights. If None, importance sampling is not used.
    """

    rollout_logprobs: torch.Tensor | None = None
    advantages: torch.Tensor
    old_logprobs: torch.Tensor | None = None
    policy_loss_weight: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    kl_loss_weight: torch.Tensor | None = None
    global_grad_tokens: torch.Tensor | None = None
    is_weights: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> Self:
        self.shifted_labels = sp_split(self.shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100)
        self.advantages = sp_split(self.advantages, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        if self.rollout_logprobs is not None:
            self.rollout_logprobs = sp_split(self.rollout_logprobs, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        if self.is_weights is not None:
            self.is_weights = sp_split(self.is_weights, sp_mesh=sp_mesh, split_dim=1, padding_value=1.0)
        # 1. 这里不用对old_logprobs和ref_logprobs进行sp_split，因为他是模型 fwd 生成的，
        # 因为模型 fwd 前一定会对 seq_ctx 进行 sp_split。
        # 2. global_grad_tokens 是scalar Tensor, 不用进行sp_split。
        # 3. 这里也不用对各种weight(policy_loss_weight, kl_loss_weight, is_weights)进行sp_split，
        # 因为他们在LossContext.build_batches()中生成时也保证是sp_split的。
        return self

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
        if self.global_grad_tokens is not None:
            self.global_grad_tokens = self.global_grad_tokens.to(device)
        if self.policy_loss_weight is not None:
            self.policy_loss_weight = self.policy_loss_weight.to(device)
        if self.kl_loss_weight is not None:
            self.kl_loss_weight = self.kl_loss_weight.to(device)
        return self


class BaseRLLossContext(BaseLossContext[RLLossContextInputItem]):
    loss_cfg: BaseRLLossConfig
    loss_kwargs: BaseRLLossKwargs

    def compute_rollout_is(
        self, sp_mesh: DeviceMesh, num_tokens: torch.Tensor
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        shifted_labels = self.loss_kwargs.shifted_labels
        rollout_logprobs = self.loss_kwargs.rollout_logprobs
        mask = shifted_labels != self.loss_cfg.ignore_idx
        old_logprobs = self.loss_kwargs.old_logprobs

        assert rollout_logprobs is not None
        assert old_logprobs is not None

        if sp_mesh and sp_mesh.size() > 1:
            # Temporarily sp_gather old_logprobs here, but not modify loss_kwargs.old_logprobs(still in sp_split state)
            rollout_logprobs = sp_gather(rollout_logprobs, sp_mesh, dim=1)
            old_logprobs = sp_gather(old_logprobs, sp_mesh, dim=1)
            old_logprobs = old_logprobs[:, : rollout_logprobs.size(1)]  # type: ignore
            mask = sp_gather(mask, sp_mesh, dim=1)
            mask = mask[:, : rollout_logprobs.size(1)]  # type: ignore

        rollout_is_weights, rollout_is_mask, mismatch_metrics, rollout_is_metrics = (
            self.loss_cfg.rollout_is.compute_rollout_importance_weights_and_metrics(
                old_log_prob=old_logprobs,
                rollout_log_prob=rollout_logprobs,
                num_tokens=num_tokens,
                response_mask=mask,
            )
        )
        if sp_mesh and sp_mesh.size() > 1:
            rollout_is_mask = sp_split(rollout_is_mask, sp_mesh, 1, 0)
            assert rollout_is_mask.size(1) == shifted_labels.size(1), (
                f"rollout_is_mask {rollout_is_mask.size(1)} vs shifted_labels {shifted_labels.size(1)}"
            )

            if rollout_is_weights is not None:
                rollout_is_weights = sp_split(rollout_is_weights, sp_mesh, 1, 0)
                assert rollout_is_weights.size(1) == shifted_labels.size(1), (
                    f"rollout_is_weights {rollout_is_weights.size(1)} vs shifted_labels {shifted_labels.size(1)}"
                )

        shifted_labels[~rollout_is_mask.bool()] = -100  # update loss mask

        self.loss_kwargs.is_weights = rollout_is_weights
        return mismatch_metrics, rollout_is_metrics

# Copyright (c) OpenMMLab. All rights reserved.
"""
DPO (Direct Preference Optimization) Loss Implementation for XTuner v1.

This module supports multiple DPO variants and loss combinations:
- sigmoid: Standard DPO loss for preference learning
- bco_pair: Binary Classifier Optimization for absolute quality
- sft: Supervised Fine-Tuning loss to maintain generation quality
- hinge, ipo, robust, nca_pair, sppo_hard: Other DPO variants

Reference:
- DPO: https://arxiv.org/abs/2305.18290
- BCO: https://arxiv.org/abs/2404.04656
- IPO: https://arxiv.org/abs/2310.12036
"""
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import Self

from xtuner.v1.loss import BaseLossContext, BaseLossKwargs
from xtuner.v1.loss.base_loss_ctx import BaseLossConfig
from xtuner.v1.utils import get_logger

from ..utils import gather_logprobs, sp_split


logger = get_logger()


class RunningMoments:
    """Running mean tracker for BCO loss."""

    def __init__(self):
        self.mean = 0.0
        self.count = 0

    def update(self, value: torch.Tensor):
        """Update running mean with new value."""
        value = value.detach().float().mean().item()
        self.count += 1
        self.mean = self.mean + (value - self.mean) / self.count


class DPOLossContextInputItem(BaseModel):
    """Input item for DPO loss context.

    This class handles preference pair data (chosen/rejected responses).

    Args:
        chosen_shifted_labels (torch.Tensor): Shifted labels for chosen responses.
        rejected_shifted_labels (torch.Tensor): Shifted labels for rejected responses.
        ref_chosen_logprobs (torch.Tensor | None): Reference model log probs for chosen.
        ref_rejected_logprobs (torch.Tensor | None): Reference model log probs for rejected.
    """

    model_config = ConfigDict(
        title="DPOLossContextInputItem", extra="forbid", arbitrary_types_allowed=True
    )
    chosen_shifted_labels: torch.Tensor
    rejected_shifted_labels: torch.Tensor
    ref_chosen_logprobs: torch.Tensor | None = None
    ref_rejected_logprobs: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> Self:
        chosen_shifted_labels = sp_split(
            self.chosen_shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100
        )
        rejected_shifted_labels = sp_split(
            self.rejected_shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100
        )
        return type(self)(
            chosen_shifted_labels=chosen_shifted_labels,
            rejected_shifted_labels=rejected_shifted_labels,
            ref_chosen_logprobs=self.ref_chosen_logprobs,
            ref_rejected_logprobs=self.ref_rejected_logprobs,
        )

    def to(self, device: torch.device | str) -> Self:
        self.chosen_shifted_labels = self.chosen_shifted_labels.to(device)
        self.rejected_shifted_labels = self.rejected_shifted_labels.to(device)
        if self.ref_chosen_logprobs is not None:
            self.ref_chosen_logprobs = self.ref_chosen_logprobs.to(device)
        if self.ref_rejected_logprobs is not None:
            self.ref_rejected_logprobs = self.ref_rejected_logprobs.to(device)
        return self


class DPOLossKwargs(BaseLossKwargs):
    """Keyword arguments for DPO loss computation.

    Args:
        shifted_labels (torch.Tensor): Concatenated shifted labels [chosen; rejected].
        chosen_mask (torch.Tensor): Mask for chosen tokens.
        rejected_mask (torch.Tensor): Mask for rejected tokens.
        ref_chosen_logprobs (torch.Tensor | None): Reference log probs for chosen.
        ref_rejected_logprobs (torch.Tensor | None): Reference log probs for rejected.
        loss_weight (torch.Tensor): Weight for each token in loss computation.
        num_chosen_tokens (int): Number of valid tokens in chosen responses.
        num_rejected_tokens (int): Number of valid tokens in rejected responses.
    """

    chosen_mask: torch.Tensor
    rejected_mask: torch.Tensor
    ref_chosen_logprobs: torch.Tensor | None = None
    ref_rejected_logprobs: torch.Tensor | None = None
    loss_weight: torch.Tensor
    num_chosen_tokens: int = 0
    num_rejected_tokens: int = 0


class DPOLossConfig(BaseLossConfig):
    """Configuration for DPO (Direct Preference Optimization) loss.

    DPO can combine multiple loss types with configurable weights:
    - sigmoid: Standard DPO loss
    - bco_pair: Binary Classifier Optimization loss
    - sft: Supervised Fine-Tuning loss on chosen responses

    Args:
        loss_types (list[str]): List of loss types to combine.
            Supported: ["sigmoid", "bco_pair", "sft", "hinge", "ipo", "robust"]
        loss_weights (list[float] | None): Weights for each loss type.
            If None, all weights are set to 1.0.
        beta (float): Temperature parameter for DPO/BCO losses. Defaults to 0.1.
        label_smoothing (float): Label smoothing for DPO loss. Defaults to 0.0.
        reference_free (bool): Whether to use reference-free mode. Defaults to False.
        use_average_log_prob (bool): Whether to normalize log probs by sequence length.
            Defaults to False.

    Example:
        >>> # Standard DPO configuration
        >>> config = DPOLossConfig(
        ...     loss_types=["sigmoid"],
        ...     loss_weights=[1.0],
        ...     beta=0.1,
        ... )
        >>> # MPO-style configuration (multiple loss types)
        >>> config = DPOLossConfig(
        ...     loss_types=["sigmoid", "bco_pair", "sft"],
        ...     loss_weights=[0.8, 0.2, 1.0],
        ...     beta=0.1,
        ... )
    """

    loss_types: list[Literal["sigmoid", "bco_pair", "sft", "hinge", "ipo", "robust", "nca_pair", "sppo_hard"]] = Field(
        default=["sigmoid"],
        description="List of loss types to combine",
    )
    loss_weights: list[float] | None = Field(
        default=None,
        description="Weights for each loss type. If None, all weights are 1.0",
    )
    beta: float = Field(
        default=0.1,
        description="Temperature parameter for DPO loss",
    )
    label_smoothing: float = Field(
        default=0.0,
        description="Label smoothing for robust DPO variants",
    )
    reference_free: bool = Field(
        default=False,
        description="Whether to use reference-free mode",
    )
    use_average_log_prob: bool = Field(
        default=False,
        description="Whether to normalize log probs by sequence length (used in IPO)",
    )

    @property
    def loss_ctx_cls(self) -> type["DPOLossContext"]:
        return DPOLossContext

    def model_post_init(self, __context: Any) -> None:
        """Validate and set default loss weights."""
        if self.loss_weights is None:
            self.loss_weights = [1.0] * len(self.loss_types)
        elif len(self.loss_weights) != len(self.loss_types):
            raise ValueError(
                f"Length of loss_weights ({len(self.loss_weights)}) must match "
                f"length of loss_types ({len(self.loss_types)})"
            )


class DPOLossContext(BaseLossContext[DPOLossContextInputItem]):
    """DPO loss context for preference alignment training.

    This class implements the loss computation for Direct Preference Optimization
    and its variants (including MPO-style multi-loss combinations).
    """

    loss_cfg: DPOLossConfig
    loss_kwargs: DPOLossKwargs

    def __init__(self, loss_cfg: DPOLossConfig, loss_kwargs: DPOLossKwargs):
        super().__init__(loss_cfg, loss_kwargs)
        # Running moments for BCO loss
        self.running = RunningMoments()

    @staticmethod
    def build_batches(  # type: ignore[override]
        loss_ctx_list: list["DPOLossContext"], *args: Any, **kwargs: Any
    ) -> list["DPOLossContext"]:
        del args, kwargs
        batch_size = len(loss_ctx_list)
        for loss_ctx in loss_ctx_list:
            loss_ctx._batch_size = batch_size
        return loss_ctx_list

    @classmethod
    def build_batches_loss_kwargs(
        cls,
        data_batches: list[DPOLossContextInputItem],
        loss_cfg: DPOLossConfig,
        cu_seq_lens_list: list[torch.Tensor] | None = None,
        sp_mesh: DeviceMesh | None = None,
    ) -> list[DPOLossKwargs]:
        """Build loss kwargs for each batch.

        This method processes the input data batches and prepares the loss kwargs
        for the DPO loss computation.
        """
        batches_loss_kwargs = []

        # Compute global token count for loss normalization
        total_chosen_tokens = 0
        total_rejected_tokens = 0
        for item in data_batches:
            total_chosen_tokens += (item.chosen_shifted_labels != loss_cfg.ignore_idx).sum()
            total_rejected_tokens += (item.rejected_shifted_labels != loss_cfg.ignore_idx).sum()

        total_chosen_tokens = cast(torch.Tensor, total_chosen_tokens)
        total_rejected_tokens = cast(torch.Tensor, total_rejected_tokens)

        if dist.is_initialized():
            dist.all_reduce(total_chosen_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_rejected_tokens, op=dist.ReduceOp.SUM)

        global_tokens = total_chosen_tokens + total_rejected_tokens
        if global_tokens == 0:
            logger.warning(
                "Global tokens is 0, which may lead to division by zero in loss weight calculation."
            )
            global_tokens = global_tokens + 1

        for item in data_batches:
            # Concatenate chosen and rejected labels
            # Shape: [1, chosen_len + rejected_len]
            shifted_labels = torch.cat(
                [item.chosen_shifted_labels, item.rejected_shifted_labels], dim=1
            )

            # Create masks
            chosen_len = item.chosen_shifted_labels.shape[1]
            rejected_len = item.rejected_shifted_labels.shape[1]
            total_len = chosen_len + rejected_len

            chosen_mask = torch.zeros(1, total_len, device=shifted_labels.device, dtype=torch.bool)
            rejected_mask = torch.zeros(1, total_len, device=shifted_labels.device, dtype=torch.bool)
            chosen_mask[:, :chosen_len] = item.chosen_shifted_labels != loss_cfg.ignore_idx
            rejected_mask[:, chosen_len:] = item.rejected_shifted_labels != loss_cfg.ignore_idx

            # Compute loss weight
            loss_weight = torch.zeros_like(shifted_labels, dtype=torch.float32)
            loss_weight[chosen_mask] = 1.0 / global_tokens.float()
            loss_weight[rejected_mask] = 1.0 / global_tokens.float()

            num_chosen_tokens = chosen_mask.sum().item()
            num_rejected_tokens = rejected_mask.sum().item()

            loss_kwargs = DPOLossKwargs(
                shifted_labels=shifted_labels,
                chosen_mask=chosen_mask,
                rejected_mask=rejected_mask,
                ref_chosen_logprobs=item.ref_chosen_logprobs,
                ref_rejected_logprobs=item.ref_rejected_logprobs,
                loss_weight=loss_weight,
                num_chosen_tokens=int(num_chosen_tokens),
                num_rejected_tokens=int(num_rejected_tokens),
            )
            batches_loss_kwargs.append(loss_kwargs)

        return batches_loss_kwargs

    def _compute_logprobs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for the given labels.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            mask: Valid token mask [batch, seq_len]

        Returns:
            Sum of log probabilities for valid tokens [batch]
        """
        # Gather log probs for target tokens
        logprobs = gather_logprobs(logits, labels)  # [batch, seq_len]
        logprobs = logprobs * mask.float()

        if self.loss_cfg.use_average_log_prob:
            # Normalize by sequence length (used in IPO)
            return logprobs.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        else:
            return logprobs.sum(dim=-1)

    def _dpo_loss_sigmoid(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute standard DPO sigmoid loss."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.loss_cfg.reference_free:
            ref_logratios = torch.zeros_like(pi_logratios)
        else:
            ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios

        loss = (
            -F.logsigmoid(self.loss_cfg.beta * logits) * (1 - self.loss_cfg.label_smoothing)
            - F.logsigmoid(-self.loss_cfg.beta * logits) * self.loss_cfg.label_smoothing
        )
        return loss

    def _dpo_loss_robust(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute robust DPO loss."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.loss_cfg.reference_free:
            ref_logratios = torch.zeros_like(pi_logratios)
        else:
            ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios

        loss = (
            -F.logsigmoid(self.loss_cfg.beta * logits) * (1 - self.loss_cfg.label_smoothing)
            + F.logsigmoid(-self.loss_cfg.beta * logits) * self.loss_cfg.label_smoothing
        ) / (1 - 2 * self.loss_cfg.label_smoothing + 1e-8)
        return loss

    def _dpo_loss_hinge(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hinge loss (SLiC style)."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.loss_cfg.reference_free:
            ref_logratios = torch.zeros_like(pi_logratios)
        else:
            ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios
        return torch.relu(1 - self.loss_cfg.beta * logits)

    def _dpo_loss_ipo(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IPO loss."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.loss_cfg.reference_free:
            ref_logratios = torch.zeros_like(pi_logratios)
        else:
            ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios
        return (logits - 1 / (2 * self.loss_cfg.beta)) ** 2

    def _bco_pair_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCO (Binary Classifier Optimization) pairwise loss.

        BCO optimizes for absolute quality rather than relative preference.
        """
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.loss_cfg.beta * chosen_logratios
        rejected_rewards = self.loss_cfg.beta * rejected_logratios

        # Update running mean
        rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
        self.running.update(rewards)
        delta = self.running.mean

        loss = -F.logsigmoid(chosen_rewards - delta) - F.logsigmoid(-(rejected_rewards - delta))
        return loss

    def _nca_pair_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NCA pairwise loss."""
        chosen_rewards = (policy_chosen_logps - ref_chosen_logps) * self.loss_cfg.beta
        rejected_rewards = (policy_rejected_logps - ref_rejected_logps) * self.loss_cfg.beta

        loss = (
            -F.logsigmoid(chosen_rewards)
            - 0.5 * F.logsigmoid(-chosen_rewards)
            - 0.5 * F.logsigmoid(-rejected_rewards)
        )
        return loss

    def _sppo_hard_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SPPO hard loss."""
        a = policy_chosen_logps - ref_chosen_logps
        b = policy_rejected_logps - ref_rejected_logps
        return (a - 0.5 / self.loss_cfg.beta) ** 2 + (b + 0.5 / self.loss_cfg.beta) ** 2

    def _sft_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        loss_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SFT (Supervised Fine-Tuning) loss on chosen responses.

        This maintains generation quality during preference optimization.
        """
        # Only compute loss on chosen tokens (mask is for chosen part)
        sft_logits = logits[:, :mask.shape[1]]
        sft_labels = labels[:, :mask.shape[1]]
        sft_loss_weight = loss_weight[:, :mask.shape[1]]

        # Cross entropy loss
        vocab_size = sft_logits.shape[-1]
        sft_loss = F.cross_entropy(
            sft_logits.reshape(-1, vocab_size),
            sft_labels.reshape(-1),
            reduction="none",
            ignore_index=self.loss_cfg.ignore_idx,
        ).reshape(sft_labels.shape)

        # Apply loss weight and sum
        return (sft_loss * sft_loss_weight * mask.float()).sum()

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: DPOLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        """Compute combined DPO loss.

        This method computes all configured loss types and combines them
        with their respective weights.
        """
        # Compute logits
        logits = F.linear(hidden_states, head_weight, head_bias)
        logits = logits.float()

        shifted_labels = loss_kwargs.shifted_labels
        chosen_mask = loss_kwargs.chosen_mask
        rejected_mask = loss_kwargs.rejected_mask
        loss_weight = loss_kwargs.loss_weight

        # Compute per-token log probabilities
        all_logprobs = gather_logprobs(logits, shifted_labels)

        # Split into chosen and rejected parts
        chosen_len = chosen_mask.shape[1]
        if chosen_mask.any():
            # Get chosen logprobs - using the chosen part of the sequence
            chosen_logprobs = all_logprobs * chosen_mask.float()
            if self.loss_cfg.use_average_log_prob:
                policy_chosen_logps = chosen_logprobs.sum(dim=-1) / chosen_mask.sum(dim=-1).clamp(min=1)
            else:
                policy_chosen_logps = chosen_logprobs.sum(dim=-1)
        else:
            policy_chosen_logps = torch.zeros(1, device=logits.device)

        if rejected_mask.any():
            # Get rejected logprobs - using the rejected part of the sequence
            rejected_logprobs = all_logprobs * rejected_mask.float()
            if self.loss_cfg.use_average_log_prob:
                policy_rejected_logps = rejected_logprobs.sum(dim=-1) / rejected_mask.sum(dim=-1).clamp(min=1)
            else:
                policy_rejected_logps = rejected_logprobs.sum(dim=-1)
        else:
            policy_rejected_logps = torch.zeros(1, device=logits.device)

        # Get reference logprobs
        ref_chosen_logps = loss_kwargs.ref_chosen_logprobs
        ref_rejected_logps = loss_kwargs.ref_rejected_logprobs

        if ref_chosen_logps is None:
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
        if ref_rejected_logps is None:
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

        # Compute combined loss
        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        extra_info = {}

        for loss_type, weight in zip(self.loss_cfg.loss_types, self.loss_cfg.loss_weights):
            if loss_type == "sigmoid":
                loss = self._dpo_loss_sigmoid(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                loss = loss.mean() * weight
                extra_info["dpo_sigmoid_loss"] = loss.detach()

            elif loss_type == "robust":
                loss = self._dpo_loss_robust(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                loss = loss.mean() * weight
                extra_info["dpo_robust_loss"] = loss.detach()

            elif loss_type == "hinge":
                loss = self._dpo_loss_hinge(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                loss = loss.mean() * weight
                extra_info["dpo_hinge_loss"] = loss.detach()

            elif loss_type == "ipo":
                loss = self._dpo_loss_ipo(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                loss = loss.mean() * weight
                extra_info["dpo_ipo_loss"] = loss.detach()

            elif loss_type == "bco_pair":
                loss = self._bco_pair_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                loss = loss.mean() * weight
                extra_info["bco_pair_loss"] = loss.detach()

            elif loss_type == "nca_pair":
                loss = self._nca_pair_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                loss = loss.mean() * weight
                extra_info["nca_pair_loss"] = loss.detach()

            elif loss_type == "sppo_hard":
                loss = self._sppo_hard_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                loss = loss.mean() * weight
                extra_info["sppo_hard_loss"] = loss.detach()

            elif loss_type == "sft":
                # SFT loss only on chosen responses
                chosen_len_actual = chosen_mask.shape[1]
                # Note: We need to handle the chosen part separately
                chosen_labels = shifted_labels[:, :chosen_len_actual]
                chosen_logits = logits[:, :chosen_len_actual]
                chosen_loss_weight = loss_weight[:, :chosen_len_actual]

                loss = self._sft_loss(
                    chosen_logits,
                    chosen_labels,
                    chosen_mask[:, :chosen_len_actual],
                    chosen_loss_weight,
                )
                loss = loss * weight
                extra_info["sft_loss"] = loss.detach()

            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            total_loss = total_loss + loss

        # Compute rewards for logging
        chosen_rewards = self.loss_cfg.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.loss_cfg.beta * (policy_rejected_logps - ref_rejected_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        extra_info.update({
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean(),
            "reward_accuracy": reward_accuracies.mean(),
            "policy_chosen_logps": policy_chosen_logps.mean().detach(),
            "policy_rejected_logps": policy_rejected_logps.mean().detach(),
        })

        return total_loss, (logits, extra_info)

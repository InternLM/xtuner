# Copyright (c) OpenMMLab. All rights reserved.
"""Score centering for training-inference mismatch (Marek & Ryabinin, arXiv:2609.20807).

With sampler distribution ``q``, current policy ``p``, sampled token ``y`` and a token-level weight ``f(r)`` of the
ratio ``r = p / q``, the centered REINFORCE gradient of one token is

    A * (f(p_y / q_y) * grad log p_y - sum_v q_v * f(p_v / q_v) * grad log p_v)

Subtracting it makes the weighted score mean-zero under ``y ~ q``, which cancels the drift term
``E_q[A] * E_q[f * grad log p]`` that a mismatch ``q != p`` adds to the policy gradient. The sampler
records only a top-k head ``H`` of ``q``; its tail is modeled as the trainer's tail rescaled to the sampler's tail
mass, ``q_v = rho * p_v`` outside ``H`` with ``rho = (1 - Q_H) / (1 - P_H)``. Because ``sum_v p_v grad log p_v = 0``,
the tail term folds into the head with ``alpha = rho * f(1 / rho)``:

    sum_v q_v f(p_v / q_v) grad log p_v = sum_{v in H} (q_v f(p_v / q_v) - alpha p_v) grad log p_v
"""

import math
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import Field, model_validator
from typing_extensions import Self

from .base_loss import (
    BaseRLLossConfig,
    BaseRLLossContext,
    BaseRLLossKwargs,
    calibrate_token_mean_loss_weights,
)
from .loss_fn import kl_penalty


ScoreCenteringWeight = Literal["none", "tis", "mis"]

_SUM_METRIC_KEYS = (
    "reduced_sc_correction_sum",
    "reduced_sc_sampler_head_mass_sum",
    "reduced_sc_train_head_mass_sum",
    "reduced_sc_tail_ratio_sum",
    "reduced_sc_weight_sum",
    "reduced_sc_weight_zero_count",
    "reduced_sc_rollout_kl3_sum",
    "reduced_sc_rollout_abs_logprob_diff_sum",
    "reduced_sc_valid_count",
)
_MAX_METRIC_KEYS = ("reduced_sc_tail_ratio_max",)
_MEAN_METRICS = {
    "sc_correction": "reduced_sc_correction_sum",
    "sc_sampler_head_mass": "reduced_sc_sampler_head_mass_sum",
    "sc_train_head_mass": "reduced_sc_train_head_mass_sum",
    "sc_tail_ratio_mean": "reduced_sc_tail_ratio_sum",
    "sc_weight": "reduced_sc_weight_sum",
    "sc_weight_zero_frac": "reduced_sc_weight_zero_count",
    "sc_rollout_kl3": "reduced_sc_rollout_kl3_sum",
    "sc_rollout_abs_logprob_diff": "reduced_sc_rollout_abs_logprob_diff_sum",
}


class ScoreCenteringLossConfig(BaseRLLossConfig):
    """Configuration of the score-centered REINFORCE loss.

    The loss is a REINFORCE objective, not a PPO clip objective, and it does not read old-policy log probabilities:
    the token weights are computed inside the loss from the current policy and the sampler's distribution, so they
    are a valid off-policy correction for any number of optimizer steps on one batch. Every pack must carry
    ``rollout_logprobs`` (the sampler's log probability of the sampled token) and the sampler's top-k head
    (``sampler_topk_ids`` / ``sampler_topk_logprobs``, full-vocabulary normalized). The sampler must have drawn from
    exactly the recorded distribution: fixed temperature, no top-p/top-k/min-p truncation, no penalties, no
    constrained or speculative decoding. The trainer cannot check this; the data producer must guarantee it.

    Args:
        policy_loss_cfg (dict[str, Any]): Unused by this loss and must stay empty. Defaults to ``{}``.
        weight (Literal["none", "tis", "mis"]): Token-level weight ``f(r)`` of ``r = p / q``. ``"none"`` is 1,
            ``"tis"`` is ``min(r, upper)``, ``"mis"`` is ``r`` inside ``[lower, upper]`` and 0 outside.
            Defaults to ``"none"``.
        weight_bounds (tuple[float, float]): ``(upper, lower)`` bounds of ``weight``, following the
            ``RolloutImportanceSampling`` convention; ``"tis"`` uses ``upper`` only. Defaults to ``(2.0, 0.0)``.
        tail_eps (float): Floor of the head-complement masses in the tail ratio ``rho``. Defaults to ``1e-6``.
        use_kl_loss (bool): Whether to add the KL penalty to a reference model. Defaults to False.
        kl_loss_coef (float): Coefficient of the KL penalty. Defaults to 0.001.
        kl_loss_type (Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None): KL estimator.
            Defaults to None.
        rollout_is (RolloutImportanceSampling): Must keep its default (no thresholds): the weights are applied
            inside this loss. Defaults to ``RolloutImportanceSampling()``.

    **Examples:**

    Example score-centering loss with truncated importance weights::

        config = ScoreCenteringLossConfig(weight="tis", weight_bounds=(2.0, 0.0), mode="chunk", chunk_size=1024)
    """

    policy_loss_cfg: dict[str, Any] = {}
    weight: ScoreCenteringWeight = "none"
    weight_bounds: tuple[float, float] = (2.0, 0.0)
    tail_eps: float = Field(default=1e-6, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_options(self) -> Self:
        if self.policy_loss_cfg:
            raise ValueError(
                "Score centering is a REINFORCE objective and reads no policy_loss_cfg; "
                f"remove {sorted(self.policy_loss_cfg)}"
            )
        rollout_is = self.rollout_is
        if (
            rollout_is.rollout_is_threshold is not None
            or rollout_is.rollout_is_mask_threshold is not None
            or rollout_is.rollout_is_veto_threshold is not None
        ):
            raise ValueError(
                "Score centering applies its token weights inside the loss; leave rollout_is at its default "
                "and set weight / weight_bounds instead"
            )
        upper, lower = self.weight_bounds
        if not (math.isfinite(upper) and math.isfinite(lower) and upper > lower >= 0.0):
            raise ValueError(
                f"weight_bounds must be finite (upper, lower) with upper > lower >= 0, got {self.weight_bounds}"
            )
        if self.mode == "liger":
            raise ValueError("Score centering needs the logits and supports only mode='eager' or 'chunk'")
        return self

    @property
    def loss_ctx_cls(self) -> type["ScoreCenteringLossContext"]:
        return ScoreCenteringLossContext

    @property
    def required_loss_keys(self) -> list[str]:
        """Pack fields read by the score-centering loss.

        Returns:
            list[str]: ``rollout_logprobs``, ``sampler_topk_ids`` and ``sampler_topk_logprobs``.
        """
        return ["rollout_logprobs", "sampler_topk_ids", "sampler_topk_logprobs"]

    @property
    def _loss_kwargs_cls(self) -> type["ScoreCenteringLossKwargs"]:
        return ScoreCenteringLossKwargs

    def finalize_metrics(self, extra_info_dict: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
        """Reduce the score-centering metrics across ranks into token means.

        Args:
            extra_info_dict (dict[str, Any]): Per-rank summed metrics of one optimizer step.
            device (str | torch.device): Device used for the cross-rank reduction.

        Returns:
            dict[str, Any]: Metrics with the ``reduced_sc_*`` sums replaced by ``sc_*`` means and maxima.
        """
        extra_info_dict = super().finalize_metrics(extra_info_dict, device)
        if "reduced_sc_valid_count" not in extra_info_dict:
            return extra_info_dict

        sums = torch.tensor(
            [float(extra_info_dict.pop(key, 0.0)) for key in _SUM_METRIC_KEYS], dtype=torch.float32, device=device
        )
        maxima = torch.tensor(
            [float(extra_info_dict.pop(key, 0.0)) for key in _MAX_METRIC_KEYS], dtype=torch.float32, device=device
        )
        if dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        sum_values = dict(zip(_SUM_METRIC_KEYS, sums.tolist()))
        valid_count = sum_values["reduced_sc_valid_count"]
        for output_key, sum_key in _MEAN_METRICS.items():
            extra_info_dict[output_key] = sum_values[sum_key] / valid_count if valid_count > 0 else 0.0
        extra_info_dict["sc_tail_ratio_max"] = maxima[0].item() if valid_count > 0 else 0.0
        return extra_info_dict


class ScoreCenteringLossKwargs(BaseRLLossKwargs):
    """Keyword arguments of the score-centering loss.

    Args:
        shifted_labels (torch.Tensor): Sampled tokens ``y``, ``[1, T]``.
        advantages (torch.Tensor): Detached advantages, ``[1, T]``.
        rollout_logprobs (torch.Tensor | None): Sampler log probability of ``y``, ``[1, T]``.
        sampler_topk_ids (torch.Tensor | None): Sampler head token ids, ``[1, T, k]``, ``-1`` on unused slots.
        sampler_topk_logprobs (torch.Tensor | None): Full-vocabulary sampler log probabilities of the head,
            ``[1, T, k]``, ``-inf`` on unused slots.
        policy_loss_weight (torch.Tensor | None): Global token-mean weights set by ``build_batches``.
        ref_logprobs (torch.Tensor | None): Reference log probabilities for the KL penalty, if used.
        kl_loss_weight (torch.Tensor | None): KL penalty weights set by ``build_batches``, if used.
    """


class ScoreCenteringLossContext(BaseRLLossContext):
    """Score-centering loss context.

    Args:
        loss_cfg (ScoreCenteringLossConfig): Loss configuration.
        loss_kwargs (ScoreCenteringLossKwargs): Local loss inputs of one pack.
    """

    loss_cfg: ScoreCenteringLossConfig
    loss_kwargs: ScoreCenteringLossKwargs

    @staticmethod
    def build_batches(  # type: ignore[override]
        loss_ctx_list: list["ScoreCenteringLossContext"],
    ) -> list["ScoreCenteringLossContext"]:
        """Calibrate one optimizer step's contexts to a global token mean.

        Args:
            loss_ctx_list (list[ScoreCenteringLossContext]): Local contexts of one optimizer step.

        Returns:
            list[ScoreCenteringLossContext]: The same contexts with policy and KL weights set.
        """
        calibrate_token_mean_loss_weights(loss_ctx_list)
        return loss_ctx_list

    def loss_fn(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: ScoreCenteringLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        """Compute the score-centered loss of one chunk.

        Args:
            hidden_states (torch.Tensor): Final hidden states of the chunk, ``[1, c, H]``.
            head_weight (torch.Tensor): LM head weight, ``[V, H]``.
            head_bias (torch.Tensor | None): LM head bias.
            loss_kwargs (ScoreCenteringLossKwargs): Loss inputs of the chunk.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]: The weighted loss sum, the logits and
            the per-chunk metric sums.
        """
        cfg = self.loss_cfg
        shifted_labels = loss_kwargs.shifted_labels
        advantages = loss_kwargs.advantages
        policy_loss_weight = cast(torch.Tensor, loss_kwargs.policy_loss_weight)
        rollout_logprobs = loss_kwargs.rollout_logprobs
        head_ids = loss_kwargs.sampler_topk_ids
        sampler_head_logprobs = loss_kwargs.sampler_topk_logprobs
        if rollout_logprobs is None or head_ids is None or sampler_head_logprobs is None:
            raise ValueError("Score centering requires rollout_logprobs, sampler_topk_ids and sampler_topk_logprobs")

        logits = F.linear(hidden_states, head_weight, head_bias).float()
        # One normalizer for the sampled token and the head; no full-vocabulary log-softmax is kept.
        log_normalizer = logits.logsumexp(dim=-1, keepdim=True)
        logp_y = logits.gather(-1, shifted_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1) - log_normalizer.squeeze(-1)
        in_head = head_ids >= 0
        # Unused slots gather token 0; their coefficient is exactly 0, so they add neither value nor gradient.
        logp_head = logits.gather(-1, head_ids.clamp_min(0)) - log_normalizer

        with torch.no_grad():
            valid = shifted_labels != cfg.ignore_idx
            # -inf never reaches exp or the autograd graph: unused slots are replaced by 0 before any arithmetic.
            q_logp_head = sampler_head_logprobs.masked_fill(~in_head, 0.0)
            p_head = logp_head.detach().exp() * in_head
            q_head = q_logp_head.exp() * in_head
            train_head_mass = p_head.sum(-1)
            sampler_head_mass = q_head.sum(-1)
            log_tail_ratio = (1.0 - sampler_head_mass).clamp_min(cfg.tail_eps).log() - (
                1.0 - train_head_mass
            ).clamp_min(cfg.tail_eps).log()
            tail_ratio = log_tail_ratio.exp()
            alpha = tail_ratio * self._weight(-log_tail_ratio)
            head_is_weight = self._weight(logp_head.detach() - q_logp_head)
            coeff = torch.where(in_head, q_head * head_is_weight - alpha.unsqueeze(-1) * p_head, 0.0)
            log_ratio_y = logp_y.detach() - rollout_logprobs
            weight_y = self._weight(log_ratio_y)

        correction = (coeff * logp_head).sum(-1)
        per_token_loss = -advantages * (weight_y * logp_y - correction)
        loss = (per_token_loss * policy_loss_weight).sum()

        if cfg.use_kl_loss:
            ref_logprobs = loss_kwargs.ref_logprobs
            kl_loss_weight = loss_kwargs.kl_loss_weight
            assert ref_logprobs is not None and kl_loss_weight is not None, (
                "loss_kwargs.ref_logprobs and loss_kwargs.kl_loss_weight can not be None when use_kl_loss=True"
            )
            loss = loss + kl_penalty(logp_y, ref_logprobs, kl_loss_weight, cfg.kl_loss_type)

        with torch.no_grad():
            valid_float = valid.float()
            log_ratio_safe = log_ratio_y.clamp(min=-20.0, max=20.0)
            extra_info = {
                "reduced_sc_correction_sum": (correction.detach() * valid_float).sum(),
                "reduced_sc_sampler_head_mass_sum": (sampler_head_mass * valid_float).sum(),
                "reduced_sc_train_head_mass_sum": (train_head_mass * valid_float).sum(),
                "reduced_sc_tail_ratio_sum": (tail_ratio * valid_float).sum(),
                "reduced_sc_tail_ratio_max": tail_ratio.masked_fill(~valid, 0.0).max(),
                "reduced_sc_weight_sum": (weight_y * valid_float).sum(),
                "reduced_sc_weight_zero_count": ((weight_y == 0) & valid).float().sum(),
                "reduced_sc_rollout_kl3_sum": ((log_ratio_safe.exp() - 1.0 - log_ratio_safe) * valid_float).sum(),
                "reduced_sc_rollout_abs_logprob_diff_sum": (log_ratio_y.abs() * valid_float).sum(),
                "reduced_sc_valid_count": valid_float.sum(),
            }
        return loss, (logits, extra_info)

    def _weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        # f(r) evaluated from log r, so neither r nor the bounds overflow; never differentiated.
        weight = self.loss_cfg.weight
        if weight == "none":
            return torch.ones_like(log_ratio)
        upper, lower = self.loss_cfg.weight_bounds
        log_upper = math.log(upper)
        ratio = log_ratio.clamp(max=log_upper).exp()
        if weight == "tis":
            return ratio
        log_lower = math.log(lower) if lower > 0 else float("-inf")
        in_band = (log_ratio >= log_lower) & (log_ratio <= log_upper)
        return torch.where(in_band, ratio, 0.0)

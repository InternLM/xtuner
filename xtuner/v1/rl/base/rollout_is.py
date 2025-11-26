# modified from https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/mismatch_helper.py
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rollout Importance Sampling (IS) Helper Module.

This module handles importance sampling weight computation for correcting
distribution mismatch between rollout policy (e.g., vLLM BFloat16) and
training policy (e.g., FSDP FP32).

Key Features:
1. Three aggregation levels: token, sequence, geometric
2. Two handling modes: truncate, mask
3. Per-token veto mechanism for catastrophic outliers
4. Memory-efficient computation to prevent CUDA OOM
5. Comprehensive metrics tracking

Usage Notes:
- compute_rollout_importance_weights() computes both IS weights and mismatch metrics
- Used in ray_trainer.py via compute_rollout_importance_weights_and_add_to_batch()
- Also used in dp_actor.py for distributed worker computations
- compute_mismatch_metrics() is called internally by compute_rollout_importance_weights()

References:
- When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda
- Off-policy RL: https://fengyao.notion.site/off-policy-rl
"""

from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.utils import convert_packed_to_padded, convert_padded_to_packed, masked_mean, masked_sum


class RolloutImportanceSampling(BaseModel):
    model_config = ConfigDict(title="RolloutImportanceSamplingConfig", extra="forbid", arbitrary_types_allowed=True)
    rollout_is_level: str = "token"
    rollout_is_mode: str = "truncate"
    rollout_is_threshold: Optional[Tuple[float, float]] = None
    rollout_is_mask_threshold: Optional[Tuple[float, float]] = None
    rollout_is_veto_threshold: Optional[Tuple[float, float]] = None

    def compute_rollout_importance_weights(
        self,
        old_log_prob: torch.Tensor,
        rollout_log_prob: torch.Tensor,
        num_tokens: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, dict[str, Any]]:
        return compute_rollout_importance_weights(
            old_log_prob,
            rollout_log_prob,
            num_tokens,
            response_mask,
            rollout_is_level=self.rollout_is_level,
            rollout_is_mode=self.rollout_is_mode,
            rollout_is_threshold=self.rollout_is_threshold,
            rollout_is_mask_threshold=self.rollout_is_mask_threshold,
            rollout_is_veto_threshold=self.rollout_is_veto_threshold,
        )


def compute_rollout_importance_weights(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    num_tokens: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is_level: str = "token",
    rollout_is_mode: str = "truncate",
    rollout_is_threshold: Optional[Tuple[float, float]] = None,
    rollout_is_mask_threshold: Optional[Tuple[float, float]] = None,
    rollout_is_veto_threshold: Optional[Tuple[float, float]] = None,
) -> tuple[Optional[torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Compute importance sampling weights and rejection mask for rollout-
    training mismatch.

    This function computes IS weights to correct for distribution mismatch between rollout
    and training policies, and applies rejection sampling for outliers.

    Key Design: Separation of IS Weights and Rejection Sampling
    - IS weights (rollout_is_weights): Ratios π_train/π_rollout with processing applied:
      * Safety-bounded to prevent overflow:
        - Token level: exp(clamp(log_ratio, -20, 20)) per token
        - Sequence level: exp(clamp(sum(log_ratio), -20, 20)) broadcast to all tokens
        - Geometric level: exp(clamp(mean(log_ratio), -20, 20)) broadcast to all tokens
      * Truncate mode: upper clamped via .clamp(max=upper_threshold)
      * Mask mode: safety-bounded ratios preserved (no threshold clamping)
      * All modes: zeroed at padding positions
      Used for policy gradient calculations
    - Response mask (modified_response_mask): Has rejection applied (mask mode + veto)
      Used for loss aggregation to exclude rejected samples from training

    Reference:
        When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda

    Memory-efficient implementation:
    - Log-space computation to prevent overflow
    - Safety bounds (exp(±20)) on all exponentiations
    - Metrics computed without large intermediate tensors

    Args:
        old_log_prob: Log probs from training policy (FSDP FP32), shape (batch_size, seq_length)
        rollout_log_prob: Log probs from rollout policy (vLLM BF16), shape (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding), shape (batch_size, seq_length)
        rollout_is_level: IS weight aggregation level
            - "token": Per-token ratios ρ_t = π_train(t)/π_rollout(t) (biased but low variance)
            - "sequence": Sequence product ρ_seq = ∏ρ_t (unbiased but high variance)
            - "geometric": Geometric mean ρ_geo = (∏ρ_t)^(1/T) (experimental trade-off)
        rollout_is_mode: Treatment of outlier IS weights
            - "truncate": Clamp weights at upper threshold only. No rejection for outlier ratios,
              but veto can still apply (TIS)
            - "mask": Reject tokens/sequences outside [lower, upper] via response_mask (MIS/rejection sampling)
        rollout_is_threshold: Tuple of (upper threshold, lower threshold) for IS weights (e.g., (2.0, 0.5))
        rollout_is_mask_threshold: Tuple of (upper threshold, lower threshold) for mask mode (e.g., (2.0, 0.5))
        rollout_is_veto_threshold: Tuple of (upper threshold, lower threshold), if any token has ratio < lower threshold or > upper threshold,
            reject entire sequence. Applied independently of rollout_is_mode. If None, veto disabled.

    Returns:
        Tuple of (is_weights, modified_response_mask, metrics):
            is_weights: IS weights, shape (batch_size, seq_length).
            modified_response_mask: Response mask with rejection applied:
                - truncate mode: unchanged for outlier ratios, but veto rejection still applied
                - mask mode: tokens outside [lower, upper] masked to 0
                - veto: sequences with catastrophic tokens masked to 0 (applied in both modes)
                Shape (batch_size, seq_length).
            metrics: Dict of IS and mismatch metrics, all scalars with "mismatch/" prefix
    """
    if rollout_is_threshold is None:
        return None, response_mask, {}

    assert rollout_is_mode in ["truncate", "mask", "both"], (
        f"Invalid rollout_is_mode: {rollout_is_mode}. Must be 'truncate', 'mask', or 'both'."
    )
    assert rollout_is_level in ["token", "sequence", "geometric"], (
        f"Invalid rollout_is_level: {rollout_is_level}. Must be 'token', 'sequence', or 'geometric'."
    )
    # Parse thresholds: if lower not specified, use 1/upper (reciprocal)
    upper_threshold, lower_threshold = rollout_is_threshold
    assert upper_threshold > lower_threshold, (
        f"upper_threshold must be greater than lower_threshold, but got {upper_threshold} and {lower_threshold}"
    )
    if rollout_is_mask_threshold is None:
        rollout_is_mask_threshold = rollout_is_threshold
    upper_mask_threshold, lower_mask_threshold = rollout_is_mask_threshold
    assert upper_mask_threshold > lower_mask_threshold, (
        f"upper_mask_threshold must be greater than lower_mask_threshold, but got {upper_mask_threshold} and {lower_mask_threshold}"
    )

    # convert packed sequence to padded sequence
    old_log_prob = convert_packed_to_padded(old_log_prob, num_tokens, padding_value=0, padding_side="right")
    rollout_log_prob = convert_packed_to_padded(rollout_log_prob, num_tokens, padding_value=0, padding_side="right")
    response_mask = convert_packed_to_padded(response_mask, num_tokens, padding_value=0, padding_side="right")

    # Step 1: Compute raw importance weights based on the specified level
    log_ratio = old_log_prob - rollout_log_prob

    # Pre-compute log thresholds
    device = old_log_prob.device
    log_threshold_upper = torch.log(torch.tensor(upper_threshold, device=device))
    log_threshold_lower = torch.log(torch.tensor(lower_threshold, device=device))

    # Safety bound to prevent numerical overflow (exp(20) ≈ 485M)
    SAFETY_BOUND = 20.0

    # Store unclamped values in log-space for accurate metrics
    if rollout_is_level == "token":
        # Token-level IS: π_train(a|s) / π_rollout(a|s) per token
        log_ratio_for_metrics = log_ratio

        # Apply safety bound to prevent overflow
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_safe)

    elif rollout_is_level == "sequence":
        # Sequence-level IS: π_train(y|x) / π_rollout(y|x) for entire sequence
        # Product of token ratios: exp(Σ log(π_train/π_rollout))
        log_ratio_sum = masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
        log_ratio_for_metrics = log_ratio_sum  # Store for metrics

        # Apply safety bound to prevent overflow
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(old_log_prob)

    elif rollout_is_level == "geometric":
        # Geometric mean IS: (∏ π_train/π_rollout)^(1/T)
        # Equivalent to exp(mean(log(π_train/π_rollout)))
        log_ratio_mean = masked_mean(log_ratio, response_mask, axis=-1).unsqueeze(-1)
        log_ratio_for_metrics = log_ratio_mean  # Store for metrics

        # Geometric mean rarely explodes due to averaging, but apply safety bound anyway
        log_ratio_mean_safe = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_mean_safe).expand_as(old_log_prob)

    else:
        raise ValueError(f"Invalid rollout_is_level: {rollout_is_level}. Must be 'token', 'sequence', or 'geometric'.")

    # Step 1.5: Apply per-token veto check in log space (memory efficient)
    if rollout_is_veto_threshold is not None:
        upper_veto_threshold, lower_veto_threshold = rollout_is_veto_threshold
        log_veto_threshold_lower = torch.log(torch.tensor(lower_veto_threshold, device=device))
        log_veto_threshold_upper = torch.log(torch.tensor(upper_veto_threshold, device=device))

        # Check if any token ratio is below veto threshold (in log space)
        # log(π_train/π_rollout) < log(veto_threshold) ⟺ π_train/π_rollout < veto_threshold
        catastrophic_tokens = (
            (log_ratio < log_veto_threshold_lower) | (log_ratio > log_veto_threshold_upper)
        ) & response_mask.bool()

        # For each sequence, check if it has any catastrophic token
        # Use broadcasting instead of expand_as to save memory
        has_catastrophic = catastrophic_tokens.any(dim=-1, keepdim=True)

        # Create veto mask: 0 if sequence has catastrophic token, 1 otherwise
        veto_mask = (~has_catastrophic).float()
    else:
        # No veto mechanism
        catastrophic_tokens = torch.zeros_like(response_mask, dtype=torch.bool)
        has_catastrophic = torch.zeros((old_log_prob.size(0), 1), dtype=torch.bool, device=device)
        veto_mask = torch.ones((old_log_prob.size(0), 1), dtype=torch.float32, device=device)

    # Step 2: Compute comprehensive metrics
    metrics = compute_is_metrics(
        rollout_is_weights=rollout_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        rollout_is_level=rollout_is_level,
        rollout_is_threshold_upper=upper_threshold,
        rollout_is_threshold_lower=lower_threshold,
        log_threshold_upper=log_threshold_upper,
        log_threshold_lower=log_threshold_lower,
        has_catastrophic=has_catastrophic,
        catastrophic_tokens=catastrophic_tokens,
        SAFETY_BOUND=SAFETY_BOUND,
    )

    # Step 3: Apply outlier handling and rejection sampling
    # Key design principle: IS weights and rejection are separate mechanisms
    # - rollout_is_weights: IS weight ratios with mode-specific processing
    #   * Truncate mode: upper clamped to prevent extreme values
    #   * Mask mode: safety-bounded ratios preserved (no threshold clamping, rejection via mask)
    #   Used for policy gradient calculations
    # - modified_response_mask: Has rejection applied (excludes outliers from training)
    #   Used for loss denominator: ensures rejected samples don't dilute gradients

    modified_response_mask = response_mask
    if rollout_is_mode in ["mask", "both"]:
        # Masked IS (MIS): rejection sampling for outlier IS weights
        # Reject tokens/sequences with IS ratios outside [lower, upper] via response_mask
        # IS weights themselves are NOT threshold-clamped (remain safety-bounded only)
        mask = (rollout_is_weights >= lower_mask_threshold) & (rollout_is_weights <= upper_mask_threshold)
        mask = mask.float()

        # Compute rejection rate metrics
        metrics["rollout_is_masked_fraction"] = masked_mean(1 - mask, response_mask)
        if rollout_is_level in ["sequence", "geometric"]:
            # Sequence-level: all tokens have same weight, check first token
            metrics["rollout_is_seq_masked_fraction"] = (1 - mask[:, 0]).mean()
        else:
            # Token-level: sequence rejected if ANY token is rejected
            seq_has_masked = masked_sum(1 - mask, response_mask, axis=-1) > 0
            metrics["rollout_is_seq_masked_fraction"] = seq_has_masked.float().mean()

        # Apply rejection via response_mask (NOT by clamping IS weights)
        modified_response_mask = response_mask * mask
        # rollout_is_weights kept as safety-bounded ratios (no threshold clamping)
    if rollout_is_mode in ["truncate", "both"]:
        # Truncated IS (TIS): clamp weights to prevent extreme importance ratios
        # Weights are modified by clamping; no rejection via mask for outlier ratios
        # Veto rejection (if enabled) will still be applied to modified_response_mask below
        rollout_is_weights = rollout_is_weights.clamp(max=upper_threshold, min=lower_threshold)

    # Apply veto: reject entire sequences with catastrophic tokens (ratio < veto_threshold)
    # Veto is independent of mode - it applies to modified_response_mask after mode-specific handling
    modified_response_mask = modified_response_mask * veto_mask
    # Note: rollout_is_weights unaffected by veto (already clamped in truncate mode, or kept as-is in mask mode)

    # Zero out padding positions in IS weights for correct aggregation
    # This is different from rejection - padding must be zeroed regardless of mode
    rollout_is_weights = rollout_is_weights * response_mask

    # Compute mismatch metrics (KL, PPL, etc.) and merge with IS metrics
    mismatch_metrics = compute_mismatch_metrics(
        old_log_prob=old_log_prob, rollout_log_prob=rollout_log_prob, response_mask=response_mask
    )
    metrics.update(mismatch_metrics)

    # Convert all tensor metrics to scalars for logging
    # Note: No need to detach since old_log_prob and rollout_log_prob are computed with torch.no_grad()
    metrics_scalar = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_scalar[f"mismatch/{key}"] = value.item()
        else:
            metrics_scalar[f"mismatch/{key}"] = value

    rollout_is_weights = convert_padded_to_packed(rollout_is_weights, num_tokens, padding_side="right")
    modified_response_mask = convert_padded_to_packed(modified_response_mask, num_tokens, padding_side="right")
    return rollout_is_weights, modified_response_mask, metrics_scalar


def compute_is_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is_level: str,
    rollout_is_threshold_upper: float,
    rollout_is_threshold_lower: float,
    log_threshold_upper: torch.Tensor,
    log_threshold_lower: torch.Tensor,
    has_catastrophic: torch.Tensor,
    catastrophic_tokens: torch.Tensor,
    SAFETY_BOUND: float,
) -> dict[str, Any]:
    """Compute comprehensive metrics for importance sampling weights.

    Reference:
        When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda

    This function computes metrics using a mix of true unclamped values (for max/min/fractions
    in sequence/geometric mode via log-space) and safety-clamped values (for mean/std/ESS)
    to balance accuracy with numerical stability and avoid overflow.
    """
    # Validate that we have at least one valid sample
    assert response_mask.any(), "Expected at least one valid sample in response_mask"

    metrics = {}
    device = rollout_is_weights.device

    # Track veto statistics
    metrics["rollout_is_veto_fraction"] = has_catastrophic.float().mean()
    metrics["rollout_is_catastrophic_token_fraction"] = masked_mean(catastrophic_tokens.float(), response_mask)

    # Compute metrics based on IS level
    if rollout_is_level in ["sequence", "geometric"]:
        # For sequence/geometric, compute true statistics from log-space
        # This reflects the actual distribution before clamping

        # True max/min in log space
        log_max = log_ratio_for_metrics.max()
        log_min = log_ratio_for_metrics.min()

        # Convert to regular space with safety bound
        metrics["rollout_is_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND))
        metrics["rollout_is_min"] = torch.exp(log_min)

        # Mean uses clamped weights to avoid overflow
        metrics["rollout_is_mean"] = masked_mean(rollout_is_weights, response_mask)

        # Compute fraction exceeding threshold in log space (accurate)
        exceeds_upper = log_ratio_for_metrics > log_threshold_upper
        below_lower = log_ratio_for_metrics < log_threshold_lower

        if rollout_is_level == "sequence":
            # For sequence level, all tokens in a sequence have the same weight
            metrics["rollout_is_ratio_fraction_high"] = exceeds_upper.float().mean()
            metrics["rollout_is_ratio_fraction_low"] = below_lower.float().mean()
        else:  # geometric
            # Need to expand to match token dimensions
            exceeds_upper_expanded = exceeds_upper.expand_as(response_mask)
            below_lower_expanded = below_lower.expand_as(response_mask)
            metrics["rollout_is_ratio_fraction_high"] = masked_mean(exceeds_upper_expanded.float(), response_mask)
            metrics["rollout_is_ratio_fraction_low"] = masked_mean(below_lower_expanded.float(), response_mask)

    else:
        # Token-level: compute directly from weights
        metrics["rollout_is_mean"] = masked_mean(rollout_is_weights, response_mask)

        # Fraction exceeding thresholds
        rollout_is_above_threshold = rollout_is_weights > rollout_is_threshold_upper
        rollout_is_below_threshold = rollout_is_weights < rollout_is_threshold_lower
        metrics["rollout_is_ratio_fraction_high"] = masked_mean(rollout_is_above_threshold.float(), response_mask)
        metrics["rollout_is_ratio_fraction_low"] = masked_mean(rollout_is_below_threshold.float(), response_mask)

        # Max/min for token level
        mask_bool = response_mask.bool()
        metrics["rollout_is_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max()
        metrics["rollout_is_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min()

    # Compute standard deviation using clamped weights to avoid overflow
    mask_count = response_mask.sum()
    if mask_count > 1:
        # Use clamped weights for variance to avoid squaring huge values
        weights_for_std = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold_upper)
        # Use mean from clamped weights for consistency
        mean_clamped = masked_mean(weights_for_std, response_mask)
        rollout_is_var = masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        metrics["rollout_is_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0))
    else:
        metrics["rollout_is_std"] = torch.tensor(0.0, device=device)

    # Effective sample size (use clamped weights to avoid overflow)
    weights_for_ess = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold_upper)
    mean_for_ess = masked_mean(weights_for_ess, response_mask)
    is_weights_normalized = weights_for_ess / (mean_for_ess + 1e-8)
    metrics["rollout_is_eff_sample_size"] = 1.0 / masked_mean(is_weights_normalized.square(), response_mask)

    # Per-sequence breakdown metrics
    if rollout_is_weights.dim() > 1:
        # Compute mean IS weight per sequence
        seq_mean_weights = masked_mean(rollout_is_weights, response_mask, axis=-1)

        # Per-sequence statistics
        metrics["rollout_is_seq_mean"] = seq_mean_weights.mean()
        metrics["rollout_is_seq_std"] = (
            seq_mean_weights.std() if seq_mean_weights.numel() > 1 else torch.tensor(0.0, device=device)
        )
        metrics["rollout_is_seq_max"] = seq_mean_weights.max()
        metrics["rollout_is_seq_min"] = seq_mean_weights.min()

        # Identify most problematic sequences
        seq_deviation = (seq_mean_weights - 1.0).abs()
        metrics["rollout_is_seq_max_deviation"] = seq_deviation.max()

        # Fraction of sequences with high IS weights
        metrics["rollout_is_seq_fraction_high"] = (seq_mean_weights > rollout_is_threshold_upper).float().mean()
        metrics["rollout_is_seq_fraction_low"] = (seq_mean_weights < rollout_is_threshold_lower).float().mean()

    return metrics


def compute_mismatch_metrics(
    old_log_prob: torch.Tensor,
    rollout_log_prob: Optional[torch.Tensor],
    response_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compute training-inference mismatch metrics (helper function).

    This helper function operates on raw tensors and is used internally by:
    - compute_rollout_importance_weights() in this module (automatically included)
    - Tests (test_rollout_is.py, test_rollout_is_integration.py)

    These metrics help diagnose the mismatch between the rollout policy (e.g., vLLM)
    and the training policy (e.g., FSDP), which can cause training instability.

    Key metrics:
    - mismatch_kl: Direct KL divergence estimator KL(π_rollout || π_training)
    - mismatch_k3_kl: K3 KL estimator for stability (more stable for small KL)
    - training_ppl: Perplexity of training policy
    - rollout_ppl: Perplexity of rollout policy
    - log_ppl_diff: Difference in log perplexities
    - ppl_ratio: Ratio of training PPL to rollout PPL

    Args:
        old_log_prob: Log probabilities from training policy, shape (batch_size, seq_length)
        rollout_log_prob: Log probabilities from rollout policy, shape (batch_size, seq_length)
        response_mask: Mask for valid tokens, shape (batch_size, seq_length)

    Returns:
        Dictionary of mismatch metrics (without prefix)

    Reference:
    - When Speed Kills Stability: https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda
    """
    # Validate that we have at least one valid token
    assert response_mask.any(), "Expected at least one valid token in response_mask"

    metrics = {}

    # 1. Training policy perplexity (always available)
    # Formula: exp(-1/|T| * Σ log π_training(y_t|y_<t))
    # where |T| is the number of tokens generated by the model
    mean_log_prob_training = masked_mean(old_log_prob, response_mask, axis=-1)  # (batch_size,)
    training_ppl = torch.exp(-mean_log_prob_training).mean()  # Batch mean of per-sequence PPL
    metrics["mismatch_training_ppl"] = training_ppl.detach().item()

    # Also log log-ppl for easier analysis (avoids exponential scale)
    metrics["mismatch_training_log_ppl"] = (-mean_log_prob_training).mean().detach().item()

    # 2. Compute rollout mismatch metrics (only if rollout_log_probs available)
    if rollout_log_prob is not None:
        # 2a. mismatch_kl: Direct estimator for KL(π_rollout || π_training)
        # This is the standard KL divergence: E[log(π_rollout) - log(π_training)]
        # Positive value means rollout policy is more confident than training policy
        metrics["mismatch_kl"] = masked_mean(rollout_log_prob - old_log_prob, response_mask).detach().item()

        # 2b. mismatch_k3_kl: K3 estimator for KL(π_rollout || π_training)
        # More stable for small KL values using: E[exp(log_ratio) - log_ratio - 1]
        # Formula: KL ≈ E[r - log(r) - 1] where r = π_training/π_rollout
        log_ratio = old_log_prob - rollout_log_prob
        mismatch_k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
        metrics["mismatch_k3_kl"] = masked_mean(mismatch_k3_kl_matrix, response_mask).detach().item()

        # 2c. Rollout policy perplexity
        mean_log_prob_rollout = masked_mean(rollout_log_prob, response_mask, axis=-1)  # (batch_size,)
        rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()  # Batch mean of per-sequence PPL
        metrics["mismatch_rollout_ppl"] = rollout_ppl.detach().item()
        metrics["mismatch_rollout_log_ppl"] = (-mean_log_prob_rollout).mean().detach().item()

        # 2d. Log PPL difference (sequence-level perplexity difference)
        # log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        # Since ppl = exp(-log_prob), we have:
        #   log(ppl_ratio) = log(training_ppl/rollout_ppl) = log_ppl_diff
        # Positive value means training assigns lower probability (higher PPL) than rollout
        log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        metrics["mismatch_log_ppl_diff"] = log_ppl_diff.mean().detach().item()
        metrics["mismatch_log_ppl_abs_diff"] = log_ppl_diff.abs().mean().detach().item()
        metrics["mismatch_log_ppl_diff_max"] = log_ppl_diff.max().detach().item()
        metrics["mismatch_log_ppl_diff_min"] = log_ppl_diff.min().detach().item()

        # 2e. PPL ratio (how much higher is training PPL vs rollout PPL)
        # IMPORTANT: Compute per-sequence ratio first, then average
        # For numerical stability, compute in log space using log_ppl_diff
        # Note: log_ppl_diff = log(ppl_ratio), so ppl_ratio = exp(log_ppl_diff)
        # This is the inverse of geometric IS: ppl_ratio_i = 1 / geometric_is_i for each sequence
        ppl_ratio = torch.exp(log_ppl_diff).mean()  # mean(exp(log_ppl_diff)) = mean(ppl_ratio_i)
        metrics["mismatch_ppl_ratio"] = ppl_ratio.detach().item()

    return metrics


def merge_rollout_is_metrics(rollout_is_metrics: list[dict[str, float]], device="cuda") -> dict[str, float]:
    metrics = {}
    for key in rollout_is_metrics[0].keys():
        all_values = [m[key] for m in rollout_is_metrics]
        if "max" in key:
            max_value = torch.tensor(all_values).max().to(torch.float32).to(device)
            dist.all_reduce(max_value, op=dist.ReduceOp.MAX)
            metrics[key] = max_value.item()
        elif "min" in key:
            min_value = torch.tensor(all_values).min().to(torch.float32).to(device)
            dist.all_reduce(min_value, op=dist.ReduceOp.MIN)
            metrics[key] = min_value.item()
        else:
            mean_value = torch.tensor(all_values).mean().to(torch.float32).to(device)
            dist.all_reduce(mean_value, op=dist.ReduceOp.AVG)
            metrics[key] = mean_value.item()
    return metrics

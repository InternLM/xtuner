from typing import Any, Callable

import torch
import torch.nn.functional as F

from xtuner.v1.data_proto.utils import convert_packed_to_padded


PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # loss_weights
        dict,  # config
    ],
    torch.Tensor,  # loss
]


POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        if "." in loss_name:  # try to manually import the loss function from a custom path
            try:
                import importlib

                package_name, module_name = loss_name.rsplit(".", 1)
                print(f"Importing loss function from {package_name}.{module_name}")
                module = importlib.import_module(package_name)
                loss_fn = getattr(module, module_name)
                POLICY_LOSS_REGISTRY[loss_name] = loss_fn
                return loss_fn
            except ImportError as e:
                raise ImportError(f"Failed to import loss function: {loss_name}, error: {e}")
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


def check_config(keys_needed: list[str], config: dict[str, Any]) -> None:
    """Check if the config contains all the required keys."""
    for key in keys_needed:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in config. Available keys: {list(config.keys())}")


@register_policy_loss("vanilla")
def pg_loss_fn(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    loss_weights: torch.Tensor,
    policy_loss_cfg: dict,
    enable_chunk_linear: bool = False,
    num_tokens: list[int] | None = None,
    shifted_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    check_config(["cliprange_low", "cliprange_high"], policy_loss_cfg)
    cliprange_low = policy_loss_cfg["cliprange_low"]
    cliprange_high = policy_loss_cfg["cliprange_high"]
    clip_ratio_c = policy_loss_cfg.get("clip_ratio_c", 3.0)
    log_prob_diff_min = policy_loss_cfg.get("log_prob_diff_min", -20.0)
    log_prob_diff_max = policy_loss_cfg.get("log_prob_diff_max", 20.0)
    advantages = advantages.to(log_prob.dtype)
    negative_approx_kl = log_prob - old_log_prob.detach()
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=log_prob_diff_min, max=log_prob_diff_max)
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -ratio * advantages
    pg_losses2 = -torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -clip_ratio_c * advantages
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    loss = (pg_losses * loss_weights.to(pg_losses.dtype)).sum()
    return loss


@register_policy_loss("gspo")
def gspo_loss_fn(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    loss_weights: torch.Tensor,
    policy_loss_cfg: dict,
    enable_chunk_linear: bool = True,
    num_tokens: list[int] | None = None,
    shifted_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    assert num_tokens is not None
    assert shifted_labels is not None
    assert enable_chunk_linear, "GSPO can only be supported when enable_chunk_linear"
    check_config(["cliprange_low", "cliprange_high"], policy_loss_cfg)
    cliprange_low = policy_loss_cfg["cliprange_low"]
    cliprange_high = policy_loss_cfg["cliprange_high"]
    clip_ratio_c = policy_loss_cfg.get("clip_ratio_c", 3.0)
    log_prob_diff_min = policy_loss_cfg.get("log_prob_diff_min", -20.0)
    log_prob_diff_max = policy_loss_cfg.get("log_prob_diff_max", 20.0)
    advantages = advantages.to(log_prob.dtype)
    response_mask = shifted_labels != -100

    # 1. unpack
    old_log_prob = convert_packed_to_padded(old_log_prob, num_tokens, padding_value=0, padding_side="right")
    log_prob = convert_packed_to_padded(log_prob, num_tokens, padding_value=0, padding_side="right")
    advantages = convert_packed_to_padded(advantages, num_tokens, padding_value=0, padding_side="right")
    response_mask = convert_packed_to_padded(response_mask, num_tokens, padding_value=0, padding_side="right")
    loss_weights = convert_packed_to_padded(loss_weights, num_tokens, padding_value=0, padding_side="right")

    # 2. calculate loss
    # adapted from https://github.com/verl-project/verl/blob/de9880d76467af6bcb9b5c12fad6dfa980e83d57/verl/trainer/ppo/core_algos.py#L1254
    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(
        log_seq_importance_ratio, min=log_prob_diff_min, max=log_prob_diff_max
    )  # clamp for numerical stability

    # finally exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -clip_ratio_c * advantages
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    loss = (pg_losses * loss_weights.to(pg_losses.dtype)).sum()

    return loss


def sft_loss_fn(
    logits: torch.Tensor,  # [1, seq_len, vocab_size]
    shifted_labels: torch.Tensor,  # [1, seq_len]
    loss_weights: torch.Tensor,  # [1, seq_len]
    ignore_idx: int = -100,
) -> torch.Tensor:
    sft_loss = F.cross_entropy(
        logits.squeeze(),
        shifted_labels.squeeze(),
        reduction="none",
        ignore_index=ignore_idx,
    ).unsqueeze(0)  # [1, seq_len]
    sft_loss = (sft_loss * loss_weights.to(sft_loss.dtype)).sum()
    return sft_loss


def kl_penalty(
    logprobs: torch.Tensor, ref_logprobs: torch.Tensor, loss_weights: torch.Tensor, kl_penalty
) -> torch.Tensor:
    """
    Modified from https://github.com/volcengine/verl/blob/313366fd85e95ad43d567a808dd647089723a255/verl/trainer/ppo/core_algos.py#L1272
    Compute KL divergence given logprobs and ref_logprobs.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html
    """
    if kl_penalty in ("kl", "k1"):
        loss = logprobs - ref_logprobs
        # return logprobs - ref_logprobs
    elif kl_penalty == "abs":
        loss = (logprobs - ref_logprobs).abs()
    elif kl_penalty in ("mse", "k2"):
        loss = 0.5 * (logprobs - ref_logprobs).square()
    elif kl_penalty in ("low_var_kl", "k3"):
        # J. Schulman. Approximating kl divergence, 2020.
        # # URL http://joschu.net/blog/kl-approx.html.
        kl = ref_logprobs - logprobs
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        loss = torch.clamp(kld, min=-10, max=10)
    else:
        raise NotImplementedError

    return (loss * loss_weights.to(loss.dtype)).sum()

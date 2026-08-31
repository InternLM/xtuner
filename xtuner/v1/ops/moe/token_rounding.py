from typing import Literal

import torch


TokenRoundingMode = Literal["nearest", "up", "down"]


def build_token_rounding_metadata(
    router_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    rounding_quantum: int,
    rounding_mode: TokenRoundingMode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build SonicMoE token-rounding routing metadata.

    This follows SonicMoE's token-choice rounding algorithm. Original top-k
    assignments have priority for each expert. Rounding up may then add the
    highest-scoring non-top-k tokens, while rounding down removes the
    lowest-priority original assignments.

    Args:
        router_weights (torch.Tensor): Full router probabilities with shape
            ``[num_tokens, num_experts]``.
        topk_ids (torch.Tensor): Original token-choice expert ids with shape
            ``[num_tokens, top_k]``.
        num_experts (int): Number of routed experts.
        rounding_quantum (int): Expert token-count tile size.
        rounding_mode (TokenRoundingMode): ``"nearest"``, ``"up"``, or
            ``"down"``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Flattened router
        scores, token indices, and expert indices accepted by SonicMoE's
        general-routing API.
    """
    if router_weights.ndim != 2 or router_weights.shape[1] != num_experts:
        raise ValueError(
            "router_weights must have shape [num_tokens, num_experts], "
            f"got {tuple(router_weights.shape)} for num_experts={num_experts}."
        )
    if topk_ids.ndim != 2 or topk_ids.shape[0] != router_weights.shape[0]:
        raise ValueError(
            "topk_ids must have shape [num_tokens, top_k] with the same token dimension as router_weights."
        )
    if rounding_quantum <= 0:
        raise ValueError(f"rounding_quantum must be positive, got {rounding_quantum}.")

    num_tokens = router_weights.shape[0]
    if num_tokens == 0:
        empty_indices = torch.empty(0, dtype=torch.int32, device=router_weights.device)
        return router_weights.new_empty(0, dtype=torch.float32), empty_indices, empty_indices.clone()

    topk_ids_long = topk_ids.to(dtype=torch.long)
    selected_priority = router_weights.gather(1, topk_ids_long)
    selected_priority = selected_priority / selected_priority.sum(dim=-1, keepdim=True).clamp_min(1e-20)

    # Keep every original token-choice assignment ahead of every expert-choice
    # candidate. Routing selection is discrete, so detach the priority matrix
    # while gathering the final scores from router_weights to preserve drouter.
    routing_priority = router_weights.detach() - 1
    routing_priority = routing_priority.scatter(1, topk_ids_long, selected_priority.detach())
    ranked_token_indices = routing_priority.argsort(dim=0, descending=True).to(torch.int32)

    expert_frequency = torch.bincount(topk_ids_long.reshape(-1), minlength=num_experts).to(torch.int32)
    if rounding_mode == "up":
        rounded_frequency = (
            torch.div(
                expert_frequency + rounding_quantum - 1,
                rounding_quantum,
                rounding_mode="floor",
            )
            * rounding_quantum
        )
    elif rounding_mode == "down":
        rounded_frequency = (
            torch.div(
                expert_frequency,
                rounding_quantum,
                rounding_mode="floor",
            )
            * rounding_quantum
        )
    elif rounding_mode == "nearest":
        rounded_frequency = torch.round(expert_frequency.float() / rounding_quantum).to(torch.int32)
        rounded_frequency = rounded_frequency * rounding_quantum
    else:
        raise ValueError(f"Unsupported token rounding mode: {rounding_mode!r}.")
    rounded_frequency = rounded_frequency.clamp(min=0, max=num_tokens)

    rank_mask = (
        torch.arange(num_tokens, device=router_weights.device, dtype=torch.int32)[:, None] < rounded_frequency[None, :]
    )
    token_indices = ranked_token_indices[rank_mask]
    expert_indices = torch.arange(num_experts, device=router_weights.device, dtype=torch.int32)[None, :].expand(
        num_tokens, -1
    )[rank_mask]

    # SonicMoE's reduction expects assignments ordered by token id.
    token_order = token_indices.argsort()
    token_indices = token_indices[token_order].contiguous()
    expert_indices = expert_indices[token_order].contiguous()
    router_scores = router_weights[token_indices.long(), expert_indices.long()].to(torch.float32).contiguous()
    return router_scores, token_indices, expert_indices

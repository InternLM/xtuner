from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NextTokenData:
    """A trajectory aligned to next-token prediction positions.

    Args:
        model_input_ids (torch.Tensor): Tokens consumed by the model.
        shifted_labels (torch.Tensor): Labels predicted at each model input position.
        action_mask (torch.Tensor): Whether each predicted token is controlled by the actor.
    """

    model_input_ids: torch.Tensor
    shifted_labels: torch.Tensor
    action_mask: torch.Tensor


def align_next_token_data(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    ignore_idx: int = -100,
) -> NextTokenData:
    """Align full token and label sequences to next-token prediction positions.

    Args:
        input_ids (torch.Tensor): One-dimensional full token sequence.
        labels (torch.Tensor): One-dimensional labels aligned with ``input_ids``.
        ignore_idx (int): Label value for non-action tokens.

    Returns:
        NextTokenData: Model inputs, shifted labels, and the action mask with equal lengths.
    """
    _validate_1d_tokens(input_ids, "input_ids")
    _validate_1d_tokens(labels, "labels")
    if input_ids.shape != labels.shape:
        raise ValueError(f"input_ids and labels must have the same shape, got {input_ids.shape} and {labels.shape}")
    if input_ids.numel() < 2:
        raise ValueError("A trajectory must contain at least two tokens for next-token prediction.")

    model_input_ids = input_ids[:-1]
    shifted_labels = labels[1:]
    action_mask = shifted_labels != ignore_idx
    return NextTokenData(model_input_ids, shifted_labels, action_mask)


def align_single_turn(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor | None = None,
    ignore_idx: int = -100,
) -> NextTokenData:
    """Build next-token-aligned data for a single-turn prompt and response.

    The first response token is predicted by the final prompt token. Consequently,
    the prompt prefix in ``action_mask`` has ``len(prompt_ids) - 1`` entries.

    Args:
        prompt_ids (torch.Tensor): One-dimensional prompt token IDs.
        response_ids (torch.Tensor): One-dimensional response token IDs, including the final token.
        response_mask (torch.Tensor | None): Per-response-token action mask. Defaults to all action tokens.
        ignore_idx (int): Label value for non-action tokens.

    Returns:
        NextTokenData: Model inputs, shifted labels, and the action mask with equal lengths.
    """
    _validate_1d_tokens(prompt_ids, "prompt_ids")
    _validate_1d_tokens(response_ids, "response_ids")
    if prompt_ids.numel() == 0:
        raise ValueError("prompt_ids must not be empty.")
    if response_ids.numel() == 0:
        raise ValueError("response_ids must not be empty.")

    if response_mask is None:
        action_response_mask = torch.ones_like(response_ids, dtype=torch.bool)
    else:
        if response_mask.ndim != 1 or response_mask.shape != response_ids.shape:
            raise ValueError(
                "response_mask must be one-dimensional and match response_ids, "
                f"got {response_mask.shape} and {response_ids.shape}"
            )
        action_response_mask = response_mask.to(device=response_ids.device, dtype=torch.bool)

    if prompt_ids.device != response_ids.device:
        raise ValueError(
            f"prompt_ids and response_ids must be on the same device, got {prompt_ids.device} and "
            f"{response_ids.device}"
        )

    model_input_ids = torch.cat((prompt_ids, response_ids[:-1]))
    prompt_labels = torch.full(
        (prompt_ids.numel() - 1,),
        ignore_idx,
        dtype=response_ids.dtype,
        device=response_ids.device,
    )
    response_labels = response_ids.masked_fill(~action_response_mask, ignore_idx)
    shifted_labels = torch.cat((prompt_labels, response_labels))
    action_mask = torch.cat(
        (
            torch.zeros(prompt_ids.numel() - 1, dtype=torch.bool, device=response_ids.device),
            action_response_mask,
        )
    )

    expected_length = model_input_ids.numel()
    if shifted_labels.numel() != expected_length or action_mask.numel() != expected_length:
        raise AssertionError("Next-token fields must have identical lengths.")
    return NextTokenData(model_input_ids, shifted_labels, action_mask)


def _validate_1d_tokens(tokens: torch.Tensor, name: str) -> None:
    if tokens.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tokens.shape}")

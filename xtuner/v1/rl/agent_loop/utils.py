import time

import numpy as np
import ray
import torch

from xtuner.v1.data_proto import RolloutState, Status, update_seq_staleness
from xtuner.v1.utils import get_logger


logger = get_logger()


def _resolve_routed_experts(routed_experts: torch.Tensor | ray.ObjectRef | None) -> torch.Tensor | None:
    if routed_experts is None:
        return None
    if isinstance(routed_experts, ray.ObjectRef):
        routed_experts = ray.get(routed_experts)
    if isinstance(routed_experts, np.ndarray):
        return torch.from_numpy(routed_experts)
    assert isinstance(routed_experts, torch.Tensor), f"Unexpected routed_experts type: {type(routed_experts)}"
    return routed_experts


class PartialRolloutHandler:
    """Handle preprocessing and postprocessing for partial rollout
    continuation."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    def preprocess(self, rollout_state: RolloutState, enable_partial_rollout: bool = False) -> RolloutState:
        # for partial rollout
        if not enable_partial_rollout or not rollout_state.response_ids or rollout_state.status == Status.COMPLETED:
            return rollout_state

        # If status is EXPIRED, reset tokens, sample_params and responses for fresh generation
        if rollout_state.status == Status.EXPIRED:
            rollout_state.tokens = rollout_state.prompt_ids
            rollout_state.sample_params = rollout_state.sample_params.copy(update={"max_tokens": self.max_tokens})
            rollout_state.response_ids = []
            rollout_state.response = ""
            rollout_state.logprobs = []
            rollout_state.response_mask = []
            rollout_state.response_rollout_steps = []
            return rollout_state

        # Set up token and length variable
        response_ids = rollout_state.response_ids
        prompt_ids = list(rollout_state.prompt_ids or [])
        response_len = len(response_ids)
        prompt_len = len(prompt_ids)

        rollout_state.tokens = prompt_ids + response_ids  # concatenate for partial rollout continuation
        remaining_tokens = self.max_tokens - response_len  # compute remaining max_tokens budget
        rollout_state.sample_params = rollout_state.sample_params.copy(update={"max_tokens": remaining_tokens})

        logger.debug(
            f"[PartialRolloutHandler] Sample {rollout_state.uid} continue rollout | Remaining tokens allowed: {remaining_tokens} | Status: {rollout_state.status} | Prompt len: {prompt_len} | Response len: {response_len} | Staleness: {rollout_state.seq_staleness} | Total tokens: {len(rollout_state.tokens)}"
        )
        # TODO: handle routed_experts
        rollout_state.extra_fields["history_response_dict"] = {
            "response_ids": rollout_state.tokens[prompt_len:] if rollout_state.tokens else [],
            "response": rollout_state.response or "",
            "logprobs": rollout_state.logprobs or [],
            "response_mask": rollout_state.response_mask or [],
            "routed_experts": rollout_state.routed_experts,
        }
        return rollout_state

    def postprocess(self, rollout_state: RolloutState, rollout_step: int) -> RolloutState:
        # Update seq_staleness
        rollout_state = update_seq_staleness(rollout_state, rollout_step)

        # Concatenate history response fields
        history_dict = rollout_state.extra_fields.pop("history_response_dict", None)
        if not history_dict:
            return rollout_state

        rollout_state.response_ids = history_dict.get("response_ids", []) + (rollout_state.response_ids or [])
        rollout_state.response = history_dict.get("response", "") + (rollout_state.response or "")
        rollout_state.logprobs = history_dict.get("logprobs", []) + (rollout_state.logprobs or [])
        rollout_state.response_mask = history_dict.get("response_mask", []) + (rollout_state.response_mask or [])
        history_routed_experts = _resolve_routed_experts(history_dict.get("routed_experts"))
        cur_routed_experts = _resolve_routed_experts(rollout_state.routed_experts)
        if history_routed_experts is not None and cur_routed_experts is not None:
            start_time = time.time()
            cur_routed_experts_shape = cur_routed_experts.shape
            history_routed_experts_len = history_routed_experts.shape[0]
            assert history_routed_experts_len - 1 <= cur_routed_experts_shape[0], (
                f"Existing routed_experts shape: {history_routed_experts.shape}, current routed_experts shape: {cur_routed_experts_shape}"
            )
            cur_routed_experts = cur_routed_experts[history_routed_experts_len:, :, :]
            concat_routed_experts = torch.cat([history_routed_experts, cur_routed_experts], dim=0)

            prompt_ids = rollout_state.prompt_ids or []
            response_ids = rollout_state.response_ids or []
            expect_tokens_num = len(prompt_ids) + len(response_ids) - 1
            assert concat_routed_experts.shape[0] == expect_tokens_num, (
                f"After concatenation, routed_experts shape: {concat_routed_experts.shape}, expected tokens num: {expect_tokens_num}"
            )
            logger.info(
                f"[PartialRolloutHandler] Postprocess rollout {rollout_state.uid}: "
                f"concat routed_experts {concat_routed_experts.shape} "
                f"(history={history_routed_experts.shape[0]}, new={cur_routed_experts_shape[0]}), "
                f"prompt={len(prompt_ids)}, response={len(response_ids)}"
            )
            rollout_state.routed_experts = concat_routed_experts
            end_time = time.time()
            logger.info(
                f"[PartialRolloutHandler] Postprocess routed_experts concatenation time: {end_time - start_time:.4f} seconds"
            )
        elif history_routed_experts is None and cur_routed_experts is not None:
            rollout_state.routed_experts = cur_routed_experts
        elif history_routed_experts is not None and cur_routed_experts is None:
            rollout_state.routed_experts = history_routed_experts

        return rollout_state

import time

import ray

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.utils import clear_rollout_response_for_rerun
from xtuner.v1.utils import get_logger


def _resolve_routed_experts(routed_experts: list[int] | ray.ObjectRef) -> list[int]:
    if isinstance(routed_experts, ray.ObjectRef):
        routed_experts = ray.get(routed_experts)
    if hasattr(routed_experts, "tolist"):
        routed_experts = routed_experts.tolist()
    assert isinstance(routed_experts, list), f"Unexpected routed_experts type: {type(routed_experts)}"
    return routed_experts


class PartialRolloutHandler:
    """Handle preprocessing and postprocessing for partial rollout
    continuation."""

    def __init__(self, max_tokens: int) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.max_tokens = max_tokens

    def preprocess(self, rollout_state: RolloutState, enable_partial_rollout: bool = False) -> RolloutState:
        if rollout_state.status == Status.EXPIRED or (
            not enable_partial_rollout and rollout_state.status == Status.ABORTED
        ):
            rollout_state = clear_rollout_response_for_rerun(rollout_state)
            rollout_state.sample_params = rollout_state.sample_params.model_copy(
                update={"max_tokens": self.max_tokens}
            )
            rollout_state.response = ""
            rollout_state.status = Status.INIT

        if not rollout_state.response_ids or rollout_state.status == Status.COMPLETED:
            return rollout_state

        # Set up token and length variable
        response_ids = rollout_state.response_ids
        prompt_ids = list(rollout_state.prompt_ids or [])
        response_len = len(response_ids)
        prompt_len = len(prompt_ids)

        rollout_state.tokens = prompt_ids + response_ids  # concatenate for partial rollout continuation
        remaining_tokens = self.max_tokens - response_len  # compute remaining max_tokens budget
        rollout_state.sample_params = rollout_state.sample_params.copy(update={"max_tokens": remaining_tokens})

        self.logger.debug(
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

    def postprocess(self, rollout_state: RolloutState) -> RolloutState:
        # TODO: if not enable partial rollout, return directly?

        # Concatenate history response fields
        history_dict = rollout_state.extra_fields.pop("history_response_dict", None)
        if not history_dict:
            return rollout_state

        rollout_state.response_ids = history_dict.get("response_ids", []) + (rollout_state.response_ids or [])
        rollout_state.response = history_dict.get("response", "") + (rollout_state.response or "")
        rollout_state.logprobs = history_dict.get("logprobs", []) + (rollout_state.logprobs or [])
        rollout_state.response_mask = history_dict.get("response_mask", []) + (rollout_state.response_mask or [])
        history_routed_experts_ref = history_dict.get("routed_experts")
        cur_routed_experts_ref = rollout_state.routed_experts
        if history_routed_experts_ref is not None and cur_routed_experts_ref is not None:
            start_time = time.time()
            history_routed_experts = _resolve_routed_experts(history_routed_experts_ref)
            cur_routed_experts = _resolve_routed_experts(cur_routed_experts_ref)
            cur_routed_experts_len = len(cur_routed_experts)
            history_routed_experts_len = len(history_routed_experts)
            assert history_routed_experts_len - 1 <= cur_routed_experts_len, (
                f"Existing routed_experts len: {history_routed_experts_len}, current routed_experts len: {cur_routed_experts_len}"
            )
            cur_routed_experts = cur_routed_experts[history_routed_experts_len:]
            concat_routed_experts = history_routed_experts + cur_routed_experts
            rollout_state.routed_experts = ray.put(concat_routed_experts)
            # free_object_refs(
            #     [ref for ref in (history_routed_experts_ref, cur_routed_experts_ref) if isinstance(ref, ray.ObjectRef)]
            # )
            end_time = time.time()
            self.logger.info(
                f"[PartialRolloutHandler] Postprocess routed_experts concatenation time: {end_time - start_time:.4f} seconds"
            )
        elif history_routed_experts_ref is None and cur_routed_experts_ref is not None:
            rollout_state.routed_experts = cur_routed_experts_ref
        elif history_routed_experts_ref is not None and cur_routed_experts_ref is None:
            rollout_state.routed_experts = history_routed_experts_ref

        return rollout_state

import os
import time
from typing import cast

import numpy as np
import ray
from ray import ObjectRef as RayObjectRef

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.utils import free_object_refs
from xtuner.v1.utils import get_logger


ROLLOUT_RAY_GET_TIMEOUT = int(os.getenv("XTUNER_ROLLOUT_RAY_GET_TIMEOUT", str(5 * 3600)))  # default 5 hours
logger = get_logger()


async def _resolve_routed_experts(routed_experts: np.ndarray | RayObjectRef) -> np.ndarray:
    if isinstance(routed_experts, RayObjectRef):
        routed_experts = await routed_experts
    assert routed_experts is not None, "routed_experts should not be empty after resolution"
    return np.asarray(routed_experts)


class PartialRolloutHandler:
    """Handle worker-level partial rollout continuation for one inference
    request.

    This handler only knows how to continue a single interrupted generation by reusing the previous response as the
    next engine input. Agent-loop level multi-turn rollout, including tool messages and response masks, must be handled
    by the agent loop itself.
    """

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def preprocess(self, rollout_state: RolloutState, max_tokens: int) -> RolloutState:
        # Set up token and length variable
        response_ids = list(rollout_state.response_ids or [])
        prompt_ids = list(rollout_state.prompt_ids or [])
        response_len = len(response_ids)
        prompt_len = len(prompt_ids)

        rollout_state.tokens = prompt_ids + response_ids  # concatenate for partial rollout continuation
        remaining_tokens = max_tokens - response_len  # compute remaining max_tokens budget
        rollout_state.sample_params = rollout_state.sample_params.copy(update={"max_tokens": remaining_tokens})

        self.logger.debug(
            f"[PartialRolloutHandler] Sample {rollout_state.uid} continue rollout | Remaining tokens allowed: {remaining_tokens} | Status: {rollout_state.status} | Prompt len: {prompt_len} | Response len: {response_len} | Staleness: {rollout_state.seq_staleness} | Total tokens: {len(rollout_state.tokens)}"
        )
        return rollout_state

    async def postprocess(
        self,
        rollout_state: RolloutState,
        *,
        response: str,
        response_ids: list[int],
        logprobs: list[float],
        routed_experts: np.ndarray | RayObjectRef | None,
        finish_reason: str,
        status: Status,
        routed_experts_expect_len: int | None = None,
        enable_partial_rollout: bool | None = None,
    ) -> RolloutState:
        rollout_state.finish_reason = finish_reason
        rollout_state.status = status
        history_response = rollout_state.response or ""
        history_response_ids = list(rollout_state.response_ids or [])
        current_response_ids = list(response_ids or [])
        history_logprobs = list(rollout_state.logprobs or [])
        current_logprobs = list(logprobs or [])

        rollout_state.response = history_response + response
        rollout_state.response_ids = history_response_ids + current_response_ids
        rollout_state.logprobs = history_logprobs + current_logprobs

        history_routed_experts = rollout_state.routed_experts
        if history_routed_experts is not None and routed_experts is not None:
            # case 1: 上一次 rolloutstate 有 response, 本次推理也有 response，需要对 routed experts 进行拼接
            start_time = time.perf_counter()
            history_routed_experts_ref = history_routed_experts
            cur_routed_experts_ref = routed_experts
            history_routed_experts = await _resolve_routed_experts(history_routed_experts_ref)  # type: ignore[assignment]
            cur_routed_experts = await _resolve_routed_experts(routed_experts)  # type: ignore[assignment]
            history_routed_experts_len = len(history_routed_experts)
            cur_routed_experts_len = len(cur_routed_experts)
            assert history_routed_experts_len - 1 <= cur_routed_experts_len, (
                f"Existing routed_experts len: {history_routed_experts_len}, current routed_experts len: {cur_routed_experts_len}, history_response_ids len: {len(history_response_ids)}, current response_ids len: {len(response_ids)}"
            )
            cur_routed_experts = cur_routed_experts[history_routed_experts_len:]
            if isinstance(history_routed_experts, list) and isinstance(cur_routed_experts, list):
                concat_routed_experts = history_routed_experts + cur_routed_experts
            else:
                concat_routed_experts = np.concatenate([history_routed_experts, cur_routed_experts], axis=0)
            rollout_state.routed_experts = ray.put(concat_routed_experts)
            if routed_experts_expect_len is not None:
                expected_len = len(cast(list[int], rollout_state.prompt_ids)) + len(rollout_state.response_ids) - 1
                assert expected_len == routed_experts_expect_len, (
                    f"Expected routed_experts len: {expected_len}, routed_experts_expect_len: {routed_experts_expect_len}, prompt_ids_len: {len(cast(list[int], rollout_state.prompt_ids))}, response_ids_len: {len(rollout_state.response_ids)}"
                )
                assert len(concat_routed_experts) == routed_experts_expect_len, (
                    f"After concatenation, routed_experts len: {len(concat_routed_experts)}, expected len: {expected_len}, history_routed_experts_len: {history_routed_experts_len}, current_routed_experts_len: {len(cur_routed_experts)}, prompt_ids_len: {len(cast(list[int], rollout_state.prompt_ids))}, response_ids_len: {len(rollout_state.response_ids)}"
                )
            free_object_refs(
                [
                    ref
                    for ref in (history_routed_experts_ref, cur_routed_experts_ref)
                    if isinstance(ref, RayObjectRef)
                ]
            )
            end_time = time.perf_counter()
            self.logger.debug(
                f"[PartialRolloutHandler] Postprocess routed_experts concatenation time: {end_time - start_time:.4f} seconds"
            )
        elif history_routed_experts is None and routed_experts is not None:
            # case 2: 上一次 rolloutstate 没有 response, 需要历史的 routed_experts， response_ids, logprobs 为空，直接赋值本次的 routed_experts 即可
            assert not history_response_ids and not history_logprobs, (
                "Got None historical routed_experts, but historical response_ids or logprobs exist: "
                f"history_response_ids_len={len(history_response_ids)}, "
                f"history_logprobs_len={len(history_logprobs)}, "
            )
            rollout_state.routed_experts = routed_experts
        elif history_routed_experts is not None and routed_experts is None:
            # case3: 本次推理为超发的任务, token 还未生成时就被 abort了，所以本次 routed_experts 为空，并且response_ids, logprobs 需要也为空
            assert not current_response_ids and not current_logprobs, (
                "Got None current routed_experts, but new response_ids or logprobs exist: "
                f"current_response_ids_len={len(current_response_ids)}, "
                f"current_logprobs_len={len(current_logprobs)}, "
            )
            rollout_state.routed_experts = history_routed_experts
        return rollout_state

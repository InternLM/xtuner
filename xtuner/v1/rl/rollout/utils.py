import asyncio
import threading
import time
from collections import OrderedDict
from enum import Enum
from itertools import cycle
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import ray
from ray import ObjectRef as RayObjectRef

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.utils import free_object_refs
from xtuner.v1.utils import get_logger


if TYPE_CHECKING:
    from .controller import WorkerInfo

logger = get_logger()

__all__ = [
    "PartialRolloutHandler",
    "SessionRouter",
    "WorkerLifecycleState",
]


class WorkerLifecycleState(str, Enum):
    # Can serve rollout generation and control requests.
    ACTIVE = "active"
    # Not serving rollout requests; the rollout server may still hold resources.
    INACTIVE = "inactive"
    # Temporarily owned by recovery shutdown/init/check_health.
    RECOVERING = "recovering"


class SessionRouter:
    def __init__(
        self,
        worker_infos: dict[int, "WorkerInfo"],
        worker_infos_lock: Optional[threading.RLock] = None,
        max_sessions: int = 10000,
        max_idle_seconds: Optional[float] = 3600.0,
    ):
        self._worker_infos = worker_infos
        self._worker_infos_lock = worker_infos_lock
        self._max_sessions = max_sessions
        self._max_idle = max_idle_seconds

        # OrderedDict: key=session_id -> value=(worker_rank, last_used_ts)
        self._map: OrderedDict[int, tuple[int, float]] = OrderedDict()

        self._worker_cycler = cycle(worker_infos.keys())
        self._lock = asyncio.Lock()
        self.logger = get_logger()

    def _now(self) -> float:
        return time.time()

    def _evict_expired(self):
        if self._max_idle is None:
            return
        now = self._now()

        to_delete = []
        for sid, (_, last_used) in self._map.items():
            if now - last_used > self._max_idle:
                to_delete.append(sid)
            else:
                break
        for sid in to_delete:
            self._map.pop(sid, None)

    def _evict_lru_to_capacity(self):
        while len(self._map) > self._max_sessions:
            self._map.popitem(last=False)

    def _choose_next_active_worker(self) -> tuple[int, Any]:
        n = len(self._worker_infos)
        for _ in range(n):
            rank = next(self._worker_cycler)
            if self._worker_infos_lock is None:
                info = self._worker_infos[rank]
                if info and info.is_active() and info.is_request_entrypoint:
                    return rank, info.actor
            else:
                with self._worker_infos_lock:
                    info = self._worker_infos[rank]
                    if info and info.is_active() and info.is_request_entrypoint:
                        return rank, info.actor
        return -1, None

    async def get_worker(self, session_id: int) -> Optional[Any]:
        async with self._lock:
            self._evict_expired()

            if session_id in self._map:
                worker_rank, _ = self._map.pop(session_id)
                if self._worker_infos_lock is None:
                    info = self._worker_infos.get(worker_rank)
                else:
                    with self._worker_infos_lock:
                        info = self._worker_infos.get(worker_rank)
                if info and info.is_active() and info.is_request_entrypoint:
                    self._map[session_id] = (worker_rank, self._now())
                    return info.actor

            rank, worker = self._choose_next_active_worker()
            if rank == -1:
                return None
            self._map[session_id] = (rank, self._now())
            self._evict_lru_to_capacity()
            return worker


async def _resolve_routed_experts(routed_experts: np.ndarray | RayObjectRef) -> np.ndarray:
    if isinstance(routed_experts, RayObjectRef):
        routed_experts_value = await routed_experts
        free_object_refs([routed_experts])
    else:
        routed_experts_value = routed_experts
    assert routed_experts_value is not None, "routed_experts should not be empty after resolution"
    return np.asarray(routed_experts_value)


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
            f"[PartialRolloutHandler] Sample {rollout_state.rollout_id} continue rollout | Remaining tokens allowed: {remaining_tokens} | Status: {rollout_state.status} | Prompt len: {prompt_len} | Response len: {response_len} | Staleness: {rollout_state.seq_staleness} | Total tokens: {len(rollout_state.tokens)}"
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
        prompt_tokens: int,
        completion_tokens: int,
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
            routed_experts_expect_len = prompt_tokens + completion_tokens - 1
            history_routed_experts_expect_len = prompt_tokens - 1

            # case 1: 上一次 rolloutstate 有 response, 本次推理也有 response，需要对 routed experts 进行拼接
            start_time = time.perf_counter()
            history_routed_experts = await _resolve_routed_experts(history_routed_experts)  # type: ignore[assignment]
            cur_routed_experts = await _resolve_routed_experts(routed_experts)  # type: ignore[assignment]
            history_routed_experts_len = len(history_routed_experts)
            cur_routed_experts_len = len(cur_routed_experts)
            assert history_routed_experts_len == history_routed_experts_expect_len, (
                f"History routed_experts len mismatch before partial rollout concatenation: "
                f"history_routed_experts_len={history_routed_experts_len}, "
                f"expected_len={history_routed_experts_expect_len}, "
                f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
                f"history_response_ids_len={len(history_response_ids)}, "
                f"current_response_ids_len={len(response_ids)}"
            )
            assert cur_routed_experts_len == routed_experts_expect_len, (
                f"Current routed_experts len mismatch before partial rollout concatenation: "
                f"cur_routed_experts_len={cur_routed_experts_len}, expected_len={routed_experts_expect_len}, "
                f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
                f"history_routed_experts_len={history_routed_experts_len}, "
                f"current_response_ids_len={len(response_ids)}"
            )
            assert history_routed_experts_len - 1 <= cur_routed_experts_len, (
                f"Current routed_experts len is shorter than history during partial rollout concatenation: "
                f"history_routed_experts_len={history_routed_experts_len}, "
                f"cur_routed_experts_len={cur_routed_experts_len}, "
                f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
                f"history_response_ids_len={len(history_response_ids)}, "
                f"current_response_ids_len={len(response_ids)}"
            )
            cur_routed_experts = cur_routed_experts[history_routed_experts_len:]
            concat_routed_experts = np.concatenate([history_routed_experts, cur_routed_experts], axis=0)
            assert len(concat_routed_experts) == routed_experts_expect_len, (
                f"Concatenated routed_experts len mismatch after partial rollout concatenation: "
                f"concat_routed_experts_len={len(concat_routed_experts)}, "
                f"expected_len={routed_experts_expect_len}, "
                f"history_routed_experts_len={history_routed_experts_len}, "
                f"current_routed_experts_len={len(cur_routed_experts)}, "
                f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"
            )
            rollout_state.routed_experts = ray.put(concat_routed_experts)
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

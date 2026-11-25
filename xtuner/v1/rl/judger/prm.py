from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Awaitable, Callable

import httpx
from pydantic import Field

from xtuner.v1.data_proto.rl_data import RolloutState

from .native import Judger, JudgerConfig, JudgerOutput, JudgerOutputBatch, JudgerPayloadBatch


PRMTransport = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

# 步分是 value head 对 0/1 标签的 MSE 回归，未过 sigmoid，实测会轻微越界；
# 容差内钳制，超出容差视为服务异常 fail-closed。
_SCORE_RANGE_TOLERANCE = 0.01


def _validate_step_scores(body: dict[str, Any]) -> list[float]:
    raw_scores = body.get("step_scores")
    if not isinstance(raw_scores, list) or not raw_scores:
        raise ValueError("PRM server response must contain a non-empty step_scores list.")
    scores: list[float] = []
    for index, raw in enumerate(raw_scores):
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
            raise ValueError(f"PRM step score {index} must be a finite number, got {raw!r}.")
        score = float(raw)
        if not -_SCORE_RANGE_TOLERANCE <= score <= 1.0 + _SCORE_RANGE_TOLERANCE:
            raise ValueError(f"PRM step score {index}={score} is outside [0, 1].")
        scores.append(min(1.0, max(0.0, score)))
    n_steps = body.get("n_steps")
    if n_steps is not None and n_steps != len(scores):
        raise ValueError(f"PRM server n_steps {n_steps!r} does not match {len(scores)} step scores.")
    return scores


def _judge_error_type(error: Exception) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "transport"
    if isinstance(error, httpx.HTTPStatusError):
        return "http"
    if isinstance(error, httpx.HTTPError):
        return "transport"
    return "protocol"


class PRMJudger(Judger):
    """Score rollout responses with a value-head PRM served over HTTP.

    The PRM server (see ``prm_server.py``) scores every step of the response
    and returns ``step_scores`` in ``[0, 1]``. This judger aggregates them by
    mean and returns the raw ``[0, 1]`` score as ``score``/``process_score``;
    mapping to ``[-1, 1]`` and fusion with the outcome branch stay in
    ``merge_outcome_and_process``, same as the rubric process branch.

    Args:
        endpoint (str): URL of the PRM server ``/score`` route.
        judger_name (str): Logical judger name used in logs and reward output.
            Defaults to ``"prm"``.
        step_separator (str): Separator between reasoning steps in the
            response. Defaults to ``"\\n\\n"``.
        request_timeout (float): Timeout in seconds for one scoring request.
            Defaults to 60.0.
        max_retries (int): Number of retries after the first failed attempt.
            Defaults to 2.
        retry_backoff (float): Base seconds of exponential backoff between
            retries. Defaults to 1.0.
        transport (PRMTransport | None): Injected request executor used by
            tests; defaults to HTTP POST against ``endpoint``.
    """

    def __init__(
        self,
        endpoint: str,
        judger_name: str = "prm",
        step_separator: str = "\n\n",
        request_timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        transport: PRMTransport | None = None,
    ) -> None:
        super().__init__(judger_name=judger_name)
        if not endpoint:
            raise ValueError("PRMJudger requires an endpoint.")
        if not step_separator:
            raise ValueError("PRMJudger requires a non-empty step_separator.")
        if request_timeout <= 0 or retry_backoff < 0 or max_retries < 0:
            raise ValueError("Invalid PRMJudger timeout or retry configuration.")
        self.endpoint = endpoint
        self.step_separator = step_separator
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._transport = transport

    def preprocess(self, rollout_state: RolloutState) -> dict[str, Any]:
        return {
            "text": rollout_state.response,
            "rollout_id": rollout_state.rollout_id,
            "group_id": rollout_state.group_id,
            "task_name": rollout_state.task_name,
        }

    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutputBatch:
        if isinstance(payload, list):
            return list(await asyncio.gather(*(self._score(item) for item in payload)))
        return await self._score(payload)

    async def _score(self, payload: dict[str, Any]) -> JudgerOutput:
        text = payload.get("text")
        if text is None:
            raise ValueError("PRM judging requires a detokenized response.")
        request = {"text": text, "step_separator": self.step_separator}
        started_at = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                body = await self._request(request)
                step_scores = _validate_step_scores(body)
                score = sum(step_scores) / len(step_scores)
                return {
                    "score": score,
                    "process_score": score,
                    "step_scores": step_scores,
                    "n_steps": len(step_scores),
                    "judge_status": "ok",
                    "judge_attempts": attempt + 1,
                    "judge_latency_seconds": time.monotonic() - started_at,
                    "rollout_id": payload.get("rollout_id"),
                }
            except (httpx.HTTPError, ValueError, TypeError, KeyError) as error:
                last_error = error
                if attempt < self.max_retries and self.retry_backoff:
                    await asyncio.sleep(self.retry_backoff * 2**attempt)
        diagnostics = {
            "rollout_id": payload.get("rollout_id"),
            "group_id": payload.get("group_id"),
            "task_name": payload.get("task_name"),
            "error_type": _judge_error_type(last_error) if last_error is not None else "protocol",
            "judge_latency_seconds": time.monotonic() - started_at,
        }
        raise RuntimeError(
            f"PRM judge failed after {self.max_retries + 1} attempts; "
            f"diagnostics={json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)}; error={last_error}"
        ) from last_error

    async def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return await self._transport(request)
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.post(self.endpoint, json=request)
            response.raise_for_status()
            body = response.json()
        if not isinstance(body, dict):
            raise ValueError("PRM server response must be a JSON object.")
        return body


class PRMJudgerConfig(JudgerConfig):
    """Build a PRM judger while retaining the existing CPU replica options."""

    endpoint: str
    step_separator: str = "\n\n"
    max_retries: int = Field(default=2, ge=0)
    retry_backoff: float = Field(default=1.0, ge=0)
    request_timeout: float = Field(default=60.0, gt=0)

    def build_local(self) -> PRMJudger:
        return PRMJudger(
            endpoint=self.endpoint,
            judger_name=self.judger_name,
            step_separator=self.step_separator,
            request_timeout=self.request_timeout,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
        )

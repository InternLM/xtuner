from __future__ import annotations

import asyncio
import math
import time
from typing import Literal, cast

import httpx
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.advantage import AdvantageEstimator


class OPDTeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    endpoint: str
    api_key: str | None = None
    request_timeout_s: float = Field(default=1200.0, gt=0.0)
    max_retry_per_sample: int = Field(default=2, ge=0)
    max_concurrency: int = Field(default=128, gt=0)


class OPDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["pg-opd"] = "pg-opd"
    task_adv_weight: float = Field(default=0.0, ge=0.0)
    opd_adv_weight: float = Field(default=1.0, ge=0.0)
    teachers: list[OPDTeacherConfig] = Field(min_length=1)
    data_source_teacher_map: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_teacher_names(self) -> OPDConfig:
        teacher_names = [teacher.name for teacher in self.teachers]
        if len(teacher_names) != len(set(teacher_names)):
            raise ValueError("OPD teacher names must be unique")
        unknown_teachers = set(self.data_source_teacher_map.values()) - set(teacher_names)
        if unknown_teachers:
            raise ValueError(f"data_source_teacher_map references unknown teachers: {sorted(unknown_teachers)}")
        return self


class TeacherLogprobClient:
    """Minimal asynchronous client for one external SGLang teacher."""

    def __init__(self, config: OPDTeacherConfig) -> None:
        self.config = config
        self.name = config.name
        self.url = f"{config.endpoint.rstrip('/')}/generate"
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

        headers = {"Content-Type": "application/json"}
        if config.api_key is not None:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=config.request_timeout_s)

    async def compute_logprobs(self, state: RolloutState) -> RolloutState:
        start = time.perf_counter()
        try:
            prompt_ids = cast(list[int], state.prompt_ids)
            response_ids = cast(list[int], state.response_ids)
            payload = {
                "input_ids": prompt_ids + response_ids,
                "sampling_params": {
                    "max_new_tokens": 0,
                    "temperature": 1.0,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": max(len(prompt_ids) - 1, 0),
                "top_logprobs_num": 0,
                "stream": False,
            }

            retries = 0
            while True:
                try:
                    async with self._semaphore:
                        response = await self._client.post(self.url, json=payload)
                        response.raise_for_status()
                    teacher_tokens, teacher_logprobs = self._parse_response(response, response_ids)
                    state.teacher_tokens = teacher_tokens
                    state.teacher_logprobs = teacher_logprobs
                    return state
                except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as exc:
                    if retries >= self.config.max_retry_per_sample:
                        state.status = Status.FAILED
                        state.error_msg = f"Teacher {self.name!r} scoring failed after {retries + 1} attempts: {exc}"
                        return state
                    retries += 1
                    await asyncio.sleep(0.1)
        finally:
            state.extra_fields["teacher_score_time_s"] = time.perf_counter() - start

    def _parse_response(
        self,
        response: httpx.Response,
        response_ids: list[int],
    ) -> tuple[list[int], list[float]]:
        try:
            raw_logprobs = response.json()["meta_info"]["input_token_logprobs"]
            teacher_tokens = [item[1] for item in raw_logprobs[1:]]
            teacher_logprobs = [float(item[0]) for item in raw_logprobs[1:]]
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("Invalid teacher response") from exc

        if len(raw_logprobs) != len(response_ids) + 1:
            raise ValueError("Teacher logprob length mismatch")
        if teacher_tokens != response_ids:
            raise ValueError("Teacher token ids mismatch")
        if not all(math.isfinite(logprob) for logprob in teacher_logprobs):
            raise ValueError("Teacher logprobs contain NaN or Inf")
        return teacher_tokens, teacher_logprobs

    async def close(self) -> None:
        await self._client.aclose()


def route_teacher_client(
    state: RolloutState,
    *,
    data_source_teacher_map: dict[str, str],
    teacher_clients: dict[str, TeacherLogprobClient],
) -> TeacherLogprobClient:
    data_source = state.extra_fields["origin_data_source"]
    teacher_name = data_source_teacher_map[data_source]
    return teacher_clients[teacher_name]


def compute_pg_opd_token_advantages(
    group: list[RolloutState],
    *,
    config: OPDConfig,
    task_adv_estimator: AdvantageEstimator | None,
) -> list[torch.Tensor]:
    opd_advantages: list[torch.Tensor] = []
    response_masks: list[torch.Tensor] = []
    for state in group:
        response_ids = cast(list[int], state.response_ids)
        behavior_logprobs = cast(list[float], state.logprobs)
        teacher_logprobs = cast(list[float], state.teacher_logprobs)

        behavior_logprobs_t = torch.tensor(behavior_logprobs, dtype=torch.float32)
        teacher_logprobs_t = torch.tensor(teacher_logprobs, dtype=torch.float32)
        response_mask = state.response_mask
        if not response_mask:
            response_mask_t = torch.ones(len(response_ids), dtype=torch.float32)
        else:
            response_mask_t = torch.tensor(response_mask, dtype=torch.float32)
        opd_advantages.append(teacher_logprobs_t - behavior_logprobs_t)
        response_masks.append(response_mask_t)

    task_advantages = [0.0] * len(group)
    if config.task_adv_weight > 0:
        rewards: list[float] = []
        for state in group:
            if state.reward is None or "score" not in state.reward:
                raise ValueError(f"Reward score is required for mixed PG-OPD rollout {state.rollout_id}")
            rewards.append(float(state.reward["score"]))

        task_advantages = (
            cast(AdvantageEstimator, task_adv_estimator)
            .compute(
                torch.tensor(rewards, dtype=torch.float32),
                group,
            )
            .tolist()
        )

    return [
        (config.task_adv_weight * task_advantage + config.opd_adv_weight * opd_advantage) * response_mask
        for task_advantage, opd_advantage, response_mask in zip(
            task_advantages,
            opd_advantages,
            response_masks,
            strict=True,
        )
    ]

from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.loss.base_loss import BaseRLLossContext


class OPDTeacherLaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str | Path
    num_workers: int = Field(default=1, gt=0)
    server_port: int = Field(gt=0, le=65535)
    dtype: Literal["auto", "float16", "bfloat16"] = "bfloat16"
    tensor_parallel_size: int = Field(default=1, gt=0)
    expert_parallel_size: int = Field(default=1, gt=0)
    context_length: int | None = Field(default=None, gt=0)
    max_batch_size: int | None = Field(default=None, gt=0)
    chunked_prefill_size: int | None = Field(default=4096, gt=0)
    max_prefill_token_num: int | None = Field(default=4096, gt=0)
    gpu_memory_utilization: float = Field(default=0.6, gt=0.0, le=1.0)


class OPDTeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    endpoint: str | None = None
    api_key: str | None = None
    request_timeout_s: float = Field(default=1200.0, gt=0.0)
    max_retry_per_sample: int = Field(default=2, ge=0)
    max_concurrency: int = Field(default=128, gt=0)
    launch_config: OPDTeacherLaunchConfig | None = None


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

    def resolve_teacher_endpoints(self, endpoint_map: dict[str, str]) -> OPDConfig:
        teachers = []
        for teacher in self.teachers:
            if teacher.launch_config is None:
                endpoint = teacher.endpoint
            else:
                endpoint = endpoint_map[teacher.name]
            if not endpoint:
                raise ValueError(f"Teacher {teacher.name!r} needs endpoint")
            teachers.append(teacher.model_copy(update={"endpoint": endpoint}))
        return self.model_copy(update={"teachers": teachers})


def validate_opd_sample_params(sample_params: SampleParams) -> None:
    identity_sampling_params: dict[str, Any] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "min_tokens": 0,
    }
    non_identity_params = {
        name: getattr(sample_params, name)
        for name, expected in identity_sampling_params.items()
        if getattr(sample_params, name) != expected
    }
    if non_identity_params:
        raise ValueError(f"PG-OPD requires identity student sampling, got {non_identity_params}")
    if not sample_params.return_logprob or not sample_params.return_token_ids:
        raise ValueError("PG-OPD requires return_logprob=True and return_token_ids=True")


class TeacherLogprobClient:
    """Asynchronous teacher client scoped to one AgentLoop."""

    def __init__(self, config: OPDTeacherConfig) -> None:
        self.config = config
        self.name = config.name
        self.backend = self._resolve_backend_from_env()
        if not config.endpoint:
            raise ValueError(f"Teacher {config.name!r} needs endpoint")
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
            if not prompt_ids or not response_ids:
                state.status = Status.FAILED
                state.error_msg = f"Teacher {self.name!r} scoring requires non-empty prompt_ids and response_ids"
                return state
            payload = self._construct_payload(prompt_ids, response_ids)

            retries = 0
            while True:
                try:
                    async with self._semaphore:
                        response = await self._client.post(self.url, json=payload)
                        response.raise_for_status()
                    teacher_tokens, teacher_logprobs = self._parse_response(response, prompt_ids, response_ids)
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

    @staticmethod
    def _resolve_backend_from_env() -> Literal["sglang", "lmdeploy"]:
        use_sglang = os.environ.get("XTUNER_USE_SGLANG", "0") == "1"
        use_lmdeploy = os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "1"
        use_vllm = os.environ.get("XTUNER_USE_VLLM", "0") == "1"

        if use_vllm:
            raise RuntimeError("TeacherLogprobClient supports only SGLang or LMDeploy, not vLLM")
        if use_sglang == use_lmdeploy:
            raise RuntimeError("Exactly one of XTUNER_USE_SGLANG and XTUNER_USE_LMDEPLOY must be set to 1")
        return "sglang" if use_sglang else "lmdeploy"

    def _construct_payload(self, prompt_ids: list[int], response_ids: list[int]) -> dict[str, Any]:
        if self.backend == "sglang":
            return self._construct_sglang_payload(prompt_ids, response_ids)
        if self.backend == "lmdeploy":
            return self._construct_lmdeploy_payload(prompt_ids, response_ids)
        raise RuntimeError(f"Unsupported teacher backend: {self.backend}")

    @staticmethod
    def _construct_sglang_payload(prompt_ids: list[int], response_ids: list[int]) -> dict[str, Any]:
        return {
            "input_ids": prompt_ids + response_ids,
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 0,
            "stream": False,
        }

    @staticmethod
    def _construct_lmdeploy_payload(prompt_ids: list[int], response_ids: list[int]) -> dict[str, Any]:
        return {
            "input_ids": prompt_ids + response_ids,
            "return_logprob": True,
            "logprob_start_len": 0,
            "max_tokens": 0,
            "stream": False,
        }

    def _parse_response(
        self,
        response: httpx.Response,
        prompt_ids: list[int],
        response_ids: list[int],
    ) -> tuple[list[int], list[float]]:
        if self.backend == "sglang":
            response_logprobs = self._parse_sglang_response(response, prompt_ids, response_ids)
        else:
            response_logprobs = self._parse_lmdeploy_response(response, prompt_ids, response_ids)
        return self._validate_response_logprobs(response_logprobs, response_ids)

    @staticmethod
    def _parse_sglang_response(
        response: httpx.Response,
        prompt_ids: list[int],
        response_ids: list[int],
    ) -> list[Any]:
        raw_logprobs = TeacherLogprobClient._get_input_token_logprobs(response)
        expected_length = len(prompt_ids) + len(response_ids)
        if len(raw_logprobs) != expected_length:
            raise ValueError(
                "SGLang teacher logprob length mismatch: "
                f"expected {expected_length} rows for the full input, got {len(raw_logprobs)}"
            )
        return raw_logprobs[-len(response_ids) :]

    @staticmethod
    def _parse_lmdeploy_response(
        response: httpx.Response,
        prompt_ids: list[int],
        response_ids: list[int],
    ) -> list[Any]:
        raw_logprobs = TeacherLogprobClient._get_input_token_logprobs(response)
        expected_length = len(prompt_ids) + len(response_ids) - 1
        if len(raw_logprobs) != expected_length:
            raise ValueError(
                "LMDeploy teacher logprob length mismatch: "
                f"expected {expected_length} rows after the boundary token, got {len(raw_logprobs)}"
            )
        return raw_logprobs[-len(response_ids) :]

    @staticmethod
    def _get_input_token_logprobs(response: httpx.Response) -> list[Any]:
        try:
            raw_logprobs = response.json()["meta_info"]["input_token_logprobs"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid teacher response") from exc
        if not isinstance(raw_logprobs, list):
            raise ValueError("Invalid teacher response")
        return raw_logprobs

    @staticmethod
    def _validate_response_logprobs(
        response_logprobs: list[Any],
        response_ids: list[int],
    ) -> tuple[list[int], list[float]]:
        try:
            teacher_tokens = [item[1] for item in response_logprobs]
            teacher_logprobs = [float(item[0]) for item in response_logprobs]
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError("Invalid teacher response") from exc

        if len(teacher_logprobs) != len(response_ids):
            raise ValueError("Teacher logprob length mismatch")
        if teacher_tokens != response_ids:
            raise ValueError("Teacher token ids mismatch")
        if not all(math.isfinite(logprob) for logprob in teacher_logprobs):
            raise ValueError("Teacher logprobs contain NaN or Inf")
        return teacher_tokens, teacher_logprobs


def route_teacher_client(
    state: RolloutState,
    *,
    data_source_teacher_map: dict[str, str],
    teacher_clients: dict[str, TeacherLogprobClient],
) -> TeacherLogprobClient:
    data_source = state.extra_fields["origin_data_source"]
    teacher_name = data_source_teacher_map[data_source]
    return teacher_clients[teacher_name]


def apply_opd_kl_to_advantages(
    loss_ctx: BaseRLLossContext,
    *,
    config: OPDConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the OPD reverse-KL penalty and return signed and absolute valid-
    token sums."""
    loss_kwargs = loss_ctx.loss_kwargs
    old_logprobs = cast(torch.Tensor, loss_kwargs.old_logprobs)
    teacher_logprobs = cast(torch.Tensor, loss_kwargs.teacher_logprobs)
    response_mask = loss_kwargs.shifted_labels != loss_ctx.loss_cfg.ignore_idx
    reverse_kl = old_logprobs - teacher_logprobs
    reverse_kl_sum = (reverse_kl * response_mask).sum().detach()
    abs_logprob_loss_sum = (reverse_kl.abs() * response_mask).sum().detach()
    loss_kwargs.advantages = torch.where(
        response_mask,
        loss_kwargs.advantages - config.opd_adv_weight * reverse_kl,
        loss_kwargs.advantages,
    )
    loss_kwargs.teacher_logprobs = None
    return reverse_kl_sum, abs_logprob_loss_sum

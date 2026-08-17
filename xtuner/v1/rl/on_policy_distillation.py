from __future__ import annotations

import asyncio
import math
import os
import time
from hashlib import blake2b
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.loss.base_loss import BaseRLLossContext
from xtuner.v1.utils import get_logger


logger = get_logger()


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
    log_level: Literal["critical", "error", "warning", "info", "debug"] | None = None
    chunked_prefill_size: int | None = Field(default=4096, gt=0)
    max_prefill_token_num: int | None = Field(default=4096, gt=0)
    gpu_memory_utilization: float = Field(default=0.6, gt=0.0, le=1.0)


class OPDTeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    num_replicas: int = Field(default=1, gt=0)
    endpoints: list[str] = Field(default_factory=list)
    api_key: str | None = None
    request_timeout_s: float = Field(default=1200.0, gt=0.0)
    max_retry_per_sample: int = Field(default=2, ge=0)
    max_concurrency: int = Field(default=128, gt=0)
    enable_prefix_caching: bool = False
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

    def resolve_teacher_endpoints(
        self,
        endpoint_map: dict[str, list[str]],
    ) -> OPDConfig:
        teachers = [
            teacher
            if teacher.launch_config is None
            else teacher.model_copy(update={"endpoints": endpoint_map[teacher.name]})
            for teacher in self.teachers
        ]
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


class TeacherReplicaRouter:
    """Resolve a physical teacher replica for one scoring request."""

    def __init__(self, num_replicas: int) -> None:
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        self._num_replicas = num_replicas

    def resolve_replica_idx(
        self,
        *,
        teacher_name: str,
        data_source: str,
        group_id: int | None,
    ) -> int:
        if self._num_replicas == 1:
            return 0
        routing_key = f"{teacher_name}\0{data_source}\0{group_id}".encode()
        digest = blake2b(routing_key, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big") % self._num_replicas


class TeacherLogprobClient:
    """Asynchronous teacher client scoped to one AgentLoop."""

    def __init__(self, config: OPDTeacherConfig) -> None:
        self.config = config
        self.name = config.name
        self.backend = self._resolve_backend_from_env()
        self.urls = [f"{endpoint.rstrip('/')}/generate" for endpoint in config.endpoints]
        self._semaphores = [asyncio.Semaphore(config.max_concurrency) for _ in self.urls]
        self._replica_router = TeacherReplicaRouter(len(self.urls))

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
            try:
                routed_replica_idx = self._replica_router.resolve_replica_idx(
                    teacher_name=self.name,
                    data_source=str(state.extra_fields.get("origin_data_source", "")),
                    group_id=state.group_id,
                )
            except ValueError as exc:
                state.status = Status.FAILED
                state.error_msg = f"Teacher {self.name!r} replica routing failed: {exc}"
                logger.warning(state.error_msg)
                return state
            image_data = state.extra_fields.get("image_data")
            expanded_prompt_len = None
            # Recompute the final prompt token so the first response token's
            # logprob remains available while the earlier prompt can be reused.
            logprob_start_len = 0
            if self.config.enable_prefix_caching:
                if not image_data:
                    logprob_start_len = len(prompt_ids) - 1
                else:
                    expanded_prompt_len = len(state.extra_fields["train_prompt_ids"])
                    logprob_start_len = expanded_prompt_len - 1
            payload = self._construct_payload(
                prompt_ids,
                response_ids,
                logprob_start_len=logprob_start_len,
                image_data=image_data,
            )

            for attempt_idx in range(self.config.max_retry_per_sample + 1):
                replica_idx = (routed_replica_idx + attempt_idx) % len(self.urls)
                url = self.urls[replica_idx]
                try:
                    async with self._semaphores[replica_idx]:
                        response = await self._client.post(url, json=payload)
                        response.raise_for_status()
                    teacher_tokens, teacher_logprobs = self._parse_response(
                        response,
                        response_ids,
                        logprob_start_len=logprob_start_len,
                        expanded_prompt_len=expanded_prompt_len,
                    )
                    state.teacher_tokens = teacher_tokens
                    state.teacher_logprobs = teacher_logprobs
                    return state
                except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as exc:
                    if attempt_idx >= self.config.max_retry_per_sample:
                        state.status = Status.FAILED
                        state.error_msg = (
                            f"Teacher {self.name!r} logprobs calculation failed after {attempt_idx + 1} attempts; "
                            f"group_id={state.group_id}; routed_replica={routed_replica_idx}; "
                            f"replica={replica_idx}; endpoint={url}; last_error={exc}"
                        )
                        logger.warning(state.error_msg)
                        return state
                    await asyncio.sleep(0.1)
            raise RuntimeError("Teacher scoring retry loop exited unexpectedly")
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

    def _construct_payload(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        *,
        logprob_start_len: int,
        image_data: Any | None = None,
    ) -> dict[str, Any]:
        if self.backend == "sglang":
            payload = self._construct_sglang_payload(
                prompt_ids,
                response_ids,
                logprob_start_len=logprob_start_len,
            )
        elif self.backend == "lmdeploy":
            payload = self._construct_lmdeploy_payload(
                prompt_ids,
                response_ids,
                logprob_start_len=logprob_start_len,
            )
        else:
            raise RuntimeError(f"Unsupported teacher backend: {self.backend}")
        if image_data:
            payload["image_data"] = image_data
        return payload

    @staticmethod
    def _construct_sglang_payload(
        prompt_ids: list[int],
        response_ids: list[int],
        *,
        logprob_start_len: int,
    ) -> dict[str, Any]:
        return {
            "input_ids": prompt_ids + response_ids,
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": 0,
            "stream": False,
        }

    @staticmethod
    def _construct_lmdeploy_payload(
        prompt_ids: list[int],
        response_ids: list[int],
        *,
        logprob_start_len: int,
    ) -> dict[str, Any]:
        return {
            "input_ids": prompt_ids + response_ids,
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            "max_tokens": 0,
            "stream": False,
        }

    def _parse_response(
        self,
        response: httpx.Response,
        response_ids: list[int],
        *,
        logprob_start_len: int,
        expanded_prompt_len: int | None = None,
    ) -> tuple[list[int], list[float]]:
        if self.backend == "sglang":
            response_logprobs = self._parse_sglang_response(
                response,
                response_ids,
                logprob_start_len=logprob_start_len,
                expanded_prompt_len=expanded_prompt_len,
            )
        else:
            response_logprobs = self._parse_lmdeploy_response(
                response,
                response_ids,
                logprob_start_len=logprob_start_len,
                expanded_prompt_len=expanded_prompt_len,
            )
        return self._validate_response_logprobs(response_logprobs, response_ids)

    @staticmethod
    def _parse_sglang_response(
        response: httpx.Response,
        response_ids: list[int],
        *,
        logprob_start_len: int,
        expanded_prompt_len: int | None = None,
    ) -> list[Any]:
        raw_logprobs = TeacherLogprobClient._get_input_token_logprobs(response)
        prompt_token_count = response.json()["meta_info"]["prompt_tokens"]
        if expanded_prompt_len is not None:
            expected_prompt_token_count = expanded_prompt_len + len(response_ids)
            if prompt_token_count != expected_prompt_token_count:
                raise ValueError(
                    "SGLang expanded prompt length mismatch: "
                    f"expected {expected_prompt_token_count} total input tokens "
                    f"({expanded_prompt_len} prompt + {len(response_ids)} response), "
                    f"got {prompt_token_count}"
                )
        # SGLang includes an unscorable placeholder at the requested boundary,
        # so N processed tokens with boundary S produce N-S rows.
        expected_rows = prompt_token_count - logprob_start_len
        if len(raw_logprobs) != expected_rows:
            raise ValueError(
                "SGLang teacher logprob length mismatch: "
                f"expected {expected_rows} rows for logprob_start_len={logprob_start_len}, "
                f"got {len(raw_logprobs)}"
            )
        return raw_logprobs[-len(response_ids) :]

    @staticmethod
    def _parse_lmdeploy_response(
        response: httpx.Response,
        response_ids: list[int],
        *,
        logprob_start_len: int,
        expanded_prompt_len: int | None = None,
    ) -> list[Any]:
        raw_logprobs = TeacherLogprobClient._get_input_token_logprobs(response)
        prompt_token_count = response.json()["meta_info"]["prompt_tokens"]
        if expanded_prompt_len is not None:
            expected_prompt_token_count = expanded_prompt_len + len(response_ids)
            if prompt_token_count != expected_prompt_token_count:
                raise ValueError(
                    "LMDeploy expanded prompt length mismatch: "
                    f"expected {expected_prompt_token_count} total input tokens "
                    f"({expanded_prompt_len} prompt + {len(response_ids)} response), "
                    f"got {prompt_token_count}"
                )
        # LMDeploy omits the unscorable boundary row, hence one fewer row than
        # SGLang for the same processed input and logprob boundary.
        expected_rows = prompt_token_count - logprob_start_len - 1
        if len(raw_logprobs) != expected_rows:
            raise ValueError(
                "LMDeploy teacher logprob length mismatch: "
                f"expected {expected_rows} rows for logprob_start_len={logprob_start_len}, "
                f"got {len(raw_logprobs)}"
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

from __future__ import annotations

import asyncio
import math
import os
import time
from hashlib import blake2b
from typing import Any, Literal, cast

import httpx

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.loss import DistillationLossConfig
from xtuner.v1.utils import get_logger

from .config import RolloutTeacherConfig


logger = get_logger()


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


class RolloutTeacherReplicaRouter:
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


class RolloutTeacherClient:
    """Asynchronous teacher client scoped to one AgentLoop."""

    def __init__(self, config: RolloutTeacherConfig, loss_config: DistillationLossConfig) -> None:
        self.config = config
        self.loss_config = loss_config
        self.name = config.name
        self.backend = self._resolve_backend_from_env()
        if self.loss_config.uses_topk_targets and self.backend != "lmdeploy":
            raise RuntimeError("Rollout Teacher Top-K targets currently require LMDeploy")
        self.urls = [f"{endpoint.rstrip('/')}/generate" for endpoint in config.endpoints]
        self._semaphores = [asyncio.Semaphore(config.max_concurrency) for _ in self.urls]
        self._replica_router = RolloutTeacherReplicaRouter(len(self.urls))

        headers = {"Content-Type": "application/json"}
        if config.api_key is not None:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=config.request_timeout_s)

    async def compute_logprobs(self, state: RolloutState) -> RolloutState:
        start = time.perf_counter()
        try:
            scoring_input = self._prepare_scoring_input(state)
            if scoring_input is None:
                return state
            prompt_ids, response_ids = scoring_input
            routed_replica_idx = self._replica_router.resolve_replica_idx(
                teacher_name=self.name,
                data_source=str(state.extra_fields.get("origin_data_source", "")),
                group_id=state.group_id,
            )
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

            attempt_idx = 0
            while True:
                replica_idx = (routed_replica_idx + attempt_idx) % len(self.urls)
                url = self.urls[replica_idx]
                try:
                    async with self._semaphores[replica_idx]:
                        response = await self._client.post(url, json=payload)
                        response.raise_for_status()
                    teacher_tokens: list[int] | list[list[int]]
                    teacher_logprobs: list[float] | list[list[float]]
                    if self.loss_config.uses_topk_targets:
                        teacher_tokens, teacher_logprobs = self._parse_topk_response(
                            response,
                            response_ids,
                            logprob_start_len=logprob_start_len,
                            expanded_prompt_len=expanded_prompt_len,
                        )
                    else:
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
                    attempt_idx += 1
                    await asyncio.sleep(0.1)
        finally:
            state.extra_fields["teacher_score_time_s"] = time.perf_counter() - start

    def _prepare_scoring_input(self, state: RolloutState) -> tuple[list[int], list[int]] | None:
        if state.input_ids is not None:
            input_ids = state.input_ids
            labels = state.labels
            if len(input_ids) < 2:
                state.status = Status.FAILED
                state.error_msg = f"Teacher {self.name!r} trace scoring requires at least two input_ids"
                return None
            if labels is None or len(labels) != len(input_ids):
                state.status = Status.FAILED
                state.error_msg = (
                    f"Teacher {self.name!r} trace scoring requires input_ids and labels with equal lengths; "
                    f"got {len(input_ids)} and {None if labels is None else len(labels)}"
                )
                return None
            scoring_start = next((index for index, label in enumerate(labels[1:], start=1) if label != -100), None)
            if scoring_start is None:
                state.status = Status.FAILED
                state.error_msg = f"Teacher {self.name!r} trace scoring requires at least one trainable label"
                return None
            # Keep later masked turns in the scored suffix so subsequent
            # assistant turns retain their complete causal context.
            return input_ids[:scoring_start], input_ids[scoring_start:]

        prompt_ids = cast(list[int] | None, state.prompt_ids)
        response_ids = cast(list[int] | None, state.response_ids)
        if not prompt_ids or not response_ids:
            state.status = Status.FAILED
            state.error_msg = f"Teacher {self.name!r} scoring requires non-empty prompt_ids and response_ids"
            return None
        return prompt_ids, response_ids

    @staticmethod
    def _resolve_backend_from_env() -> Literal["sglang", "lmdeploy"]:
        use_sglang = os.environ.get("XTUNER_USE_SGLANG", "0") == "1"
        use_lmdeploy = os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "1"
        use_vllm = os.environ.get("XTUNER_USE_VLLM", "0") == "1"

        if use_vllm:
            raise RuntimeError("RolloutTeacherClient supports only SGLang or LMDeploy, not vLLM")
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
        if self.loss_config.uses_topk_targets:
            payload["top_logprobs_num"] = cast(int, self.loss_config.top_k)
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

    def _parse_topk_response(
        self,
        response: httpx.Response,
        response_ids: list[int],
        *,
        logprob_start_len: int,
        expanded_prompt_len: int | None = None,
    ) -> tuple[list[list[int]], list[list[float]]]:
        meta_info = self._get_response_meta_info(response)
        prompt_token_count = self._get_prompt_token_count(meta_info)
        if expanded_prompt_len is not None:
            expected_prompt_token_count = expanded_prompt_len + len(response_ids)
            if prompt_token_count != expected_prompt_token_count:
                raise ValueError(
                    "LMDeploy expanded prompt length mismatch: "
                    f"expected {expected_prompt_token_count} total input tokens "
                    f"({expanded_prompt_len} prompt + {len(response_ids)} response), "
                    f"got {prompt_token_count}"
                )

        raw_topk = self._get_meta_list(meta_info, "input_top_logprobs")
        expected_rows = prompt_token_count - logprob_start_len - 1
        if len(raw_topk) != expected_rows:
            raise ValueError(
                "LMDeploy teacher Top-K length mismatch: "
                f"expected {expected_rows} rows for logprob_start_len={logprob_start_len}, got {len(raw_topk)}"
            )
        if len(raw_topk) < len(response_ids):
            raise ValueError(
                "LMDeploy teacher Top-K response is shorter than the scored response: "
                f"{len(raw_topk)} vs {len(response_ids)}"
            )

        top_k = cast(int, self.loss_config.top_k)
        teacher_tokens: list[list[int]] = []
        teacher_logprobs: list[list[float]] = []
        for row_idx, row in enumerate(raw_topk[-len(response_ids) :]):
            if not isinstance(row, list) or len(row) != top_k:
                raise ValueError(f"Teacher Top-K row {row_idx} must contain exactly {top_k} entries")

            token_row: list[int] = []
            logprob_row: list[float] = []
            for entry_idx, entry in enumerate(row):
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ValueError(f"Teacher Top-K row {row_idx} entry {entry_idx} must be [logprob, token_id]")
                raw_logprob, raw_token_id = entry
                if isinstance(raw_logprob, bool) or not isinstance(raw_logprob, (int, float)):
                    raise ValueError(f"Teacher Top-K row {row_idx} entry {entry_idx} has a non-numeric logprob")
                if isinstance(raw_token_id, bool) or not isinstance(raw_token_id, int) or raw_token_id < 0:
                    raise ValueError(
                        f"Teacher Top-K row {row_idx} entry {entry_idx} has an invalid non-negative token id"
                    )
                logprob = float(raw_logprob)
                if not math.isfinite(logprob):
                    raise ValueError("Teacher Top-K logprobs contain NaN or Inf")
                logprob_row.append(logprob)
                token_row.append(raw_token_id)
            if len(token_row) != len(set(token_row)):
                raise ValueError(f"Teacher Top-K row {row_idx} contains duplicate token ids")
            teacher_tokens.append(token_row)
            teacher_logprobs.append(logprob_row)
        return teacher_tokens, teacher_logprobs

    @staticmethod
    def _parse_sglang_response(
        response: httpx.Response,
        response_ids: list[int],
        *,
        logprob_start_len: int,
        expanded_prompt_len: int | None = None,
    ) -> list[Any]:
        meta_info = RolloutTeacherClient._get_response_meta_info(response)
        raw_logprobs = RolloutTeacherClient._get_meta_list(meta_info, "input_token_logprobs")
        prompt_token_count = RolloutTeacherClient._get_prompt_token_count(meta_info)
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
        meta_info = RolloutTeacherClient._get_response_meta_info(response)
        raw_logprobs = RolloutTeacherClient._get_meta_list(meta_info, "input_token_logprobs")
        prompt_token_count = RolloutTeacherClient._get_prompt_token_count(meta_info)
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
    def _get_response_meta_info(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid teacher response") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("meta_info"), dict):
            raise ValueError("Invalid teacher response")
        return cast(dict[str, Any], payload["meta_info"])

    @staticmethod
    def _get_prompt_token_count(meta_info: dict[str, Any]) -> int:
        prompt_token_count = meta_info.get("prompt_tokens")
        if isinstance(prompt_token_count, bool) or not isinstance(prompt_token_count, int) or prompt_token_count < 1:
            raise ValueError("Invalid teacher response prompt_tokens")
        return prompt_token_count

    @staticmethod
    def _get_meta_list(meta_info: dict[str, Any], field_name: str) -> list[Any]:
        value = meta_info.get(field_name)
        if not isinstance(value, list):
            raise ValueError(f"Invalid teacher response {field_name}")
        return value

    @staticmethod
    def _validate_response_logprobs(
        response_logprobs: list[Any],
        response_ids: list[int],
    ) -> tuple[list[int], list[float]]:
        teacher_tokens: list[int] = []
        teacher_logprobs: list[float] = []
        for row_idx, item in enumerate(response_logprobs):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"Teacher logprob row {row_idx} must be [logprob, token_id]")
            raw_logprob, raw_token_id = item
            if isinstance(raw_logprob, bool) or not isinstance(raw_logprob, (int, float)):
                raise ValueError(f"Teacher logprob row {row_idx} has a non-numeric logprob")
            if isinstance(raw_token_id, bool) or not isinstance(raw_token_id, int) or raw_token_id < 0:
                raise ValueError(f"Teacher logprob row {row_idx} has an invalid non-negative token id")
            teacher_tokens.append(raw_token_id)
            teacher_logprobs.append(float(raw_logprob))

        if len(teacher_logprobs) != len(response_ids):
            raise ValueError("Teacher logprob length mismatch")
        if teacher_tokens != response_ids:
            raise ValueError("Teacher token ids mismatch")
        if not all(math.isfinite(logprob) for logprob in teacher_logprobs):
            raise ValueError("Teacher logprobs contain NaN or Inf")
        return teacher_tokens, teacher_logprobs


def route_rollout_teacher_client(
    state: RolloutState,
    *,
    data_source_teacher_map: dict[str, str],
    teacher_clients: dict[str, RolloutTeacherClient],
) -> RolloutTeacherClient:
    data_source = state.extra_fields["origin_data_source"]
    teacher_name = data_source_teacher_map[data_source]
    return teacher_clients[teacher_name]

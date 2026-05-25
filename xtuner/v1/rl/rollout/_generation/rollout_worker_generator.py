from __future__ import annotations

import asyncio
import base64
import copy
import json
import threading
import traceback
from typing import Any, cast

import httpx
import numpy as np
import ray
from transformers import AutoConfig, AutoTokenizer

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    SampleParams,
    Status,
    reset_rollout_response,
    update_status_from_finish_reason,
)
from xtuner.v1.rl.utils import cancel_and_drain, get_eos_token
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType, HttpRequestResult

from ..utils import PartialRolloutHandler
from ..worker import RolloutConfig


LMDEPLOY_SHARED_STORE = "shared_store"
LMDEPLOY_SHARED_STORE_NAMESPACE = "lmdeploy"


class RolloutWorkerGenerator:
    """Generator bound to one rollout worker backend URL."""

    def __init__(self, config: RolloutConfig, rank: int, server_url: str) -> None:
        self.config = config
        self.rank = rank
        self.server_url = server_url
        self.backend = config.rollout_backend
        self.logger = get_logger(log_dir=config.worker_log_dir, tag=f"RolloutWorkerGenerator-{rank}")
        self.endpoints = self._build_endpoints()
        tokenizer_path = config.tokenizer_path or config.model_path
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None
        self.model_name = config.model_name
        self.enable_return_routed_experts = config.enable_return_routed_experts
        self._partial_rollout_handler: PartialRolloutHandler | None = None
        self.receive_abort_request = threading.Event()
        self.abort_timeout = 10.0
        http_concurrency = config.rollout_max_batch_size_per_instance * config.allow_over_concurrency_ratio
        limits = httpx.Limits(max_connections=http_concurrency, max_keepalive_connections=100)
        self.client = httpx.AsyncClient(limits=limits, timeout=config.rollout_timeout)
        eos_token = get_eos_token(config.model_path)
        self.eos_token: list[int] = [eos_token] if isinstance(eos_token, int) else eos_token
        self.lmdeploy_actor = None
        self.routed_experts_num_hidden_layers = None
        self.routed_experts_num_experts_per_tok = None
        if self.backend == "sglang":
            model_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
            text_config = getattr(model_config, "text_config", model_config)
            self.routed_experts_num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
            self.routed_experts_num_experts_per_tok = getattr(text_config, "num_experts_per_tok", None)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        return self._tokenizer

    @property
    def partial_rollout_handler(self) -> PartialRolloutHandler:
        if self._partial_rollout_handler is None:
            self._partial_rollout_handler = PartialRolloutHandler()
        return self._partial_rollout_handler

    def _build_endpoints(self) -> dict[str, str]:
        if self.backend == "lmdeploy":
            return {
                "generate": "generate",
                "v1/chat/completions": "v1/chat/completions",
                "abort_request": "abort_request",
            }
        if self.backend == "sglang":
            return {
                "generate": "generate",
                "v1/chat/completions": "v1/chat/completions",
                "abort_request": "abort_request",
            }
        if self.backend == "vllm":
            return {
                "generate": "v1/chat/completions",
                "v1/chat/completions": "v1/chat/completions",
                "abort_request": "abort_request",
            }
        raise ValueError(f"Unsupported rollout backend: {self.backend}")

    def update_server_url(self, server_url: str) -> None:
        self.server_url = server_url
        self.receive_abort_request.clear()

    async def pause_generation(self) -> bool:
        self.receive_abort_request.set()
        return await self._send_abort_request()

    def continue_generation(self) -> None:
        self.receive_abort_request.clear()

    async def _send_abort_request(self) -> bool:
        endpoint = self.endpoints.get("abort_request", "abort_request")
        url = f"{self.server_url}/{endpoint}"
        try:
            response = await self.client.post(url)
            response.raise_for_status()
            return True
        except Exception as exc:
            self.logger.warning(f"Failed to send abort request to {url}: {exc}")
            return False

    async def _wait_abort_request(self) -> None:
        while not self.receive_abort_request.is_set():
            await asyncio.sleep(1)

    async def generate(self, rollout_state: RolloutState, *, enable_partial_rollout: bool = False) -> RolloutState:
        if self.receive_abort_request.is_set():
            rollout_state.finish_reason = "abort"
            rollout_state.status = Status.ABORTED
            return rollout_state

        uid = rollout_state.uid
        sample_params: SampleParams = rollout_state.sample_params
        max_tokens = sample_params.max_tokens
        if sample_params.return_token_ids:
            endpoint_url = f"{self.server_url}/{self.endpoints['generate']}"
        else:
            endpoint_url = f"{self.server_url}/{self.endpoints['v1/chat/completions']}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        if enable_partial_rollout:
            rollout_state = self.partial_rollout_handler.preprocess(rollout_state, max_tokens)
        elif rollout_state.status == Status.ABORTED:
            rollout_state = reset_rollout_response(rollout_state)
            rollout_state.sample_params = rollout_state.sample_params.model_copy(update={"max_tokens": max_tokens})
            rollout_state.status = Status.INIT

        payload = self._get_request_payload(rollout_state)
        max_retries = self.config.max_retry_per_sample

        if rollout_state.status == Status.COMPLETED:
            self.logger.debug(f"Request {uid} is already marked as COMPLETED, skipping generation.")
            return rollout_state

        input_ids = payload.get("input_ids", [])
        max_tokens = payload.get("max_tokens") or payload.get("max_new_tokens")
        sampling_params = payload.get("sampling_params")
        if max_tokens is None and isinstance(sampling_params, dict):
            max_tokens = sampling_params.get("max_tokens") or sampling_params.get("max_new_tokens")
        max_tokens = cast(int | None, max_tokens)
        last_id = input_ids[-1] if len(input_ids) > 0 else "None"
        is_max_tokens_zero = max_tokens is not None and max_tokens <= 0
        is_eos_reached = len(input_ids) > 0 and input_ids[-1] in self.eos_token
        if is_max_tokens_zero or is_eos_reached:
            self.logger.debug(
                f"No generation needed for request {uid}: max_tokens={max_tokens} or last input_id={last_id} is in eos_token."
            )
            rollout_state.finish_reason = "stop" if is_eos_reached else "length"
            rollout_state.status = Status.COMPLETED
            return rollout_state

        for attempt in range(max_retries + 1):
            is_last_attempt = attempt == max_retries
            http_result = await self._safe_post_request(endpoint_url, headers=headers, payload=payload)
            if http_result.response:
                rollout_state = await self._safe_handle_response(rollout_state, http_result.response)
                if rollout_state.status in [Status.COMPLETED, Status.ABORTED]:
                    return rollout_state
                if is_last_attempt:
                    self.logger.warning(
                        f"Invalid rollout response for request {uid} after {max_retries} attempts, marking as FAILED."
                    )
                    rollout_state.status = Status.FAILED
                    rollout_state.error_msg = f"Invalid rollout response after {max_retries} attempts."
                    return rollout_state
                self.logger.warning(
                    f"Invalid rollout response for request {uid}, retrying {attempt + 1}/{max_retries}."
                )
                await asyncio.sleep(0.1)
                continue

            if http_result.error_type == HttpRequestErrorType.REQUEST_ABORTED:
                rollout_state.finish_reason = "abort"
                rollout_state.status = update_status_from_finish_reason("abort")
                return rollout_state

            if http_result.is_client_error:
                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} was skipped due to client error {http_result.error_type} with {http_result.error_msg}"
                )
                rollout_state.error_msg = (
                    f"Client error {http_result.error_type} with message: {http_result.error_msg}"
                )
                rollout_state.status = Status.FAILED
                return rollout_state

            if http_result.is_server_error:
                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} failed due to server error {http_result.error_type} with {http_result.error_msg}"
                )
                rollout_state.error_msg = (
                    f"Server error {http_result.error_type} with message: {http_result.error_msg}"
                )
                rollout_state.status = Status.FAILED
                return rollout_state

            if http_result.is_retryable:
                if is_last_attempt:
                    self.logger.warning(
                        f"rollout request {uid} to {http_result.url} failed after {max_retries} attempts due to retryable error {http_result.error_type} with {http_result.error_msg}"
                    )
                    rollout_state.error_msg = f"Request failed after {max_retries} attempts due to retryable error {http_result.error_type} with message: {http_result.error_msg}"
                    rollout_state.status = Status.FAILED
                    return rollout_state
                self.logger.warning(
                    f"rollout request {uid} to {http_result.url} failed due to retryable error {http_result.error_type} with {http_result.error_msg}, retrying {attempt + 1}/{max_retries}."
                )
                await asyncio.sleep(0.1)
                continue

            if http_result.is_unknown_error:
                raise RuntimeError(
                    f"Unexpected error during rollout request {uid} to {http_result.url}: {http_result.exception}"
                )
        return rollout_state

    async def _safe_post_request(self, url, headers, payload) -> HttpRequestResult:
        send_task = None
        abort_task = None
        try:
            if self.receive_abort_request.is_set():
                self.logger.debug(f"Request to {url} was cancelled before sending due to an abort signal.")
                return HttpRequestResult(error_type=HttpRequestErrorType.REQUEST_ABORTED, url=url, payload=payload)
            req = self.client.build_request("POST", url, headers=headers, json=payload)
            send_task = asyncio.create_task(self.client.send(req))
            abort_task = asyncio.create_task(self._wait_abort_request())
            done, _ = await asyncio.wait({send_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
            if send_task in done:
                response = await send_task
            else:
                try:
                    response = await asyncio.wait_for(asyncio.shield(send_task), timeout=self.abort_timeout)
                except asyncio.TimeoutError:
                    self.logger.debug(
                        f"Request to {url} did not return within {self.abort_timeout:.2f}s after abort signal."
                    )
                    await cancel_and_drain([send_task])
                    return HttpRequestResult(error_type=HttpRequestErrorType.REQUEST_ABORTED, url=url, payload=payload)
            response.raise_for_status()
            return HttpRequestResult(response=response)
        except asyncio.CancelledError:
            self.logger.debug(f"Request to {url} was cancelled while waiting for the response.")
            await cancel_and_drain([send_task, abort_task])
            self.receive_abort_request.set()
            return HttpRequestResult(error_type=HttpRequestErrorType.REQUEST_ABORTED, url=url, payload=payload)
        except Exception as exc:
            error_type = HttpRequestErrorType.from_exception(exc)
            return HttpRequestResult(error_type=error_type, exception=exc, url=url, payload=payload)
        finally:
            await cancel_and_drain([abort_task])

    def _get_request_payload(self, rollout_state: RolloutState) -> dict[str, Any]:
        if self.backend == "lmdeploy":
            return self._get_lmdeploy_request_payload(rollout_state)
        if self.backend == "sglang":
            return self._get_sglang_request_payload(rollout_state)
        if self.backend == "vllm":
            return self._get_vllm_request_payload(rollout_state)
        raise ValueError(f"Unsupported rollout backend: {self.backend}")

    def _get_lmdeploy_request_payload(self, rollout_state: RolloutState) -> dict[str, Any]:
        sample_params = rollout_state.sample_params
        optional_fields: dict[str, object] = {}
        if rollout_state.tools is not None:
            optional_fields["tools"] = rollout_state.tools
        if rollout_state.tool_choice is not None:
            optional_fields["tool_choice"] = rollout_state.tool_choice

        if sample_params.return_token_ids:
            payload: dict[str, Any] = {"model": self.model_name, **optional_fields}
            if "image_data" in rollout_state.extra_fields:
                assert rollout_state.tokens is not None, "input_tokens is required when image_data is provided."
                payload["image_data"] = rollout_state.extra_fields["image_data"]
            if rollout_state.tokens is not None:
                payload["input_ids"] = rollout_state.tokens
            else:
                text_prompt = self.tokenizer.apply_chat_template(
                    rollout_state.message, tokenize=False, add_generation_prompt=True
                )
                payload["input_ids"] = self.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
            sample_params.return_routed_experts = True if self.enable_return_routed_experts else False
            payload.update(sample_params.model_dump(exclude_none=True))
            return payload

        payload = {"model": self.model_name, "messages": rollout_state.message, **optional_fields}
        lmdeploy_sample_params: dict[str, Any] = {
            "temperature": sample_params.temperature,
            "top_p": sample_params.top_p,
            "n": sample_params.n,
            "stream": sample_params.stream,
            "max_tokens": sample_params.max_tokens,
            "repetition_penalty": sample_params.repetition_penalty,
            "top_k": sample_params.top_k,
            "skip_special_tokens": sample_params.skip_special_tokens,
        }
        if sample_params.stops:
            lmdeploy_sample_params["stop"] = sample_params.stops
        if sample_params.min_tokens > 0:
            lmdeploy_sample_params["min_new_tokens"] = sample_params.min_tokens
        payload.update(lmdeploy_sample_params)
        return payload

    def _get_sglang_request_payload(self, rollout_state: RolloutState) -> dict[str, Any]:
        sample_params = rollout_state.sample_params
        payload: dict[str, Any] = {"model": self.model_name}
        if rollout_state.tools is not None:
            payload["tools"] = rollout_state.tools
        if rollout_state.tool_choice is not None:
            payload["tool_choice"] = rollout_state.tool_choice

        sample_params_dict = sample_params.model_dump()
        sglang_sample_params = self._transform_sglang_sample_params(sample_params_dict)
        sglang_extra_params = self._transform_sglang_extra_params(sample_params_dict)
        payload.update(sglang_extra_params)
        if self.enable_return_routed_experts and not rollout_state.extra_fields.get("disable_routed_experts", False):
            payload["return_routed_experts"] = True

        if sample_params.return_token_ids:
            if "image_data" in rollout_state.extra_fields:
                assert rollout_state.tokens is not None, "input_ids is required when image_data is provided."
                payload["image_data"] = rollout_state.extra_fields["image_data"]
            if rollout_state.tokens is not None:
                payload["input_ids"] = rollout_state.tokens
            else:
                text_prompt = self.tokenizer.apply_chat_template(
                    rollout_state.message, tokenize=False, add_generation_prompt=True
                )
                payload["input_ids"] = self.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
            payload["sampling_params"] = sglang_sample_params
            return payload

        payload["messages"] = rollout_state.message
        payload.update(sglang_sample_params)
        payload["max_tokens"] = sglang_sample_params["max_new_tokens"]
        payload["min_tokens"] = sglang_sample_params["min_new_tokens"]
        payload.pop("max_new_tokens", None)
        payload.pop("min_new_tokens", None)
        return payload

    def _get_vllm_request_payload(self, rollout_state: RolloutState) -> dict[str, Any]:
        sample_params = rollout_state.sample_params
        prompt = copy.deepcopy(rollout_state.message)
        extra_fields = rollout_state.extra_fields
        if "image_data" in extra_fields:
            image_index = 0
            for message in prompt:
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                new_content = []
                for content_part in message.get("content", []):
                    if not isinstance(content_part, dict):
                        new_content.append(content_part)
                        continue
                    if content_part.get("type") == "image_url":
                        content_part["image_url"]["url"] = f"file://{extra_fields['image_data'][image_index]}"
                        content_part["image_url"].pop("image_wh", None)
                        image_index += 1
                    new_content.append(content_part)
                message["content"] = new_content
            assert image_index == len(extra_fields["image_data"]), (
                f"Expected {len(extra_fields['image_data'])} images, but processed {image_index}."
            )

        payload: dict[str, Any] = {
            "model": self.config.model_path,
            "messages": prompt,
            "stream": sample_params.stream,
        }
        if rollout_state.tokens is not None:
            payload["input_ids"] = rollout_state.tokens
        elif "train_prompt_ids" in extra_fields:
            payload["input_ids"] = extra_fields["train_prompt_ids"]
        payload.update(self._transform_vllm_sample_params(sample_params.model_dump(), sample_params.model_dump()))
        return payload

    def _transform_sglang_sample_params(self, sample_params: dict[str, Any]) -> dict[str, Any]:
        if sample_params["top_p"] > 0:
            sample_params["top_k"] = -1
        sglang_sample_params = {
            "n": sample_params["n"],
            "top_k": sample_params["top_k"],
            "top_p": sample_params["top_p"],
            "temperature": sample_params["temperature"],
            "repetition_penalty": sample_params["repetition_penalty"],
            "presence_penalty": sample_params["presence_penalty"],
            "frequency_penalty": sample_params["frequency_penalty"],
            "max_new_tokens": sample_params["max_tokens"],
            "min_new_tokens": sample_params["min_tokens"],
            "stop": sample_params["stops"],
            "stop_token_ids": sample_params["stop_token_ids"],
            "skip_special_tokens": sample_params["skip_special_tokens"],
        }
        sampling_seed = sample_params.get("sampling_seed")
        if sampling_seed is None and XTUNER_DETERMINISTIC:
            sampling_seed = self.config.random_seed
        if sampling_seed is not None:
            sglang_sample_params["sampling_seed"] = sampling_seed
        return sglang_sample_params

    def _transform_sglang_extra_params(self, extra_params: dict[str, Any]) -> dict[str, Any]:
        return {
            "stream": extra_params["stream"],
            "return_logprob": extra_params["return_logprob"],
            "include_stop_str_in_output": extra_params["include_stop_str_in_output"],
            "no_stop_trim": extra_params.get("no_stop_trim", False),
            "spaces_between_special_tokens": extra_params.get("spaces_between_special_tokens", False),
        }

    def _transform_vllm_sample_params(
        self, sample_params: dict[str, Any], extra_params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        vllm_sample_params = copy.deepcopy(sample_params)
        if extra_params:
            vllm_sample_params.update(extra_params)
        if "stops" in vllm_sample_params:
            vllm_sample_params["stop"] = vllm_sample_params.pop("stops")
        if "no_stop_trim" in vllm_sample_params:
            vllm_sample_params["include_stop_str_in_output"] = vllm_sample_params.pop("no_stop_trim")
        if "top_logprobs" in vllm_sample_params and "return_logprob" in vllm_sample_params:
            vllm_sample_params["logprobs"] = vllm_sample_params.pop("return_logprob")
        return vllm_sample_params

    async def _decode_routed_experts(self, routed_experts: Any) -> Any:
        if self.backend == "lmdeploy":
            if isinstance(routed_experts, str):
                if self.lmdeploy_actor is None:
                    self.lmdeploy_actor = ray.get_actor(LMDEPLOY_SHARED_STORE, namespace=LMDEPLOY_SHARED_STORE_NAMESPACE)
                routed_experts_data = await self.lmdeploy_actor.get.remote(routed_experts)
                return ray.put(np.asarray(routed_experts_data))
            return np.asarray(routed_experts)
        if self.backend == "sglang":
            if isinstance(routed_experts, str):
                routed_experts_flat = np.frombuffer(base64.b64decode(routed_experts), dtype=np.int32)
                routed_experts_array = routed_experts_flat.reshape(
                    -1,
                    self.routed_experts_num_hidden_layers,
                    self.routed_experts_num_experts_per_tok,
                )
                return routed_experts_array.copy()
            return np.asarray(routed_experts)
        if self.backend == "vllm":
            if isinstance(routed_experts, str):
                routed_experts = ray.cloudpickle.loads(base64.b64decode(routed_experts))
            return np.asarray(routed_experts)
        return routed_experts

    async def _safe_handle_response(self, rollout_state: RolloutState, http_response: httpx.Response) -> RolloutState:
        if self.backend == "vllm":
            return await self._safe_handle_vllm_response(rollout_state, http_response)
        return await self._safe_handle_openai_or_token_response(rollout_state, http_response)

    async def _safe_handle_openai_or_token_response(
        self, rollout_state: RolloutState, http_response: httpx.Response
    ) -> RolloutState:
        uid = rollout_state.message_uid
        sample_params = rollout_state.sample_params
        response = http_response.json()

        if sample_params.return_token_ids:
            response_ids: list[int] = []
            logprobs: list[float] = []
            routed_experts = None
            returned_response = ""
            try:
                meta_info = response.get("meta_info") or {}
                finish_reason_info = meta_info.get("finish_reason") or {}
                finish_reason = finish_reason_info.get("type")
                if finish_reason is None:
                    if self.receive_abort_request.is_set():
                        rollout_state.finish_reason = "abort"
                        rollout_state.status = Status.ABORTED
                    else:
                        rollout_state.finish_reason = "error"
                        rollout_state.status = Status.FAILED
                    rollout_state.error_msg = "Missing finish_reason in response meta_info"
                    return rollout_state
                returned_response = response.get("text", "")
                if meta_info.get("output_token_logprobs") is not None:
                    response_ids = [item[1] for item in meta_info["output_token_logprobs"]]
                    logprobs = [item[0] for item in meta_info["output_token_logprobs"]]
                else:
                    num_return_tokens = meta_info.get("completion_tokens", 0)
                    response_ids = response["output_ids"][-num_return_tokens:] if num_return_tokens > 0 else []

                if self.enable_return_routed_experts:
                    assert "routed_experts" in meta_info, (
                        "enable_return_routed_experts is True, but routed_experts is not in meta_info"
                    )
                    routed_experts = meta_info["routed_experts"]
                    if routed_experts is not None:
                        routed_experts = await self._decode_routed_experts(routed_experts)
                        if not isinstance(routed_experts, ray.ObjectRef):
                            routed_experts = ray.put(routed_experts)

                rollout_status = update_status_from_finish_reason(finish_reason)
                if rollout_status == Status.COMPLETED:
                    validation_errors = []
                    if not response_ids:
                        validation_errors.append("empty response_ids")
                    if not returned_response:
                        validation_errors.append("empty response text")
                    if sample_params.return_logprob and not logprobs:
                        validation_errors.append("missing logprobs")
                    if self.enable_return_routed_experts and routed_experts is None:
                        validation_errors.append("missing routed_experts")
                    if validation_errors:
                        error_msg = f"Incomplete rollout data for msg {uid}: {', '.join(validation_errors)}"
                        self.logger.error(error_msg)
                        rollout_state.status = Status.FAILED
                        rollout_state.error_msg = error_msg
                        return rollout_state
                elif rollout_status == Status.FAILED:
                    error_msg = f"Rollout failed for msg {uid} with finish_reason {finish_reason}"
                    self.logger.error(error_msg)
                    rollout_state.status = Status.FAILED
                    rollout_state.error_msg = error_msg
                    return rollout_state

                if enable_partial_rollout:
                    expect_len = meta_info.get("prompt_tokens", 0) + meta_info.get("completion_tokens", 0) - 1
                    rollout_state = await self.partial_rollout_handler.postprocess(
                        rollout_state,
                        response=returned_response,
                        response_ids=response_ids,
                        logprobs=logprobs,
                        routed_experts=routed_experts,
                        finish_reason=finish_reason,
                        status=rollout_status,
                        routed_experts_expect_len=expect_len,
                    )
                else:
                    rollout_state.response = returned_response
                    rollout_state.response_ids = response_ids
                    rollout_state.logprobs = logprobs
                    rollout_state.routed_experts = routed_experts
                    rollout_state.finish_reason = finish_reason
                    rollout_state.status = rollout_status
                return rollout_state
            except Exception as exc:
                raise self._response_error(exc, response, uid)

        try:
            returned_response = response["choices"][0]["message"]["content"]
            finish_reason = response["choices"][0]["finish_reason"]
            rollout_status = update_status_from_finish_reason(finish_reason)
            if rollout_status == Status.COMPLETED and not returned_response:
                rollout_state.status = Status.FAILED
                rollout_state.error_msg = "Empty response text"
                return rollout_state
            rollout_state.response = returned_response
            rollout_state.finish_reason = finish_reason
            rollout_state.status = rollout_status
            return rollout_state
        except Exception as exc:
            raise self._response_error(exc, response, uid)

    async def _safe_handle_vllm_response(self, rollout_state: RolloutState, http_response) -> RolloutState:
        uid = rollout_state.uid or rollout_state.message_uid
        sample_params = rollout_state.sample_params
        last_token_ids: list[int] = []
        last_logprobs: list[float] = []
        routed_experts = None

        response_json = http_response.json()
        try:
            response_choice = response_json["choices"][0]
            if response_choice.get("logprobs") is not None:
                last_token_ids = response_choice.get("token_ids", response_json.get("token_ids", []))
                last_logprobs = [
                    item["logprob"] for item in response_choice["logprobs"].get("content", []) if "logprob" in item
                ]
                assert len(last_token_ids) == len(last_logprobs)
                assert len(last_token_ids) <= sample_params.max_tokens, (
                    f"Generation length exceeds limit: generated {len(last_token_ids)}, limit {sample_params.max_tokens}"
                )

            last_trajectory = response_choice["message"].get("content") or ""
            finish_reason = response_choice.get("finish_reason")
            if finish_reason == "abort" and not self.receive_abort_request.is_set():
                self.receive_abort_request.set()
                self.logger.info(f"Setting receive_abort_request to True for rank {self.rank}")

            if self.enable_return_routed_experts:
                routed_experts = response_choice.get("routed_experts", response_json.get("routed_experts"))
                if routed_experts is not None:
                    routed_experts = await self._decode_routed_experts(routed_experts)
                    if not isinstance(routed_experts, ray.ObjectRef):
                        routed_experts = ray.put(routed_experts)

            rollout_status = update_status_from_finish_reason(finish_reason)
            if rollout_status == Status.COMPLETED:
                validation_errors = []
                if sample_params.return_token_ids and len(last_token_ids) == 0:
                    validation_errors.append("empty response_ids")
                if sample_params.return_logprob and len(last_logprobs) == 0:
                    validation_errors.append("missing logprobs")
                if not last_trajectory:
                    validation_errors.append("empty response text")
                if self.enable_return_routed_experts and routed_experts is None:
                    validation_errors.append("missing routed_experts")
                if validation_errors:
                    error_msg = f"Incomplete rollout data for request {uid}: {', '.join(validation_errors)}"
                    self.logger.error(f"{error_msg}. Raw response: {response_json}")
                    rollout_state.status = Status.FAILED
                    rollout_state.error_msg = error_msg
                    return rollout_state

            rollout_state.response = last_trajectory
            rollout_state.response_ids = last_token_ids if len(last_token_ids) > 0 else None
            rollout_state.logprobs = last_logprobs if len(last_logprobs) > 0 else None
            rollout_state.routed_experts = routed_experts
            rollout_state.finish_reason = finish_reason
            rollout_state.status = rollout_status
            return rollout_state
        except Exception as exc:
            raise self._response_error(exc, response_json, uid)

    def _response_error(self, exc: Exception, response: dict[str, Any], uid: Any) -> RuntimeError:
        response_for_log = {k: v for k, v in response.items() if k not in ("logprobs", "response_ids")}
        if isinstance(exc, KeyError):
            return RuntimeError(f"Missing expected key {exc} in response {response_for_log} for {uid}")
        if isinstance(exc, IndexError):
            return RuntimeError(f"Index error {exc} while processing response {response_for_log} for {uid}")
        if isinstance(exc, AssertionError):
            return RuntimeError(f"AssertionError: {exc} when processing response {response_for_log} for {uid}")
        if isinstance(exc, json.JSONDecodeError):
            return RuntimeError(f"JSONDecodeError: {exc} when processing response {response} for {uid}")
        if isinstance(exc, TypeError):
            return RuntimeError(f"TypeError: {exc} when processing response {response_for_log} for {uid}")
        return RuntimeError(
            f"Unexpected error: {exc} when processing response {response_for_log} for {uid}\nTraceback: {traceback.format_exc()}"
        )

RayRolloutWorkerGenerator = ray.remote(RolloutWorkerGenerator)

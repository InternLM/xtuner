from __future__ import annotations

import asyncio
import base64
import json
import math
import mimetypes
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

import httpx
from pydantic import Field

from xtuner.v1.data_proto.rl_data import RolloutState

from .native import Judger, JudgerConfig, JudgerOutput, JudgerOutputBatch, JudgerPayloadBatch


RubricTransport = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

_SYSTEM_PROMPT = """You are a strict multimodal rubric judge.
Evaluate the assistant response against every supplied rubric independently.
A criterion is met only when the assistant response itself explicitly states or demonstrates it.
Use the original question and images only to verify the response's visual claims.
Never use the question or images to supply reasoning, facts, or steps missing from the response.
Return only one JSON object with this schema:
{"results":[{"local_id":"a1","verdict":"met","evidence":"an exact contiguous quote from the response"}]}
The verdict must be either "met" or "unmet". Include every rubric exactly once.
For "met", evidence must be a verbatim substring of the assistant response proving the criterion.
For example, if the response says "The grid has two shaded squares.", evidence can be exactly "The grid has two shaded squares.".
Never copy the criterion into evidence, and never use phrases such as "The response identifies" as evidence.
For "unmet", give a short explanation of what is missing or incorrect.
Do not invent criteria, omit criteria, or reveal hidden reference answers.
"""


def _validate_rubrics(raw_rubrics: Any, total_weight: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rubrics, list) or not raw_rubrics:
        raise ValueError("Rubric judging requires a non-empty extra_info.rubrics list.")
    rubrics: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    weight_sum = 0.0
    for index, raw in enumerate(raw_rubrics):
        if not isinstance(raw, dict):
            raise TypeError(f"Rubric {index} must be a dict.")
        local_id = raw.get("local_id")
        criterion = raw.get("criterion")
        kind = raw.get("type")
        weight = raw.get("weight")
        if not isinstance(local_id, str) or not local_id.strip():
            raise ValueError(f"Rubric {index} requires a non-empty string local_id.")
        if local_id in seen_ids:
            raise ValueError(f"Duplicate rubric local_id: {local_id!r}.")
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(f"Rubric {local_id!r} requires a non-empty criterion.")
        if kind not in {"visual_grounding", "answer_coverage"}:
            raise ValueError(f"Rubric {local_id!r} has unsupported type {kind!r}.")
        if isinstance(weight, bool) or not isinstance(weight, int | float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Rubric {local_id!r} requires a finite positive weight.")
        seen_ids.add(local_id)
        weight_sum += float(weight)
        rubrics.append(
            {
                "local_id": local_id,
                "type": kind,
                "criterion": criterion,
                "weight": float(weight),
            }
        )
    if total_weight is not None:
        if (
            isinstance(total_weight, bool)
            or not isinstance(total_weight, int | float)
            or not math.isfinite(total_weight)
            or not math.isclose(float(total_weight), weight_sum, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise ValueError(f"total_weight {total_weight!r} does not match rubric weight sum {weight_sum}.")
    return rubrics


def _question_text(messages: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                part["text"] for part in content if part.get("type") == "text" and isinstance(part.get("text"), str)
            )
    return "\n".join(texts)


def _message_image_sources(messages: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image" and isinstance(part.get("image"), dict):
                url = part["image"].get("url")
            elif part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                url = part["image_url"].get("url")
            else:
                continue
            if isinstance(url, str) and url:
                sources.append(url)
    return sources


def _image_data_uri(source: str) -> str:
    if source.startswith("data:"):
        return source
    path = Path(source)
    data = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _response_content(body: dict[str, Any]) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Judge response does not contain choices[0].message.content.") from error
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raise ValueError("Judge response content must be a string or text-part list.")


def _parse_results(content: str, rubrics: list[dict[str, Any]], response: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    parsed: dict[str, Any] | None = None
    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "results" in candidate:
            parsed = candidate
            break
    if parsed is None or not isinstance(parsed.get("results"), list):
        raise ValueError("Judge output must contain a JSON object with a results list.")

    expected = {rubric["local_id"]: rubric for rubric in rubrics}
    results_by_id: dict[str, dict[str, Any]] = {}
    for raw in parsed["results"]:
        if not isinstance(raw, dict):
            raise ValueError("Each rubric result must be an object.")
        local_id = raw.get("local_id")
        verdict = raw.get("verdict")
        evidence = raw.get("evidence")
        if not isinstance(local_id, str):
            raise ValueError("Rubric result local_id must be a string.")
        if local_id not in expected:
            raise ValueError(f"Judge returned unknown rubric local_id {local_id!r}.")
        if local_id in results_by_id:
            raise ValueError(f"Judge returned duplicate rubric local_id {local_id!r}.")
        if verdict not in {"met", "unmet"}:
            raise ValueError(f"Rubric {local_id!r} verdict must be met or unmet.")
        if not isinstance(evidence, str):
            raise ValueError(f"Rubric {local_id!r} evidence must be a string.")
        result = {
            **expected[local_id],
            "verdict": verdict,
            "score": 1.0 if verdict == "met" else 0.0,
            "evidence": evidence,
        }
        if verdict == "met" and (not evidence or evidence not in response):
            result.update(
                judge_verdict="met",
                verdict="unmet",
                score=0.0,
                validation_error="met evidence is not a verbatim substring of the response",
            )
        results_by_id[local_id] = result
    missing = set(expected) - set(results_by_id)
    if missing:
        raise ValueError(f"Judge omitted rubric local_ids: {sorted(missing)}.")
    return [results_by_id[rubric["local_id"]] for rubric in rubrics]


def merge_outcome_and_process(
    rollout_states: RolloutState | list[RolloutState],
    judged: dict[str, Any],
    *,
    process_weight: float = 0.3,
    outcome_key: str = "outcome",
    process_key: str = "process",
) -> RolloutState | list[RolloutState]:
    """Merge outcome and process branches into the standard RL reward.

    Process scores are mapped from [0, 1] to [-1, 1] before fusion.

    Args:
        rollout_states (RolloutState | list[RolloutState]): States being judged.
        judged (dict[str, Any]): Raw outputs keyed by outcome and process branch.
        process_weight (float): Process-score weight in ``[0, 1]``. Defaults to 0.3.
        outcome_key (str): Outcome branch key. Defaults to ``outcome``.
        process_key (str): Process branch key. Defaults to ``process``.

    Returns:
        RolloutState | list[RolloutState]: States with fused ``reward["score"]`` and branch details.
    """
    if not 0.0 <= process_weight <= 1.0:
        raise ValueError("process_weight must be in [0, 1].")
    if isinstance(rollout_states, list):
        outputs = judged
        if not isinstance(outputs.get(outcome_key), list) or not isinstance(outputs.get(process_key), list):
            raise TypeError("Batch rubric merge requires list outputs for outcome and process branches.")
        outcome_outputs = outputs[outcome_key]
        process_outputs = outputs[process_key]
        if len(outcome_outputs) != len(rollout_states) or len(process_outputs) != len(rollout_states):
            raise ValueError("Batch rubric merge output lengths must match rollout states.")
        return [
            _merge_one(state, outcome, process, process_weight)
            for state, outcome, process in zip(rollout_states, outcome_outputs, process_outputs)
        ]
    return _merge_one(rollout_states, judged[outcome_key], judged[process_key], process_weight)


def _merge_one(
    rollout_state: RolloutState,
    outcome: JudgerOutput,
    process: JudgerOutput,
    process_weight: float,
) -> RolloutState:
    outcome_score = float(outcome["score"])
    process_score = 2.0 * float(process["score"]) - 1.0
    if not math.isfinite(outcome_score) or not math.isfinite(process_score):
        raise ValueError("Outcome and process scores must be finite.")
    rollout_state.reward = {
        "score": (1.0 - process_weight) * outcome_score + process_weight * process_score,
        "outcome_score": outcome_score,
        "process_score": process_score,
        "rubric_results": process.get("rubric_results", []),
        "outcome_detail": outcome,
        "process_detail": process,
    }
    return rollout_state


class RubricJudger(Judger):
    """Score rollout responses against per-sample rubrics through a multimodal
    judge."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        judger_name: str = "rubric",
        request_timeout: float = 120.0,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        max_tokens: int = 4096,
        transport: RubricTransport | None = None,
    ) -> None:
        super().__init__(judger_name=judger_name)
        if not endpoint or not model:
            raise ValueError("RubricJudger requires endpoint and model.")
        if request_timeout <= 0 or retry_backoff < 0 or max_retries < 0 or max_tokens <= 0:
            raise ValueError("Invalid RubricJudger timeout, retry, or max_tokens configuration.")
        self.endpoint = endpoint
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_tokens = max_tokens
        self._transport = transport

    def preprocess(self, rollout_state: RolloutState) -> dict[str, Any]:
        extra = rollout_state.extra_fields or {}
        rubrics = _validate_rubrics(extra.get("rubrics"), extra.get("total_weight"))
        image_sources = list(extra.get("image_data") or []) or _message_image_sources(rollout_state.message)
        if any(rubric["type"] == "visual_grounding" for rubric in rubrics) and not image_sources:
            raise ValueError("visual_grounding rubrics require at least one input image.")
        return {
            "response": rollout_state.response,
            "question": _question_text(rollout_state.message),
            "rubrics": rubrics,
            "image_sources": image_sources,
        }

    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutputBatch:
        if isinstance(payload, list):
            return list(await asyncio.gather(*(self._score(item) for item in payload)))
        return await self._score(payload)

    async def _score(self, payload: dict[str, Any]) -> JudgerOutput:
        if payload.get("response") is None:
            raise ValueError("Rubric judging requires a detokenized response.")
        rubrics = cast(list[dict[str, Any]], payload["rubrics"])
        request = self._build_request(payload, rubrics)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                body = await self._request(request)
                results = _parse_results(_response_content(body), rubrics, cast(str, payload["response"]))
                weight_sum = sum(result["weight"] for result in results)
                score = sum(result["weight"] * result["score"] for result in results) / weight_sum
                invalid_evidence_count = sum("validation_error" in result for result in results)
                return {
                    "score": score,
                    "process_score": score,
                    "rubric_results": results,
                    "judge_status": "ok_with_invalid_evidence" if invalid_evidence_count else "ok",
                    "invalid_evidence_count": invalid_evidence_count,
                    "judge_model": self.model,
                    "judge_attempts": attempt + 1,
                }
            except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as error:
                last_error = error
                if attempt < self.max_retries and self.retry_backoff:
                    await asyncio.sleep(self.retry_backoff * 2**attempt)
        raise RuntimeError(f"Rubric judge failed after {self.max_retries + 1} attempts: {last_error}") from last_error

    def _build_request(self, payload: dict[str, Any], rubrics: list[dict[str, Any]]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Original user question:\n" + (payload.get("question") or "[no question text]"),
            }
        ]
        content.extend(
            {"type": "image_url", "image_url": {"url": _image_data_uri(source)}}
            for source in payload.get("image_sources", [])
        )
        content.append(
            {
                "type": "text",
                "text": (
                    "Assistant response to evaluate:\n"
                    + cast(str, payload["response"])
                    + "\nRubrics:\n"
                    + json.dumps(rubrics, ensure_ascii=False)
                ),
            }
        )
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"reasoning_effort": "low", "clear_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "rubric_judgment",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "minItems": len(rubrics),
                                "maxItems": len(rubrics),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "local_id": {
                                            "type": "string",
                                            "enum": [rubric["local_id"] for rubric in rubrics],
                                        },
                                        "verdict": {"type": "string", "enum": ["met", "unmet"]},
                                        "evidence": {"type": "string"},
                                    },
                                    "required": ["local_id", "verdict", "evidence"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["results"],
                        "additionalProperties": False,
                    },
                },
            },
        }

    async def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return await self._transport(request)
        headers = {}
        api_key = os.environ.get("RUBRIC_JUDGE_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.post(self.endpoint, json=request, headers=headers)
            response.raise_for_status()
            body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Judge HTTP response must be a JSON object.")
        return body


class RubricJudgerConfig(JudgerConfig):
    """Build a rubric judger while retaining the existing CPU replica
    options."""

    endpoint: str
    model: str
    max_retries: int = Field(default=2, ge=0)
    retry_backoff: float = Field(default=1.0, ge=0)
    max_tokens: int = Field(default=4096, gt=0)
    request_timeout: float = Field(default=120.0, gt=0)

    def build_local(self) -> RubricJudger:
        return RubricJudger(
            endpoint=self.endpoint,
            model=self.model,
            judger_name=self.judger_name,
            request_timeout=self.request_timeout,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
            max_tokens=self.max_tokens,
        )

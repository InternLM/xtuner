from __future__ import annotations

import asyncio
import base64
import json
import math
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, cast

import httpx
from pydantic import Field

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.utils.logger import get_logger

from .native import Judger, JudgerConfig, JudgerOutput, JudgerOutputBatch, JudgerPayloadBatch


logger = get_logger()


RubricTransport = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
FailurePolicy = Literal["raise", "zero"]


class _JudgeModelOutputError(ValueError):
    pass


# evidence 只需引用证明评分点的最短原文片段；上限与请求 schema 中的 maxLength 保持一致。
_MAX_EVIDENCE_CHARS = 256
_DEFAULT_MAX_RUBRICS_PER_REQUEST = 8
_POSITIVE_RUBRIC_TYPES = frozenset({"visual_grounding", "answer_coverage"})
_SUPPORTED_RUBRIC_TYPES = frozenset({"visual_grounding", "answer_coverage", "dodged_bullet"})


def _build_system_prompt(max_evidence_chars: int = _MAX_EVIDENCE_CHARS) -> str:
    unmet_cap = min(200, max_evidence_chars)
    return (
        "You are a strict multimodal rubric judge.\n"
        "Evaluate the assistant response against every supplied rubric independently.\n"
        "Use the original question and images only to verify the response's visual claims.\n"
        "Never use the question or images to supply reasoning, facts, or steps missing from the response.\n"
        "Return only one JSON object with this schema:\n"
        '{"results":[{"local_id":"a1","verdict":"met","evidence":"an exact contiguous quote from the response"}]}\n'
        'The verdict must be either "met" or "unmet". Include every rubric exactly once.\n'
        f"Keep evidence short: at most {max_evidence_chars} characters and never more than 50 words.\n"
        'For positive criteria (type "visual_grounding" or "answer_coverage"):\n'
        '- A criterion is "met" only when the assistant response itself explicitly states or demonstrates it.\n'
        '- For "met", evidence must be a verbatim contiguous substring of the assistant response proving the '
        'criterion (never use ellipsis, paraphrase, or phrases such as "The response identifies", and never '
        "copy the criterion into evidence).\n"
        f'- For "unmet", give one short sentence (<={unmet_cap} characters) stating only what is missing or '
        "incorrect without quoting long passages.\n"
        'For negative pitfall criteria (type "dodged_bullet", e.g. "The response should avoid ..."):\n'
        '- Mark "met" when the response avoids the prohibited error (including when the topic is not mentioned); '
        'set evidence to a brief note such as "Avoided." (verbatim quote not required).\n'
        '- Mark "unmet" only when the response actively commits the prohibited error; '
        f"briefly cite or state the offending claim in one short sentence (<={unmet_cap} characters). "
        "Omitting a topic never violates a dodged_bullet.\n"
        "Do not invent criteria, omit criteria, or reveal hidden reference answers.\n"
    )


_SYSTEM_PROMPT = _build_system_prompt(_MAX_EVIDENCE_CHARS)


def _validate_rubrics(raw_rubrics: Any, total_weight: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rubrics, list) or not raw_rubrics:
        raise ValueError("Rubric judging requires a non-empty extra_info.rubrics list.")
    rubrics: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    all_weight_sum = 0.0
    pos_weight_sum = 0.0
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
        if kind not in _SUPPORTED_RUBRIC_TYPES:
            raise ValueError(f"Rubric {local_id!r} has unsupported type {kind!r}.")
        if isinstance(weight, bool) or not isinstance(weight, int | float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Rubric {local_id!r} requires a finite positive weight.")
        seen_ids.add(local_id)
        fweight = float(weight)
        all_weight_sum += fweight
        if kind in _POSITIVE_RUBRIC_TYPES:
            pos_weight_sum += fweight
        rubrics.append(
            {
                "local_id": local_id,
                "type": kind,
                "criterion": criterion,
                "weight": fweight,
            }
        )
    if pos_weight_sum <= 0.0:
        raise ValueError("Rubric judging requires at least one positive rubric.")
    if total_weight is not None:
        if (
            isinstance(total_weight, bool)
            or not isinstance(total_weight, int | float)
            or not math.isfinite(total_weight)
            or not (
                math.isclose(float(total_weight), pos_weight_sum, rel_tol=1e-6, abs_tol=1e-6)
                or math.isclose(float(total_weight), all_weight_sum, rel_tol=1e-6, abs_tol=1e-6)
            )
        ):
            raise ValueError(
                f"total_weight {total_weight!r} does not match rubric weight sum "
                f"(positive={pos_weight_sum}, all={all_weight_sum})."
            )
    return rubrics


def _question_text(messages: list[dict[str, Any]]) -> str:
    # 只取首个 user 消息的文本。agentic RL 的多轮轨迹会把工具输出、观察结果等
    # 追加为后续 user/observation 消息，全部拼接会把轨迹内容污染进 judge 的
    # 判分上下文；首个 user 消息才是原始任务。reasoning RL 单轮场景行为不变。
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part["text"] for part in content if part.get("type") == "text" and isinstance(part.get("text"), str)
            )
        return ""
    return ""


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


def _collect_image_sources(extra: dict[str, Any], messages: list[dict[str, Any]]) -> list[str]:
    # 首轮图优先取 extra_fields["image_data"]（dataset 展开路径/data URI），否则回退解析 message；
    # 再追加 agent 多轮调用 get_page_image 写入 extra_fields["agent_tool_images"] 的页图，并保序去重。
    base_sources = list(extra.get("image_data") or []) or _message_image_sources(messages)
    raw_tool_images = extra.get("agent_tool_images") or []
    tool_sources = [item.get("image_data") if isinstance(item, dict) else item for item in raw_tool_images]
    seen: set[str] = set()
    deduped: list[str] = []
    for source in (*base_sources, *tool_sources):
        if isinstance(source, str) and source and source not in seen:
            seen.add(source)
            deduped.append(source)
    return deduped


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


def _response_metadata(body: dict[str, Any]) -> dict[str, Any]:
    finish_reason = None
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason")
    raw_usage = body.get("usage")
    usage = {}
    if isinstance(raw_usage, dict):
        usage = {
            key: raw_usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens") if key in raw_usage
        }
    return {"finish_reason": finish_reason, "content_chars": None, "usage": usage}


def _judge_error_type(error: Exception, metadata: dict[str, Any]) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "transport"
    if isinstance(error, httpx.HTTPStatusError):
        return "http"
    if isinstance(error, httpx.HTTPError):
        return "transport"
    if metadata.get("finish_reason") == "length":
        return "output_truncated"
    if isinstance(error, _JudgeModelOutputError):
        return "model_output"
    return "protocol"


def _dump_judge_failure(
    payload: dict[str, Any],
    diagnostics: dict[str, Any],
    error: Exception,
    body: dict[str, Any] | None,
) -> Path | None:
    # RUBRIC_JUDGE_DEBUG_DIR 设置后，重试耗尽的判分失败会把完整原始响应落盘，
    # 用于区分服务端返回的 content / reasoning_content 和诊断前缀之外的输出形态。
    # payload 只含图片路径，raw body 原样保存；目录由调用方保证不进入共享产物。
    debug_dir = os.environ.get("RUBRIC_JUDGE_DEBUG_DIR")
    if not debug_dir:
        return None
    try:
        path = Path(debug_dir)
        path.mkdir(parents=True, exist_ok=True)
        name = (
            f"group{diagnostics.get('group_id')}_rollout{diagnostics.get('rollout_id')}"
            f"_{diagnostics.get('error_type')}.json"
        )
        record = {
            "diagnostics": diagnostics,
            "error": repr(error),
            "payload": payload,
            "raw_response": body,
        }
        dump_path = path / name
        dump_path.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str))
    except OSError:
        logger.warning("Failed to dump rubric judge failure record.", exc_info=True)
        return None
    return dump_path


def _try_parse_top_level(content: str) -> dict[str, Any] | None:
    # strict=False 放宽字符串内控制字符：实测 judge 引用回复中的 LaTeX 时会输出
    # 字面 \x08 转义残留（\boldsymbol -> \x08oldsymbol），严格模式整份输出无法解析；
    # 放宽后证据无法逐字匹配回复，仍由 met 证据校验 fail-closed 为 unmet。
    try:
        top_level = json.loads(content, strict=False)
    except json.JSONDecodeError:
        return None
    if isinstance(top_level, list):
        return {"results": top_level}
    if isinstance(top_level, dict) and isinstance(top_level.get("results"), list):
        return top_level
    return None


def _try_parse_embedded_object(content: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder(strict=False)
    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "results" in candidate:
            return candidate
    return None


def _try_repair_missing_prefix(content: str) -> dict[str, Any] | None:
    # 长输入下实测会丢失 '{"results"' 标记的头部若干字符（甚至成对的外层花括号
    # 首尾各缺一个）。只认第一个匹配的截断点：对同一截断点依次尝试原样与补
    # 结尾 "}" 修复；raw_decode 允许尾随字符，故两式解析结果不会冲突。
    decoder = json.JSONDecoder(strict=False)
    stripped = content.lstrip()
    marker = '{"results"'
    for offset in range(1, len(marker)):
        if not stripped.startswith(marker[offset:]):
            continue
        for suffix in ("", "}"):
            try:
                candidate, _ = decoder.raw_decode(marker[:offset] + stripped + suffix)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and isinstance(candidate.get("results"), list):
                return candidate
        return None
    return None


def _try_parse_bare_array(content: str) -> dict[str, Any] | None:
    # 长输入下模型还可能在裸数组前输出垃圾前缀（实测出现 ":\n[...]"）；
    # 从首个 [ 起按完整裸数组解析，之后仍执行全部严格校验。
    decoder = json.JSONDecoder(strict=False)
    bracket = content.find("[")
    if bracket == -1:
        return None
    try:
        candidate, _ = decoder.raw_decode(content[bracket:])
    except json.JSONDecodeError:
        return None
    if isinstance(candidate, list):
        return {"results": candidate}
    return None


def _build_validated_result(
    raw: Any,
    expected: dict[str, dict[str, Any]],
    seen_ids: set[str],
    response: str,
    max_evidence_chars: int,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Each rubric result must be an object.")
    local_id = raw.get("local_id")
    verdict = raw.get("verdict")
    evidence = raw.get("evidence")
    if not isinstance(local_id, str):
        raise ValueError("Rubric result local_id must be a string.")
    if local_id not in expected:
        raise ValueError(f"Judge returned unknown rubric local_id {local_id!r}.")
    if local_id in seen_ids:
        raise ValueError(f"Judge returned duplicate rubric local_id {local_id!r}.")
    rubric = expected[local_id]
    if rubric["type"] == "dodged_bullet" and verdict == "violated":
        verdict = "unmet"
    if verdict not in {"met", "unmet"}:
        raise ValueError(f"Rubric {local_id!r} verdict must be met or unmet.")
    if not isinstance(evidence, str):
        raise ValueError(f"Rubric {local_id!r} evidence must be a string.")

    if rubric["type"] == "dodged_bullet":
        # dodged_bullet 只扣分不加分：met 表示未踩坑（0 分，回复天然不含错误原文，不做子串校验）；
        # unmet 表示踩坑违规（记 -1.0 * weight 扣分；整体性错误常需概括说明，翻成 met 会 fail-open 漏扣分）。
        return {
            **rubric,
            "verdict": verdict,
            "score": 0.0 if verdict == "met" else -1.0,
            "evidence": evidence,
        }

    result = {
        **rubric,
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
    elif verdict == "met" and len(evidence) > max_evidence_chars:
        # response_format 的 maxLength 只在 guided decoding 生效的后端有强制力；
        # 判分侧独立执行同一上限，防止超长证据耗尽后续样本的输出预算。
        result.update(
            judge_verdict="met",
            verdict="unmet",
            score=0.0,
            validation_error=f"met evidence exceeds {max_evidence_chars} characters",
        )
    return result


def _parse_results(
    content: str,
    rubrics: list[dict[str, Any]],
    response: str,
    max_evidence_chars: int = _MAX_EVIDENCE_CHARS,
) -> list[dict[str, Any]]:
    parsed = (
        _try_parse_top_level(content)
        or _try_parse_embedded_object(content)
        or _try_repair_missing_prefix(content)
        or _try_parse_bare_array(content)
    )
    if parsed is None or not isinstance(parsed.get("results"), list):
        raise _JudgeModelOutputError(
            f"Judge output must contain a JSON object with a results list; content prefix: {content[:200]!r}"
        )

    expected = {rubric["local_id"]: rubric for rubric in rubrics}
    results_by_id: dict[str, dict[str, Any]] = {}
    for raw in parsed["results"]:
        result = _build_validated_result(raw, expected, set(results_by_id), response, max_evidence_chars)
        results_by_id[result["local_id"]] = result
    missing = set(expected) - set(results_by_id)
    if missing:
        raise ValueError(f"Judge omitted rubric local_ids: {sorted(missing)}.")
    return [results_by_id[rubric["local_id"]] for rubric in rubrics]


def _extract_partial_array_items(content: str) -> list[dict[str, Any]]:
    # max_tokens 截断时外层 JSON 未闭合，但截断点之前的若干条目已是合法对象；
    # 定位 results 数组起始 "[" 后用 raw_decode 逐个回收已闭合的对象，遇到截断尾巴即停。
    key_pos = content.find('"results"')
    if key_pos != -1:
        bracket = content.find("[", key_pos + len('"results"'))
    else:
        bracket = content.find("[")
    if bracket == -1:
        return []
    decoder = json.JSONDecoder(strict=False)
    pos = bracket + 1
    length = len(content)
    items: list[dict[str, Any]] = []
    while pos < length:
        while pos < length and content[pos] in " \t\r\n,":
            pos += 1
        if pos >= length or content[pos] != "{":
            break
        try:
            item, next_pos = decoder.raw_decode(content, pos)
        except json.JSONDecodeError:
            break
        if not isinstance(item, dict):
            break
        items.append(item)
        pos = next_pos
    return items


def _recover_truncated_results(
    content: str,
    rubrics: list[dict[str, Any]],
    response: str,
    max_evidence_chars: int = _MAX_EVIDENCE_CHARS,
) -> list[dict[str, Any]]:
    expected = {rubric["local_id"]: rubric for rubric in rubrics}
    recovered: dict[str, dict[str, Any]] = {}
    for raw in _extract_partial_array_items(content):
        try:
            result = _build_validated_result(raw, expected, set(recovered), response, max_evidence_chars)
        except ValueError:
            break
        recovered[result["local_id"]] = result
    return [recovered[rubric["local_id"]] for rubric in rubrics if rubric["local_id"] in recovered]


def _aggregate_rubric_results(results: list[dict[str, Any]]) -> tuple[float, bool]:
    pos_results = [r for r in results if r["type"] in _POSITIVE_RUBRIC_TYPES]
    neg_results = [r for r in results if r["type"] == "dodged_bullet"]
    pos_weight_sum = sum(r["weight"] for r in pos_results)
    pos_earned = sum(r["weight"] * r["score"] for r in pos_results)
    neg_penalty = sum(r["weight"] for r in neg_results if r["verdict"] == "unmet")
    score = min(1.0, max(0.0, pos_earned - neg_penalty) / pos_weight_sum)
    all_pass = bool(results) and all(r["verdict"] == "met" for r in results)
    return score, all_pass


def _merge_usage(target: dict[str, int], source: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = source.get(key)
        if isinstance(val, int) and not isinstance(val, bool):
            target[key] = target.get(key, 0) + val


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
    if "all_pass" in process:
        rollout_state.reward["all_pass"] = process["all_pass"]
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
        max_evidence_chars: int = _MAX_EVIDENCE_CHARS,
        max_rubrics_per_request: int | None = _DEFAULT_MAX_RUBRICS_PER_REQUEST,
        on_failure: FailurePolicy = "raise",
        max_concurrent_requests: int = 16,
        chat_template_kwargs: dict[str, Any] | None = None,
        transport: RubricTransport | None = None,
    ) -> None:
        super().__init__(judger_name=judger_name)
        if not endpoint or not model:
            raise ValueError("RubricJudger requires endpoint and model.")
        if request_timeout <= 0 or retry_backoff < 0 or max_retries < 0 or max_tokens <= 0 or max_evidence_chars <= 0:
            raise ValueError("Invalid RubricJudger timeout, retry, max_tokens, or max_evidence_chars configuration.")
        if max_rubrics_per_request is not None and max_rubrics_per_request < 1:
            raise ValueError("RubricJudger max_rubrics_per_request must be >= 1 or None.")
        if on_failure not in {"raise", "zero"}:
            raise ValueError(f"RubricJudger on_failure must be 'raise' or 'zero', got {on_failure!r}.")
        if max_concurrent_requests < 1:
            raise ValueError("RubricJudger max_concurrent_requests must be >= 1.")
        self.endpoint = endpoint
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_tokens = max_tokens
        self.max_evidence_chars = max_evidence_chars
        self.max_rubrics_per_request = max_rubrics_per_request
        self.on_failure: FailurePolicy = on_failure
        self.max_concurrent_requests = max_concurrent_requests
        self._system_prompt = _build_system_prompt(max_evidence_chars)
        # chat_template_kwargs 面向 GLM 系 judge 服务（控制思考开销）；默认 None 表示
        # 沿用 GLM 值，传 {} 可整体关闭，传其他 dict 覆盖，便于接入非 GLM judge。
        self.chat_template_kwargs = (
            {"reasoning_effort": "low", "clear_thinking": False}
            if chat_template_kwargs is None
            else chat_template_kwargs
        )
        self._transport = transport
        self._semaphore: asyncio.Semaphore | None = None

    def preprocess(self, rollout_state: RolloutState) -> dict[str, Any]:
        extra = rollout_state.extra_fields or {}
        rubrics = _validate_rubrics(extra.get("rubrics"), extra.get("total_weight"))
        image_sources = _collect_image_sources(extra, rollout_state.message)
        if any(rubric["type"] == "visual_grounding" for rubric in rubrics) and not image_sources:
            raise ValueError("visual_grounding rubrics require at least one input image.")
        return {
            "response": rollout_state.response,
            "question": _question_text(rollout_state.message),
            "rubrics": rubrics,
            "image_sources": image_sources,
            "rollout_id": rollout_state.rollout_id,
            "group_id": rollout_state.group_id,
            "task_name": rollout_state.task_name,
        }

    def postprocess(self, rollout_state: RolloutState, output: JudgerOutput) -> RolloutState:
        rollout_state.reward = output
        if rollout_state.extra_fields is not None:
            rollout_state.extra_fields["all_pass"] = bool(output.get("all_pass", False))
            if output.get("judge_failed"):
                rollout_state.extra_fields["judge_failed"] = True
                rollout_state.extra_fields["error_type"] = output.get("error_type")
        return rollout_state

    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutputBatch:
        if isinstance(payload, list):
            # 惰性创建信号量：batch 并发打向同一 judge 端点需要上限，避免大 batch
            # 打满服务端；信号量必须与消费它的事件循环同生命周期，故不在 __init__ 建。
            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            return list(await asyncio.gather(*(self._score_bounded(item) for item in payload)))
        return await self._score(payload)

    async def _score_bounded(self, payload: dict[str, Any]) -> JudgerOutput:
        assert self._semaphore is not None
        async with self._semaphore:
            return await self._score(payload)

    async def _score(self, payload: dict[str, Any]) -> JudgerOutput:
        if payload.get("response") is None:
            raise ValueError("Rubric judging requires a detokenized response.")
        response_text = cast(str, payload["response"])
        rubrics = cast(list[dict[str, Any]], payload["rubrics"])
        # 图片读盘与 base64 编码只做一次并放 worker 线程，避免多 chunk/续判重复读盘或阻塞事件循环。
        image_uris = await asyncio.gather(
            *(asyncio.to_thread(_image_data_uri, source) for source in payload.get("image_sources", []))
        )
        chunk_size = self.max_rubrics_per_request or len(rubrics)
        chunks = [rubrics[i : i + chunk_size] for i in range(0, len(rubrics), chunk_size)]

        results_by_id: dict[str, dict[str, Any]] = {}
        agg_usage: dict[str, int] = {}
        total_attempts = 0
        last_finish_reason: str | None = None
        started_at = time.monotonic()

        for chunk in chunks:
            pending = list(chunk)
            while pending:
                request = await self._build_request(payload, pending, image_uris=image_uris)
                last_error: Exception | None = None
                last_metadata: dict[str, Any] = {}
                last_body: dict[str, Any] | None = None
                progressed = False
                for attempt in range(self.max_retries + 1):
                    metadata: dict[str, Any] = {}
                    total_attempts += 1
                    try:
                        body = await self._request(request)
                        last_body = body
                        metadata = _response_metadata(body)
                        content = _response_content(body)
                        metadata["content_chars"] = len(content)
                        try:
                            chunk_results = _parse_results(
                                content,
                                pending,
                                response_text,
                                max_evidence_chars=self.max_evidence_chars,
                            )
                        except (_JudgeModelOutputError, ValueError):
                            if metadata.get("finish_reason") == "length":
                                recovered = _recover_truncated_results(
                                    content,
                                    pending,
                                    response_text,
                                    max_evidence_chars=self.max_evidence_chars,
                                )
                                if recovered:
                                    for item in recovered:
                                        results_by_id[item["local_id"]] = item
                                    _merge_usage(agg_usage, metadata["usage"])
                                    last_finish_reason = metadata["finish_reason"]
                                    pending = [r for r in pending if r["local_id"] not in results_by_id]
                                    progressed = True
                                    break
                            raise
                        for item in chunk_results:
                            results_by_id[item["local_id"]] = item
                        _merge_usage(agg_usage, metadata["usage"])
                        last_finish_reason = metadata["finish_reason"]
                        pending = []
                        progressed = True
                        break
                    except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as error:
                        last_error = error
                        last_metadata = metadata
                        if attempt < self.max_retries and self.retry_backoff:
                            await asyncio.sleep(self.retry_backoff * 2**attempt)

                if not progressed:
                    return self._handle_failure(
                        payload=payload,
                        last_error=cast(Exception, last_error),
                        last_metadata=last_metadata,
                        last_body=last_body,
                        started_at=started_at,
                        total_attempts=total_attempts,
                    )

        ordered_results = [results_by_id[rubric["local_id"]] for rubric in rubrics]
        score, all_pass = _aggregate_rubric_results(ordered_results)
        invalid_evidence_count = sum("validation_error" in result for result in ordered_results)
        return {
            "score": score,
            # 与 score 同值；该键供 merge_outcome_and_process 等多分支融合消费，
            # 单分支路由场景属于冗余字段，仅为契约统一保留。
            "process_score": score,
            "all_pass": all_pass,
            "judge_failed": False,
            "rubric_results": ordered_results,
            "judge_status": "ok_with_invalid_evidence" if invalid_evidence_count else "ok",
            "invalid_evidence_count": invalid_evidence_count,
            "judge_model": self.model,
            "judge_attempts": total_attempts,
            "judge_finish_reason": last_finish_reason,
            "judge_usage": agg_usage,
            "judge_latency_seconds": time.monotonic() - started_at,
        }

    def _handle_failure(
        self,
        *,
        payload: dict[str, Any],
        last_error: Exception,
        last_metadata: dict[str, Any],
        last_body: dict[str, Any] | None,
        started_at: float,
        total_attempts: int,
    ) -> JudgerOutput:
        latency = time.monotonic() - started_at
        error_type = _judge_error_type(last_error, last_metadata)
        diagnostics = {
            "rollout_id": payload.get("rollout_id"),
            "group_id": payload.get("group_id"),
            "task_name": payload.get("task_name"),
            "error_type": error_type,
            "judge_latency_seconds": latency,
            **last_metadata,
        }
        debug_dump = _dump_judge_failure(payload, diagnostics, last_error, last_body)
        if debug_dump is not None:
            diagnostics["debug_dump"] = str(debug_dump)
        message = (
            f"Rubric judge failed after {self.max_retries + 1} attempts; "
            f"diagnostics={json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)}; error={last_error}"
        )
        if self.on_failure == "zero":
            logger.warning(f"{message} (falling back to score=0.0)")
            return {
                "score": 0.0,
                "process_score": 0.0,
                "all_pass": False,
                "judge_failed": True,
                "error_type": error_type,
                "judge_error": str(last_error),
                "rubric_results": [],
                "judge_status": "failed",
                "invalid_evidence_count": 0,
                "judge_model": self.model,
                "judge_attempts": total_attempts,
                "judge_finish_reason": last_metadata.get("finish_reason"),
                "judge_usage": last_metadata.get("usage", {}),
                "judge_latency_seconds": latency,
            }
        raise RuntimeError(message) from last_error

    async def _build_request(
        self,
        payload: dict[str, Any],
        rubrics: list[dict[str, Any]],
        image_uris: list[str] | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Original user question:\n" + (payload.get("question") or "[no question text]"),
            }
        ]
        if image_uris is None:
            image_uris = await asyncio.gather(
                *(asyncio.to_thread(_image_data_uri, source) for source in payload.get("image_sources", []))
            )
        content.extend({"type": "image_url", "image_url": {"url": uri}} for uri in image_uris)
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
        request = {
            "model": self.model,
            "messages": [{"role": "system", "content": self._system_prompt}, {"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
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
                                        # 证据只需证明评分点的最短原文片段；不限制长度会让 judge
                                        # 在长回复上逐段引用，耗尽 max_tokens 导致输出被截断。
                                        "evidence": {"type": "string", "maxLength": self.max_evidence_chars},
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
        if self.chat_template_kwargs:
            request["chat_template_kwargs"] = self.chat_template_kwargs
        return request

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
    max_evidence_chars: int = Field(default=_MAX_EVIDENCE_CHARS, gt=0)
    max_rubrics_per_request: int | None = Field(default=_DEFAULT_MAX_RUBRICS_PER_REQUEST, ge=1)
    on_failure: FailurePolicy = "raise"
    request_timeout: float = Field(default=120.0, gt=0)
    max_concurrent_requests: int = Field(default=16, ge=1)
    # GLM judge 的思考控制参数；默认 None 沿用 GLM 值，传 {} 关闭，传 dict 覆盖。
    chat_template_kwargs: dict[str, Any] | None = None

    def build_local(self) -> RubricJudger:
        return RubricJudger(
            endpoint=self.endpoint,
            model=self.model,
            judger_name=self.judger_name,
            request_timeout=self.request_timeout,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
            max_tokens=self.max_tokens,
            max_evidence_chars=self.max_evidence_chars,
            max_rubrics_per_request=self.max_rubrics_per_request,
            on_failure=self.on_failure,
            max_concurrent_requests=self.max_concurrent_requests,
            chat_template_kwargs=self.chat_template_kwargs,
        )

"""Rule judgers that score a math response against a reference answer.

DAPO, AIME and AMC compare integers. MATH-500 compares LaTeX. OlympiadBench compares symbolic
answers, including multi-answer sets and an explicit tolerance. Minerva is scored by CompassVerifier.
A correct answer scores ``+1``. An incorrect, unextractable or length-truncated answer scores ``-1``.
"""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from pydantic import Field

from xtuner.v1.data_proto.rl_data import RolloutState

from .native import Judger, JudgerConfig, JudgerOutput, JudgerOutputBatch, JudgerPayloadBatch


STATUS_OK = "ok"
STATUS_NO_ANSWER = "no_answer"
STATUS_PARSE_ERROR = "parse_error"
_FINISHED_REASONS = {"stop", "finished"}
_CORRECT_REWARD = 1.0
_INCORRECT_REWARD = -1.0
_COMPASS_TAIL_CHARS = 1500

_FRAC_COMMANDS = ("\\dfrac", "\\tfrac", "\\frac")
_UNIT_SUFFIX_RE = re.compile(r"(?<=[\d}\)])\s*\\(?:text|mbox|mathrm)\s*\{[^{}]*\}\s*$")
_CHOICE_PAREN_RE = re.compile(r"^\(\s*([A-Za-z])\s*\)$")
_VAR_PREFIX_RE = re.compile(r"^[A-Za-z]\s*(?:\\in|=|\\to)\s*(?=\S)")


@dataclass
class JudgeResult:
    """One rule-judge result.

    Args:
        correct (bool): Whether the extracted answer matches. Meaningful only when status is ``ok``.
        status (str): ``ok``, ``no_answer`` or ``parse_error``.
        extracted (str | None): Text extracted from the response.
        detail (str): Short comparison note.
    """

    correct: bool
    status: str
    extracted: str | None = None
    detail: str = ""


def extract_boxed(text: str | None) -> str | None:
    """Extract the last ``\\boxed{...}`` body, including nested braces.

    Args:
        text (str | None): Model response.

    Returns:
        str | None: Boxed body, or None when no box is present.
    """
    if not text:
        return None
    marker = "\\boxed"
    start = text.rfind(marker)
    if start < 0:
        return None
    index = start + len(marker)
    while index < len(text) and text[index] in " \t":
        index += 1
    if index >= len(text):
        return None
    if text[index] != "{":
        rest = text[index:].strip().split()
        return rest[0] if rest else None
    depth = 0
    out: list[str] = []
    for char in text[index:]:
        if char == "{":
            depth += 1
            if depth == 1:
                continue
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(char)
    return None


def extract_answer(text: str | None) -> str | None:
    """Extract an answer, preferring a box and otherwise the last ``Answer:``
    line.

    Args:
        text (str | None): Model response.

    Returns:
        str | None: Extracted answer text, or None.
    """
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    match = None
    for match in re.finditer(r"(?:final answer|answer)\s*(?:is)?\s*[:：]\s*(.+)", text or "", re.IGNORECASE):
        pass
    if match is None:
        return None
    tail = match.group(1).strip().split("\n")[0].strip()
    return tail.rstrip(".。").strip() or None


def _read_latex_group(text: str, index: int) -> tuple[str | None, int]:
    while index < len(text) and text[index] == " ":
        index += 1
    if index >= len(text):
        return None, index
    if text[index] != "{":
        return text[index], index + 1
    depth = 0
    for pos in range(index, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[index + 1 : pos], pos + 1
    return None, index


def _normalize_frac(text: str) -> str:
    out = text
    for command in _FRAC_COMMANDS:
        pos = 0
        while True:
            start = out.find(command, pos)
            if start < 0:
                break
            cursor = start + len(command)
            numerator, cursor = _read_latex_group(out, cursor)
            denominator, cursor = _read_latex_group(out, cursor)
            if numerator is None or denominator is None:
                pos = start + len(command)
                continue
            replacement = f"\\frac{{{_normalize_frac(numerator)}}}{{{_normalize_frac(denominator)}}}"
            out = out[:start] + replacement + out[cursor:]
            pos = start + len(replacement)
    return out


def _strip_latex_wrappers(text: str) -> str:
    out = text.strip()
    out = re.sub(r"^\$+|\$+$", "", out).strip()
    out = out.replace("\\$", "")
    out = re.sub(r"^\\\[|\\\]$", "", out).strip()
    out = re.sub(r"^\\\(|\\\)$", "", out).strip()
    out = _normalize_frac(out)
    out = _UNIT_SUFFIX_RE.sub("", out).strip()
    for pattern in (r"\\left", r"\\right", r"\\!", r"\\,", r"\\;", r"\\ "):
        out = re.sub(pattern, "", out)
    out = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\mbox\s*\{([^{}]*)\}", r"\1", out)
    out = out.replace("\\%", "").replace("%", "")
    out = out.replace("^{\\circ}", "").replace("^\\circ", "")
    out = _VAR_PREFIX_RE.sub("", out).strip()
    out = _CHOICE_PAREN_RE.sub(r"\1", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _normalize_number_text(text: str) -> str:
    out = _strip_latex_wrappers(text)
    out = out.replace(",", "").replace(" ", "")
    out = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", out)
    out = out.replace("\\times10^", "e").replace("\\cdot10^", "e")
    out = re.sub(r"\\times\s*10\^\{([^{}]+)\}", r"e\1", out)
    out = re.sub(r"10\^\{([^{}]+)\}", r"1e\1", out)
    return out.rstrip(".")


def parse_integer(text: str | None) -> int | None:
    """Parse an integer answer, allowing LaTeX wrappers, thousands separators
    and ``x.0``.

    Args:
        text (str | None): Answer text.

    Returns:
        int | None: Parsed integer, or None.
    """
    if text is None:
        return None
    norm = _normalize_number_text(text)
    if re.fullmatch(r"[+-]?\d+", norm):
        return int(norm)
    if re.fullmatch(r"[+-]?\d+\.0*", norm):
        return int(norm.split(".")[0])
    match = re.fullmatch(r"\(([+-]?\d+)\)/\(([+-]?\d+)\)", norm)
    if match:
        numerator, denominator = int(match.group(1)), int(match.group(2))
        if denominator != 0 and numerator % denominator == 0:
            return numerator // denominator
    return None


def judge_integer(response: str, ground_truth: str) -> JudgeResult:
    """Score an integer answer for DAPO, AIME and AMC.

    Args:
        response (str): Model response.
        ground_truth (str): Reference answer.

    Returns:
        JudgeResult: Comparison result.
    """
    gold = parse_integer(ground_truth)
    if gold is None:
        return JudgeResult(False, STATUS_PARSE_ERROR, None, f"bad ground_truth: {ground_truth!r}")
    raw = extract_answer(response)
    if raw is None:
        return JudgeResult(False, STATUS_NO_ANSWER, None, "no boxed or Answer: found")
    got = parse_integer(raw)
    if got is None:
        return JudgeResult(False, STATUS_PARSE_ERROR, raw, "cannot parse as integer")
    return JudgeResult(got == gold, STATUS_OK, raw)


def _to_fraction(text: str) -> Fraction | None:
    norm = _normalize_number_text(text)
    try:
        return Fraction(norm)
    except (ValueError, ZeroDivisionError):
        pass
    match = re.fullmatch(r"\(([^()]+)\)/\(([^()]+)\)", norm)
    if match:
        try:
            return Fraction(match.group(1)) / Fraction(match.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return Fraction(float(norm))
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


def _sympy_equal(left: str, right: str) -> bool | None:
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
    except ImportError:
        return None
    try:
        diff = simplify(parse_latex(left) - parse_latex(right))
        return bool(diff == 0)
    except Exception:
        return None


def judge_latex(response: str, ground_truth: str) -> JudgeResult:
    """Score a LaTeX answer for MATH-500.

    Args:
        response (str): Model response.
        ground_truth (str): Reference answer.

    Returns:
        JudgeResult: Comparison result.
    """
    raw = extract_answer(response)
    if raw is None:
        return JudgeResult(False, STATUS_NO_ANSWER, None, "no boxed or Answer: found")

    gold_frac, got_frac = _to_fraction(ground_truth), _to_fraction(raw)
    if gold_frac is not None and got_frac is not None:
        return JudgeResult(gold_frac == got_frac, STATUS_OK, raw, "numeric")

    gold_norm, got_norm = _strip_latex_wrappers(ground_truth), _strip_latex_wrappers(raw)
    if gold_norm.replace(" ", "") == got_norm.replace(" ", ""):
        return JudgeResult(True, STATUS_OK, raw, "string")

    symbolic = _sympy_equal(gold_norm, got_norm)
    if symbolic is None:
        return JudgeResult(False, STATUS_PARSE_ERROR, raw, "sympy unavailable or failed")
    return JudgeResult(symbolic, STATUS_OK, raw, "sympy")


def _split_multi_answer(text: str) -> list[str]:
    parts = re.split(r"\s*(?:,|;|\\quad|\\qquad|\sor\s|、)\s*", text)
    return [part.strip() for part in parts if part.strip()]


def judge_symbolic_single(candidate: str, gold: str) -> JudgeResult:
    """Compare one symbolic answer fragment.

    Args:
        candidate (str): Predicted fragment.
        gold (str): Reference fragment.

    Returns:
        JudgeResult: Comparison result. ``extracted`` is the candidate.
    """
    gold_frac, got_frac = _to_fraction(gold), _to_fraction(candidate)
    if gold_frac is not None and got_frac is not None:
        return JudgeResult(gold_frac == got_frac, STATUS_OK, candidate, "numeric")

    gold_norm, got_norm = _strip_latex_wrappers(gold), _strip_latex_wrappers(candidate)
    if gold_norm.replace(" ", "") == got_norm.replace(" ", ""):
        return JudgeResult(True, STATUS_OK, candidate, "string")

    symbolic = _sympy_equal(gold_norm, got_norm)
    if symbolic is None:
        return JudgeResult(False, STATUS_PARSE_ERROR, candidate, "sympy unavailable or failed")
    return JudgeResult(symbolic, STATUS_OK, candidate, "sympy")


def judge_symbolic(
    response: str,
    ground_truth: str,
    answer_type: str | None = None,
    is_multiple_answer: bool = False,
    error: str | None = None,
) -> JudgeResult:
    """Score a symbolic answer for OlympiadBench.

    Args:
        response (str): Model response.
        ground_truth (str): Reference answer.
        answer_type (str | None): Source answer type, recorded in the detail string.
        is_multiple_answer (bool): Whether the reference is an unordered set of answers.
        error (str | None): Absolute tolerance, such as ``1e-1``.

    Returns:
        JudgeResult: Comparison result.
    """
    raw = extract_answer(response)
    if raw is None:
        return JudgeResult(False, STATUS_NO_ANSWER, None, "no boxed or Answer: found")

    tolerance = None
    if error:
        try:
            tolerance = abs(float(error))
        except ValueError:
            tolerance = None

    if tolerance is not None:
        gold_frac, got_frac = _to_fraction(ground_truth), _to_fraction(raw)
        if gold_frac is not None and got_frac is not None:
            return JudgeResult(abs(float(gold_frac) - float(got_frac)) <= tolerance, STATUS_OK, raw, "tolerance")

    if is_multiple_answer:
        gold_parts, got_parts = _split_multi_answer(ground_truth), _split_multi_answer(raw)
        if len(gold_parts) != len(got_parts):
            return JudgeResult(False, STATUS_OK, raw, f"multi-answer count {len(got_parts)} != {len(gold_parts)}")
        remaining = list(got_parts)
        for gold_part in gold_parts:
            match_index = None
            for index, got_part in enumerate(remaining):
                if judge_symbolic_single(got_part, gold_part).correct:
                    match_index = index
                    break
            if match_index is None:
                return JudgeResult(False, STATUS_OK, raw, "multi-answer mismatch")
            remaining.pop(match_index)
        return JudgeResult(True, STATUS_OK, raw, "multi-answer")

    result = judge_symbolic_single(raw, ground_truth)
    return JudgeResult(result.correct, result.status, raw, f"{answer_type or 'unknown'}/{result.detail}")


def _response_text(response: str | None) -> str:
    text = (response or "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
    text = text.replace("<｜end▁of▁sentence｜>", "")
    return text.split("</think>")[-1].strip()


def _question_text(message: list[dict[str, Any]] | None) -> str:
    if not message:
        return ""
    content = message[-1].get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _reward_from_result(result: JudgeResult, kind: str) -> dict[str, Any]:
    correct = result.status == STATUS_OK and result.correct
    return {
        "score": _CORRECT_REWARD if correct else _INCORRECT_REWARD,
        "acc": correct,
        "judge_status": result.status,
        "judge_kind": kind,
        "extracted": result.extracted,
        "no_answer": result.status == STATUS_NO_ANSWER,
        "parse_error": result.status == STATUS_PARSE_ERROR,
        "judge_detail": result.detail,
    }


class _LazyCompass:
    """Connect to CompassVerifier on the first request."""

    def __init__(self, hosts: list[str], request_timeout: float, max_retries: int) -> None:
        self.hosts = hosts
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self._model_name: str | None = None

    async def score(self, question: str, response: str, ground_truth: str) -> float:
        """Ask CompassVerifier whether the response matches the reference.

        Args:
            question (str): User question.
            response (str): Candidate answer text.
            ground_truth (str): Reference answer.

        Returns:
            float: ``1`` when the verifier returns ``A``, otherwise ``-1``.
        """
        import aiohttp

        from xtuner.v1.rl.judger.compass_verifier_v2 import verify_prompt

        if self._model_name is None:
            import requests

            self._model_name = requests.get(
                f"http://{self.hosts[0]}/v1/models",
                headers={"Authorization": "Bearer "},
                timeout=self.request_timeout,
            ).json()["data"][0]["id"]
        prompt = verify_prompt.format(question=question, llm_response=response, gold_answer=ground_truth)
        data = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            host = random.choice(self.hosts)
            url = f"http://{host}/v1/chat/completions"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                    ) as response_http:
                        body = await response_http.json()
                        if response_http.status != 200:
                            message = body.get("error", {}).get("message", "Unknown error")
                            raise RuntimeError(f"API request failed with status {response_http.status}: {message}")
                        verdict = body["choices"][0]["message"]["content"]
                        return 1.0 if str(verdict).strip() == "A" else -1.0
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(1)
        raise RuntimeError(f"Cannot connect to judger service: {last_error}")


class MathRuleJudger(Judger):
    """Score one rollout with integer, LaTeX or symbolic comparison.

    Args:
        kind (str): ``integer``, ``latex`` or ``symbolic``.
        judger_name (str): Name recorded on the judger.
        compass_hosts (list[str] | None): When set, a ``no_answer`` result is sent to CompassVerifier.
        compass_request_timeout (float): Compass request timeout in seconds.
        compass_max_retries (int): Compass attempts before keeping the rule score.
    """

    def __init__(
        self,
        kind: Literal["integer", "latex", "symbolic"],
        judger_name: str = "math_rule",
        compass_hosts: list[str] | None = None,
        compass_request_timeout: float = 60.0,
        compass_max_retries: int = 3,
    ) -> None:
        super().__init__(judger_name=judger_name)
        if kind not in {"integer", "latex", "symbolic"}:
            raise ValueError(f"unknown judge kind: {kind}")
        self.kind = kind
        self._compass = None
        if compass_hosts:
            self._compass = _LazyCompass(compass_hosts, compass_request_timeout, compass_max_retries)

    def preprocess(self, rollout_state: RolloutState) -> dict[str, Any]:
        extra = rollout_state.extra_fields or {}
        return {
            "response": rollout_state.response,
            "label": rollout_state.reward_model.get("ground_truth") if rollout_state.reward_model else None,
            "finish_reason": rollout_state.finish_reason,
            "message": rollout_state.message,
            "answer_type": extra.get("answer_type"),
            "is_multiple_answer": bool(extra.get("is_multiple_answer", False)),
            "error": extra.get("error"),
        }

    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutputBatch:
        if isinstance(payload, list):
            return [await self._score(item) for item in payload]
        return await self._score(payload)

    async def _score(self, payload: dict[str, Any]) -> JudgerOutput:
        if payload.get("finish_reason") not in _FINISHED_REASONS:
            return {
                "score": _INCORRECT_REWARD,
                "acc": False,
                "judge_status": "truncated",
                "judge_kind": self.kind,
                "finish_reason": payload.get("finish_reason"),
            }
        response = _response_text(payload.get("response"))
        ground_truth = "" if payload.get("label") is None else str(payload["label"])
        result = self._judge_text(response, ground_truth, payload)
        reward = _reward_from_result(result, self.kind)
        if self._compass is None or result.status != STATUS_NO_ANSWER:
            return reward
        question = _question_text(payload.get("message"))
        tail = response[-_COMPASS_TAIL_CHARS:]
        try:
            verdict = await self._compass.score(question, tail, ground_truth)
        except Exception as exc:
            return {**reward, "compass_fallback": "error", "compass_error": str(exc)[:200]}
        correct = verdict > 0
        return {
            **reward,
            "score": _CORRECT_REWARD if correct else _INCORRECT_REWARD,
            "acc": correct,
            "judge_status": "ok_compass" if correct else result.status,
            "compass_fallback": "hit" if correct else "miss",
        }

    def _judge_text(self, response: str, ground_truth: str, payload: dict[str, Any]) -> JudgeResult:
        if self.kind == "integer":
            return judge_integer(response, ground_truth)
        if self.kind == "latex":
            return judge_latex(response, ground_truth)
        return judge_symbolic(
            response,
            ground_truth,
            answer_type=payload.get("answer_type"),
            is_multiple_answer=bool(payload.get("is_multiple_answer", False)),
            error=None if payload.get("error") is None else str(payload.get("error")),
        )


class MathCompassJudger(Judger):
    """Score Minerva with CompassVerifier. Length-truncated samples score
    ``-1`` without a request.

    Args:
        hosts (list[str]): CompassVerifier ``host:port`` addresses.
        judger_name (str): Name recorded on the judger.
        request_timeout (float): Request timeout in seconds.
        max_retries (int): Attempts before scoring ``-1``.
    """

    def __init__(
        self,
        hosts: list[str],
        judger_name: str = "math_compass",
        request_timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(judger_name=judger_name)
        if not hosts:
            raise ValueError("MathCompassJudger requires at least one host.")
        self._compass = _LazyCompass(hosts, request_timeout, max_retries)

    def preprocess(self, rollout_state: RolloutState) -> dict[str, Any]:
        return {
            "response": rollout_state.response,
            "label": rollout_state.reward_model.get("ground_truth") if rollout_state.reward_model else None,
            "finish_reason": rollout_state.finish_reason,
            "message": rollout_state.message,
        }

    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutputBatch:
        if isinstance(payload, list):
            return [await self._score(item) for item in payload]
        return await self._score(payload)

    async def _score(self, payload: dict[str, Any]) -> JudgerOutput:
        if payload.get("finish_reason") not in _FINISHED_REASONS:
            return {
                "score": _INCORRECT_REWARD,
                "acc": False,
                "judge_status": "truncated",
                "judge_kind": "compass",
                "finish_reason": payload.get("finish_reason"),
            }
        question = _question_text(payload.get("message"))
        response = _response_text(payload.get("response"))[-_COMPASS_TAIL_CHARS:]
        ground_truth = "" if payload.get("label") is None else str(payload["label"])
        try:
            verdict = await self._compass.score(question, response, ground_truth)
        except Exception as exc:
            return {
                "score": _INCORRECT_REWARD,
                "acc": False,
                "judge_status": "compass_error",
                "judge_kind": "compass",
                "compass_error": str(exc)[:200],
            }
        correct = verdict > 0
        return {
            "score": _CORRECT_REWARD if correct else _INCORRECT_REWARD,
            "acc": correct,
            "judge_status": STATUS_OK,
            "judge_kind": "compass",
        }


class MathRuleJudgerConfig(JudgerConfig):
    """Configuration for one baseline math rule branch.

    Args:
        kind (str): ``integer``, ``latex`` or ``symbolic``.
        compass_hosts (list[str]): Hosts used only when the rule extractor finds no answer.
        compass_request_timeout (float): Compass request timeout in seconds.
        compass_max_retries (int): Compass attempts before keeping the rule score.
    """

    kind: Literal["integer", "latex", "symbolic"]
    compass_hosts: list[str] = Field(default_factory=list)
    compass_request_timeout: float = 60.0
    compass_max_retries: int = 3

    def build_local(self) -> Judger:
        return MathRuleJudger(
            kind=self.kind,
            judger_name=self.judger_name,
            compass_hosts=list(self.compass_hosts),
            compass_request_timeout=self.compass_request_timeout,
            compass_max_retries=self.compass_max_retries,
        )


class MathCompassJudgerConfig(JudgerConfig):
    """Configuration for the Minerva CompassVerifier branch.

    Args:
        compass_hosts (list[str]): CompassVerifier ``host:port`` addresses.
        compass_request_timeout (float): Request timeout in seconds.
        compass_max_retries (int): Attempts before scoring ``-1``.
    """

    compass_hosts: list[str]
    compass_request_timeout: float = 60.0
    compass_max_retries: int = 3

    def build_local(self) -> Judger:
        return MathCompassJudger(
            hosts=list(self.compass_hosts),
            judger_name=self.judger_name,
            request_timeout=self.compass_request_timeout,
            max_retries=self.compass_max_retries,
        )

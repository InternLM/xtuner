from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.trace import set_trace_attributes, set_trace_error, trace_event


TRACE_ATTR_JUDGER_BATCH_SIZE = "judger.batch_size"
TRACE_ATTR_JUDGER_NAME = "judger.name"
TRACE_ATTR_ROLLOUT_BACKEND = "rollout.backend"
TRACE_ATTR_PROMPT_TOKENS = "prompt.tokens"
TRACE_ATTR_COMPLETION_TOKENS = "completion.tokens"
TRACE_ATTR_XTUNER_ROLLOUT_ID = "xtuner.rollout_id"
TRACE_ATTR_XTUNER_GROUP_ID = "xtuner.group_id"
TRACE_ATTR_XTUNER_TASK_NAME = "xtuner.task_name"
TRACE_ATTR_XTUNER_SESSION_ID = "xtuner.session_id"
TRACE_ATTR_XTUNER_STATUS = "xtuner.status"
TRACE_ATTR_XTUNER_FINISH_REASON = "xtuner.finish_reason"
TRACE_ATTR_XTUNER_SEQ_STALENESS = "xtuner.seq_staleness"
TRACE_ATTR_XTUNER_ERROR_MSG = "xtuner.error_msg"
TRACE_ATTR_XTUNER_PRODUCER_FUTURE_STEP = "xtuner.producer_future_step"
TRACE_ATTR_XTUNER_GLOBAL_BATCH_SIZE = "xtuner.global_batch_size"
TRACE_ATTR_XTUNER_PROMPT_REPEAT_K = "xtuner.prompt_repeat_k"
TRACE_ATTR_XTUNER_EXPECTED_GROUPS = "xtuner.expected_groups"
TRACE_ATTR_XTUNER_EXPECTED_SAMPLES = "xtuner.expected_samples"
TRACE_ATTR_XTUNER_STALE = "xtuner.stale"

TRACE_ATTR_REWARD_SCORE = "reward.score"
TRACE_ATTR_REWARD_PASS = "reward.pass"
TRACE_ATTR_FILTER_DECISION = "filter.decision"
TRACE_ATTR_FILTER_REASON = "filter.reason"
TRACE_ATTR_TRAIN_INCLUDED = "train.included"
TRACE_ATTR_OVERSAMPLE_SOURCE = "oversample.source"
TRACE_ATTR_DROP_REASON = "drop.reason"

TRACE_EVENT_ROLLOUT_STATUS = "rollout.status"

TRACE_SPAN_ROLLOUT_BATCH = "rollout.batch"
TRACE_SPAN_ROLLOUT_CONTROLLER_GENERATE = "rollout_controller.generate"
TRACE_SPAN_ROLLOUT_WORKER_GENERATE = "rollout_worker.generate"
TRACE_SPAN_AGENT_LOOP_RUN = "agent_loop.run"
TraceAttributes = Mapping[str, Any]


_ERROR_STATUS_VALUES = {"error", "exception", "failed", "timeout", "timed_out"}
_PASSTHROUGH_EXTRA_FIELD_KEYS = (
    TRACE_ATTR_XTUNER_GLOBAL_BATCH_SIZE,
    TRACE_ATTR_XTUNER_PROMPT_REPEAT_K,
    TRACE_ATTR_XTUNER_EXPECTED_GROUPS,
    TRACE_ATTR_XTUNER_EXPECTED_SAMPLES,
    TRACE_ATTR_XTUNER_STALE,
    TRACE_ATTR_FILTER_DECISION,
    TRACE_ATTR_FILTER_REASON,
    TRACE_ATTR_TRAIN_INCLUDED,
    TRACE_ATTR_OVERSAMPLE_SOURCE,
    TRACE_ATTR_DROP_REASON,
)


def rollout_state_initial_attributes(
    rollout_state: RolloutState,
    *,
    task_name: str | None = None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    _set_if_not_none(attributes, TRACE_ATTR_XTUNER_ROLLOUT_ID, rollout_state.rollout_id)
    _set_if_not_none(attributes, TRACE_ATTR_XTUNER_GROUP_ID, rollout_state.group_id)
    _set_if_not_none(attributes, TRACE_ATTR_XTUNER_SESSION_ID, rollout_state.session_id)
    _set_if_not_none(attributes, TRACE_ATTR_XTUNER_TASK_NAME, task_name or rollout_state.task_name)
    extra_fields = getattr(rollout_state, "extra_fields", None)
    if isinstance(extra_fields, Mapping):
        producer_future_step = extra_fields.get("producer_future_step")
        if producer_future_step is None:
            producer_future_step = extra_fields.get("train_step")
        _set_if_not_none(attributes, TRACE_ATTR_XTUNER_PRODUCER_FUTURE_STEP, producer_future_step)
        _copy_passthrough_extra_fields(attributes, extra_fields)
    attributes[TRACE_ATTR_XTUNER_STATUS] = _status_value(rollout_state.status)
    attributes[TRACE_ATTR_XTUNER_SEQ_STALENESS] = rollout_state.seq_staleness
    if rollout_state.prompt_ids is not None:
        attributes[TRACE_ATTR_PROMPT_TOKENS] = len(rollout_state.prompt_ids)
    return attributes


def rollout_state_final_attributes(
    rollout_state: RolloutState,
    *,
    task_name: str | None = None,
) -> dict[str, Any]:
    attributes = rollout_state_initial_attributes(rollout_state, task_name=task_name)
    _set_if_not_none(attributes, TRACE_ATTR_XTUNER_FINISH_REASON, rollout_state.finish_reason)
    _set_if_not_none(attributes, TRACE_ATTR_XTUNER_ERROR_MSG, rollout_state.error_msg)
    if rollout_state.response_ids is not None:
        attributes[TRACE_ATTR_COMPLETION_TOKENS] = len(rollout_state.response_ids)
    attributes.update(_reward_payload_trace_attributes(rollout_state.reward))
    return attributes


def rollout_state_trace_attributes(
    rollout_state: RolloutState,
    *,
    task_name: str | None = None,
) -> dict[str, Any]:
    return rollout_state_final_attributes(rollout_state, task_name=task_name)


def judger_trace_attributes(
    judger: Any,
    rollout_state: RolloutState | list[RolloutState],
) -> dict[str, Any]:
    attributes: dict[str, Any] = {TRACE_ATTR_JUDGER_NAME: judger.get_judger_name()}
    if isinstance(rollout_state, list):
        attributes[TRACE_ATTR_JUDGER_BATCH_SIZE] = len(rollout_state)
        return attributes
    attributes.update(rollout_state_final_attributes(rollout_state))
    return attributes


def reward_trace_attributes(score: Any, *, passed: Any | None = None) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    normalized_score = _numeric_score(score)
    if normalized_score is not None:
        attributes[TRACE_ATTR_REWARD_SCORE] = normalized_score
        attributes["reward_score"] = normalized_score
        if passed is None:
            passed = normalized_score > 0
    normalized_pass = _bool_value(passed)
    if normalized_pass is not None:
        attributes[TRACE_ATTR_REWARD_PASS] = normalized_pass
    return attributes


def add_rollout_status_event(
    rollout_state: RolloutState,
    *,
    task_name: str | None = None,
) -> None:
    trace_event(TRACE_EVENT_ROLLOUT_STATUS, rollout_state_final_attributes(rollout_state, task_name=task_name))


def record_rollout_state_result(
    rollout_state: Any,
    *,
    task_name: str | None = None,
) -> None:
    if not isinstance(rollout_state, RolloutState):
        return
    set_trace_attributes(rollout_state_final_attributes(rollout_state, task_name=task_name))
    add_rollout_status_event(rollout_state, task_name=task_name)
    if _is_error_status(rollout_state.status):
        set_trace_error(_rollout_error_message(rollout_state))


def _reward_payload_trace_attributes(reward: Any) -> dict[str, Any]:
    if not isinstance(reward, Mapping):
        return {}
    score = reward.get("score")
    passed = reward.get("pass", reward.get("passed"))
    return reward_trace_attributes(score, passed=passed)


def _numeric_score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "pass", "passed"}:
        return True
    if text in {"0", "false", "no", "n", "off", "fail", "failed"}:
        return False
    return None


def _status_value(status: Status | Any) -> str:
    if isinstance(status, Status):
        return status.value
    return str(getattr(status, "value", status))


def _is_error_status(status: Any) -> bool:
    return _status_value(status).strip().lower() in _ERROR_STATUS_VALUES


def _rollout_error_message(rollout_state: RolloutState) -> str:
    if rollout_state.error_msg:
        return str(rollout_state.error_msg)
    return f"rollout status={_status_value(rollout_state.status)}"


def _set_if_not_none(attributes: dict[str, Any], key: str, value: Any | None) -> None:
    if value is not None:
        attributes[key] = value


def _copy_passthrough_extra_fields(attributes: dict[str, Any], extra_fields: Mapping[str, Any]) -> None:
    for key in _PASSTHROUGH_EXTRA_FIELD_KEYS:
        _set_if_not_none(attributes, key, extra_fields.get(key))


__all__ = [
    "TRACE_ATTR_COMPLETION_TOKENS",
    "TRACE_ATTR_DROP_REASON",
    "TRACE_ATTR_FILTER_DECISION",
    "TRACE_ATTR_FILTER_REASON",
    "TRACE_ATTR_JUDGER_BATCH_SIZE",
    "TRACE_ATTR_JUDGER_NAME",
    "TRACE_ATTR_OVERSAMPLE_SOURCE",
    "TRACE_ATTR_PROMPT_TOKENS",
    "TRACE_ATTR_REWARD_PASS",
    "TRACE_ATTR_REWARD_SCORE",
    "TRACE_ATTR_ROLLOUT_BACKEND",
    "TRACE_ATTR_TRAIN_INCLUDED",
    "TRACE_ATTR_XTUNER_ERROR_MSG",
    "TRACE_ATTR_XTUNER_EXPECTED_GROUPS",
    "TRACE_ATTR_XTUNER_EXPECTED_SAMPLES",
    "TRACE_ATTR_XTUNER_FINISH_REASON",
    "TRACE_ATTR_XTUNER_GLOBAL_BATCH_SIZE",
    "TRACE_ATTR_XTUNER_GROUP_ID",
    "TRACE_ATTR_XTUNER_PRODUCER_FUTURE_STEP",
    "TRACE_ATTR_XTUNER_PROMPT_REPEAT_K",
    "TRACE_ATTR_XTUNER_ROLLOUT_ID",
    "TRACE_ATTR_XTUNER_SEQ_STALENESS",
    "TRACE_ATTR_XTUNER_SESSION_ID",
    "TRACE_ATTR_XTUNER_STALE",
    "TRACE_ATTR_XTUNER_STATUS",
    "TRACE_ATTR_XTUNER_TASK_NAME",
    "TRACE_EVENT_ROLLOUT_STATUS",
    "TRACE_SPAN_ROLLOUT_BATCH",
    "TRACE_SPAN_ROLLOUT_CONTROLLER_GENERATE",
    "TRACE_SPAN_ROLLOUT_WORKER_GENERATE",
    "TRACE_SPAN_AGENT_LOOP_RUN",
    "TraceAttributes",
    "add_rollout_status_event",
    "judger_trace_attributes",
    "record_rollout_state_result",
    "reward_trace_attributes",
    "rollout_state_final_attributes",
    "rollout_state_initial_attributes",
    "rollout_state_trace_attributes",
]

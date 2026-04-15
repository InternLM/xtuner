from __future__ import annotations

from contextvars import ContextVar, Token

from xtuner.v1.data_proto.rl_data import RolloutState


_CURRENT_TRACE_COLLECTOR: ContextVar[list[RolloutState] | None] = ContextVar(
    "xtuner_rollout_trace_collector",
    default=None,
)


def set_current_trace_collector(collector: list[RolloutState]) -> Token:
    return _CURRENT_TRACE_COLLECTOR.set(collector)


def reset_current_trace_collector(token: Token) -> None:
    _CURRENT_TRACE_COLLECTOR.reset(token)


def append_current_trace_rollout_state(rollout_state: RolloutState) -> None:
    collector = _CURRENT_TRACE_COLLECTOR.get()
    if collector is None:
        return
    collector.append(rollout_state.model_copy(deep=True))

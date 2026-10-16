"""Public rollout checks and the separate pre-training field contract."""

from xtuner.v1.data_proto.rl_data import RolloutState, Status


def validate_rollout_group(group: list[RolloutState], *, task_name: str) -> None:
    """Check producer result structure, final status and existing version
    bounds.

    This function is read-only: it neither counts tokens nor fills model versions.
    """
    allowed = {Status.COMPLETED, Status.FAILED, Status.FILTERED, Status.ABORTED, Status.EXPIRED}
    for state in group:
        context = f"task={task_name!r}, group_id={state.group_id}, rollout_id={state.rollout_id}"
        if state.status not in allowed:
            raise ValueError(f"{context}: invalid generated status {state.status}")
        response_length = len(state.response_ids or [])
        version_length = len(state.response_model_steps or [])
        if version_length > response_length:
            raise ValueError(
                f"{context}: response_model_steps length={version_length}, expected at most response_ids={response_length}"
            )


def validate_train_group(group: list[RolloutState], *, task_name: str) -> None:
    """Check a completed group's full fields immediately before training.

    Raise ValueError so the training boundary can discard the entire group. This function does not fill source versions
    or modify labels or metadata.
    """
    if not group or any(not isinstance(state, RolloutState) or state.status != Status.COMPLETED for state in group):
        raise ValueError(f"task={task_name!r}: training requires a nonempty COMPLETED rollout group")
    for state in group:
        context = f"task={task_name!r}, group_id={state.group_id}, rollout_id={state.rollout_id}"
        input_ids = state.input_ids
        if input_ids is None or len(input_ids) < 2:
            length = None if input_ids is None else len(input_ids)
            raise ValueError(f"{context}: input_ids must contain at least 2 tokens, got {length}")
        for field in ("labels", "logprobs"):
            value = getattr(state, field)
            if field == "logprobs" and value is None:
                continue
            length = None if value is None else len(value)
            if length != len(input_ids):
                raise ValueError(f"{context}: {field} length mismatch: {length}, expected input_ids={len(input_ids)}")
        labels = state.labels
        assert labels is not None
        response_ids = state.response_ids or []
        version_length = len(state.response_model_steps or [])
        if version_length != len(response_ids):
            raise ValueError(
                f"{context}: response_model_steps length={version_length}, expected response_ids={len(response_ids)}"
            )
        position_ids = state.position_ids
        expected = {
            "train_response_length": len(response_ids),
            "supervised_tokens": sum(label != -100 for label in labels[1:]),
            "position_layout": "mrope_3d" if position_ids is not None and position_ids.ndim == 3 else "1d",
            "has_rollout_logprobs": state.logprobs is not None,
            "has_routed_experts": state.routed_experts is not None,
        }
        if state.num_tokens != len(input_ids) - 1:
            raise ValueError(
                f"{context}: num_tokens={state.num_tokens}, expected={len(input_ids) - 1}; missing or stale train meta"
            )
        prompt_length = state.extra_fields.get("train_prompt_length")
        if not isinstance(prompt_length, int) or prompt_length < 0:
            raise ValueError(f"{context}: missing or invalid train_prompt_length={prompt_length}")
        for key, value in expected.items():
            actual = state.extra_fields.get(key)
            if actual != value:
                raise ValueError(f"{context}: train meta {key}={actual!r}, expected={value!r}")

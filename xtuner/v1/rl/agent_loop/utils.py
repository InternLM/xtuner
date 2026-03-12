from xtuner.v1.data_proto import RolloutState, Status, update_seq_staleness


class PartialRolloutHandler:
    """处理 Partial Rollout 的状态预处理与后处理."""

    def __init__(self, max_tokens: int, logger):
        self.max_tokens = max_tokens
        self.logger = logger

    def preprocess(self, rollout_state: RolloutState, enable_partial_rollout: bool = False) -> RolloutState:
        # for partial rollout
        if not enable_partial_rollout or not rollout_state.response_ids or rollout_state.status == Status.COMPLETED:
            return rollout_state

        # 如果状态是 EXPIRED，重置 tokens, sample_params和responses, 重新生成
        if rollout_state.status == Status.EXPIRED:
            rollout_state.tokens = rollout_state.prompt_ids
            rollout_state.sample_params = rollout_state.sample_params.copy(update={"max_tokens": self.max_tokens})
            rollout_state.response_ids = []
            rollout_state.response = ""
            rollout_state.logprobs = []
            rollout_state.response_mask = []
            rollout_state.response_steps = []
            return rollout_state

        # Set up token and length variable
        response_ids = rollout_state.response_ids
        prompt_ids = list(rollout_state.prompt_ids or [])
        response_len = len(response_ids)
        prompt_len = len(prompt_ids)

        rollout_state.tokens = prompt_ids + response_ids  # partial rollout 拼接逻辑
        remaining_tokens = self.max_tokens - response_len  # partial rollout max_tokens 计算逻辑
        rollout_state.sample_params = rollout_state.sample_params.copy(update={"max_tokens": remaining_tokens})

        self.logger.debug(
            f"Sample {rollout_state.uid} continue rollout | Remaining tokens allowed: {remaining_tokens} | Status: {rollout_state.status} | Prompt len: {prompt_len} | Response len: {response_len} | Total tokens: {len(rollout_state.tokens)}"
        )
        # TODO: 处理 routed_experts
        rollout_state.extra_fields["history_response_dict"] = {
            "response_ids": rollout_state.tokens[prompt_len:] if rollout_state.tokens else [],
            "response": rollout_state.response or "",
            "logprobs": rollout_state.logprobs or [],
            "response_mask": rollout_state.response_mask or [],
        }
        return rollout_state

    def postprocess(self, rollout_state: RolloutState, rollout_step: int) -> RolloutState:
        history_dict = rollout_state.extra_fields.pop("history_response_dict", None)
        if not history_dict:
            return rollout_state

        # 需要在拼接历史response_ids前更新seq_staleness
        rollout_state = update_seq_staleness(rollout_state, rollout_step)  # 计算 seq_staleness
        rollout_state.response_ids = history_dict.get("response_ids", []) + (rollout_state.response_ids or [])
        rollout_state.response = history_dict.get("response", "") + (rollout_state.response or "")
        rollout_state.logprobs = history_dict.get("logprobs", []) + (rollout_state.logprobs or [])
        rollout_state.response_mask = history_dict.get("response_mask", []) + (rollout_state.response_mask or [])

        return rollout_state

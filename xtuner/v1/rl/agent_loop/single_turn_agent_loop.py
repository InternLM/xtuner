from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.trace.rollout_api import trace_rollout_endpoint, trace_rollout_remote

from .agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    mark_training_artifacts_failed,
    normalize_token_ids,
    validate_training_artifacts,
)


class SingleTurnAgentLoopConfig(AgentLoopConfig):
    """Configuration for the built-in single-turn agent loop.

    ``SingleTurnAgentLoopConfig`` runs one model generation for each input
    ``RolloutState`` and optionally sends the completed output to a judger. It
    is the default choice for math, QA, and other single-response RL tasks.

    Args:
        sample_params (SampleParams): Sampling parameters used by the rollout
            backend, such as temperature and maximum generation length.
        hf_checkpoint (str): Hugging Face checkpoint path used to identify the
            policy checkpoint for the agent loop.
        cpu_resources (CPUResourcesConfig | None): PG-external CPU resources
            used to run this agent loop as Ray actors. ``None`` runs the loop
            in local mode. Defaults to None.
        enable_batch_judge (bool): Whether to judge a generated group in one
            batch in ``generate_group``. Defaults to False.

    **Examples:**

    Example configuration for a single-turn task::

        config = SingleTurnAgentLoopConfig(
            sample_params=SampleParams(max_tokens=1024, temperature=1.0),
            hf_checkpoint="Qwen/Qwen3-8B",
            enable_batch_judge=True,
        )
    """

    def build_local(self, rollout_controller, judger: Judger | None = None, logger=None) -> "SingleTurnAgentLoop":
        return SingleTurnAgentLoop(
            rollout_ctl=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
            enable_batch_judge=self.enable_batch_judge,
        )


class SingleTurnAgentLoop(AgentLoop):
    def __init__(
        self,
        rollout_ctl: RolloutController,
        sample_params: SampleParams | None,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
        enable_batch_judge: bool = False,
    ):
        super().__init__(
            rollout_ctl=rollout_ctl,
            sample_params=sample_params,
            hf_checkpoint=hf_checkpoint,
            judger=judger,
            logger=logger,
            enable_batch_judge=enable_batch_judge,
        )

    @trace_rollout_endpoint("single_turn_agent_loop.run")
    async def generate_sample(
        self,
        rollout_state: RolloutState,
        **kwargs,
    ) -> RolloutState:
        if not rollout_state.tokens:
            rollout_state.tokens = rollout_state.prompt_ids

        # 推理引擎generate, 生成的结果会覆盖到 rollout_state.response_ids 上
        assert self.rollout_ctl is not None
        rollout_state = await trace_rollout_remote(
            self.rollout_ctl.generate,  # type: ignore[attr-defined]
            rollout_state,
        )
        # 非 COMPLETED 状态（如被截断、放弃等）直接早退，不触发打分
        if rollout_state.status != Status.COMPLETED:
            return rollout_state
        if self.judger is not None and not self.enable_batch_judge:
            # 如果开启了批量打分，则在 generate_group 里统一打分，不在这里逐条打分
            rollout_state = await self.run_judger(rollout_state)
        return rollout_state

    async def prepare_training_artifacts(self, rollout_state: RolloutState) -> RolloutState:
        """Prepare list-based training artifacts for a completed single-turn
        rollout."""
        try:
            if rollout_state.status != Status.COMPLETED:
                return rollout_state
            prompt_ids = rollout_state.prompt_ids or rollout_state.extra_fields.get("train_prompt_ids")
            prompt_ids = normalize_token_ids(prompt_ids)
            response_ids = normalize_token_ids(rollout_state.response_ids)
            response_mask = rollout_state.response_mask
            if response_mask is None:
                response_mask = [1] * len(response_ids)
            response_labels = [
                response_id if mask_id != 0 else -100 for response_id, mask_id in zip(response_ids, response_mask)
            ]
            rollout_state.input_ids = prompt_ids + response_ids[:-1]
            rollout_state.labels = [-100] * (len(prompt_ids) - 1) + response_labels
            if rollout_state.logprobs is not None:
                rollout_state.logprobs = [0.0] * (len(prompt_ids) - 1) + [
                    float(value) for value in rollout_state.logprobs
                ]
            validate_training_artifacts(rollout_state)
            rollout_state.response_ids = response_ids
            rollout_state.input_ids = normalize_token_ids(rollout_state.input_ids)
            rollout_state.labels = normalize_token_ids(rollout_state.labels)
            return rollout_state
        except Exception as exc:
            return mark_training_artifacts_failed(rollout_state, exc, self.logger)

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController

from .agent_loop import AgentLoop, AgentLoopConfig
from .utils import PartialRolloutHandler


class SingleTurnAgentLoopConfig(AgentLoopConfig):
    def build_local(self, rollout_controller, judger: Judger | None = None, logger=None) -> "SingleTurnAgentLoop":
        return SingleTurnAgentLoop(
            rollout_ctl=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
        )


class SingleTurnAgentLoop(AgentLoop):
    def __init__(
        self,
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
    ):
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judger, logger)
        self.max_tokens = self.sample_params.max_tokens
        self.partial_rollout_handler = PartialRolloutHandler(max_tokens=self.max_tokens)

    async def generate_sample(
        self,
        rollout_state: RolloutState,
        **kwargs,
    ) -> RolloutState:
        enable_partial_rollout = kwargs.get("enable_partial_rollout", False)

        # rollout state 预处理, enable_partial_rollout = True 会在这里拼接 token 和修正 max_token
        rollout_state = self.partial_rollout_handler.preprocess(rollout_state, enable_partial_rollout)
        if not rollout_state.tokens:
            rollout_state.tokens = rollout_state.prompt_ids

        # 推理引擎generate, 生成的结果会覆盖到 rollout_state.response_ids 上
        rollout_state = await self.rollout_ctl.generate.remote(rollout_state)  # type: ignore[attr-defined]
        # rollout state 后处理: 合并 partial rollout 的历史上下文
        rollout_state = self.partial_rollout_handler.postprocess(rollout_state)
        # 非 COMPLETED 状态（如被截断、放弃等）直接早退，不触发打分
        if rollout_state.status != Status.COMPLETED:
            return rollout_state
        if self.judger is not None:
            rollout_state = await self.judger.judge(rollout_state)
        return rollout_state

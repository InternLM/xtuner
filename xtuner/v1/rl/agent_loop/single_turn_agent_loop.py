import asyncio
from typing import overload

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import cancel_and_drain, create_task

from .agent_loop import DEFAULT_JUDGER_CANCEL_TIMEOUT_S, AgentLoop, AgentLoopConfig


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

    enable_batch_judge: bool = False

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
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
        enable_batch_judge: bool = False,
    ):
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judger, logger)
        self.enable_batch_judge = enable_batch_judge
        self._pause_event = asyncio.Event()

    @overload
    async def run_judger(self, rollout_state: RolloutState) -> RolloutState: ...

    @overload
    async def run_judger(self, rollout_state: list[RolloutState]) -> list[RolloutState]: ...

    async def run_judger(self, rollout_state: RolloutState | list[RolloutState]) -> RolloutState | list[RolloutState]:
        assert self.judger is not None
        judge_task = create_task(self.judger.judge(rollout_state))
        pause_task = create_task(self._pause_event.wait())
        try:
            done, _ = await asyncio.wait({judge_task, pause_task}, return_when=asyncio.FIRST_COMPLETED)
            if judge_task in done:
                return await judge_task
            try:
                return await asyncio.wait_for(
                    asyncio.shield(judge_task),
                    timeout=DEFAULT_JUDGER_CANCEL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                await cancel_and_drain([judge_task])
                for sample in rollout_state if isinstance(rollout_state, list) else [rollout_state]:
                    sample.status = Status.ABORTED
                    sample.finish_reason = "abort"
                    sample.reward = None
                return rollout_state
        except asyncio.CancelledError:
            await cancel_and_drain([judge_task])
            for sample in rollout_state if isinstance(rollout_state, list) else [rollout_state]:
                sample.status = Status.ABORTED
                sample.finish_reason = "abort"
                sample.reward = None
            return rollout_state
        finally:
            await cancel_and_drain([pause_task])

    async def pause(self) -> None:
        self._pause_event.set()
        # TODO: Decide whether Judger needs an explicit pause API for resources not owned by SingleTurnAgentLoop.
        try:
            await super().pause()
        finally:
            self._pause_event.clear()

    async def generate_sample(
        self,
        rollout_state: RolloutState,
        **kwargs,
    ) -> RolloutState:
        if not rollout_state.tokens:
            rollout_state.tokens = rollout_state.prompt_ids

        # 推理引擎generate, 生成的结果会覆盖到 rollout_state.response_ids 上
        rollout_state = await self.rollout_ctl.generate.remote(rollout_state)  # type: ignore[attr-defined]
        # 非 COMPLETED 状态（如被截断、放弃等）直接早退，不触发打分
        if rollout_state.status != Status.COMPLETED:
            return rollout_state
        if self.judger is not None and not self.enable_batch_judge:
            # 如果开启了批量打分，则在 generate_group 里统一打分，不在这里逐条打分
            rollout_state = await self.run_judger(rollout_state)
        return rollout_state

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self.generate_sample(state, **kwargs))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        if self.judger is not None and self.enable_batch_judge:
            if not any(sample.status == Status.ABORTED for sample in group_samples):
                # 批量打分
                group_samples = await self.run_judger(group_samples)
        return group_samples

from typing import Any, Callable, Optional

from omegaconf import DictConfig
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, DictConfigWrap
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.utils.dataset.rl_dataset import get_dataset_class
from verl.workers.rollout.replica import TokenOutput

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.ray.judger import NativeJudger, RouterJudger
from xtuner.v1.ray.rollout.controller import RolloutControllerProxy

from .agent_loop import AgentLoop, AgentLoopConfig


class VerlToolAgentLoopConfig(AgentLoopConfig):
    config: DictConfig

    def build(
        self,
        rollout_controller: RolloutControllerProxy,
        judger: Callable | NativeJudger | RouterJudger | None = None,
        logger=None,
    ) -> "VerlToolAgentLoop":
        verl_tool_agent_loop = VerlToolAgentLoop(
            rollout_controller=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            config=self.config,
            judger=judger,
        )
        return verl_tool_agent_loop


class XtunerAsyncLLMServerManager:
    def __init__(self, rollout_controller: RolloutControllerProxy):
        self.rollout_controller = rollout_controller

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        sample_params = SampleParams(
            return_token_ids=True,
            temperature=1.0,  # sampling_params.get("temperature", 1.0),  # TODO
            top_p=1.0,  # sampling_params.get("top_p", 1.0),
            top_k=0,  # sampling_params.get("top_k", 0),
            repetition_penalty=1.0,
            return_logprob=True,  # bool(sampling_params.get("logprobs", True)),
        )

        rollout_state = RolloutState(
            message=[],
            tokens=prompt_ids,
            session_uid=hash(request_id) if request_id is not None else None,
            sample_params=sample_params,
        )

        response: RolloutState = await self.rollout_controller.generate.remote(
            rollout_state=rollout_state,
        )

        finish_reason = response.finish_reason

        return TokenOutput(
            token_ids=response.response_ids or [],
            log_probs=response.logprobs,
            routed_experts=response.routed_experts,
            stop_reason=finish_reason,
        )


class VerlToolAgentLoop(AgentLoop):
    def __init__(
        self,
        rollout_controller: RolloutControllerProxy,
        sample_params: SampleParams,
        hf_checkpoint: str,
        config: DictConfig,
        judger: Callable | NativeJudger | RouterJudger | None = None,
        logger=None,
    ):
        super().__init__(rollout_controller, sample_params, hf_checkpoint, judger, logger)

        server_manager = XtunerAsyncLLMServerManager(rollout_controller)

        dataset_cls = get_dataset_class(config.data)

        self.verl_tool_agent_loop = ToolAgentLoop(
            trainer_config=DictConfigWrap(config=config),
            server_manager=server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=dataset_cls,
            data_config=DictConfigWrap(config.data),
        )

    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState:
        assert rollout_state.sample_params is not None, "sample_params must be set in rollout_state"
        # self.verl_tool_agent_loop.loop = asyncio.get_running_loop()  # TODO: check if this is needed

        # convert rollout_state to verl_tool_agent_loop input
        sp = rollout_state.sample_params
        sampling_params = dict(
            temperature=sp.temperature,
            top_p=sp.top_p,
            top_k=sp.top_k,
            repetition_penalty=sp.repetition_penalty,
            logprobs=sp.return_logprob,
        )

        input_kwargs = {
            "raw_prompt": rollout_state.message,
            "tools_kwargs": rollout_state.extra_fields.get("tools_kwargs", {}),
        }

        # run verl_tool_agent_loop
        try:
            output: AgentLoopOutput = await self.verl_tool_agent_loop.run(sampling_params, **input_kwargs)
        except Exception as e:
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = str(e)
            self.logger.error(f"[VerlToolAgentLoop][{rollout_state.session_uid}] generate_sample failed: {e}")
            return rollout_state
        # TODO: handle samples with corrupted tool tokens ?

        # convert verl_tool_agent_loop output to rollout_state
        rollout_state.prompt_ids = output.prompt_ids
        rollout_state.response_ids = output.response_ids
        rollout_state.logprobs = output.response_logprobs
        rollout_state.routed_experts = output.routed_experts
        rollout_state.loss_mask = [0] * len(
            output.prompt_ids
        ) + output.response_mask  # TODO: use loss_mask in Training
        rollout_state.status = Status.COMPLETED
        rollout_state.extra_fields.update(output.extra_fields)
        # judger needs response in text format
        rollout_state.response = self.tokenizer.decode(rollout_state.response_ids)
        # for trajectory dump, we need to add raw_prompt to extra_fields
        # raw_prompt is updated in tool_agent_loop: apply_chat_template of tools
        rollout_state.extra_fields["raw_prompt"] = self.tokenizer.decode(rollout_state.prompt_ids)

        # judge rollout_state
        rollout_state = await self.judge_sample(rollout_state)
        # self.logger.info(f"[VerlToolAgentLoop][{rollout_state.session_uid}] generate_sample completed with raw_prompt:\n    {rollout_state.extra_fields['raw_prompt']}\n and response:\n    {rollout_state.response}")

        return rollout_state

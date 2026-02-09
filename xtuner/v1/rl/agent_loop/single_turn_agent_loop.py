import ray

from xtuner.v1.data_proto import RolloutState, Status

from .agent_loop import AgentLoop


class SingleTurnAgentLoop(AgentLoop):
    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState:
        assert rollout_state.sample_parms is not None, "sample_parms should not be None"
        assert rollout_state.prompt_ids is not None, "prompt_ids should not be None"

        rollout_state.tokens = rollout_state.prompt_ids
        # TODO: 多模态
        rollout_state = await self.rollout_ctl.generate(rollout_state)
        if rollout_state.state == Status.COMPLETED:
            if self.judger is not None:
                if callable(self.judger):
                    rollout_state = await self.judger(rollout_state)
                elif isinstance(self.judger, ray.actor.ActorHandle):
                    rollout_state = await self.judger.remote(rollout_state)
        return rollout_state

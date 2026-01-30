from xtuner.v1.ray.rollout import RolloutController as V1RolloutController
from .rollout_state import RolloutState, Status

reason_map = {
    "length": Status.COMPLETED,
    'aborted': Status.ABORTED,
    "failed": Status.FAILED,
}

# 临时方案
class RolloutController(V1RolloutController):

    async def generate(self, rollout_state: RolloutState):

        # 简单包一层
        input_ids = rollout_state.input_ids
        sample_params = rollout_state.sample_params
        session_id = rollout_state.session_id
        
        response = await super().rollout(
            input_ids=input_ids, 
            sample_params=sample_params, 
            session_id=session_id
        )
        
        rollout_state.response = response.response
        rollout_state.response_ids = response.response_ids
        rollout_state.logprobs = response.logprobs
        rollout_state.state = response.state
        
        return rollout_state

from .simple_env_runner import SimpleEnvRunner
from .rollout_state import RolloutState

# 用户无感
class AsyncProxyEnvRuner:

    def __init__(self):
        self.base_env_runner: SimpleEnvRunner = None

    def set_base_env_runner(self, base_env_runner: SimpleEnvRunner):
        self.base_env_runner = base_env_runner
    
    async def async_generate_batch(self, 
                                    batch_size: int,
                                    prompt_repeat_k: int,
                                    staleness_threshold: float = 0.0,
                                    enable_partial_rollout: bool =False,
                                    ) -> list[RolloutState]:
        raise NotImplementedError("Please implement async_generate_batch method for your custom generation strategy.")

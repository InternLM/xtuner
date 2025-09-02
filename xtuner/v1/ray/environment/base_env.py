from abc import ABC, abstractmethod
from typing import Any, List, Union


class BaseEnvironment(ABC):
    def __init__(self, environment: str, placement_group: Any, rollout_cfg: Any = None, judger_cfg: Any = None):
        self.environment = environment
        self.rollout_controller = self.init_rollout_controller(placement_group, rollout_cfg)
        self.judger_controller = self.init_judger_controller(placement_group, judger_cfg)

    def init_rollout_controller(self, placement_group: Any, rollout_cfg: Any):
        from xtuner.v1.ray.accelerator import AutoAcceleratorWorkers

        rollout_controller = None
        if rollout_cfg is None:
            return rollout_controller
        if rollout_cfg.backend == "lmdeploy":
            from xtuner.v1.ray.rollout import LMDeployWorker

            rollout_workers_map = AutoAcceleratorWorkers.from_placement_group(
                LMDeployWorker, rollout_cfg, placement_group
            )
        elif rollout_cfg.backend == "vllm":
            from xtuner.v1.ray.rollout import vLLMWorker

            rollout_workers_map = AutoAcceleratorWorkers.from_placement_group(vLLMWorker, rollout_cfg, placement_group)
        else:
            raise NotImplementedError(f"Rollout backend '{rollout_cfg.backend}' is not supported.")

        from xtuner.v1.ray.rollout.controller import RolloutController

        rollout_controller = RolloutController.remote(rollout_cfg, rollout_workers_map)  # type: ignore[attr-defined]
        return rollout_controller

    def init_judger_controller(self, placement_group: Any, judger_cfg: Any):
        judger_controller = None
        if judger_cfg:
            from xtuner.v1.ray.judger.controller import JudgerController

            judger_controller = JudgerController.remote(judger_cfg)  # type: ignore[attr-defined]
        return judger_controller

    @abstractmethod
    async def generate(self, data: Union[list, Any, List[Any]], sample_params: Any) -> Union[list, Any, List[Any]]:
        """Generates responses from the model for the given data using the
        inference engine. This method is primarily used for single-step
        inference.

        Args:
            data: The input data, which can be a single prompt, RLTextDataItem, or a list of RLTextDataItem.
            sample_params: Sampling parameters for the generation process.

        Returns:
            A list of generated samples, each populated with 'response_str' and 'state'
        """
        pass

    @abstractmethod
    async def run(self, data: Union[list, Any, List[Any]], sample_params: Any) -> Union[list, Any, List[Any]]:
        """Executes a full cycle of generation and interpretation, such as
        generating a response and then evaluating it with a judger. This method
        can be extended to support complex interactions like multi-turn
        dialogues.

        Args:
            data: The input data for the generation process.
            sample_params: Sampling parameters for generation.

        Returns:
            A list of generated samples
        """
        pass

    def _call_rollout_func(self, method_name: str, block: bool):
        import ray

        assert self.rollout_controller, "Rollout controller is not initialized."
        if block:
            return ray.get(getattr(self.rollout_controller, method_name).remote())
        return getattr(self.rollout_controller, method_name).remote()

    def pause(self, block=True):
        return self._call_rollout_func("pause", block)

    def shutdown(self, block=True):
        return self._call_rollout_func("shutdown", block)

    def restart(self, block=True):
        return self._call_rollout_func("restart", block)

    def get_rollout_info(self, block=True):
        return self._call_rollout_func("get_rollout_info", block)

    def onload_weights(self, block=True):
        return self._call_rollout_func("onload_weights", block)

    def onload_kvcache(self, block=True):
        return self._call_rollout_func("onload_kvcache", block)

    def offload(self, block=True):
        return self._call_rollout_func("offload", block)

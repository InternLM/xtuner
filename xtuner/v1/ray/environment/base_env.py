from abc import ABC, abstractmethod
from typing import Any, List, Union


class BaseEnvironment(ABC):
    """The BaseEnvironment class provides a foundational structure for managing
    rollout and judger controllers for single-turn generation or multi-turn
    generation.

    This class is responsible for initializing the necessary controllers based on the provided
    configurations and placement group. It defines abstract methods for generation and
    execution, which must be implemented by subclasses.

    Args:
        environment (str): The name or identifier of the environment.
        placement_group (Any): The placement group for scheduling Ray actors.
        rollout_cfg (Any, optional): The configuration for the rollout controller. Defaults to None.
        judger_cfg (Any, optional): The configuration for the judger controller. Defaults to None.
    """

    def __init__(self, environment: str, placement_group: Any, rollout_cfg: Any = None, judger_cfg: Any = None):
        self.environment = environment
        self.rollout_controller = self.init_rollout_controller(placement_group, rollout_cfg)
        self.judger_controller = self.init_judger_controller(placement_group, judger_cfg)

    def init_rollout_controller(self, placement_group: Any, rollout_cfg: Any):
        """Initializes the rollout controller with the appropriate worker
        backend.

        Based on the `rollout_cfg`, this method selects and initializes the corresponding
        rollout worker (e.g., `LMDeployWorker` or `vLLMWorker`). It then creates and
        returns a `RolloutController` to manage these workers.

        Args:
            placement_group (Any): The placement group for scheduling Ray actors.
            rollout_cfg (Any): The configuration for the rollout controller.

        Returns:
            The initialized rollout controller, or None if `rollout_cfg` is not provided.

        Raises:
            NotImplementedError: If the specified rollout backend is not supported.
        """
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
        """Initializes the judger controller.

        If a `judger_cfg` is provided, this method creates and returns a `JudgerController`
        to handle evaluation and judging tasks.

        Args:
            placement_group (Any): The placement group for scheduling Ray actors.
            judger_cfg (Any): The configuration for the judger controller.

        Returns:
            The initialized judger controller, or None if `judger_cfg` is not provided.
        """
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
        """A helper function to dynamically call a method on the rollout
        controller.

        Args:
            method_name (str): The name of the method to call.
            block (bool): Whether to block until the call completes.

        Returns:
            The result of the method call.
        """
        import ray

        assert self.rollout_controller, "Rollout controller is not initialized."
        if block:
            return ray.get(getattr(self.rollout_controller, method_name).remote())
        return getattr(self.rollout_controller, method_name).remote()

    def pause(self, block=True):
        """Pauses the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("pause", block)

    def shutdown(self, block=True):
        """Shuts down the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("shutdown", block)

    def restart(self, block=True):
        """Restarts the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("restart", block)

    def get_rollout_info(self, block=True):
        """Gets information about the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("get_rollout_info", block)

    def onload_weights(self, block=True):
        """Loads weights onto the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("onload_weights", block)

    def onload_kvcache(self, block=True):
        """Loads the KV cache onto the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("onload_kvcache", block)

    def offload(self, block=True):
        """Offloads weights and the KV cache from the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("offload", block)


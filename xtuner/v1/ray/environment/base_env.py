import os
from abc import ABC, abstractmethod
from typing import Any, List

import ray

from xtuner.v1.data_proto.rl_data import RLDataFlowItem
from xtuner.v1.utils import ray_method


class BaseEnvironment(ABC):
    """The BaseEnvironment class provides a foundational structure for managing
    rollout and judger controllers for single-turn generation or multi-turn
    generation.

    This class is responsible for initializing the necessary controllers based on the provided
    configurations and placement group. It defines abstract methods for generation and
    execution, which must be implemented by subclasses.

    Args:
        environment (str): The name or identifier of the environment.
        rollout_pg (Any): The placement group for scheduling rollout Ray actors.
        rollout_cfg (Any, optional): The configuration for the rollout controller. Defaults to None.
        judger_pg (Any): The placement group for scheduling judger Ray actors.
                         Defaults to None indicates using the rollout_pg.
        judger_cfg (Any, optional): The configuration for the judger controller. Defaults to None.
    """

    def __init__(
        self,
        environment: str,
        rollout_pg: Any,
        rollout_cfg: Any,
        judger_pg: Any = None,
        judger_cfg: Any = None,
        rollout_controller=None,
        judger_controller=None,
    ):
        # judger_pg = judger_pg if judger_pg else rollout_pg
        self.environment = environment
        if rollout_controller:
            self.rollout_controller = rollout_controller
        else:
            self.rollout_controller = self.init_rollout_controller(rollout_cfg, rollout_pg)
        if judger_controller:
            self.judger_controller = judger_controller
        else:
            self.judger_controller = self.init_judger_controller(judger_cfg, judger_pg)

    def init_rollout_controller(self, rollout_cfg: Any, placement_group: Any):
        """Initializes the rollout controller with the appropriate worker
        backend.

        Based on the `rollout_cfg`, this method selects and initializes the corresponding
        rollout worker (e.g., `LMDeployWorker` or `vLLMWorker`). It then creates and
        returns a `RolloutController` to manage these workers.

        Args:
            rollout_cfg (Any): The configuration for the rollout controller.
            placement_group (Any): The placement group for scheduling Ray actors.

        Returns:
            The initialized rollout controller, or None if `rollout_cfg` is not provided.

        Raises:
            NotImplementedError: If the specified rollout backend is not supported.
        """

        rollout_controller = None
        if rollout_cfg is None:
            return rollout_controller

        from xtuner.v1.ray.rollout.controller import RolloutController

        rollout_controller = (
            ray.remote(RolloutController)
            .options(max_concurrency=int(os.environ.get("RAY_MAX_CONCURRENCY", 1000)))
            .remote(rollout_cfg, placement_group)
        )  # type: ignore[attr-defined]
        return rollout_controller

    def init_judger_controller(self, judger_cfg: Any, placement_group: Any):
        """Initializes the judger controller.

        If a `judger_cfg` is provided, this method creates and returns a `JudgerController`
        to handle evaluation and judging tasks.

        Args:
            judger_cfg (Any): The configuration for the judger controller.
            placement_group (Any): The placement group for scheduling Ray actors.

        Returns:
            The initialized judger controller, or None if `judger_cfg` is not provided.
        """
        judger_controller = None
        if judger_cfg:
            from xtuner.v1.ray.judger.controller import JudgerController

            judger_controller = JudgerController.remote(judger_cfg, placement_group)  # type: ignore[attr-defined]
        return judger_controller

    @abstractmethod
    @ray_method
    async def generate(
        self, data: List[RLDataFlowItem], sample_params: Any, extra_params: Any
    ) -> List[RLDataFlowItem]:
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
    @ray_method
    async def run(self, data: List[RLDataFlowItem], sample_params: Any, extra_params: Any) -> List[RLDataFlowItem]:
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

    @ray_method
    def pause(self, block=True) -> None:
        """Pauses the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("pause", block)

    @ray_method
    def shutdown(self, block=True) -> None:
        """Shuts down the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("shutdown", block)

    @ray_method
    def restart(self, block=True) -> None:
        """Restarts the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("restart", block)

    @ray_method
    def get_rollout_info(self, block=True) -> dict[str, Any]:
        """Gets information about the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("get_rollout_info", block)

    @ray_method
    def onload_weights(self, block=True) -> None:
        """Loads weights onto the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("onload_weights", block)

    @ray_method
    def onload_kvcache(self, block=True) -> str:
        """Loads the KV cache onto the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("onload_kvcache", block)

    @ray_method
    def offload(self, block=True) -> str:
        """Offloads weights and the KV cache from the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("offload", block)

    @ray_method
    def update_active_workers(self, block=True) -> None:
        """Checks the status of active rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("update_active_workers", block)

    @ray_method
    def get_rollout_stats(self, block=True) -> dict[str, Any]:
        """Gets statistics from the rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        return self._call_rollout_func("get_rollout_stats", block)

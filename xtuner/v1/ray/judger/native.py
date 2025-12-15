import inspect
from typing import Any, Callable, List, Optional

import httpx
import ray
from pydantic import BaseModel
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RLDataFlowItem, RLJudgerResponseItem
from xtuner.v1.utils import get_logger


class NativeJudgerConfig(BaseModel):
    """Base configuration class for judgers."""

    model_config = {"arbitrary_types_allowed": True}
    judger_name: str
    num_ray_actors: int = 1
    reward_func: Optional[Callable] = None
    remote_url: Optional[str] = None
    preprocess_func: Optional[Callable] = None
    postprocess_func: Optional[Callable] = None
    request_timeout: float = 30.0
    extra_info: dict = {}

    def build_actor(self, pg: PlacementGroup, start_bundle_idx: int) -> List[ray.actor.ActorClass]:
        """Create and launch Ray actor instances for the GSM8K judger.

        This method instantiates multiple NativeJudger Ray actors according to `num_ray_actors`,
        assigning each to a specific bundle in the provided placement group for resource isolation.
        Each actor is initialized with the judger's configuration and reward function.

        Args:
            pg: The Ray PlacementGroup used to allocate resources for the actors.
            start_bundle_idx: The starting bundle index in the placement group for actor placement.

        Returns:
            List[ActorClass]: A list of Ray actor handles representing the launched judger workers.
        """
        workers_list = []
        pg_options = {"num_cpus": pg.bundle_specs[0].get("CPU", 1)}
        for idx in range(self.num_ray_actors):
            worker = (
                ray.remote(NativeJudger)
                .options(
                    placement_group=pg,
                    placement_group_bundle_index=(start_bundle_idx + idx),
                    **pg_options,
                )
                .remote(
                    judger_name=self.judger_name,
                    reward_func=self.reward_func,
                    remote_url=self.remote_url,
                    preprocess_func=self.preprocess_func,
                    postprocess_func=self.postprocess_func,
                    request_timeout=self.request_timeout,
                    extra_info=self.extra_info,
                )
            )
            workers_list.append(worker)
        return workers_list


class NativeJudger:
    """Base class for judgers, providing a standard interface for executing a
    judging process, which can be either a local function or a remote service.

    The judger orchestrates a three-step pipeline:
    1. Pre-process the input data.
    2. Execute the core logic (local function or remote HTTP call).
    3. Post-process the result.
    """

    def __init__(
        self,
        judger_name: str = "native_judger",
        reward_func: Optional[Callable] = None,
        remote_url: Optional[str] = None,
        preprocess_func: Optional[Callable] = None,
        postprocess_func: Optional[Callable] = None,
        request_timeout: float = 30.0,
        extra_info: dict = {},
    ):
        """Initialize the NativeJudger.

        Args:
            reward_func (Optional[Callable]): A local function to compute the
                reward. Exactly one of `reward_func` or `remote_url` must be
                provided. Defaults to None.
            remote_url (Optional[str]): The URL of a remote service for
                judging. Exactly one of `reward_func` or `remote_url` must be
                provided. Defaults to None.
            preprocess_func (Optional[Callable]): A function to preprocess the
                input data before judger execution. Defaults to None.
            postprocess_func (Optional[Callable]): A function to postprocess
                the judger result. Defaults to None.
            request_timeout (float): Timeout for remote requests in seconds.
                Defaults to 30.0.
            extra_info (dict): Extra information to be passed to the reward
                function. Defaults to {}.

        Raises:
            ValueError: If both or neither of `reward_func` and `remote_url`
                are provided.
        """
        if (reward_func is None and remote_url is None) or (reward_func is not None and remote_url is not None):
            raise ValueError("Exactly one of 'reward_func' or 'remote_url' must be provided.")
        self.judger_name = judger_name
        self.extra_info = extra_info
        self.reward_func = reward_func
        self.remote_url = remote_url

        self.preprocess_func = preprocess_func or self._default_preprocess
        self.postprocess_func = postprocess_func or self._default_postprocess

        self.http_client = None
        self.execute_func = None

        if self.reward_func:
            self.execute_func = self._local_executor
        elif self.remote_url:
            self.http_client = httpx.AsyncClient(timeout=request_timeout)
            self.execute_func = self._remote_executor

    def _default_preprocess(self, data_item: List[RLDataFlowItem], extra_info: dict) -> Any:
        """Default preprocessing function.

        Args:
            data_item (RLDataFlowItem | List[RLDataFlowItem]): The data item(s) to preprocess.

        Returns:
            Any: A dictionary containing the responses, labels, and extra info.
        """

        assert len(data_item) == 1, "Default preprocess only supports single data item."
        # TODO: Support batch reward calculation via API server
        response = data_item[0].env.rollout.response
        assert data_item[0].data.reward_model is not None
        label = data_item[0].data.reward_model["ground_truth"]
        return {
            "response": response,
            "label": label,
            "extra_info": extra_info,
        }

    def _default_postprocess(self, result: Any) -> List[RLJudgerResponseItem]:
        ## 将结果包装成 RLJudgerResponseItem
        """Default postprocessing function.

        Args:
            result (Any): The result from the execution step.

        Returns:
            Any: The result, unchanged.
        """
        if not isinstance(result, list):
            result = [result]
        # todo: 支持多个judger结果的返回
        judger_response_item = [RLJudgerResponseItem(reward=result[i]) for i in range(len(result))]
        return judger_response_item

    async def _local_executor(self, data_item: List[RLDataFlowItem]) -> List[RLJudgerResponseItem]:
        """Executes the reward function locally.

        Args:
            responses (str | List[str]): The model's response(s).
            labels (str | List[str]): The ground-truth label(s).

        Returns:
            Any: The postprocessed result of the reward function.
        """
        assert self.reward_func is not None, "reward_func cannot be None for local execution."
        # 记录每个judger请求的uid, 方便后续结果合并
        uid_list = [item.uid.observation_id for item in data_item]
        kwargs = self.preprocess_func(data_item, self.extra_info)
        if inspect.iscoroutinefunction(self.reward_func):
            json_result = await self.reward_func(**kwargs)
        else:
            json_result = self.reward_func(**kwargs)

        # transform json to RLJudgerResponseItem
        result = self.postprocess_func(json_result)
        for i in range(len(result)):
            result[i].uid = uid_list[i]
        return result

    async def _remote_executor(self, data_item: List[RLDataFlowItem]) -> List[RLJudgerResponseItem]:
        """Executes the reward function by calling a remote service.

        Args:
            responses (str | List[str]): The model's response(s).
            labels (str | List[str]): The ground-truth label(s).

        Returns:
            Any: The postprocessed result from the remote service, or None if
                an error occurs.
        """
        assert self.remote_url is not None and self.http_client is not None, (
            "remote_url cannot be None for remote execution."
        )
        payload = self.preprocess_func(data_item, self.extra_info)
        try:
            response = await self.http_client.post(self.remote_url, json=payload)
            response.raise_for_status()
            json_result = response.json()
            # 重要，必须加
            json_result["uid"] = data_item[0].uid.observation_id
            # transform json to RLJudgerResponseItem
            return self.postprocess_func(json_result)
        except httpx.RequestError as exc:
            get_logger().error(f"An error occurred while requesting {exc.request.url}: {exc}")
            return []

    async def judge(self, data_item: List[RLDataFlowItem]) -> List[RLJudgerResponseItem]:
        """The main public method to run the judging pipeline.

        Args:
            responses (str | List[str]): The model's response(s) to be judged.
            labels (str | List[str]): The ground-truth label(s).

        Returns:
            Any: The final result after the full
                preprocess-execute-postprocess pipeline.

        Raises:
            RuntimeError: If the judger is not properly initialized.
        """
        if self.execute_func is None:
            raise RuntimeError("Judger is not properly initialized.")
        return await self.execute_func(data_item)

    def get_judger_name(self) -> str:
        """Get the name of the judger.

        Returns:
            str: The name of the judger.
        """
        return self.judger_name

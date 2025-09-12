import inspect
from typing import Any, Callable, List, Optional

import httpx

from xtuner.v1.data_proto.rl_data import RLDataFlowItem, RLJudgerResponseItem
from xtuner.v1.utils import get_logger


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
        self.logger = get_logger(__name__)

    def _default_preprocess(self, data_item: List[RLDataFlowItem], extra_info: dict) -> Any:
        ## 将输入RLDataFlowItem转换成 reward_func需要的输入格式
        ## batch_reward: 传入list of RLDataFlowItem
        ## single_reward: 传入单个 RLDataFlowItem
        """Default preprocessing function.

        Args:
            data_item (RLDataFlowItem | List[RLDataFlowItem]): The data item(s) to preprocess.

        Returns:
            Any: A dictionary containing the responses, labels, and extra info.
        """

        if len(data_item) == 1:
            response = data_item[0].env.rollout.response
            label = data_item[0].data.reward_model["ground_truth"]
        else:
            # 当有多个数据计算batch_reward时，传入list，custom_reward_func的输入参数需要是list
            response = [item.env.rollout.response for item in data_item]
            label = [item.data.reward_model["ground_truth"] for item in data_item]
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
        judger_response_item = [RLJudgerResponseItem(reward={self.judger_name: result[i]}) for i in range(len(result))]
        self.logger.info(f"Postprocessed result: {judger_response_item}")
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
        uid_list = (
            data_item.uid.action_id
            if isinstance(data_item, dict)
            else [item.uid.action_id for item in data_item]
        )
        kwargs = self.preprocess_func(data_item, self.extra_info)
        if inspect.iscoroutinefunction(self.reward_func):
            result = await self.reward_func(**kwargs)
        else:
            result = self.reward_func(**kwargs)
        
        result = self.postprocess_func(result)

        if isinstance(result, list):
            for i in range(len(result)):
                result[i].uid = uid_list[i]
        else:
            result.uid = uid_list[0] if isinstance(uid_list, list) else uid_list
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
            result = response.json()
            self.logger.info(f"Received response from {result}")
            return self.postprocess_func(result)
        except httpx.RequestError as exc:
            self.logger.error(f"An error occurred while requesting {exc.request.url}: {exc}")
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

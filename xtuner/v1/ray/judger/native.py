import inspect
from typing import Any, Callable, List, Optional

import httpx


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
        reward_func: Optional[Callable] = None,
        remote_url: Optional[str] = None,
        preprocess_func: Optional[Callable] = None,
        postprocess_func: Optional[Callable] = None,
        request_timeout: float = 30.0,
        extra_info: dict = {},
    ):
        if (reward_func is None and remote_url is None) or (reward_func is not None and remote_url is not None):
            raise ValueError("Exactly one of 'reward_func' or 'remote_url' must be provided.")

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

    def _default_preprocess(self, responses: str | List[str], labels: str | List[str]) -> Any:
        return {"response": responses, "label": labels, "extra_info": self.extra_info}

    def _default_postprocess(self, result: Any) -> Any:
        return result

    async def _local_executor(self, responses: str | List[str], labels: str | List[str]) -> Any:
        assert self.reward_func is not None, "reward_func cannot be None for local execution."
        kwargs = self.preprocess_func(responses, labels)
        if inspect.iscoroutinefunction(self.reward_func):
            result = await self.reward_func(**kwargs)
        else:
            result = self.reward_func(**kwargs)
        return self.postprocess_func(result)

    async def _remote_executor(self, responses: str | List[str], labels: str | List[str]) -> Any:
        assert self.remote_url is not None and self.http_client is not None, (
            "remote_url cannot be None for remote execution."
        )
        payload = self.preprocess_func(responses, labels)
        try:
            response = await self.http_client.post(self.remote_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return self.postprocess_func(result)
        except httpx.RequestError as exc:
            print(f"An error occurred while requesting {exc.request.url}: {exc}")
            return None

    async def judge(self, responses: str | List[str], labels: str | List[str]) -> Any:
        """The main public method to run the judging pipeline.

        Args:
            data: A list of data items to be judged.

        Returns:
            The final result after the full preprocess-execute-postprocess pipeline.
        """
        if self.execute_func is None:
            raise RuntimeError("Judger is not properly initialized.")
        return await self.execute_func(responses, labels)

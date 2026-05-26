import asyncio
import os
import traceback
import uuid
from typing import Any

import httpx

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status, update_status_from_finish_reason
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.trace_store import get_store
from xtuner.v1.rl.utils import create_task

from .agent_loop import AgentLoop, AgentLoopConfig


ROUTED_APIPROXY_BASE_URL = "http://s-20260104203038-22bhb.ailab-evalservice.pjh-service.org.cn/v1"
ROUTED_APIPROXY_API_KEY = "sk-admin"
ROUTED_APIPROXY_TIMEOUT = 3600.0
ROUTED_APIPROXY_MAX_CONNECTIONS = 512
ROUTED_APIPROXY_MAX_KEEPALIVE_CONNECTIONS = 128


class SingleTurnAgentLoopConfig(AgentLoopConfig):
    """Configuration for the built-in single-turn agent loop.

    ``SingleTurnAgentLoopConfig`` runs one model generation for each input
    ``RolloutState`` and optionally sends the completed output to a judger. It
    is the default choice for math, QA, and other single-response RL tasks.

    Args:
        sample_params (SampleParams): Sampling parameters used by the rollout
            backend, such as temperature and maximum generation length.
        hf_checkpoint (str): Hugging Face checkpoint path used to identify the
            policy checkpoint for the agent loop.
        cpu_resources (CPUResourcesConfig | None): PG-external CPU resources
            used to run this agent loop as Ray actors. ``None`` runs the loop
            in local mode. Defaults to None.
        enable_batch_judge (bool): Whether to judge a generated group in one
            batch in ``generate_group``. Defaults to False.

    **Examples:**

    Example configuration for a single-turn task::

        config = SingleTurnAgentLoopConfig(
            sample_params=SampleParams(max_tokens=1024, temperature=1.0),
            hf_checkpoint="Qwen/Qwen3-8B",
            enable_batch_judge=True,
        )
    """

    enable_batch_judge: bool = False
    api_base_url: str = ROUTED_APIPROXY_BASE_URL
    api_key: str = ROUTED_APIPROXY_API_KEY
    api_timeout: float = ROUTED_APIPROXY_TIMEOUT
    api_max_connections: int = ROUTED_APIPROXY_MAX_CONNECTIONS
    api_max_keepalive_connections: int = ROUTED_APIPROXY_MAX_KEEPALIVE_CONNECTIONS

    def build_local(self, rollout_controller, judger: Judger | None = None, logger=None) -> "SingleTurnAgentLoop":
        return SingleTurnAgentLoop(
            rollout_ctl=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
            enable_batch_judge=self.enable_batch_judge,
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            api_timeout=self.api_timeout,
            api_max_connections=self.api_max_connections,
            api_max_keepalive_connections=self.api_max_keepalive_connections,
        )


class SingleTurnAgentLoop(AgentLoop):
    def __init__(
        self,
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
        enable_batch_judge: bool = False,
        api_base_url: str = ROUTED_APIPROXY_BASE_URL,
        api_key: str = ROUTED_APIPROXY_API_KEY,
        api_timeout: float = ROUTED_APIPROXY_TIMEOUT,
        api_max_connections: int = ROUTED_APIPROXY_MAX_CONNECTIONS,
        api_max_keepalive_connections: int = ROUTED_APIPROXY_MAX_KEEPALIVE_CONNECTIONS,
    ):
        super().__init__(rollout_ctl, sample_params, hf_checkpoint, judger, logger)
        self.enable_batch_judge = enable_batch_judge
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.api_timeout = api_timeout
        self.api_max_connections = api_max_connections
        self.api_max_keepalive_connections = api_max_keepalive_connections
        self._model_name = os.environ.get("MODEL_NAME")
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            timeout = httpx.Timeout(self.api_timeout)
            limits = httpx.Limits(
                max_connections=self.api_max_connections,
                max_keepalive_connections=self.api_max_keepalive_connections,
            )
            self._http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
        return self._http_client

    async def generate_sample(
        self,
        rollout_state: RolloutState,
        **kwargs,
    ) -> RolloutState:
        try:
            if rollout_state.uid is None:
                rollout_state.uid = uuid.uuid4().int
            response = await self._chat_completions(rollout_state)
            await self._fill_rollout_state_from_response(rollout_state, response)
        except Exception as exc:
            rollout_state.status = Status.FAILED
            rollout_state.finish_reason = "error"
            rollout_state.error_msg = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"[SingleTurnAgentLoop] failed: {exc}\n{traceback.format_exc()}")
            return rollout_state

        if rollout_state.status != Status.COMPLETED:
            # 非 COMPLETED 状态（如被截断、放弃等）直接早退，不触发打分
            return rollout_state
        if self.judger is not None and not self.enable_batch_judge:
            # 如果开启了批量打分，则在 generate_group 里统一打分，不在这里逐条打分
            rollout_state = await self.judger.judge(rollout_state)
        return rollout_state

    def _build_http_payload(self, rollout_state: RolloutState, model_name: str) -> dict[str, Any]:
        sample_params = rollout_state.sample_params
        payload: dict[str, Any] = {
            "model": model_name,
            "session_id": str(rollout_state.uid),
            "messages": rollout_state.message,
            "max_tokens": sample_params.max_tokens,
            "temperature": sample_params.temperature,
            "top_p": sample_params.top_p,
            "extra_body": {"spaces_between_special_tokens": sample_params.spaces_between_special_tokens},
        }
        if sample_params.stops:
            payload["stop"] = sample_params.stops
        if rollout_state.tools is not None:
            payload["tools"] = rollout_state.tools
        if rollout_state.tool_choice is not None:
            payload["tool_choice"] = rollout_state.tool_choice
        return payload

    async def _chat_completions(self, rollout_state: RolloutState) -> dict[str, Any]:
        model_name = self._model_name
        if model_name is None:
            raise ValueError("RL_LLM_MODEL environment variable is required for routed API rollout.")
        url = f"{self.api_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        client = self._get_http_client()
        payload = self._build_http_payload(rollout_state, model_name)
        response = await client.post(url, headers=headers, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"HTTP rollout failed: {exc}. response={response.text}") from exc
        data = response.json()
        if "choices" not in data:
            raise RuntimeError(f"HTTP rollout response missing choices: {data}")
        return data

    async def _fill_rollout_state_from_response(
        self,
        rollout_state: RolloutState,
        response: dict[str, Any],
    ) -> None:
        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        finish_reason = choice.get("finish_reason")
        status = update_status_from_finish_reason(finish_reason)

        rollout_state.response = content if content is not None else reasoning_content
        rollout_state.finish_reason = finish_reason
        rollout_state.status = status
        if status != Status.COMPLETED:
            rollout_state.error_msg = f"HTTP rollout finished with status={status.value}, finish_reason={finish_reason}"
            return

        messages = [dict(item) for item in rollout_state.message]
        messages.append(message)
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=rollout_state.tools,
            tokenize=False,
            add_generation_prompt=False,
        ).rstrip()

        trace_store = get_store()
        data = await trace_store.export_training_trace.remote(str(rollout_state.uid), text)
        rollout_state.input_ids = data["input_ids"]
        rollout_state.labels = data["labels"]
        rollout_state.response_ids = [
            token_id
            for token_id, label in zip(data["input_ids"][1:], data["labels"][1:])
            if label != -100
        ]
        if rollout_state.response is None:
            rollout_state.response = self.tokenizer.decode(
                rollout_state.response_ids,
                skip_special_tokens=True,
            )
        rollout_state.response_mask = [1] * len(rollout_state.response_ids)
        rollout_state.logprobs = data["logprobs"]
        rollout_state.routed_experts = data["routed_experts"]

    async def generate_group(self, rollout_state: list[RolloutState], **kwargs) -> list[RolloutState]:
        pending_tasks = []
        for state in rollout_state:
            state.sample_params = self.sample_params
            task = create_task(self.generate_sample(state, **kwargs))
            pending_tasks.append(task)
        generated_samples = asyncio.gather(*pending_tasks)
        group_samples = await generated_samples
        if self.judger is not None and self.enable_batch_judge:
            # 批量打分
            group_samples = await self.judger.judge(group_samples)
        return group_samples

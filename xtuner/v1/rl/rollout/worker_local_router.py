from __future__ import annotations

import asyncio
from uuid import uuid4

import ray
from ray.exceptions import RayActorError
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger

from .health_manager import RolloutHealthManagerProxy
from .parser.factory import build_reasoning_parser, build_tool_call_parser
from .parser.reasoning_parser import ReasoningParser
from .parser.tool_parser import ToolCallParser
from .worker import RolloutConfig


class WorkerLocalRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    timeout_multiplier: float = Field(default=2.0, gt=0)

    def build(self, rollout_controller) -> "WorkerLocalRouter":
        rollout_metadata = ray.get(rollout_controller.get_rollout_metadata.remote())
        health_manager = ray.get(rollout_controller.get_health_manager.remote())
        return WorkerLocalRouter(
            rollout_controller=rollout_controller,
            health_manager=health_manager,
            rollout_config=rollout_metadata["rollout_config"],
            timeout_multiplier=self.timeout_multiplier,
        )


class WorkerLocalRouter:
    def __init__(
        self,
        rollout_controller,
        health_manager: RolloutHealthManagerProxy,
        rollout_config: RolloutConfig,
        timeout_multiplier: float = 2.0,
    ) -> None:
        self.rollout_controller = rollout_controller
        self.health_manager = health_manager
        self.config = rollout_config
        self.timeout_multiplier = timeout_multiplier
        self.logger = get_logger(log_dir=rollout_config.worker_log_dir, tag="WorkerLocalRouter")
        self._tool_call_parser, self._reasoning_parser = self._build_output_parsers()

    def _build_output_parsers(self) -> tuple[ToolCallParser | None, ReasoningParser | None]:
        tool_call_parser = None
        reasoning_parser = None

        if self.config.tool_call_parser != "none":
            tool_call_parser = build_tool_call_parser(self.config.tool_call_parser)

        if self.config.reasoning_parser != "none":
            tokenizer_path = self.config.tokenizer_path or self.config.model_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            reasoning_parser = build_reasoning_parser(self.config.reasoning_parser, tokenizer)

        return tool_call_parser, reasoning_parser

    def _apply_output_parsers(self, rollout_state: RolloutState) -> None:
        if self._tool_call_parser is not None:
            parsed = self._tool_call_parser.parse(rollout_state)
            rollout_state.tool_calls = parsed.tool_calls
            rollout_state.response = parsed.remaining_text or None
        if self._reasoning_parser is not None:
            parsed_reasoning = self._reasoning_parser.parse(rollout_state)
            rollout_state.response = parsed_reasoning.remaining_text
            if parsed_reasoning.reasoning_text:
                rollout_state.extra_fields["reasoning_text"] = parsed_reasoning.reasoning_text
            else:
                rollout_state.extra_fields.pop("reasoning_text", None)

    async def generate(self, rollout_state: RolloutState) -> RolloutState:
        if XTUNER_DETERMINISTIC:
            sample_params = rollout_state.sample_params.model_copy(deep=True)
            sample_params.sampling_seed = self.config.random_seed + (
                (rollout_state.uid or 0) - (rollout_state.message_uid or 0)
            )
            rollout_state.sample_params = sample_params

        session_id = rollout_state.session_uid if rollout_state.session_uid is not None else uuid4().int
        worker_info = await self.health_manager.get_worker_route_info.remote(session_id)
        worker = worker_info.actor if worker_info is not None else None
        if worker is None:
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = "No active rollout worker available."
            return rollout_state

        response_ref = worker.generate.remote(rollout_state=rollout_state)
        try:
            response_rollout_state = await asyncio.wait_for(
                response_ref, timeout=self.config.rollout_timeout * self.timeout_multiplier
            )
            self._apply_output_parsers(response_rollout_state)
            return response_rollout_state
        except (asyncio.TimeoutError, RayActorError) as e:
            await self.health_manager.report_worker_failure.remote(worker_info.rank, str(e))
            self.logger.error(f"Rollout failed for worker {worker}. Skipping sample. Error: {e}")
            rollout_state.status = Status.FAILED
            if isinstance(e, asyncio.TimeoutError):
                rollout_state.error_msg = (
                    f"Rollout request timed out after {self.config.rollout_timeout * self.timeout_multiplier} seconds."
                )
            else:
                rollout_state.error_msg = f"Rollout worker failed with error: {e}"
            return rollout_state

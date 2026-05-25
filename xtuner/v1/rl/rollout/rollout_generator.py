from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

import ray
from pydantic import BaseModel, ConfigDict, Field
from ray.exceptions import RayActorError
from transformers import AutoTokenizer

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger

from ._generation.external_http_entry import ExternalRolloutHttpEntryConfig
from ._generation.internal_http_entry import InternalRolloutHttpEntryConfig
from ._generation.session_worker_selector import RolloutWorkerHandle, RolloutWorkerUrlSource, SessionWorkerSelector
from .parser.factory import build_reasoning_parser, build_tool_call_parser
from .parser.reasoning_parser import ReasoningParser
from .parser.tool_parser import ToolCallParser
from .worker import RolloutConfig


class LocalRolloutGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    timeout_multiplier: float = Field(default=2.0, gt=0)

    def build(self, rollout_controller) -> "LocalRolloutGenerator":
        rollout_metadata = ray.get(rollout_controller.get_rollout_metadata.remote())
        return LocalRolloutGenerator(
            worker_handles=rollout_metadata["worker_handles"],
            rollout_config=rollout_metadata["rollout_config"],
            timeout_multiplier=self.timeout_multiplier,
        )


class LocalRolloutGenerator:
    """Local AgentLoop generation path.

    It chooses one active rollout worker by session id, then calls the worker's
    bound RolloutWorkerGenerator actor directly. The RolloutController is not
    on the runtime generation path.
    """

    def __init__(
        self,
        worker_handles: list[RolloutWorkerHandle],
        rollout_config: RolloutConfig,
        timeout_multiplier: float = 2.0,
    ) -> None:
        self.worker_handles = worker_handles
        self.worker_selector = SessionWorkerSelector(worker_handles)
        self.config = rollout_config
        self.timeout_multiplier = timeout_multiplier
        self.logger = get_logger(log_dir=rollout_config.worker_log_dir, tag="LocalRolloutGenerator")
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

    async def generate(self, rollout_state: RolloutState, *, enable_partial_rollout: bool = False) -> RolloutState:
        if XTUNER_DETERMINISTIC:
            sample_params = rollout_state.sample_params.model_copy(deep=True)
            sample_params.sampling_seed = self.config.random_seed + (
                (rollout_state.uid or 0) - (rollout_state.message_uid or 0)
            )
            rollout_state.sample_params = sample_params

        session_id = rollout_state.session_uid if rollout_state.session_uid is not None else uuid4().int
        worker = await self.worker_selector.select(session_id)
        if worker is None:
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = "No rollout worker available."
            return rollout_state

        try:
            response_rollout_state = await asyncio.wait_for(
                worker.generator_actor.generate.remote(
                    rollout_state=rollout_state,
                    enable_partial_rollout=enable_partial_rollout,
                ),
                timeout=self.config.rollout_timeout * self.timeout_multiplier,
            )
            self._apply_output_parsers(response_rollout_state)
            return response_rollout_state
        except (asyncio.TimeoutError, RayActorError) as exc:
            self.logger.error(f"Rollout failed for worker {worker.rank}. Skipping sample. Error: {exc}")
            rollout_state.status = Status.FAILED
            if isinstance(exc, asyncio.TimeoutError):
                rollout_state.error_msg = (
                    f"Rollout request timed out after {self.config.rollout_timeout * self.timeout_multiplier} seconds."
                )
            else:
                rollout_state.error_msg = f"Rollout worker generator actor failed with error: {exc}"
            return rollout_state


RolloutGenerateKind = Literal["local", "http"]
RolloutHttpEntryKind = Literal["internal", "external"]


@dataclass
class RolloutGenerateHandle:
    kind: RolloutGenerateKind
    local_generator: LocalRolloutGenerator | None = None
    base_url: str | None = None
    rollout_controller: Any | None = None

    def require_local_generator(self) -> LocalRolloutGenerator:
        if self.local_generator is None:
            raise RuntimeError(f"Rollout generate handle {self.kind!r} does not provide a local generator.")
        return self.local_generator

    def require_base_url(self) -> str:
        if self.base_url is None:
            raise RuntimeError(f"Rollout generate handle {self.kind!r} does not provide a base URL.")
        return self.base_url


class RolloutGenerateHandleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kind: RolloutGenerateKind = "local"
    http_entry: RolloutHttpEntryKind = "internal"
    local_generator_config: LocalRolloutGeneratorConfig = Field(default_factory=LocalRolloutGeneratorConfig)

    base_url: str | None = None

    internal_http_entry_host: str = "0.0.0.0"
    internal_http_entry_port: int = 8081
    internal_http_entry_title: str = "XTuner Internal Rollout Router"
    internal_http_entry_version: str = "0.1.0"
    internal_http_entry_log_level: str = "warning"
    internal_http_entry_request_timeout: float | None = None
    internal_http_entry_stream_timeout: float | None = None
    http_worker_url_source: RolloutWorkerUrlSource = "backend"

    external_http_entry_delete_existing: bool = True
    external_http_entry_check_worker_urls: bool = True
    external_http_entry_check_base_url: bool = True

    def build(self, rollout_controller) -> RolloutGenerateHandle:
        if self.kind == "local":
            return RolloutGenerateHandle(
                kind=self.kind,
                local_generator=self.local_generator_config.build(rollout_controller),
                rollout_controller=rollout_controller,
            )

        if self.kind != "http":
            raise ValueError(f"Unsupported rollout generate kind: {self.kind!r}")

        base_url = self._resolve_http_base_url(rollout_controller)
        if base_url is None:
            raise ValueError(f"Rollout generate handle {self.kind!r} requires base_url.")
        return RolloutGenerateHandle(kind=self.kind, base_url=base_url, rollout_controller=rollout_controller)

    def _resolve_http_base_url(self, rollout_controller) -> str | None:
        if self.base_url is not None:
            return self.base_url

        if self.http_entry == "internal":
            rollout_metadata = ray.get(rollout_controller.get_rollout_metadata.remote())
            base_url = rollout_metadata.get("internal_http_entry_url")
            if base_url is None:
                raise ValueError(
                    "Rollout generate handle kind='http' and http_entry='internal' requires the internal HTTP "
                    "entry to be started before building AgentLoop, or base_url to be provided as a runtime override."
                )
            return base_url

        if self.http_entry == "external":
            raise ValueError("Rollout generate handle kind='http' and http_entry='external' requires base_url.")

        raise ValueError(f"Unsupported rollout HTTP entry: {self.http_entry!r}")

    def build_internal_http_entry_config(self) -> InternalRolloutHttpEntryConfig | None:
        if self.kind != "http" or self.http_entry != "internal":
            return None
        return InternalRolloutHttpEntryConfig(
            host=self.internal_http_entry_host,
            port=self.internal_http_entry_port,
            title=self.internal_http_entry_title,
            version=self.internal_http_entry_version,
            log_level=self.internal_http_entry_log_level,
            request_timeout=self.internal_http_entry_request_timeout,
            stream_timeout=self.internal_http_entry_stream_timeout,
            worker_url_source=self.http_worker_url_source,
        )

    def build_external_http_entry_config(self) -> ExternalRolloutHttpEntryConfig | None:
        if self.kind != "http" or self.http_entry != "external":
            return None
        if self.base_url is None:
            raise ValueError("Rollout generate handle kind='http' and http_entry='external' requires base_url.")
        return ExternalRolloutHttpEntryConfig(
            base_url=self.base_url,
            worker_url_source=self.http_worker_url_source,
            delete_existing=self.external_http_entry_delete_existing,
            check_worker_urls=self.external_http_entry_check_worker_urls,
            check_base_url=self.external_http_entry_check_base_url,
        )

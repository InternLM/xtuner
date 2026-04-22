from __future__ import annotations

import asyncio
import copy
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from pydantic import ConfigDict, Field

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop.agent_loop import AgentLoop, AgentLoopConfig
from xtuner.v1.rl.judger.native import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.utils import chat_trace_records_to_rollout_states


DEFAULT_READONLY_INSTRUCTION = (
    "You are running inside an automated rollout collection job. "
    "Work in read-only mode: inspect files and report findings, but do not edit, create, delete, move, "
    "format, commit, push, install dependencies, or run commands that write to the repository or external services."
)


class ClaudeCodeAgentLoopConfig(AgentLoopConfig):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    claude_command: list[str] = Field(default_factory=lambda: ["$HOME/.local/bin/claude"])
    cwd: str | None = None
    timeout_s: float = 600.0
    api_timeout_ms: int = 600000
    max_turns: int = 5
    output_format: str = "json"
    permission_mode: str = "plan"
    tools: str | None = "Read,Grep,Glob,LS,Bash"
    allowed_tools: str | None = None
    disallowed_tools: str | None = "Edit,Write,MultiEdit,NotebookEdit"
    mcp_config: list[str] = Field(default_factory=list)
    strict_mcp_config: bool = False
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    readonly_instruction: str = DEFAULT_READONLY_INSTRUCTION
    bare: bool = True
    extra_env: dict[str, str] = Field(default_factory=dict)

    def build_local(
        self,
        rollout_controller,
        judger: Judger | None = None,
        logger=None,
    ) -> ClaudeCodeAgentLoop:
        return ClaudeCodeAgentLoop(
            claude_command=self.claude_command,
            cwd=self.cwd,
            timeout_s=self.timeout_s,
            api_timeout_ms=self.api_timeout_ms,
            max_turns=self.max_turns,
            output_format=self.output_format,
            permission_mode=self.permission_mode,
            tools=self.tools,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            mcp_config=self.mcp_config,
            strict_mcp_config=self.strict_mcp_config,
            system_prompt=self.system_prompt,
            append_system_prompt=self.append_system_prompt,
            readonly_instruction=self.readonly_instruction,
            bare=self.bare,
            extra_env=self.extra_env,
            rollout_ctl=rollout_controller,
            sample_params=self.sample_params,
            hf_checkpoint=self.hf_checkpoint,
            judger=judger,
            logger=logger,
        )


class ClaudeCodeAgentLoop(AgentLoop):
    def __init__(
        self,
        claude_command: list[str],
        cwd: str | None,
        timeout_s: float,
        api_timeout_ms: int,
        max_turns: int,
        output_format: str,
        permission_mode: str,
        tools: str | None,
        allowed_tools: str | None,
        disallowed_tools: str | None,
        mcp_config: list[str],
        strict_mcp_config: bool,
        system_prompt: str | None,
        append_system_prompt: str | None,
        readonly_instruction: str,
        bare: bool,
        extra_env: dict[str, str],
        rollout_ctl: RolloutController,
        sample_params: SampleParams,
        hf_checkpoint: str,
        judger: Judger | None = None,
        logger=None,
    ) -> None:
        super().__init__(
            rollout_ctl=rollout_ctl,
            sample_params=sample_params,
            hf_checkpoint=hf_checkpoint,
            judger=judger,
            logger=logger,
        )
        self.claude_command = claude_command
        self.cwd = cwd
        self.timeout_s = timeout_s
        self.api_timeout_ms = api_timeout_ms
        self.max_turns = max_turns
        self.output_format = output_format
        self.permission_mode = permission_mode
        self.tools = tools
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools
        self.mcp_config = mcp_config
        self.strict_mcp_config = strict_mcp_config
        self.system_prompt = system_prompt
        self.append_system_prompt = append_system_prompt
        self.readonly_instruction = readonly_instruction
        self.bare = bare
        self.extra_env = extra_env

    async def generate_sample(  # type: ignore[override]
        self, rollout_state: RolloutState, **kwargs
    ) -> list[RolloutState]:
        try:
            metadata = await self.rollout_ctl.get_rollout_metadata.remote()  # type: ignore[attr-defined]
            gateway_url = metadata.get("api_server_url")
            rollout_config = metadata.get("rollout_config")
            model_name = getattr(rollout_config, "model_name", None) or "rollout-controller"
            if not gateway_url:
                return [
                    self._failed_state(
                        rollout_state,
                        "Gateway is not started. Configure GatewayConfig(auto_start=True) "
                        "before using ClaudeCodeAgentLoop.",
                    )
                ]

            api_key = f"claudecode_{uuid4().hex}"
            command = self._build_command(rollout_state, model_name=model_name)
            returncode, stdout, stderr = await self._run_claude(command, gateway_url, model_name, api_key)
            records = await self._pop_trace_store_records(gateway_url, api_key)
            rollout_extra_fields = {
                "claudecode_api_key": api_key,
                "claudecode_cli_returncode": returncode,
                "claudecode_cli_stdout": self._truncate(stdout),
                "claudecode_cli_stderr": self._truncate(stderr),
            }

            if not records:
                reason = "Claude Code finished without trace store records for this api_key."
                if returncode != 0:
                    reason += f" returncode={returncode}, stderr={self._truncate(stderr)}"
                return [self._failed_state(rollout_state, reason, extra_fields=rollout_extra_fields)]

            reward = None
            if self.judger is not None:
                judge_state = rollout_state.model_copy(deep=True)
                judge_state.extra_fields = {
                    **copy.deepcopy(rollout_state.extra_fields),
                    **copy.deepcopy(rollout_extra_fields),
                }
                judged_state = await self.judger.judge(judge_state)
                if judged_state.reward is None:
                    return [
                        self._failed_state(
                            rollout_state,
                            "Judger completed without setting reward.",
                            extra_fields=rollout_extra_fields,
                        )
                    ]
                reward = copy.deepcopy(judged_state.reward)

            states = chat_trace_records_to_rollout_states(
                rollout_state=rollout_state,
                records=records,
                tokenizer=self.tokenizer,
                extra_fields=rollout_extra_fields,
            )
            if not states:
                return [
                    self._failed_state(
                        rollout_state,
                        "Gateway trace records did not contain trainable turns.",
                        extra_fields=rollout_extra_fields,
                    )
                ]

            completed_states = [state for state in states if state.status == Status.COMPLETED]
            if reward is not None:
                for state in completed_states:
                    state.reward = copy.deepcopy(reward)
            return states
        except Exception as exc:
            return [self._failed_state(rollout_state, f"ClaudeCodeAgentLoop failed: {exc}")]

    def _build_command(self, rollout_state: RolloutState, *, model_name: str) -> list[str]:
        command = [os.path.expandvars(os.path.expanduser(part)) for part in self.claude_command]
        prompt = self._build_prompt(rollout_state)
        if self.bare:
            command.append("--bare")
        if self.system_prompt:
            command.extend(["--system-prompt", self.system_prompt])
        if self.append_system_prompt:
            command.extend(["--append-system-prompt", self.append_system_prompt])
        for config in self.mcp_config:
            command.extend(["--mcp-config", os.path.expandvars(os.path.expanduser(config))])
        if self.strict_mcp_config:
            command.append("--strict-mcp-config")
        command.extend(
            [
                "-p",
                prompt,
                "--output-format",
                self.output_format,
                "--permission-mode",
                self.permission_mode,
                "--model",
                model_name,
                "--max-turns",
                str(self.max_turns),
                "--no-session-persistence",
            ]
        )
        if self.tools is not None:
            command.extend(["--tools", self.tools])
        if self.allowed_tools:
            command.extend(["--allowedTools", self.allowed_tools])
        if self.disallowed_tools:
            command.extend(["--disallowedTools", self.disallowed_tools])
        return command

    def _build_prompt(self, rollout_state: RolloutState) -> str:
        content = ""
        for message in reversed(rollout_state.message):
            if message.get("role") == "user":
                content = self._message_content_to_text(message.get("content"))
                break
        if not content and rollout_state.message:
            content = self._message_content_to_text(rollout_state.message[-1].get("content"))
        if not self.readonly_instruction:
            return content
        return f"{self.readonly_instruction}\n\nTask:\n{content}"

    def _message_content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                    elif item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    async def _run_claude(
        self,
        command: list[str],
        gateway_url: str,
        model_name: str,
        api_key: str,
    ) -> tuple[int, str, str]:
        env = os.environ.copy()
        env.update(
            {
                "ANTHROPIC_BASE_URL": gateway_url,
                "ANTHROPIC_AUTH_TOKEN": api_key,
                "ANTHROPIC_API_KEY": api_key,
                "ANTHROPIC_MODEL": model_name,
                "API_TIMEOUT_MS": str(self.api_timeout_ms),
                "PATH": f"{Path.home() / '.local' / 'bin'}:{env.get('PATH', '')}",
            }
        )
        env.update({key: os.path.expandvars(os.path.expanduser(value)) for key, value in self.extra_env.items()})

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(Path(self.cwd or os.getcwd()).resolve()),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            returncode = process.returncode if process.returncode is not None else -9
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            stderr = f"Claude Code timed out after {self.timeout_s}s.\n{stderr}"
            return returncode, stdout_bytes.decode("utf-8", errors="replace"), stderr

        return (
            process.returncode if process.returncode is not None else 0,
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
        )

    async def _pop_trace_store_records(self, gateway_url: str, api_key: str) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{gateway_url.rstrip('/')}/trace_store/pop",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
        records = payload.get("records", [])
        if not isinstance(records, list):
            return []
        return records

    def _failed_state(
        self,
        rollout_state: RolloutState,
        error_msg: str,
        *,
        extra_fields: dict[str, Any] | None = None,
    ) -> RolloutState:
        failed = rollout_state.model_copy(deep=True)
        failed.status = Status.FAILED
        failed.error_msg = error_msg
        if extra_fields:
            failed.extra_fields = {
                **copy.deepcopy(rollout_state.extra_fields),
                **copy.deepcopy(extra_fields),
            }
        return failed

    def _truncate(self, text: str, max_chars: int = 4096) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...<truncated>"

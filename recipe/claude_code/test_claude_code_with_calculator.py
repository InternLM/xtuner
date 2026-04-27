from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from calculator_tool import (
    CalculatorJudger,
    CALCULATOR_PROMPT,
    CALCULATOR_SYSTEM_PROMPT,
    CALCULATOR_TOOL_NAME,
    normalize_answer,
    write_calculator_mcp_server,
)
from claudecode_agent_loop import ClaudeCodeAgentLoopConfig
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.gateway import wait_for_gateway_ready
from xtuner.v1.rl.utils import find_free_ports


RESOURCE_MAP = {
    "npu": "NPU",
    "cuda": "GPU",
}


async def test_claude_code_with_calculator(model_path: str) -> list[RolloutState]:
    import ray

    os.environ.setdefault("XTUNER_USE_FA3", "1")
    os.environ.setdefault("LMDEPLOY_SKIP_WARMUP", "1")
    os.environ.pop("RAY_ADDRESS", None)
    ray.init(address="local", ignore_reinit_error=True)

    temp_dir = tempfile.TemporaryDirectory()
    work_dir = Path(temp_dir.name)
    worker_log_dir = work_dir / "work_dirs"
    output_dir = Path(os.environ.get("XTUNER_CLAUDECODE_TOOL_OUTPUT_DIR", work_dir / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    controller = None
    placement_group = None

    try:
        _, mcp_config_path = write_calculator_mcp_server(work_dir)
        gateway_url, controller, placement_group = _start_rollout_controller_and_gateway(
            model_path=model_path,
            worker_log_dir=worker_log_dir,
        )
        cfg = ClaudeCodeAgentLoopConfig(
            hf_checkpoint=model_path,
            sample_params=SampleParams(
                max_tokens=int(os.environ.get("XTUNER_CLAUDECODE_TOOL_MAX_TOKENS", "1024")),
                temperature=0.0,
            ),
            claude_command=[os.environ.get("XTUNER_CLAUDE_BIN", str(Path.home() / ".local" / "bin" / "claude"))],
            cwd=str(work_dir),
            timeout_s=float(os.environ.get("XTUNER_CLAUDECODE_TOOL_TIMEOUT_S", "600")),
            api_timeout_ms=int(os.environ.get("XTUNER_CLAUDECODE_TOOL_API_TIMEOUT_MS", "600000")),
            max_turns=int(os.environ.get("XTUNER_CLAUDECODE_TOOL_MAX_TURNS", "4")),
            output_format="json",
            permission_mode=os.environ.get("XTUNER_CLAUDECODE_PERMISSION_MODE", "bypassPermissions"),
            tools=None,
            allowed_tools=CALCULATOR_TOOL_NAME,
            disallowed_tools="Bash,Edit,Read,Grep,Glob,LS,WebFetch,WebSearch",
            mcp_config=[str(mcp_config_path)],
            strict_mcp_config=True,
            system_prompt=CALCULATOR_SYSTEM_PROMPT,
            readonly_instruction="",
        )
        agent_loop = cfg.build(rollout_controller=controller, judger=CalculatorJudger())
        rollout_state = RolloutState(
            message=[{"role": "user", "content": CALCULATOR_PROMPT}],
            task_name="calculator_tool_call",
            extra_fields={"gateway_url": gateway_url},
        )

        states = await agent_loop.generate_sample(rollout_state)
        _dump_rollout_states(output_dir, "calculator", states)
        failed = [state for state in states if state.status == Status.FAILED]
        if failed:
            raise AssertionError(f"ClaudeCodeAgentLoop returned failed states: {[state.error_msg for state in failed]}")

        completed = [state for state in states if state.status == Status.COMPLETED]
        if len(completed) < 2:
            raise AssertionError("Expected one tool-call turn and one final-answer turn.")

        api_keys = {state.extra_fields["claudecode_api_key"] for state in completed}
        if len(api_keys) != 1:
            raise AssertionError(f"Expected one Claude Code api key, got {api_keys}.")
        for index, state in enumerate(completed):
            if state.prompt_ids is None:
                raise AssertionError(f"State {index} is missing prompt_ids.")
            if state.tokens != state.prompt_ids:
                raise AssertionError(f"State {index} tokens must equal prompt_ids.")
            if not state.response_ids:
                raise AssertionError(f"State {index} is missing response_ids.")
            if state.logprobs is None:
                raise AssertionError(f"State {index} is missing logprobs.")
            if len(state.logprobs) != len(state.response_ids):
                raise AssertionError(f"State {index} logprobs length does not match response_ids length.")
            if state.response_mask != [1] * len(state.response_ids):
                raise AssertionError(f"State {index} response_mask does not match response_ids length.")
            if state.response is None:
                raise AssertionError(f"State {index} is missing response text.")
            if "gateway_trace_records" not in state.extra_fields:
                raise AssertionError(f"State {index} is missing gateway_trace_records.")
            if state.extra_fields["gateway_trace_count"] != len(states):
                raise AssertionError(f"State {index} gateway_trace_count does not match trace count.")
            if state.extra_fields["claudecode_cli_returncode"] != 0:
                raise AssertionError(f"State {index} Claude Code returncode is not 0.")

        tool_blocks = []
        for state in completed:
            snapshot = state.extra_fields.get("gateway_response_snapshot") or {}
            for block in snapshot.get("content") or []:
                if block.get("type") == "tool_use":
                    tool_blocks.append(block)
        if not tool_blocks:
            raise AssertionError("Expected at least one calculator tool_use block.")
        calculator_blocks = [block for block in tool_blocks if "calculator" in str(block.get("name"))]
        if not calculator_blocks:
            raise AssertionError(f"Expected calculator tool call, got: {tool_blocks}")
        if calculator_blocks[0].get("input", {}).get("expression") != "23 + 19":
            raise AssertionError(f"Unexpected calculator input: {calculator_blocks[0].get('input')}")

        final_answer = normalize_answer(completed[-1].response)
        if final_answer != "42":
            raise AssertionError(f"Expected final answer 42, got {final_answer!r}.")
        if completed[-1].reward != {"score": 1.0, "answer": "42"}:
            raise AssertionError(f"Expected reward score 1.0 for answer 42, got {completed[-1].reward!r}.")
        return states
    finally:
        _cleanup_ray(controller=controller, placement_group=placement_group)
        temp_dir.cleanup()


def _start_rollout_controller_and_gateway(
    *,
    model_path: str,
    worker_log_dir: Path,
) -> tuple[str, Any, Any]:
    import ray
    import torch

    from xtuner.v1.rl.gateway.config import GatewayConfig
    from xtuner.v1.rl.rollout.worker import RolloutConfig
    from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers

    accelerator = RESOURCE_MAP[torch.accelerator.current_accelerator().type]
    tensor_parallel_size = int(os.environ.get("XTUNER_CLAUDECODE_TOOL_TP", "1"))
    num_workers = int(os.environ.get("XTUNER_CLAUDECODE_TOOL_NUM_WORKERS", str(tensor_parallel_size)))
    resource_config = AcceleratorResourcesConfig(
        accelerator=accelerator,
        num_workers=num_workers,
        num_cpus_per_worker=int(os.environ.get("XTUNER_CLAUDECODE_TOOL_CPUS_PER_WORKER", "8")),
        cpu_memory_per_worker=int(os.environ.get("XTUNER_CLAUDECODE_TOOL_CPU_MEMORY", str(16 * 1024**3))),
    )
    placement_group = AutoAcceleratorWorkers.build_placement_group(
        resource_config,
        name=f"claudecode_tool_pg_{uuid4().hex[:8]}",
    )
    rollout_config = RolloutConfig(
        env=f"claudecode_tool_{uuid4().hex[:8]}",
        model_path=model_path,
        model_name=os.path.basename(model_path).lower(),
        tokenizer_path=model_path,
        context_length=int(os.environ.get("XTUNER_CLAUDECODE_TOOL_CONTEXT_LENGTH", "32768")),
        worker_log_dir=worker_log_dir / "rollout",
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=1,
        dist_port_base=int(
            os.environ.get("XTUNER_CLAUDECODE_TOOL_DIST_PORT_BASE", str(find_free_ports(nums=8, contiguous=True)[0]))
        ),
        tool_call_parser=os.environ.get("XTUNER_CLAUDECODE_TOOL_CALL_PARSER", "qwen3p5"),
        reasoning_parser=os.environ.get("XTUNER_CLAUDECODE_REASONING_PARSER", "qwen3"),
        api_host="127.0.0.1",
        api_port=find_free_ports()[0],
    )
    controller = rollout_config.build(placement_group)
    gateway_host = ray.util.get_node_ip_address()
    gateway_config = GatewayConfig(
        host=gateway_host,
        port=find_free_ports(host=gateway_host)[0],
        capture_folder=str(worker_log_dir / "gateway_captures"),
    )
    gateway_url = ray.get(controller.start_gateway.remote(gateway_config), timeout=1800)
    wait_for_gateway_ready(gateway_url)
    return gateway_url, controller, placement_group


def _dump_rollout_states(output_dir: Path, case_name: str, states: list[RolloutState]) -> None:
    output_path = output_dir / f"rollout-states-{case_name}.json"
    payload = [_redact_rollout_state_for_dump(state) for state in states]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Claude Code rollout states written to {output_path}")


def _redact_rollout_state_for_dump(state: RolloutState) -> dict:
    payload = state.model_dump(mode="json")
    for key in ("prompt_ids", "response_ids"):
        payload.pop(key, None)
    extra_fields = payload.get("extra_fields")
    if isinstance(extra_fields, dict):
        records = extra_fields.get("gateway_trace_records")
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    record.pop("prompt_ids", None)
                    record.pop("response_ids", None)
                    record.pop("tokens", None)
    return payload


def _cleanup_ray(*, controller: Any, placement_group: Any) -> None:
    import ray

    if controller is not None:
        try:
            ray.get(controller.shutdown.remote(), timeout=300)
        except Exception:
            pass
        try:
            ray.kill(controller, no_restart=True)
        except Exception:
            pass
    if placement_group is not None:
        ray.util.remove_placement_group(placement_group)
    if ray.is_initialized():
        ray.shutdown()


async def _main_async() -> None:
    states = await test_claude_code_with_calculator(model_path=os.environ["ROLLOUT_MODEL_PATH"])
    completed_count = sum(state.status == Status.COMPLETED for state in states)
    print(f"Claude Code calculator tool E2E passed with {completed_count} completed rollout states.")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()

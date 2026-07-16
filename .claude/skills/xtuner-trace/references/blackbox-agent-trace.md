# Blackbox Agent Trace Reference

This reference captures the XTuner black-box agent trace reconstruction workflow. It is intended for implementation and review work, not as user-facing docs.

Read this reference when `xtuner-trace/SKILL.md` routes an agent trace request here. The design background lives in `docs/design/blackbox_agent_trace_reconstruction.md`.

## Contents

- [Decision Flow](#decision-flow)
- [Path A: Trajectory-Backed Reconstruction](#path-a-trajectory-backed-reconstruction)
- [Path B: SessionServer-Only Fallback](#path-b-sessionserver-only-fallback)
- [`record_synthetic_span(...)` Usage](#record_synthetic_span-usage)
- [Reconstructed Span Shape](#reconstructed-span-shape)
- [Viewer And Jaeger Requirements](#viewer-and-jaeger-requirements)
- [Example: mini-SWE-agent Trajectory Conversion](#example-mini-swe-agent-trajectory-conversion)
- [Verification](#verification)
- [Common Failure Modes](#common-failure-modes)

## Decision Flow

When adding trace support for a black-box agent, classify the available evidence before deciding where to instrument.

1. Confirm this is a black-box agent path: XTuner exposes an OpenAI-compatible inference endpoint to the agent through `SessionServer`; the agent runs its native loop and calls that endpoint for LLM generation. Tool execution, environment interaction, and observation handling happen inside the agent runtime and are not directly instrumented by XTuner.
2. Confirm whether the agent can save a trajectory for each run.
3. If a trajectory exists, check whether it contains tool-call metadata such as tool name, action/command, status, return code, error information, observation, or timestamp.
4. Use Path A when the trajectory contains tool-call metadata. The trajectory supplies tool semantics, and `session_server.request` spans supply the LLM request timeline.
5. Use Path B only when no useful trajectory exists. This can recover request gaps, but cannot recover reliable tool names, return codes, errors, or observations.

Do not require the trajectory to contain complete LLM generate timing before using Path A. `SessionServer` remains the stable source for LLM request start/end times. If the trajectory also records LLM calls, use them as an optional consistency check.

## Path A: Trajectory-Backed Reconstruction

Use this path when the agent can save a trajectory with tool-call metadata.

### Data Contract

The `session_server.request` spans must provide:

- `xtuner.session_id` or `xtuner.rollout_id`
- span start time
- span end time

The trajectory should provide best-effort tool metadata:

- tool event order, id, or enough context to infer order
- tool name, action, or command
- status, return code, or exception information
- observation or output summary
- tool start/end timestamps, or a single tool-related timestamp with documented semantics, if the agent records one

Do not define a universal trajectory schema in trace core. Agent trajectory formats are agent-specific; keep parsing inside that agent's recipe or trajectory-to-trace adapter until multiple agents prove a shared model is worth extracting.

### Where To Add Trace

Add trace at the three black-box observation boundaries. These instructions must be enough to re-create the instrumentation even when no example implementation exists in the checkout.

#### 1. SessionServer Request Trace

First check whether the `SessionServer` HTTP request handler already emits `session_server.request` spans. If not, add this instrumentation to the handler that receives the agent's OpenAI-compatible request.

Implementation requirements:

- Locate the outer request handler, such as an aiohttp/FastAPI handler named like `_handle_request(request)`.
- Read enough request headers/body to determine:
  - `session_id`
  - optional rollout id
  - whether the request is streaming
  - optional trace carrier from standard headers or a request-body carrier field
- Wrap the full inference request lifecycle in:

```python
with trace_span(
    "session_server.request",
    attributes={
        "xtuner.stage": "llm_generate",
        "xtuner.session_id": session_id,
        "xtuner.rollout_id": rollout_id or session_id,
        "session.stream": is_streaming,
    },
    parent_carrier=parent_carrier,
):
    ...
```

- Keep the outer `session_server.request` span around prepare, backend request, response read, response post-processing, and return-to-agent.
- Use `set_trace_attributes(...)` before exiting the span to record request-level status or error metadata when available.
- If adding child spans, use these names:
  - `session_server.prepare_request`
  - `session_server.send_request`
  - `session_server.read_response`
  - `session_server.record_response`
- Put `xtuner.session_id` on every child span too. This makes later trajectory-to-trace adapters able to filter a single agent run reliably.

Do not parse agent trajectory or infer tool calls inside `SessionServer`. Its black-box responsibility is to provide the LLM request timeline.

#### 2. Agent Run Boundary Trace

In the black-box recipe or launcher, find the call site that invokes the agent's native run loop, such as `agent.run(...)`.

Wrap only the outer run boundary:

```python
with trace_span(
    "<agent_name>.run",
    attributes={
        "xtuner.stage": "agent.run",
        "xtuner.session_id": session_id,
        "xtuner.rollout_id": rollout_id or session_id,
        "agent.name": agent_name,
    },
):
    agent.run(...)
```

This span is a lifecycle boundary. Do not wrap the agent's internal model, environment, or tool methods as the default black-box trace path. LLM/tool detail should come from `SessionServer` spans plus trajectory reconstruction.

#### 3. Trajectory-To-Trace Reconstruction

After the native agent run finishes, call the trajectory-to-trace adapter before `close_trace()`. Prefer a `finally` block so partial trajectories can still be converted after failures:

```python
return_code = 1
try:
    with trace_span("<agent_name>.run", attributes=agent_run_attributes):
        return_code = agent.run(...)
finally:
    emit_reconstructed_trace(
        trace_jsonl=trace_jsonl_path,
        trajectory=trajectory_path,
        session_id=session_id,
    )
    close_trace()
return return_code
```

The adapter must read the trajectory, read same-session `session_server.request` spans, derive tool timing from trajectory timestamps when possible, use request gaps only for correlation or inferred fallback, and emit reconstructed synthetic spans with `record_synthetic_span(...)`.

### Adapter Responsibilities

Write a trajectory-to-trace adapter for the specific agent. The adapter should:

1. Load the agent trajectory.
2. Extract ordered tool events and their metadata.
3. Load `session_server.request` spans from the same `traces.jsonl`.
4. Filter request spans by `session_id`.
5. Sort request spans by start time and assign local indexes.
6. Extract tool timestamps from the trajectory and interpret their semantics:
   - If the trajectory provides reliable tool start/end times, use them as the `reconstructed_tool_call.N` span boundaries and set `xtuner.inferred=false`.
   - If the trajectory provides only one timestamp, record what it means if known, such as tool start, tool end, observation write, or trajectory write. Use it to correlate the tool event with the surrounding `session_server.request` window.
   - If the trajectory timestamp semantics are unknown or incomplete, use request-gap boundaries only as an inferred fallback and set `xtuner.inferred=true`.
7. Use `session_server.request` spans as the LLM timeline and correlation windows, not as the default source of trajectory-backed tool timing.
8. When using request-gap fallback for a bounded tool event, use:

```text
tool_i.start = request_i.end
tool_i.end   = request_{i+1}.start
```

9. Emit `reconstructed_agent_run`, `reconstructed_llm_call.N`, and bounded `reconstructed_tool_call.N` spans with `record_synthetic_span(...)`.
10. Treat final tool events without a next LLM request as unbounded unless the trajectory has trustworthy start/end times.

When request-gap fallback is used, the interval is not pure tool runtime. It can include LLM-response parsing, user confirmation time, tool execution, observation formatting, trajectory write time, and loop overhead before the next LLM request.

## Path B: SessionServer-Only Fallback

Use this path only when the agent does not save a useful trajectory.

### Where To Add Trace

Add trace to the `SessionServer` request path so every LLM request produces a reliable timeline. At minimum, `session_server.request` spans must include:

- `xtuner.session_id`
- `xtuner.rollout_id`, falling back to the session id when there is no rollout id
- request start time
- request end time

Supporting spans such as `session_server.prepare_request`, `session_server.send_request`, `session_server.read_response`, and `session_server.record_response` may remain useful for debugging latency, but the fallback reconstruction only requires the outer `session_server.request` start/end times.

Do not put tool inference inside `SessionServer`. `SessionServer` should not parse trajectories, infer tool names, track return codes, or maintain complex per-session reconstruction state.

### Fallback Reconstruction

After the agent run, a fallback adapter may infer intervals from adjacent request spans:

```text
inferred_tool_call_i.start = request_i.end
inferred_tool_call_i.end   = request_{i+1}.start
```

The fallback can emit synthetic spans such as `inferred_tool_call.N` with:

```text
xtuner.stage=tool_call
xtuner.synthetic=true
xtuner.inferred=true
xtuner.session_id=<session id>
reconstruction.source=session_server_only
reconstruction.method=session_server_gap
```

Do not claim unavailable metadata in this mode. SessionServer-only fallback cannot reliably recover tool name, command, return code, stderr/stdout, agent exception, skipped action, or observation content.

Do not infer the final tool event without a next LLM request. The interval has no reliable end time.

## `record_synthetic_span(...)` Usage

Use `record_synthetic_span(...)` to write reconstructed historical intervals back into the active XTuner trace runtime.

```python
record_synthetic_span(
    name: str,
    *,
    start_time_unix_ns: int,
    end_time_unix_ns: int,
    attributes: Mapping[str, Any] | None = None,
    parent_carrier: Mapping[str, str] | None = None,
    status: str = "completed",
    error_message: str | None = None,
) -> dict[str, str] | None
```

Rules:

- Call it before `close_trace()`.
- Use absolute Unix nanoseconds, not monotonic timestamps.
- Ensure `end_time_unix_ns > start_time_unix_ns`.
- Let the API default `xtuner.synthetic=true`.
- Set `xtuner.inferred=true` only when the time boundary is inferred, such as from `session_server.request` gaps.
- Use `status="error"` and `error_message=...` for real errors.
- Use `parent_carrier` to attach reconstructed children under one synthetic root trace.
- Treat `None` return as trace-disabled no-op.

Typical root and child pattern:

```python
root_ids = record_synthetic_span(
    "reconstructed_agent_run",
    start_time_unix_ns=first_request.start_time_unix_ns,
    end_time_unix_ns=last_request.end_time_unix_ns,
    attributes={
        "xtuner.stage": "agent.run",
        "xtuner.session_id": session_id,
        "xtuner.inferred": False,
        "reconstruction.method": "trajectory_session_timeline",
    },
)
root_carrier = None
if root_ids:
    root_carrier = {"traceparent": f"00-{root_ids['trace_id']}-{root_ids['span_id']}-01"}

record_synthetic_span(
    "reconstructed_tool_call.1",
    start_time_unix_ns=request_1.end_time_unix_ns,
    end_time_unix_ns=request_2.start_time_unix_ns,
    attributes={
        "xtuner.stage": "tool_call",
        "xtuner.session_id": session_id,
        "xtuner.inferred": True,
        "agent.tool_call_index": 1,
        "agent.tool.name": "bash",
        "reconstruction.source": "agent_trajectory",
        "reconstruction.method": "session_server_gap",
    },
    parent_carrier=root_carrier,
)
```

Do not call OpenTelemetry SDK directly from the adapter. Use XTuner trace APIs.

## Reconstructed Span Shape

Trajectory-backed reconstruction should generate one synthetic root trace:

```text
reconstructed_agent_run
|-- reconstructed_llm_call.1
|-- reconstructed_tool_call.1
|-- reconstructed_llm_call.2
|-- reconstructed_tool_call.2
...
```

Use these naming rules:

- `reconstructed_agent_run` for the synthetic root.
- `reconstructed_llm_call.N` for LLM request intervals copied from `session_server.request`.
- `reconstructed_tool_call.N` for trajectory-backed tool metadata.
- `inferred_tool_call.N` only for SessionServer-only fallback without trajectory metadata.

Recommended root attributes:

```text
xtuner.stage=agent.run
xtuner.synthetic=true
xtuner.inferred=false
xtuner.session_id=<session id>
xtuner.rollout_id=<session or rollout id>
agent.name=<agent name>
agent.llm_call_count=<count>
agent.tool_call_count=<count>
reconstruction.source=<agent trajectory source>
reconstruction.method=trajectory_session_timeline
```

Recommended LLM attributes:

```text
xtuner.stage=llm_generate
xtuner.synthetic=true
xtuner.inferred=false
agent.llm_call_index=N
reconstruction.source=session_server_request
reconstruction.method=session_server_request_copy
reconstruction.original_trace_id=<raw request trace id>
reconstruction.original_span_id=<raw request span id>
reconstruction.original_operation=session_server.request
```

Recommended tool attributes:

```text
xtuner.stage=tool_call
xtuner.synthetic=true
xtuner.inferred=true
agent.tool_call_index=N
agent.tool.name=<tool name>
agent.tool.status=<completed|error|skipped>
agent.tool.returncode=<return code, if available>
agent.tool.command_preview=<short command/action preview>
reconstruction.source=<agent trajectory source>
reconstruction.method=session_server_gap
reconstruction.timestamp_check=<inside_gap|outside_gap|missing>
```

Do not call trajectory-backed spans `inferred_tool_call.N`. Use `reconstructed_tool_call.N` and put uncertainty in attributes.

## Viewer And Jaeger Requirements

XTuner viewer and Jaeger must consume the same reconstructed spans. Do not implement one inference path in the viewer and another inference path for Jaeger.

Viewer expectations:

- Show one sample per agent run/session after live and completed records are merged.
- Prefer the trace containing `reconstructed_agent_run` as the primary sample trace.
- Keep raw `session_server.request` trace ids available for audit.
- Display the reconstructed path:

```text
reconstructed_agent_run -> reconstructed_llm_call.1 -> reconstructed_tool_call.1 -> ...
```

Jaeger expectations:

- The viewer's Jaeger link should open the trace containing `reconstructed_agent_run`.
- Opening the first raw `session_server.request` trace is a bug for reconstructed black-box samples.
- A healthy reconstructed trace contains `1 + llm_count + bounded_tool_count` reconstructed spans.
# XTuner Trace Patterns

These patterns describe the retained trace surface in this branch: runtime, basic API, viewer, and local setup tooling.

## Basic API

Use the public facade:

```python
from xtuner.v1.rl.trace import (
    inject_trace_context,
    set_trace_attributes,
    trace_event,
    trace_function,
    trace_span,
)
```

### Local Span

```python
with trace_span("phase.name", attributes={"xtuner.stage": "phase"}):
    ...
    set_trace_attributes({"phase.count": 3})
```

### Decorated Function

```python
@trace_function("phase.load")
def load_item(path: str) -> object:
    ...
```

### Parent Carrier

```python
carrier = inject_trace_context()

with trace_span("child.phase", parent_carrier=carrier):
    ...
```

`parent_carrier` is a basic W3C context carrier. It is not tied to `RolloutState`, Ray, aiohttp, or any XTuner business object.

## End-to-End Trace Wiring

Use this pattern when adding trace support to an XTuner training run.

### Launch Script

Enable trace bootstrap from the launcher, guarded by `XTUNER_TRACE_ENABLED`:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${XTUNER_TRACE_ENABLED:-0}" = "1" ]; then
  source "${SCRIPT_DIR}/setup_trace.sh"
fi
```

Keep training stdout/stderr in the main training log so runtime trace messages
are visible:

```bash
python xtuner/v1/train/cli/rl.py \
  --config "$CONFIG_PATH" \
  --num-workers "$XTUNER_RL_NUM_WORKERS" \
  2>&1 | tee -a "${WORK_DIR}/training_log_${current_time}.txt"
```

### Training Config

Add `TraceConfig` to the training config and pass it to the trainer config:

```python
import os
from pathlib import Path

from xtuner.v1.rl.trace import TraceConfig


trace_config = TraceConfig(
    enabled=os.environ.get("XTUNER_TRACE_ENABLED") == "1",
    output_dir=Path(work_dir) / "otel",
    service_name="xtuner-agent-rollout",
    viewer_enabled=True,
    viewer_host="0.0.0.0",
    viewer_port=18080,
    viewer_jaeger_query_url="http://127.0.0.1:16686",
)

trainer = RLColocateTrainerConfig(
    ...,
    trace_config=trace_config,
)
```

### Stage Spans

Before adding spans, ask the user which stages they want to observe and which
metrics each stage should expose.

Put fields known when entering the stage, especially fields used for grouping
or filtering, in `trace_span(..., attributes=...)`. Put results known only
after or during execution, such as status, duration, token counts, errors, and
reward values, in `set_trace_attributes(...)`. This lets the viewer reconstruct
the call chain and identify latency hotspots or abnormal stages.

Infer stage:

```python
import time


started_at = time.monotonic()
attributes = {
    "xtuner.stage": "infer",
    "xtuner.task_name": task_name,
    "xtuner.sample_id": sample_id,
    "rollout.backend": backend,
}

with trace_span("agent.infer", attributes=attributes):
    result = await run_infer(...)
    set_trace_attributes(
        {
            "xtuner.status": "ok" if result.ok else "error",
            "prompt.tokens": result.prompt_tokens,
            "completion.tokens": result.completion_tokens,
            "stage.duration_ms": int((time.monotonic() - started_at) * 1000),
        }
    )
```

Resource acquire stage:

```python
with trace_span(
    "sandbox.acquire",
    attributes={
        "xtuner.stage": "acquire",
        "xtuner.sample_id": sample_id,
        "sandbox.name": sandbox_name,
    },
):
    client = await pool.get(sandbox_name)
    set_trace_attributes(
        {
            "sandbox.env_id": env_id,
            "sandbox.image": image,
            "sandbox.reused": reused,
        }
    )
```

Validation or reward stage:

```python
with trace_span(
    "sample.validate",
    attributes={
        "xtuner.stage": "validate",
        "xtuner.sample_id": sample_id,
        "validator.name": validator_name,
    },
):
    record = await validate(...)
    set_trace_attributes(
        {
            "xtuner.status": "ok" if record.ok else "error",
            "reward.score": record.score,
            "reward.passed": record.passed,
        }
    )

    if record.error is not None:
        set_trace_attributes(
            {
                "error": True,
                "error.type": record.error.type,
                "error.message": record.error.message,
            }
        )
```

Do not record prompts, responses, full configs, secrets, raw headers, stack
traces, or large payloads.

### Cross-Boundary Propagation

When a traced operation crosses a process, actor, or HTTP boundary, keep the
propagation helper in the business or transport module that owns that boundary.
Do not put Ray, HTTP, rollout, agent, or judger helpers back into
`xtuner/v1/rl/trace`.

#### Ray Remote Calls

For Ray calls, inject the current context into a plain carrier and temporarily
attach that carrier to a serializable domain object or explicit call argument.

```python
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

from xtuner.v1.rl.trace import inject_trace_context, trace_span


_TRACE_CARRIER_FIELD = "_xtuner_trace_carrier"


@contextmanager
def attach_trace_carrier_temporarily(target: Any, carrier: Mapping[str, str]):
    if not carrier:
        yield
        return

    extra_fields = getattr(target, "extra_fields", None)
    if extra_fields is None:
        extra_fields = {}
        target.extra_fields = extra_fields
    had_previous = _TRACE_CARRIER_FIELD in extra_fields
    previous = extra_fields.get(_TRACE_CARRIER_FIELD)
    extra_fields[_TRACE_CARRIER_FIELD] = dict(carrier)
    try:
        yield
    finally:
        if had_previous:
            extra_fields[_TRACE_CARRIER_FIELD] = previous
        else:
            extra_fields.pop(_TRACE_CARRIER_FIELD, None)


def call_ray_remote_with_trace(remote_method, *args, trace_target: Any, **kwargs):
    carrier: dict[str, str] = {}
    inject_trace_context(carrier)
    with attach_trace_carrier_temporarily(trace_target, carrier):
        return remote_method.remote(*args, **kwargs)


def pop_trace_parent_carrier(target: Any) -> dict[str, str] | None:
    extra_fields = getattr(target, "extra_fields", None)
    if not isinstance(extra_fields, dict):
        return None
    carrier = extra_fields.pop(_TRACE_CARRIER_FIELD, None)
    if not isinstance(carrier, Mapping):
        return None
    return {str(key): str(value) for key, value in carrier.items()}
```

On the receiver side, attach the carrier to the new span:

```python
parent_carrier = pop_trace_parent_carrier(trace_target)

with trace_span("worker.stage", attributes=attributes, parent_carrier=parent_carrier):
    ...
```

Keep the helper domain-specific. Validate that exactly one trace target is used
when the transport needs one target object; reject collections if restoring the
carrier to the right child span would be ambiguous.

#### HTTP Calls

For HTTP calls, prefer W3C headers. If a third-party client or proxy may drop
custom headers, also put a copied carrier into the JSON body under an internal
field and remove it before forwarding the payload downstream.

```python
from collections.abc import Mapping
from typing import Any

from xtuner.v1.rl.trace import inject_trace_context, trace_span


_TRACE_CARRIER_FIELD = "_xtuner_trace_carrier"


def inject_trace_carrier_into_json_body(kwargs: dict[str, Any], carrier: Mapping[str, str]) -> None:
    if not carrier:
        return
    payload = kwargs.get("json")
    if not isinstance(payload, dict):
        return
    payload = dict(payload)
    payload.setdefault(_TRACE_CARRIER_FIELD, dict(carrier))
    kwargs["json"] = payload


def extract_trace_carrier_from_mapping(payload: Mapping[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(payload, Mapping):
        return None
    carrier = payload.get(_TRACE_CARRIER_FIELD)
    if not isinstance(carrier, Mapping):
        return None
    return {str(key): str(value) for key, value in carrier.items()}


def remove_trace_carrier_from_mapping(payload: dict[str, Any]) -> None:
    payload.pop(_TRACE_CARRIER_FIELD, None)


def extract_http_parent_carrier(headers: Mapping[str, Any], payload: Mapping[str, Any] | None) -> dict[str, str] | None:
    header_carrier = {
        str(key): str(value)
        for key, value in headers.items()
        if str(key).lower() in {"traceparent", "tracestate", "baggage"}
    }
    return header_carrier or extract_trace_carrier_from_mapping(payload)
```

Caller:

```python
headers = dict(headers or {})
request_kwargs = {"json": payload}
carrier: dict[str, str] = {}
inject_trace_context(carrier)
headers.update(carrier)
inject_trace_carrier_into_json_body(request_kwargs, carrier)

with trace_span("http.client.request", attributes={"http.method": "POST", "http.url": url}):
    response = await client.post(url, headers=headers, **request_kwargs)
```

Receiver:

```python
headers = dict(request.headers)
payload = await request.json()
parent_carrier = extract_http_parent_carrier(headers, payload)

with trace_span("http.server.request", attributes=attributes, parent_carrier=parent_carrier):
    remove_trace_carrier_from_mapping(payload)
    ...
```

Do not record raw headers or full payloads as span attributes. Only record
derived fields such as method, route, status code, stage, IDs, and timing.

### Viewer Output

When `viewer_enabled=True`, the trace runtime logs the trace JSONL path, viewer
URL, and restart command. Keep those runtime log lines visible in the main
training log and report them to the user after the run.

Useful reference points from the full trace implementation:

- `examples/v1/scripts/run_rl_run.sh`
- `examples/v1/scripts/setup_trace.sh`
- `examples/v1/config/agentic_rl_qwen3p5vl_mtp_ep_code.py`

## Runtime

Configure tracing explicitly:

```python
from xtuner.v1.rl.trace import TraceConfig, close_trace, configure_trace

runtime = configure_trace(
    TraceConfig(
        enabled=True,
        output_dir="work_dirs/example/otel",
        service_name="xtuner",
        viewer_enabled=True,
    )
)

try:
    ...
finally:
    close_trace()
```

The runtime owns OTel collector setup, trace JSONL output, live JSONL output, and optional viewer process startup.

## Viewer

The viewer reads Jaeger-style traces or OTel JSONL converted into Jaeger-style payloads. It should not depend on hard-coded rollout/agent span registries.

Stage display rules:

1. Use `xtuner.stage` when present.
2. Else use `stage` when present.
3. Else use `stage.name` when present.
4. Else fall back to the raw span name.

Useful stable attributes:

- IDs: `xtuner.rollout_id`, `xtuner.group_id`, `xtuner.session_id`, `xtuner.task_name`
- Status: `xtuner.status`
- Stage: `xtuner.stage`, `stage`, `stage.name`
- Counts/timing: `prompt.tokens`, `completion.tokens`, `http.status_code`
- Errors: `error`, `error.message`, `exception.type`

Avoid recording prompts, responses, full configs, secrets, raw headers, or large payloads.

## Local Setup

`examples/v1/scripts/setup_trace.sh` and `recipe/otle/` are local helper tooling for installing OTel collector binaries and starting local Jaeger/viewer dependencies. Keep these as setup assets; do not use them to add automatic trace behavior to training configs or launch scripts unless that integration is explicitly requested.

## Removed Semantics

Do not use or recreate these removed surfaces in this branch:

- `trace_remote`
- `traced_rollout_endpoint`
- `traced_agent_item_endpoint`
- `traced_judger_endpoint`
- `xtuner/v1/rl/trace/trace_utils.py`
- `xtuner/v1/rl/trace/context_propagation.py`
- fixed `TRACE_SPAN_*` registries
- rollout/agent/judger attribute builders

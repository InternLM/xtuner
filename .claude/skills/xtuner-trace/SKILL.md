---
name: xtuner-trace
description: Use when the user wants to instrument, inspect, or debug XTuner traces to reconstruct a sample/request execution path, follow cross-process call chains, or identify latency hotspots, bottleneck stages, and abnormal paths from span data.
---

# XTuner Trace

Use this skill for the current trace layer in this repository: OpenTelemetry runtime setup, the basic public API, local setup scripts, and the trace viewer.

## Current Boundary

Keep this package infrastructure-only:

- Runtime/configuration: `xtuner/v1/rl/trace/runtime.py`
- OTel SDK adapter: `xtuner/v1/rl/trace/otel_utils.py`
- Public facade: `xtuner/v1/rl/trace/__init__.py`
- Basic API: `xtuner/v1/rl/trace/api.py`
- Rollout starter preset: `xtuner/v1/rl/trace/rollout_api.py`
- Viewer: `xtuner/tools/trace_viewer/`
- Local tooling: `recipe/otle/` and `examples/v1/scripts/setup_trace.sh`

Do not add rollout, agent, judger, Ray remote, HTTP proxy, reward, status, or session-server business semantics to `api.py`, `runtime.py`, or `otel_utils.py`. Keep rollout-specific starter behavior inside `rollout_api.py` and gated by `TraceConfig.enable_rollout_trace`.

## Basic API

These are the interfaces defined in `xtuner/v1/rl/trace/api.py` and re-exported
from `xtuner.v1.rl.trace`:

- `trace_span(name, attributes=None, parent_carrier=None)`
- `trace_function(name=None, attributes=None)`
- `trace_event(name, attributes=None)`
- `set_trace_attributes(attributes)`
- `inject_trace_context(carrier=None)`

## Runtime API

Use these `xtuner/v1/rl/trace/runtime.py` interfaces for explicit trace
runtime setup:

- `TraceConfig`
- `configure_trace(...)`
- `close_trace()`

## Rollout Starter Preset

`TraceConfig(enable_rollout_trace=True)` enables the built-in rollout starter
trace. Its helpers live in `xtuner/v1/rl/trace/rollout_api.py`:

- `trace_rollout_endpoint(...)`
- `trace_rollout_remote(...)`

Keep this layer narrow: it may depend on `RolloutState` and Ray call boundaries,
but it must not leak those dependencies into the basic API.

## Add Trace Workflow

When adding trace instrumentation to an XTuner run:

1. Ask for or locate the launch script and training config before editing.
2. In the launch script, source `examples/v1/scripts/setup_trace.sh` when
   `XTUNER_TRACE_ENABLED=1`.
3. In the training config, add `TraceConfig` and set `enabled=True`; set
   `viewer_enabled=True` when the user needs interactive inspection. Set
   `enable_rollout_trace=True` only when the user wants the built-in rollout
   starter trace.
4. Before adding any `trace_span(...)` instrumentation, you mask ask the user which
   stages they want to observe and which metrics each stage should expose.
   Do not infer default stages unless the user explicitly asks you to choose.
5. Add `trace_span(...)` only around the user-confirmed observed stages. Put fields known at span
   start in initial attributes, and update runtime or final fields with
   `set_trace_attributes(...)`.
6. For cross-process or request boundaries, inject a carrier with
   `inject_trace_context(...)` and pass it to downstream
   `trace_span(..., parent_carrier=carrier)`.
7. Keep transport-specific propagation at the caller boundary; do not move
   rollout, agent, judger, Ray, or HTTP semantics into the basic trace package.
8. Ensure the main training log includes the trace output path, viewer URL, and
   a restart command for the viewer.

## Guardrails

- Do not reintroduce the old generic `trace_remote`, `traced_rollout_endpoint`,
  `traced_agent_item_endpoint`, or `traced_judger_endpoint` surfaces. The
  supported rollout starter helpers are `trace_rollout_endpoint` and
  `trace_rollout_remote` in `rollout_api.py`.
- Do not recreate `trace_utils.py`, `context_propagation.py`, span-name registries, or business attribute builders under `xtuner/v1/rl/trace`.
- Do not import `RolloutState`, agent item classes, judgers, rollout workers, Ray actors, aiohttp clients, or trainer configs from the basic trace package.
- Do not call OpenTelemetry SDK directly from business code; use the basic API only when trace instrumentation is explicitly requested.
- Do not record prompts, responses, full configs, secrets, raw headers, stack traces, or large payloads as attributes.
- Viewer stage grouping must come from span attributes such as `xtuner.stage`, `stage`, or `stage.name`; otherwise fall back to the raw span name.

For concrete patterns, read [references/trace-patterns.md](references/trace-patterns.md) before editing trace code.

## Verification

Use focused checks:

- `PYTHONPATH=. python -m compileall -q xtuner/v1/rl/trace xtuner/tools/trace_viewer`
- `PYTHONPATH=. python -m unittest discover -s tests/rl -p 'test_trace*.py' -v` when the trace tests exist in the worktree.
- `git diff --check`

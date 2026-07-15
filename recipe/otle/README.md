# XTuner OTel Trace

XTuner exports rollout traces through OpenTelemetry. For local inspection, start
Jaeger with `jaeger/jaeger-memory.yaml`, enable trace in the training config, and
open the XTuner rollout viewer.

The reference Jaeger config exposes:

- Jaeger UI and Query API: `http://127.0.0.1:16686`
- OTLP gRPC receiver: `http://127.0.0.1:14317`
- OTLP HTTP receiver: `http://127.0.0.1:14318/v1/traces`

Install local binaries:

```bash
bash recipe/otle/install_otel_tools.sh
export PATH=/tmp/xtuner_otel/bin:$PATH
```

Start Jaeger:

```bash
jaeger --config recipe/otle/jaeger/jaeger-memory.yaml
```

For local smoke tests, restart the in-memory Jaeger before each experiment so
old services, operations, and trace ids cannot be mixed with the new run:

```bash
bash recipe/otle/restart_jaeger_memory.sh
```

Run XTuner with trace enabled:

```bash
export XTUNER_TRACE_ENABLED=1
```

By default, XTuner starts a local collector that writes
`<trace_run_dir>/traces/traces.jsonl` and forwards spans to the reference Jaeger
OTLP gRPC endpoint `http://127.0.0.1:14317`.

Set `TraceConfig(xtuner_viewer_enabled=True, ...)` to start the XTuner rollout
viewer with the trace runtime. The viewer output goes to the same terminal or
training log as the training process. The default viewer port is `18080`.
`examples/v1/scripts/setup_trace.sh` clears stale `recipe.trace_viewer.server`
processes on `XTUNER_TRACE_VIEWER_PORT` or `18080` before preparing the local
trace dependencies.

Open the rollout viewer:

```bash
python -m recipe.trace_viewer.server \
  --trace-jsonl <trace_run_dir>/traces/traces.jsonl \
  --jaeger-query-url http://127.0.0.1:16686 \
  --service xtuner-rollout
```

The viewer reads `traces.jsonl` directly and uses Jaeger only for trace links
and the optional same-origin proxy. XTuner groups spans by rollout metadata such
as `xtuner.rollout_id`, `xtuner.group_id`, `xtuner.task_name`, and
`xtuner.status`.

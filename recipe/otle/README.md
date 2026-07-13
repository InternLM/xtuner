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

Open the rollout viewer:

```bash
python -m xtuner.tools.trace_viewer.server \
  --jaeger-query-url http://127.0.0.1:16686 \
  --service xtuner-rollout
```

The viewer is a thin Jaeger Query API adapter. Jaeger remains the trace backend;
XTuner only groups spans by rollout metadata such as `xtuner.rollout_id`,
`xtuner.group_id`, `xtuner.task_name`, and `xtuner.status`.

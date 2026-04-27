#!/usr/bin/env python
"""Start a RolloutController-backed Gateway for manual protocol debugging.

This script is intended for end-to-end debugging with real clients such as
Claude Code, Codex, curl, or the OpenAI SDK. It starts the RolloutController,
waits for rollout workers to become ready, then serves the Gateway in the
current process.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_WORK_DIR = Path("/tmp/xtuner_debug_gateway")
DEFAULT_MODEL_PATH = os.environ.get("ROLLOUT_MODEL_PATH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start a local XTuner Gateway backed by a RolloutController for manual protocol debugging.\n\n"
            "Example:\n"
            "  python .dev_scripts/debug_gateway.py --model-path /path/to/model --model-name local-test"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        required=DEFAULT_MODEL_PATH is None,
        help="Model path for rollout workers. Defaults to the ROLLOUT_MODEL_PATH environment variable.",
    )
    parser.add_argument("--model-name", default=None, help="Model name exposed by the Gateway.")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path. Defaults to --model-path.")
    parser.add_argument("--rollout-env", default="debug_gateway", help="Rollout environment name.")
    parser.add_argument("--ray-address", default="local", help="Ray cluster address. Use 'local' to start one.")
    parser.add_argument("--ray-namespace", default="xtuner-debug-gateway", help="Ray namespace for this debug run.")
    parser.add_argument("--controller-name", default=None, help="Optional Ray actor name for the RolloutController.")
    parser.add_argument(
        "--ray-max-concurrency",
        type=int,
        default=int(os.environ.get("RAY_MAX_CONCURRENCY", 1000)),
        help="max_concurrency for the RolloutController actor.",
    )

    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-cpus-per-worker", type=int, default=16)
    parser.add_argument("--cpu-memory-per-worker-gb", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--dist-port-base", type=int, default=42000)
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=30080)
    parser.add_argument("--worker-log-dir", default=str(DEFAULT_WORK_DIR / "worker_logs"))
    parser.add_argument("--placement-group-name", default="xtuner_debug_gateway_pg")
    parser.add_argument(
        "--ready-poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval while waiting for rollout workers to become ready.",
    )
    parser.add_argument("--tool-call-parser", default="qwen3", help="Tool call parser used by the rollout backend.")
    parser.add_argument("--reasoning-parser", default="qwen3", help="Reasoning parser used by the rollout backend.")

    parser.add_argument("--host", default="127.0.0.1", help="Gateway bind host.")
    parser.add_argument("--port", type=int, default=8091, help="Gateway bind port.")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level.")
    parser.add_argument(
        "--capture-folder",
        default=None,
        help="Optional request capture folder. If omitted, defaults to <worker-log-dir>/gateway_captures.",
    )

    return parser.parse_args()


def resolve_capture_output_file(capture_folder: str | Path | None) -> Path | None:
    if capture_folder is None:
        return None
    from xtuner.v1.rl.gateway.adapters.capture import resolve_capture_output_path

    return resolve_capture_output_path(capture_folder)


def describe_capture_output(capture_folder: str | Path | None) -> str:
    capture_output_file = resolve_capture_output_file(capture_folder)
    if capture_output_file is None:
        return "disabled"
    return f"{capture_output_file} (requests with API keys are split into api_key_<hash>.jsonl)"


def init_ray(address: str, namespace: str) -> dict[str, Any]:
    import ray

    ctx = ray.init(address=address, namespace=namespace, ignore_reinit_error=True)
    address_info = getattr(ctx, "address_info", {}) or {}
    return {
        "requested_ray_address": address,
        "ray_address": address_info.get("address") or address_info.get("gcs_address") or address,
        "namespace": namespace,
        "ray_context": address_info,
    }


def build_rollout_config(args: argparse.Namespace):
    from xtuner.v1.rl.rollout.worker import RolloutConfig

    model_path = str(args.model_path)
    tokenizer_path = str(args.tokenizer_path or args.model_path)
    model_name = args.model_name or Path(model_path).name.lower()
    return RolloutConfig(
        env=args.rollout_env,
        device="GPU",
        model_path=model_path,
        model_name=model_name,
        tokenizer_path=tokenizer_path,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        context_length=args.context_length,
        worker_log_dir=args.worker_log_dir,
        dist_port_base=args.dist_port_base,
        api_host=args.api_host,
        api_port=args.api_port,
        tool_call_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
    )


def build_controller(args: argparse.Namespace):
    import ray

    from xtuner.v1.rl.rollout.controller import RolloutController
    from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers

    resource_config = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_workers=args.num_workers,
        num_cpus_per_worker=args.num_cpus_per_worker,
        cpu_memory_per_worker=args.cpu_memory_per_worker_gb * 1024**3,
    )
    placement_group = AutoAcceleratorWorkers.build_placement_group(
        resource_config,
        name=args.placement_group_name,
    )
    rollout_config = build_rollout_config(args)
    actor_options: dict[str, Any] = {
        "max_concurrency": args.ray_max_concurrency,
    }
    if args.controller_name:
        actor_options["name"] = args.controller_name
    controller = ray.remote(RolloutController).options(**actor_options).remote(rollout_config, placement_group)
    print("Created rollout controller.")
    return controller, placement_group


def wait_for_controller_ready(controller, poll_seconds: float) -> dict[str, Any]:
    import ray

    while True:
        ready, status = ray.get(controller.get_ready_status.remote())
        if ready:
            print(f"Rollout controller ready: {status}")
            return status
        print(f"Waiting for rollout workers... {status}")
        time.sleep(poll_seconds)


def start_gateway(args: argparse.Namespace, controller) -> None:
    from xtuner.v1.rl.gateway.config import GatewayConfig
    from xtuner.v1.rl.gateway.server import build_local_gateway_app, serve_gateway

    capture_folder = args.capture_folder
    if capture_folder is None:
        capture_folder = str(Path(args.worker_log_dir) / GatewayConfig._CAPTURE_PATH_FOLDER)

    cfg = GatewayConfig(
        host=args.host,
        port=args.port,
        auto_start=False,
        capture_folder=capture_folder,
        log_level=args.log_level,
    )

    app = build_local_gateway_app(controller, config=cfg)
    print(f"Starting gateway at http://{cfg.host}:{cfg.port}")
    print(f"Gateway capture output: {describe_capture_output(cfg.capture_folder)}")
    serve_gateway(app, cfg)


def cleanup_controller(controller, placement_group) -> None:
    import ray

    try:
        ray.get(controller.shutdown.remote(), timeout=300)
    except Exception as exc:
        print(f"Failed to shutdown rollout controller cleanly: {exc}", file=sys.stderr)
    try:
        ray.kill(controller, no_restart=True)
    except Exception as exc:
        print(f"Failed to kill rollout controller: {exc}", file=sys.stderr)
    if placement_group is not None:
        try:
            ray.util.remove_placement_group(placement_group)
        except Exception as exc:
            print(f"Failed to remove placement group: {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    controller = None
    placement_group = None
    try:
        init_info = init_ray(args.ray_address, args.ray_namespace)
        print(
            "Initialized Ray: "
            f"requested_address={init_info['requested_ray_address']}, "
            f"address={init_info['ray_address']}, namespace={init_info['namespace']}"
        )
        controller, placement_group = build_controller(args)
        wait_for_controller_ready(controller, args.ready_poll_seconds)
        start_gateway(args, controller)
    finally:
        ray_module = sys.modules.get("ray")
        if ray_module is not None and ray_module.is_initialized():
            if controller is not None:
                cleanup_controller(controller, placement_group)
            ray_module.shutdown()


if __name__ == "__main__":
    main()

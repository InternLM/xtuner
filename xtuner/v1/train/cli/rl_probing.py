"""RL training entry point with Probing telemetry enabled.

This is an opt-in variant of ``xtuner/v1/train/cli/rl.py``. It installs the
Probing XTuner adapter in the driver process before the RL CLI starts, asks
Probing to propagate its environment to Ray actors, and optionally registers
LMDeploy rollout engines for the Probing Inference page.

Usage::

    PROBING=1 PROBING_PORT=8080 \
        python xtuner/v1/train/cli/rl_probing.py \
        --config <config.py> --num-workers 8

A Ray cluster must already be running; ``examples/v1/scripts/run_rl.sh``
starts one. ``PROBING_PORT`` enables the driver's Web UI, while Ray workers
use per-process Unix sockets. Set ``PROBING_LMDEPLOY_LOG`` to the launcher
log to register LMDeploy engines; ``examples/v1/scripts/run_rl_probing.sh``
sets it automatically.
"""

import os
import sys


def _install_probing() -> None:
    probing_flag = (os.environ.get("PROBING_ORIGINAL") or os.environ.get("PROBING") or "0").lower()
    if probing_flag not in {"1", "true", "yes", "on"}:
        print(
            "[probing] PROBING is not enabled; set PROBING=1 before starting Python to collect telemetry.",
            file=sys.stderr,
            flush=True,
        )

    try:
        from probing.ext.xtuner import install
    except ImportError as exc:
        raise RuntimeError(
            "rl_probing.py requires a probing build with the XTuner adapter "
            "(probing.ext.xtuner, probing >= 0.2.5). Install it with "
            "`pip install -U probing` or use xtuner/v1/train/cli/rl.py without telemetry."
        ) from exc

    installed = install()
    print(f"[probing] XTuner adapter installed: {installed}", file=sys.stderr, flush=True)
    if not installed:
        print(
            "[probing] adapter installation failed; RL telemetry will be unavailable.",
            file=sys.stderr,
            flush=True,
        )


def _init_ray_with_probing() -> None:
    try:
        import ray

        if ray.is_initialized():
            return
        from probing.ext.ray import init_job

        init_job(address="auto")
        print("[probing] Ray initialized with probing env propagation", file=sys.stderr, flush=True)
    except Exception as exc:  # telemetry must not prevent training from starting
        print(f"[probing] Ray probing setup skipped: {exc}", file=sys.stderr, flush=True)


def _register_lmdeploy_engines() -> None:
    log_path = os.environ.get("PROBING_LMDEPLOY_LOG", "").strip()
    if not log_path:
        return
    try:
        from probing.ext.engines.lmdeploy import register_lmdeploy_engines

        engines = register_lmdeploy_engines(log_path)
        print(
            f"[probing] LMDeploy engines registered: {[engine.engine_id for engine in engines]}",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:  # inference metrics are optional
        print(f"[probing] LMDeploy engine registration skipped: {exc}", file=sys.stderr, flush=True)


def main() -> None:
    _install_probing()
    _init_ray_with_probing()
    _register_lmdeploy_engines()

    from xtuner.v1.rl.utils import register_cleanup
    from xtuner.v1.train.cli.rl import app

    register_cleanup()
    app(exit_on_error=False)


if __name__ == "__main__":
    main()

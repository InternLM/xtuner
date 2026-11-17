"""Fresh-process fixtures using the actual Runner/Pool and loopback management."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path

import sandbox_cleanup_cpu as runtime


class Infer:
    sandbox = "main"

    async def run(self, client, item, record):
        item.artifacts["infer_pid"] = os.getpid()
        item.artifacts["bytes"] = b"unchanged\x00artifact"
        marker = Path(item.metadata["marker"])
        marker.with_suffix(".staging").write_text(str(Path(os.environ["XTUNER_SANDBOX_TRIAL_MANIFEST"]).parent))
        marker.write_text(str(os.getpid()))
        if item.metadata.get("crash"):
            os._exit(71)
        if item.metadata.get("business_error"):
            record.error = runtime.schemas.RolloutError(
                stage="infer", category="business_error", message="original business failure"
            )
            return runtime.schemas.StageResult(error="business_error")
        if item.metadata.get("block"):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            while True:
                time.sleep(10)
        if item.metadata.get("slow"):
            await asyncio.sleep(0.7)
        if item.metadata.get("cancel_resistant"):
            while True:
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    continue
        return runtime.schemas.StageResult()


class Verify:
    name = "verify"

    async def run(self, item, pool, record):
        await pool.get("verifier", record=record)
        item.artifacts["verify_pid"] = os.getpid()
        return 1.0


class Pipeline:
    async def run(self, item):
        pool = runtime.sandbox.SandboxPool(
            provider=runtime.Provider(item.metadata["port"]),
            specs={
                "main": runtime.schemas.SandboxSpec(image="main"),
                "verifier": runtime.schemas.SandboxSpec(image="verifier"),
            },
            close_provider=True,
            max_attempts=1,
            creates_per_sec=None,
            cleanup_timeout_sec=0.5,
            cleanup_retry_timeout_sec=0.5,
        )
        return await runtime.Runner(pool=pool, infer=Infer(), validate=Verify()).run(item)


def provider(key):
    if key != "trial-test":
        raise ValueError("unexpected provider key")
    return runtime.Provider(int(os.environ["TRIAL_TEST_PORT"]))


def hanging_provider(key):
    Path(os.environ["TRIAL_TEST_RECOVERY_STARTED"]).write_text("started")
    time.sleep(60)
    return provider(key)

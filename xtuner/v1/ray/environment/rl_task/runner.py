"""Runner — top-level orchestrator for one task rollout.

Owns the infer sandbox lifecycle, threads a context dict through the two
stages (infer → validate), assembles the result envelope.  All real work
is hook-driven (:class:`sandbox.SandboxStage` pre/entry/post) +
:class:`validator.JudgerValidator`.

Invocation:

    # config.py (user-written)
    from claw_bench import ClawBench           # bench project on PYTHONPATH
    dataset = ClawBench(tasks_root="/data/bench/claw-bench/tasks")

    # Run:
    python runner.py --config config.py TASK_DIR [TASK_DIR ...]
    python runner.py --config config.py --limit 10     # iterate whole dataset

``dataset`` in the config provides ``pipeline`` (a :class:`Runner` shared
by all its tasks) and ``load_task(task_dir) → TaskData``.  No
type-dispatched config — the pipeline is composed in pure Python by the
project's factory.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lagent.serving.sandbox.providers.gateway import GatewayProvider  # noqa: E402

from xtuner.v1.ray.environment.rl_task.sandbox import SandboxStage  # noqa: E402
from xtuner.v1.ray.environment.rl_task.schemas import TaskData  # noqa: E402
from xtuner.v1.ray.environment.rl_task.validator import JudgerValidator  # noqa: E402
from xtuner.v1.utils import get_logger  # noqa: E402


# ─────────────────────────────────────────────────────────────────
# Daemon-silence watchdog
# ─────────────────────────────────────────────────────────────────
#
# The sandbox /exec call has a single host-side timeout of
# ``timeout_sec + 10 = 10810s`` (3h). If the sandbox daemon silently
# hangs part-way through the agent loop (no LLM traffic, no tool activity,
# no log output), the whole batch is held up for nearly 3h before the
# host httpx read-timeout fires. The agent config's LLM timeout (1800s)
# and tool-level timeouts bound individual operations but don't cover
# "daemon stuck between operations" — that's what this watchdog is for.
#
# Strategy: poll ``/tmp/agent_daemon.log`` size concurrently with the
# in-flight /exec. If the log hasn't grown for longer than
# ``XTUNER_SANDBOX_WATCHDOG_STALE_SEC`` we call the daemon dead, cancel
# /exec, and fail fast with the last known tail.
#
# The stale threshold must exceed the LLM timeout, because a single
# legit slow LLM call can produce no log output for its full duration.
# Default 2400s = 1800s (AsyncAPIClient.timeout default for interndp
# config) + 600s margin.

_WATCHDOG_STALE_SEC = int(os.environ.get("XTUNER_SANDBOX_WATCHDOG_STALE_SEC", "2400"))
_WATCHDOG_POLL_SEC = float(os.environ.get("XTUNER_SANDBOX_WATCHDOG_POLL_SEC", "60"))
_WATCHDOG_DOWNLOAD_TIMEOUT_SEC = 10.0
_WATCHDOG_MAX_CONSECUTIVE_POLL_FAILURES = 5
_WATCHDOG_DAEMON_LOG_PATH = "/tmp/agent_daemon.log"
_WATCHDOG_DAEMON_SOCK_PATH = "/tmp/lagent_agent.sock"
_WATCHDOG_PING_TIMEOUT_SEC = 5
_WATCHDOG_MAX_CONSECUTIVE_PING_FAILURES = 3


class DaemonStuckError(RuntimeError):
    """Raised by the watchdog when the daemon log stops growing."""


async def _ping_daemon(
    client: Any,
    sock_path: str = _WATCHDOG_DAEMON_SOCK_PATH,
    timeout: int = _WATCHDOG_PING_TIMEOUT_SEC,
) -> tuple[bool, str]:
    """Send ``{"cmd": "ping"}`` to the in-sandbox daemon socket.

    The daemon's asyncio accept loop handles each client connection in a
    new ``_handle_client`` task, so a ping can get a fresh
    ``_dispatch("ping")`` response even while another task is stuck in
    ``await self.agent(*messages)``.  That makes this a precise
    liveness signal: ping-alive + log-unchanged = agent deadlocked;
    ping-dead = daemon process / network itself is gone.

    We implement the wire protocol inline with the sandbox's stdlib
    ``python3``: a prior version tried ``python3 -m lagent.serving.sandbox.daemon
    call`` but the sandbox's ``/usr/bin/python3`` has no ``lagent`` on
    its ``sys.path`` (lagent lives in the host-side shared NFS that the
    sandbox container doesn't mount), so the subprocess failed with
    ``ModuleNotFoundError`` and every ping produced false "daemon dead"
    verdicts.  Speaking the socket protocol directly (4-byte big-endian
    length header, then JSON body — matches ``daemon._send_msg`` /
    ``_recv_msg``) is robust to whatever python environment the sandbox
    has.

    Args:
        client (Any): :class:`SandboxClient` that can run ``/exec``.
        sock_path (str): Unix socket path inside the sandbox.
        timeout (int): Max seconds to wait for ping response.  Used as
            both the in-sandbox socket timeout and the /exec timeout
            budget.

    Returns:
        tuple[bool, str]: ``(alive, detail)``.  ``detail`` carries the
            daemon's JSON response on success or the failure reason.
    """
    # Inline python one-liner: open the unix socket, send the length-
    # prefixed JSON, read the response, print to stdout.  Any exception
    # → write a PING_FAIL line to stderr and exit 1.  No imports
    # beyond stdlib — robust to broken PYTHONPATHs / missing packages.
    inline = (
        "import socket,struct,json,sys\n"
        "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)\n"
        f"s.settimeout({timeout})\n"
        "try:\n"
        f"  s.connect({sock_path!r})\n"
        '  m=json.dumps({"cmd":"ping"}).encode()\n'
        '  s.sendall(struct.pack("!I",len(m))+m)\n'
        "  h=s.recv(4)\n"
        '  assert len(h)==4,"short header"\n'
        '  (n,)=struct.unpack("!I",h)\n'
        '  buf=b""\n'
        "  while len(buf)<n:\n"
        "    chunk=s.recv(n-len(buf))\n"
        '    assert chunk,"short body"\n'
        "    buf+=chunk\n"
        "  sys.stdout.write(buf.decode())\n"
        "except Exception as e:\n"
        '  sys.stderr.write(f"PING_FAIL: {type(e).__name__}: {e}")\n'
        "  sys.exit(1)\n"
    )
    # Use bash -c with stdin-feed so the inline script survives shell
    # quoting without needing carefully-escaped single-quotes in the
    # outer command.
    cmd = f"python3 - <<'PING_EOF'\n{inline}\nPING_EOF"

    try:
        res = await asyncio.wait_for(
            client.execute(cmd, timeout_sec=timeout),
            timeout=timeout + 5,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return False, f"/exec failed: {type(exc).__name__}: {exc}"

    rc = res.get("return_code")
    stdout = (res.get("stdout") or "").strip()
    stderr = (res.get("stderr") or "").strip()
    if rc != 0:
        return False, f"daemon ping rc={rc} stderr={stderr!r}"
    try:
        parsed = json.loads(stdout)
    except Exception:
        return False, f"non-JSON ping response: {stdout!r}"
    if parsed.get("status") == "ok":
        return True, stdout
    # Response parsed but status != ok — daemon is technically alive but
    # in a weird state (shouldn't happen for ping, but be explicit).
    return False, f"ping response not ok: {stdout!r}"


async def _daemon_silence_watchdog(
    client: Any,
    tid: str,
    *,
    stale_threshold_sec: int | None = None,
    poll_interval_sec: float | None = None,
) -> None:
    """Watch the sandbox daemon via two liveness signals.

    Runs alongside the in-flight ``/exec`` call via
    :func:`asyncio.wait(..., FIRST_COMPLETED)`.  A clean exit is
    impossible — this coroutine only returns by raising, or by being
    cancelled when the caller observes that the infer stage finished
    first.

    Two signals, in order of precision:

    * **Daemon ping** (primary).  Cheap unix-socket round-trip via the
      sandbox ``/exec`` endpoint.  A dead daemon fails ping immediately;
      an agent stuck inside ``self.agent(*messages)`` will still answer
      ping because the daemon accept loop is independent.  Consecutive
      ping failures (``_WATCHDOG_MAX_CONSECUTIVE_PING_FAILURES``) fire
      the watchdog with reason ``"daemon dead/unreachable"``.

    * **Daemon-log size** (secondary).  Detects the
      ping-alive-but-agent-deadlocked case — daemon process is fine,
      accept loop answers, but the agent coroutine made no progress
      for longer than ``stale_threshold_sec``.  Requires threshold
      ``> LLM per-call timeout`` so a single legit slow LLM doesn't
      false-positive.

    Defaults for the two tunables come from module-level constants
    (read at call time, so tests / env-var tuning take effect without
    process restart).

    Args:
        client (Any): SandboxClient with ``download_file`` and
            ``execute`` APIs.
        tid (str): Task id used in log tags.
        stale_threshold_sec (int): How long the log can go without
            growing before we declare the agent deadlocked.  Defaults
            to ``_WATCHDOG_STALE_SEC`` when None.
        poll_interval_sec (float): Seconds between polls.  Defaults to
            ``_WATCHDOG_POLL_SEC`` when None.

    Raises:
        DaemonStuckError: Either ping failed ``N`` consecutive times
            (daemon dead) or log unchanged past threshold (agent
            deadlocked).  Message distinguishes the two cases.
    """
    if stale_threshold_sec is None:
        stale_threshold_sec = _WATCHDOG_STALE_SEC
    if poll_interval_sec is None:
        poll_interval_sec = _WATCHDOG_POLL_SEC
    last_size = -1
    last_change_monotonic = time.monotonic()
    last_blob = b""
    consecutive_download_failures = 0
    consecutive_ping_failures = 0
    last_ping_detail = "(none yet)"

    while True:
        await asyncio.sleep(poll_interval_sec)

        # --- Signal 1: daemon ping. Fast, precise. ---
        try:
            alive, detail = await _ping_daemon(client)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            alive, detail = False, f"ping raised: {type(exc).__name__}: {exc}"
        if alive:
            consecutive_ping_failures = 0
            last_ping_detail = detail
        else:
            consecutive_ping_failures += 1
            last_ping_detail = detail
            get_logger().warning(
                f"[{tid}] watchdog ping {consecutive_ping_failures}/"
                f"{_WATCHDOG_MAX_CONSECUTIVE_PING_FAILURES} failed: {detail}"
            )
            if consecutive_ping_failures >= _WATCHDOG_MAX_CONSECUTIVE_PING_FAILURES:
                raise DaemonStuckError(
                    f"[{tid}] daemon dead/unreachable: ping failed "
                    f"{consecutive_ping_failures} consecutive times; "
                    f"last detail: {detail}"
                )

        # --- Signal 2: daemon-log size. Catches ping-alive-deadlock. ---
        try:
            blob = await asyncio.wait_for(
                client.download_file(_WATCHDOG_DAEMON_LOG_PATH),
                timeout=_WATCHDOG_DOWNLOAD_TIMEOUT_SEC,
            )
            consecutive_download_failures = 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            consecutive_download_failures += 1
            get_logger().warning(
                f"[{tid}] watchdog: daemon log download failed "
                f"({consecutive_download_failures}/{_WATCHDOG_MAX_CONSECUTIVE_POLL_FAILURES}): "
                f"{type(exc).__name__}: {exc}"
            )
            if consecutive_download_failures >= _WATCHDOG_MAX_CONSECUTIVE_POLL_FAILURES:
                # Can't download log but ping was working — something is
                # wrong with the FS or /download endpoint specifically.
                # Still actionable, but keep the message specific.
                raise DaemonStuckError(
                    f"[{tid}] daemon log unavailable: "
                    f"{consecutive_download_failures} consecutive download failures; "
                    f"last error {type(exc).__name__}: {exc}. "
                    f"Last ping detail: {last_ping_detail}"
                ) from exc
            continue

        cur_size = len(blob)
        now = time.monotonic()
        if cur_size != last_size:
            last_size = cur_size
            last_blob = blob
            last_change_monotonic = now
            continue

        stale_for = now - last_change_monotonic
        if stale_for >= stale_threshold_sec:
            tail_lines = last_blob.decode(errors="replace").splitlines()[-200:]
            raise DaemonStuckError(
                f"[{tid}] agent deadlocked: ping OK but daemon log unchanged for "
                f"{stale_for:.0f}s (size={cur_size}, threshold={stale_threshold_sec}s); "
                f"last ping detail: {last_ping_detail}\ntail:\n" + "\n".join(tail_lines)
            )


async def _run_infer_with_watchdog(
    infer: SandboxStage,
    client: Any,
    ctx: dict[str, Any],
    tid: str,
) -> Any:
    """Race ``infer.run(client, ctx)`` against the daemon-silence watchdog.

    If the infer stage finishes first, cancel the watchdog and return
    its result.  If the watchdog fires first, cancel the in-flight
    ``/exec`` (which will propagate ``CancelledError`` through httpx)
    and raise the :class:`DaemonStuckError` so the caller's exception
    handler can surface the stuck-tail and mark the sample failed.
    """
    infer_task = asyncio.create_task(infer.run(client, ctx), name=f"infer-{tid}")
    watchdog_task = asyncio.create_task(_daemon_silence_watchdog(client, tid), name=f"watchdog-{tid}")

    try:
        done, _ = await asyncio.wait(
            {infer_task, watchdog_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        for t in (infer_task, watchdog_task):
            if not t.done():
                t.cancel()
        raise

    if infer_task in done:
        # Normal path: /exec returned first.
        watchdog_task.cancel()
        try:
            await watchdog_task
        except (asyncio.CancelledError, Exception):
            pass
        return await infer_task

    # Watchdog fired — kill the in-flight /exec and re-raise its error.
    infer_task.cancel()
    try:
        await asyncio.wait_for(infer_task, timeout=30)
    except (asyncio.CancelledError, asyncio.TimeoutError, Exception) as exc:
        get_logger().debug(f"[{tid}] infer cancel after watchdog: {type(exc).__name__}: {exc}")
    # Re-raise the watchdog error (it's in the done-set as an exception).
    return await watchdog_task  # raises DaemonStuckError


# ─────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────


class Runner:
    """Pairs one infer stage with one validator."""

    def __init__(self, infer: SandboxStage, validate: JudgerValidator):
        self.infer = infer
        self.validate = validate

    async def run_single(
        self,
        task_root: Path,
        data: TaskData,
        uid: dict[str, int],
        *,
        provider: Any,
        lagent_src_dir: str | Path | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
    ) -> dict[str, Any]:
        """Run one rollout end-to-end.

        Returns an RLDataFlowItem-shaped dict.
        """
        ctx: dict[str, Any] = {
            "task_root": task_root,
            "data": data,
            "uid": uid,
            "runtime": {
                "lagent_src_dir": lagent_src_dir,
                "llm_model": llm_model,
                "llm_base_url": llm_base_url,
                "llm_api_key": llm_api_key,
            },
            "workspace": self.infer.sandbox.workspace_path,
            "sandbox_image": self.infer.sandbox.image,
        }

        client = None
        env_id: str | None = None
        tid = data.id
        try:
            get_logger().info(f"[{tid}] acquiring sandbox (image={self.infer.sandbox.image})")
            t0 = time.monotonic()
            client, env_id = await _acquire_ready_sandbox(
                provider,
                self.infer.sandbox,
            )
            get_logger().info(f"[{tid}] sandbox ready env_id={env_id} ({time.monotonic() - t0:.1f}s)")

            get_logger().info(f"[{tid}] infer: start ({len(self.infer.pre)} pre-hooks)")
            t1 = time.monotonic()
            infer_result = await _run_infer_with_watchdog(self.infer, client, ctx, tid)
            get_logger().info(f"[{tid}] infer: done rc={infer_result.return_code} ({time.monotonic() - t1:.1f}s)")
            if not infer_result.ok:
                return _mark_failed(
                    data,
                    uid,
                    f"infer failed: {infer_result.error}",
                    metadata=_infer_metadata(ctx),
                )

            get_logger().info(f"[{tid}] validate: start ({len(self.validate.judgers)} judgers)")
            t2 = time.monotonic()
            aggregated = await self.validate.run(
                client,
                ctx,
                provider,
                self.infer.sandbox.workspace_path,
            )
            get_logger().info(f"[{tid}] validate: done total={aggregated.total:.4f} ({time.monotonic() - t2:.1f}s)")

            return _mark_completed(data, uid, metadata=_infer_metadata(ctx), judge=aggregated, infer=infer_result)
        except Exception as exc:
            get_logger().error(f"[{tid}] runner failed: {exc}\n{traceback.format_exc()}")
            return _mark_failed(
                data,
                uid,
                f"{type(exc).__name__}: {exc}",
                metadata=_infer_metadata(ctx),
            )
        finally:
            if env_id is not None:
                try:
                    await provider.delete(env_id)
                except Exception as exc:
                    get_logger().warning(f"[{tid}] gateway delete failed: {exc}")
            if client is not None:
                try:
                    await client.aclose()
                except Exception as exc:
                    get_logger().warning(f"[{tid}] client aclose failed: {exc}")


# ─────────────────────────────────────────────────────────────────
# Result envelope
# ─────────────────────────────────────────────────────────────────


def _infer_metadata(ctx: dict[str, Any]) -> dict[str, Any]:
    md: dict[str, Any] = {}
    chosen = ctx.get("chosen_agent")
    if chosen is not None:
        md["agent_name"] = chosen.name
    return md


_ACQUIRE_MAX_ATTEMPTS = 3
# Sandbox cold-start can take 30-60s under load; pathological boots run
# longer.  With async waits this budget is cheap (no thread tied up), so
# stay generous to avoid spurious "unhealthy" → delete → recreate churn.
_HEALTH_MAX_WAIT_SEC = 600
_HEALTH_POLL_INTERVAL_SEC = 2


async def _acquire_ready_sandbox(provider: Any, spec: Any) -> tuple[Any, str]:
    """Create a sandbox + wait for /health to respond before returning.

    Retries on create-fail or /health-never-becomes-ok (up to
    :data:`_ACQUIRE_MAX_ATTEMPTS` times).  Burst protection is enforced
    inside ``provider.create`` via a token-bucket rate limiter.

    Args:
        provider (Any): Async sandbox provider exposing ``create`` /
            ``delete``.
        spec (Any): Sandbox spec with ``image`` and ``ttl_seconds``.

    Returns:
        tuple[Any, str]: Client and env_id of a healthy sandbox.
    """

    last_err: Exception | None = None
    for attempt in range(1, _ACQUIRE_MAX_ATTEMPTS + 1):
        try:
            client, env_id = await provider.create(
                image_tag=spec.image,
                ttl_seconds=spec.ttl_seconds,
            )
        except Exception as exc:
            last_err = exc
            get_logger().warning(f"provider.create attempt {attempt} failed: {exc}")
            await asyncio.sleep(min(2**attempt, 8))
            continue

        if await _wait_healthy(client):
            return client, env_id

        get_logger().warning(f"sandbox {env_id} never became healthy; deleting and retrying")
        try:
            await provider.delete(env_id)
        except Exception as exc:
            get_logger().warning(f"delete of unhealthy {env_id} failed: {exc}")
        try:
            await client.aclose()
        except Exception as exc:
            get_logger().warning(f"aclose of unhealthy {env_id} failed: {exc}")
        last_err = RuntimeError(f"sandbox {env_id} unhealthy")

    raise RuntimeError(f"could not acquire a healthy sandbox after {_ACQUIRE_MAX_ATTEMPTS} attempts: {last_err}")


async def _wait_healthy(client: Any) -> bool:
    """Poll /health until it returns ``ok: True`` or the budget is
    exhausted."""
    deadline = time.monotonic() + _HEALTH_MAX_WAIT_SEC
    while time.monotonic() < deadline:
        try:
            h = await client.health_check()
            if h.get("ok"):
                return True
        except Exception as exc:
            get_logger().debug(f"health poll error: {exc}")
        await asyncio.sleep(_HEALTH_POLL_INTERVAL_SEC)
    return False


def _mark_completed(
    data: TaskData,
    uid: dict[str, int],
    *,
    metadata: dict[str, Any],
    judge,
    infer,
) -> dict[str, Any]:
    return {
        "uid": uid,
        "data": {
            "extra_info": {
                "task_id": data.id,
                "data_source": data.data_source,
                "ability": data.ability,
                "tags": data.tags,
            },
        },
        "env": {
            "rollout": {
                "state": "completed",
                "finish_reason": "stop",
                "extra_info": {**metadata},
            },
            "judger": {
                "total": judge.total,
                "per_judger": [r.model_dump() for r in judge.per_judger],
                "step_rewards": [sr.model_dump() for sr in judge.step_rewards],
                "failed": judge.failed,
            },
            "agent": {
                "message_dict": json.loads(infer.pulled.get("message", "{}")),
                "daemon_log": infer.pulled.get("daemon_log", ""),
            },
        },
    }


def _mark_failed(
    data: TaskData,
    uid: dict[str, int],
    reason: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "uid": uid,
        "data": {
            "extra_info": {
                "task_id": data.id,
                "data_source": data.data_source,
            },
        },
        "env": {
            "rollout": {
                "state": "failed",
                "finish_reason": "failed",
                "extra_info": {
                    "failure_reason": reason,
                    **(metadata or {}),
                },
            },
            "judger": None,
            "agent": None,
        },
    }


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────


DEFAULT_GATEWAY = "http://env-gateway.ailab.ailab.ai"
DEFAULT_LAGENT_SRC = "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent"
# DEFAULT_LAGENT_SRC = "/mnt/shared-storage-user/llmit/user/wangziyi/projs/lagent"


def _load_dataset_from_config(config_path: Path) -> Any:
    """Exec a config file on the host; return its ``dataset`` binding.

    The config is just a Python file that imports a project package and
    instantiates a dataset class.  Project must already be on PYTHONPATH;
    runner doesn't auto-add anything.

    Example config::

        # configs/claw_bench_calendar.py
        from claw_bench.dataset import ClawBench
        from claw_bench.pipeline import claw_pipeline
        dataset = ClawBench(tasks_root="/data/bench/claw-bench/tasks",
                            pipeline=claw_pipeline())
    """
    ns: dict[str, Any] = {}
    exec(compile(config_path.read_text(encoding="utf-8"), str(config_path), "exec"), ns)
    if "dataset" not in ns:
        raise KeyError(f"`dataset` not defined in {config_path}")
    return ns["dataset"]


async def _run_one(
    dataset: Any,
    task_dir: Path,
    provider: Any,
    *,
    uid: dict[str, int],
    lagent_src_dir: str | None,
    llm_model: str | None,
    llm_base_url: str | None,
    llm_api_key: str | None,
) -> dict[str, Any]:
    data = dataset.load_task(task_dir)
    runner: Runner = dataset.pipeline
    get_logger().info(f"running task={data.id} (dataset={dataset.name}, uid={uid})")
    return await runner.run_single(
        task_dir,
        data,
        uid,
        provider=provider,
        lagent_src_dir=lagent_src_dir,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
    )


async def main_async(args: argparse.Namespace) -> int:
    # Size the default thread pool so ``asyncio.to_thread`` can actually
    # fan out to ``args.concurrency`` concurrent sync calls.  Python's
    # default is ``min(32, cpu+4)`` — without this override, 400-way
    # concurrency gets paced into batches of 32 and looks serial.
    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=max(args.concurrency * 2, 64),
            thread_name_prefix="runner-io",
        )
    )

    provider = GatewayProvider(args.gateway)
    dataset = _load_dataset_from_config(Path(args.config))
    lagent_src = args.lagent_src or None

    if args.task_dirs:
        base_dirs = [Path(p) for p in args.task_dirs]
    else:
        base_dirs = [td for td, _ in dataset.iter_tasks()]
        if args.limit:
            base_dirs = base_dirs[: args.limit]

    if not base_dirs:
        get_logger().error("no tasks to run")
        return 1

    # Stress: each repeat runs the full task list again with a distinct uid
    # root_id — fresh sandboxes per iteration, exercises gateway concurrency.
    repeats = max(1, args.repeat)
    jobs: list[tuple[Path, dict[str, int]]] = [
        (td, {"root_id": r, "action_id": 0, "observation_id": 0}) for r in range(repeats) for td in base_dirs
    ]
    total = len(jobs)

    print(f"TotalTask: {total} (base={len(base_dirs)} × repeat={repeats}, concurrency={args.concurrency})")
    sem = asyncio.Semaphore(max(1, args.concurrency))
    run_start = time.monotonic()
    completed = 0
    completed_lock = asyncio.Lock()

    async def _guarded(job_idx: int, td: Path, uid: dict[str, int]) -> dict[str, Any]:
        async with sem:
            started = time.monotonic()
            result = await _run_one(
                dataset,
                td,
                provider,
                uid=uid,
                lagent_src_dir=lagent_src,
                llm_model=args.llm_model,
                llm_base_url=args.llm_base_url,
                llm_api_key=args.llm_api_key,
            )
            nonlocal completed
            async with completed_lock:
                completed += 1
                elapsed = time.monotonic() - run_start
                rate = completed / max(elapsed, 0.001)
                remaining = (total - completed) / max(rate, 0.001)
                state = (result.get("env", {}).get("rollout") or {}).get("state", "?")
                tid = (result.get("data", {}).get("extra_info") or {}).get("task_id", "?")
                took = time.monotonic() - started
                get_logger().info(
                    f"[{completed}/{total}] {tid} {state} took={took:.1f}s | "
                    f"elapsed={int(elapsed)}s eta={int(remaining)}s (rate={rate:.1f}/s)"
                )
            return result

    results = await asyncio.gather(*[_guarded(i, td, uid) for i, (td, uid) in enumerate(jobs)])
    if args.verbose:
        for r in results:
            print(json.dumps(r, ensure_ascii=False, indent=2, default=str))

    report_path = _write_run_report(results, Path(args.report_dir))
    _print_summary(results, report_path)
    any_failed = any(r["env"]["rollout"]["state"] == "failed" for r in results)
    return 1 if any_failed else 0


# ─────────────────────────────────────────────────────────────────
# Post-run report
# ─────────────────────────────────────────────────────────────────


# Stderr patterns we can confidently categorize.  Keep these narrow — the
# "other" bucket catches anything ambiguous.
_CATEGORIES: list[tuple[str, str]] = [
    # Sandbox / HTTP transport
    ("sandbox_unreachable", r"404 Client Error.*Not Found for url.*/(exec|upload|download|health)"),
    ("sandbox_http_error", r"HTTPError|ConnectionError|ReadTimeout|5\d\d Server Error"),
    # Missing upstream data (pre_entry.sh cp fails, or awk/python open on missing file)
    (
        "upstream_data_missing",
        r"(?:cp: cannot stat|cannot open|No such file or directory) ['\"]?([^'\"\n]+environment/data[^'\"\n]+)",
    ),
    # Missing deps
    ("missing_python_pkg", r"ModuleNotFoundError: No module named ['\"]([^'\"]+)"),
    ("missing_system_cmd", r"(\S+): command not found"),
    # Script bugs
    ("script_syntax_error", r"SyntaxError:|syntax error near unexpected token"),
    ("script_name_error", r"NameError:"),
    ("script_bash_unbound", r"unbound variable"),
    ("script_type_error", r"TypeError:"),
    ("script_file_not_found", r"FileNotFoundError:"),
    ("timeout", r"return_code=124|Command timed out"),
]


def _categorize_failure(reason: str) -> tuple[str, str | None]:
    """Return (category, detail).

    ``detail`` carries the captured group for
    missing_python_pkg / missing_system_cmd; None for script-bug categories.
    """
    for category, pattern in _CATEGORIES:
        m = re.search(pattern, reason)
        if m:
            detail = m.group(1) if m.groups() else None
            return category, detail
    return "other", None


def _summarize(r: dict) -> dict:
    rollout = r.get("env", {}).get("rollout", {}) or {}
    state = rollout.get("state")
    task_id = r.get("data", {}).get("extra_info", {}).get("task_id")
    if state == "completed":
        total = (r.get("env", {}).get("judger") or {}).get("total")
        return {"state": "passed", "task_id": task_id, "total": total}

    extra = rollout.get("extra_info", {}) or {}
    reason = extra.get("failure_reason", "") or ""
    category, detail = _categorize_failure(reason)
    return {
        "state": "failed",
        "task_id": task_id,
        "category": category,
        "detail": detail,
        "reason": reason[:800],
    }


def _write_run_report(results: list[dict], out_dir: Path) -> Path:
    summaries = [_summarize(r) for r in results]
    passed = [s for s in summaries if s["state"] == "passed"]
    failed = [s for s in summaries if s["state"] == "failed"]

    # Aggregate dep asks so user can add to pre_entry.sh in one place.
    missing_pkgs = sorted(
        {s["detail"] for s in failed if s.get("category") == "missing_python_pkg" and s.get("detail")}
    )
    missing_cmds = sorted(
        {s["detail"] for s in failed if s.get("category") == "missing_system_cmd" and s.get("detail")}
    )
    script_bugs = sorted({s["task_id"] for s in failed if s.get("category", "").startswith("script_")})

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    path = out_dir / f"run_{ts}.json"
    path.write_text(
        json.dumps(
            {
                "ts": ts,
                "totals": {
                    "total": len(summaries),
                    "passed": len(passed),
                    "failed": len(failed),
                },
                "missing_python_pkgs": missing_pkgs,
                "missing_system_cmds": missing_cmds,
                "script_bug_task_ids": script_bugs,
                "per_task": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _print_summary(results: list[dict], report_path: Path) -> None:
    summaries = [_summarize(r) for r in results]
    passed = [s for s in summaries if s["state"] == "passed"]
    failed = [s for s in summaries if s["state"] == "failed"]

    print(f"\n{'=' * 60}")
    print(f"  {len(passed)}/{len(summaries)} passed  |  {len(failed)} failed")
    print(f"  report: {report_path}")
    if failed:
        by_cat: dict[str, list[str]] = {}
        for s in failed:
            by_cat.setdefault(s.get("category") or "other", []).append(s["task_id"])
        for cat, ids in sorted(by_cat.items()):
            print(f"  [{cat}] {len(ids)}: {', '.join(ids[:5])}" + (" …" if len(ids) > 5 else ""))
    print(f"{'=' * 60}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "task_dirs",
        nargs="*",
        help="Task root directories (omit to iterate the dataset).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a config .py that imports a project package and binds `dataset = <DatasetClass>(...)`.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max tasks when iterating.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the whole task list N times (stress test; each repeat has a "
        "distinct uid.root_id, so sandboxes are not shared across runs).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1024,
        help="Max tasks running in parallel (semaphore-gated).",
    )
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    parser.add_argument(
        "--lagent-src",
        default=DEFAULT_LAGENT_SRC,
        help="Local path to lagent source.  Pass '' to skip upload.",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument(
        "--report-dir",
        default="/mnt/shared-storage-user/llmit/user/liukuikun/workspace/xtuner/work_dir/reports",
        help="Dir to write per-run summary reports (totals + failure categories + dep asks).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also dump every task's full result JSON to stdout.",
    )
    args = parser.parse_args()
    if args.lagent_src == "":
        args.lagent_src = None

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

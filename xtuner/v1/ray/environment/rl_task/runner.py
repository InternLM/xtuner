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
from xtuner.v1.ray.environment.trace import emit_diagnostic, span  # noqa: E402
from xtuner.v1.utils import get_logger  # noqa: E402


# ─────────────────────────────────────────────────────────────────
# Failure-path diagnostics
# ─────────────────────────────────────────────────────────────────
#
# The in-sandbox lagent daemon writes its log to ``_DAEMON_LOG_PATH``.
# On failure we pull the whole log to shared storage so post-mortem
# survives Ray driver-log dedup.  The budget below is generous since
# daemon logs can grow to several MB on long rollouts.
_DAEMON_LOG_PATH = "/tmp/agent_daemon.log"
_DAEMON_LOG_DUMP_TIMEOUT_SEC = 30.0


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
            # Structured failure metadata populated as run_single proceeds; the
            # fields below end up in ``_mark_failed``'s ``extra_info`` so the
            # install_agent_env classifier and fates.jsonl get the exact stage
            # + rc the runner saw, rather than a best-effort regex guess.
            "failed_stage": None,
            "entry_rc": None,
            "entry_stderr": None,
            "exception_type": None,
            "sandbox_env_id": None,
            "sandbox_url": None,
        }

        client = None
        env_id: str | None = None
        tid = data.id
        uid_obs = str(uid.get("observation_id") or "")
        try:
            with span(uid_obs, "run_single_total", task_id=tid) as total_span:
                get_logger().info(f"[{tid}] acquiring sandbox (image={self.infer.sandbox.image})")
                t0 = time.monotonic()
                try:
                    with span(uid_obs, "acquire", task_id=tid) as acquire_span:
                        client, env_id = await _acquire_ready_sandbox(
                            provider,
                            self.infer.sandbox,
                        )
                        # Annotate the span BEFORE it exits so the sandbox_url
                        # lands in the emitted acquire span record.  This is
                        # the primary observability signal for in-flight
                        # debugging (``scripts/trace/inflight.py`` uses it to
                        # surface "which sandbox is this sample stuck in").
                        ctx["sandbox_env_id"] = env_id
                        ctx["sandbox_url"] = _sandbox_url_of(client)
                        acquire_span.annotate(
                            sandbox_env_id=env_id,
                            sandbox_url=ctx["sandbox_url"],
                            sandbox_image=self.infer.sandbox.image,
                        )
                except Exception:
                    ctx["failed_stage"] = "acquire"
                    raise
                get_logger().info(f"[{tid}] sandbox ready env_id={env_id} ({time.monotonic() - t0:.1f}s)")

                get_logger().info(f"[{tid}] infer: start ({len(self.infer.pre)} pre-hooks)")
                t1 = time.monotonic()
                try:
                    with span(uid_obs, "infer", task_id=tid) as infer_span:
                        # Detached-entry model: SandboxStage.run launches the
                        # entry with detach=True and polls process liveness +
                        # rc file to detect completion.  Post-hooks on entry
                        # failure swallow their own exceptions (see sandbox.py)
                        # so the real entry rc reaches us here.
                        infer_result = await self.infer.run(client, ctx)
                        if not infer_result.ok:
                            infer_span.mark_error(f"rc={infer_result.return_code}: {infer_result.error}")
                except Exception as exc:
                    # SandboxStage.run itself raised.  Distinguish pre-hook
                    # failure (no entry output yet) from post-hook failure
                    # (entry rc available via ctx["result"]).
                    prev = ctx.get("result")
                    if prev is not None and prev.return_code != 0:
                        ctx["failed_stage"] = _classify_entry_rc(prev.return_code)
                        ctx["entry_rc"] = prev.return_code
                        ctx["entry_stderr"] = (prev.stderr or "")[:400]
                    elif prev is not None:
                        ctx["failed_stage"] = "infer_posthook"
                        ctx["entry_rc"] = 0
                    else:
                        ctx["failed_stage"] = "infer_prehook"
                    ctx["exception_type"] = type(exc).__name__
                    raise
                get_logger().info(f"[{tid}] infer: done rc={infer_result.return_code} ({time.monotonic() - t1:.1f}s)")
                if not infer_result.ok:
                    ctx["failed_stage"] = _classify_entry_rc(infer_result.return_code)
                    ctx["entry_rc"] = infer_result.return_code
                    ctx["entry_stderr"] = (infer_result.stderr or "")[:400]
                    total_span.mark_error(f"infer failed: {infer_result.error}")
                    # Also dump diagnostics on non-exception rc!=0 paths (rc=-4
                    # sandbox_unreachable, rc=-3 timeout, rc=5/6/7 entry-script
                    # errors, etc.).  The dump records the attempt even when
                    # the daemon log download itself fails (e.g. sandbox is
                    # already gone in the rc=-4 case) so post-mortems always
                    # have *something* beyond the terse fate string.
                    await _dump_skipped_diagnostic(
                        client,
                        tid,
                        data,
                        RuntimeError(f"infer failed: return_code={infer_result.return_code}: {infer_result.error}"),
                        uid=uid_obs,
                    )
                    return _mark_failed(
                        data,
                        uid,
                        f"infer failed: {infer_result.error}",
                        metadata=_infer_metadata(ctx),
                    )

                get_logger().info(f"[{tid}] validate: start ({len(self.validate.judgers)} judgers)")
                t2 = time.monotonic()
                try:
                    with span(uid_obs, "validate", task_id=tid):
                        aggregated = await self.validate.run(
                            client,
                            ctx,
                            provider,
                            self.infer.sandbox.workspace_path,
                        )
                except Exception:
                    ctx["failed_stage"] = "validate"
                    raise
                get_logger().info(
                    f"[{tid}] validate: done total={aggregated.total:.4f} ({time.monotonic() - t2:.1f}s)"
                )

                return _mark_completed(data, uid, metadata=_infer_metadata(ctx), judge=aggregated, infer=infer_result)
        except Exception as exc:
            # Any stage still without a tag at this point is an uncategorized
            # runner-level exception — the span for the throwing stage already
            # has the exception recorded, so this label is just a fate-bucket
            # fallback.
            if ctx.get("failed_stage") is None:
                ctx["failed_stage"] = "runner_exc"
            ctx["exception_type"] = ctx.get("exception_type") or type(exc).__name__
            # Try to capture daemon log tail to a per-task file on shared
            # storage so the post-mortem survives Ray driver-log dedup.
            # ``client.download_file`` may well fail when the daemon is dead
            # (same upstream /download endpoint) — we swallow that and record
            # the attempt outcome either way.
            await _dump_skipped_diagnostic(client, tid, data, exc, uid=uid_obs)
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
    """Pack structured runner state into the ``extra_info`` metadata that
    ``_mark_failed`` / ``_mark_completed`` attach to their result envelope.

    The install_agent_env fate emitter reads these fields directly instead
    of regex-parsing ``failure_reason``, which is both fragile and opaque
    (rc22 had 3468 fates misclassified as ``posthook_download_404`` when
    the real cause was ``infer_timeout`` — the post-hook error string
    simply overwrote the entry rc in a reason-string-only world).
    """
    md: dict[str, Any] = {}
    chosen = ctx.get("chosen_agent")
    if chosen is not None:
        md["agent_name"] = chosen.name
    for key in (
        "failed_stage",
        "entry_rc",
        "entry_stderr",
        "exception_type",
        "sandbox_image",
        "sandbox_env_id",
        "sandbox_url",
    ):
        val = ctx.get(key)
        if val is not None:
            md[key] = val
    return md


def _classify_entry_rc(rc: int) -> str:
    """Map a detached-entry return code to a fate ``failed_stage`` label.

    Mirrors the rc encodings produced by
    ``wait_for_detached_entry`` + entry scripts:

      * ``-4`` — sandbox env unreachable (gateway evicted, TTL expired,
        container killed).  Separates "sandbox died under us" from
        "entry script misbehaved" so fate stats attribute the loss
        correctly.
      * ``-3`` — host-side poll loop exceeded ``SandboxStage.timeout``
        (agent hung through the window).
      * ``-2`` — daemon process disappeared from pgrep while entry ran.
      * ``-1`` — wrapper shell died without writing the rc file
        (SIGKILL / container eviction / fatal shell error).
      * positive — entry script's own exit code (e.g. lagent_entry.sh
        uses 4 = daemon boot failure, 5 = chat call error, 6 = state_dict
        error, 7 = get_messages error).
    """
    if rc == -4:
        return "sandbox_unreachable"
    if rc == -3:
        return "infer_timeout"
    if rc == -2:
        return "infer_daemon_gone"
    if rc == -1:
        return "infer_pid_lost"
    if rc == -9 or rc == 137:
        return "infer_oom"
    if rc > 0:
        return f"infer_rc_{rc}"
    return "infer_rc_unknown"


def _sandbox_url_of(client: Any) -> str | None:
    """Best-effort extraction of the sandbox base URL from a client.

    Different client libraries expose this under different attributes —
    try the common ones and fall back to ``str(client)`` so fates always
    have *something* pointing at the right sandbox.
    """
    for attr in ("base_url", "url", "endpoint", "_base_url"):
        val = getattr(client, attr, None)
        if val:
            return str(val)
    return None


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


async def _dump_skipped_diagnostic(
    client: Any | None,
    tid: str,
    data: TaskData,
    exc: BaseException,
    uid: str | None = None,
) -> None:
    """Download the daemon log and hand it off to ``trace.emit_diagnostic`` so
    per-failure bundles land under ``$WORK_DIR/trace/diagnostics/``.

    Runner owns the sandbox client (so the daemon log fetch lives here),
    but the file-writing convention is centralized in
    :mod:`xtuner.v1.ray.environment.trace` to keep the full observability
    namespace (``fates.jsonl`` / ``spans.jsonl`` / ``llm_calls.jsonl`` /
    ``diagnostics/``) under one roof.  Both are best-effort — download
    timeout or write errors are swallowed.

    Args:
        client (Any | None): SandboxClient or None if acquire failed.
        tid (str): Task id for log tags.
        data (TaskData): Owning task record.
        exc (BaseException): Exception being handled.
        uid (str | None): Per-sample observation id; threaded in so
            diagnostics can be cross-referenced with fates.jsonl /
            spans.jsonl entries of the same sample.
    """
    daemon_log: bytes | None = None
    download_err: str | None = None
    if client is not None:
        try:
            daemon_log = await asyncio.wait_for(
                client.download_file(_DAEMON_LOG_PATH),
                timeout=_DAEMON_LOG_DUMP_TIMEOUT_SEC,
            )
        except Exception as dl_exc:
            download_err = f"{type(dl_exc).__name__}: {dl_exc}"
    else:
        download_err = "no sandbox client available"

    emit_diagnostic(
        task_id=data.id,
        uid=uid,
        data_source=getattr(data, "data_source", None),
        exception_type=type(exc).__name__,
        exception_msg=str(exc),
        daemon_log=daemon_log,
        download_err=download_err,
    )


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

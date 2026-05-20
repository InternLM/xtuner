"""Runner — top-level orchestrator for one task rollout.

Owns sandbox lifecycle, threads one :class:`AgentRolloutItem` through the
stages (infer → validate), and fills the result fields in place. All real work
is hook-driven (:class:`sandbox.SandboxStage` pre/entry/post) +
:class:`validator.JudgerValidator`.

Invocation:

    # config.py (user-written)
    from claw_bench import ClawBench           # bench project on PYTHONPATH
    dataset = ClawBench(tasks_root="/data/bench/claw-bench/tasks")

    # Run:
    python runner.py --config config.py TASK_DIR [TASK_DIR ...]
    python runner.py --config config.py --limit 10     # iterate whole dataset

``dataset`` in the config provides either a :class:`Runner` object or a
lagent-style runner config (``dict(type=Runner, ...)``), plus
``load_task(task_dir) → AgentRolloutItem``.
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
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lagent.utils import create_object  # noqa: E402
from lagent.serving.sandbox.providers.gateway import GatewayProvider  # noqa: E402

from xtuner.v1.ray.environment.rl_task.sandbox import SandboxStage  # noqa: E402
from xtuner.v1.ray.environment.rl_task.schemas import (  # noqa: E402
    AgentRolloutItem,
    RolloutError,
    RolloutStatus,
    SandboxSpec,
    StageStatus,
)
from xtuner.v1.ray.environment.rl_task.validator import JudgerValidator  # noqa: E402
from xtuner.v1.ray.environment.trace import span  # noqa: E402
from xtuner.v1.utils import get_logger  # noqa: E402


# ─────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────


class Runner:
    """Pairs one infer stage with one validator."""

    def __init__(
        self,
        *,
        provider: Any | None = None,
        sandboxes: Mapping[str, SandboxSpec | dict[str, Any]],
        infer: SandboxStage | dict[str, Any],
        validate: JudgerValidator | dict[str, Any],
        acquire_max_attempts: int = 3,
        health_max_wait_sec: float = 600.0,
        health_poll_interval_sec: float = 2.0,
    ):
        self.provider = create_object(provider)
        self.sandboxes = {name: create_object(spec) for name, spec in sandboxes.items()}
        self.infer = create_object(infer)
        self.validate = create_object(validate)
        self.acquire_max_attempts = acquire_max_attempts
        self.health_max_wait_sec = health_max_wait_sec
        self.health_poll_interval_sec = health_poll_interval_sec

    async def run(self, item: AgentRolloutItem) -> AgentRolloutItem:
        """Run one rollout sample and return the same item with result fields filled."""
        provider = self.provider
        if provider is None:
            raise RuntimeError("Runner has no sandbox provider; pass provider at build time")

        if item.task_root is None:
            raise ValueError("AgentRolloutItem.task_root is required by Runner.run")
        if not item.uid:
            raise ValueError("AgentRolloutItem.uid is required by Runner.run")

        uid = item.uid
        infer_sandbox = self._stage_sandbox_name(self.infer)
        infer_spec = self.sandboxes[infer_sandbox]
        sandbox_clients: dict[str, Any] = {}
        sandbox_env_ids: dict[str, str] = {}
        sandbox_urls: dict[str, str | None] = {}

        async def get_sandbox(name: str) -> Any:
            if name in sandbox_clients:
                return sandbox_clients[name]
            spec = self.sandboxes[name]
            client, env_id = await _acquire_ready_sandbox(
                provider,
                spec,
                max_attempts=self.acquire_max_attempts,
                health_max_wait_sec=self.health_max_wait_sec,
                health_poll_interval_sec=self.health_poll_interval_sec,
            )
            sandbox_clients[name] = client
            sandbox_env_ids[name] = env_id
            sandbox_urls[name] = _sandbox_url_of(client)
            return client

        item.status = RolloutStatus.RUNNING
        item.infer.runtime.update(dict(getattr(self.infer, "runtime", {}) or {}))
        item.infer.sandbox_name = infer_sandbox
        item.infer.sandbox_image = infer_spec.image
        item.infer.workspace = infer_spec.workspace_path

        tid = item.id
        uid_obs = str(uid.get("observation_id") or "")
        try:
            with span(uid_obs, "run_total", task_id=tid) as total_span:
                get_logger().info(f"[{tid}] acquiring sandbox name={infer_sandbox} image={infer_spec.image}")
                t0 = time.monotonic()
                try:
                    with span(uid_obs, "acquire", task_id=tid) as acquire_span:
                        infer_client = await get_sandbox(infer_sandbox)
                        # Annotate the span BEFORE it exits so the sandbox_url
                        # lands in the emitted acquire span record.  This is
                        # the primary observability signal for in-flight
                        # debugging (``scripts/trace/inflight.py`` uses it to
                        # surface "which sandbox is this sample stuck in").
                        item.infer.sandbox_env_id = sandbox_env_ids[infer_sandbox]
                        sandbox_url = sandbox_urls[infer_sandbox]
                        if sandbox_url is not None:
                            item.infer.sandbox_url = sandbox_url
                        acquire_span.annotate(
                            sandbox_name=infer_sandbox,
                            sandbox_env_id=item.infer.sandbox_env_id,
                            sandbox_url=sandbox_url,
                            sandbox_image=infer_spec.image,
                        )
                except Exception as exc:
                    item.infer.status = StageStatus.FAILED
                    item.infer.error = RolloutError(
                        stage="infer",
                        category="acquire",
                        type=type(exc).__name__,
                        message=str(exc),
                    )
                    raise
                get_logger().info(
                    f"[{tid}] sandbox ready name={infer_sandbox} env_id={sandbox_env_ids[infer_sandbox]} "
                    f"({time.monotonic() - t0:.1f}s)"
                )

                get_logger().info(f"[{tid}] infer: start ({len(self.infer.pre)} pre-hooks)")
                t1 = time.monotonic()
                try:
                    with span(uid_obs, "infer", task_id=tid) as infer_span:
                        infer_result = await self.infer.run(infer_client, item, item.infer)
                        if not infer_result.ok:
                            infer_span.mark_error(f"rc={infer_result.return_code}: {infer_result.error}")
                except Exception as exc:
                    prev = item.infer.entry_result
                    if item.infer.error is not None:
                        category = item.infer.error.category
                    elif prev is not None:
                        category = "infer_posthook"
                    else:
                        category = "infer_prehook"
                    item.infer.error = item.infer.error or RolloutError(
                        stage="infer",
                        category=category,
                        type=type(exc).__name__,
                        message=str(exc),
                    )
                    raise
                get_logger().info(f"[{tid}] infer: done rc={infer_result.return_code} ({time.monotonic() - t1:.1f}s)")
                if not infer_result.ok:
                    if item.infer.error is not None:
                        category = item.infer.error.category
                    else:
                        category = "infer"
                    total_span.mark_error(f"infer failed: {infer_result.error}")
                    item.status = RolloutStatus.FAILED
                    item.error = item.infer.error or RolloutError(
                        stage="infer",
                        category=category,
                        type="StageResult",
                        message=f"infer failed: {infer_result.error}",
                    )
                    return item

                get_logger().info(f"[{tid}] validate: start ({len(self.validate.judgers)} judgers)")
                t2 = time.monotonic()
                try:
                    with span(uid_obs, "validate", task_id=tid):
                        aggregated = await self.validate.run(
                            item,
                            get_sandbox,
                            sandboxes=self.sandboxes,
                            infer_sandbox=infer_sandbox,
                            infer_workspace=infer_spec.workspace_path,
                        )
                except Exception as exc:
                    item.validation.status = StageStatus.FAILED
                    item.validation.error = RolloutError(
                        stage="validate",
                        category="validate",
                        type=type(exc).__name__,
                        message=str(exc),
                    )
                    raise
                get_logger().info(
                    f"[{tid}] validate: done total={aggregated.total:.4f} ({time.monotonic() - t2:.1f}s)"
                )

                item.status = RolloutStatus.COMPLETED
                item.score = aggregated.total
                return item
        except Exception as exc:
            # Any stage still without a tag at this point is an uncategorized
            # runner-level exception. The span for the throwing stage already
            # has the exception recorded, so this label is just a structured
            # error-category fallback.
            if item.error is None:
                stage_error = item.validation.error if item.validation.error is not None else item.infer.error
                if stage_error is not None:
                    item.error = stage_error
                else:
                    item.error = RolloutError(
                        stage="runner",
                        category="runner_exc",
                        type=type(exc).__name__,
                        message=str(exc),
                    )
            get_logger().error(f"[{tid}] runner failed: {exc}\n{traceback.format_exc()}")
            item.status = RolloutStatus.FAILED
            return item
        finally:
            for name, client in reversed(list(sandbox_clients.items())):
                try:
                    await provider.delete(sandbox_env_ids[name])
                except Exception as exc:
                    get_logger().warning(f"[{tid}] gateway delete failed for sandbox {name}: {exc}")
                try:
                    await client.aclose()
                except Exception as exc:
                    get_logger().warning(f"[{tid}] client aclose failed for sandbox {name}: {exc}")

    def _stage_sandbox_name(self, stage: SandboxStage) -> str:
        name = stage.sandbox
        if not isinstance(name, str):
            raise TypeError(f"SandboxStage.sandbox must be a sandbox name, got {type(name).__name__}")
        if name not in self.sandboxes:
            raise KeyError(f"unknown sandbox {name!r}; known sandboxes: {sorted(self.sandboxes)}")
        return name


def _sandbox_url_of(client: Any) -> str | None:
    """Best-effort extraction of the sandbox base URL from a client.

    Different client libraries expose this under different attributes;
    try the common ones so downstream observability has a sandbox pointer.
    """
    for attr in ("base_url", "url", "endpoint", "_base_url"):
        val = getattr(client, attr, None)
        if val:
            return str(val)
    return None


async def _acquire_ready_sandbox(
    provider: Any,
    spec: Any,
    *,
    max_attempts: int,
    health_max_wait_sec: float,
    health_poll_interval_sec: float,
) -> tuple[Any, str]:
    """Create a sandbox + wait for /health to respond before returning.

    Retries on create-fail or /health-never-becomes-ok. Burst protection
    is enforced inside ``provider.create`` via a token-bucket rate limiter.

    Args:
        provider (Any): Async sandbox provider exposing ``create`` /
            ``delete``.
        spec (Any): Sandbox spec with ``image`` and ``ttl_seconds``.

    Returns:
        tuple[Any, str]: Client and env_id of a healthy sandbox.
    """

    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            create_kwargs: dict[str, Any] = {}
            if getattr(spec, "key", None):
                create_kwargs["key"] = spec.key
            if getattr(spec, "env_vars", None):
                create_kwargs["env_vars"] = spec.env_vars
            if getattr(spec, "resources", None):
                create_kwargs["resources"] = spec.resources
            client, env_id = await provider.create(
                image_tag=spec.image,
                ttl_seconds=spec.ttl_seconds,
                **create_kwargs,
            )
        except Exception as exc:
            last_err = exc
            get_logger().warning(f"provider.create attempt {attempt} failed: {exc}")
            await asyncio.sleep(min(2**attempt, 8))
            continue

        if await _wait_healthy(
            client,
            max_wait_sec=health_max_wait_sec,
            poll_interval_sec=health_poll_interval_sec,
        ):
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

    raise RuntimeError(f"could not acquire a healthy sandbox after {max_attempts} attempts: {last_err}")


async def _wait_healthy(client: Any, *, max_wait_sec: float, poll_interval_sec: float) -> bool:
    """Poll /health until it returns ``ok: True`` or the budget is
    exhausted."""
    deadline = time.monotonic() + max_wait_sec
    while time.monotonic() < deadline:
        try:
            h = await client.health_check()
            if h.get("ok"):
                return True
        except Exception as exc:
            get_logger().debug(f"health poll error: {exc}")
        await asyncio.sleep(poll_interval_sec)
    return False


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────


def _load_dataset_from_config(config_path: Path) -> Any:
    """Exec a config file on the host; return its ``dataset`` binding.

    The config is just a Python file that imports a project package and
    instantiates a dataset class.  Project must already be on PYTHONPATH;
    runner doesn't auto-add anything.

    Example config::

        # configs/claw_bench_calendar.py
        from claw_bench.dataset import ClawBench
        from claw_bench.pipeline import runner
        dataset = ClawBench(tasks_root="/data/bench/claw-bench/tasks",
                            pipeline=runner)
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
    item = dataset.load_task(task_dir)
    pipeline_spec = item.pipeline or dataset.pipeline
    runner_config = deepcopy(pipeline_spec) if isinstance(pipeline_spec, dict) else pipeline_spec
    if isinstance(runner_config, dict):
        runner_config["provider"] = provider
        infer = dict(runner_config.get("infer") or {})
        runtime = dict(infer.get("runtime") or {})
        runtime.update(
            {
                "lagent_src_dir": lagent_src_dir,
                "llm_model": llm_model,
                "llm_base_url": llm_base_url,
                "llm_api_key": llm_api_key,
            }
        )
        infer["runtime"] = runtime
        runner_config["infer"] = infer
    runner: Runner = create_object(runner_config)
    get_logger().info(f"running task={item.id} (dataset={dataset.name}, uid={uid})")
    result = await runner.run(item.model_copy(update={"task_root": Path(task_dir), "uid": uid}))
    dumped = result.model_dump(mode="json", exclude={"artifacts", "pipeline"})
    dumped["artifacts"] = {
        key: f"<{len(value)} bytes>" if isinstance(value, (bytes, bytearray)) else value
        for key, value in result.artifacts.items()
    }
    return dumped


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
                state = result.get("status", "?")
                tid = result.get("id", "?")
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
    any_failed = any(r.get("status") == RolloutStatus.FAILED.value for r in results)
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
    ("timeout", r"Command timed out"),
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
    state = r.get("status")
    task_id = r.get("id")
    if state == RolloutStatus.COMPLETED.value:
        validation = r.get("validation") or {}
        judge = validation.get("result") or {}
        total = r.get("score", judge.get("total"))
        return {"state": "passed", "task_id": task_id, "total": total}

    err = r.get("error") or (r.get("infer") or {}).get("error") or (r.get("validation") or {}).get("error") or {}
    reason = err.get("message") or state or ""
    category, detail = _categorize_failure(reason)
    return {
        "state": "failed",
        "task_id": task_id,
        "category": err.get("category") or category,
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
    parser.add_argument("--gateway", default="http://env-gateway.ailab.ailab.ai")
    parser.add_argument(
        "--lagent-src",
        default="/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent",
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

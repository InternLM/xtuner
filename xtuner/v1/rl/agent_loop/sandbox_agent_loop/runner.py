"""Runner — top-level orchestrator for one task rollout.

Owns sandbox lifecycle, threads one :class:`AgentRolloutItem` through the
stages (infer → validate), and fills the result fields in place. All real work
is hook-driven (:class:`sandbox.SandboxStage` pre/entry/post) and the configured
validation judger.

``dataset`` in the config provides either a :class:`Runner` object or a
lagent-style runner config (``dict(type=Runner, ...)``), plus
``load_task(task_dir) → AgentRolloutItem``.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from lagent.utils import create_object

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.finalization import CleanupError
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.judger import Judger
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import SandboxPool, SandboxStage
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    RolloutError,
    RolloutStatus,
    StageRecord,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.trace import span
from xtuner.v1.utils import get_logger


@dataclass
class _RunTimings:
    acquire: float | None = None
    infer: float | None = None
    validate: float | None = None


class Runner:
    """Pairs one infer stage with one validation judger."""

    def __init__(
        self,
        *,
        pool: SandboxPool | dict[str, Any],
        infer: SandboxStage | dict[str, Any],
        validate: Judger | dict[str, Any],
    ):
        # ``pool`` is stored as a *template*: pass a dict for normal use (a
        # fresh SandboxPool is built per ``run`` call), or pass an already-built
        # SandboxPool for one-shot reuse (testing).  Sharing one SandboxPool
        # instance across concurrent ``run`` calls is unsafe.
        self._pool = pool
        self.infer = create_object(infer)
        self.validate = create_object(validate)

    async def run(self, item: AgentRolloutItem) -> AgentRolloutItem:
        """Run a sample and preserve cancellation after bounded cleanup.

        Returns:
            AgentRolloutItem: The input item with result fields filled.
        """
        return await self.run_with_pool(item, self.prepare_trial_pool(item))

    def prepare_trial_pool(self, item: AgentRolloutItem) -> SandboxPool:
        """Validate input and apply this trial's overrides to a fresh Pool.

        Recipes that need the Pool before execution can use this operation and
        then :meth:`run_with_pool`. They own closing the Pool if their setup
        fails before execution starts. Neither the template nor the item's
        overrides are modified. A pre-built Pool remains a one-shot alternative
        and cannot accept overrides.
        """
        self._validate_input(item)
        if isinstance(self._pool, SandboxPool):
            if item.pipeline_overrides:
                raise RuntimeError("pipeline_overrides require dict-form pool config; got pre-built SandboxPool")
            pool = self._pool
        else:
            pool_cfg = deepcopy(self._pool)
            override = (item.pipeline_overrides or {}).get("pool")
            if override:
                _deep_merge(pool_cfg, deepcopy(override))
            pool = create_object(pool_cfg)
        return pool

    async def run_with_pool(self, item: AgentRolloutItem, pool: SandboxPool) -> AgentRolloutItem:
        """Run stages with an already prepared trial Pool and finalize it.

        Pool configuration, including overrides, must already be applied via
        :meth:`prepare_trial_pool`. This operation never rebuilds the Pool or
        reapplies overrides; it retains the normal result and cancellation
        rules. The caller must not share the Pool between concurrent trials.
        """
        timings = _RunTimings()
        try:
            return await _run_sandbox_trial(pool, item, lambda: self._run_stages(item, pool, timings), self._fail)
        finally:
            self._log_final(item.id, item, timings.acquire, timings.infer, timings.validate)

    async def _run_stages(self, item: AgentRolloutItem, pool: SandboxPool, timings: _RunTimings) -> AgentRolloutItem:
        self._validate_input(item)
        tid = item.id
        uid_obs = str(item.uid) if item.uid is not None else ""
        try:
            infer_sandbox = _stage_sandbox_name(self.infer, pool)
            infer_spec = pool.spec(infer_sandbox)

            item.status = RolloutStatus.RUNNING
            item.infer.sandbox_name = infer_sandbox
            item.infer.sandbox_image = infer_spec.image
            item.infer.workspace = infer_spec.workspace_path

            with span(uid_obs, "run_total", task_id=tid) as total_span:
                # ─── acquire infer sandbox ───────────────────────────────
                t0 = time.monotonic()
                with span(uid_obs, "acquire", task_id=tid) as acquire_span:
                    infer_client = await pool.get(infer_sandbox, record=item.infer)
                    item.infer.sandbox_env_id = pool.env_id(infer_sandbox)
                    sandbox_url = pool.url(infer_sandbox)
                    if sandbox_url is not None:
                        item.infer.sandbox_url = sandbox_url
                    acquire_span.annotate(
                        sandbox_name=infer_sandbox,
                        sandbox_env_id=item.infer.sandbox_env_id,
                        sandbox_url=sandbox_url,
                        sandbox_image=infer_spec.image,
                    )
                timings.acquire = time.monotonic() - t0

                # ─── infer ──────────────────────────────────────────────
                t1 = time.monotonic()
                with span(uid_obs, "infer", task_id=tid) as infer_span:
                    infer_result = await self.infer.run(infer_client, item, item.infer)
                    if not infer_result.ok:
                        infer_span.mark_error(_format_error(item.infer.error))
                timings.infer = time.monotonic() - t1
                if not infer_result.ok:
                    total_span.mark_error(_format_error(item.infer.error))
                    return self._fail(item, item.infer.error)

                # ─── validate ───────────────────────────────────────────
                t2 = time.monotonic()
                with span(uid_obs, "validate", task_id=tid):
                    validate_name = self.validate.name
                    validate_record = item.judgers.setdefault(
                        validate_name,
                        StageRecord(judger_name=validate_name),
                    )
                    score = float(await self.validate.run(item, pool, validate_record))
                timings.validate = time.monotonic() - t2
                item.reward = score

                item.status = RolloutStatus.COMPLETED
                return item
        except Exception as exc:
            promoted = (
                item.infer.error
                or _first_validate_error(item)
                or RolloutError(
                    stage="runner",
                    category="runner_exception",
                    type=type(exc).__name__,
                    message=str(exc),
                )
            )
            get_logger().error(f"[{tid}] traceback:\n{traceback.format_exc()}")
            return self._fail(item, promoted)

    def _log_final(
        self,
        tid: str,
        item: AgentRolloutItem,
        t_acquire: float | None,
        t_infer: float | None,
        t_validate: float | None,
    ) -> None:
        def format_seconds(label: str, val: Any) -> str | None:
            if isinstance(val, (int, float)):
                return f"{label}={val:.1f}s"
            return None

        parts: list[str] = [f"status={item.status.value}"]
        if item.reward is not None:
            parts.append(f"reward={item.reward:.4f}")
        if item.infer.agent is not None:
            parts.append(f"agent={item.infer.agent.name}")
        timing_parts = [
            format_seconds("t_acquire", t_acquire),
            format_seconds("t_acquire_rate_limit", item.infer.metadata.get("sandbox_acquire_rate_limit_wait_s")),
            format_seconds("t_sandbox_ready", item.infer.metadata.get("sandbox_create_to_ready_time_s")),
            format_seconds("t_install_agent", item.infer.metadata.get("install_agent_time_s")),
            format_seconds("t_infer", t_infer),
            format_seconds("t_validate", t_validate),
        ]
        parts.extend(part for part in timing_parts if part is not None)
        attempts = item.infer.metadata.get("sandbox_create_attempts")
        if isinstance(attempts, int):
            parts.append(f"sandbox_create_attempts={attempts}")
        if item.infer.sandbox_image:
            parts.append(f"sandbox_image={item.infer.sandbox_image}")
        if item.infer.sandbox_url:
            parts.append(f"sandbox_url={item.infer.sandbox_url}")
        if item.status == RolloutStatus.FAILED and item.error is not None:
            parts.append(f"error={item.error.stage or '?'}/{item.error.category}")
        get_logger().info(f"[{tid}] done {' '.join(parts)}")

    def _validate_input(self, item: AgentRolloutItem) -> None:
        if item.task_root is None:
            raise ValueError("AgentRolloutItem.task_root is required by Runner.run")
        if item.uid is None:
            raise ValueError("AgentRolloutItem.uid is required by Runner.run")

    def _fail(self, item: AgentRolloutItem, error: RolloutError | None) -> AgentRolloutItem:
        item.status = RolloutStatus.FAILED
        if item.error is None:
            item.error = error
        if error is not None:
            get_logger().error(f"[{item.id}] failed: {_format_error(error)}")
        return item


async def _run_sandbox_trial(
    pool: SandboxPool,
    item: AgentRolloutItem,
    operation: Callable[[], Awaitable[AgentRolloutItem]],
    fail: Callable[[AgentRolloutItem, RolloutError | None], AgentRolloutItem],
) -> AgentRolloutItem:
    """Own trial lifetime and translate cleanup facts into the rollout result.

    Invoke work only after scope entry succeeds. Keep the existing failure
    callback so Runner subclasses retain their error policy, including when
    scope entry or final close fails. Always publish the final snapshot before
    the caller logs the result.
    """
    try:
        async with pool.task_scope(str(item.uid) if item.uid is not None else item.id):
            return await operation()
    except asyncio.CancelledError:
        fail(item, RolloutError(stage="runner", category="rollout_cancelled", message="Rollout cancelled"))
        raise
    except CleanupError as exc:
        return fail(item, item.error or RolloutError(stage="cleanup", category=exc.kind, message=str(exc)))
    finally:
        item.artifacts["sandbox_cleanup"] = pool.cleanup


def _stage_sandbox_name(stage: SandboxStage, pool: SandboxPool) -> str:
    name = stage.sandbox
    if not isinstance(name, str):
        raise TypeError(f"SandboxStage.sandbox must be a sandbox name, got {type(name).__name__}")
    pool.validate_name(name)
    return name


def _first_validate_error(item: AgentRolloutItem) -> RolloutError | None:
    for record in item.judgers.values():
        if record.error is not None:
            return record.error
    return None


def _format_error(error: RolloutError | None) -> str:
    """Render a RolloutError uniformly for span / log output."""
    if error is None:
        return "unknown error"
    stage = f"{error.stage}/" if error.stage else ""
    typ = f" ({error.type})" if error.type else ""
    return f"{stage}{error.category}{typ}: {error.message}"


def _deep_merge(dst: dict, src: dict) -> dict:
    """In-place deep-merge ``src`` into ``dst``. Returns ``dst``.

    Dict values are merged recursively; non-dict values (including lists) are
    replaced. Used to apply ``item.pipeline_overrides`` onto the pool config.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

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

import time
import traceback
from copy import deepcopy
from typing import Any

from lagent.utils import create_object

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.judger import ComposeJudger, Judger
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import SandboxPool, SandboxStage
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import (
    AgentRolloutItem,
    RolloutError,
    RolloutStatus,
    StageRecord,
)
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.trace import span
from xtuner.v1.utils import get_logger


# ─────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────


class Runner:
    """Pairs one infer stage with one validation judger."""

    def __init__(
        self,
        *,
        pool: SandboxPool | dict[str, Any],
        infer: SandboxStage | dict[str, Any],
        validate: Judger | ComposeJudger | dict[str, Any],
    ):
        # ``pool`` is stored as a *template*: pass a dict for normal use (a
        # fresh SandboxPool is built per ``run`` call), or pass an already-built
        # SandboxPool for one-shot reuse (testing).  Sharing one SandboxPool
        # instance across concurrent ``run`` calls is unsafe.
        self._pool = pool
        self.infer = create_object(infer)
        self.validate = create_object(validate)

    async def run(self, item: AgentRolloutItem) -> AgentRolloutItem:
        """Run one rollout sample and return the same item with result fields
        filled.

        Each worker (SandboxStage / Judger / SandboxPool) writes its
        own ``record.status`` / ``record.error``.  The runner only orchestrates
        and promotes the first stage error to ``item.error`` when something
        fails.
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
                _deep_merge(pool_cfg, override)
            pool = create_object(pool_cfg)
        infer_sandbox = _stage_sandbox_name(self.infer, pool)
        infer_spec = pool.spec(infer_sandbox)

        item.status = RolloutStatus.RUNNING
        item.infer.sandbox_name = infer_sandbox
        item.infer.sandbox_image = infer_spec.image
        item.infer.workspace = infer_spec.workspace_path

        tid = item.id
        uid_obs = str(item.uid) if item.uid is not None else ""
        t_acquire: float | None = None
        t_infer: float | None = None
        t_validate: float | None = None
        try:
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
                t_acquire = time.monotonic() - t0

                # ─── infer ──────────────────────────────────────────────
                t1 = time.monotonic()
                with span(uid_obs, "infer", task_id=tid) as infer_span:
                    infer_result = await self.infer.run(infer_client, item, item.infer)
                    if not infer_result.ok:
                        infer_span.mark_error(_format_error(item.infer.error))
                t_infer = time.monotonic() - t1
                if not infer_result.ok:
                    total_span.mark_error(_format_error(item.infer.error))
                    return self._fail(item, item.infer.error)

                # ─── validate ───────────────────────────────────────────
                t2 = time.monotonic()
                with span(uid_obs, "validate", task_id=tid):
                    validate_name = getattr(self.validate, "name", "validate")
                    validate_record = item.judgers.setdefault(
                        validate_name,
                        StageRecord(judger_name=validate_name),
                    )
                    score = float(await self.validate.run(item, pool, validate_record))
                t_validate = time.monotonic() - t2
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
        finally:
            self._log_final(tid, item, t_acquire, t_infer, t_validate)
            await pool.release_all()

    def _log_final(
        self,
        tid: str,
        item: AgentRolloutItem,
        t_acquire: float | None,
        t_infer: float | None,
        t_validate: float | None,
    ) -> None:
        parts: list[str] = [f"status={item.status.value}"]
        if item.reward is not None:
            parts.append(f"reward={item.reward:.4f}")
        if t_acquire is not None:
            parts.append(f"t_acquire={t_acquire:.1f}s")
        if t_infer is not None:
            parts.append(f"t_infer={t_infer:.1f}s")
        if t_validate is not None:
            parts.append(f"t_validate={t_validate:.1f}s")
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

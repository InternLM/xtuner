"""Validator — fans out to a list of :class:`Judger` stages, aggregates scores.

Judgers are self-contained: each one's stage declares everything it needs
(uploads, env vars, entry, post-hooks).  The validator just orchestrates.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Literal

from xtuner.v1.ray.environment.rl_task.judgers import Judger
from xtuner.v1.ray.environment.rl_task.sandbox import exec_in, http_upload
from xtuner.v1.ray.environment.rl_task.schemas import AggregatedScore, JudgerResult, SandboxSpec


logger = logging.getLogger(__name__)


class JudgerValidator:
    """Run every judger, aggregate results.

    Init args:
        judgers (list[Judger]): Judger instances to fan out to.
        aggregator (str): ``"weighted_sum"`` / ``"mean"`` / ``"max"`` /
            ``"min"`` / ``"all_or_nothing"``.
        on_error (str): ``"zero"`` (sum over usable) or ``"fail"`` (any
            error → total=0, failed=True).
    """

    def __init__(
        self,
        judgers: list[Judger],
        *,
        aggregator: Literal[
            "weighted_sum",
            "mean",
            "max",
            "min",
            "all_or_nothing",
        ] = "weighted_sum",
        on_error: Literal["zero", "fail"] = "zero",
    ):
        self.judgers = list(judgers)
        self.aggregator = aggregator
        self.on_error = on_error

    async def run(
        self,
        infer_client: Any,
        ctx: dict[str, Any],
        provider: Any,
        infer_workspace: str,
    ) -> AggregatedScore:
        owned: dict[str, tuple[Any, str]] = {}
        results: list[JudgerResult] = []
        try:
            for j in self.judgers:
                results.append(await self._run_one(j, infer_client, ctx, provider, infer_workspace, owned))
        finally:
            for _name, (_c, env_id) in owned.items():
                try:
                    await asyncio.to_thread(_c.close)
                except Exception as exc:
                    logger.warning("isolated judger teardown: %s", exc)
        return self._aggregate(results)

    # -- internals --

    async def _run_one(
        self,
        j: Judger,
        infer_client: Any,
        ctx: dict[str, Any],
        provider: Any,
        infer_workspace: str,
        owned: dict[str, tuple[Any, str]],
    ) -> JudgerResult:
        try:
            client, j_workspace = await self._acquire_client(
                j,
                infer_client,
                ctx,
                provider,
                infer_workspace,
                owned,
            )
            j_ctx = {
                **ctx,
                "workspace": j_workspace,
                "judger_name": j.name,
            }
            await j.stage.run(client, j_ctx)
            return j_ctx.get("judger_result") or JudgerResult(
                judger_name=j.name,
                total=0.0,
                error="no judger_result produced",
            )
        except Exception as exc:
            return JudgerResult(
                judger_name=j.name,
                total=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _acquire_client(
        self,
        j: Judger,
        infer_client: Any,
        ctx: dict[str, Any],
        provider: Any,
        infer_workspace: str,
        owned: dict[str, tuple[Any, str]],
    ) -> tuple[Any, str]:
        if j.sandbox == "shared":
            return infer_client, infer_workspace

        assert isinstance(j.sandbox, SandboxSpec)
        client, env_id = await asyncio.to_thread(
            provider.create,
            image_tag=j.sandbox.image,
            ttl_seconds=j.sandbox.ttl_seconds,
        )
        owned[j.name] = (client, env_id)

        # Seed the isolated sandbox with the agent's workspace.
        ws = j.sandbox.workspace_path
        await exec_in(client, f"mkdir -p {ws}")
        try:
            blob = await asyncio.to_thread(infer_client.download_file, infer_workspace)
            await http_upload(
                client,
                f"/tmp/_ws_{j.name}.tar.gz",
                base64.b64encode(blob).decode(),
            )
            await exec_in(
                client,
                f"cd {ws} && tar xzf /tmp/_ws_{j.name}.tar.gz && rm /tmp/_ws_{j.name}.tar.gz",
                raise_on_error=False,
            )
        except Exception as exc:
            logger.warning("isolated workspace copy for %s failed: %s", j.name, exc)

        for hook in j.on_isolated_pre:
            await hook(client, ctx)

        return client, ws

    def _aggregate(self, results: list[JudgerResult]) -> AggregatedScore:
        weights = {j.name: j.weight for j in self.judgers}
        errors = [r for r in results if r.error]
        usable = [r for r in results if not r.error]

        if errors and self.on_error == "fail":
            total, failed = 0.0, True
        elif not usable:
            total, failed = 0.0, True
        elif self.aggregator == "weighted_sum":
            tw = sum(weights.get(r.judger_name, 1.0) for r in usable)
            total = sum(r.total * weights.get(r.judger_name, 1.0) for r in usable) / tw if tw else 0.0
            failed = False
        elif self.aggregator == "mean":
            total, failed = sum(r.total for r in usable) / len(usable), False
        elif self.aggregator == "max":
            total, failed = max(r.total for r in usable), False
        elif self.aggregator == "min":
            total, failed = min(r.total for r in usable), False
        elif self.aggregator == "all_or_nothing":
            total = 1.0 if all(r.total >= 1.0 for r in usable) else 0.0
            failed = False
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        return AggregatedScore(
            total=total,
            per_judger=results,
            step_rewards=[sr for r in results for sr in r.step_rewards],
            failed=failed,
        )

"""Runner — top-level orchestrator for one task rollout.

Tiny: owns the infer sandbox lifecycle, threads a context dict through
the two stages (infer → validate), and assembles the result envelope.
All real work is done by :class:`SandboxStage` hooks (pre/entry/post)
and :class:`JudgerValidator`.

Task format:

    # task.py
    from common.pipelines.claw_bench import claw_pipeline
    from judgers import pytest_ctrf_judger

    data = {...}
    runner = claw_pipeline(judgers=[pytest_ctrf_judger(name="rule_grader", target="verifier/test_output.py")])

``runner`` is a :class:`Runner` instance (no type-dispatched config — the
pipeline factory composes it in pure Python).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lagent.serving.sandbox.providers.gateway import GatewayProvider  # noqa: E402

from sandbox import SandboxStage                      # noqa: E402
from schemas import TaskData                          # noqa: E402
from validator import JudgerValidator                 # noqa: E402


logger = logging.getLogger(__name__)


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
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
    ) -> dict[str, Any]:
        """Run one rollout end-to-end.  Returns an RLDataFlowItem-shaped dict."""
        ctx: dict[str, Any] = {
            "task_root": task_root,
            "data": data,
            "uid": uid,
            "runtime": {
                "lagent_src_dir": lagent_src_dir,
                "llm_base_url": llm_base_url,
                "llm_api_key": llm_api_key,
            },
            "workspace": self.infer.sandbox.workspace_path,
        }

        client = None
        env_id: str | None = None
        try:
            client, env_id = await asyncio.to_thread(
                provider.create,
                image_tag=self.infer.sandbox.image,
                ttl_seconds=self.infer.sandbox.ttl_seconds,
            )

            infer_result = await self.infer.run(client, ctx)
            if not infer_result.ok:
                await _dump_daemon_log(client)
                return _mark_failed(
                    data, uid, f"infer failed: {infer_result.error}",
                    metadata=_infer_metadata(ctx),
                )

            artifacts = {
                "missing_required": ctx.get("missing_required", []),
                "present": ctx.get("present_outputs", []),
            }

            aggregated = await self.validate.run(
                client, ctx, provider, self.infer.sandbox.workspace_path,
            )

            return _mark_completed(
                data, uid,
                metadata=_infer_metadata(ctx),
                artifact_check=artifacts,
                judge=aggregated,
            )
        except Exception as exc:
            logger.error(
                "runner failed for %s: %s\n%s",
                data.id, exc, traceback.format_exc(),
            )
            return _mark_failed(
                data, uid, f"{type(exc).__name__}: {exc}",
                metadata=_infer_metadata(ctx),
            )
        finally:
            if env_id:
                try:
                    await asyncio.to_thread(provider.delete, env_id)
                except Exception as exc:
                    logger.warning("teardown failed: %s", exc)


# ─────────────────────────────────────────────────────────────────
# Result envelope
# ─────────────────────────────────────────────────────────────────


def _infer_metadata(ctx: dict[str, Any]) -> dict[str, Any]:
    md: dict[str, Any] = {}
    chosen = ctx.get("chosen_agent")
    if chosen is not None:
        md["agent_name"] = chosen.name
    return md


async def _dump_daemon_log(client) -> None:
    try:
        data = await asyncio.to_thread(client.download_file, "/tmp/agent_daemon.log")
        logger.error(
            "agent daemon log tail:\n%s",
            data.decode(errors="replace")[-4000:],
        )
    except Exception as exc:
        logger.warning("could not download daemon log: %s", exc)


def _mark_completed(
    data: TaskData,
    uid: dict[str, int],
    *,
    metadata: dict[str, Any],
    artifact_check: dict[str, Any],
    judge,
) -> dict[str, Any]:
    missing = artifact_check.get("missing_required") or []
    state = "failed" if missing else "completed"
    reason = "missing_artifacts" if missing else "stop"
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
                "state": state,
                "finish_reason": reason,
                "extra_info": {
                    **metadata,
                    "artifact_check": artifact_check,
                },
            },
            "judger": {
                "total": judge.total,
                "per_judger": [r.model_dump() for r in judge.per_judger],
                "step_rewards": [sr.model_dump() for sr in judge.step_rewards],
                "failed": judge.failed,
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
        },
    }


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────


DEFAULT_GATEWAY = "http://env-gateway.ailab.ailab.ai"
DEFAULT_LAGENT_SRC = "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/lagent"


def _load_dataset(name: str, tasks_root: str) -> Any:
    """Import ``datasets.<name>`` and instantiate its main class.

    Convention: each dataset module has a single class whose ``name``
    attribute matches the module name (e.g., ``ClawBench.name = "claw-bench"``).
    """
    import importlib
    mod = importlib.import_module(f"datasets.{name}")
    # Find the class whose ``.name`` matches the module name (hyphenated).
    hyphen = name.replace("_", "-")
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and getattr(obj, "name", None) in (name, hyphen):
            return obj(tasks_root=tasks_root)
    raise ValueError(f"no dataset class with name={name!r} in datasets.{name}")


async def _run_one(
    dataset: Any,
    task_dir: Path,
    provider: Any,
    *,
    lagent_src_dir: str | None,
    llm_base_url: str | None,
    llm_api_key: str | None,
) -> dict[str, Any]:
    data = dataset.load_task(task_dir)
    runner: Runner = dataset.pipeline
    uid = {"root_id": 0, "action_id": 0, "observation_id": 0}
    logger.info("running task=%s (dataset=%s)", data.id, dataset.name)
    return await runner.run_single(
        task_dir, data, uid,
        provider=provider,
        lagent_src_dir=lagent_src_dir,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
    )


async def main_async(args: argparse.Namespace) -> int:
    provider = GatewayProvider(args.gateway)
    dataset = _load_dataset(args.dataset, args.tasks_root)
    lagent_src = args.lagent_src or None

    if args.task_dirs:
        task_dirs = [Path(p) for p in args.task_dirs]
    else:
        task_dirs = [td for td, _ in dataset.iter_tasks()]
        if args.limit:
            task_dirs = task_dirs[: args.limit]

    if not task_dirs:
        logger.error("no tasks to run")
        return 1

    results = await asyncio.gather(*[
        _run_one(
            dataset, td, provider,
            lagent_src_dir=lagent_src,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
        )
        for td in task_dirs
    ])
    for r in results:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    any_failed = any(r["env"]["rollout"]["state"] == "failed" for r in results)
    return 1 if any_failed else 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "task_dirs", nargs="*",
        help="Task root directories (omit to iterate the dataset).",
    )
    parser.add_argument(
        "--dataset", default="claw_bench",
        help="Dataset module name under datasets/ (without .py).",
    )
    parser.add_argument(
        "--tasks-root", required=True,
        help="Root dir containing the dataset's tasks.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max tasks when iterating.")
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    parser.add_argument(
        "--lagent-src", default=DEFAULT_LAGENT_SRC,
        help="Local path to lagent source.  Pass '' to skip upload.",
    )
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    args = parser.parse_args()
    if args.lagent_src == "":
        args.lagent_src = None

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

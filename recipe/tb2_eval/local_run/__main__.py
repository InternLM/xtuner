"""CLI for running rollout samples through this recipe's pipeline.

Usage:
    python -m recipe.tb2_eval.local_run --limit 5 --output results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

from lagent.utils import create_object

from xtuner.v1.rl.agent_loop.rl_task.schemas import AgentRolloutItem
from xtuner.v1.rl.agent_loop.rl_task.trace import init_writer


def _load_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("local_run_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _inject_session_id(runner_cfg: dict[str, Any], session_id: str) -> None:
    for entry in runner_cfg.get("infer", {}).get("entries", []):
        if isinstance(entry, dict) and entry.get("name") == "start_agent_daemon":
            entry.setdefault("env", {})["XTUNER_SESSION_ID"] = session_id


async def _run_one(dataset: Any, item: AgentRolloutItem) -> dict[str, Any]:
    runner_cfg = item.pipeline or dataset.pipeline
    if isinstance(runner_cfg, dict):
        runner_cfg = deepcopy(runner_cfg)
        _inject_session_id(runner_cfg, str(item.uid))
        runner = create_object(runner_cfg)
    else:
        runner = runner_cfg
    result = await runner.run(item)
    dumped = result.model_dump(mode="json", exclude={"artifacts", "pipeline"})
    dumped["artifacts"] = _serialize_artifacts(result.artifacts)
    return dumped


def _serialize_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Keep text artifacts as-is; collapse bytes blobs to a size placeholder."""
    out: dict[str, Any] = {}
    for key, value in artifacts.items():
        if isinstance(value, (bytes, bytearray)):
            out[key] = f"<{len(value)} bytes>"
        else:
            out[key] = value
    return out


async def main_async(args: argparse.Namespace) -> int:
    init_writer()
    cfg = _load_config(Path(args.config))
    dataset = cfg.dataset

    pairs: list[tuple[Path, AgentRolloutItem]]
    if args.tasks:
        wanted = {str(Path(p).resolve()) for p in args.tasks}
        pairs = [(td, item) for td, item in dataset.iter_tasks() if str(td.resolve()) in wanted]
    else:
        pairs = list(dataset.iter_tasks())
        if args.limit:
            pairs = pairs[: args.limit]
    if not pairs:
        print("no tasks to run", file=sys.stderr)
        return 1

    print(f"running {len(pairs)} task(s) (concurrency={args.concurrency})", file=sys.stderr)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def guarded(idx: int, td: Path, item: AgentRolloutItem) -> dict[str, Any]:
        async with sem:
            item = item.model_copy(update={"group_id": 0, "uid": idx})
            try:
                return await _run_one(dataset, item)
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[{item.id}] uncaught: {type(exc).__name__}: {exc}\n{tb}", file=sys.stderr)
                return {"id": item.id, "task_dir": str(td), "error": f"{type(exc).__name__}: {exc}", "traceback": tb}

    out_fp = open(args.output, "w") if args.output else None
    try:
        coros = [guarded(i, td, item) for i, (td, item) in enumerate(pairs)]
        for coro in asyncio.as_completed(coros):
            result = await coro
            line = json.dumps(result, ensure_ascii=False)
            if out_fp is not None:
                out_fp.write(line + "\n")
                out_fp.flush()
            print(json.dumps({k: result.get(k) for k in ("id", "status", "reward", "error")}, ensure_ascii=False))
    finally:
        if out_fp is not None:
            out_fp.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rollout samples through this recipe's pipeline.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.py"))
    parser.add_argument("--tasks", nargs="*", help="Specific task dirs to run; default: all from dataset")
    parser.add_argument("--limit", type=int, default=0, help="Limit total tasks (0=all)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", help="Optional JSONL path to dump full per-sample results")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())

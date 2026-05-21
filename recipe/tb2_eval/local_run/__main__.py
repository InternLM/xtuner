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
from copy import deepcopy
from pathlib import Path
from typing import Any

from lagent.utils import create_object


def _load_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("local_run_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def _run_one(dataset: Any, task_dir: Path, uid: dict[str, int]) -> dict[str, Any]:
    item = dataset.load_task(task_dir)
    runner_cfg = item.pipeline or dataset.pipeline
    runner = create_object(deepcopy(runner_cfg)) if isinstance(runner_cfg, dict) else runner_cfg
    item = item.model_copy(update={"task_root": Path(task_dir), "uid": uid})
    result = await runner.run(item)
    return result.model_dump(mode="json", exclude={"artifacts", "pipeline"})


async def main_async(args: argparse.Namespace) -> int:
    cfg = _load_config(Path(args.config))
    dataset = cfg.dataset

    if args.tasks:
        dirs = [Path(p) for p in args.tasks]
    else:
        dirs = [td for td, _ in dataset.iter_tasks()]
        if args.limit:
            dirs = dirs[: args.limit]
    if not dirs:
        print("no tasks to run", file=sys.stderr)
        return 1

    print(f"running {len(dirs)} task(s) (concurrency={args.concurrency})", file=sys.stderr)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def guarded(idx: int, td: Path) -> dict[str, Any]:
        async with sem:
            uid = {"root_id": 0, "action_id": idx, "observation_id": idx}
            try:
                return await _run_one(dataset, td, uid)
            except Exception as exc:
                return {"task_dir": str(td), "error": f"{type(exc).__name__}: {exc}"}

    out_fp = open(args.output, "w") if args.output else None
    try:
        coros = [guarded(i, td) for i, td in enumerate(dirs)]
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

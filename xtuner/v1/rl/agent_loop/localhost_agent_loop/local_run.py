from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

from lagent.utils import create_object, ctx_session_id

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.agent_loop.localhost_agent_loop import AgentInLocalhostLoop, AgentRolloutItem


def _load_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("local_run_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def _run_runner(dataset: Any, item: AgentRolloutItem) -> dict[str, Any]:
    runner_cfg = item.pipeline or dataset.pipeline
    runner = create_object(deepcopy(runner_cfg)) if isinstance(runner_cfg, dict) else runner_cfg
    with ctx_session_id.set(str(item.uid)):
        result = await runner.run(item)
    dumped = result.model_dump(mode="json", exclude={"artifacts", "pipeline"})
    dumped["artifacts"] = _serialize_artifacts(result.artifacts)
    return dumped


async def _run_agentloop(dataset: Any, item: AgentRolloutItem, agent_loop: AgentInLocalhostLoop) -> dict[str, Any]:
    item = item.model_copy(update={"pipeline": item.pipeline or dataset.pipeline}, deep=True)
    content = _instruction_text(item)
    prompt_ids = agent_loop.tokenizer.encode(content, add_special_tokens=False)

    rollout_state = RolloutState(
        message=[{"role": "user", "content": content}],
        prompt_ids=prompt_ids,
        num_tokens=len(prompt_ids),
        data_source={item.data_source: 1.0},
        reward_model=item.reward_model or {"style": item.data_source},
        uid=item.uid,
        message_uid=item.group_id,
        extra_fields={"rollout_item": item},
    )
    result = await agent_loop.generate_sample(rollout_state)
    return {
        "id": item.id,
        "status": result.status.value,
        "reward": result.reward["score"] if result.reward and "score" in result.reward else None,
        "error": result.error_msg,
        "finish_reason": result.finish_reason,
        "response": result.response,
        "response_ids_len": len(result.response_ids or []),
        "prompt_ids_len": len(result.prompt_ids or []),
        "input_ids_len": len(result.input_ids or []),
        "agent_artifacts": _serialize_artifacts(result.extra_fields.get("agent_artifacts", {})),
    }


def _instruction_text(item: AgentRolloutItem) -> str:
    if item.task_root is not None:
        path = Path(item.task_root) / item.instruction
        if path.exists():
            return path.read_text(encoding="utf-8")
    return item.instruction


def _serialize_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in artifacts.items():
        if isinstance(value, (bytes, bytearray)):
            out[key] = f"<{len(value)} bytes>"
        else:
            out[key] = value
    return out


async def main_async(args: argparse.Namespace) -> int:
    cfg = _load_config(Path(args.config))
    dataset = cfg.build_dataset(args.jsonl_path)

    if args.mode == "agentloop":
        import ray

        ray.init(address=args.ray_address)

    if args.tasks:
        wanted = set(args.tasks)
        pairs = [(tid, item) for tid, item in dataset.iter_tasks() if tid in wanted]
    else:
        pairs = list(dataset.iter_tasks())
        if args.limit:
            pairs = pairs[: args.limit]
    if not pairs:
        print("no tasks to run", file=sys.stderr)
        return 1

    print(
        f"running {len(pairs)} task(s) mode={args.mode} concurrency={args.concurrency}",
        file=sys.stderr,
    )
    sem = asyncio.Semaphore(max(1, args.concurrency))
    agent_loop = None
    if args.mode == "agentloop":
        if not args.hf_checkpoint:
            raise ValueError("--hf-checkpoint is required in agentloop mode.")
        agent_loop = AgentInLocalhostLoop(hf_checkpoint=args.hf_checkpoint, mode=args.agentloop_mode)

    async def guarded(idx: int, tid: str, item: AgentRolloutItem) -> dict[str, Any]:
        async with sem:
            item = item.model_copy(update={"group_id": 0, "uid": idx})
            try:
                if args.mode == "agentloop":
                    assert agent_loop is not None
                    return await _run_agentloop(dataset, item, agent_loop)
                return await _run_runner(dataset, item)
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[{item.id}] uncaught: {type(exc).__name__}: {exc}\n{tb}", file=sys.stderr)
                return {"id": item.id, "error": f"{type(exc).__name__}: {exc}", "traceback": tb}

    out_fp = open(args.output, "w", encoding="utf-8") if args.output else None
    try:
        coros = [guarded(i, tid, item) for i, (tid, item) in enumerate(pairs)]
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


def main(default_config: Path, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run localhost_agent_loop tasks for this recipe.")
    parser.add_argument("--jsonl-path", required=True, help="Path to JSONL data file.")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--tasks", nargs="*", help="Specific task ids to run; default: all from dataset")
    parser.add_argument("--limit", type=int, default=0, help="Limit total tasks (0=all)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", help="Optional JSONL path to dump full per-sample results")
    parser.add_argument("--mode", choices=("runner", "agentloop"), default="runner")
    parser.add_argument("--agentloop-mode", choices=("train", "eval"), default="eval")
    parser.add_argument(
        "--hf-checkpoint",
        default=os.environ.get("HF_CHECKPOINT") or os.environ.get("QWEN3P5_VL_MODEL_PATH"),
        help="Tokenizer/processor checkpoint used by agentloop mode.",
    )
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", "auto"),
        help="Ray address used by agentloop mode.",
    )
    return asyncio.run(main_async(parser.parse_args(argv)))


__all__ = ["main", "main_async"]

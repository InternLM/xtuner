"""DCP round trip and HF export checks for the decoupled EP/FSDP layout (DESIGN.md §6 L3).

For every mode:

1. ``from_hf`` the tiny checkpoint and immediately ``save_hf`` → must be bit-identical to the source;
2. train 5 steps, ``save_dcp``, train 5 more steps, ``save_hf`` (the "continuous" run);
3. build a fresh engine, ``load_dcp`` the step-5 checkpoint, train steps 5-9 → losses must continue
   the continuous curve, and its ``save_hf`` must match the continuous export;
4. the step-10 HF exports of all modes are compared against the ``ep=1`` baseline export.

Optionally the step-5 DCP checkpoint of one layout is loaded into another layout
(``--cross-load SRC:DST``) to check DCP resharding across the switch.

    torchrun --nproc-per-node 8 tests/model/run_decoupled_ep_fsdp_ckpt.py \\
        --modes A:ep=1 C:ep=8,decouple=1 C4:ep=4,decouple=1 H41:ep=4,decouple=1,hsdp=4 --out /tmp/l3
"""

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file


sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_decoupled_ep_fsdp_numerics import build_hf_checkpoint, make_batch, parse_mode  # noqa: E402

from xtuner.v1.config import AdamWConfig, FSDPConfig  # noqa: E402
from xtuner.v1.engine.train_engine import TrainEngine  # noqa: E402
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig  # noqa: E402


def build_engine(mode: dict[str, Any], hf_path: Path, tag: str) -> TrainEngine:
    model_cfg = Qwen3MoEConfig.from_hf(hf_path)
    model_cfg.ep_size = mode["ep"]
    model_cfg.dispatcher = "all2all"
    model_cfg.compile_cfg = False
    model_cfg.mesh_prefix = f"ckpt_{mode['name']}_{tag}"
    fsdp_cfg = FSDPConfig(
        ep_size=mode["ep"],
        decouple_ep_fsdp=mode["decouple"],
        hsdp_sharding_size=mode["hsdp"],
        torch_compile=False,
    )
    return TrainEngine(model_cfg=model_cfg, optim_cfg=AdamWConfig(lr=1e-4), fsdp_cfg=fsdp_cfg)


def train(engine: TrainEngine, steps: range, vocab_size: int, seq_len: int) -> list[float]:
    losses = []
    for step in steps:
        info = engine.train_step([make_batch(step, dist.get_rank(), vocab_size, seq_len)])
        grad_norm = engine.clip_grad_norm()
        engine.step_optimizer(grad_norm)
        losses.append(float(info["logs_info"]["reduced_llm_loss"]))
    return losses


def load_hf_dir(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for file in sorted(path.glob("*.safetensors")):
        tensors.update(load_file(str(file)))
    return tensors


def compare_hf(lhs: Path, rhs: Path) -> dict[str, Any]:
    a = load_hf_dir(lhs)
    b = load_hf_dir(rhs)
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    max_abs = 0.0
    max_rel = 0.0
    mismatched = 0
    total = 0
    worst = ""
    for key in sorted(set(a) & set(b)):
        x = a[key].to(torch.float32)
        y = b[key].to(torch.float32)
        if x.shape != y.shape:
            return {"error": f"shape mismatch for {key}: {tuple(x.shape)} vs {tuple(y.shape)}"}
        diff = (x - y).abs()
        cur = float(diff.max())
        if cur > max_abs:
            max_abs, worst = cur, key
        max_rel = max(max_rel, float((diff / x.abs().clamp_min(1e-6)).max()))
        mismatched += int((diff > 0).sum())
        total += diff.numel()
    return {
        "keys": len(set(a) & set(b)),
        "missing_in_rhs": missing[:5],
        "extra_in_rhs": extra[:5],
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mismatched_elements": mismatched,
        "total_elements": total,
        "worst_key": worst,
    }


def release(engine: TrainEngine) -> None:
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--cross-load", nargs="*", default=[], help="SRC:DST mode-name pairs")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--hf-dir", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    out = Path(args.out)
    hf_dir = Path(args.hf_dir) if args.hf_dir else out / "tiny_qwen3_moe_hf"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if not (hf_dir / "config.json").exists():
            build_hf_checkpoint(hf_dir, 0, "tiny")
    dist.barrier()
    vocab_size = Qwen3MoEConfig.from_hf(hf_dir).vocab_size

    # Source checkpoint in bf16, the dtype `save_hf` writes.
    src_bf16 = out / "source_bf16"
    if rank == 0:
        src_bf16.mkdir(exist_ok=True)
        from safetensors.torch import save_file

        tensors = {k: v.to(torch.bfloat16).contiguous() for k, v in load_hf_dir(hf_dir).items()}
        save_file(tensors, str(src_bf16 / "model.safetensors"))
    dist.barrier()

    report: dict[str, Any] = {"modes": {}, "cross_hf_vs_baseline": {}, "cross_load": {}}
    modes = [parse_mode(spec) for spec in args.modes]
    for mode in modes:
        name = mode["name"]
        mode_dir = out / name
        if rank == 0:
            shutil.rmtree(mode_dir, ignore_errors=True)
            mode_dir.mkdir(parents=True)
        dist.barrier()
        if rank == 0:
            print(f"===== {mode}", flush=True)

        engine = build_engine(mode, hf_dir, "main")
        engine.from_hf(hf_path=hf_dir, strict=True)
        engine.save_hf(str(mode_dir / "hf_step0"))
        dist.barrier()
        entry: dict[str, Any] = {"mode": mode}
        if rank == 0:
            entry["hf_step0_vs_source"] = compare_hf(src_bf16, mode_dir / "hf_step0")

        losses_first = train(engine, range(0, 5), vocab_size, args.seq_len)
        engine.save_dcp(mode_dir / "dcp_step5")
        dist.barrier()
        losses_cont = train(engine, range(5, 10), vocab_size, args.seq_len)
        engine.save_hf(str(mode_dir / "hf_step10"))
        dist.barrier()
        release(engine)

        resumed = build_engine(mode, hf_dir, "resume")
        resumed.load_dcp(mode_dir / "dcp_step5")
        losses_resumed = train(resumed, range(5, 10), vocab_size, args.seq_len)
        resumed.save_hf(str(mode_dir / "hf_step10_resumed"))
        dist.barrier()
        release(resumed)

        entry["losses_steps_0_4"] = losses_first
        entry["losses_steps_5_9_continuous"] = losses_cont
        entry["losses_steps_5_9_resumed"] = losses_resumed
        entry["resume_max_rel_loss_diff"] = max(
            abs(a - b) / abs(b) for a, b in zip(losses_resumed, losses_cont, strict=True)
        )
        if rank == 0:
            entry["hf_step10_resumed_vs_continuous"] = compare_hf(
                mode_dir / "hf_step10", mode_dir / "hf_step10_resumed"
            )
            print(json.dumps({k: v for k, v in entry.items() if k != "mode"}, indent=1), flush=True)
        report["modes"][name] = entry

    if rank == 0:
        baseline = modes[0]["name"]
        for mode in modes[1:]:
            report["cross_hf_vs_baseline"][mode["name"]] = compare_hf(
                out / baseline / "hf_step10", out / mode["name"] / "hf_step10"
            )

    for pair in args.cross_load:
        src, dst = pair.split(":")
        dst_mode = next(m for m in modes if m["name"] == dst)
        if rank == 0:
            print(f"===== cross load {src} -> {dst}", flush=True)
        engine = build_engine(dst_mode, hf_dir, f"cross_{src}")
        engine.load_dcp(out / src / "dcp_step5")
        losses = train(engine, range(5, 10), vocab_size, args.seq_len)
        release(engine)
        ref = report["modes"][src]["losses_steps_5_9_continuous"]
        report["cross_load"][pair] = {
            "losses_steps_5_9": losses,
            "max_rel_loss_diff_vs_src_continuous": max(abs(a - b) / abs(b) for a, b in zip(losses, ref, strict=True)),
        }

    if rank == 0:
        (out / "l3.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k] for k in ("cross_hf_vs_baseline", "cross_load")}, indent=1), flush=True)
        print(f"wrote {out / 'l3.json'}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

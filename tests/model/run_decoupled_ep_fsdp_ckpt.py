"""DCP round trip and HF export checks for the decoupled EP/FSDP layout (DESIGN.md §6 L3).

For every mode:

1. ``from_hf`` the tiny checkpoint and immediately ``save_hf`` → must be bit-identical to the source;
2. train 5 steps, ``save_dcp``, train 5 more steps, ``save_hf`` (the "continuous" run);
3. build a fresh engine, ``load_dcp`` the step-5 checkpoint, train steps 5-9 → losses must continue
   the continuous curve, and its ``save_hf`` must match the continuous export;
4. the step-10 HF exports of all modes are compared against the ``ep=1`` baseline export.

Optionally the step-5 DCP checkpoint of one layout is loaded into another layout
(``--cross-load SRC:DST``) to check DCP resharding across the switch.

The pytest gate for these checks lives in ``tests/engine/test_decoupled_ep_fsdp_train_engine.py``;
this script writes the JSON behind the markdown report.

    torchrun --nproc-per-node 8 tests/model/run_decoupled_ep_fsdp_ckpt.py \\
        --modes A:ep=1 C:ep=8,decouple=1 C4:ep=4,decouple=1 H41:ep=4,decouple=1,hsdp=4 --out /tmp/l3
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from xtuner._testing.decoupled_ep_fsdp import (
    build_engine,
    build_hf_checkpoint,
    compare_hf,
    load_hf_dir,
    max_rel_diff,
    parse_mode,
    release,
    train,
)
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig


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

        engine = build_engine(mode, hf_dir, "ckpt_main")
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

        resumed = build_engine(mode, hf_dir, "ckpt_resume")
        resumed.load_dcp(mode_dir / "dcp_step5")
        losses_resumed = train(resumed, range(5, 10), vocab_size, args.seq_len)
        resumed.save_hf(str(mode_dir / "hf_step10_resumed"))
        dist.barrier()
        release(resumed)

        entry["losses_steps_0_4"] = losses_first
        entry["losses_steps_5_9_continuous"] = losses_cont
        entry["losses_steps_5_9_resumed"] = losses_resumed
        entry["resume_max_rel_loss_diff"] = max_rel_diff(losses_resumed, losses_cont)
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
        engine = build_engine(dst_mode, hf_dir, f"ckpt_cross_{src}")
        engine.load_dcp(out / src / "dcp_step5")
        losses = train(engine, range(5, 10), vocab_size, args.seq_len)
        release(engine)
        ref = report["modes"][src]["losses_steps_5_9_continuous"]
        report["cross_load"][pair] = {
            "losses_steps_5_9": losses,
            "max_rel_loss_diff_vs_src_continuous": max_rel_diff(losses, ref),
        }

    if rank == 0:
        (out / "l3.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k] for k in ("cross_hf_vs_baseline", "cross_load")}, indent=1), flush=True)
        print(f"wrote {out / 'l3.json'}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Multi-GPU numerical-equivalence runs for the EP/FSDP decoupling (DESIGN.md §6 L1 / L2).

Every mode trains the same tiny Qwen3-MoE (random HF checkpoint built once from a fixed seed)
on the same token stream, so loss / grad-norm curves of different layouts can be compared. The
pytest gate for the same comparison lives in ``tests/engine/test_decoupled_ep_fsdp_train_engine.py``;
this script exists for the larger ``--model-size medium`` memory / step-time runs, ``--fp8`` and
``--dispatcher deepep``, and for producing the JSON behind the markdown reports
(``summarize_decoupled_ep_fsdp_numerics.py``).

Example (L1):

    torchrun --nproc-per-node 8 tests/model/run_decoupled_ep_fsdp_numerics.py \\
        --modes A:ep=1 B:ep=8 C:ep=8,decouple=1 --steps 50 --out reports/l1.json

Mode spec: ``<name>:key=value[,key=value...]`` with keys ``ep`` (int), ``decouple`` (0/1) and
``hsdp`` (hsdp_sharding_size, int).
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from xtuner._testing.decoupled_ep_fsdp import MODEL_SIZES, build_hf_checkpoint, parse_mode, run_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-size", choices=tuple(MODEL_SIZES), default="tiny")
    parser.add_argument("--dispatcher", choices=("all2all", "deepep"), default="all2all")
    parser.add_argument("--fp8", action="store_true", help="tile-wise float8 for linear and grouped linear")
    parser.add_argument("--hf-dir", type=str, default=None, help="where to build / reuse the tiny HF checkpoint")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    hf_dir = Path(args.hf_dir or Path(args.out).with_suffix("")).with_name(f"{args.model_size}_qwen3_moe_hf")
    if rank == 0 and not (hf_dir / "config.json").exists():
        hf_dir.mkdir(parents=True, exist_ok=True)
        build_hf_checkpoint(hf_dir, args.seed, args.model_size)
    dist.barrier()

    results = []
    for spec in args.modes:
        mode = parse_mode(spec)
        if rank == 0:
            print(f"===== running mode {mode}", flush=True)
        results.append(run_mode(mode, hf_dir, args.steps, args.seq_len, args.lr, args.dispatcher, args.fp8))
        dist.barrier()
        if rank == 0:
            last = results[-1]
            print(
                f"[{mode['name']}] loss[0]={last['losses'][0]:.6f} loss[-1]={last['losses'][-1]:.6f} "
                f"grad_norm[0]={last['grad_norms'][0]:.6f} step_time={last['step_time_s'] * 1000:.1f}ms "
                f"mem={last['memory']}",
                flush=True,
            )

    if rank == 0:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"args": vars(args), "world_size": dist.get_world_size(), "results": results}, indent=2)
        )
        print(f"wrote {out}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

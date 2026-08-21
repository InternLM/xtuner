"""Multi-GPU numerical-equivalence runs for the EP/FSDP decoupling (DESIGN.md §6 L1 / L2).

Every mode trains the same tiny Qwen3-MoE (random HF checkpoint built once from a fixed seed)
on the same token stream, so loss / grad-norm curves of different layouts can be compared.

Example (L1):

    torchrun --nproc-per-node 8 tests/model/run_decoupled_ep_fsdp_numerics.py \\
        --modes A:ep=1 B:ep=8 C:ep=8,decouple=1 --steps 50 --out reports/l1.json

Mode spec: ``<name>:key=value[,key=value...]`` with keys ``ep`` (int), ``decouple`` (0/1) and
``hsdp`` (hsdp_sharding_size, int).
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig
from xtuner.v1.utils.device import get_device
from xtuner.v1.utils.dtensor import cal_total_norm


DEVICE = get_device()
GRAD_NORM_STEPS = (0, 1, 25)


def parse_mode(spec: str) -> dict[str, Any]:
    name, _, kv = spec.partition(":")
    mode: dict[str, Any] = {"name": name, "ep": 1, "decouple": False, "hsdp": None}
    for item in filter(None, kv.split(",")):
        key, value = item.split("=")
        if key == "ep":
            mode["ep"] = int(value)
        elif key == "decouple":
            mode["decouple"] = bool(int(value))
        elif key == "hsdp":
            mode["hsdp"] = int(value)
        else:
            raise ValueError(f"unknown mode key {key}")
    return mode


MODEL_SIZES: dict[str, dict[str, int]] = {
    # ~108M params: numerics (fast, 50 steps in < 10 s)
    "tiny": {
        "vocab_size": 4096,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "moe_intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "num_experts": 64,
        "num_experts_per_tok": 4,
    },
    # ~3.4B params (3.2B in routed experts): step-time / memory comparison
    "medium": {
        "vocab_size": 32768,
        "hidden_size": 2048,
        "intermediate_size": 4096,
        "moe_intermediate_size": 1024,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 64,
        "num_experts_per_tok": 4,
    },
}


def build_hf_checkpoint(path: Path, seed: int, model_size: str) -> None:
    config = Qwen3MoeConfig(
        **MODEL_SIZES[model_size],
        max_position_embeddings=8192,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=1000000.0,
        use_sliding_window=False,
        max_window_layers=MODEL_SIZES[model_size]["num_hidden_layers"],
    )
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(config)
    model.save_pretrained(path, safe_serialization=True)


def make_batch(step: int, rank: int, vocab_size: int, seq_len: int) -> ModelItem:
    generator = torch.Generator().manual_seed(100_000 + step * 1024 + rank)
    tokens = torch.randint(0, vocab_size, (1, seq_len + 1), generator=generator)
    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:].to(DEVICE)
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
    loss_cfg = CELossConfig()
    loss_ctx = loss_cfg.build(data={"shifted_labels": labels}, sp_mesh=None)
    loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
    return ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})


def clean_name(name: str) -> str:
    return name.replace("_checkpoint_wrapped_module.", "")


def param_memory(model: torch.nn.Module) -> dict[str, float]:
    expert_bytes = 0
    dense_bytes = 0
    for name, param in model.named_parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        nbytes = local.numel() * local.element_size()
        if ".experts" in name:
            expert_bytes += nbytes
        else:
            dense_bytes += nbytes
    return {"expert_param_mib": expert_bytes / 2**20, "dense_param_mib": dense_bytes / 2**20}


def per_param_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if isinstance(param.grad, DTensor):
            norm = cal_total_norm([param.grad], norm_type=2.0, foreach=True, dtype=torch.float32)
        else:
            norm = torch.linalg.vector_norm(param.grad, 2.0, dtype=torch.float32)
        norms[clean_name(name)] = float(norm)
    return norms


def run_mode(
    mode: dict[str, Any], hf_path: Path, steps: int, seq_len: int, lr: float, dispatcher: str
) -> dict[str, Any]:
    rank = dist.get_rank()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    base_allocated = torch.cuda.memory_allocated()

    model_cfg = Qwen3MoEConfig.from_hf(hf_path)
    model_cfg.ep_size = mode["ep"]
    model_cfg.dispatcher = dispatcher
    model_cfg.compile_cfg = False
    model_cfg.mesh_prefix = f"numerics_{mode['name']}"
    fsdp_cfg = FSDPConfig(
        ep_size=mode["ep"],
        decouple_ep_fsdp=mode["decouple"],
        hsdp_sharding_size=mode["hsdp"],
        torch_compile=False,
    )
    optim_cfg = AdamWConfig(lr=lr)

    engine = TrainEngine(model_cfg=model_cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg)
    engine.from_hf(hf_path=hf_path, strict=True)
    torch.cuda.synchronize()
    result: dict[str, Any] = {
        "mode": mode,
        "memory": param_memory(engine.model),
        "losses": [],
        "grad_norms": [],
        "param_grad_norms": {},
        "step_time_s": None,
    }
    result["memory"]["allocated_after_load_mib"] = (torch.cuda.memory_allocated() - base_allocated) / 2**20

    step_times = []
    for step in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = make_batch(step, rank, model_cfg.vocab_size, seq_len)
        info = engine.train_step([batch])
        grad_norm = engine.clip_grad_norm()
        if step in GRAD_NORM_STEPS:
            result["param_grad_norms"][str(step)] = per_param_grad_norms(engine.model)
        engine.step_optimizer(grad_norm)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        result["losses"].append(float(info["logs_info"]["reduced_llm_loss"]))
        result["grad_norms"].append(float(grad_norm))
        if step == 0:
            result["memory"]["allocated_after_step0_mib"] = (torch.cuda.memory_allocated() - base_allocated) / 2**20

    warm = step_times[5:] if len(step_times) > 10 else step_times
    result["step_time_s"] = sum(warm) / len(warm)
    result["memory"]["peak_allocated_mib"] = (torch.cuda.max_memory_allocated() - base_allocated) / 2**20
    result["memory"]["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20

    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-size", choices=tuple(MODEL_SIZES), default="tiny")
    parser.add_argument("--dispatcher", choices=("all2all", "deepep"), default="all2all")
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
        results.append(run_mode(mode, hf_dir, args.steps, args.seq_len, args.lr, args.dispatcher))
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

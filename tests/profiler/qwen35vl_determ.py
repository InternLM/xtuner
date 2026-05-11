"""Qwen3.5-VL 在不同 TRITON_CACHE_DIR 下的最小确定性复现脚本。

典型用法由 qwen35vl_determ.sh 负责：

    bash tests/profiler/qwen35vl_determ.sh 8
    bash tests/profiler/qwen35vl_determ.sh 4

Python 脚本本身只负责一次 torchrun 进程内的记录/比较：

    XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 tests/profiler/qwen35vl_determ.py \
        --record-path /tmp/qwen35vl/run1 --num-layers 8 --deterministic

    XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 tests/profiler/qwen35vl_determ.py \
        --record-path /tmp/qwen35vl/run2 --num-layers 8 --deterministic \
        --compare /tmp/qwen35vl/run1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.config import FSDPConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model.base import ModelItem
from xtuner.v1.profiler.prober import ProberList
from xtuner.v1.profiler.prober_utils import setup_prober_list
from xtuner.v1.utils import XTUNER_DETERMINISTIC
from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache


DEFAULT_MODEL_PATH = (
    "/mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/"
    "models--Qwen--Qwen3.5-35B-A3B/snapshots/"
    "ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
)
DEFAULT_DATA_PATH = "/mnt/shared-storage-user/llmrazor-share/data/ci_vl"


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _print_rank0(msg: str) -> None:
    if _rank() == 0:
        print(msg, flush=True)


def _record_path_for_rank(base_path: str | Path, rank: int) -> Path:
    return Path(f"{base_path}_rank{rank}.json")


def _resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32", "torch.float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _init_distributed(deterministic: bool) -> tuple[int, int, int]:
    # NCCL 环境变量必须在 init_process_group 前设置。
    if deterministic:
        os.environ.setdefault("NCCL_ALGO", "Ring")
        os.environ.setdefault("NCCL_PROTO", "Simple")
        os.environ.setdefault("NCCL_NUM_CHANNELS", "1")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    torch.accelerator.set_device_index(local_rank)

    # 和 Trainer 保持一致：初始化后先做一次小通信，提前建好通信上下文。
    warmup = torch.ones(4, 4, device="cuda")
    dist.all_reduce(warmup)
    return rank, world_size, local_rank


def _build_first_batch(
    *,
    model_path: str,
    data_path: str,
    media_root: str,
    max_len: int,
    global_batch_size: int,
    seed: int,
    world_size: int,
):
    from transformers import AutoTokenizer

    if global_batch_size % world_size != 0:
        raise ValueError(f"global_batch_size={global_batch_size} must be divisible by world_size={world_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset_config = [
        {
            "dataset": DatasetConfig(
                name="sft",
                anno_path=data_path,
                class_name="VLMJsonlDataset",
                media_root=media_root,
                sample_ratio=1.0,
            ),
            "tokenize_fn": Qwen3VLTokenizeFnConfig(
                processor_path=model_path,
                max_length=max_len,
                add_vision_id=True,
            ),
        }
    ]
    dataloader_config = DataloaderConfig(
        dataset_config_list=dataset_config,
        pack_max_length=max_len,
        collator="qwen3_vl_sft_collator",
    )

    # Trainer 在 cpu mesh 上做 dp sampler；这里照抄，保证每个 rank 取到的 batch 顺序一致。
    data_mesh = init_device_mesh("cpu", (world_size, 1, 1), mesh_dim_names=("dp", "sp", "tp"))
    dataloader = dataloader_config.build(
        tokenizer=tokenizer,
        dp_mesh=data_mesh["dp"],
        global_batch_size=global_batch_size,
        micro_batch_size=global_batch_size // world_size,
        seed=seed,
        total_step=1,
    )
    return next(iter(dataloader))


def _build_model(
    *,
    model_path: str,
    num_layers: int,
    compile_model: bool,
    reduce_dtype: torch.dtype,
):
    model_cfg = Qwen3_5_VLMoE35BA3Config(compile_cfg=None if compile_model else False)
    model_cfg.text_config.num_hidden_layers = num_layers

    fsdp_cfg = FSDPConfig(
        torch_compile=compile_model,
        cpu_offload=False,
        reduce_dtype=reduce_dtype,
    )

    with torch.device("meta"):
        model = model_cfg.build()

    model = model.fully_shard(fsdp_config=fsdp_cfg)
    model.from_hf(model_path, strict=False)
    model.train()
    return model


def _prepare_model_input(data_batch: list[dict[str, Any]], loss_cfg: CELossConfig) -> list[ModelItem]:
    seq_ctx_list = []
    loss_ctx_list = []

    for data in data_batch:
        seq_ctx = data["seq_ctx"].to("cuda")
        seq_ctx_list.append(seq_ctx)
        loss_ctx_list.append(loss_cfg.build(shifted_labels=data["shifted_labels"], sp_mesh=None))

    cu_seq_lens_list = [seq_ctx.cu_seq_lens_q for seq_ctx in seq_ctx_list]
    loss_ctx_list = CELossContext.build_batches(loss_ctx_list, cu_seq_lens_list=cu_seq_lens_list, sp_mesh=None)
    return [ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx) for seq_ctx, loss_ctx in zip(seq_ctx_list, loss_ctx_list)]


def _total_loss(outputs: Any) -> torch.Tensor:
    loss = torch.tensor(0.0, device="cuda")
    for key in type(outputs).model_fields:
        value = getattr(outputs, key)
        if "loss" in key and isinstance(value, torch.Tensor):
            loss = loss + value
    return loss


def _run_one_step(model: torch.nn.Module, data_batches: list[ModelItem]) -> float:
    model.zero_grad(set_to_none=True)
    total_loss = torch.tensor(0.0, device="cuda")

    for micro_iter, data in enumerate(data_batches):
        ProberList.set_micro_batch_iter(micro_iter)
        outputs = model(seq_ctx=data["seq_ctx"], loss_ctx=data["loss_ctx"])
        if hasattr(outputs, "free_nongrad_feature"):
            outputs.free_nongrad_feature()
        loss = _total_loss(outputs)
        loss.backward()
        total_loss = total_loss + loss.detach()
        ProberList.after_micro_iter_forward()
        _print_rank0(f"[qwen35vl_determ] micro_iter={micro_iter} loss={loss.item():.8f}")

    torch.cuda.synchronize()
    return float(total_loss.item())


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    return tensor.detach().contiguous()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.cpu().view(torch.uint8)
    return hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest()


def collect_grad_records(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = _local_tensor(param.grad)
            grad_float = grad.float()
            records[name] = {
                "shape": list(grad.shape),
                "dtype": str(grad.dtype),
                "numel": grad.numel(),
                "grad_sum": float(grad_float.sum().item()),
                "grad_mean": float(grad_float.mean().item()),
                "grad_std": float(grad_float.std(unbiased=False).item()) if grad.numel() > 1 else 0.0,
                "grad_absmax": float(grad_float.abs().max().item()) if grad.numel() > 0 else 0.0,
                "sha256": _tensor_sha256(grad),
            }
    return records


def save_record(
    records: dict[str, dict[str, Any]],
    *,
    base_path: str,
    rank: int,
    args: argparse.Namespace,
    total_loss: float | None,
) -> None:
    path = _record_path_for_rank(base_path, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": rank,
        "world_size": dist.get_world_size(),
        "total_loss": total_loss,
        "meta": {
            "num_layers": args.num_layers,
            "max_len": args.max_len,
            "global_batch_size": args.global_batch_size,
            "reduce_dtype": args.reduce_dtype,
            "compile": args.compile,
            "deterministic_arg": args.deterministic,
            "XTUNER_DETERMINISTIC": XTUNER_DETERMINISTIC,
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "param_grads": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_record(base_path: str, rank: int) -> dict[str, dict[str, Any]]:
    path = _record_path_for_rank(base_path, rank)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["param_grads"]


def compare_records(
    new_records: dict[str, dict[str, Any]],
    old_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    diffs = []
    for name, new in new_records.items():
        old = old_records.get(name)
        if old is None:
            continue
        if new["sha256"] == old["sha256"]:
            continue

        old_sum = float(old["grad_sum"])
        new_sum = float(new["grad_sum"])
        abs_sum_diff = abs(new_sum - old_sum)
        rel_sum_diff = abs_sum_diff / (abs(old_sum) + 1e-12)
        diffs.append(
            {
                "name": name,
                "old_sum": old_sum,
                "new_sum": new_sum,
                "abs_sum_diff": abs_sum_diff,
                "rel_sum_diff": rel_sum_diff,
                "old_hash": old["sha256"][:12],
                "new_hash": new["sha256"][:12],
            }
        )
    return diffs


def _format_diff_line(diff: dict[str, Any]) -> str:
    return (
        f"{diff['name']}  old={diff['old_sum']:.8e}  new={diff['new_sum']:.8e}  "
        f"abs={diff['abs_sum_diff']:.3e}  rel={diff['rel_sum_diff']:.3e}  "
        f"hash={diff['old_hash']}->{diff['new_hash']}"
    )


def _print_compare_summary(diffs: list[dict[str, Any]], diff_limit: int) -> int:
    local_summary = {
        "rank": dist.get_rank(),
        "diff_count": len(diffs),
        "examples": diffs[:diff_limit],
        "gdn_examples": [
            d
            for d in diffs
            if "self_attn.A_log" in d["name"] or "self_attn.in_proj_a.weight" in d["name"]
        ][:diff_limit],
    }
    all_summaries: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_summaries, local_summary)

    diff_count = torch.tensor([len(diffs)], dtype=torch.int64, device="cuda")
    max_rel = max((d["rel_sum_diff"] for d in diffs), default=0.0)
    max_rel_t = torch.tensor([max_rel], dtype=torch.float64, device="cuda")
    dist.all_reduce(diff_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(max_rel_t, op=dist.ReduceOp.MAX)

    total_diffs = int(diff_count.item())
    if dist.get_rank() == 0:
        for summary in all_summaries:
            assert summary is not None
            rank = summary["rank"]
            if summary["diff_count"] == 0:
                print(f"[Rank {rank}] all gradients are bitwise identical", flush=True)
                continue

            print(f"[Rank {rank}] {summary['diff_count']} param grads differ. Examples:", flush=True)
            for diff in summary["examples"]:
                print("  " + _format_diff_line(diff), flush=True)

            if summary["gdn_examples"]:
                print(f"[Rank {rank}] GDN focus diffs:", flush=True)
                for diff in summary["gdn_examples"]:
                    print("  " + _format_diff_line(diff), flush=True)

        if total_diffs == 0:
            print("RESULT: DETERMINISTIC - all per-rank gradient hashes are identical.", flush=True)
        else:
            print(
                "RESULT: NON-DETERMINISTIC - "
                f"{total_diffs} per-rank parameter gradients differ; "
                f"max grad_sum relative diff is {max_rel_t.item():.3e}.",
                flush=True,
            )
    return 2 if total_diffs else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-VL TRITON_CACHE_DIR determinism repro")
    parser.add_argument("--record-path", required=True, help="Output base path; _rank<N>.json is appended")
    parser.add_argument("--compare", default=None, help="Previous output base path to compare against")
    parser.add_argument("--skip-train", action="store_true", help="Only load --record-path and compare")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--data-path", default=os.environ.get("DATA_PATH", DEFAULT_DATA_PATH))
    parser.add_argument("--media-root", default=os.environ.get("MEDIA_ROOT", DEFAULT_DATA_PATH))
    parser.add_argument("--num-layers", type=int, default=int(os.environ.get("NUM_LAYERS", "8")))
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reduce-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--compile", action="store_true", help="Enable model compile; default matches sft_qwen35_mengke")
    parser.add_argument("--deterministic", action="store_true", help="Set deterministic torch/NCCL knobs")
    parser.add_argument("--no-prober", action="store_true", help="Disable AccProber wrappers/dumps")
    parser.add_argument("--diff-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_train and not args.compare:
        print("ERROR: --skip-train requires --compare", file=sys.stderr)
        sys.exit(2)

    monkey_patch_hf_modules_cache()
    rank, world_size, local_rank = _init_distributed(args.deterministic)
    set_random_seed(args.seed, deterministic=args.deterministic)

    _print_rank0(
        "[qwen35vl_determ] "
        f"world_size={world_size} local_rank0={local_rank} layers={args.num_layers} "
        f"max_len={args.max_len} gbs={args.global_batch_size} compile={args.compile} "
        f"reduce_dtype={args.reduce_dtype} XTUNER_DETERMINISTIC={XTUNER_DETERMINISTIC} "
        f"TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR')}"
    )
    if args.deterministic and not XTUNER_DETERMINISTIC:
        _print_rank0(
            "[qwen35vl_determ] WARNING: --deterministic only set torch/NCCL knobs here; "
            "XTUNER_DETERMINISTIC was not true at import time. Prefer qwen35vl_determ.sh "
            "or export XTUNER_DETERMINISTIC=true before torchrun."
        )

    total_loss = None
    if args.skip_train:
        records = load_record(args.record_path, rank)
    else:
        t0 = time.time()
        _print_rank0("[qwen35vl_determ] building first dataloader batch ...")
        data_batch = _build_first_batch(
            model_path=args.model_path,
            data_path=args.data_path,
            media_root=args.media_root,
            max_len=args.max_len,
            global_batch_size=args.global_batch_size,
            seed=args.seed,
            world_size=world_size,
        )

        set_random_seed(args.seed, deterministic=args.deterministic)
        _print_rank0("[qwen35vl_determ] building FSDP Qwen3.5-VL model ...")
        torch._dynamo.reset()
        model = _build_model(
            model_path=args.model_path,
            num_layers=args.num_layers,
            compile_model=args.compile,
            reduce_dtype=_resolve_dtype(args.reduce_dtype),
        )
        if not args.no_prober:
            setup_prober_list(Path(args.record_path).parent, [1], model, ["AccProber"])

        loss_cfg = CELossConfig(mode="chunk", chunk_size=512)
        model_input = _prepare_model_input(data_batch, loss_cfg)
        _print_rank0(f"[qwen35vl_determ] running {len(model_input)} micro-batches ...")
        ProberList.set_step(1)
        total_loss = _run_one_step(model, model_input)
        if not args.no_prober:
            ProberList.before_clip_grad_norm(model)
        records = collect_grad_records(model)
        dist.barrier()
        save_record(records, base_path=args.record_path, rank=rank, args=args, total_loss=total_loss)
        dist.barrier()
        _print_rank0(
            f"[qwen35vl_determ] saved {len(records)} grad records to {args.record_path}_rank*.json "
            f"in {time.time() - t0:.1f}s"
        )

    exit_code = 0
    if args.compare:
        old_records = load_record(args.compare, rank)
        diffs = compare_records(records, old_records)
        exit_code = _print_compare_summary(diffs, args.diff_limit)

    dist.destroy_process_group()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

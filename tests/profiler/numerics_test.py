"""Text-only Qwen3.5 MoE gradient determinism recorder.

Typical usage:

    XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 tests/profiler/numerics_test.py \
        --record-path /tmp/qwen35_text/run1 --hf-path /path/to/Qwen3.5-35B-A3B --deterministic

    XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 tests/profiler/numerics_test.py \
        --record-path /tmp/qwen35_text/run2 --hf-path /path/to/Qwen3.5-35B-A3B --deterministic \
        --compare /tmp/qwen35_text/run1

Each rank writes ``<record-path>_rank<N>.json``. Comparison is bitwise on local
gradient shards; any hash mismatch returns exit code 2.
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

from xtuner.v1.config import FSDPConfig
from xtuner.v1.loss import CELossContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.profiler.prober import ProberList
from xtuner.v1.profiler.prober_utils import setup_prober_list
from xtuner.v1.utils import IGNORE_INDEX, XTUNER_DETERMINISTIC, set_deterministic
from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache


GRAD_ACCUM_STEPS = 2


def _print_rank0(message: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message, flush=True)


def _record_path_for_rank(base_path: str | Path, rank: int) -> Path:
    return Path(f"{base_path}_rank{rank}.json")


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    return int(torch.randint(low, high, (1,), generator=generator).item())


def _random_partition(generator: torch.Generator, total: int, num_parts: int, min_part: int) -> list[int]:
    assert total >= num_parts * min_part
    parts = [min_part] * num_parts
    for _ in range(total - num_parts * min_part):
        parts[_randint(generator, 0, num_parts)] += 1
    return parts


def _build_simple_batch(
    *,
    generator: torch.Generator,
    vocab_size: int,
    seq_len: int,
) -> tuple[Any, torch.Tensor]:
    from xtuner.v1.data_proto import SequenceContext

    full = torch.randint(0, vocab_size, (1, seq_len + 1), generator=generator, dtype=torch.long)
    shifted_labels = full[:, 1:].clone()
    shift_input_ids = full[:, :-1]
    seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids,), device="cpu")
    return seq_ctx, shifted_labels


def _build_realistic_batch(
    *,
    generator: torch.Generator,
    vocab_size: int,
    seq_len: int,
    padding_token_idx: int,
) -> tuple[Any, torch.Tensor]:
    from xtuner.v1.datasets.collator import build_text_ctx_labels

    min_segment_len = 8
    num_segments = _randint(generator, 3, 9)
    min_total = num_segments * min_segment_len + 2
    if min_total > seq_len:
        num_segments = max(2, seq_len // (2 * min_segment_len))
        min_total = num_segments * min_segment_len + 2
    total_len = _randint(generator, min_total, seq_len + 1)
    segment_lens = _random_partition(generator, total_len, num_segments, min_segment_len)

    instances = []
    for segment_len in segment_lens:
        input_ids = torch.randint(0, vocab_size, (segment_len,), generator=generator).tolist()
        labels = list(input_ids)
        labels[0] = IGNORE_INDEX
        user_len = _randint(generator, 1, max(2, segment_len - 1))
        for idx in range(1, 1 + user_len):
            labels[idx] = IGNORE_INDEX
        instances.append({"input_ids": input_ids, "labels": labels, "num_tokens": segment_len})

    return build_text_ctx_labels(
        instances,
        seq_len,
        padding_token_idx,
        pack_to_max_length=True,
        pad_chunk_size=256,
    )[:2]


def _precompute_micro_batches(
    *,
    rank: int,
    vocab_size: int,
    padding_token_idx: int,
    seq_len: int,
    batch_style: str,
) -> list[tuple[Any, torch.Tensor]]:
    # 先在 CPU 上固定输入，避免输入 CUDA 分配影响后续模型/compile workspace 地址布局。
    micro_batches = []
    for micro_iter in range(GRAD_ACCUM_STEPS):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(rank * GRAD_ACCUM_STEPS + micro_iter)
        if batch_style == "realistic":
            micro_batches.append(
                _build_realistic_batch(
                    generator=generator,
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                    padding_token_idx=padding_token_idx,
                )
            )
        else:
            micro_batches.append(
                _build_simple_batch(
                    generator=generator,
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                )
            )
    return micro_batches


def _build_model(
    *,
    hf_path: str,
    num_layers: int,
    compile_model: bool,
    reduce_dtype: torch.dtype,
):
    model_cfg = Qwen3_5_VLMoE35BA3Config()
    model_cfg.text_config.num_hidden_layers = num_layers
    model_cfg.text_config.ep_size = 1
    model_cfg.only_llm_forward = True
    model_cfg.compile_cfg = True if compile_model else False

    with torch.device("meta"):
        model = model_cfg.build()

    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1, reduce_dtype=reduce_dtype)
    model = model.fully_shard(fsdp_config=fsdp_cfg)
    model.from_hf(hf_path, strict=False)
    model.train()
    return model


def _total_loss(outputs: Any) -> torch.Tensor:
    loss = torch.tensor(0.0, device="cuda")
    for key in type(outputs).model_fields:
        value = getattr(outputs, key)
        if "loss" in key and isinstance(value, torch.Tensor):
            loss = loss + value
    return loss


def _run_forward_backward(model, micro_batches: list[tuple[Any, torch.Tensor]]) -> float:
    loss_cfg = CELossConfig(mode="chunk")
    model.zero_grad(set_to_none=True)
    total_loss = torch.tensor(0.0, device="cuda")

    for micro_iter, (seq_ctx, shifted_labels) in enumerate(micro_batches):
        ProberList.set_micro_batch_iter(micro_iter)
        seq_ctx = seq_ctx.to("cuda")
        loss_ctx = loss_cfg.build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
        assert loss_ctx is not None
        loss_ctx = CELossContext.build_batches([loss_ctx])[0]
        outputs = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
        loss = _total_loss(outputs)
        loss.backward()
        total_loss = total_loss + loss.detach()
        ProberList.after_micro_iter_forward()
        _print_rank0(f"[numerics_test] micro_iter={micro_iter} loss={loss.item():.8f}")

    torch.cuda.synchronize()
    return float(total_loss.item())


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    return tensor.detach().contiguous()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.cpu().view(torch.uint8)
    return hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest()


def collect_grad_records(model) -> dict[str, dict[str, Any]]:
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
                "grad_sum": float(grad_float.sum().item()),
                "grad_mean": float(grad_float.mean().item()),
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
            "seq_len": args.seq_len,
            "batch_style": args.batch_style,
            "compile": not args.no_compile,
            "reduce_dtype": args.reduce_dtype,
            "deterministic_arg": args.deterministic,
            "XTUNER_DETERMINISTIC": XTUNER_DETERMINISTIC,
            "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK": os.environ.get("TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK"),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
        },
        "param_grads": records,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_record(base_path: str, rank: int) -> dict[str, dict[str, Any]]:
    path = _record_path_for_rank(base_path, rank)
    with path.open(encoding="utf-8") as f:
        return json.load(f)["param_grads"]


def compare_records(
    new_records: dict[str, dict[str, Any]],
    old_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    diffs = []
    for name, new in new_records.items():
        old = old_records.get(name)
        if old is None or new["sha256"] == old["sha256"]:
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


def _format_diff(diff: dict[str, Any]) -> str:
    return (
        f"{diff['name']} old={diff['old_sum']:.8e} new={diff['new_sum']:.8e} "
        f"abs={diff['abs_sum_diff']:.3e} rel={diff['rel_sum_diff']:.3e} "
        f"hash={diff['old_hash']}->{diff['new_hash']}"
    )


def _print_compare_summary(diffs: list[dict[str, Any]], diff_limit: int) -> int:
    local_summary = {
        "rank": dist.get_rank(),
        "diff_count": len(diffs),
        "examples": diffs[:diff_limit],
    }
    all_summaries: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_summaries, local_summary)

    diff_count = torch.tensor([len(diffs)], dtype=torch.int64, device="cuda")
    max_rel = max((diff["rel_sum_diff"] for diff in diffs), default=0.0)
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
                print("  " + _format_diff(diff), flush=True)

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


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported reduce dtype: {dtype_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-only Qwen3.5 MoE gradient determinism recorder")
    parser.add_argument("--record-path", required=True, help="Output base path; _rank<N>.json is appended")
    parser.add_argument("--compare", default=None, help="Previous output base path to compare against")
    parser.add_argument("--skip-train", action="store_true", help="Only load --record-path and compare")
    parser.add_argument("--hf-path", default=os.environ.get("QWEN35_MOE_PATH"), help="HF path or QWEN35_MOE_PATH")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-style", choices=("simple", "realistic"), default="simple")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reduce-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--deterministic", action="store_true", help="Set deterministic torch/NCCL knobs")
    parser.add_argument("--prober", action="store_true", help="Enable AccProber dumps under record-path/prober")
    parser.add_argument("--diff-limit", type=int, default=20)
    return parser.parse_args()


def _init_distributed(deterministic: bool) -> tuple[int, int]:
    if deterministic:
        os.environ.setdefault("NCCL_ALGO", "Ring")
        os.environ.setdefault("NCCL_PROTO", "Simple")
        os.environ.setdefault("NCCL_NUM_CHANNELS", "1")
        set_deterministic(deterministic=True)

    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    torch.accelerator.set_device_index(local_rank)
    return rank, local_rank


def main() -> None:
    args = parse_args()
    if args.skip_train and not args.compare:
        print("ERROR: --skip-train requires --compare", file=sys.stderr)
        sys.exit(2)
    if not args.skip_train and not args.hf_path:
        print("ERROR: --hf-path or QWEN35_MOE_PATH must be set", file=sys.stderr)
        sys.exit(2)

    monkey_patch_hf_modules_cache()
    rank, local_rank = _init_distributed(args.deterministic)
    world_size = dist.get_world_size()
    set_random_seed(args.seed, deterministic=args.deterministic)

    _print_rank0(
        "[numerics_test] "
        f"world_size={world_size} local_rank0={local_rank} layers={args.num_layers} "
        f"seq_len={args.seq_len} batch_style={args.batch_style} compile={not args.no_compile} "
        f"reduce_dtype={args.reduce_dtype} XTUNER_DETERMINISTIC={XTUNER_DETERMINISTIC} "
        f"TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK={os.environ.get('TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK')}"
    )

    total_loss = None
    if args.skip_train:
        records = load_record(args.record_path, rank)
    else:
        t0 = time.time()
        _print_rank0("[numerics_test] precomputing CPU micro-batches ...")
        micro_batches = _precompute_micro_batches(
            rank=rank,
            vocab_size=248320,
            padding_token_idx=0,
            seq_len=args.seq_len,
            batch_style=args.batch_style,
        )

        set_random_seed(args.seed, deterministic=args.deterministic)
        _print_rank0("[numerics_test] building FSDP text-only Qwen3.5 MoE model ...")
        torch._dynamo.reset()
        model = _build_model(
            hf_path=args.hf_path,
            num_layers=args.num_layers,
            compile_model=not args.no_compile,
            reduce_dtype=_resolve_dtype(args.reduce_dtype),
        )
        if args.prober:
            setup_prober_list(Path(args.record_path) / "prober", [0, 1], model, ["AccProber"])

        total_loss = _run_forward_backward(model, micro_batches)
        if args.prober:
            ProberList.before_clip_grad_norm(model)
        records = collect_grad_records(model)
        dist.barrier()
        save_record(records, base_path=args.record_path, rank=rank, args=args, total_loss=total_loss)
        dist.barrier()
        _print_rank0(
            f"[numerics_test] saved {len(records)} grad records to {args.record_path}_rank*.json "
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

"""Replay serialized GLM image/text inputs through HF or XTuner; compare CE and logits.

Run using torchrun, including --nproc-per-node=1 for the HF reference.
See doc/glm53_forward_parity.md for the input contract and commands.
"""

import argparse
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from xtuner._testing.logits import check_logits
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.compose.glm53 import Glm53BaseConfig


def main() -> None:
    """所有 DP 组回放相同样本；NLL 和计数只在 SP 组求和一次。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--backend", choices=("hf", "xt"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--sp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1 or args.ep_size < 1:
        raise ValueError("limit and ep-size must be positive")
    if args.reference is not None:
        reference_metadata = json.loads((args.reference / "metadata-rank0.json").read_text())
        if Path(reference_metadata["model"]).resolve() != Path(args.model).resolve():
            raise ValueError("Reference checkpoint path differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world % args.ep_size or world % args.sp_size:
        raise ValueError("EP/SP must divide world size")
    if args.backend == "hf" and (world, args.ep_size, args.sp_size) != (1, 1, 1):
        raise ValueError("HF reference requires world=EP=SP=1")
    if args.output.exists():
        raise FileExistsError(f"请使用新的输出目录：{args.output}")
    dist.barrier()
    if rank == 0:
        args.output.mkdir(parents=True)
    dist.barrier()
    torch.manual_seed(1234)
    mesh = init_device_mesh("cuda", (world // args.sp_size, args.sp_size), mesh_dim_names=("dp", "sp"))
    sp_mesh = mesh["sp"]
    if args.backend == "hf":
        from transformers import AutoConfig, Glm5NextForConditionalGeneration

        config = AutoConfig.from_pretrained(args.model)
        config.text_config.num_nextn_predict_layers = 0
        model = Glm5NextForConditionalGeneration.from_pretrained(
            args.model,
            config=config,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="eager",
        )
        assert model.config.vision_config._attn_implementation == "eager"
    else:
        with torch.device("meta"):
            cfg = Glm53BaseConfig.from_hf(args.model)
            cfg.text_config.mtp_config = None
            cfg.text_config.num_nextn_predict_layers = 0
            cfg.text_config.dispatcher = "all2all"
            cfg.text_config.ep_size = args.ep_size
            cfg.text_config.compile_cfg = False
            cfg.compile_cfg = False
            cfg.text_config.attention.sparse_mla_backend = "torch"
            cfg.text_config.attention.indexer_backend = "torch"
            cfg.vision_config.attn_impl = "flash_attention"
            model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)
        model.fully_shard(FSDPConfig(ep_size=args.ep_size, cpu_offload=False))
        model.from_hf(args.model, strict=False)
    model.eval()
    metadata = {
        "backend": args.backend,
        "model": args.model,
        "rank": rank,
        "world": world,
        "ep": args.ep_size,
        "sp": args.sp_size,
        "sp_ranks": sp_mesh.mesh.tolist(),
        "vision_attention": "eager" if args.backend == "hf" else "flash_attention",
        "text_attention": "eager" if args.backend == "hf" else "torch",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "eval": True,
        "no_grad": True,
        "packing": False,
        "mtp": 0,
    }
    if args.backend == "xt":
        ep_meshes = [m.ep_mesh for m in model.modules() if getattr(m, "ep_mesh", None) is not None]
        if not ep_meshes or any(m.size() != args.ep_size for m in ep_meshes):
            raise ValueError("Actual expert mesh does not match requested EP")
        metadata["ep_ranks"] = ep_meshes[0].mesh.tolist()
        metadata["expert_shards"] = [
            {"name": n, "global_shape": list(p.shape), "local_shape": list(p.to_local().shape)}
            for n, p in model.named_parameters()
            if isinstance(p, DTensor) and "experts." in n and "shared_experts" not in n
        ][:3]
    (args.output / f"metadata-rank{rank}.json").write_text(json.dumps(metadata, indent=2))
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    samples = manifest["samples"][: args.limit]
    if len({s["id"] for s in samples}) != len(samples) or len({s["index"] for s in samples}) != len(samples):
        raise ValueError("Duplicate input sample IDs or indices")
    if len(samples) != args.limit:
        raise ValueError("输入样本数不足")
    log = (args.output / "loss.jsonl").open("w") if rank == 0 else None
    comparisons = []
    reference_rows = None
    if args.reference is not None:
        reference_rows = [json.loads(line) for line in (args.reference / "loss.jsonl").read_text().splitlines()]
        if len(reference_rows) != len(samples):
            raise ValueError("Reference sample count differs")
    try:
        with torch.no_grad():
            for ordinal, sample in enumerate(samples):
                path = Path(sample["file"])
                if not path.is_absolute():
                    path = args.inputs / path
                if hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
                    raise ValueError("输入 hash 改变")
                data = load_file(str(path), device=f"cuda:{torch.cuda.current_device()}")
                ids = data["input_ids"][:, :-1].contiguous()
                labels = data["labels"][:, 1:].contiguous()
                ctx = SequenceContext.from_input_ids((ids,), device="cuda")
                ctx.mm_token_type_ids = data["mm_token_type_ids"][:, :-1].contiguous()
                # split 只补尾部到 SP 的倍数；真实样本仍只有一条，padding 不参与 CE。
                if args.sp_size > 1:
                    ctx = ctx.split(sp_mesh)
                ctx.pixel_values = data["pixel_values"].to(torch.bfloat16)
                ctx.image_grid_thw = data["image_grid_thw"]
                ctx.num_img_tokens = [[int(ctx.image_grid_thw.prod(-1).sum())]]
                if args.backend == "hf":
                    logits = model(
                        input_ids=ids,
                        pixel_values=ctx.pixel_values,
                        image_grid_thw=data["image_grid_thw"],
                        use_cache=False,
                    ).logits
                else:
                    logits = model(seq_ctx=ctx, loss_ctx=None).logits
                local_length = ctx.input_ids.shape[1]
                start = sp_mesh.get_local_rank() * local_length
                padded_labels = F.pad(labels, (0, local_length * args.sp_size - labels.shape[1]), value=-100)
                targets = padded_labels[:, start : start + local_length]
                if logits.shape[:2] != targets.shape:
                    raise ValueError(f"logits/labels mismatch: {logits.shape}, {targets.shape}")
                valid = targets != -100
                nll = (
                    F.cross_entropy(logits[valid].float(), targets[valid], reduction="sum")
                    if valid.any()
                    else logits.new_zeros((), dtype=torch.float32)
                )
                stats = torch.stack((nll.double(), valid.sum().double()))
                # EP 组不是 loss 归约组；DP 是重复验证样本，不再次求和。
                if args.sp_size > 1:
                    dist.all_reduce(stats, group=sp_mesh.get_group())
                if int(stats[1].item()) != sample["valid_tokens"] or not torch.isfinite(stats).all():
                    raise ValueError("全局监督分母或 NLL 非法")
                positions = data["logit_positions"]
                expected_positions = torch.nonzero(labels[0] != -100).flatten()[-8:]
                if not torch.equal(positions, expected_positions):
                    raise ValueError("Expected the final eight supervised global positions")
                local_mask = (positions >= start) & (positions < start + local_length)
                chosen = torch.zeros((positions.numel(), logits.shape[-1]), device=logits.device, dtype=torch.float32)
                chosen[local_mask] = logits[0, positions[local_mask] - start].float()
                if args.sp_size > 1:
                    dist.all_reduce(chosen, group=sp_mesh.get_group())
                if not torch.isfinite(chosen).all():
                    raise ValueError("抽样 logits 非有限")
                # 记录不同 DP 复制组的偏差；它们应得到相同样本统计。
                low, high = stats.clone(), stats.clone()
                dist.all_reduce(low, op=dist.ReduceOp.MIN)
                dist.all_reduce(high, op=dist.ReduceOp.MAX)
                row = {
                    "sample_id": sample["id"],
                    "index": sample["index"],
                    "input_sha256": sample["sha256"],
                    "valid_tokens": int(stats[1].item()),
                    "nll_sum": stats[0].item(),
                    "mean_ce": (stats[0] / stats[1]).item(),
                    "dp_nll_span": (high[0] - low[0]).item(),
                    "global_input_length": ids.shape[1],
                    "padded_length": local_length * args.sp_size,
                    "ep": args.ep_size,
                    "sp": args.sp_size,
                }
                if reference_rows is not None:
                    ref = reference_rows[ordinal]
                    for key in ("sample_id", "input_sha256", "valid_tokens"):
                        if row[key] != ref[key]:
                            raise ValueError(f"Reference contract mismatch: {key}")
                    reference = load_file(str(args.reference / f"{sample['index']:03d}-logits.safetensors"))
                    if not torch.equal(reference["positions"], positions.cpu()) or not torch.equal(
                        reference["labels"], labels[0, positions].cpu()
                    ):
                        raise ValueError("Reference positions/labels differ")
                    metrics = check_logits(chosen, reference["logits"])
                    comparisons.append(
                        {
                            "sample_id": sample["id"],
                            "reference_ce": ref["mean_ce"],
                            "candidate_ce": row["mean_ce"],
                            **metrics,
                        }
                    )
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    save_file(
                        {
                            "positions": positions.cpu(),
                            "labels": labels[0, positions].cpu(),
                            "logits": chosen.cpu().contiguous(),
                        },
                        str(args.output / f"{sample['index']:03d}-logits.safetensors"),
                    )
                del logits, chosen, data, ctx
        if comparisons:
            expected = torch.tensor([r["reference_ce"] for r in comparisons], dtype=torch.float64)
            actual = torch.tensor([r["candidate_ce"] for r in comparisons], dtype=torch.float64)
            mean_relative = ((actual - expected).abs() / expected.abs()).mean().item()
            cosine = F.cosine_similarity(actual, expected, dim=0).item()
            if rank == 0:
                (args.output / "comparison.json").write_text(
                    json.dumps(
                        {"mean_ce_relative": mean_relative, "loss_cosine": cosine, "samples": comparisons}, indent=2
                    )
                )
            assert mean_relative < 0.03 and cosine > 0.97, (mean_relative, cosine)
    finally:
        if log is not None:
            log.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    with torch.backends.cudnn.flags(allow_tf32=False):
        main()

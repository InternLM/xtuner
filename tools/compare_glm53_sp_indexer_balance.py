"""Check SP8 global IDs and measure paired unbalanced/balanced cooperative indexers."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.glm53.nope_dsa_mla import KPoolIndexer
from xtuner.v1.ops.sparse_mla import kpool as kpool_module
from xtuner.v1.ops.sparse_mla.sequence_parallel_indexer import (
    balanced_query_partition,
    redistribute_indexer_queries,
)


def check_routes(mesh: Any) -> dict[str, Any]:
    """Check exact multihop round trips, including destination ranks with no rows."""
    rank = dist.get_rank()
    source = tuple(3 * r for r in range(9))
    target = (0, 0, 2, 5, 5, 14, 14, 20, 24)
    ids = torch.arange(source[rank], source[rank + 1], device="cuda", dtype=torch.int64)
    features = ids[:, None].expand(-1, 7).to(torch.bfloat16).contiguous()
    moved_ids, moved_features = redistribute_indexer_queries((ids, features), source, target, mesh)
    expected = torch.arange(target[rank], target[rank + 1], device="cuda", dtype=torch.int64)
    assert torch.equal(moved_ids, expected)
    assert torch.equal(moved_features, expected[:, None].expand(-1, 7).to(torch.bfloat16))
    restored_ids, restored_features = redistribute_indexer_queries((moved_ids, moved_features), target, source, mesh)
    assert torch.equal(restored_ids, ids) and torch.equal(restored_features, features)
    return {"rank": rank, "source": source, "target": target, "round_trip_exact": True}


def context(lengths: list[int], mesh: Any) -> SequenceContext:
    """Create a packed context and split it into uniform attention shards."""
    ids = tuple(torch.zeros(1, length, dtype=torch.long) for length in lengths)
    return SequenceContext.from_input_ids(ids, device="cuda").split(sequence_parallel_mesh=mesh)


def check_ids(output: torch.Tensor, ctx: SequenceContext, lengths: list[int], topk: int) -> None:
    """Check document isolation, causality, uniqueness, complete pools and visible tails."""
    rows = output[:, 0].sort(dim=-1).values
    positions = torch.arange(ctx.shard_start, ctx.shard_start + rows.shape[0], device="cuda")
    boundaries = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device="cuda")
    documents = torch.bucketize(positions, boundaries[1:], right=True)
    starts = boundaries[documents]
    local_positions = positions - starts + 1
    valid = rows >= 0
    assert bool(((~valid) | ((rows >= starts[:, None]) & (rows <= positions[:, None]))).all())
    assert bool(((~valid[:, 1:]) | (rows[:, 1:] != rows[:, :-1])).all())
    counts = torch.minimum(local_positions // 4, torch.tensor(topk // 4, device="cuda")) * 4
    counts += local_positions % 4
    assert torch.equal(valid.sum(dim=1), counts)
    for offset in range(3):
        tail = positions - local_positions % 4 + 1 + offset
        visible = offset < local_positions % 4
        assert bool(((rows == tail[:, None]).any(dim=1) | (~visible)).all())


def run_case(module: KPoolIndexer, mesh: Any, lengths: list[int], repeats: int, ratio: float = 0.5) -> dict[str, Any]:
    """Compare selected IDs and paired rankwise elapsed timings on identical input features."""
    rank = dist.get_rank()
    module.sp_full_pool_work_ratio = ratio
    ctx = context(lengths, mesh)
    local = sum(lengths) // 8
    torch.manual_seed(711 + rank)
    hidden = torch.randn(1, local, 4096, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(1, local, 1536, device="cuda", dtype=torch.bfloat16)
    module.balance_sp = False
    baseline = module(hidden, residual, ctx)
    module.balance_sp = True
    balanced = module(hidden, residual, ctx)
    check_ids(baseline, ctx, lengths, module.index_topk)
    check_ids(balanced, ctx, lengths, module.index_topk)
    assert torch.equal(baseline.sort(dim=-1).values, balanced.sort(dim=-1).values), f"Different IDs on rank {rank}"
    assert torch.equal(baseline, balanced), f"Different canonical ID order on rank {rank}"
    del baseline, balanced
    metadata: list[dict[str, Any]] = []
    scoring: list[dict[str, Any]] = []
    cooperative_module = importlib.import_module("xtuner.v1.ops.sparse_mla.cooperative_kpool")
    original_scorer = cooperative_module.cooperative_kpool_topk
    original_redistribute = kpool_module.redistribute_indexer_queries

    def measured_redistribute(tensors: Any, source: Any, target: Any, active_mesh: Any) -> Any:
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        result = original_redistribute(tensors, source, target, active_mesh)
        end.record()
        metadata.append(
            {"source_rows": tensors[0].shape[0], "target_rows": result[0].shape[0], "begin": begin, "end": end}
        )
        return result

    def measured_scorer(*positional: Any, **keywords: Any) -> Any:
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        result = original_scorer(*positional, **keywords)
        end.record()
        scoring.append(
            {"query_rows": positional[0].shape[0], "shard_start": positional[4], "begin": begin, "end": end}
        )
        return result

    kpool_module.redistribute_indexer_queries = measured_redistribute
    cooperative_module.cooperative_kpool_topk = measured_scorer
    samples: dict[str, list[dict[str, Any]]] = {"off": [], "on": []}
    try:
        for iteration in range(2 + repeats):
            for enabled in [False, True] if iteration % 2 == 0 else [True, False]:
                module.balance_sp = enabled
                metadata.clear()
                scoring.clear()
                dist.barrier()
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                started = time.perf_counter()
                begin.record()
                output = module(hidden, residual, ctx)
                end.record()
                torch.cuda.synchronize()
                measurement = {
                    "gpu_ms": begin.elapsed_time(end),
                    "wall_ms": (time.perf_counter() - started) * 1000,
                    "redistribution": [
                        {
                            "source_rows": entry["source_rows"],
                            "target_rows": entry["target_rows"],
                            "gpu_ms": entry["begin"].elapsed_time(entry["end"]),
                        }
                        for entry in metadata
                    ],
                    "scoring": [
                        {
                            "query_rows": entry["query_rows"],
                            "shard_start": entry["shard_start"],
                            "gpu_ms": entry["begin"].elapsed_time(entry["end"]),
                        }
                        for entry in scoring
                    ],
                }
                if iteration >= 2:
                    samples["on" if enabled else "off"].append(measurement)
                del output
    finally:
        kpool_module.redistribute_indexer_queries = original_redistribute
        cooperative_module.cooperative_kpool_topk = original_scorer
    cu = tuple([0] + list(torch.tensor(lengths).cumsum(0).tolist()))
    pools = sum((length + 3) // 4 for length in lengths)
    partition = balanced_query_partition(cu, 8, 4, query_alignment=4, per_query_work=round(pools * ratio))
    return {
        "rank": rank,
        "lengths": lengths,
        "exact_selected_ids": True,
        "causal_document_tail_checks": True,
        "exact_canonical_id_order": True,
        "sp_full_pool_work_ratio": ratio,
        "partition": {
            "attention": partition.attention,
            "indexer": partition.indexer,
            "candidate_work": partition.candidate_work,
            "estimated_work": partition.estimated_work,
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("work_dirs/glm53_block_profile/sp_balance"))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.0])
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    assert dist.get_world_size() == 8
    mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("sp",))
    dist.barrier()
    routes = check_routes(mesh)
    torch.manual_seed(173)
    module = (
        KPoolIndexer(
            hidden_size=4096,
            q_lora_rank=1536,
            index_head_dim=128,
            index_n_heads=32,
            index_topk=2048,
            index_kpool=4,
            index_kpool_always_select_tail=True,
            indexer_backend="tilelang_cooperative",
            alignment=512,
            topk_query_chunk_size=1024,
            balance_sp=True,
            sp_full_pool_work_ratio=0.5,
        )
        .cuda()
        .bfloat16()
    )
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1)
            elif name.endswith("weight"):
                param.normal_(0, 0.02)
            else:
                param.zero_()
    module.requires_grad_(False)
    all_results: list[dict[str, Any]] = []
    cases = [([3, 5, 257, 759], 0.5), ([1] * 8, 0.5)]
    cases += [([131072], ratio) for ratio in args.ratios]
    for lengths, ratio in cases:
        with torch.no_grad():
            result = run_case(module, mesh, lengths, args.repeats if sum(lengths) == 131072 else 1, ratio)
        collected: list[Any] = [None] * 8
        dist.all_gather_object(collected, result)
        all_results.append({"lengths": lengths, "sp_full_pool_work_ratio": ratio, "ranks": collected})
        if rank == 0:
            print(f"Exact selected-ID/causal/tail checks passed: {lengths}, ratio={ratio}", flush=True)
    route_results: list[Any] = [None] * 8
    dist.all_gather_object(route_results, routes)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {}
        for case in all_results:
            if sum(case["lengths"]) != 131072:
                continue
            pair: dict[str, Any] = {}
            for label in ("off", "on"):
                rank_maxima = [
                    max(r["samples"][label][i]["wall_ms"] for r in case["ranks"]) for i in range(args.repeats)
                ]
                pair[label] = {
                    "rank_max_wall_ms_mean": statistics.mean(rank_maxima),
                    "rank_max_wall_ms_median": statistics.median(rank_maxima),
                    "rank_max_wall_ms_samples": rank_maxima,
                    "rankwise_gpu_ms_mean": [
                        statistics.mean(x["gpu_ms"] for x in r["samples"][label]) for r in case["ranks"]
                    ],
                }
            pair["speedup"] = pair["off"]["rank_max_wall_ms_mean"] / pair["on"]["rank_max_wall_ms_mean"]
            summary[str(case["sp_full_pool_work_ratio"])] = pair
        (args.output / "indexer_comparison.json").write_text(
            json.dumps({"route_validation": route_results, "cases": all_results, "summary": summary}, indent=2)
        )
        print(json.dumps(summary, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

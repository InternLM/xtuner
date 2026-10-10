# Copyright (c) OpenMMLab. All rights reserved.
"""Compare original GLM53 DSA with cooperative BF16 + DeepSelect on SM90.

Use the existing pt29_glm2 interpreter; set CUDA_HOME to the CUDA 12.8 toolkit.
The run writes JSON incrementally, excluding JIT compilation from timings.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.glm53.nope_dsa_mla import KPoolIndexer
from xtuner.v1.ops.sparse_mla import get_kpool_topk_indices
from xtuner.v1.ops.sparse_mla.cooperative_kpool import cooperative_kpool_topk
from xtuner.v1.ops.sparse_mla.tilelang import tilelang_indexer_topk_from_ranges
from xtuner.v1.ops.sparse_mla.tilelang_indexer_fwd import indexer_fwd_interface
from xtuner.v1.ops.sparse_mla.tilelang_indexer_scoring4_cooperative_paired_query_m64_bf16_full_m64n128_producer import (
    scorer_full_interface,
)


def context(lengths):
    return SequenceContext.from_input_ids(tuple(torch.zeros(1, n, dtype=torch.long) for n in lengths), device="cuda")


def errors(original, new):
    a, b = original.double(), new.double()
    valid = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[valid], b[valid]
    diff = b - a
    return dict(
        max_abs=diff.abs().max().item() if diff.numel() else 0,
        rms=diff.square().mean().sqrt().item() if diff.numel() else 0,
        relative_l2=(diff.norm() / a.norm().clamp_min(1e-30)).item(),
    )


def selections(original, new):
    a = original.squeeze(1).sort(dim=-1).values.contiguous()
    b = new.squeeze(1).sort(dim=-1).values.contiguous()
    assert a.shape == b.shape
    count_a, count_b = (a >= 0).sum(1), (b >= 0).sum(1)
    assert torch.equal(count_a, count_b), "Different valid selection counts"
    positions = torch.searchsorted(a, b).clamp_max(a.shape[-1] - 1)
    matches = (torch.gather(a, 1, positions) == b) & (b >= 0)
    valid_total = int(count_b.sum().item())
    return dict(
        equal_rows=int((a == b).all(1).sum().item()),
        total_rows=a.shape[0],
        overlap=float(matches.sum().item() / max(valid_total, 1)),
    )


def measure(fn, repeats):
    for _ in range(2):
        value = fn()
        del value
    torch.cuda.synchronize()
    gc.collect()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    gpu, wall = [], []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        begin.record()
        value = fn()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1000)
        gpu.append(begin.elapsed_time(end))
        del value
    return dict(
        gpu_ms=statistics.median(gpu),
        wall_ms=statistics.median(wall),
        peak_extra_mib=(torch.cuda.max_memory_allocated() - base) / 2**20,
        gpu_samples_ms=gpu,
        wall_samples_ms=wall,
    )


def initialize(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.fill_(1)
            elif name.endswith("bias"):
                p.zero_()
            else:
                p.normal_(0, 0.02)


def score_check():
    torch.manual_seed(71)
    nq, nk = 2049, 1025
    q = torch.randn(nq, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(nk, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(nq, 32, device="cuda") * (32 * 128) ** -0.5
    starts = torch.zeros(nq, dtype=torch.int32, device="cuda")
    ends = torch.arange(nq, device="cuda", dtype=torch.int32) // 2 + 1
    padded = (-nq) % 4
    old = indexer_fwd_interface(
        F.pad(q, (0, 0, 0, 0, 0, padded)),
        k,
        F.pad(weights, (0, 0, 0, padded)),
        F.pad(starts, (0, padded)),
        F.pad(ends, (0, padded)),
    )[:nq]
    new = scorer_full_interface(q, k, weights, starts, ends)
    visible = torch.arange(nk, device="cuda")[None] < ends[:, None]
    reference = (torch.einsum("qhd,kd->qhk", q.float(), k.float()).relu() * weights[:, :, None]).sum(1)
    metrics = dict(
        original_vs_new=errors(old[visible], new[visible]),
        original_vs_torch=errors(reference[visible], old[visible]),
        new_vs_torch=errors(reference[visible], new[visible]),
    )
    torch.testing.assert_close(old[visible], new[visible], atol=2e-5, rtol=2e-5)
    return metrics


def selector_case(n, repeats):
    torch.manual_seed(101 + n)
    p = (n + 3) // 4
    q = torch.randn(n, 32, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(p, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(n, 32, device="cuda") * (32 * 128) ** -0.5
    starts = torch.zeros(n, dtype=torch.int32, device="cuda")
    ends = (torch.arange(n, dtype=torch.int32, device="cuda") + 1) // 4
    cu = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    fns = {
        "original_unchunked": lambda: tilelang_indexer_topk_from_ranges(q, k, weights, starts, ends, 512),
        "original_chunk1024": lambda: tilelang_indexer_topk_from_ranges(
            q, k, weights, starts, ends, 512, query_chunk_size=1024
        ),
        "cooperative_chunk1024": lambda: cooperative_kpool_topk(q, k, weights, cu, 0, 4, 512, 1024),
    }
    old, new = fns["original_chunk1024"](), fns["cooperative_chunk1024"]()
    result = dict(stage="scorer_and_selector", lengths=[n], selection=selections(old, new))
    del old, new
    result["timing"] = {name: measure(fn, repeats) for name, fn in fns.items()}
    return result


def pipeline_case(lengths, repeats):
    n = sum(lengths)
    torch.manual_seed(201 + n)
    q = torch.randn(1, n, 32, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, n, 128, dtype=torch.bfloat16, device="cuda")
    gates = torch.randn_like(k)
    weights = torch.randn(1, n, 32, device="cuda")
    ape = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    ctx = context(lengths)
    kwargs = dict(index_head_dim=128, index_topk=2048, index_kpool=4, alignment=512)

    def run(backend, chunk):
        return get_kpool_topk_indices(backend)(q, k, gates, weights, ape, ctx, query_chunk_size=chunk, **kwargs)

    old, new = run("tilelang", 1024), run("tilelang_cooperative", 1024)
    result = dict(stage="pooling_scoring_selection_expansion", lengths=lengths, selection=selections(old, new))
    del old, new
    result["timing"] = {
        name: measure(fn, repeats)
        for name, fn in {
            "original_unchunked": lambda: run("tilelang", None),
            "original_chunk1024": lambda: run("tilelang", 1024),
            "cooperative_chunk1024": lambda: run("tilelang_cooperative", 1024),
        }.items()
    }
    return result


def indexer_case(n, repeats):
    torch.manual_seed(301 + n)
    module = (
        KPoolIndexer(
            hidden_size=4096,
            q_lora_rank=1536,
            index_head_dim=128,
            index_n_heads=32,
            index_topk=2048,
            index_kpool=4,
            index_kpool_always_select_tail=True,
            indexer_backend="tilelang",
            alignment=512,
            topk_query_chunk_size=1024,
        )
        .cuda()
        .bfloat16()
    )
    initialize(module)
    hidden = torch.randn(1, n, 4096, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(1, n, 1536, device="cuda", dtype=torch.bfloat16)
    ctx = context([n])

    def run(backend, chunk):
        module._topk_indices_fn = get_kpool_topk_indices(backend)
        module.topk_query_chunk_size = chunk
        with torch.no_grad():
            return module(hidden, residual, ctx)

    old, new = run("tilelang", 1024), run("tilelang_cooperative", 1024)
    result = dict(stage="full_indexer_with_projections", lengths=[n], selection=selections(old, new))
    del old, new
    result["timing"] = {
        name: measure(fn, repeats)
        for name, fn in {
            "original_unchunked": lambda: run("tilelang", None),
            "original_chunk1024": lambda: run("tilelang", 1024),
            "cooperative_chunk1024": lambda: run("tilelang_cooperative", 1024),
        }.items()
    }
    return result


def component_case(n, repeats):
    """Isolate scoring and selection for the final 1024 causal query rows."""
    import deep_select

    from xtuner.v1.ops.sparse_mla.tilelang_indexer_fwd import clean_logits_, tl_indexer_fwd_impl
    from xtuner.v1.ops.sparse_mla.tilelang_indexer_scoring4_cooperative_paired_query_m64_bf16_full_m64n128_producer import (
        allocate_scores,
    )

    query_len, key_len = min(1024, n), (n + 3) // 4
    q = torch.randn(query_len, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(key_len, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(query_len, 32, device="cuda") * (32 * 128) ** -0.5
    starts = torch.zeros(query_len, device="cuda", dtype=torch.int32)
    ends = (torch.arange(n - query_len, n, device="cuda", dtype=torch.int32) + 1) // 4
    old = torch.empty(query_len, key_len, dtype=torch.float32, device="cuda")
    new = allocate_scores(query_len, key_len, "cuda")
    ids = torch.empty(query_len, 512, device="cuda", dtype=torch.int32)
    kernel, clean = tl_indexer_fwd_impl(heads=32, index_dim=128), clean_logits_()

    def oldscore():
        kernel(q.view(-1, 128), k, old, w, starts, ends)
        clean(old, starts, ends)
        return old

    def newscore():
        return scorer_full_interface(q, k, w, starts, ends, out=new)

    oldscore()
    # DeepSelect needs an aligned score stride; copy once outside selection timing.
    selector_input = allocate_scores(query_len, key_len, "cuda")
    selector_input.copy_(old)

    def oldtopk():
        values, indices = selector_input.topk(min(512, key_len), dim=-1)
        return indices.masked_fill(values == -torch.inf, -1).to(torch.int32)

    def newtopk():
        return deep_select.topk(
            selector_input,
            512,
            end=ends,
            indices_type=torch.int32,
            sorted=False,
            sorted_index=False,
            return_value=False,
            output_idx=ids,
            idx_oob_fill_value=-1,
        )[1]

    return dict(
        stage="isolated_components",
        lengths=[n],
        query_rows=query_len,
        key_rows=key_len,
        window="final causal query chunk; preallocated score storage; identical selector input",
        timing={
            name: measure(fn, repeats)
            for name, fn in {
                "original_scorer_with_clean": oldscore,
                "cooperative_scorer": newscore,
                "torch_topk_with_padding": oldtopk,
                "deepselect_topk": newtopk,
            }.items()
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="work_dirs/glm53_cooperative_comparison/results.json")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384, 32768, 65536, 131072])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    import deep_select
    import tilelang

    report = dict(
        environment=dict(
            torch=torch.__version__,
            tilelang=tilelang.__version__,
            gpu=torch.cuda.get_device_name(),
            cuda_home=__import__("os").environ.get("CUDA_HOME"),
            interpreter=__import__("sys").executable,
            deepselect_alignment=deep_select.get_stride_requirement(),
        ),
        chunk_size=1024,
        repeats=args.repeats,
        cases=[],
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save(result):
        report["cases"].append(result)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(result), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    save(dict(stage="score_accuracy", result=score_check()))
    for n in args.lengths:
        save(component_case(n, max(15, args.repeats)))
        save(selector_case(n, args.repeats))
        save(pipeline_case([n], args.repeats))
        save(indexer_case(n, args.repeats))
    save(pipeline_case([8191, 16389, 8188], args.repeats))


if __name__ == "__main__":
    main()

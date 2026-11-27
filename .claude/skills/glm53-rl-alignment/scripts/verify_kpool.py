"""Compare one GLM-5.3-Flash BF16 checkpoint layer across pinned KPool implementations."""

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


@torch.inference_mode()
def main() -> None:
    """Run the selected backend and write or compare the reference indices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("main", "preview", "lmdeploy"))
    parser.add_argument("checkpoint", type=Path, help="Local sharded BF16 Hugging Face checkpoint")
    parser.add_argument("reference", type=Path, help="Tensor file written by main and read by the other backends")
    args = parser.parse_args()
    cfg = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    layer, weights = _load_first_indexer(args.checkpoint)
    length = cfg["index_topk"] + 257
    rng = torch.Generator().manual_seed(20261127)
    hidden, latent = [
        torch.randn(1, length, cfg[dim], generator=rng, device="cpu").to(device="cuda", dtype=torch.bfloat16)
        for dim in ("hidden_size", "q_lora_rank")
    ]
    if args.backend == "lmdeploy":
        ids = _rollout_indices(cfg, weights, hidden, latent)
    else:
        ids = _training_indices(args.backend, args.checkpoint, weights, hidden, latent)
    if ids.shape[0] != length:
        raise ValueError(f"Expected {length} rows, got {ids.shape[0]}")
    selected = _token_sets(ids, length, cfg["index_topk"], cfg["index_kpool"])
    result = {"backend": args.backend, "layer": layer, "shape": list(ids.shape)}
    if args.backend == "main":
        torch.save(ids, args.reference)
        result["sha256"] = hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()
    else:
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
        if args.backend == "preview":
            torch.testing.assert_close(ids, reference, rtol=0, atol=0)
            result["identical"] = True
        else:
            expected = _token_sets(reference, length, cfg["index_topk"], cfg["index_kpool"])
            different = (selected != expected).any(dim=1).sum().item()
            result["rows_with_different_token_sets"] = different
            if different:
                raise ValueError(f"LMDeploy prefill differs from XTuner in {different}/{length} rows")
    print(json.dumps(result))


def _load_first_indexer(checkpoint: Path) -> tuple[int, dict[str, torch.Tensor]]:
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    pattern = re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.indexer\.(.+)")
    by_layer = defaultdict(list)
    for name, shard in weight_map.items():
        match = pattern.fullmatch(name)
        if match:
            by_layer[int(match[1])].append((shard, name, match[2]))
    if not by_layer:
        raise ValueError("Expected a sharded GLM-5.3 checkpoint with language_model indexer weights")
    layer = min(by_layer)
    weights = {}
    for shard, name, parameter in by_layer[layer]:
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as reader:
            weights[parameter] = reader.get_tensor(name)
    return layer, weights


def _training_indices(
    branch: str, checkpoint: Path, weights: dict[str, torch.Tensor], hidden: torch.Tensor, latent: torch.Tensor
) -> torch.Tensor:
    from xtuner.v1.data_proto import SequenceContext
    from xtuner.v1.model import get_model_config_from_hf

    model = get_model_config_from_hf(checkpoint)
    attn = model.attention
    attn.indexer_balance_sp = False
    attn.freeze_dsa_indexer = True
    attn.indexer_topk_query_chunk_size = 1024
    attn.sparse_mla_backend = "flash_mla_cudnn"
    if branch == "main":
        attn.indexer_backend = "lmdeploy_bf16"
        attn.indexer_kpool_fp32_prep = True
    else:
        from xtuner.v1.ops.sparse_mla.kpool import _lmdeploy_kpool_is_available

        attn.indexer_backend = "tilelang"
        if not _lmdeploy_kpool_is_available(hidden.device, bf16=True):
            raise RuntimeError("Preview requires XTUNER_GLM53_KPOOL_BF16=1 and LMDeploy BF16 compression")

    with torch.device("meta"):
        indexer = attn.build(hidden_size=model.hidden_size, float8_cfg=None).indexer
    indexer.to_empty(device="cuda").to(dtype=torch.bfloat16)
    indexer.load_state_dict(weights, strict=True)
    tokens = torch.zeros(1, hidden.shape[1], dtype=torch.long)
    ctx = SequenceContext.from_input_ids((tokens,), device="cuda")
    return indexer(hidden, latent, ctx).cpu()


def _rollout_indices(
    cfg: dict[str, int], weights: dict[str, torch.Tensor], hidden: torch.Tensor, latent: torch.Tensor
) -> torch.Tensor:
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_prefill_update_cuda, kpool_select_prefill_cuda
    from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheGeometry, BlockCacheRequestContext
    from lmdeploy.pytorch.nn.kpool import KPOOL_PAGE_SIZE, KPoolIndexer

    dimensions = ("hidden_size", "q_lora_rank", "index_n_heads", "index_head_dim", "index_topk", "index_kpool")
    indexer = KPoolIndexer(**{k: cfg[k] for k in dimensions}, dtype=torch.bfloat16, device="cuda")
    if indexer.cache_dtype != torch.bfloat16:
        raise RuntimeError("Set LMDEPLOY_GLM53_KPOOL_BF16=1 before starting Python")
    indexer.load_state_dict(weights, strict=True)

    length = hidden.shape[1]
    pool = cfg["index_kpool"]
    pages = math.ceil(length / KPOOL_PAGE_SIZE)
    geometry = BlockCacheGeometry(KPOOL_PAGE_SIZE, KPOOL_PAGE_SIZE)
    (request,) = indexer.get_block_cache_requests(BlockCacheRequestContext(geometry))
    cache = torch.zeros(pages, *request.shape, dtype=request.dtype, device="cuda")
    blocks = torch.arange(pages, device="cuda", dtype=torch.int32).unsqueeze(0)
    lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
    keys = indexer.project_key(hidden)[0]
    scores = indexer.project_compress_score(hidden)[0]
    tail = keys.new_zeros(1, pool, cfg["index_head_dim"])
    state_ids = torch.zeros(1, device="cuda", dtype=torch.int64)
    kpool_prefill_update_cuda(
        keys,
        scores,
        tail,
        tail,
        state_ids,
        lengths,
        lengths,
        cache,
        blocks,
        indexer.index_kpool_compress_ape,
        pool,
        indexer.scale_fmt is not None,
    )
    queries = indexer.project_query(latent)[0]
    head_gates = indexer.project_head_gate(hidden)[0] * indexer.softmax_scale
    return kpool_select_prefill_cuda(
        queries, head_gates, cache, lengths, lengths, blocks, length, pool, cfg["index_topk"]
    ).cpu()


def _token_sets(indices: torch.Tensor, length: int, topk: int, pool: int) -> torch.Tensor:
    # Sort with padding last so training and serving can use different buffer widths.
    rows = indices.reshape(length, -1)
    if rows.dtype != torch.int32 or (rows < -1).any() or (rows > torch.arange(length)[:, None]).any():
        raise ValueError("Unexpected index dtype, padding or noncausal token ID")
    prefix = torch.arange(1, length + 1)
    counts = torch.minimum(prefix // pool * pool, torch.tensor(topk)) + prefix % pool
    if not torch.equal((rows >= 0).sum(1), counts):
        raise ValueError("Unexpected number of selected tokens")
    ordered = rows.masked_fill(rows < 0, length).sort(dim=1).values
    if ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] < length)).any():
        raise ValueError("Duplicate token in Top-K")
    return ordered[:, : counts.max().item()]


if __name__ == "__main__":
    main()

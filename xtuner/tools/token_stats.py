# Copyright (c) OpenMMLab. All rights reserved.
"""Token lengths, truncation losses and packing mixture changes for XTuner v1 text datasets."""

import argparse
import csv
import logging
import os
import sys
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


Row = dict[str, str | int | float | None]
_LENGTH_KEYS = ("num_tokens", "original_num_tokens")


def summarize_samples(name: str, shards: Sequence[Mapping[str, np.ndarray]], scope: str) -> Row:
    """Summarize original lengths and tokenization losses for one subset.

    Args:
        name (str): Dataset/subset name.
        shards (Sequence[Mapping[str, np.ndarray]]): Metadata in sample order, before packing.
        scope (str): Counting population, e.g. source samples or sampled instances.

    Returns:
        Row: One CSV row. Ratios are fractions in [0, 1], undefined metrics are None.
    """
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in _LENGTH_KEYS}
    for shard in shards:
        if "chunks" in shard:
            raise ValueError("Long-text chunk caches are not original-sample metadata")
        n = len(shard["num_tokens"])
        for key in _LENGTH_KEYS:
            values = np.asarray(shard[key] if key in shard else np.full(n, -1, dtype=np.int64))
            # Empty ranks can promote cached counts to float. Reject non-integral values.
            if (
                values.ndim != 1
                or len(values) != n
                or values.dtype.kind not in "iuf"
                or (
                    values.dtype.kind == "f"
                    and (not np.all(np.isfinite(values)) or np.any(values != np.floor(values)))
                )
            ):
                raise ValueError(f"Invalid {key} array in subset {name!r}")
            arrays[key].append(values.astype(np.int64, copy=False))
    merged = {
        key: vals[0] if len(vals) == 1 else np.concatenate(vals) if vals else np.empty(0, dtype=np.int64)
        for key, vals in arrays.items()
    }
    kept, original = (merged[key] for key in _LENGTH_KEYS)
    if np.any(kept < 0) or np.any(original < -1):
        raise ValueError(f"Invalid negative token counts in subset {name!r}")
    known = original >= 0
    known_count = int(known.sum())
    known_original = original if known_count == len(original) else original[known]
    # num_tokens is recorded after tokenization truncation, before packing and collation.
    lost = known_original - (kept if known_count == len(kept) else kept[known])
    if np.any(lost < 0):
        raise ValueError(f"Inconsistent tokenization truncation metadata in subset {name!r}")
    truncated_count = int((lost > 0).sum())
    lost_total = int(lost.sum())
    del lost
    denominator = int(known_original.sum())
    return {
        "name": name,
        "scope": scope,
        "truncation_stage": "tokenization",
        "num_shards": len(shards),
        "count": len(kept),
        "damaged_or_empty_count": int((kept == 0).sum()),
        "known_original_count": known_count,
        "unknown_original_count": len(kept) - known_count,
        "complete_count": known_count,
        "incomplete_count": len(kept) - known_count,
        **_distribution(known_original, "original"),
        "retained_tokens_all_samples": int(kept.sum()),
        "complete_original_tokens": denominator,
        "truncated_count": truncated_count if known_count else None,
        "truncated_sample_ratio": truncated_count / known_count if known_count else None,
        "truncated_tokens": lost_total if known_count else None,
        "truncated_token_ratio": lost_total / denominator if denominator else None,
    }


def summarize_datasets(datasets: Sequence[Any]) -> list[Row]:
    """Summarize datasets after filtering and sampling.

    Args:
        datasets (Sequence[Any]): Dataset instances, each supplied once (not once per rank).
            Distinct instances may use the same source file.

    Returns:
        list[Row]: One row per dataset name; repetitions from sample_ratio count as instances.
    """
    grouped: dict[str, list[Mapping[str, np.ndarray]]] = defaultdict(list)
    seen: set[int] = set()
    for dataset in datasets:
        instance_id = id(dataset)
        if instance_id in seen:
            raise ValueError(f"Dataset object supplied more than once: {dataset.name!r} ({dataset.path})")
        seen.add(instance_id)
        grouped[dataset.name].append(dataset._meta)
    return [
        summarize_samples(name, shards, "sampled_instances_before_packing") for name, shards in sorted(grouped.items())
    ]


def summarize_cache_manifest(manifest: Sequence[Mapping[str, Any]], workers: int = 1) -> list[Row]:
    """Summarize existing caches by subset.

    Args:
        manifest (Sequence[Mapping[str, Any]]): Entries with name and meta_dirs (jsonl_meta directories).
            A cache may be shared by different subsets, but supplied only once within each subset.
        workers (int): Parallel subset readers; only the parent writes CSVs.

    Returns:
        list[Row]: Original-source statistics before filtering and sample_ratio.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, Path]] = set()
    for entry in manifest:
        name = entry["name"]
        for directory in entry["meta_dirs"]:
            path = Path(directory).resolve()
            key = (name, path)
            if key in seen:
                raise ValueError(f"Cache supplied more than once for subset {name!r}: {path}")
            seen.add(key)
            grouped[name].append(str(path))
    jobs = sorted(grouped.items())
    if workers == 1:
        return [_summarize_cache_subset(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_summarize_cache_subset, jobs))


def summarize_packing(packed_dataset: Any, dataloader_config: Any, *, subset_rows: Sequence[Row] | None = None) -> Row:
    """Measure packing/collator losses and changes in subset token shares.

    Args:
        packed_dataset (Any): Existing text soft/hard packed dataset or unpacked ConcatDataset.
        dataloader_config (Any): DataloaderConfig supplying the actual collator and length settings.
        subset_rows (Sequence[Row] | None): Enrich these rows in place; each must describe the same sampled
            population and include a matching retained_tokens_all_samples total.

    Returns:
        Row: Global pack lengths and losses, before the distributed sampler. Padding and label shift are separate.
    """
    from xtuner.v1.datasets.collator import build_text_ctx_labels
    from xtuner.v1.datasets.packing import HardPackDataset
    from xtuner.v1.utils import IGNORE_INDEX

    if dataloader_config.collator != "sft_llm_collator":
        raise ValueError("Packing statistics require sft_llm_collator")
    if dataloader_config.pack_level not in ("none", "soft", "hard", "__legacy"):
        raise ValueError("Packing statistics support text none/soft/hard/__legacy only")
    sources = _source_datasets(packed_dataset)
    fields = (
        "packing_input_tokens",
        "packing_retained_tokens",
        "packing_truncated_instances",
        "packing_truncated_tokens",
        "packing_dropped_instances",
        "packing_dropped_tokens",
        "collator_input_sequences",
        "collator_truncated_sequences",
        "collator_truncated_tokens",
        "collator_dropped_sequences",
        "collator_dropped_tokens",
        "label_shift_tokens",
    )
    totals = {source.name: dict.fromkeys(fields, 0) for source in sources}
    supervised = dict.fromkeys(totals, 0)
    hard_pack = isinstance(packed_dataset, HardPackDataset)
    used = {id(source): np.zeros(len(source), dtype=np.int64) for source in sources} if hard_pack else {}
    for source in sources:
        totals[source.name]["packing_input_tokens"] += int(source.num_tokens.sum())
    lengths: list[int] = []
    padding = 0
    zero_supervision_packs = 0
    for index in range(len(packed_dataset)):
        items = packed_dataset[index]
        items = [items] if isinstance(items, dict) else items
        refs = _pack_sources(packed_dataset, index)
        if len(refs) != len(items):
            raise ValueError("Pack source indices do not match the returned sequences")
        before = [item["num_tokens"] for item in items]
        # The collator may replace fields when truncating. Do not mutate cached/user-owned items.
        ctx, shifted_labels, retained = build_text_ctx_labels(
            [dict(item) for item in items],
            pack_max_length=dataloader_config.pack_max_length,
            padding_token_idx=dataloader_config.pad_token_id or 0,
            pack_to_max_length=dataloader_config.pack_to_max_length,
            pad_chunk_size=256,
        )
        after = [item["num_tokens"] for item in retained]
        effective = ctx.input_ids.numel() - ctx.num_padding
        lengths.append(effective)
        padding += ctx.num_padding
        supervised_mask = (shifted_labels != IGNORE_INDEX).reshape(-1)
        zero_supervision_packs += int(not supervised_mask.any())
        label_start = 0
        for position, ((source, sample_index), count) in enumerate(zip(refs, before)):
            row = totals[source.name]
            row["packing_retained_tokens"] += count
            row["collator_input_sequences"] += 1
            if hard_pack:
                used[id(source)][sample_index] += count
            if position >= len(after):
                row["collator_dropped_sequences"] += 1
                row["collator_dropped_tokens"] += count
            else:
                loss = count - after[position]
                row["collator_truncated_sequences"] += int(loss > 0)
                row["collator_truncated_tokens"] += loss
                # Map each source's retained label interval through the one-position shift.
                label_end = label_start + after[position]
                supervised[source.name] += int(supervised_mask[max(label_start - 1, 0) : max(label_end - 1, 0)].sum())
                label_start = label_end
        # The text collator shifts once per pack, removing its last retained input token.
        totals[refs[len(after) - 1][0].name]["label_shift_tokens"] += sum(after) - effective

    if hard_pack:
        for source in sources:
            retained_tokens = used[id(source)]
            original = source.num_tokens
            if np.any(retained_tokens > original):
                raise ValueError("Hard-pack usage exceeds the cached sample lengths")
            partial = (retained_tokens > 0) & (retained_tokens < original)
            dropped = (retained_tokens == 0) & (original > 0)
            row = totals[source.name]
            row["packing_truncated_instances"] += int(partial.sum())
            row["packing_truncated_tokens"] += int((original[partial] - retained_tokens[partial]).sum())
            row["packing_dropped_instances"] += int(dropped.sum())
            row["packing_dropped_tokens"] += int(original[dropped].sum())

    for row in totals.values():
        if row["packing_input_tokens"] - row["packing_retained_tokens"] != (
            row["packing_truncated_tokens"] + row["packing_dropped_tokens"]
        ):
            raise ValueError("Packing token accounting does not match cached lengths; check the config/cache pairing")
        row["after_collator_tokens"] = (
            row["packing_retained_tokens"] - row["collator_truncated_tokens"] - row["collator_dropped_tokens"]
        )
        row["packing_collator_lost_tokens"] = row["packing_input_tokens"] - row["after_collator_tokens"]
        row["effective_tokens"] = row["after_collator_tokens"] - row["label_shift_tokens"]
    sums = {
        key: sum(row[key] for row in totals.values()) for key in next(iter(totals.values()), dict.fromkeys(fields, 0))
    }
    supervised_total = sum(supervised.values())
    if subset_rows is not None:
        if {row["name"] for row in subset_rows} != set(totals):
            raise ValueError("Packing sources and subset report names do not match")
        for row in subset_rows:
            expected = totals[str(row["name"])]["packing_input_tokens"]
            if row.get("retained_tokens_all_samples") != expected:
                raise ValueError("Subset and packing populations do not match")
        for row in subset_rows:
            counts = totals[str(row["name"])]
            row.update(counts)
            row["packing_collator_loss_ratio"] = _ratio(
                counts["packing_collator_lost_tokens"], counts["packing_input_tokens"]
            )
            row["token_share_before_packing"] = _ratio(counts["packing_input_tokens"], sums["packing_input_tokens"])
            row["token_share_after_packing"] = _ratio(counts["after_collator_tokens"], sums["after_collator_tokens"])
            row["effective_token_share"] = _ratio(counts["effective_tokens"], sums["effective_tokens"])
            before_share, after_share = row["token_share_before_packing"], row["token_share_after_packing"]
            row["token_share_change"] = (
                after_share - before_share if before_share is not None and after_share is not None else None
            )
            row["supervised_tokens_after_collator"] = supervised[str(row["name"])]
            row["supervised_token_share"] = _ratio(supervised[str(row["name"])], supervised_total)
    effective_total = sum(lengths)
    return {
        "scope": "all_packs_once_before_distributed_sampler",
        "pack_level": dataloader_config.pack_level,
        "pack_max_length": dataloader_config.pack_max_length,
        "count": len(lengths),
        **_distribution(np.asarray(lengths, dtype=np.int64), "effective"),
        "sampled_input_tokens": sums["packing_input_tokens"],
        "tokens_before_collation": sums["packing_retained_tokens"],
        "packing_unassigned_tokens": sums["packing_input_tokens"] - sums["packing_retained_tokens"],
        **{key: sums[key] for key in fields[2:]},
        "collator_truncated_sequence_ratio": _ratio(
            sums["collator_truncated_sequences"], sums["collator_input_sequences"]
        ),
        "collator_truncated_token_ratio": _ratio(sums["collator_truncated_tokens"], sums["packing_retained_tokens"]),
        "packing_collator_lost_tokens": sums.get("packing_collator_lost_tokens", 0),
        "packing_collator_loss_ratio": _ratio(
            sums.get("packing_collator_lost_tokens", 0), sums["packing_input_tokens"]
        ),
        "padding_tokens": padding,
        "supervised_tokens_after_collator": supervised_total,
        "supervision_ratio": _ratio(supervised_total, effective_total),
        "zero_supervision_pack_count": zero_supervision_packs,
        "padding_ratio": _ratio(padding, effective_total + padding),
    }


def save_reports(
    rows: Sequence[Row], output_dir: str | Path, packing: Row | None = None, *, seed: int | None = None
) -> tuple[Path, str]:
    """Save CSV reports in a new directory and return the log summary.

    Args:
        rows (Sequence[Row]): Final subset aggregates (also used for the log summary).
        output_dir (str | Path): Existing work/output directory.
        packing (Row | None): Optional global packing/collator summary.
        seed (int | None): Seed used to sample the datasets; None for cache-only or unknown runs.

    Returns:
        tuple[Path, str]: Output directory and short log message derived from these same rows.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_")
    run_dir = Path(tempfile.mkdtemp(prefix=f"token_stats_{stamp}", dir=output))
    _write_csv(run_dir / "subset_token_stats.csv", [{**row, "seed": seed} for row in rows])
    if packing is not None:
        _write_csv(run_dir / "packing_token_stats.csv", [{**packing, "seed": seed}])
    count = sum(int(row["count"]) for row in rows)
    known = sum(int(row["known_original_count"]) for row in rows)
    complete = sum(int(row["complete_count"]) for row in rows)
    lost = sum(int(row["truncated_tokens"] or 0) for row in rows)
    truncated = sum(int(row["truncated_count"] or 0) for row in rows)
    truncation_summary = (
        f"{truncated} samples, {lost} tokens (complete coverage only)" if complete else "unknown (no complete samples)"
    )
    message = (
        f"Token statistics: {len(rows)} subsets, {count} samples/instances; original length known {known}/{count}, "
        f"complete {complete}/{count}; tokenization truncated {truncation_summary}. "
        f"CSV: {run_dir / 'subset_token_stats.csv'}"
    )
    if seed is not None:
        message += f" Seed: {seed}."
    if packing is not None:
        message += (
            f" Packing/collator lost {packing['packing_collator_lost_tokens']} tokens "
            f"(excluding label shift and padding). CSV: {run_dir / 'packing_token_stats.csv'}"
        )
    return run_dir, message


def main() -> None:
    """Run token statistics from a config or cache manifest."""
    import json

    os.environ.setdefault("TQDM_DISABLE", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache-manifest", type=Path, help="JSON list of {name, meta_dirs}; never rebuilds caches")
    source.add_argument("--config", type=Path, help="SFT config defining trainer, or build_stats_inputs(work_dir)")
    parser.add_argument(
        "--packing", action="store_true", help="Also measure packing/collator losses and subset token shares"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Existing work/output directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel subset readers in cache mode")
    args = parser.parse_args()
    # Avoid reporting replicated datasets once per rank.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Run this offline coordinator once, without torchrun; use --workers for parallel cache reads")
    if args.cache_manifest and args.packing:
        parser.error("--packing requires --config; length caches alone do not contain the packing configuration")
    seed = None
    packing = None
    if args.cache_manifest:
        manifest = json.loads(args.cache_manifest.read_text())
        for entry in manifest:
            entry["meta_dirs"] = [str(args.cache_manifest.parent / path) for path in entry["meta_dirs"]]
        rows = summarize_cache_manifest(manifest, args.workers)
        log = logging.getLogger("xtuner.token_stats")
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        from mmengine.runner import set_random_seed

        from xtuner.v1.utils import Config, get_logger, log_format

        log = get_logger()
        # Keep per-sample logs out of the report.
        log.remove()
        log.add(sys.stderr, format=log_format(), filter=lambda record: record["name"] == __name__)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config = Config.fromfile(args.config)
        factory = config.get("build_stats_inputs")
        trainer = config.get("trainer")
        if factory is None and trainer is None:
            parser.error("Config must define trainer or build_stats_inputs(work_dir)")
        seed = config.get("seed", 42) if factory is not None else trainer.seed
        if type(seed) is not int or not 0 <= seed < 2**32:
            parser.error("Config seed must be an integer in [0, 2**32)")
        set_random_seed(seed)
        inputs = factory(args.output_dir) if factory is not None else {"datasets": _build_training_datasets(trainer)}
        rows = summarize_datasets(inputs["datasets"])
        if args.packing or "packed_dataset" in inputs:
            loader = inputs.get("dataloader_config") if factory is not None else trainer.dataloader_cfg
            if loader is None:
                parser.error("Packing statistics require dataloader_config in build_stats_inputs()")
            if factory is None and trainer.sp_size > 1:
                # Match Trainer's forced padding without mutating the input config.
                loader = loader.model_copy(update={"pack_to_max_length": True})
            packed = inputs.get("packed_dataset")
            if packed is None:
                packed = _build_packed_dataset(inputs["datasets"], loader, seed)
            elif Counter(id(source) for source in inputs["datasets"]) != Counter(
                id(source) for source in _source_datasets(packed)
            ):
                raise ValueError("datasets and packed_dataset must share sampled sources")
            packing = summarize_packing(packed, loader, subset_rows=rows)
    _, message = save_reports(rows, args.output_dir, packing, seed=seed)
    log.info(message)


def _build_training_datasets(trainer: Any) -> list[Any]:
    from transformers import AutoTokenizer
    from xtuner.v1.datasets import DataloaderConfig, build_datasets
    from xtuner.v1.datasets.token_stats import TokenStatsConfig

    loader = trainer.dataloader_cfg
    if type(loader) is not DataloaderConfig or loader.pack_level not in ("none", "soft", "hard", "__legacy"):
        raise ValueError("Direct training-config statistics require a standard text SFT DataloaderConfig")
    # Match Trainer's precedence for the deprecated dataset_cfg field.
    configs = trainer.dataset_cfg if trainer.dataset_cfg is not None else loader.dataset_config_list
    if not configs or trainer.tokenizer_path is None:
        raise ValueError("Training config must provide datasets and tokenizer_path; otherwise use build_stats_inputs")
    if any(entry["dataset"].class_name != "JsonlDataset" for entry in configs):
        raise ValueError("Direct training-config statistics require text JsonlDataset sources")
    configs = [
        {
            **entry,
            "tokenize_fn": entry["tokenize_fn"]
            if isinstance(entry["tokenize_fn"], TokenStatsConfig)
            else TokenStatsConfig(entry["tokenize_fn"]),
        }
        for entry in configs
    ]
    tokenizer = AutoTokenizer.from_pretrained(trainer.tokenizer_path, trust_remote_code=True)
    return build_datasets(configs, tokenizer, tokenizer_hash=loader.tokenizer_hash)


def _build_packed_dataset(datasets: Sequence[Any], config: Any, seed: int) -> Any:
    from torch.utils.data import ConcatDataset

    from xtuner.v1.datasets.packing import ExpandSoftPackDataset, HardPackDataset, _LegacySoftPackDataset

    if config.collator != "sft_llm_collator":
        raise ValueError("Packing statistics require sft_llm_collator")
    if config.pack_level == "none":
        return ConcatDataset(datasets)
    options = {"pack_max_length": config.pack_max_length, "global_pack": config.global_pack, "seed": seed}
    if config.pack_level == "__legacy":
        return _LegacySoftPackDataset(datasets, **options)
    options.update(pack_workers=config.pack_workers, pack_chunk_size=config.pack_chunk_size)
    if config.pack_level == "soft":
        return ExpandSoftPackDataset(datasets, pack_extra_buffer_size=config.pack_extra_buffer_size, **options)
    if config.pack_level == "hard":
        return HardPackDataset(datasets, **options)
    raise ValueError("Packing statistics support text none/soft/hard/__legacy only")


def _source_datasets(dataset: Any) -> list[Any]:
    from torch.utils.data import ConcatDataset

    from xtuner.v1.datasets.packing import _LegacySoftPackDataset

    if isinstance(dataset, (ConcatDataset, _LegacySoftPackDataset)):
        return [source for child in dataset.datasets for source in _source_datasets(child)]
    return [dataset]


def _locate_source(dataset: Any, index: int) -> tuple[Any, int]:
    from torch.utils.data import ConcatDataset

    while isinstance(dataset, ConcatDataset):
        child = bisect_right(dataset.cumulative_sizes, index)
        index -= dataset.cumulative_sizes[child - 1] if child else 0
        dataset = dataset.datasets[child]
    return dataset, index


def _pack_sources(packed: Any, index: int) -> list[tuple[Any, int]]:
    from torch.utils.data import ConcatDataset

    from xtuner.v1.datasets.packing import HardPackDataset, _LegacySoftPackDataset

    if isinstance(packed, ConcatDataset):
        return [_locate_source(packed, index)]
    if isinstance(packed, HardPackDataset):
        infos = packed.pack_infos
        start = int(infos["indices_cu_len"][index - 1]) if index else 0
        end = int(infos["indices_cu_len"][index])
        indices = infos["indices"][start:end]
        dataset = packed.datasets[int(infos["dataset_id"][index])]
    elif isinstance(packed, _LegacySoftPackDataset):
        info = packed.pack_infos[index]
        indices = info["indices"]
        dataset = packed.datasets[info["dataset_id"]]
    else:
        raise ValueError("Packing provenance requires a standard text packed dataset or ConcatDataset")
    return [_locate_source(dataset, int(sample)) for sample in indices]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _distribution(values: np.ndarray, prefix: str) -> Row:
    median, p99 = np.percentile(values, [50, 99]) if len(values) else (None, None)
    return {
        f"{prefix}_total_tokens": int(values.sum()) if len(values) else None,
        f"{prefix}_mean_tokens": float(values.mean()) if len(values) else None,
        f"{prefix}_median_tokens": float(median) if len(values) else None,
        f"{prefix}_p99_tokens": float(p99) if len(values) else None,
        f"{prefix}_max_tokens": int(values.max()) if len(values) else None,
    }


def _summarize_cache_subset(job: tuple[str, list[str]]) -> Row:
    name, directories = job
    shards = []
    for directory in directories:
        path = Path(directory)
        if (path / "chunks.npy").exists():
            raise ValueError(f"Chunked caches have no original-document coverage: {path}")
        shard = {"num_tokens": np.load(path / "num_tokens.npy", allow_pickle=False)}
        for key in _LENGTH_KEYS[1:]:
            if (path / f"{key}.npy").exists():
                shard[key] = np.load(path / f"{key}.npy", allow_pickle=False)
        shards.append(shard)
    return summarize_samples(name, shards, "source_samples_before_filtering")


def _write_csv(path: Path, rows: Sequence[Row]) -> None:
    if not rows:
        raise ValueError("No subsets to report")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

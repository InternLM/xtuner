from __future__ import annotations

# %%
"""Notebook-style analyzer for XTuner memory-stage profile JSONL files.

Open this file as a Jupyter/VSCode notebook-style script and run cells from top
to bottom, or run it as a CLI to export CSV summaries.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# %%
# Jupyter parameters. Override these in a notebook cell for drill-down.
PROFILE_DIR = Path("work_dirs/sft_glm5p2/profile_30b_mtp_ep1_16k_alpaca_2step_gc_memstage_0707043733/memory_stage_profile")
RANKS: list[int] | None = [0]
SOURCES = ("active", "interval_alloc")
MIN_GIB = 0.1
TOP_N = 30
MAX_STACK_DEPTH = 8


# %%
def load_records(profile_dir: str | Path, ranks: Iterable[int] | None = None) -> list[dict[str, Any]]:
    profile_dir = Path(profile_dir)
    rank_filter = None if ranks is None else set(ranks)
    records: list[dict[str, Any]] = []
    for path in sorted(profile_dir.glob("rank*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if rank_filter is not None and int(record["rank"]) not in rank_filter:
                continue
            records.append(record)
    return records


def stage_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "rank": record["rank"],
                "step": record["step"],
                "micro_batch": record["micro_batch"],
                "stage": record["stage"],
                "allocated_gib": record["allocated_gib"],
                "reserved_gib": record["reserved_gib"],
                "max_allocated_gib": record["max_allocated_gib"],
                "max_reserved_gib": record["max_reserved_gib"],
                "active_gib": record["active"]["active_gib"],
                "inactive_gib": record["active"]["inactive_gib"],
                "interval_alloc_gib": record["interval"]["alloc_gib"],
                "interval_free_requested_gib": record["interval"]["free_requested_gib"],
                "delta_allocated_from_prev_gib": record["delta_allocated_from_prev_gib"],
                "active_top_coverage_gib": sum(item["gib"] for item in record["active"]["top_stacks"]),
                "interval_top_coverage_gib": sum(item["gib"] for item in record["interval"]["top_alloc_stacks"]),
            }
        )
    return pd.DataFrame(rows)


def _stack_text(item: dict[str, Any]) -> str:
    return "\n".join([item.get("frame", ""), *item.get("stack", [])])


def _is_xtuner_frame(frame: str) -> bool:
    return "xtuner/" in frame or frame.startswith("xtuner/")


def _normalize_frame(frame: str) -> str:
    marker = "xtuner/"
    if marker in frame:
        frame = frame[frame.index(marker) :]
    return frame


def _format_snapshot_frame(frame: dict[str, Any]) -> str:
    filename = str(frame.get("filename", ""))
    line = frame.get("line", 0)
    name = frame.get("name", "")
    return _normalize_frame(f"{filename}:{line}:{name}")


def _snapshot_item(size: int, frames: list[dict[str, Any]]) -> dict[str, Any]:
    stack = [_format_snapshot_frame(frame) for frame in frames]
    return {
        "frame": stack[0] if stack else "<unknown>",
        "stack": stack,
        "gib": size / 1024**3,
        "requested_gib": size / 1024**3,
        "count": 1,
    }


def _xtuner_call_path(item: dict[str, Any], max_depth: int = MAX_STACK_DEPTH) -> list[str]:
    frames = [_normalize_frame(frame) for frame in item.get("stack", []) if _is_xtuner_frame(str(frame))]
    if not frames and _is_xtuner_frame(str(item.get("frame", ""))):
        frames = [_normalize_frame(str(item["frame"]))]
    # PyTorch memory frames are direct allocation first, caller later. Reverse
    # the project-only frames to show outer XTuner call -> inner allocation site.
    return list(reversed(frames))[:max_depth]


def classify_stack(item: dict[str, Any], *, source: str, stage: str) -> tuple[str, str]:
    text = _stack_text(item)
    lower = text.lower()
    frame_lower = str(item.get("frame", "")).lower()

    # FSDP allocator snapshots do not record tensor dtype directly. The split
    # below is inferred from lifetime and stack: persistent <unknown> active
    # blocks are local optimizer-side FP32 shards, while FSDP unshard/all-gather
    # and gradient reduce paths are temporary BF16 communication buffers.
    if "step_optimizer" in lower:
        return "optimizer_states", "adamw_dtensor_state_or_update_buffer"
    if "_to_dtype_if_needed" in frame_lower or "cal_grad_norm" in frame_lower:
        return "fp32_sharded_gradients", "optimizer_fp32_grad_shards"
    if "foreach_reduce" in lower:
        return "bf16_reduce_scatter_gradients", "temporary_bf16_reduce_scatter_grads"
    if "_fully_shard" in lower or "all_gather" in lower or "unshard" in lower:
        return "bf16_all_gather_parameters", "temporary_bf16_all_gather_params"
    if "<unknown>" in lower and source in {"active", "peak_active"}:
        return "fp32_sharded_parameters", "optimizer_fp32_param_shards"

    if "ce_loss.py" in lower or "chunk_loss.py" in lower or "cross_entropy" in lower:
        return "activations", "lm_head_loss"
    if "tilelang_dsa_topk_indices" in lower or "dsa_topk_indices" in lower:
        return "activations", "dsa_indexer_topk_cache"
    if "sparse_mla" in lower or "dsa_mla.py" in lower:
        return "activations", "dsa_sparse_mla"
    if "grouped_gemm" in lower or "permute_unpermute" in lower or "native_swiglu" in lower:
        return "activations", "moe_grouped_gemm_dispatch"
    if "rms_norm" in lower:
        return "activations", "rms_norm"
    if "mtp_block" in lower:
        return "activations", "mtp_block"
    if "autograd/graph.py" in lower or "torch/utils/checkpoint.py" in lower:
        return "activations", "checkpoint_recompute_autograd"
    if "xtuner/" in lower:
        return "activations", "xtuner_other"
    if "<unknown>" in lower:
        return "allocator_or_unknown", "no_python_frame"
    return "other", "external_or_small"


def explode_stack_items(records: list[dict[str, Any]], source: str) -> pd.DataFrame:
    if source not in {"active", "interval_alloc", "interval_segment_alloc"}:
        raise ValueError(f"unknown source={source!r}")
    rows: list[dict[str, Any]] = []
    for record in records:
        if source == "active":
            items = record["active"]["top_stacks"]
        elif source == "interval_alloc":
            items = record["interval"]["top_alloc_stacks"]
        else:
            items = record["interval"]["top_segment_alloc_stacks"]

        for item in items:
            category, subcategory = classify_stack(item, source=source, stage=record["stage"])
            call_path = _xtuner_call_path(item)
            row: dict[str, Any] = {
                "rank": record["rank"],
                "step": record["step"],
                "micro_batch": record["micro_batch"],
                "stage": record["stage"],
                "source": source,
                "category": category,
                "subcategory": subcategory,
                "gib": item["gib"],
                "requested_gib": item.get("requested_gib", item["gib"]),
                "count": item["count"],
                "frame": item["frame"],
                "call_path": " -> ".join(call_path) if call_path else "<no_xtuner_frame>",
            }
            for level in range(MAX_STACK_DEPTH):
                row[f"level_{level}"] = call_path[level] if level < len(call_path) else ""
            rows.append(row)
    return pd.DataFrame(rows)


def _snapshot_path(profile_dir: Path, record: dict[str, Any]) -> Path:
    step_name = "none" if record["step"] is None else str(record["step"])
    micro_name = "none" if record["micro_batch"] is None else str(record["micro_batch"])
    name = f"rank{record['rank']}_step{step_name}_micro{micro_name}_{record['stage']}.pickle"
    return profile_dir / "snapshots" / name


def _load_snapshot(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _active_blocks(snapshot: dict[str, Any]) -> dict[int, dict[str, Any]]:
    blocks: dict[int, dict[str, Any]] = {}
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            if str(block.get("state", "")).startswith("active"):
                address = int(block["address"])
                size = int(block.get("requested_size", block.get("size", 0)))
                blocks[address] = {"size": size, "frames": block.get("frames") or []}
    return blocks


def _device_trace(snapshot: dict[str, Any], device: int) -> list[dict[str, Any]]:
    traces = snapshot.get("device_traces", [])
    return traces[device] if device < len(traces) else []


def _replay_peak_between(
    start_snapshot: dict[str, Any],
    end_snapshot: dict[str, Any],
    *,
    device: int,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    active = _active_blocks(start_snapshot)
    current = sum(block["size"] for block in active.values())
    peak = current
    peak_active = dict(active)
    peak_event: dict[str, Any] = {
        "peak_event_index": None,
        "peak_interval_event_index": None,
        "peak_event_action": "initial_active",
        "peak_event_size_gib": 0.0,
        "peak_event_time_us": None,
        "peak_event_since_start_ms": None,
        "peak_event_frame": "<initial_active>",
        "peak_event_call_path": "<initial_active>",
    }

    start_trace = _device_trace(start_snapshot, device)
    end_trace = _device_trace(end_snapshot, device)
    start_trace_len = len(start_trace)
    start_time_us = start_trace[-1].get("time_us") if start_trace else None
    for interval_index, event in enumerate(end_trace[start_trace_len:]):
        action = event.get("action")
        address = int(event.get("addr", 0))
        size = int(event.get("size", 0))
        if action == "alloc":
            active[address] = {"size": size, "frames": event.get("frames") or []}
            current += size
            if current > peak:
                peak = current
                peak_active = dict(active)
                event_item = _snapshot_item(size, event.get("frames") or [])
                call_path = _xtuner_call_path(event_item)
                event_time_us = event.get("time_us")
                peak_event = {
                    "peak_event_index": start_trace_len + interval_index,
                    "peak_interval_event_index": interval_index,
                    "peak_event_action": action,
                    "peak_event_size_gib": size / 1024**3,
                    "peak_event_time_us": event_time_us,
                    "peak_event_since_start_ms": None
                    if start_time_us is None or event_time_us is None
                    else (event_time_us - start_time_us) / 1000,
                    "peak_event_frame": event_item["frame"],
                    "peak_event_call_path": " -> ".join(call_path) if call_path else "<no_xtuner_frame>",
                }
        elif action == "free_requested":
            block = active.pop(address, None)
            current -= int(block["size"] if block is not None else size)

    peak_items = [_snapshot_item(block["size"], block["frames"]) for block in peak_active.values()]
    return peak / 1024**3, peak_items, peak_event


def replay_interval_peaks(profile_dir: str | Path, ranks: Iterable[int] | None = RANKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_dir = Path(profile_dir)
    records = load_records(profile_dir, ranks)
    if not (profile_dir / "snapshots").exists():
        return pd.DataFrame(), pd.DataFrame()

    peak_rows: list[dict[str, Any]] = []
    peak_stack_rows: list[dict[str, Any]] = []
    previous_by_rank: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}

    for record in records:
        path = _snapshot_path(profile_dir, record)
        if not path.exists():
            continue
        snapshot = _load_snapshot(path)
        rank = int(record["rank"])
        previous = previous_by_rank.get(rank)
        if previous is None:
            previous_by_rank[rank] = (record, snapshot)
            continue

        previous_record, previous_snapshot = previous
        peak_gib, peak_items, peak_event = _replay_peak_between(
            previous_snapshot,
            snapshot,
            device=int(record["device"]),
        )
        peak_rows.append(
            {
                "rank": rank,
                "step": record["step"],
                "micro_batch": record["micro_batch"],
                "stage": record["stage"],
                "start_step": previous_record["step"],
                "start_micro_batch": previous_record["micro_batch"],
                "start_stage": previous_record["stage"],
                "peak_allocated_gib": peak_gib,
                **peak_event,
            }
        )
        for item in peak_items:
            category, subcategory = classify_stack(item, source="peak_active", stage=record["stage"])
            call_path = _xtuner_call_path(item)
            row: dict[str, Any] = {
                "rank": rank,
                "step": record["step"],
                "micro_batch": record["micro_batch"],
                "stage": record["stage"],
                "source": "peak_active",
                "category": category,
                "subcategory": subcategory,
                "gib": item["gib"],
                "count": 1,
                "frame": item["frame"],
                "call_path": " -> ".join(call_path) if call_path else "<no_xtuner_frame>",
            }
            for level in range(MAX_STACK_DEPTH):
                row[f"level_{level}"] = call_path[level] if level < len(call_path) else ""
            peak_stack_rows.append(row)

        previous_by_rank[rank] = (record, snapshot)

    return pd.DataFrame(peak_rows), pd.DataFrame(peak_stack_rows)


def category_summary(rows: pd.DataFrame, *, min_gib: float = MIN_GIB) -> pd.DataFrame:
    if rows.empty:
        return rows
    grouped = (
        rows.groupby(
            ["rank", "step", "micro_batch", "stage", "source", "category", "subcategory"],
            as_index=False,
            dropna=False,
        )
        .agg(gib=("gib", "sum"), count=("count", "sum"))
        .sort_values(
            ["step", "micro_batch", "stage", "source", "gib"],
            ascending=[True, True, True, True, False],
        )
    )
    return grouped[grouped["gib"] >= min_gib]


def hierarchy_summary(rows: pd.DataFrame, *, min_gib: float = MIN_GIB, max_depth: int = MAX_STACK_DEPTH) -> pd.DataFrame:
    if rows.empty:
        return rows
    levels = [f"level_{idx}" for idx in range(max_depth)]
    grouped = (
        rows.groupby(
            ["rank", "step", "micro_batch", "stage", "source", "category", "subcategory", *levels],
            as_index=False,
            dropna=False,
        )
        .agg(gib=("gib", "sum"), count=("count", "sum"))
        .sort_values("gib", ascending=False)
    )
    return grouped[grouped["gib"] >= min_gib]


def filter_rows(
    rows: pd.DataFrame,
    *,
    step: int | None = None,
    micro_batch: int | None = None,
    stage: str | None = None,
    source: str | None = None,
    category: str | None = None,
    min_gib: float = MIN_GIB,
) -> pd.DataFrame:
    result = rows
    if step is not None:
        result = result[result["step"] == step]
    if micro_batch is not None and "micro_batch" in result.columns:
        result = result[result["micro_batch"] == micro_batch]
    if stage is not None:
        result = result[result["stage"] == stage]
    if source is not None:
        result = result[result["source"] == source]
    if category is not None:
        result = result[result["category"] == category]
    return result[result["gib"] >= min_gib].sort_values("gib", ascending=False)


_SUBCATEGORY_ABBR = {
    "adamw_dtensor_state_or_update_buffer": "adamw_state",
    "optimizer_fp32_grad_shards": "fp32_grad",
    "optimizer_fp32_param_shards": "fp32_param",
    "temporary_bf16_all_gather_params": "bf16_gather_param",
    "temporary_bf16_reduce_scatter_grads": "bf16_reduce_grad",
    "lm_head_loss": "lm_loss",
    "dsa_indexer_topk_cache": "dsa_topk",
    "dsa_sparse_mla": "dsa_mla",
    "moe_grouped_gemm_dispatch": "moe_gemm",
    "rms_norm": "rms_norm",
    "mtp_block": "mtp",
    "checkpoint_recompute_autograd": "ckpt_recompute",
    "xtuner_other": "xtuner_other",
    "no_python_frame": "unknown",
    "external_or_small": "external",
}


def _abbr_subcategory(subcategory: str) -> str:
    if subcategory in _SUBCATEGORY_ABBR:
        return _SUBCATEGORY_ABBR[subcategory]
    parts = [part for part in str(subcategory).split("_") if part]
    if not parts:
        return "unknown"
    return "_".join(parts[:3])[:24]


def _abbr_frame(frame: str) -> str:
    frame = str(frame)
    if not frame or frame == "<no_xtuner_frame>":
        return frame
    pieces = frame.split(":")
    path = pieces[0]
    line = pieces[1] if len(pieces) > 1 else ""
    name = pieces[2] if len(pieces) > 2 else ""
    path_parts = path.split("/")
    short_path = "/".join(path_parts[-2:]) if len(path_parts) >= 2 else path
    suffix = f":{line}" if line else ""
    if name:
        suffix += f":{name}"
    return f"{short_path}{suffix}"


def _abbr_call_path(call_path: str) -> str:
    frames = [frame for frame in str(call_path).split(" -> ") if frame]
    if not frames:
        return "<no_xtuner_frame>"
    short_frames = [_abbr_frame(frame) for frame in frames]
    if len(short_frames) <= 2:
        return " -> ".join(short_frames)
    return f"{short_frames[0]} -> ... -> {short_frames[-1]}"


def hierarchy_label_table(
    rows: pd.DataFrame,
    *,
    step: int,
    micro_batch: int | None = None,
    stage: str,
    source: str = "active",
    category: str | None = "activations",
    min_gib: float = MIN_GIB,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Return the y-axis label mapping used by ``plot_hierarchy_bar``.

    The plot intentionally uses compact ``#NN short_label`` y tick labels so
    long call paths do not dominate the figure. This table preserves the full
    subcategory and call path mapping for interpretation.
    """

    data = filter_rows(
        rows,
        step=step,
        micro_batch=micro_batch,
        stage=stage,
        source=source,
        category=category,
        min_gib=min_gib,
    ).head(top_n)
    if data.empty:
        return data
    result = data.reset_index(drop=True).copy()
    result["label_id"] = [f"#{idx:02d}" for idx in range(1, len(result) + 1)]
    result["short_path"] = result["call_path"].map(_abbr_call_path)
    result["short_label"] = result["subcategory"].map(_abbr_subcategory) + " | " + result["short_path"]
    result["plot_label"] = result["label_id"] + " " + result["subcategory"].map(_abbr_subcategory)
    return result[
        [
            "label_id",
            "plot_label",
            "short_label",
            "category",
            "subcategory",
            "gib",
            "count",
            "call_path",
        ]
    ]


# %%
def plot_stage_memory(stage_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    data = stage_df.sort_values(["step", "rank"]).copy()
    data["point"] = data["step"].astype(str) + ":" + data["stage"]
    ax = data.plot(
        x="point",
        y=["allocated_gib", "reserved_gib", "max_allocated_gib"],
        marker="o",
        figsize=(max(12, len(data) * 0.7), 4),
    )
    ax.set_ylabel("GiB")
    ax.set_title("Stage memory")
    # Matplotlib may thin categorical ticks when there are many stages. Keep
    # every sampled stage visible because adjacent stage names are meaningful.
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data["point"].tolist(), rotation=35, ha="right")
    plt.tight_layout()
    return ax


def plot_category_bar(rows: pd.DataFrame, *, step: int, stage: str, source: str = "active", min_gib: float = MIN_GIB):
    import matplotlib.pyplot as plt

    data = filter_rows(rows, step=step, stage=stage, source=source, min_gib=min_gib)
    if data.empty:
        raise ValueError("no rows after filtering")
    grouped = data.groupby(["category", "subcategory"], as_index=False)["gib"].sum().sort_values("gib", ascending=True)
    labels = grouped["category"] + "/" + grouped["subcategory"]
    ax = grouped.plot.barh(x=None, y="gib", legend=False, figsize=(10, max(4, len(grouped) * 0.35)))
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("GiB")
    ax.set_title(f"{source} memory by category: step={step}, stage={stage}")
    plt.tight_layout()
    return ax


def plot_hierarchy_bar(
    rows: pd.DataFrame,
    *,
    step: int,
    micro_batch: int | None = None,
    stage: str,
    source: str = "active",
    category: str | None = "activations",
    min_gib: float = MIN_GIB,
    top_n: int = TOP_N,
):
    import matplotlib.pyplot as plt

    labels = hierarchy_label_table(
        rows,
        step=step,
        micro_batch=micro_batch,
        stage=stage,
        source=source,
        category=category,
        min_gib=min_gib,
        top_n=top_n,
    )
    if labels.empty:
        raise ValueError("no rows after filtering")
    plot_data = labels.sort_values("gib")
    ax = plot_data.plot.barh(x=None, y="gib", legend=False, figsize=(12, max(4, len(plot_data) * 0.45)))
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data["plot_label"])
    ax.set_xlabel("GiB")
    if micro_batch is None:
        point = f"step={step}, stage={stage}"
    else:
        point = f"step={step}, micro_batch={micro_batch}, stage={stage}"
    ax.set_title(f"{source} hierarchy: {point}, category={category}")
    ax.xtuner_label_map = labels
    plt.tight_layout()
    return ax


# %%
def build_analysis_tables(profile_dir: str | Path, ranks: Iterable[int] | None = RANKS) -> dict[str, pd.DataFrame]:
    records = load_records(profile_dir, ranks)
    stage_df = stage_summary(records)
    stack_rows = pd.concat([explode_stack_items(records, source) for source in SOURCES], ignore_index=True)
    tables = {
        "stages": stage_df,
        "stacks": stack_rows,
        "categories": category_summary(stack_rows),
        "hierarchy": hierarchy_summary(stack_rows),
    }
    peak_df, peak_rows = replay_interval_peaks(profile_dir, ranks)
    if not peak_df.empty:
        tables["peaks"] = peak_df
        tables["peak_stacks"] = peak_rows
        tables["peak_categories"] = category_summary(peak_rows, min_gib=MIN_GIB)
        tables["peak_hierarchy"] = hierarchy_summary(peak_rows, min_gib=MIN_GIB)
    return tables


def export_tables(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", nargs="?", default=str(PROFILE_DIR))
    parser.add_argument("--ranks", default="0", help="'all' or comma-separated rank ids")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    ranks = None if args.ranks == "all" else [int(rank) for rank in args.ranks.split(",") if rank]
    tables = build_analysis_tables(args.profile_dir, ranks)
    print("\n[stages]")
    print(tables["stages"].to_string(index=False))
    print("\n[categories]")
    print(tables["categories"].to_string(index=False))
    if "peaks" in tables:
        print("\n[peaks]")
        print(tables["peaks"].to_string(index=False))
        print("\n[peak_categories]")
        print(tables["peak_categories"].to_string(index=False))
    if args.out_dir:
        export_tables(tables, args.out_dir)
        print(f"\nExported CSV tables to {args.out_dir}")

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import prod
from pathlib import Path
from typing import Any

import pandas as pd
from safetensors import safe_open


GIB = 1024**3


def _to_gib(num_bytes: int | float) -> float:
    return float(num_bytes) / GIB


def _load_config(model_dir: str | Path) -> dict[str, Any]:
    with (Path(model_dir) / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_tensor_shapes(model_dir: str | Path):
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                shape = f.get_slice(name).get_shape()
                yield name, tuple(shape), prod(shape)


def _tensor_group(name: str, config: dict[str, Any]) -> tuple[str, str, int | None]:
    num_hidden_layers = int(config.get("num_hidden_layers", 0))
    if name.startswith("model.layers."):
        layer_idx = int(name.split(".")[2])
        if layer_idx < num_hidden_layers:
            return f"layer_{layer_idx}", "main_layer", layer_idx
        return f"mtp_layer_{layer_idx}", "mtp_layer", layer_idx
    if name.startswith("model.embed_tokens"):
        return "embed_tokens", "embedding", None
    if name.startswith("lm_head"):
        return "lm_head", "lm_head", None
    return name.rsplit(".", 1)[0], "other", None


def build_weight_tables(
    model_dir: str | Path,
    *,
    world_size: int = 8,
    optimizer_state_count: int = 2,
) -> dict[str, pd.DataFrame]:
    """Estimate weight-related memory from HF safetensor shapes.

    The calculation is shape-only and does not load tensor contents. It assumes
    bf16 model/all-gather buffers, fp32 local optimizer-side parameter shards,
    fp32 local gradient shards, and AdamW-style fp32 ``m``/``v`` states.
    """

    config = _load_config(model_dir)
    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    for name, shape, numel in _iter_tensor_shapes(model_dir):
        group, group_type, layer_idx = _tensor_group(name, config)
        item = grouped.setdefault(
            group,
            {
                "group": group,
                "group_type": group_type,
                "layer_idx": layer_idx,
                "params": 0,
                "tensor_count": 0,
            },
        )
        item["params"] += numel
        item["tensor_count"] += 1
        rows.append(
            {
                "name": name,
                "shape": str(shape),
                "params": numel,
                "bf16_full_gib": _to_gib(numel * 2),
                "fp32_shard_gib": _to_gib(numel * 4 / world_size),
                "group": group,
                "group_type": group_type,
                "layer_idx": layer_idx,
            }
        )

    tensors = pd.DataFrame(rows)
    groups = pd.DataFrame(grouped.values()).sort_values(["group_type", "layer_idx", "group"], na_position="last")
    groups["params_b"] = groups["params"] / 1e9
    groups["bf16_full_gib"] = groups["params"].map(lambda value: _to_gib(value * 2))
    groups["fp32_shard_gib"] = groups["params"].map(lambda value: _to_gib(value * 4 / world_size))

    total_params = int(tensors["params"].sum())
    summary = pd.DataFrame(
        [
            {
                "item": "total_params",
                "params_b": total_params / 1e9,
                "estimated_gib": None,
                "formula": "sum(safetensors numel)",
            },
            {
                "item": "bf16_full_model",
                "params_b": total_params / 1e9,
                "estimated_gib": _to_gib(total_params * 2),
                "formula": "total_params * 2 bytes",
            },
            {
                "item": "fp32_sharded_parameters",
                "params_b": total_params / 1e9,
                "estimated_gib": _to_gib(total_params * 4 / world_size),
                "formula": f"total_params * 4 bytes / world_size({world_size})",
            },
            {
                "item": "fp32_sharded_gradients",
                "params_b": total_params / 1e9,
                "estimated_gib": _to_gib(total_params * 4 / world_size),
                "formula": f"same as fp32_sharded_parameters",
            },
            {
                "item": "optimizer_states",
                "params_b": total_params / 1e9,
                "estimated_gib": _to_gib(total_params * 4 * optimizer_state_count / world_size),
                "formula": f"AdamW m/v: total_params * 4 bytes * {optimizer_state_count} / world_size({world_size})",
            },
        ]
    )

    first_k_dense_replace = int(config.get("first_k_dense_replace", 0))
    num_hidden_layers = int(config.get("num_hidden_layers", 0))
    main_moe = groups[
        (groups["group_type"] == "main_layer")
        & groups["layer_idx"].notna()
        & (groups["layer_idx"] >= first_k_dense_replace)
        & (groups["layer_idx"] < num_hidden_layers)
    ]
    dense = groups[
        (groups["group_type"] == "main_layer")
        & groups["layer_idx"].notna()
        & (groups["layer_idx"] < first_k_dense_replace)
    ]
    mtp = groups[groups["group_type"] == "mtp_layer"]

    main_moe_bf16 = float(main_moe["bf16_full_gib"].max()) if not main_moe.empty else 0.0
    dense_bf16 = float(dense["bf16_full_gib"].max()) if not dense.empty else 0.0
    mtp_bf16 = float(mtp["bf16_full_gib"].max()) if not mtp.empty else 0.0
    lm_head_bf16 = float(groups.loc[groups["group"] == "lm_head", "bf16_full_gib"].sum())

    communication = pd.DataFrame(
        [
            {
                "item": "single_main_moe_layer_bf16",
                "estimated_gib": main_moe_bf16,
                "formula": "largest main MoE FSDP unit full bf16 params",
            },
            {
                "item": "single_mtp_layer_bf16",
                "estimated_gib": mtp_bf16,
                "formula": "largest MTP FSDP unit full bf16 params",
            },
            {
                "item": "single_dense_layer_bf16",
                "estimated_gib": dense_bf16,
                "formula": "largest dense FSDP unit full bf16 params",
            },
            {
                "item": "lm_head_bf16",
                "estimated_gib": lm_head_bf16,
                "formula": "lm_head full bf16 params",
            },
            {
                "item": "expected_backward_peak_bf16_param_overlap",
                "estimated_gib": main_moe_bf16 + lm_head_bf16 + dense_bf16,
                "formula": "main MoE + lm_head + dense layer overlap observed in backward peak",
            },
            {
                "item": "expected_forward_peak_bf16_param_overlap",
                "estimated_gib": 2 * main_moe_bf16 + mtp_bf16,
                "formula": "two main MoE layers + MTP layer overlap observed in forward peak",
            },
            {
                "item": "expected_single_moe_bf16_reduce_scatter_grad",
                "estimated_gib": main_moe_bf16,
                "formula": "one main MoE layer full bf16 gradient communication buffer",
            },
        ]
    )

    return {
        "summary": summary,
        "groups": groups,
        "tensors": tensors,
        "communication": communication,
    }


def build_profile_alignment(
    tables: dict[str, pd.DataFrame],
    *,
    analysis_dir: str | Path,
) -> pd.DataFrame:
    analysis_dir = Path(analysis_dir)
    categories = pd.read_csv(analysis_dir / "categories.csv")
    peak_categories = pd.read_csv(analysis_dir / "peak_categories.csv")

    summary = tables["summary"].set_index("item")
    communication = tables["communication"].set_index("item")

    def category_sum(df: pd.DataFrame, *, step: int, stage: str, source: str, category: str) -> float:
        rows = df[(df["step"] == step) & (df["stage"] == stage) & (df["source"] == source) & (df["category"] == category)]
        return float(rows["gib"].sum())

    rows = [
        {
            "item": "fp32_sharded_parameters steady",
            "estimated_gib": summary.loc["fp32_sharded_parameters", "estimated_gib"],
            "profile_gib": category_sum(categories, step=2, stage="forward_start", source="active", category="fp32_sharded_parameters"),
            "profile_point": "categories: step2 forward_start active",
        },
        {
            "item": "fp32_sharded_gradients complete",
            "estimated_gib": summary.loc["fp32_sharded_gradients", "estimated_gib"],
            "profile_gib": category_sum(categories, step=2, stage="optimizer_step_start", source="active", category="fp32_sharded_gradients"),
            "profile_point": "categories: step2 optimizer_step_start active",
        },
        {
            "item": "optimizer_states",
            "estimated_gib": summary.loc["optimizer_states", "estimated_gib"],
            "profile_gib": category_sum(categories, step=2, stage="forward_start", source="active", category="optimizer_states"),
            "profile_point": "categories: step2 forward_start active",
        },
        {
            "item": "bf16 reduce-scatter one MoE layer",
            "estimated_gib": communication.loc["expected_single_moe_bf16_reduce_scatter_grad", "estimated_gib"],
            "profile_gib": category_sum(
                peak_categories,
                step=2,
                stage="backward_end",
                source="peak_active",
                category="bf16_reduce_scatter_gradients",
            ),
            "profile_point": "peak_categories: step2 backward_end peak_active",
        },
        {
            "item": "bf16 all-gather backward overlap",
            "estimated_gib": communication.loc["expected_backward_peak_bf16_param_overlap", "estimated_gib"],
            "profile_gib": category_sum(
                peak_categories,
                step=2,
                stage="backward_end",
                source="peak_active",
                category="bf16_all_gather_parameters",
            ),
            "profile_point": "peak_categories: step2 backward_end peak_active",
        },
        {
            "item": "bf16 all-gather forward overlap",
            "estimated_gib": communication.loc["expected_forward_peak_bf16_param_overlap", "estimated_gib"],
            "profile_gib": category_sum(
                peak_categories,
                step=2,
                stage="forward_end",
                source="peak_active",
                category="bf16_all_gather_parameters",
            ),
            "profile_point": "peak_categories: step2 forward_end peak_active",
        },
    ]
    result = pd.DataFrame(rows)
    result["diff_gib"] = result["profile_gib"] - result["estimated_gib"]
    result["diff_pct"] = result["diff_gib"] / result["estimated_gib"] * 100
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()

    tables = build_weight_tables(args.model_dir, world_size=args.world_size)
    for name in ["summary", "communication", "groups"]:
        print(f"\n[{name}]")
        print(tables[name].to_string(index=False))
    if args.analysis_dir is not None:
        print("\n[profile_alignment]")
        print(build_profile_alignment(tables, analysis_dir=args.analysis_dir).to_string(index=False))


if __name__ == "__main__":
    main()

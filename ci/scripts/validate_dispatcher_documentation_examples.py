from __future__ import annotations

import argparse
import json
from fractions import Fraction
from typing import Any


K = 2
EP_EXPERTS = {
    0: (0, 1, 2),
    1: (3, 4, 5),
}
SOURCE_TOKENS = {
    "ep0": ("A0", "A1", "A2", "A3"),
    "ep1": ("B0", "B1", "B2", "B3"),
}
TOKEN_VALUE = {
    "A0": 10,
    "A1": 11,
    "A2": 12,
    "A3": 13,
    "B0": 20,
    "B1": 21,
    "B2": 22,
    "B3": 23,
}
TOPK_IDS = {
    "A0": (0, 4),
    "A1": (3, 1),
    "A2": (2, 5),
    "A3": (4, 0),
    "B0": (1, 3),
    "B1": (4, 2),
    "B2": (5, 0),
    "B3": (3, 1),
}
TOPK_WEIGHTS = {
    "A0": (Fraction(1, 4), Fraction(3, 4)),
    "A1": (Fraction(2, 5), Fraction(3, 5)),
    "A2": (Fraction(7, 10), Fraction(3, 10)),
    "A3": (Fraction(4, 5), Fraction(1, 5)),
    "B0": (Fraction(1, 5), Fraction(4, 5)),
    "B1": (Fraction(1, 2), Fraction(1, 2)),
    "B2": (Fraction(9, 10), Fraction(1, 10)),
    "B3": (Fraction(7, 20), Fraction(13, 20)),
}
FORWARD_ORDER = [
    "DeepEP dispatch receives source-token rows",
    "TP AllGather hidden, topK ids, and topK weights",
    "dispatch_postprocess builds local route-copy layout",
    "local experts produce ExpertTP partial outputs",
    "combine_preprocess performs Expert-side topK folding",
    "TP ReduceScatterRowsSum returns each TP rank source-token slice",
    "DeepEP combine sends reduced source-token rows back",
]


def _number(value: Fraction | int) -> int | float:
    if isinstance(value, int) or value.denominator == 1:
        return int(value)
    return float(value)


def _numbers(values: list[Fraction] | tuple[Fraction, ...]) -> list[int | float]:
    return [_number(value) for value in values]


def _matrix_numbers(values: list[list[Fraction]]) -> list[list[int | float]]:
    return [_numbers(row) for row in values]


def _local_expert_id(global_expert: int, ep_rank: int) -> int:
    if global_expert not in EP_EXPERTS[ep_rank]:
        return -1
    return global_expert - min(EP_EXPERTS[ep_rank])


def _source_preprocess(source_rank: str) -> dict[str, Any]:
    tokens = SOURCE_TOKENS[source_rank]
    flat_copies = []
    for slot in range(K):
        for token_index, token in enumerate(tokens):
            flat_copies.append(
                {
                    "flat_pos": slot * len(tokens) + token_index,
                    "source_rank": source_rank,
                    "source_row": token_index,
                    "token": token,
                    "global_expert": TOPK_IDS[token][slot],
                    "topk_slot": slot,
                }
            )

    sorted_copies = sorted(flat_copies, key=lambda row: (row["global_expert"], row["source_row"]))
    row_id_map = [-1] * len(flat_copies)
    for sorted_row, copy in enumerate(sorted_copies):
        row_id_map[copy["flat_pos"]] = sorted_row

    return {
        "tokens": [row["token"] for row in sorted_copies],
        "global_experts": [row["global_expert"] for row in sorted_copies],
        "row_id_map": row_id_map,
        "rows": sorted_copies,
    }


def _all2all_dispatch_rows(preprocessed_sources: dict[str, dict[str, Any]], target_ep_rank: int) -> list[dict[str, Any]]:
    rows = []
    for source_rank in ("ep0", "ep1"):
        for row in preprocessed_sources[source_rank]["rows"]:
            if row["global_expert"] not in EP_EXPERTS[target_ep_rank]:
                continue
            rows.append(
                {
                    **row,
                    "target_ep_rank": target_ep_rank,
                    "local_expert": _local_expert_id(row["global_expert"], target_ep_rank),
                }
            )
    return rows


def _permute_route_rows_by_local_expert(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_input_indices = sorted(range(len(rows)), key=lambda index: (rows[index]["local_expert"], index))
    row_ids_map = [-1] * len(rows)
    post_rows = []
    for post_row, input_index in enumerate(sorted_input_indices):
        row_ids_map[input_index] = post_row
        post_rows.append(rows[input_index])

    tokens_per_expert = [0, 0, 0]
    for row in post_rows:
        tokens_per_expert[row["local_expert"]] += 1

    return {
        "tokens": [row["token"] for row in post_rows],
        "local_experts": [row["local_expert"] for row in post_rows],
        "row_ids_map": row_ids_map,
        "tokens_per_expert": tokens_per_expert,
        "rows": post_rows,
    }


def _received_rows_for_ep(ep_rank: int) -> list[dict[str, Any]]:
    rows = []
    all_tokens = SOURCE_TOKENS["ep0"] + SOURCE_TOKENS["ep1"]
    for token in all_tokens:
        topk_ids = []
        topk_weights = []
        for slot, global_expert in enumerate(TOPK_IDS[token]):
            local_expert = _local_expert_id(global_expert, ep_rank)
            topk_ids.append(local_expert)
            topk_weights.append(TOPK_WEIGHTS[token][slot] if local_expert >= 0 else Fraction(0))
        if any(expert >= 0 for expert in topk_ids):
            rows.append(
                {
                    "token": token,
                    "hidden": TOKEN_VALUE[token],
                    "topk_ids": topk_ids,
                    "topk_weights": topk_weights,
                }
            )
    return rows


def _local_route_copy_layout(received_rows: list[dict[str, Any]]) -> dict[str, Any]:
    route_copies = []
    row_count = len(received_rows)
    for slot in range(K):
        for received_row, row in enumerate(received_rows):
            local_expert = row["topk_ids"][slot]
            if local_expert < 0:
                continue
            route_copies.append(
                {
                    "flat_pos": slot * row_count + received_row,
                    "received_row": received_row,
                    "topk_slot": slot,
                    "token": row["token"],
                    "hidden": row["hidden"],
                    "local_expert": local_expert,
                    "topk_weight": row["topk_weights"][slot],
                }
            )

    # 中文注释：DeepEP dispatch 收到的是 source-token rows；这里才展开成 expert route-copy rows。
    post_rows = sorted(route_copies, key=lambda row: (row["local_expert"], row["received_row"]))
    row_ids_map = [-1] * (row_count * K)
    for post_row, row in enumerate(post_rows):
        row_ids_map[row["flat_pos"]] = post_row

    tokens_per_expert = [0, 0, 0]
    for row in post_rows:
        tokens_per_expert[row["local_expert"]] += 1

    return {
        "tokens": [row["token"] for row in post_rows],
        "local_experts": [row["local_expert"] for row in post_rows],
        "row_ids_map": row_ids_map,
        "tokens_per_expert": tokens_per_expert,
        "rows": post_rows,
    }


def _fold_topk(
    *,
    route_outputs: list[Fraction],
    row_ids_map: list[int],
    received_rows: list[dict[str, Any]],
) -> list[Fraction]:
    row_count = len(received_rows)
    folded = [Fraction(0) for _ in range(row_count)]
    for flat_pos, post_row in enumerate(row_ids_map):
        if post_row < 0:
            continue
        slot = flat_pos // row_count
        received_row = flat_pos % row_count
        folded[received_row] += route_outputs[post_row] * received_rows[received_row]["topk_weights"][slot]
    return folded


def validate_all2all_example() -> dict[str, Any]:
    preprocessed = {source_rank: _source_preprocess(source_rank) for source_rank in SOURCE_TOKENS}
    assert preprocessed["ep0"]["row_id_map"] == [0, 4, 3, 6, 5, 2, 7, 1]
    assert preprocessed["ep1"]["row_id_map"] == [1, 6, 7, 5, 4, 3, 0, 2]

    dispatched_ep0 = _all2all_dispatch_rows(preprocessed, target_ep_rank=0)
    dispatched_ep1 = _all2all_dispatch_rows(preprocessed, target_ep_rank=1)
    assert [row["token"] for row in dispatched_ep0] == ["A0", "A3", "A1", "A2", "B2", "B0", "B3", "B1"]
    assert [row["token"] for row in dispatched_ep1] == ["A1", "A0", "A3", "A2", "B0", "B3", "B1", "B2"]

    post_ep0 = _permute_route_rows_by_local_expert(dispatched_ep0)
    post_ep1 = _permute_route_rows_by_local_expert(dispatched_ep1)
    assert post_ep0["tokens"] == ["A0", "A3", "B2", "A1", "B0", "B3", "A2", "B1"]
    assert post_ep1["tokens"] == ["A1", "B0", "B3", "A0", "A3", "B1", "A2", "B2"]
    assert post_ep0["row_ids_map"] == [0, 1, 3, 6, 2, 4, 5, 7]
    assert post_ep1["row_ids_map"] == [0, 3, 4, 6, 1, 2, 5, 7]

    return {
        "passed": True,
        "ep0_dispatch_rows": [row["token"] for row in dispatched_ep0],
        "ep1_dispatch_rows": [row["token"] for row in dispatched_ep1],
        "ep0_tokens_per_expert": post_ep0["tokens_per_expert"],
        "ep1_tokens_per_expert": post_ep1["tokens_per_expert"],
    }


def validate_deepep_example() -> dict[str, Any]:
    received_by_ep = {ep_rank: _received_rows_for_ep(ep_rank) for ep_rank in EP_EXPERTS}
    layouts = {ep_rank: _local_route_copy_layout(rows) for ep_rank, rows in received_by_ep.items()}
    assert layouts[0]["tokens"] == ["A0", "A3", "B2", "A1", "B0", "B3", "A2", "B1"]
    assert layouts[1]["tokens"] == ["A1", "B0", "B3", "A0", "A3", "B1", "A2", "B2"]
    assert layouts[0]["tokens_per_expert"] == [3, 3, 2]
    assert layouts[1]["tokens_per_expert"] == [3, 3, 2]

    pre_combined_by_ep: dict[int, list[Fraction]] = {}
    for ep_rank, layout in layouts.items():
        route_outputs = [
            Fraction(row["hidden"] + (row["local_expert"] + min(EP_EXPERTS[ep_rank])) * 100)
            for row in layout["rows"]
        ]
        pre_combined_by_ep[ep_rank] = _fold_topk(
            route_outputs=route_outputs,
            row_ids_map=layout["row_ids_map"],
            received_rows=received_by_ep[ep_rank],
        )

    expected_ep0 = [
        Fraction(5, 2),
        Fraction(333, 5),
        Fraction(742, 5),
        Fraction(13, 5),
        Fraction(24),
        Fraction(221, 2),
        Fraction(11, 5),
        Fraction(1599, 20),
    ]
    expected_ep1 = [
        Fraction(615, 2),
        Fraction(622, 5),
        Fraction(768, 5),
        Fraction(1652, 5),
        Fraction(256),
        Fraction(421, 2),
        Fraction(2349, 5),
        Fraction(2261, 20),
    ]
    assert pre_combined_by_ep[0] == expected_ep0
    assert pre_combined_by_ep[1] == expected_ep1

    source_ep0 = [pre_combined_by_ep[0][i] + pre_combined_by_ep[1][i] for i in range(4)]
    source_ep1 = [pre_combined_by_ep[0][i] + pre_combined_by_ep[1][i] for i in range(4, 8)]
    assert source_ep0 == [Fraction(310), Fraction(191), Fraction(302), Fraction(333)]
    assert source_ep1 == [Fraction(280), Fraction(321), Fraction(472), Fraction(193)]

    return {
        "passed": True,
        "ep0_pre_combined": _numbers(pre_combined_by_ep[0]),
        "ep1_pre_combined": _numbers(pre_combined_by_ep[1]),
        "source_outputs": {
            "ep0": _numbers(source_ep0),
            "ep1": _numbers(source_ep1),
        },
    }


def validate_deepep_expert_tp_example() -> dict[str, Any]:
    received_rows_by_tp_rank = [
        [
            {"token": "S0", "hidden": 10, "topk_ids": [0, 1], "topk_weights": [Fraction(1, 4), Fraction(3, 4)]},
            {"token": "S1", "hidden": 20, "topk_ids": [2, -1], "topk_weights": [Fraction(3, 5), Fraction(0)]},
            {"token": "S2", "hidden": 30, "topk_ids": [-1, 0], "topk_weights": [Fraction(0), Fraction(2, 5)]},
        ],
        [
            {"token": "S3", "hidden": 40, "topk_ids": [1, 2], "topk_weights": [Fraction(3, 10), Fraction(7, 10)]},
            {"token": "S4", "hidden": 50, "topk_ids": [-1, 1], "topk_weights": [Fraction(0), Fraction(1, 2)]},
        ],
    ]
    tp_rank_row_counts = [len(rows) for rows in received_rows_by_tp_rank]
    gathered_rows = [row for rows in received_rows_by_tp_rank for row in rows]
    gathered_topk_ids = [row["topk_ids"] for row in gathered_rows]
    gathered_topk_weights = [row["topk_weights"] for row in gathered_rows]

    assert tp_rank_row_counts == [3, 2]
    assert gathered_topk_ids == [[0, 1], [2, -1], [-1, 0], [1, 2], [-1, 1]]

    layout = _local_route_copy_layout(gathered_rows)
    assert len(gathered_rows) == 5
    assert len(layout["rows"]) == 7
    assert layout["tokens_per_expert"] == [2, 3, 2]
    assert layout["tokens"] == ["S0", "S2", "S0", "S3", "S4", "S1", "S3"]

    # 中文注释：两个 ExpertTP rank 分别给出 row-parallel partial；
    # 先在 expert 侧按 topK fold，再由 ReduceScatterRowsSum 求和并切回 source-token slice。
    tp0_route_outputs = [Fraction(row["hidden"]) for row in layout["rows"]]
    tp1_route_outputs = [Fraction(row["local_expert"] * 100) for row in layout["rows"]]
    tp0_folded = _fold_topk(
        route_outputs=tp0_route_outputs,
        row_ids_map=layout["row_ids_map"],
        received_rows=gathered_rows,
    )
    tp1_folded = _fold_topk(
        route_outputs=tp1_route_outputs,
        row_ids_map=layout["row_ids_map"],
        received_rows=gathered_rows,
    )
    folded_sum = [left + right for left, right in zip(tp0_folded, tp1_folded)]
    assert tp0_folded == [Fraction(10), Fraction(12), Fraction(12), Fraction(40), Fraction(25)]
    assert tp1_folded == [Fraction(75), Fraction(120), Fraction(0), Fraction(170), Fraction(50)]
    assert folded_sum == [Fraction(85), Fraction(132), Fraction(12), Fraction(210), Fraction(75)]

    reduce_scatter_rows_sum = {
        "tp0": folded_sum[: tp_rank_row_counts[0]],
        "tp1": folded_sum[tp_rank_row_counts[0] :],
    }
    assert reduce_scatter_rows_sum == {
        "tp0": [Fraction(85), Fraction(132), Fraction(12)],
        "tp1": [Fraction(210), Fraction(75)],
    }

    return {
        "passed": True,
        "forward_order": FORWARD_ORDER,
        "tp_rank_row_counts": tp_rank_row_counts,
        "gathered_hidden": [row["hidden"] for row in gathered_rows],
        "gathered_topk_ids": gathered_topk_ids,
        "gathered_topk_weights": _matrix_numbers(gathered_topk_weights),
        "route_copy_tokens": layout["tokens"],
        "route_copy_local_experts": layout["local_experts"],
        "row_ids_map": layout["row_ids_map"],
        "tokens_per_expert": layout["tokens_per_expert"],
        "folded_partials": {
            "tp0": _numbers(tp0_folded),
            "tp1": _numbers(tp1_folded),
        },
        "folded_sum": _numbers(folded_sum),
        "reduce_scatter_rows_sum": {
            "tp0": _numbers(reduce_scatter_rows_sum["tp0"]),
            "tp1": _numbers(reduce_scatter_rows_sum["tp1"]),
        },
        "deepep_combine_inputs": {
            "tp0": _numbers(reduce_scatter_rows_sum["tp0"]),
            "tp1": _numbers(reduce_scatter_rows_sum["tp1"]),
        },
    }


def validate_all() -> dict[str, Any]:
    return {
        "all2all": validate_all2all_example(),
        "deepep": validate_deepep_example(),
        "deepep_expert_tp": validate_deepep_expert_tp_example(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dispatcher documentation examples.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable validation results.")
    args = parser.parse_args()

    payload = validate_all()
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("dispatcher documentation examples: ok")
        print("validated: all2all, deepep, deepep_expert_tp")


if __name__ == "__main__":
    main()

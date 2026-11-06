#!/usr/bin/env python3
"""Aggregate CC8 diagnosis events + monitor samples into the duplication/memory tables.

usage: analyze_events.py <run_dir> [--out analysis.json]

Outputs (into run_dir):
  analysis.json      full tables
  analysis.md        human-readable summary
Duplication accounting rules (per plan §一.B/C):
  - media identity = content blake2b of raw bytes (path is informational only)
  - materialized = read/decode/processor/sample events (includes prefetch-unconsumed)
  - delivered    = deliver events only (packs actually yielded to the consumer)
  - PSS tree totals from monitor.jsonl (never RSS sums); phases joined from rank0 events
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def load_events(run_dir):
    events = []
    for p in sorted(glob.glob(os.path.join(run_dir, "events", "*.jsonl"))):
        with open(p) as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def role_map(events):
    mains = {e["pid"]: e["rank"] for e in events if e["kind"] == "identity" and e.get("role") == "rank-main"}
    roles = {}
    for e in events:
        pid, ppid, rank = e["pid"], e["ppid"], e["rank"]
        if pid in mains:
            roles[pid] = f"rank{rank}-main"
        elif ppid in mains:
            roles[pid] = f"rank{rank}-worker"
        else:
            roles[pid] = f"rank{rank}-aux(ppid={ppid})"
    return roles, mains


def duplication_table(events, roles):
    per_media = defaultdict(lambda: {"read": 0, "decode": 0, "decoded_bytes": 0, "process_calls": 0,
                                     "process_images": 0, "ranks": set(), "paths": set(),
                                     "sample_pixel_shas": set()})
    for e in events:
        if e["kind"] == "oss_read":
            d = per_media[e["sha"]]
            d["read"] += 1
            d["ranks"].add(e["rank"])
            d["paths"].add(e.get("path"))
        elif e["kind"] == "decode":
            d = per_media.setdefault(e["sha"], {"read": 0, "decode": 0, "decoded_bytes": 0,
                                                "process_calls": 0, "process_images": 0,
                                                "ranks": set(), "paths": set(),
                                                "sample_pixel_shas": set()})
            d["decode"] += 1
            d["decoded_bytes"] += e.get("nbytes") or 0
            d["ranks"].add(e["rank"])
            if e.get("path"):
                d["paths"].add(e.get("path"))
    return per_media


def delivery_table(events):
    by_rank = defaultdict(list)
    for e in events:
        if e["kind"] == "deliver":
            by_rank[e["rank"]].append(e)
    return by_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="analysis.json")
    args = ap.parse_args()
    run_dir = args.run_dir
    events = load_events(run_dir)
    roles, mains = role_map(events)

    media = duplication_table(events, roles)
    media_rows = []
    for sha, d in sorted(media.items(), key=lambda kv: -kv[1]["read"]):
        if sha is None:
            continue
        media_rows.append({
            "media_sha16": sha, "read_total": d["read"], "decode_total": d["decode"],
            "decoded_bytes_total": d["decoded_bytes"],
            "n_ranks_reading": len(d["ranks"]), "paths": sorted(d["paths"]),
        })
    n_ranks = len({e["rank"] for e in events if e["kind"] == "identity"})
    total_reads = sum(r["read_total"] for r in media_rows)
    unique_media = len(media_rows)

    # materialized vs delivered image counts
    proc_images = sum(e.get("n_images", 0) for e in events if e["kind"] == "processor")
    sample_mm = sum(1 for e in events if e["kind"] == "sample_mm")
    sample_text = sum(1 for e in events if e["kind"] == "sample_text")
    sample_video = sum(1 for e in events if e["kind"] == "sample_video")
    delivered_images = sum(e.get("n_images_in_pack", 0) for e in events if e["kind"] == "deliver")
    delivered_packs = sum(1 for e in events if e["kind"] == "deliver")

    # bitwise consistency across ranks, grouped per pack index
    # (same pack_idx delivered by every rank must have identical shas)
    packs_pixel = defaultdict(set)
    packs_ids = defaultdict(set)
    packs_by_rank = defaultdict(set)
    for e in events:
        if e["kind"] == "deliver":
            k = e.get("pack_idx")
            packs_by_rank[k].add(e["rank"])
            if e.get("pixel_sha"):
                packs_pixel[k].add(e["pixel_sha"])
            if e.get("input_ids_sha"):
                packs_ids[k].add(e["input_ids_sha"])
    def cross_rank_identical(table):
        bad = {k: v for k, v in table.items() if len(v) != 1}
        return {"all_identical": not bad, "packs_compared": len(table), "mismatched_packs": sorted(bad)[:5]}
    pixel_check = cross_rank_identical(packs_pixel)
    ids_check = cross_rank_identical(packs_ids)

    # per-sample pixel shas: same media processed on different ranks must give identical sha
    sample_pixel_by_rank = defaultdict(set)
    for e in events:
        if e["kind"] == "sample_mm" and e.get("pixel_sha"):
            sample_pixel_by_rank[e["rank"]].add(e["pixel_sha"])
    sample_pixel_union = set().union(*sample_pixel_by_rank.values()) if sample_pixel_by_rank else set()

    # monitor join: phases from rank0
    mon_path = os.path.join(run_dir, "monitor.jsonl")
    phases = {}
    for e in events:
        if e["kind"] == "phase" and e["rank"] == 0:
            phases.setdefault(e["name"], e["ts"])
    mon = []
    if os.path.isfile(mon_path):
        with open(mon_path) as f:
            for line in f:
                try:
                    mon.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    def tree_at(ts_name, pick="max"):
        ts = phases.get(ts_name)
        if ts is None or not mon:
            return None
        cands = [m for m in mon if m["ts"] <= ts]
        if not cands:
            return None
        m = cands[-1]
        return {"tree_pss_gib": round(m["tree_pss_kb"] / 1048576, 3),
                "tree_uss_gib": round(m["tree_uss_kb"] / 1048576, 3),
                "n_procs": m["n_procs"], "sample_ts": m["ts"]}
    peak = max((m["tree_pss_kb"] for m in mon), default=0)
    mem = {
        "idle_after_build": tree_at("dataset_build_end"),
        "idle_iterate_start": tree_at("iterate_start"),
        "after_first_deliver": tree_at("pack_0_delivered"),
        "after_release": tree_at("loader_released") or tree_at("final_barrier_done"),
        "peak_tree_pss_gib": round(peak / 1048576, 3),
        "n_procs_peak": max((m["n_procs"] for m in mon), default=0),
    }
    idle = mem["idle_iterate_start"] or mem["idle_after_build"]
    if idle and peak:
        mem["media_increment_vs_idle_gib"] = round(peak / 1048576 - idle["tree_pss_gib"], 3)

    result = {
        "run_dir": run_dir,
        "n_ranks": n_ranks,
        "duplication": {
            "unique_media": unique_media,
            "total_reads": total_reads,
            "reads_per_unique_media": round(total_reads / unique_media, 3) if unique_media else None,
            "per_rank_expected_if_dup": n_ranks,
            "media_rows": media_rows,
        },
        "counts": {
            "processor_calls_materialized": sum(1 for e in events if e["kind"] == "processor"),
            "processor_images_materialized": proc_images,
            "sample_mm": sample_mm, "sample_text": sample_text, "sample_video": sample_video,
            "packs_delivered_total": delivered_packs,
            "images_delivered_total": delivered_images,
        },
        "consistency": {
            "deliver_pixel_across_ranks": pixel_check,
            "deliver_input_ids_across_ranks": ids_check,
            "ranks_per_pack": {str(k): sorted(v) for k, v in sorted(packs_by_rank.items())},
            "n_unique_sample_pixel_shas": len(sample_pixel_union),
        },
        "memory": mem,
    }
    out_path = os.path.join(run_dir, args.out)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    md = os.path.join(run_dir, "analysis.md")
    with open(md, "w") as f:
        f.write(f"# CC8 analysis — {run_dir}\n\n")
        f.write(f"- ranks: {n_ranks}\n- unique media: {unique_media}, total OSS reads: {total_reads}"
                f" → reads/unique-media = {result['duplication']['reads_per_unique_media']}"
                f" (SP dup hypothesis: = n_ranks = {n_ranks})\n")
        f.write(f"- materialized: processor calls={result['counts']['processor_calls_materialized']},"
                f" images={proc_images}, sample_mm={sample_mm}, sample_text={sample_text}\n")
        f.write(f"- delivered: packs={delivered_packs}, images={delivered_images}\n")
        f.write(f"- deliver bitwise identical across ranks: pixel={pixel_check},"
                f" input_ids={ids_check}\n")
        f.write(f"- memory: idle={idle}, peak={mem['peak_tree_pss_gib']}GiB,"
                f" media_increment={mem.get('media_increment_vs_idle_gib')}GiB, procs_peak={mem['n_procs_peak']}\n")
        for r in media_rows:
            f.write(f"  - media {r['media_sha16']}: reads={r['read_total']} decode={r['decode_total']}"
                    f" ranks={r['n_ranks_reading']} paths={r['paths']}\n")
    print(json.dumps(result["duplication"], indent=1)[:800])
    print(f"[analyze] wrote {out_path} and analysis.md")


if __name__ == "__main__":
    main()

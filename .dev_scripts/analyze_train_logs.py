"""Analyze multi-node torchrun training logs to localise tgs drops.

The script parses ``node_*.txt`` files emitted by torchrun, extracts per-rank
metrics for every step, and surfaces three kinds of problems:

1. Steps with a clear tgs drop (median tgs below a ratio of the run baseline).
2. Whether each drop is **data-bound** (dataloader stalled some rank) or
   **compute/comm-bound** (step time itself ballooned with healthy data_time).
3. For compute-bound steps it prints the distribution of ``efficient_attn_ratio``
   / ``img_tokens`` / ``text_tokens`` so the user can tell when an outlier
   sample has dominated a pack. For data-bound steps it prints the slowest
   ranks together with their data_time.

Example:
    python xtuner/tools/analyze_train_logs.py /path/to/torchrun_logs/xxx \\
        --tgs-drop-ratio 0.7

Default heuristics (overridable via CLI):
    * ``baseline_tgs``  -- median tgs of non-warmup steps.
    * tgs drop        -- median(tgs) < 0.7 * baseline_tgs.
    * data-bound      -- dt_max > 1s OR dt_max > 10 * (median of step_median dt).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Captures every metric we currently emit in xtuner's training INFO line. The
# regex is intentionally permissive about field ordering changes (e.g. extra
# fields appearing between known ones). Rank is read from a ``[RANK N]`` token
# when the launcher injects it; otherwise ``parse_logs`` falls back to the rank
# encoded in the per-rank log filename.
_LINE_RE = re.compile(
    r"Step (?P<step>\d+)/\d+\s+"
    r"data_time:\s*(?P<dt>[\d.]+)\s+"
    r"lr:\s*[\d.eE+\-]+\s+"
    r"time:\s*(?P<st>[\d.]+)\s+"
    r"text_tokens:\s*(?P<text_tokens>\d+)\s+"
    r"img_tokens:\s*(?P<img_tokens>[\d.]+).*?"
    r"efficient_attn_ratio:\s*(?P<eff>[\d.]+),\s*"
    r"img_efficient_attn_ratio:\s*(?P<img_eff>[\d.]+).*?"
    r"max_memory:\s*(?P<max_mem>[\d.]+)\s*GB\s+"
    r"reserved_memory:\s*(?P<rsv_mem>[\d.]+)\s*GB\s+"
    r"tgs:\s*(?P<tgs>[\d.]+)\s+"
    r"exp_tgs:\s*(?P<exp_tgs>[\d.]+)"
)

_RANK_IN_LINE_RE = re.compile(r"\[RANK (\d+)\]")
_RANK_IN_FILENAME_RE = re.compile(r"rank(\d+)", re.IGNORECASE)


@dataclass(slots=True)
class RankRecord:
    """One rank's metrics at one step."""

    rank: int
    data_time: float
    step_time: float
    text_tokens: int
    img_tokens: float
    eff_attn: float
    img_eff_attn: float
    max_mem: float
    tgs: float


@dataclass(slots=True)
class StepSummary:
    """Aggregated view of a single step across all ranks."""

    step: int
    n_ranks: int
    dt_min: float
    dt_med: float
    dt_max: float
    st_min: float
    st_med: float
    st_max: float
    tgs_min: float
    tgs_med: float
    tgs_max: float
    eff_attn_med: float
    eff_attn_p90: float
    eff_attn_max: float
    img_eff_attn_max: float
    img_tokens_med: float
    img_tokens_max: float
    text_tokens_min: int
    text_tokens_max: int
    max_mem_max: float
    # Filled in later when we decide attribution.
    is_drop: bool = False
    attribution: str = ""  # "data" | "compute" | ""
    notes: list[str] = field(default_factory=list)


def parse_logs(log_dir: Path) -> dict[int, dict[int, RankRecord]]:
    """Parse all ``node_*.txt`` files under ``log_dir``.

    Args:
        log_dir (Path): Directory holding torchrun node logs.

    Returns:
        dict[int, dict[int, RankRecord]]: ``{step: {rank: RankRecord}}``. Late
            entries for the same (step, rank) silently overwrite earlier ones,
            which is fine for resumed/retried runs.
    """
    by_step: dict[int, dict[int, RankRecord]] = {}
    files = sorted(log_dir.glob("node_*.txt"))
    if not files:
        files = sorted(log_dir.glob("rank*.log"))
    if not files:
        files = sorted(log_dir.glob("*.log"))  # fallback pattern

    if not files:
        raise FileNotFoundError(
            f"No node_*.txt / rank*.log files found under {log_dir}"
        )

    for path in files:
        # Per-rank log files (e.g. ``rank3.log``) do not embed ``[RANK N]`` in
        # every line; the filename is the authoritative source. We still prefer
        # an in-line ``[RANK N]`` token when present (multi-rank aggregated logs).
        fname_match = _RANK_IN_FILENAME_RE.search(path.name)
        rank_from_file = int(fname_match.group(1)) if fname_match else None

        with path.open("r", errors="replace") as fh:
            for line in fh:
                if "data_time:" not in line:
                    continue
                m = _LINE_RE.search(line)
                if m is None:
                    continue
                rank_match = _RANK_IN_LINE_RE.search(line)
                if rank_match is not None:
                    rank = int(rank_match.group(1))
                elif rank_from_file is not None:
                    rank = rank_from_file
                else:
                    continue
                step = int(m["step"])
                rec = RankRecord(
                    rank=rank,
                    data_time=float(m["dt"]),
                    step_time=float(m["st"]),
                    text_tokens=int(m["text_tokens"]),
                    img_tokens=float(m["img_tokens"]),
                    eff_attn=float(m["eff"]),
                    img_eff_attn=float(m["img_eff"]),
                    max_mem=float(m["max_mem"]),
                    tgs=float(m["tgs"]),
                )
                by_step.setdefault(step, {})[rec.rank] = rec
    return by_step


def summarize_step(step: int, ranks: dict[int, RankRecord]) -> StepSummary:
    """Reduce all rank records of a step to a single :class:`StepSummary`."""
    recs = list(ranks.values())
    dts = sorted(r.data_time for r in recs)
    sts = sorted(r.step_time for r in recs)
    tgs = sorted(r.tgs for r in recs)
    effs = sorted(r.eff_attn for r in recs)
    imgs = sorted(r.img_tokens for r in recs)

    def pct(values: list[float], p: float) -> float:
        idx = min(len(values) - 1, max(0, int(round(p * (len(values) - 1)))))
        return values[idx]

    return StepSummary(
        step=step,
        n_ranks=len(recs),
        dt_min=dts[0],
        dt_med=statistics.median(dts),
        dt_max=dts[-1],
        st_min=sts[0],
        st_med=statistics.median(sts),
        st_max=sts[-1],
        tgs_min=tgs[0],
        tgs_med=statistics.median(tgs),
        tgs_max=tgs[-1],
        eff_attn_med=statistics.median(effs),
        eff_attn_p90=pct(effs, 0.9),
        eff_attn_max=effs[-1],
        img_eff_attn_max=max(r.img_eff_attn for r in recs),
        img_tokens_med=statistics.median(imgs),
        img_tokens_max=imgs[-1],
        text_tokens_min=min(r.text_tokens for r in recs),
        text_tokens_max=max(r.text_tokens for r in recs),
        max_mem_max=max(r.max_mem for r in recs),
    )


def find_anomalies(
    summaries: list[StepSummary],
    tgs_drop_ratio: float,
    data_bound_abs: float,
    data_bound_rel: float,
    warmup_steps: int,
) -> tuple[float, float]:
    """Mark drop steps in-place and decide data-bound vs compute-bound.

    Args:
        summaries (list[StepSummary]): Summaries sorted by ``step``, mutated in place.
        tgs_drop_ratio (float): Step counts as a drop when ``tgs_med`` falls
            below ``tgs_drop_ratio * baseline_tgs``.
        data_bound_abs (float): Absolute ``dt_max`` (in seconds) above which a
            drop is attributed to data loading.
        data_bound_rel (float): Multiplier on the run-wide median of
            ``dt_med``; if ``dt_max`` exceeds this it also counts as data-bound.
        warmup_steps (int): Number of leading steps to exclude from the
            baseline computation (the first iteration is always cold-start).

    Returns:
        tuple[float, float]: ``(baseline_tgs, dt_med_baseline)`` so the caller
            can echo the chosen baselines in its report.
    """
    eligible = [s for s in summaries if s.step > warmup_steps]
    baseline_tgs = statistics.median(s.tgs_med for s in eligible)
    dt_med_baseline = statistics.median(s.dt_med for s in eligible)
    dt_max_threshold = max(data_bound_abs, data_bound_rel * dt_med_baseline)

    for s in summaries:
        # Always tag the warmup step but never reclassify it as a real drop.
        if s.step <= warmup_steps:
            s.notes.append(f"warmup (step <= {warmup_steps})")
            continue
        if s.tgs_med >= tgs_drop_ratio * baseline_tgs:
            continue
        s.is_drop = True
        if s.dt_max > dt_max_threshold:
            s.attribution = "data"
        else:
            s.attribution = "compute"
    return baseline_tgs, dt_max_threshold


_LEGEND = """\
Columns:
  step             training step index (1-based)
  cause            'data'    -> some rank(s) stalled in the dataloader
                   'compute' -> step_time itself ballooned (pack/comm/compile)
  tgs(med)         median tgs across ranks (tokens/GPU/s); throughput of this step
  ratio            tgs(med) / baseline_tgs; <1 means slowdown
                   (e.g. 0.41 == this step ran at 41% of normal speed)
  step_t(s)        median step_time across ranks (seconds)
  data_max(s)      max  data_time across ranks (= slowest dataloader rank)
  data_med(s)      median data_time across ranks (= typical loader cost)
  attn(max)        max efficient_attn_ratio (LLM side); 1.0 means one sample
                   fills the LLM pack -> attention becomes full O(L^2)
  img_attn(max)    max img_efficient_attn_ratio (vision side); 1.0 means one
                   image dominates the vision pack
  text_tok(min)    min text_tokens across ranks; 0 means some rank received
                   no LLM tokens this step (data starvation / SP partitioning)
  text_tok(max)    max text_tokens across ranks (usually == pack length)
  img_tok(med)     median img_tokens across ranks (typical image load per pack)
  img_tok(max)     max img_tokens across ranks (heaviest pack)
"""


def format_drop_table(
    summaries: list[StepSummary], baseline_tgs: float
) -> str:
    """One-line-per-step table for every tgs drop."""
    drops = [s for s in summaries if s.is_drop]
    if not drops:
        return _LEGEND + "\n(no tgs drop detected)"

    header = (
        f"{'step':>5}  {'cause':>7}  {'tgs(med)':>9}  {'ratio':>6}  "
        f"{'step_t(s)':>10}  {'data_max(s)':>12}  {'data_med(s)':>12}  "
        f"{'attn(max)':>10}  {'img_attn(max)':>14}  "
        f"{'text_tok(min)':>14}  {'text_tok(max)':>14}  "
        f"{'img_tok(med)':>13}  {'img_tok(max)':>13}"
    )
    lines = [_LEGEND, header, "-" * len(header)]
    for s in drops:
        lines.append(
            f"{s.step:>5}  {s.attribution:>7}  {s.tgs_med:>9.1f}  "
            f"{s.tgs_med / baseline_tgs:>6.2f}  {s.st_med:>10.2f}  "
            f"{s.dt_max:>12.3f}  {s.dt_med:>12.4f}  "
            f"{s.eff_attn_max:>10.4f}  {s.img_eff_attn_max:>14.4f}  "
            f"{s.text_tokens_min:>14d}  {s.text_tokens_max:>14d}  "
            f"{s.img_tokens_med:>13.0f}  {s.img_tokens_max:>13.0f}"
        )
    return "\n".join(lines)


def format_data_bound_details(
    summary: StepSummary, ranks: dict[int, RankRecord], top_n: int
) -> str:
    """Per-rank breakdown for a data-bound step."""
    slowest = sorted(ranks.values(), key=lambda r: -r.data_time)[:top_n]
    lines = [
        f"  Top {top_n} slowest ranks by data_time "
        f"(step_time_med={summary.st_med:.2f}s):",
        f"    {'rank':>5}  {'data_time':>10}  {'step_time':>10}",
    ]
    for r in slowest:
        lines.append(
            f"    {r.rank:>5}  {r.data_time:>10.3f}  {r.step_time:>10.3f}"
        )
    # SP-group hint: if the top slow ranks are contiguous in groups of 4-8 the
    # user almost always wants to know.
    top_ranks = sorted(r.rank for r in slowest)
    contiguous = _detect_contiguous_groups(top_ranks)
    if contiguous:
        lines.append(f"    [hint] contiguous rank groups: {contiguous}")
    return "\n".join(lines)


def format_compute_bound_details(
    summary: StepSummary, ranks: dict[int, RankRecord], top_n: int
) -> str:
    """Per-step view for a compute-bound (pack-imbalance) step."""
    heaviest = sorted(ranks.values(), key=lambda r: -r.eff_attn)[:top_n]
    lines = [
        f"  attn (LLM) ratio:      med={summary.eff_attn_med:.4f}  "
        f"p90={summary.eff_attn_p90:.4f}  max={summary.eff_attn_max:.4f}",
        f"  img_attn ratio:        max={summary.img_eff_attn_max:.4f}",
        f"  text_tokens:           min={summary.text_tokens_min}  "
        f"max={summary.text_tokens_max}",
        f"  img_tokens:            med={summary.img_tokens_med:.0f}  "
        f"max={summary.img_tokens_max:.0f}",
        f"  max_mem (any rank):    {summary.max_mem_max:.2f} GB",
        f"  Top {top_n} ranks by attn(LLM) ratio (single-sample-dominated packs):",
        f"    {'rank':>5}  {'attn':>9}  {'img_attn':>9}  "
        f"{'img_tok':>9}  {'text_tok':>9}  {'max_mem':>8}",
    ]
    for r in heaviest:
        lines.append(
            f"    {r.rank:>5}  {r.eff_attn:>9.4f}  {r.img_eff_attn:>8.4f}  "
            f"{r.img_tokens:>9.0f}  {r.text_tokens:>9}  {r.max_mem:>8.2f}"
        )
    contiguous = _detect_contiguous_groups(sorted(r.rank for r in heaviest))
    if contiguous:
        lines.append(f"    [hint] contiguous rank groups: {contiguous}")
    return "\n".join(lines)


def write_json(
    summaries: list[StepSummary], path: Path, baseline_tgs: float, dt_threshold: float
) -> None:
    """Dump the analysis to JSON for downstream tooling."""
    out = {
        "baseline_tgs": baseline_tgs,
        "data_bound_dt_threshold": dt_threshold,
        "steps": [
            {
                "step": s.step,
                "is_drop": s.is_drop,
                "attribution": s.attribution,
                "tgs_med": s.tgs_med,
                "tgs_ratio": s.tgs_med / baseline_tgs if baseline_tgs else None,
                "st_med": s.st_med,
                "dt_med": s.dt_med,
                "dt_max": s.dt_max,
                "eff_attn_med": s.eff_attn_med,
                "eff_attn_p90": s.eff_attn_p90,
                "eff_attn_max": s.eff_attn_max,
                "img_eff_attn_max": s.img_eff_attn_max,
                "text_tokens_min": s.text_tokens_min,
                "text_tokens_max": s.text_tokens_max,
                "img_tokens_med": s.img_tokens_med,
                "img_tokens_max": s.img_tokens_max,
                "max_mem_max": s.max_mem_max,
                "notes": s.notes,
            }
            for s in summaries
        ],
    }
    path.write_text(json.dumps(out, indent=2))


def _detect_contiguous_groups(ranks: Iterable[int]) -> list[list[int]]:
    """Collapse a sorted rank list into contiguous runs.

    Useful for flagging that all slow ranks belong to the same SP / PP group.
    Only runs of length >= 2 are returned, since stragglers tend to come in
    SP-sized clusters (4, 8, ...).
    """
    groups: list[list[int]] = []
    current: list[int] = []
    for r in ranks:
        if not current or r == current[-1] + 1:
            current.append(r)
            continue
        if len(current) >= 2:
            groups.append(current)
        current = [r]
    if len(current) >= 2:
        groups.append(current)
    return groups


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose tgs drops in xtuner torchrun logs. Splits drops into "
            "data-bound (slow ranks) and compute-bound (pack imbalance)."
        )
    )
    parser.add_argument("log_dir", type=Path, help="Directory containing node_*.txt logs.")
    parser.add_argument(
        "--tgs-drop-ratio",
        type=float,
        default=0.7,
        help="Step is flagged when median tgs falls below this ratio of the run baseline.",
    )
    parser.add_argument(
        "--data-bound-abs",
        type=float,
        default=1.0,
        help="Absolute data_time (s) above which a drop is attributed to dataloading.",
    )
    parser.add_argument(
        "--data-bound-rel",
        type=float,
        default=10.0,
        help="Multiplier on the global median dt_med; data_time above this is data-bound.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Leading steps excluded from baseline (cold-start).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of ranks to print per anomaly.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to dump the full analysis as JSON.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    by_step = parse_logs(args.log_dir)
    summaries = [summarize_step(s, by_step[s]) for s in sorted(by_step)]

    baseline_tgs, dt_threshold = find_anomalies(
        summaries,
        tgs_drop_ratio=args.tgs_drop_ratio,
        data_bound_abs=args.data_bound_abs,
        data_bound_rel=args.data_bound_rel,
        warmup_steps=args.warmup_steps,
    )

    n_ranks = max(s.n_ranks for s in summaries)
    n_drops = sum(1 for s in summaries if s.is_drop)
    n_data = sum(1 for s in summaries if s.attribution == "data")
    n_compute = sum(1 for s in summaries if s.attribution == "compute")
    print(f"Parsed {len(summaries)} steps from {args.log_dir} ({n_ranks} ranks).")
    print(
        f"Baseline tgs (median over step>{args.warmup_steps}) = {baseline_tgs:.1f}; "
        f"data_time threshold for data-bound = {dt_threshold:.3f}s "
        f"(max of {args.data_bound_abs}s and {args.data_bound_rel}x median dt_med)."
    )
    print(
        f"Detected {n_drops} drop step(s): {n_data} data-bound, {n_compute} compute-bound."
    )

    print("\n=== 1. tgs drop summary (median tgs < {:.2f} x baseline) ===".format(
        args.tgs_drop_ratio
    ))
    print(format_drop_table(summaries, baseline_tgs))

    data_drops = [s for s in summaries if s.attribution == "data"]
    if data_drops:
        print("\n=== 2. Data-bound steps -- slowest ranks ===")
        for s in data_drops:
            print(f"\n[step {s.step}] step_time={s.st_med:.2f}s  "
                  f"dt_med={s.dt_med:.4f}s  dt_max={s.dt_max:.2f}s")
            print(format_data_bound_details(s, by_step[s.step], args.top_n))

    compute_drops = [s for s in summaries if s.attribution == "compute"]
    if compute_drops:
        print("\n=== 3. Compute-bound steps -- pack imbalance / heavy samples ===")
        for s in compute_drops:
            print(f"\n[step {s.step}] step_time={s.st_med:.2f}s  "
                  f"tgs_med={s.tgs_med:.1f} ({s.tgs_med / baseline_tgs:.2f}x baseline)")
            print(format_compute_bound_details(s, by_step[s.step], args.top_n))

    if args.json_out is not None:
        write_json(summaries, args.json_out, baseline_tgs, dt_threshold)
        print(f"\nJSON dump written to {args.json_out}")


if __name__ == "__main__":
    main()

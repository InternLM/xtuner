#!/usr/bin/env python3
"""Analyze recent ailab-llmrazor RJobs: creators + GPU/memory utilization.

Uses:
  - RJob API (optional Bearer token via PJLAB_TOKEN / --token-file)
  - Prometheus via Grafana proxy (public read)

For jobs spanning >1 day, metrics use the last 24h before last activity.
For shorter jobs, metrics use the full active span (up to 7d lookback).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any


DEFAULT_RJOB_BASE = (
    "https://h.pjlab.org.cn/kapis/rjob.brainpp.cn/v1alpha1"
    "/tenants/ailab/projects/ailab-llmrazor/rjobs"
)
DEFAULT_PROM_URL = "https://h.pjlab.org.cn/grafana/api/datasources/proxy/2/api/v1"
DEFAULT_NAMESPACE = "ailab-llmrazor"
CI_POD_RE = re.compile(r"^(sft|rl|xtuner-ci)-")


@dataclass
class JobReport:
    rjob_name: str
    creator: str
    phase: str
    created_at: str
    pods: int
    gpus: int
    span_hours: float
    metric_window: str
    gpu_util_avg: float
    gpu_util_p95: float
    mem_util_avg: float
    mem_util_p95: float
    first_active: str
    last_active: str


def curl_json(url: str, *, headers: dict[str, str] | None = None) -> Any:
    cmd = ["curl", "-sk", url]
    for key, value in (headers or {}).items():
        cmd.extend(["-H", f"{key}: {value}"])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def prom_instant(base_url: str, query: str) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/query?{urllib.parse.urlencode({'query': query})}"
    payload = curl_json(url)
    return payload.get("data", {}).get("result", [])


def prom_range(base_url: str, query: str, *, start: int, end: int, step: int) -> dict[str, list[tuple[float, float]]]:
    url = (
        f"{base_url.rstrip('/')}/query_range?"
        f"{urllib.parse.urlencode({'query': query, 'start': start, 'end': end, 'step': step})}"
    )
    payload = curl_json(url)
    out: dict[str, list[tuple[float, float]]] = {}
    for row in payload.get("data", {}).get("result", []):
        pod = row.get("metric", {}).get("pod", "__aggregate__")
        values = [
            (float(ts), float(val))
            for ts, val in row.get("values", [])
            if val not in ("NaN", "Inf", "-Inf")
        ]
        out[pod] = values
    return out


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * pct))))
    return ordered[idx]


def rjob_from_pod(pod: str) -> str:
    match = re.match(r"^(.*)-[^-]+$", pod)
    return match.group(1) if match else pod


def load_token(token: str | None, token_file: str | None) -> str | None:
    if token:
        return token.strip()
    if token_file:
        with open(token_file, encoding="utf-8") as f:
            return f.read().strip()
    return os.environ.get("PJLAB_TOKEN", "").strip() or None


def list_rjobs(base_url: str, token: str, *, creator: str, page_size: int = 100) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params: dict[str, Any] = {
        "page": 1,
        "pageSize": page_size,
        "sortBy": "creationTimestamp:desc",
        "labelSelector": "kubebrain.brainpp.cn/extraresourcetype notin (eval-job, train-job, datamaster-job)",
        "shownames": "",
        "rjobids": "",
        "creators": creator,
        "self": "true",
        "tasktype": "rjob-normal",
    }
    items: list[dict[str, Any]] = []
    while True:
        url = f"{base_url}?{urllib.parse.urlencode(params, doseq=True)}"
        payload = curl_json(url, headers=headers)
        page = payload.get("data", {}).get("items") or payload.get("items") or []
        items.extend(page)
        total = payload.get("data", {}).get("total") or payload.get("total") or len(items)
        if len(items) >= total or not page:
            break
        params["page"] = int(params["page"]) + 1
    return items


def job_name(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return str(metadata.get("name") or item.get("name") or item.get("rjobName") or "unknown")


def job_creator(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
    for key in ("brainpp.cn/creator", "kubebrain.brainpp.cn/creator", "rjob.brainpp.cn/creator", "creator"):
        if labels.get(key):
            return str(labels[key])
    spec = item.get("spec") if isinstance(item.get("spec"), dict) else {}
    if spec.get("creator"):
        return str(spec["creator"])
    return "qa-llm-cicd"


def job_phase(item: dict[str, Any]) -> str:
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    for key in ("phase", "state"):
        if isinstance(status.get(key), str):
            return status[key]
    return "Unknown"


def job_created(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return str(metadata.get("creationTimestamp") or "-")


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def discover_ci_pods(prom_url: str, namespace: str, *, days: int) -> list[str]:
    query = (
        "count by (pod) (changes(kubebrain_exporter_gpu_resources_utilization"
        f'{{busy="true", exported_namespace="{namespace}", pod=~"^(sft|rl|xtuner-ci)-.*"}}[{days}d]) > 0)'
    )
    return sorted(row["metric"]["pod"] for row in prom_instant(prom_url, query))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token")
    parser.add_argument("--token-file")
    parser.add_argument("--rjob-base-url", default=DEFAULT_RJOB_BASE)
    parser.add_argument("--prom-url", default=DEFAULT_PROM_URL)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--creator", default="qa-llm-cicd")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--output", default="/tmp/rjob_recent_analysis.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    end = int(time.time())
    start = end - args.days * 86400
    token = load_token(args.token, args.token_file)

    api_jobs: dict[str, dict[str, Any]] = {}
    if token:
        print(f"Fetching RJobs from API (creator={args.creator}) ...")
        try:
            for item in list_rjobs(args.rjob_base_url, token, creator=args.creator):
                api_jobs[job_name(item)] = item
            print(f"  API returned {len(api_jobs)} jobs")
        except Exception as exc:
            print(f"  RJob API failed: {exc}", file=sys.stderr)
    else:
        print("No PJLAB_TOKEN; inferring qa-llm-cicd jobs from sft/rl/xtuner-ci pod names.")

    print(f"Discovering CI pods in last {args.days}d ...")
    pods = discover_ci_pods(args.prom_url, args.namespace, days=args.days)
    print(f"  found {len(pods)} pods")

    pod_pattern = "^(sft|rl|xtuner-ci)-.*"
    util_q = (
        "avg by (pod) (kubebrain_exporter_gpu_resources_utilization"
        f'{{busy="true", exported_namespace="{args.namespace}", pod=~"{pod_pattern}"}})'
    )
    mem_q = (
        "avg by (pod) (kubebrain_exporter_gpu_resources_mem_utilization"
        f'{{busy="true", exported_namespace="{args.namespace}", pod=~"{pod_pattern}"}})'
    )
    gpu_q = (
        "count by (pod) (kubebrain_exporter_gpu_resources_utilization"
        f'{{busy="true", exported_namespace="{args.namespace}", pod=~"{pod_pattern}"}})'
    )

    print("Querying 7d GPU/memory time series (batched) ...")
    util_by_pod = prom_range(args.prom_url, util_q, start=start, end=end, step=600)
    mem_by_pod = prom_range(args.prom_url, mem_q, start=start, end=end, step=600)

    pod_gpu_counts = {
        row["metric"]["pod"]: int(float(row["value"][1]))
        for row in prom_instant(args.prom_url, gpu_q)
    }

    reports: list[JobReport] = []
    rjob_pods: dict[str, list[str]] = defaultdict(list)
    for pod in util_by_pod:
        rjob_pods[rjob_from_pod(pod)].append(pod)

    for rjob_name, rjob_pod_list in rjob_pods.items():
        item = api_jobs.get(rjob_name, {})
        creator = job_creator(item) if item else "qa-llm-cicd"
        phase = job_phase(item) if item else "Unknown"
        created = job_created(item) if item else "-"

        util_vals_all: list[float] = []
        mem_vals_all: list[float] = []
        first_ts = None
        last_ts = None
        gpus = sum(pod_gpu_counts.get(pod, 0) for pod in rjob_pod_list)

        for pod in rjob_pod_list:
            util_series = util_by_pod.get(pod, [])
            mem_series = mem_by_pod.get(pod, [])
            if not util_series:
                continue
            ts_list = [ts for ts, _ in util_series]
            first_ts = min(ts_list) if first_ts is None else min(first_ts, min(ts_list))
            last_ts = max(ts_list) if last_ts is None else max(last_ts, max(ts_list))

        if first_ts is None or last_ts is None:
            continue
        span_hours = max(0.5, (last_ts - first_ts) / 3600)
        metric_window = "last_1d" if span_hours > 24 else "full_span"
        win_start = last_ts - 86400 if metric_window == "last_1d" else first_ts
        for pod in rjob_pod_list:
            util_vals_all.extend(v for ts, v in util_by_pod.get(pod, []) if win_start <= ts <= last_ts)
            mem_vals_all.extend(v for ts, v in mem_by_pod.get(pod, []) if win_start <= ts <= last_ts)

        reports.append(
            JobReport(
                rjob_name=rjob_name,
                creator=creator,
                phase=phase,
                created_at=created,
                pods=len(rjob_pod_list),
                gpus=gpus,
                span_hours=round(span_hours, 1),
                metric_window=metric_window,
                gpu_util_avg=round(mean(util_vals_all), 1) if util_vals_all else 0.0,
                gpu_util_p95=round(percentile(util_vals_all, 0.95), 1),
                mem_util_avg=round(mean(mem_vals_all), 1) if mem_vals_all else 0.0,
                mem_util_p95=round(percentile(mem_vals_all, 0.95), 1),
                first_active=fmt_ts(first_ts),
                last_active=fmt_ts(last_ts),
            )
        )

    by_creator: dict[str, list[JobReport]] = defaultdict(list)
    for report in reports:
        by_creator[report.creator].append(report)

    print(f"\n=== Summary: {len(reports)} jobs in last {args.days}d ===")
    print("\nBy creator (GPU-weighted):")
    for creator, rows in sorted(by_creator.items(), key=lambda x: -sum(r.gpus for r in x[1])):
        gpu_total = sum(r.gpus for r in rows)
        if not gpu_total:
            continue
        gpu_w = sum(r.gpu_util_avg * r.gpus for r in rows) / gpu_total
        mem_w = sum(r.mem_util_avg * r.gpus for r in rows) / gpu_total
        running = sum(1 for r in rows if r.last_active >= fmt_ts(end - 3600))
        print(
            f"  {creator}: jobs={len(rows)} gpus={gpu_total} "
            f"gpu_util={gpu_w:.1f}% mem_util={mem_w:.1f}% active_1h~{running}"
        )

    print("\nRecent jobs (top 40 by last activity):")
    print("creator\tphase\twindow\tgpus\tgpu_avg%\tgpu_p95%\tmem_avg%\tmem_p95%\tspan_h\tlast_active\trjob")
    for report in sorted(reports, key=lambda r: r.last_active, reverse=True)[:40]:
        print(
            f"{report.creator}\t{report.phase}\t{report.metric_window}\t{report.gpus}\t"
            f"{report.gpu_util_avg}\t{report.gpu_util_p95}\t{report.mem_util_avg}\t{report.mem_util_p95}\t"
            f"{report.span_hours}\t{report.last_active}\t{report.rjob_name}"
        )

    low_util = sorted(reports, key=lambda r: r.gpu_util_avg)[:10]
    print("\nLowest GPU util jobs:")
    for report in low_util:
        print(
            f"  {report.gpu_util_avg:5.1f}% gpu / {report.mem_util_avg:5.1f}% mem | "
            f"{report.last_active} | {report.rjob_name}"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in reports], f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

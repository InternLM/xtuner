#!/usr/bin/env python3
"""Analyze running qa-llm-cicd RJobs: 7-day GPU utilization and power.

Data sources:
  - RJob list API (requires Bearer token)
  - Prometheus via Grafana proxy (public read, no token)

Usage:
  export PJLAB_TOKEN='your-bearer-token'
  python autotest/utils/analyze_running_rjobs_gpu.py

  python autotest/utils/analyze_running_rjobs_gpu.py --days 7 --output report.json
  python autotest/utils/analyze_running_rjobs_gpu.py --token-file ~/.pjlab_token

Never commit tokens. Prefer PJLAB_TOKEN env var or --token-file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any


DEFAULT_RJOB_BASE = (
    "https://h.pjlab.org.cn/kapis/rjob.brainpp.cn/v1alpha1"
    "/tenants/ailab/projects/ailab-llmrazor/rjobs"
)
DEFAULT_PROM_URL = "https://h.pjlab.org.cn/grafana/api/datasources/proxy/2/api/v1"
DEFAULT_NAMESPACE = "ailab-llmrazor"
DEFAULT_CREATOR = "qa-llm-cicd"


@dataclass
class PodMetrics:
    pod: str
    gpu_count: int
    gpu_util_avg: float
    gpu_util_p50: float
    gpu_util_p95: float
    gpu_util_max: float
    power_avg_w: float
    power_max_w: float
    energy_kwh: float
    sample_points: int


@dataclass
class RJobMetrics:
    rjob_id: str
    rjob_name: str
    phase: str
    created_at: str
    gpu_cards: int
    pods: list[PodMetrics] = field(default_factory=list)
    gpu_util_avg: float = 0.0
    gpu_util_p95: float = 0.0
    power_avg_w: float = 0.0
    power_max_w: float = 0.0
    energy_kwh: float = 0.0


class PjlabClient:
    def __init__(self, token: str, *, verify_ssl: bool = False) -> None:
        self.token = token
        self.ssl_context = ssl.create_default_context()
        if not verify_ssl:
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE

    def _request_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/json")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        with urllib.request.urlopen(req, context=self.ssl_context, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_rjobs(
        self,
        *,
        base_url: str,
        creator: str,
        page_size: int = 100,
        extra_params: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        params = {
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
        if extra_params:
            params.update(extra_params)

        items: list[dict[str, Any]] = []
        while True:
            payload = self._request_json(base_url, params=params)
            page_items = _extract_items(payload)
            items.extend(page_items)
            total = _extract_total(payload, fallback=len(items))
            if len(items) >= total or not page_items:
                break
            params["page"] = int(params["page"]) + 1
        return items


class PromClient:
    def __init__(self, base_url: str = DEFAULT_PROM_URL, *, verify_ssl: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.ssl_context = ssl.create_default_context()
        if not verify_ssl:
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE

    def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, context=self.ssl_context, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def instant(self, query: str) -> list[dict[str, Any]]:
        payload = self._request_json("query", {"query": query})
        return payload.get("data", {}).get("result", [])

    def range_series(self, query: str, *, start: int, end: int, step_s: int) -> list[tuple[float, float]]:
        payload = self._request_json(
            "query_range",
            {"query": query, "start": start, "end": end, "step": step_s},
        )
        results = payload.get("data", {}).get("result", [])
        if not results:
            return []
        values = results[0].get("values", [])
        return [(float(ts), float(val)) for ts, val in values if val not in ("NaN", "Inf", "-Inf")]

    def pods_for_rjob(self, rjob_name: str, namespace: str) -> list[str]:
        # Pod naming: {rjob_name}-{replica}; hstat strips the last segment as replica id.
        pattern = re.escape(rjob_name) + r"-[^-]+$"
        query = (
            "count by (pod) (kubebrain_exporter_gpu_resources_utilization"
            f'{{busy="true", exported_namespace="{namespace}", pod=~"{pattern}"}})'
        )
        pods = [row["metric"]["pod"] for row in self.instant(query)]
        if pods:
            return sorted(pods)

        # Fallback: prefix match when replica suffix format differs.
        prefix = re.escape(rjob_name)
        query = (
            "count by (pod) (kubebrain_exporter_gpu_resources_utilization"
            f'{{busy="true", exported_namespace="{namespace}", pod=~"^{prefix}.*"}})'
        )
        return sorted(row["metric"]["pod"] for row in self.instant(query))

    def gpu_count(self, pod: str, namespace: str) -> int:
        query = (
            "count(kubebrain_exporter_gpu_resources_utilization"
            f'{{busy="true", exported_namespace="{namespace}", pod="{pod}"}})'
        )
        rows = self.instant(query)
        if not rows:
            return 0
        return int(float(rows[0]["value"][1]))


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("items", "data", "result", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = value.get("items") or value.get("records")
            if isinstance(nested, list):
                return nested
    return []


def _extract_total(payload: dict[str, Any], *, fallback: int) -> int:
    for key in ("total", "totalCount", "count"):
        if key in payload and isinstance(payload[key], int):
            return payload[key]
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("total", "totalCount", "count"):
            value = data.get(key)
            if isinstance(value, int):
                return value
    return fallback


def _job_name(item: dict[str, Any]) -> str:
    for path in (
        ("metadata", "name"),
        ("name",),
        ("rjobName",),
        ("jobName",),
    ):
        cur: Any = item
        ok = True
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                ok = False
                break
            cur = cur[part]
        if ok and isinstance(cur, str):
            return cur
    return str(item.get("id") or item.get("rjobId") or "unknown")


def _job_phase(item: dict[str, Any]) -> str:
    for path in (
        ("status", "phase"),
        ("status", "state", "phase"),
        ("phase",),
        ("state",),
    ):
        cur: Any = item
        ok = True
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                ok = False
                break
            cur = cur[part]
        if ok and isinstance(cur, str):
            return cur
    return "Unknown"


def _job_created(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    for key in ("creationTimestamp", "createdAt", "createTime"):
        value = metadata.get(key) or item.get(key)
        if isinstance(value, str):
            return value
    return "-"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _integrate_energy_kwh(power_series: list[tuple[float, float]]) -> float:
    if len(power_series) < 2:
        return 0.0
    energy_ws = 0.0
    for (t0, p0), (t1, _) in zip(power_series, power_series[1:]):
        dt = max(0.0, t1 - t0)
        energy_ws += p0 * dt
    return energy_ws / 3_600_000.0


def collect_pod_metrics(
    prom: PromClient,
    *,
    pod: str,
    namespace: str,
    start: int,
    end: int,
    step_s: int,
) -> PodMetrics:
    gpu_count = prom.gpu_count(pod, namespace)
    util_query = (
        "avg(kubebrain_exporter_gpu_resources_utilization"
        f'{{busy="true", exported_namespace="{namespace}", pod="{pod}"}})'
    )
    power_query = (
        "sum(kubebrain_exporter_gpu_resources_power_usage"
        f'{{busy="true", exported_namespace="{namespace}", pod="{pod}"}})'
    )
    util_series = prom.range_series(util_query, start=start, end=end, step_s=step_s)
    power_series = prom.range_series(power_query, start=start, end=end, step_s=step_s)
    util_vals = [v for _, v in util_series]
    power_vals = [v for _, v in power_series]
    return PodMetrics(
        pod=pod,
        gpu_count=gpu_count,
        gpu_util_avg=mean(util_vals) if util_vals else 0.0,
        gpu_util_p50=_percentile(util_vals, 0.50),
        gpu_util_p95=_percentile(util_vals, 0.95),
        gpu_util_max=max(util_vals) if util_vals else 0.0,
        power_avg_w=mean(power_vals) if power_vals else 0.0,
        power_max_w=max(power_vals) if power_vals else 0.0,
        energy_kwh=_integrate_energy_kwh(power_series),
        sample_points=len(util_series),
    )


def aggregate_rjob_metrics(rjob_name: str, phase: str, created_at: str, pods: list[PodMetrics]) -> RJobMetrics:
    gpu_cards = sum(p.gpu_count for p in pods)
    util_weighted = sum(p.gpu_util_avg * p.gpu_count for p in pods)
    util_avg = util_weighted / gpu_cards if gpu_cards else 0.0
    util_p95 = max((p.gpu_util_p95 for p in pods), default=0.0)
    power_avg = sum(p.power_avg_w for p in pods)
    power_max = max((p.power_max_w for p in pods), default=0.0)
    energy = sum(p.energy_kwh for p in pods)
    return RJobMetrics(
        rjob_id=rjob_name,
        rjob_name=rjob_name,
        phase=phase,
        created_at=created_at,
        gpu_cards=gpu_cards,
        pods=pods,
        gpu_util_avg=util_avg,
        gpu_util_p95=util_p95,
        power_avg_w=power_avg,
        power_max_w=power_max,
        energy_kwh=energy,
    )


def print_table(rows: list[RJobMetrics]) -> None:
    headers = [
        "RJob",
        "Phase",
        "GPUs",
        "Pods",
        "GPU util avg%",
        "GPU util p95%",
        "Power avg W",
        "Power max W",
        "Energy kWh(7d)",
    ]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    row.rjob_name,
                    row.phase,
                    str(row.gpu_cards),
                    str(len(row.pods)),
                    f"{row.gpu_util_avg:.1f}",
                    f"{row.gpu_util_p95:.1f}",
                    f"{row.power_avg_w:.0f}",
                    f"{row.power_max_w:.0f}",
                    f"{row.energy_kwh:.1f}",
                ]
            )
        )


def write_csv(path: str, rows: list[RJobMetrics]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rjob_name",
                "phase",
                "created_at",
                "gpu_cards",
                "pod_count",
                "gpu_util_avg_pct",
                "gpu_util_p95_pct",
                "power_avg_w",
                "power_max_w",
                "energy_kwh",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.rjob_name,
                    row.phase,
                    row.created_at,
                    row.gpu_cards,
                    len(row.pods),
                    f"{row.gpu_util_avg:.2f}",
                    f"{row.gpu_util_p95:.2f}",
                    f"{row.power_avg_w:.2f}",
                    f"{row.power_max_w:.2f}",
                    f"{row.energy_kwh:.2f}",
                ]
            )


def load_token(token: str | None, token_file: str | None) -> str:
    if token:
        return token.strip()
    if token_file:
        with open(token_file, encoding="utf-8") as f:
            return f.read().strip()
    env_token = os.environ.get("PJLAB_TOKEN", "").strip()
    if env_token:
        return env_token
    raise SystemExit(
        "Missing token. Set PJLAB_TOKEN, pass --token, or pass --token-file.\n"
        "Do not commit bearer tokens into the repository."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", help="Bearer token (prefer env PJLAB_TOKEN)")
    parser.add_argument("--token-file", help="Path to file containing bearer token")
    parser.add_argument("--rjob-base-url", default=DEFAULT_RJOB_BASE)
    parser.add_argument("--prom-url", default=DEFAULT_PROM_URL)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--creator", default=DEFAULT_CREATOR)
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days")
    parser.add_argument("--step-minutes", type=int, default=60, help="Prometheus range step")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--include-non-running", action="store_true")
    parser.add_argument("--output", help="Write JSON report to this path")
    parser.add_argument("--csv", help="Write CSV summary to this path")
    parser.add_argument("--verify-ssl", action="store_true")
    parser.add_argument(
        "--debug-dump-rjobs",
        help="Print raw RJob API payload sample (first item) and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = load_token(args.token, args.token_file)
    rjob_client = PjlabClient(token, verify_ssl=args.verify_ssl)
    prom = PromClient(args.prom_url, verify_ssl=args.verify_ssl)

    end = int(time.time())
    start = end - args.days * 24 * 3600
    step_s = max(60, args.step_minutes * 60)

    print(f"Fetching RJobs from {args.rjob_base_url} (creator={args.creator}) ...")
    try:
        rjobs = rjob_client.list_rjobs(
            base_url=args.rjob_base_url,
            creator=args.creator,
            page_size=args.page_size,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"RJob API HTTP {exc.code}: {body[:500]}", file=sys.stderr)
        return 1

    if not rjobs:
        print("No RJobs returned from API.")
        return 0

    if args.debug_dump_rjobs:
        print(json.dumps(rjobs[0], indent=2, ensure_ascii=False))
        print(f"\nTotal items: {len(rjobs)}")
        return 0

    selected: list[tuple[str, str, str]] = []
    for item in rjobs:
        name = _job_name(item)
        phase = _job_phase(item)
        created = _job_created(item)
        if args.include_non_running or phase.lower() == "running":
            selected.append((name, phase, created))

    if not selected:
        print("No running RJobs after filtering.")
        phases = sorted({_job_phase(x) for x in rjobs})
        print(f"Observed phases: {', '.join(phases) or 'none'}")
        return 0

    print(f"Analyzing {len(selected)} running RJob(s), window={args.days}d, step={args.step_minutes}m")
    results: list[RJobMetrics] = []
    for rjob_name, phase, created in selected:
        pods = prom.pods_for_rjob(rjob_name, args.namespace)
        if not pods:
            print(f"[WARN] {rjob_name}: no GPU pods found in Prometheus namespace={args.namespace}")
            results.append(
                RJobMetrics(
                    rjob_id=rjob_name,
                    rjob_name=rjob_name,
                    phase=phase,
                    created_at=created,
                    gpu_cards=0,
                )
            )
            continue

        pod_metrics = [
            collect_pod_metrics(
                prom,
                pod=pod,
                namespace=args.namespace,
                start=start,
                end=end,
                step_s=step_s,
            )
            for pod in pods
        ]
        results.append(aggregate_rjob_metrics(rjob_name, phase, created, pod_metrics))
        print(
            f"  {rjob_name}: pods={len(pods)} gpus={sum(p.gpu_count for p in pod_metrics)} "
            f"gpu_avg={results[-1].gpu_util_avg:.1f}% power_avg={results[-1].power_avg_w:.0f}W"
        )

    print_table(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON report: {args.output}")
    if args.csv:
        write_csv(args.csv, results)
        print(f"Wrote CSV summary: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

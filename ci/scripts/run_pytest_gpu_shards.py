#!/usr/bin/env python3
"""Run pytest on one multi-GPU node with isolated GPU pairs per worker.

Example (8 GPUs, 4 workers × 2 GPUs):
  CUDA 0,1 / 2,3 / 4,5 / 6,7 in parallel, each worker sets XTUNER_TEST_WORLD_SIZE=2.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Shard pytest across GPU-isolated workers")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel pytest processes (default: 4)",
    )
    parser.add_argument(
        "--gpus-per-worker",
        type=int,
        default=2,
        help="GPUs assigned to each worker via CUDA_VISIBLE_DEVICES (default: 2)",
    )
    parser.add_argument(
        "--master-port-base",
        type=int,
        default=29500,
        help="MASTER_PORT for worker i = base + i * 20",
    )
    parser.add_argument(
        "pytest_argv",
        nargs=argparse.REMAINDER,
        help="pytest args after '--', e.g. pytest -ra tests/utils",
    )
    args = parser.parse_args()
    pytest_argv = args.pytest_argv
    if pytest_argv and pytest_argv[0] == "--":
        pytest_argv = pytest_argv[1:]
    if not pytest_argv:
        parser.error("missing pytest command after '--'")
    if pytest_argv[0] == "pytest":
        pytest_argv = [sys.executable, "-m", "pytest", *pytest_argv[1:]]
    elif pytest_argv[0].endswith("pytest"):
        pytest_argv = [sys.executable, "-m", "pytest", *pytest_argv[1:]]
    return args, pytest_argv


def _collect_nodeids(pytest_argv: list[str]) -> list[str]:
    cmd = [*pytest_argv, "--collect-only", "-q"]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)

    nodeids: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        if " no tests " in line or line.endswith(" tests collected"):
            continue
        if "::" in line and ".py" in line.split("::", 1)[0]:
            nodeids.append(line.split()[0])
    return nodeids


def _bucket_by_file(nodeids: list[str], workers: int) -> list[list[str]]:
    by_file: dict[str, list[str]] = defaultdict(list)
    for nid in nodeids:
        by_file[nid.split("::", 1)[0]].append(nid)

    files = sorted(by_file.keys())
    buckets: list[list[str]] = [[] for _ in range(workers)]
    for idx, path in enumerate(files):
        buckets[idx % workers].extend(by_file[path])
    return buckets


def _worker_env(worker_id: int, gpus_per_worker: int, master_port_base: int) -> dict[str, str]:
    env = os.environ.copy()
    start = worker_id * gpus_per_worker
    devices = ",".join(str(start + i) for i in range(gpus_per_worker))
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["XTUNER_TEST_WORLD_SIZE"] = str(gpus_per_worker)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port_base + worker_id * 20)
    cache = f"/tmp/.pytest_cache_shard{worker_id}"
    env["PYTEST_ADDOPTS"] = re.sub(
        r"-o\s+cache_dir=\S+",
        "",
        env.get("PYTEST_ADDOPTS", ""),
    ).strip()
    extra = f"-o cache_dir={cache}"
    env["PYTEST_ADDOPTS"] = f"{env['PYTEST_ADDOPTS']} {extra}".strip()
    return env


def main() -> None:
    args, pytest_argv = _parse_args()
    workers = args.workers
    gpus_per_worker = args.gpus_per_worker
    needed_gpus = workers * gpus_per_worker

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != "":
        n_visible = len(visible.split(","))
    else:
        try:
            import torch

            n_visible = torch.cuda.device_count()
        except Exception:
            n_visible = needed_gpus

    if n_visible < needed_gpus:
        print(
            f"Need {needed_gpus} GPUs for {workers}×{gpus_per_worker} sharding, "
            f"but see {n_visible}; falling back to single-process pytest.",
            flush=True,
        )
        raise SystemExit(subprocess.call(pytest_argv))

    nodeids = _collect_nodeids(pytest_argv)
    if not nodeids:
        print("No tests collected; nothing to run.", flush=True)
        return

    buckets = _bucket_by_file(nodeids, workers)
    procs: list[tuple[int, subprocess.Popen[bytes]]] = []

    for worker_id, bucket in enumerate(buckets):
        if not bucket:
            continue
        env = _worker_env(worker_id, gpus_per_worker, args.master_port_base)
        cmd = [*pytest_argv, *bucket]
        log_path = Path(f"/tmp/pytest_gpu_shard_{worker_id}.log")
        print(
            f"[shard {worker_id}] GPUs={env['CUDA_VISIBLE_DEVICES']} "
            f"WORLD={env['XTUNER_TEST_WORLD_SIZE']} "
            f"PORT={env['MASTER_PORT']} "
            f"tests={len(bucket)} log={log_path}",
            flush=True,
        )
        log_fh = log_path.open("wb")
        procs.append(
            (
                worker_id,
                subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                ),
            )
        )

    exit_code = 0
    for worker_id, proc in procs:
        rc = proc.wait()
        log_path = Path(f"/tmp/pytest_gpu_shard_{worker_id}.log")
        if rc != 0:
            exit_code = rc
            print(f"=== shard {worker_id} failed (exit {rc}); log {log_path} ===", flush=True)
            try:
                sys.stdout.buffer.write(log_path.read_bytes())
            except OSError as exc:
                print(f"(could not read log: {exc})", flush=True)
        else:
            print(f"[shard {worker_id}] ok", flush=True)

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

"""Opt-in process and cgroup memory traces during SFT.

Set XTUNER_MEMORY_WATCH_ENABLED=1 for per-rank JSONL traces in work_dir/memory.
Set XTUNER_MEMORY_WATCH_INTERVAL to the sampling interval in seconds (default: 1).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator


_NUMA_PAGE_TOKEN = re.compile(r"N(\d+)=(\d+)")
_SAMPLE_INTERVAL_S = 1
_NUMA_SAMPLE_INTERVAL_NS = 30_000_000_000
_STATUS_KEYS = {
    "VmRSS",
    "VmHWM",
    "RssAnon",
    "RssFile",
    "RssShmem",
    "VmSwap",
    "Cpus_allowed_list",
    "Mems_allowed_list",
}
_NODE_MEMINFO_KEYS = {"MemTotal", "MemFree", "MemUsed", "FilePages", "AnonPages", "SReclaimable"}


def _read_status(path: Path) -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    for line in path.read_text().splitlines():
        key, _, raw = line.partition(":")
        if key not in _STATUS_KEYS:
            continue
        value = raw.strip()
        values[key] = int(value.split()[0]) * 1024 if value.endswith(" kB") else value
    return values


def _controller_dir(
    controller: str | None,
    marker: str,
    root: Path = Path("/sys/fs/cgroup"),
    membership: Path = Path("/proc/self/cgroup"),
) -> Path | None:
    """Locate a controller, including when a cgroup namespace mounts it at /.

    On cgroup v1 the memory and cpuset controllers can have different paths.
    """
    mount = root / controller if controller else root
    for line in membership.read_text().splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if controller is None:
            if parts[:2] != ["0", ""]:
                continue
        elif controller not in parts[1].split(","):
            continue
        candidate = mount / parts[2].lstrip("/")
        if (candidate / marker).exists():
            return candidate
    # A container commonly exposes its current cgroup directly at the mount.
    return mount if (mount / marker).exists() else None


def _cgroup_dir() -> Path | None:
    return _controller_dir(None, "memory.events") or _controller_dir("memory", "memory.usage_in_bytes")


def _read_numbers(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].isdigit():
            result[fields[0]] = int(fields[1])
    return result


def _read_numa_stat(path: Path) -> dict[str, dict[str, int]]:
    """Keep all NUMA counters, including v1 totals and hierarchical fields."""
    if not path.exists():
        return {}
    result: dict[str, dict[str, int]] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        key = fields[0]
        values: dict[str, int] = {}
        if "=" in key:  # cgroup v1: total=123 N0=... N1=...
            key, total = key.split("=", 1)
            values["total"] = int(total)
        for field in fields[1:]:
            node, sep, amount = field.partition("=")
            if sep:
                values[node] = int(amount)
        result[key] = values
    return result


def _read_smaps_rollup(path: Path) -> dict[str, int]:
    """Read aggregate mapping counters from /proc/<pid>/smaps_rollup in bytes."""
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, sep, raw = line.partition(":")
        if sep and raw.strip().endswith(" kB"):
            values[key] = int(raw.split()[0]) * 1024
    return values


def _read_numa_maps(path: Path) -> dict[str, Any]:
    """Summarize this rank's resident pages by NUMA node and mapping policy."""
    resident_pages: dict[str, int] = {}
    resident_bytes: dict[str, int] = {}
    policy_mappings: dict[str, int] = {}
    base_page_bytes = os.sysconf("SC_PAGE_SIZE")
    with path.open() as maps:
        for line in maps:
            fields = line.split()
            if len(fields) < 2:
                continue
            policy = fields[1]
            policy_mappings[policy] = policy_mappings.get(policy, 0) + 1
            page_bytes = base_page_bytes
            for field in fields[2:]:
                if field.startswith("kernelpagesize_kB="):
                    page_bytes = int(field.split("=", 1)[1]) * 1024
                    break
            for field in fields[2:]:
                if match := _NUMA_PAGE_TOKEN.fullmatch(field):
                    node = f"N{match.group(1)}"
                    pages = int(match.group(2))
                    resident_pages[node] = resident_pages.get(node, 0) + pages
                    resident_bytes[node] = resident_bytes.get(node, 0) + pages * page_bytes
    return {
        "resident_pages_by_node": resident_pages,
        "resident_bytes_by_node": resident_bytes,
        "policy_mappings": policy_mappings,
    }


def _read_node_meminfo(root: Path = Path("/sys/devices/system/node")) -> dict[str, dict[str, int]]:
    """Read host-wide memory counters; these are not a pod's NUMA quota."""
    result: dict[str, dict[str, int]] = {}
    for path in sorted(root.glob("node[0-9]*/meminfo")):
        values: dict[str, int] = {}
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 5 and fields[2].rstrip(":") in _NODE_MEMINFO_KEYS:
                values[fields[2].rstrip(":")] = int(fields[3]) * 1024
        result[path.parent.name] = values
    return result


def _read_node_cpulists(root: Path = Path("/sys/devices/system/node")) -> dict[str, str]:
    return {path.parent.name: path.read_text().strip() for path in sorted(root.glob("node[0-9]*/cpulist"))}


def _read_cgroup(cgroup: Path | None) -> dict[str, Any]:
    if cgroup is None:
        return {"available": False}
    version = 2 if (cgroup / "memory.events").exists() else 1 if (cgroup / "memory.usage_in_bytes").exists() else None
    if version is None:
        return {"available": False, "path": str(cgroup)}
    values: dict[str, Any] = {"available": True, "version": version, "path": str(cgroup)}
    scalar_names = (
        ("memory.current", "memory.max", "memory.peak", "memory.swap.current", "memory.swap.max", "memory.swap.peak")
        if version == 2
        else (
            "memory.usage_in_bytes",
            "memory.max_usage_in_bytes",
            "memory.limit_in_bytes",
            "memory.failcnt",
            "memory.memsw.usage_in_bytes",
            "memory.memsw.max_usage_in_bytes",
            "memory.memsw.limit_in_bytes",
            "memory.memsw.failcnt",
        )
    )
    for name in scalar_names:
        path = cgroup / name
        if path.exists():
            values[name] = path.read_text().strip()
    if version == 2:
        for name in ("memory.events", "memory.events.local", "memory.swap.events"):
            values[name] = _read_numbers(cgroup / name)
        cpuset = cgroup
        cpuset_names = ("cpuset.cpus", "cpuset.mems", "cpuset.cpus.effective", "cpuset.mems.effective")
    else:
        values["memory.oom_control"] = _read_numbers(cgroup / "memory.oom_control")
        cpuset = _controller_dir("cpuset", "cpuset.cpus")
        cpuset_names = ("cpuset.cpus", "cpuset.mems", "cpuset.effective_cpus", "cpuset.effective_mems")
    if cpuset is not None:
        values["cpuset_path"] = str(cpuset)
        for name in cpuset_names:
            path = cpuset / name
            if path.exists():
                values[name] = path.read_text().strip()
    values["memory.stat"] = _read_numbers(cgroup / "memory.stat")
    numa_path = cgroup / "memory.numa_stat"
    if numa_path.exists():
        values["memory.numa_stat"] = _read_numa_stat(numa_path)
        values["memory.numa_stat_unit"] = "bytes" if version == 2 else "pages"
    return values


def _oom_kill_count(cgroup: dict[str, Any]) -> int | None:
    events = cgroup.get("memory.events") if cgroup.get("version") == 2 else cgroup.get("memory.oom_control")
    return events.get("oom_kill") if isinstance(events, dict) else None


class MemoryWatch:
    def __init__(
        self,
        output: Path,
        *,
        include_cgroup: bool,
        interval_s: float = _SAMPLE_INTERVAL_S,
    ) -> None:
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("Memory sampling interval must be a finite positive number of seconds")
        self.output = output
        self.interval_s = interval_s
        self.include_cgroup = include_cgroup
        self.cgroup = _cgroup_dir() if include_cgroup else None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="xtuner-memory-watch", daemon=True)
        self.fd: int | None = None
        self.previous_oom_kill: int | None = None
        self.previous_numa_sample_ns = 0
        self.sample_error_reported = False

    def _report_sample_error(self, exc: Exception) -> None:
        with self.lock:
            if self.sample_error_reported:
                return
            self.sample_error_reported = True
        print(f"[XTuner memory watch] sampling failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def __enter__(self) -> "MemoryWatch":
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.output, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
        self.sample("start")
        self.thread.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=_SAMPLE_INTERVAL_S + 1)
        try:
            self.sample("stop")
        except Exception as exc:
            self._report_sample_error(exc)
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            try:
                self.sample("periodic")
            except Exception as exc:
                self._report_sample_error(exc)
                return

    def sample(self, event: str) -> None:
        now_ns = time.monotonic_ns()
        with self.lock:
            read_numa = event != "periodic" or now_ns - self.previous_numa_sample_ns >= _NUMA_SAMPLE_INTERVAL_NS
            if read_numa:
                self.previous_numa_sample_ns = now_ns
        record: dict[str, Any] = {
            "time_ns": time.time_ns(),
            "monotonic_ns": now_ns,
            "event": event,
            "pid": os.getpid(),
            "rank": os.environ.get("RANK"),
            "local_rank": os.environ.get("LOCAL_RANK"),
        }
        if event == "start":
            record["host"] = os.uname().nodename
            try:
                # starttime is field 22 of /proc/self/stat, in clock ticks.
                stat_fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
                record["pid_starttime_ticks"] = int(stat_fields[19])
                record["cgroup_membership"] = Path("/proc/self/cgroup").read_text().splitlines()
            except (OSError, ValueError, IndexError) as exc:
                record["identity_error"] = str(exc)
            try:
                record["host_node_cpus"] = _read_node_cpulists()
            except (OSError, ValueError) as exc:
                record["host_node_cpus_error"] = str(exc)
            if self.include_cgroup:
                try:
                    record["cgroup_mounts"] = [
                        line
                        for line in Path("/proc/self/mountinfo").read_text().splitlines()
                        if " - cgroup " in line or " - cgroup2 " in line
                    ]
                except OSError as exc:
                    record["cgroup_mounts_error"] = str(exc)
        try:
            # /proc/self addresses the thread-group leader, even when the
            # watcher's background thread takes this sample.
            record["process"] = _read_status(Path("/proc/self/status"))
        except (OSError, ValueError) as exc:
            record["process_error"] = str(exc)
        if read_numa:
            try:
                record["numa_maps"] = _read_numa_maps(Path("/proc/self/numa_maps"))
            except (OSError, ValueError) as exc:
                record["numa_maps_error"] = str(exc)
            try:
                record["smaps_rollup"] = _read_smaps_rollup(Path("/proc/self/smaps_rollup"))
            except (OSError, ValueError) as exc:
                record["smaps_rollup_error"] = str(exc)
        if self.include_cgroup:
            try:
                record["cgroup"] = _read_cgroup(self.cgroup)
                oom_kill = _oom_kill_count(record["cgroup"])
                if oom_kill is not None:
                    with self.lock:
                        if self.previous_oom_kill is not None and oom_kill > self.previous_oom_kill:
                            record["oom_kill_delta"] = oom_kill - self.previous_oom_kill
                        self.previous_oom_kill = oom_kill
            except (OSError, ValueError) as exc:
                record["cgroup_error"] = str(exc)
            try:
                record["host_nodes"] = _read_node_meminfo()
            except (OSError, ValueError) as exc:
                record["host_nodes_error"] = str(exc)
        payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        with self.lock:
            if self.fd is not None:
                os.write(self.fd, payload)


@contextlib.contextmanager
def monitor_rank_memory(work_dir: Path | str) -> Iterator[None]:
    if os.environ.get("XTUNER_MEMORY_WATCH_ENABLED") != "1":
        yield
        return
    output_dir = (Path(work_dir) / "memory").resolve()
    rank = os.environ.get("RANK", "unknown")
    local_rank = os.environ.get("LOCAL_RANK", "unknown")
    watcher = MemoryWatch(
        output_dir / f"rank_{rank}.jsonl",
        include_cgroup=local_rank == "0",
        interval_s=float(os.environ.get("XTUNER_MEMORY_WATCH_INTERVAL", "1")),
    )
    with watcher:
        yield

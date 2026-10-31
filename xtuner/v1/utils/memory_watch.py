"""Opt-in process and cgroup memory traces during SFT.

Set XTUNER_MEMORY_WATCH_ENABLED=1 for per-rank and host-level JSONL traces in work_dir/memory. Set
XTUNER_MEMORY_WATCH_INTERVAL to the sampling interval in seconds (default: 1).
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
from typing import Any, Iterator, cast

from typing_extensions import NotRequired, TypedDict

from xtuner.v1.utils.enum_helper import StrEnum


_NUMA_PAGE_TOKEN = re.compile(r"N(\d+)=(\d+)")
_SAMPLE_INTERVAL_S = 1
_SHUTDOWN_TIMEOUT_S = 2.0
_NUMA_SAMPLE_INTERVAL_NS = 30_000_000_000


class MemoryEvent(StrEnum):
    START = "start"
    PERIODIC = "periodic"
    STOP = "stop"


class MemoryRecord(TypedDict):
    time_ns: int
    monotonic_ns: int
    event: str
    host: str
    pid: int
    rank: str | None
    local_rank: str | None


class _RankSections(TypedDict, total=False):
    pid_starttime_ticks: int
    cgroup_membership: list[str]
    identity_error: str
    process: dict[str, int | str]
    process_error: str
    numa_maps: dict[str, Any]
    numa_maps_error: str
    smaps_rollup: dict[str, int]
    smaps_rollup_error: str


class RankMemoryRecord(MemoryRecord, _RankSections):
    pass


class _HostSections(TypedDict):
    host_node_cpus: NotRequired[dict[str, str]]
    host_node_cpus_error: NotRequired[str]
    cgroup_mounts: NotRequired[list[str]]
    cgroup_mounts_error: NotRequired[str]
    cgroup: NotRequired[dict[str, Any]]
    cgroup_error: NotRequired[str]
    host_nodes: NotRequired[dict[str, dict[str, int]]]
    host_nodes_error: NotRequired[str]
    oom_kill_delta: NotRequired[int]


class HostMemoryRecord(MemoryRecord, _HostSections):
    pass


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


def _cgroup_dir(root: Path = Path("/sys/fs/cgroup"), membership: Path = Path("/proc/self/cgroup")) -> Path | None:
    return _controller_dir(None, "memory.events", root, membership) or _controller_dir(
        "memory", "memory.usage_in_bytes", root, membership
    )


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
    """Read aggregate mapping counters from /proc/<pid>/smaps_rollup in
    bytes."""
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


def _read_cgroup(
    cgroup: Path | None,
    *,
    root: Path = Path("/sys/fs/cgroup"),
    membership: Path = Path("/proc/self/cgroup"),
) -> dict[str, Any]:
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
        cpuset: Path | None = cgroup
        cpuset_names = ("cpuset.cpus", "cpuset.mems", "cpuset.cpus.effective", "cpuset.mems.effective")
    else:
        values["memory.oom_control"] = _read_numbers(cgroup / "memory.oom_control")
        cpuset = _controller_dir("cpuset", "cpuset.cpus", root, membership)
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


class _MemoryWatch:
    """Own a rank's memory sampling thread and JSONL outputs.

    Args:
        output (Path): Destination rank JSONL file, opened in append mode.
        include_cgroup (bool): Write cgroup and host node metrics to a separate host JSONL file.
        interval_s (float): Finite positive interval between periodic samples.
        proc_self (Path): Process-local procfs directory.
        cgroup_root (Path): Root of the cgroup mounts.
        node_root (Path): Root of the host NUMA node directories.
    """

    def __init__(
        self,
        output: Path,
        *,
        include_cgroup: bool,
        interval_s: float = _SAMPLE_INTERVAL_S,
        proc_self: Path = Path("/proc/self"),
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        node_root: Path = Path("/sys/devices/system/node"),
    ) -> None:
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("Memory sampling interval must be a finite positive number of seconds")
        self._output = output
        self._interval_s = interval_s
        self._include_cgroup = include_cgroup
        self._proc_self = proc_self
        self._cgroup_root = cgroup_root
        self._node_root = node_root
        self._cgroup = _cgroup_dir(cgroup_root, proc_self / "cgroup") if include_cgroup else None
        self._host = os.uname().nodename
        self._host_output = output.parent / f"host_{self._host}.jsonl"
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="xtuner-memory-watch", daemon=True)
        self._fd: int | None = None
        self._host_fd: int | None = None
        self._previous_oom_kill: int | None = None
        self._previous_numa_sample_ns = 0
        self._sample_error_reported = False

    def __enter__(self) -> _MemoryWatch:
        """Open the traces, record initial metrics, and start periodic
        sampling.

        Returns:
            _MemoryWatch: The active monitor owned by this context.
        """
        try:
            self._output.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(self._output, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
            if self._include_cgroup:
                self._host_fd = os.open(self._host_output, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
            self._sample(MemoryEvent.START)
            self._thread.start()
        except Exception:
            self._close()
            raise
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        """Stop periodic writes, record final metrics, and close the traces.

        Args:
            _exc_type (object): Exception type supplied by the context manager.
            _exc (object): Exception instance supplied by the context manager.
            _tb (object): Traceback supplied by the context manager.
        """
        self._stop_event.set()
        try:
            # Event.wait wakes immediately on set, regardless of the configured
            # interval. The timeout only bounds an in-flight sample.
            self._thread.join(timeout=_SHUTDOWN_TIMEOUT_S)
        except RuntimeError as exc:
            self._report_sample_error(exc)
        try:
            self._sample(MemoryEvent.STOP)
        except Exception as exc:
            self._report_sample_error(exc)
        finally:
            self._close()

    def _close(self) -> None:
        with self._lock:
            fd, self._fd = self._fd, None
            host_fd, self._host_fd = self._host_fd, None
        for output_fd in (fd, host_fd):
            if output_fd is not None:
                try:
                    os.close(output_fd)
                except OSError as exc:
                    self._report_sample_error(exc)

    def _report_sample_error(self, exc: Exception) -> None:
        with self._lock:
            if self._sample_error_reported:
                return
            self._sample_error_reported = True
        print(f"[XTuner memory watch] sampling failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self._sample(MemoryEvent.PERIODIC)
            except Exception as exc:
                self._report_sample_error(exc)
                return

    def _base_record(self, event: MemoryEvent, now_ns: int) -> MemoryRecord:
        return MemoryRecord(
            time_ns=time.time_ns(),
            monotonic_ns=now_ns,
            event=event.value,
            host=self._host,
            pid=os.getpid(),
            rank=os.environ.get("RANK"),
            local_rank=os.environ.get("LOCAL_RANK"),
        )

    def _claim_numa_slot(self, event: MemoryEvent, now_ns: int) -> bool:
        with self._lock:
            read_numa = (
                event != MemoryEvent.PERIODIC or now_ns - self._previous_numa_sample_ns >= _NUMA_SAMPLE_INTERVAL_NS
            )
            if read_numa:
                self._previous_numa_sample_ns = now_ns
            return read_numa

    def _read_process_sections(self, event: MemoryEvent, read_numa: bool) -> _RankSections:
        sections = _RankSections()
        if event == MemoryEvent.START:
            try:
                # starttime is field 22 of /proc/self/stat, in clock ticks.
                stat_fields = (self._proc_self / "stat").read_text().rsplit(")", 1)[1].split()
                sections["pid_starttime_ticks"] = int(stat_fields[19])
                sections["cgroup_membership"] = (self._proc_self / "cgroup").read_text().splitlines()
            except (OSError, ValueError, IndexError) as exc:
                sections["identity_error"] = str(exc)
        try:
            # /proc/self addresses the thread-group leader, even when the
            # watcher's background thread takes this sample.
            sections["process"] = _read_status(self._proc_self / "status")
        except (OSError, ValueError) as exc:
            sections["process_error"] = str(exc)
        if read_numa:
            try:
                sections["numa_maps"] = _read_numa_maps(self._proc_self / "numa_maps")
            except (OSError, ValueError) as exc:
                sections["numa_maps_error"] = str(exc)
            try:
                sections["smaps_rollup"] = _read_smaps_rollup(self._proc_self / "smaps_rollup")
            except (OSError, ValueError) as exc:
                sections["smaps_rollup_error"] = str(exc)
        return sections

    def _read_host_sections(self, event: MemoryEvent) -> _HostSections:
        sections = _HostSections()
        if event == MemoryEvent.START:
            try:
                sections["host_node_cpus"] = _read_node_cpulists(self._node_root)
            except (OSError, ValueError) as exc:
                sections["host_node_cpus_error"] = str(exc)
            try:
                sections["cgroup_mounts"] = [
                    line
                    for line in (self._proc_self / "mountinfo").read_text().splitlines()
                    if " - cgroup " in line or " - cgroup2 " in line
                ]
            except OSError as exc:
                sections["cgroup_mounts_error"] = str(exc)
        try:
            cgroup = _read_cgroup(self._cgroup, root=self._cgroup_root, membership=self._proc_self / "cgroup")
            sections["cgroup"] = cgroup
            oom_kill = _oom_kill_count(cgroup)
            if oom_kill is not None:
                with self._lock:
                    if self._previous_oom_kill is not None and oom_kill > self._previous_oom_kill:
                        sections["oom_kill_delta"] = oom_kill - self._previous_oom_kill
                    self._previous_oom_kill = oom_kill
        except (OSError, ValueError) as exc:
            sections["cgroup_error"] = str(exc)
        try:
            sections["host_nodes"] = _read_node_meminfo(self._node_root)
        except (OSError, ValueError) as exc:
            sections["host_nodes_error"] = str(exc)
        return sections

    def _write_record(self, record: MemoryRecord, event: MemoryEvent, *, host: bool = False) -> None:
        payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        with self._lock:
            fd = self._host_fd if host else self._fd
            if fd is not None and (event != MemoryEvent.PERIODIC or not self._stop_event.is_set()):
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(fd, remaining)
                    if written == 0:
                        raise OSError("Memory watch write returned zero bytes")
                    remaining = remaining[written:]

    def _sample(self, event: MemoryEvent) -> None:
        now_ns = time.monotonic_ns()
        rank_record = cast(RankMemoryRecord, self._base_record(event, now_ns))
        rank_record.update(self._read_process_sections(event, self._claim_numa_slot(event, now_ns)))
        self._write_record(rank_record, event)
        if self._include_cgroup:
            host_record = cast(HostMemoryRecord, self._base_record(event, now_ns))
            host_record.update(self._read_host_sections(event))
            self._write_record(host_record, event, host=True)


@contextlib.contextmanager
def monitor_rank_memory(work_dir: Path | str | None) -> Iterator[None]:
    """Monitor rank memory during SFT when enabled through the environment.

    Write process records to ``memory/rank_<rank>.jsonl`` under the work
    directory. Only local rank zero writes cgroup and host node records to
    ``memory/host_<hostname>.jsonl``. Set ``XTUNER_MEMORY_WATCH_ENABLED=1`` to
    enable monitoring and ``XTUNER_MEMORY_WATCH_INTERVAL`` to a positive
    interval in seconds. An invalid interval or an initialization failure logs
    a warning and lets training continue without a trace.

    Args:
        work_dir (Path | str | None): Training directory containing the memory traces.

    Yields:
        None: Control to the training code while monitoring is active, or
            directly when monitoring is disabled.
    """
    if os.environ.get("XTUNER_MEMORY_WATCH_ENABLED") != "1":
        yield
        return
    with contextlib.ExitStack() as stack:
        try:
            if work_dir is None:
                work_dir = Path.cwd() / "work_dir"
            output_dir = (Path(work_dir) / "memory").resolve()
            rank = os.environ.get("RANK", "unknown")
            local_rank = os.environ.get("LOCAL_RANK", "unknown")
            watcher = _MemoryWatch(
                output_dir / f"rank_{rank}.jsonl",
                include_cgroup=local_rank == "0",
                interval_s=float(os.environ.get("XTUNER_MEMORY_WATCH_INTERVAL", _SAMPLE_INTERVAL_S)),
            )
            stack.enter_context(watcher)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"[XTuner memory watch] disabled: {type(exc).__name__}: {exc}", file=sys.stderr)
        yield

"""Discover GPU-local NUMA resources and plan rank binding."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast


class NumaRankPlan(TypedDict):
    """GPU-local CPU and memory binding selected for a rank."""

    gpu_uuid: str
    numa_id: int
    cpu_ids: set[int]
    mems_allowed: set[int]


class _NumaPlanReport(TypedDict):
    """Local binding plan or skip status exchanged between training ranks."""

    host: str
    rank: int
    local_rank: int | None
    plan: NumaRankPlan | None
    error: str | None
    skip_reason: str | None


def plan_numa_binding(initial_cpu_affinity: set[int]) -> dict[int, NumaRankPlan]:
    """Discover resources and plan CPU and memory binding for this host.

    All training ranks must call this function in the same collective order,
    including ranks that disable binding or use a non-CUDA device. Each rank
    reports its GPU-local CPU set and memory permissions. Any local discovery
    failure or insufficient CPU capacity causes the whole host to skip binding
    through a ValueError, before any affinity is changed.

    Args:
        initial_cpu_affinity (set[int]): Calling thread CPU affinity captured
            by the caller before binding, also retained by the caller for rollback.

    Returns:
        dict[int, NumaRankPlan]: Local rank to GPU UUID, target NUMA node,
            available CPUs and allowed memory nodes.
            CPU sets may overlap.
    """
    import os

    from torch import distributed

    from .device import get_device

    host = os.uname().nodename
    report: _NumaPlanReport = {
        "host": host,
        "rank": distributed.get_rank(),
        "local_rank": None,
        "plan": None,
        "error": None,
        "skip_reason": None,
    }
    try:
        if os.getenv("XTUNER_DISABLE_NUMA", "0") == "1":
            report["skip_reason"] = "XTUNER_DISABLE_NUMA=1"
        elif get_device() != "cuda":
            report["skip_reason"] = "current device is not cuda"
        else:
            report["local_rank"] = int(os.environ["LOCAL_RANK"])
            report["plan"] = _get_rank_numa_plan(initial_cpu_affinity)
    except Exception as exc:
        report["error"] = f"rank {report['rank']}: {type(exc).__name__}: {exc}"

    reports: list[_NumaPlanReport | None] = [None] * distributed.get_world_size()
    distributed.all_gather_object(reports, report)
    local_reports = [entry for entry in cast(list[_NumaPlanReport], reports) if entry["host"] == host]
    for entry in local_reports:
        if entry["skip_reason"] is not None:
            raise ValueError(f"rank {entry['rank']}: {entry['skip_reason']}")
        if entry["error"] is not None:
            raise ValueError(f"GPU NUMA topology unavailable on this host; {entry['error']}")
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
    if local_world_size is not None and len(local_reports) != int(local_world_size):
        raise ValueError("host topology reports do not match LOCAL_WORLD_SIZE")
    plan = {cast(int, entry["local_rank"]): cast(NumaRankPlan, entry["plan"]) for entry in local_reports}
    if not plan or sorted(plan) != list(range(len(plan))):
        raise ValueError("local rank reports must cover each local rank exactly once")
    rank_counts: dict[int, int] = {}
    for binding in plan.values():
        node = binding["numa_id"]
        rank_counts[node] = rank_counts.get(node, 0) + 1
    for rank, binding in plan.items():
        node = binding["numa_id"]
        if len(binding["cpu_ids"]) < rank_counts[node]:
            raise ValueError(
                f"GPU-local CPU capacity is insufficient: local rank {rank} has "
                f"{len(binding['cpu_ids'])} available CPUs on node {node} for {rank_counts[node]} local ranks"
            )
    return dict(sorted(plan.items()))


def parse_id_list(value: str) -> set[int]:
    """Expand a Linux CPU or NUMA node list into individual IDs.

    Args:
        value (str): Comma-separated IDs and inclusive ranges, as used by
            sysfs ``cpulist`` and ``Mems_allowed_list``.

    Returns:
        set[int]: The IDs in the list.

    Example:
        >>> parse_id_list("0-2,5")
        {0, 1, 2, 5}
    """
    result: set[int] = set()
    for item in value.strip().split(","):
        if not item:
            continue
        first, separator, last = item.partition("-")
        if separator:
            start, end = int(first), int(last)
            if end < start:
                raise ValueError(f"Invalid CPU or NUMA node range: {item}")
            result.update(range(start, end + 1))
        else:
            result.add(int(first))
    return result


def format_id_list(ids: set[int]) -> str:
    """Format CPU or NUMA node IDs as compact Linux-style ranges.

    Args:
        ids (set[int]): Nonnegative CPU or NUMA node IDs.

    Returns:
        str: Comma-separated IDs and inclusive ranges, or an empty string for
            an empty set.

    Example:
        >>> format_id_list({6, 7, 8, 12})
        '6-8,12'
    """
    if not ids:
        return ""
    ordered = sorted(ids)
    ranges: list[str] = []
    start = end = ordered[0]
    for value in ordered[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append(str(start) if start == end else f"{start}-{end}")
        start = end = value
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(ranges)


def get_available_cpus_by_numa_node(allowed_cpus: set[int]) -> dict[int, set[int]]:
    """Group the calling thread's allowed logical CPUs by host NUMA node.

    Args:
        allowed_cpus (set[int]): Calling thread's effective CPU affinity,
            normally from ``os.sched_getaffinity(0)``.

    Returns:
        dict[int, set[int]]: NUMA node ID to CPUs allowed both by the job and
            by that node's host topology. Nodes with no allowed CPUs retain
            an empty set; only GPU-local nodes participate in binding checks.

    Example:
        If node 0 has CPUs ``{0, 1}``, node 1 has ``{2, 3}``, and
        ``allowed_cpus`` is ``{1, 2}``, the result is ``{0: {1}, 1: {2}}``.
    """
    from numa import info

    result: dict[int, set[int]] = {}
    mapped: set[int] = set()
    for node_dir in Path("/sys/devices/system/node").glob("node[0-9]*"):
        node = int(node_dir.name.removeprefix("node"))
        node_cpus = allowed_cpus & set(info.node_to_cpus(node))
        duplicate_cpus = node_cpus & mapped
        if duplicate_cpus:
            raise ValueError(f"CPUs appear in more than one NUMA node: {sorted(duplicate_cpus)}")
        mapped.update(node_cpus)
        result[node] = node_cpus
    missing_cpus = allowed_cpus - mapped
    if missing_cpus:
        raise ValueError(f"NUMA topology does not cover allowed CPUs: {sorted(missing_cpus)}")
    return result


def _get_rank_numa_plan(initial_cpu_affinity: set[int]) -> NumaRankPlan:
    from numa import memory
    from torch import cuda

    cpus_by_node = get_available_cpus_by_numa_node(initial_cpu_affinity)
    mems_allowed = set(memory.get_allocation_allowed_nodes())
    gpu_uuid = str(cuda.get_device_properties(cuda.current_device()).uuid)
    gpu_cpus, gpu_memory_nodes = _get_gpu_affinity(gpu_uuid)
    node = next(iter(gpu_memory_nodes))
    cpus = cpus_by_node.get(node, set()) & gpu_cpus
    if node not in mems_allowed or not cpus:
        raise ValueError(f"GPU {gpu_uuid} has no accessible GPU-local CPU/memory node")
    return {
        "gpu_uuid": gpu_uuid,
        "numa_id": node,
        "cpu_ids": cpus,
        "mems_allowed": mems_allowed,
    }


def _get_gpu_affinity(gpu_uuid: str) -> tuple[set[int], set[int]]:
    """Query GPU-local CPUs and the PCI NUMA node using legacy NVML bindings.

    Args:
        gpu_uuid (str): UUID of the actual CUDA device.

    Returns:
        tuple[set[int], set[int]]: GPU CPU affinity and the singleton NUMA node
            attached to its PCI device. The CPU mask covers all host possible
            CPU IDs; the planner intersects it with rank permissions. Missing
            PCI topology or an unknown node prevents binding on this host.
    """
    import ctypes

    import pynvml

    word_bits = ctypes.sizeof(ctypes.c_ulonglong) * 8
    host_cpus = parse_id_list(Path("/sys/devices/system/cpu/possible").read_text())
    cpu_words = (max(host_cpus) + word_bits) // word_bits
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid.encode("ascii"))
        cpu_mask = pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_words)
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        domain, bus, device = pci_info.busId.decode("ascii").split(":")
        pci_bus_id = f"{int(domain, 16):04x}:{bus.lower()}:{device.lower()}"
        numa_id = int((Path("/sys/bus/pci/devices") / pci_bus_id / "numa_node").read_text())
        return _mask_ids(cpu_mask, word_bits), {numa_id}
    finally:
        pynvml.nvmlShutdown()


def _mask_ids(mask: list[int], word_bits: int) -> set[int]:
    return {
        word * word_bits + bit for word, value in enumerate(mask) for bit in range(word_bits) if value & (1 << bit)
    }

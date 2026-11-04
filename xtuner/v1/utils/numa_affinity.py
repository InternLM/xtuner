"""Discover GPU-local NUMA resources and plan rank binding."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import TypedDict

from .device import get_device
from .logger import get_logger


logger = get_logger()


class NumaRankPlan(TypedDict):
    """GPU-local CPU and memory binding selected for a rank."""

    gpu_uuid: str
    numa_id: int
    cpu_ids: set[int]


class NumaTopology(TypedDict):
    """Host hardware topology, independent of rank permissions."""

    node_cpus: dict[int, set[int]]
    gpu_nodes: dict[str, int]


def try_bind_numa(rank: int) -> None:
    """Plan and apply this rank's NUMA binding without raising exceptions.

    Each rank independently checks its allowed CPU count on every GPU node
    against all GPUs discovered in driver procfs. Discovery or binding failures
    only skip the current rank; ranks with identical CPU permissions make the
    same capacity decision.

    Args:
        rank (int): Global training rank used in log messages.
    """
    initial_cpu_affinity: set[int] = set()
    cpu_bound = False
    try:
        if os.getenv("XTUNER_DISABLE_NUMA", "0") == "1":
            raise ValueError("XTUNER_DISABLE_NUMA=1")
        if get_device() != "cuda":
            raise ValueError("current device is not cuda")

        initial_cpu_affinity = set(os.sched_getaffinity(0))
        topology = _get_numa_topology()
        # Count CPUs within this rank's initial affinity on every GPU node, not just its own node.
        # Example: with 8 discovered GPUs split 4/4 across nodes 0/1, CPU permissions of 3/44
        # make even a node-1 rank skip because node 0 has only 3 CPUs for 4 GPUs; 4/44 passes.
        # All ranks make the same capacity decision only when they have identical CPU permissions.
        for node, gpu_count in Counter(topology["gpu_nodes"].values()).items():
            cpu_count = len(topology["node_cpus"].get(node, set()) & initial_cpu_affinity)
            if cpu_count < gpu_count:
                raise ValueError(f"NUMA node {node} has {cpu_count} available CPUs for {gpu_count} GPUs")

        binding = _get_rank_numa_plan(initial_cpu_affinity, topology)

        from numa import memory

        os.sched_setaffinity(0, binding["cpu_ids"])
        cpu_bound = True
        memory.set_membind_nodes(binding["numa_id"])
    except Exception as exc:
        if cpu_bound:
            try:
                os.sched_setaffinity(0, initial_cpu_affinity)
            except OSError as restore_exc:
                logger.info(f"Rank: {rank} failed to restore CPU affinity: {restore_exc}.")
        logger.info(f"Rank: {rank} skip numa binding: {type(exc).__name__}: {exc}.")
        return

    logger.info(
        f"Rank: {rank} success bind process to numa node: {binding['numa_id']}; "
        f"requested_cpu_affinity={format_id_list(binding['cpu_ids'])}, "
        f"memory_policy=bind:{binding['numa_id']}."
    )


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


def _get_numa_topology() -> NumaTopology:
    from numa import info

    node_cpus: dict[int, set[int]] = {}
    for node_dir in Path("/sys/devices/system/node").glob("node[0-9]*"):
        node = int(node_dir.name.removeprefix("node"))
        node_cpus[node] = set(info.node_to_cpus(node))

    gpu_nodes: dict[str, int] = {}
    for gpu_uuid, pci_bus_id in _get_gpu_pci_bus_ids().items():
        node = int((Path("/sys/bus/pci/devices") / pci_bus_id / "numa_node").read_text())
        if node < 0:
            raise ValueError(f"GPU {gpu_uuid} NUMA node is unknown")
        gpu_nodes[gpu_uuid] = node
    return {"node_cpus": node_cpus, "gpu_nodes": gpu_nodes}


def _get_rank_numa_plan(initial_cpu_affinity: set[int], topology: NumaTopology) -> NumaRankPlan:
    from numa import memory
    from torch import cuda

    device = cuda.get_device_properties(cuda.current_device())
    # PyTorch formats the raw UUID without the GPU- prefix used by driver procfs.
    gpu_uuid = f"GPU-{device.uuid}"
    if gpu_uuid not in topology["gpu_nodes"]:
        raise ValueError(f"Cannot find PCI device for GPU {gpu_uuid}")
    node = topology["gpu_nodes"][gpu_uuid]
    cpus = topology["node_cpus"][node] & initial_cpu_affinity
    mems_allowed = set(memory.get_allocation_allowed_nodes())

    if node not in mems_allowed or not cpus:
        raise ValueError(f"GPU {gpu_uuid} has no accessible GPU-local CPU/memory node")
    return {
        "gpu_uuid": gpu_uuid,
        "numa_id": node,
        "cpu_ids": cpus,
    }


def _get_gpu_pci_bus_ids() -> dict[str, str]:
    """Build a GPU UUID to PCI BDF mapping from driver procfs.

    Returns:
        dict[str, str]: GPU UUIDs mapped to their PCI device directory names.
    """
    pci_bus_ids: dict[str, str] = {}
    for info_path in Path("/proc/driver/nvidia/gpus").glob("*/information"):
        gpu_uuid = ""
        for line in info_path.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "GPU UUID":
                gpu_uuid = value.strip()
                break
        # The driver may omit UUIDs on query failure; do not undercount GPUs in the capacity check.
        if not gpu_uuid:
            raise ValueError(f"Missing GPU UUID in {info_path}")
        pci_bus_ids[gpu_uuid] = info_path.parent.name
    return pci_bus_ids

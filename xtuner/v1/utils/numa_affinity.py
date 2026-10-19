"""Read host NUMA topology and the CPUs available to this process."""

from __future__ import annotations

from pathlib import Path


def parse_id_list(value: str) -> set[int]:
    """Expand a Linux CPU or NUMA node list into individual IDs.

    Args:
        value (str): Comma-separated IDs and inclusive ranges, as used by
            sysfs ``cpulist`` and ``Mems_allowed_list``.

    Returns:
        set[int]: The IDs in the list.

    Raises:
        ValueError: If an ID is invalid or a range ends before it starts.

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


def allowed_cpus_by_node(
    allowed_cpus: set[int], node_root: Path = Path("/sys/devices/system/node")
) -> dict[int, set[int]]:
    """Group the calling thread's allowed logical CPUs by host NUMA node.

    Args:
        allowed_cpus (set[int]): Calling thread's effective CPU affinity,
            normally from ``os.sched_getaffinity(0)``.
        node_root (Path): Directory containing each host node's ``cpulist``.

    Returns:
        dict[int, set[int]]: NUMA node ID to CPUs allowed both by the job and
            by that node's host topology. Nodes with no allowed CPUs retain
            an empty set so they still participate in rank distribution checks.

    Raises:
        ValueError: If a CPU belongs to multiple nodes or no discovered node.
        OSError: If a node's ``cpulist`` cannot be read.

    Example:
        If node 0 has CPUs ``{0, 1}``, node 1 has ``{2, 3}``, and
        ``allowed_cpus`` is ``{1, 2}``, the result is ``{0: {1}, 1: {2}}``.

    """
    result: dict[int, set[int]] = {}
    mapped: set[int] = set()
    for node_dir in node_root.glob("node[0-9]*"):
        cpulist = node_dir / "cpulist"
        if not cpulist.exists():
            continue
        node_cpus = allowed_cpus & parse_id_list(cpulist.read_text())
        if node_cpus:
            if node_cpus & mapped:
                raise ValueError(f"CPUs appear in more than one NUMA node: {sorted(node_cpus & mapped)}")
            mapped.update(node_cpus)
        result[int(node_dir.name[4:])] = node_cpus
    if mapped != allowed_cpus:
        raise ValueError(f"NUMA topology does not cover allowed CPUs: {sorted(allowed_cpus - mapped)}")
    return result


def allowed_memory_nodes(status: Path = Path("/proc/self/status")) -> set[int]:
    """Read the memory nodes permitted for this process from Linux status.

    Args:
        status (Path): Process status file containing ``Mems_allowed_list``.

    Returns:
        set[int]: Allowed memory node IDs, which do not indicate where pages
            are currently resident.

    Raises:
        ValueError: If ``Mems_allowed_list`` is absent or malformed.
        OSError: If the status file cannot be read.

    Example:
        A status line ``Mems_allowed_list: 0-1`` returns ``{0, 1}``.

    """
    for line in status.read_text().splitlines():
        if line.startswith("Mems_allowed_list:"):
            return parse_id_list(line.partition(":")[2])
    raise ValueError("Mems_allowed_list is missing from process status")

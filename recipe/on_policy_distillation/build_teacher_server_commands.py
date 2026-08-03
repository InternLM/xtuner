"""Build executable Teacher server commands from an OPD config.

The NUL-delimited output starts with the Teacher replica count, resolved
endpoint mapping, total Student worker count, and current-node Student worker
count. Each replica record contains its placement, endpoint, health URLs, and
executable command arguments. An externally managed Teacher replica has a
target node rank of ``-1`` and a command-argument count of zero.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from xtuner.v1.rl.on_policy_distillation import OPDTeacherConfig  # noqa: E402
from xtuner.v1.utils.config import Config  # noqa: E402


@dataclass(frozen=True)
class TeacherReplicaRequest:
    teacher: OPDTeacherConfig
    replica_index: int

    @property
    def key(self) -> tuple[str, int]:
        return self.teacher.name, self.replica_index

    @property
    def display_name(self) -> str:
        return f"{self.teacher.name}[{self.replica_index}]"


def build_teacher_launch_server_commands(
    config_path: str,
    backend: Literal["sglang", "lmdeploy"],
) -> tuple[dict[str, list[str]], int, int, list[list[str]]]:
    """Build executable Teacher server command records.

    Args:
        config_path (str): Path to the XTuner Python config containing
            ``opd_config``.
        backend (Literal["sglang", "lmdeploy"]): Teacher serving backend.

    Returns:
        A four-item tuple containing:

        - ``endpoint_map``: Logical Teacher name to advertised replica
          endpoints.
        - ``student_num_workers``: Total number of Student GPUs in the cluster.
        - ``student_local_num_workers``: Number of Student GPUs on the current
          node.
        - ``records``: One flattened string record per Teacher replica. Each
          record is ``[display_name, target_node_rank, local_devices, endpoint,
          health_url, model_info_url, command_arg_count, *command]``.
    """
    config = Config.fromfile(config_path)
    node_count = int(os.environ.get("NODE_COUNT", "1"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    gpus_per_node = int(os.environ.get("PROC_PER_NODE", "8"))
    node_addresses = tuple(
        address.strip()
        for address in os.environ.get(
            "WORKER_ALL_SOCKET_ADDRS", "127.0.0.1"
        ).split(",")
    )
    assert node_count > 0
    assert 0 <= node_rank < node_count
    assert gpus_per_node > 0
    assert len(node_addresses) == node_count
    assert all(node_addresses)

    teachers = config.opd_config.teachers
    replica_requests = _expand_teacher_replicas(teachers)
    health_path, model_info_path = _get_teacher_server_paths(backend)
    teacher_placements, student_local_num_workers = _allocate_teacher_devices(
        replica_requests,
        node_count=node_count,
        node_rank=node_rank,
        gpus_per_node=gpus_per_node,
    )
    teacher_num_workers = sum(
        len(local_devices)
        for _, local_devices in teacher_placements.values()
    )
    student_num_workers = node_count * gpus_per_node - teacher_num_workers
    endpoint_map: dict[str, list[str]] = {teacher.name: [] for teacher in teachers}
    records: list[list[str]] = []
    used_node_ports: set[tuple[int, int]] = set()

    for replica in replica_requests:
        teacher = replica.teacher
        launch_config = teacher.launch_config
        if launch_config is None:
            target_node_rank = -1
            local_cuda_visible_devices = ""
            endpoint = teacher.endpoints[replica.replica_index].rstrip("/")
            command: list[str] = []
        else:
            target_node_rank, local_devices = teacher_placements[replica.key]
            server_port = _allocate_teacher_server_port(
                launch_config.server_port,
                target_node_rank=target_node_rank,
                used_node_ports=used_node_ports,
                replica_name=replica.display_name,
            )

            local_cuda_visible_devices = ",".join(str(device) for device in local_devices)
            endpoint = f"http://{node_addresses[target_node_rank]}:{server_port}"
            if backend == "sglang":
                command = _build_sglang_command(teacher, server_port=server_port)
            elif backend == "lmdeploy":
                command = _build_lmdeploy_command(teacher, server_port=server_port)
            else:
                raise ValueError(f"Unsupported Teacher backend: {backend}")

        endpoint_map[teacher.name].append(endpoint)
        records.append(
            [
                replica.display_name,
                str(target_node_rank),
                local_cuda_visible_devices,
                endpoint,
                f"{endpoint}/{health_path}",
                f"{endpoint}/{model_info_path}",
                str(len(command)),
                *command,
            ]
        )
    return endpoint_map, student_num_workers, student_local_num_workers, records


def _expand_teacher_replicas(
    teachers: list[OPDTeacherConfig],
) -> list[TeacherReplicaRequest]:
    return [
        TeacherReplicaRequest(
            teacher=teacher,
            replica_index=replica_index,
        )
        for teacher in teachers
        for replica_index in range(teacher.num_replicas)
    ]


def _allocate_teacher_devices(
    replica_requests: list[TeacherReplicaRequest],
    *,
    node_count: int,
    node_rank: int,
    gpus_per_node: int,
) -> tuple[dict[tuple[str, int], tuple[int, list[int]]], int]:
    """Allocate high-rank GPUs to local Teachers and leave the rest to Student.

    Teacher replicas are processed by descending ``num_workers``. Requests
    with the same size keep their expanded config order. Each replica is placed
    wholly on the highest-rank node that has enough free GPUs, using that
    node's highest free local device ordinals. Replicas without
    ``launch_config`` are externally managed and do not consume cluster GPUs.

    Args:
        replica_requests: Teacher replicas in expanded config order.
        node_count: Number of homogeneous nodes in the cluster.
        node_rank: Rank of the current node.
        gpus_per_node: Number of local GPUs available on every node.

    Returns:
        A pair of ``(placements, student_local_num_workers)``:

        - ``placements`` maps each local ``(Teacher name, replica index)`` to
          ``(target_node_rank, local_device_ordinals)``.
        - ``student_local_num_workers`` is the number of remaining GPUs assigned
          to Student on the current node.

        For inputs equivalent to:

        .. code-block:: python

            node_count = 4
            node_rank = 3
            gpus_per_node = 8
            replica_requests = [
                ("teacher1", 0, 4),
                ("teacher2", 0, 2),
            ]

        the returned values are:

        .. code-block:: python

            placements = {
                ("teacher1", 0): (3, [4, 5, 6, 7]),
                ("teacher2", 0): (3, [2, 3]),
            }
            student_local_num_workers = 2

    Raises:
        ValueError: If a Teacher cannot fit wholly on one node, or if Teacher
            allocation leaves no GPU for Student.
    """
    free_devices_by_node = [
        list(range(gpus_per_node)) for _ in range(node_count)
    ]
    local_teacher_requests: list[tuple[TeacherReplicaRequest, int]] = []
    for replica in replica_requests:
        launch_config = replica.teacher.launch_config
        if launch_config is not None:
            local_teacher_requests.append((replica, launch_config.num_workers))
    local_teacher_requests.sort(key=lambda request: request[1], reverse=True)

    placements: dict[tuple[str, int], tuple[int, list[int]]] = {}
    for replica, num_workers in local_teacher_requests:
        if num_workers > gpus_per_node:
            raise ValueError(
                f"Teacher replica {replica.display_name!r} requests {num_workers} workers, "
                f"but each node has only {gpus_per_node} GPUs"
            )

        target_node_rank = next(
            (
                node_rank
                for node_rank in range(node_count - 1, -1, -1)
                if len(free_devices_by_node[node_rank]) >= num_workers
            ),
            None,
        )
        if target_node_rank is None:
            remaining_devices = sum(
                len(local_devices) for local_devices in free_devices_by_node
            )
            raise ValueError(
                f"Teacher replica {replica.display_name!r} requests {num_workers} workers, "
                "but no single node has enough remaining GPUs; "
                f"{remaining_devices} GPUs remain across the cluster"
            )

        free_devices = free_devices_by_node[target_node_rank]
        local_devices = free_devices[-num_workers:]
        del free_devices[-num_workers:]
        placements[replica.key] = (target_node_rank, local_devices)

    if not any(free_devices_by_node):
        raise ValueError("Teacher allocation leaves no GPUs for Student workers")
    student_local_num_workers = len(free_devices_by_node[node_rank])
    return placements, student_local_num_workers


def _allocate_teacher_server_port(
    base_port: int,
    *,
    target_node_rank: int,
    used_node_ports: set[tuple[int, int]],
    replica_name: str,
) -> int:
    server_port = base_port
    while (target_node_rank, server_port) in used_node_ports:
        server_port += 1
        if server_port > 65535:
            raise ValueError(
                f"Teacher replica {replica_name!r} cannot allocate a free port "
                f"on node {target_node_rank} starting from {base_port}"
            )
    used_node_ports.add((target_node_rank, server_port))
    return server_port


def _get_teacher_server_paths(
    backend: Literal["sglang", "lmdeploy"],
) -> tuple[str, str]:
    if backend == "sglang":
        return "health_generate", "get_model_info"
    return "health", "v1/models"


def _build_sglang_command(
    teacher: OPDTeacherConfig,
    *,
    server_port: int,
) -> list[str]:
    config = teacher.launch_config
    assert config is not None
    tensor_parallel_size = config.tensor_parallel_size
    if config.expert_parallel_size > 1:
        tensor_parallel_size = config.expert_parallel_size

    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(config.model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(server_port),
        "--dtype",
        config.dtype,
        "--tp",
        str(tensor_parallel_size),
        "--ep",
        str(config.expert_parallel_size),
        "--mem-fraction-static",
        str(config.gpu_memory_utilization),
    ]
    if config.context_length is not None:
        command.extend(["--context-length", str(config.context_length)])
    if config.max_batch_size is not None:
        command.extend(["--max-running-requests", str(config.max_batch_size)])
    if config.chunked_prefill_size is not None:
        command.extend(["--chunked-prefill-size", str(config.chunked_prefill_size)])
    return command


def _build_lmdeploy_command(
    teacher: OPDTeacherConfig,
    *,
    server_port: int,
) -> list[str]:
    config = teacher.launch_config
    assert config is not None
    data_parallel_size = (
        config.expert_parallel_size
        if config.expert_parallel_size > 1
        else 1
    )
    command = [
        sys.executable,
        "-m",
        "lmdeploy",
        "serve",
        "api_server",
        str(config.model_path),
        "--backend",
        "pytorch",
        "--role",
        "Hybrid",
        "--logprobs-mode",
        "raw_logprobs",
        "--server-name",
        "0.0.0.0",
        "--server-port",
        str(server_port),
        "--dtype",
        config.dtype,
        "--tp",
        str(config.tensor_parallel_size),
        "--ep",
        str(config.expert_parallel_size),
        "--dp",
        str(data_parallel_size),
        "--cache-max-entry-count",
        str(config.gpu_memory_utilization),
    ]
    if config.context_length is not None:
        command.extend(["--session-len", str(config.context_length)])
    if config.max_batch_size is not None:
        command.extend(["--max-batch-size", str(config.max_batch_size)])
    if config.max_prefill_token_num is not None:
        command.extend(["--max-prefill-token-num", str(config.max_prefill_token_num)])
    return command


def _write_teacher_records(
    endpoint_map: dict[str, list[str]],
    student_num_workers: int,
    student_local_num_workers: int,
    records: list[list[str]],
) -> None:
    endpoint_map_json = json.dumps(
        endpoint_map,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    fields = [
        str(len(records)),
        endpoint_map_json,
        str(student_num_workers),
        str(student_local_num_workers),
    ]
    for record in records:
        fields.extend(record)

    payload = "\0".join(fields) + "\0"
    sys.stdout.buffer.write(payload.encode("utf-8"))


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("backend", choices=("sglang", "lmdeploy"))
    args = parser.parse_args()
    endpoint_map, student_num_workers, student_local_num_workers, records = (
        build_teacher_launch_server_commands(args.config_path, args.backend)
    )
    _write_teacher_records(
        endpoint_map,
        student_num_workers,
        student_local_num_workers,
        records,
    )


if __name__ == "__main__":
    _main()

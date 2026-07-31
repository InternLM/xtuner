"""Build executable Teacher server commands from an OPD config.

The NUL-delimited output starts with the Teacher count, resolved endpoint
mapping, total Student worker count, and current-node Student worker count. Each
Teacher record contains its placement, endpoint, health URLs, and executable
command arguments. An externally managed Teacher has a target node rank of
``-1`` and a command-argument count of zero.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from xtuner.v1.rl.on_policy_distillation import OPDTeacherConfig  # noqa: E402
from xtuner.v1.utils.config import Config  # noqa: E402



def build_teacher_launch_server_commands(
    config_path: str,
    backend: Literal["sglang", "lmdeploy"],
) -> tuple[dict[str, str], int, int, list[list[str]]]:
    """Build executable Teacher server command records.

    Args:
        config_path (str): Path to the XTuner Python config containing
            ``opd_config``.
        backend (Literal["sglang", "lmdeploy"]): Teacher serving backend.

    Returns:
        A four-item tuple containing:

        - ``endpoint_map``: Teacher name to advertised endpoint.
        - ``student_num_workers``: Total number of Student GPUs in the cluster.
        - ``student_local_num_workers``: Number of Student GPUs on the current
          node.
        - ``records``: One flattened string record per Teacher. Each record is
          ``[name, target_node_rank, local_devices, endpoint, health_url,
          model_info_url, command_arg_count, *command]``.
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

    health_path, model_info_path = _get_teacher_server_paths(backend)
    teacher_placements, student_local_num_workers = _allocate_teacher_devices(
        config.opd_config.teachers,
        node_count=node_count,
        node_rank=node_rank,
        gpus_per_node=gpus_per_node,
    )
    teacher_num_workers = sum(
        len(local_devices)
        for _, local_devices in teacher_placements.values()
    )
    student_num_workers = node_count * gpus_per_node - teacher_num_workers
    endpoint_map: dict[str, str] = {}
    records: list[list[str]] = []
    used_node_ports: set[tuple[int, int]] = set()

    for teacher in config.opd_config.teachers:
        launch_config = teacher.launch_config
        if launch_config is None:
            if teacher.endpoint is None:
                raise ValueError(f"Externally managed Teacher {teacher.name!r} must define endpoint")
            target_node_rank = -1
            local_cuda_visible_devices = ""
            endpoint = teacher.endpoint.rstrip("/")
            command: list[str] = []
        else:
            target_node_rank, local_devices = teacher_placements[teacher.name]
            node_port = (target_node_rank, launch_config.server_port)
            if node_port in used_node_ports:
                raise ValueError(
                    f"Teacher {teacher.name!r} reuses port {launch_config.server_port} on node "
                    f"{target_node_rank}"
                )
            used_node_ports.add(node_port)

            local_cuda_visible_devices = ",".join(str(device) for device in local_devices)
            endpoint = (
                f"http://{node_addresses[target_node_rank]}:{launch_config.server_port}"
            )
            if backend == "sglang":
                command = _build_sglang_command(teacher)
            elif backend == "lmdeploy":
                command = _build_lmdeploy_command(teacher)

        endpoint_map[teacher.name] = endpoint
        records.append(
            [
                teacher.name,
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


def _allocate_teacher_devices(
    teachers: list[OPDTeacherConfig],
    *,
    node_count: int,
    node_rank: int,
    gpus_per_node: int,
) -> tuple[dict[str, tuple[int, list[int]]], int]:
    """Allocate high-rank GPUs to local Teachers and leave the rest to Student.

    Teachers are processed by descending ``num_workers``. Requests with the
    same size keep their original config order. Each Teacher is placed wholly
    on the highest-rank node that has enough free GPUs, using that node's
    highest free local device ordinals. Teachers without ``launch_config`` are
    externally managed and do not consume cluster GPUs.

    Args:
        teachers: Teacher configs in their original config order.
        node_count: Number of homogeneous nodes in the cluster.
        node_rank: Rank of the current node.
        gpus_per_node: Number of local GPUs available on every node.

    Returns:
        A pair of ``(placements, student_local_num_workers)``:

        - ``placements`` maps each local Teacher name to
          ``(target_node_rank, local_device_ordinals)``.
        - ``student_local_num_workers`` is the number of remaining GPUs assigned
          to Student on the current node.

        For inputs equivalent to:

        .. code-block:: python

            node_count = 4
            node_rank = 3
            gpus_per_node = 8
            teacher_num_workers = {
                "teacher1": 4,
                "teacher2": 2,
            }

        the returned values are:

        .. code-block:: python

            placements = {
                "teacher1": (3, [4, 5, 6, 7]),
                "teacher2": (3, [2, 3]),
            }
            student_local_num_workers = 2

    Raises:
        ValueError: If a Teacher cannot fit wholly on one node, or if Teacher
            allocation leaves no GPU for Student.
    """
    free_devices_by_node = [
        list(range(gpus_per_node)) for _ in range(node_count)
    ]
    local_teacher_requests: list[tuple[OPDTeacherConfig, int]] = []
    for teacher in teachers:
        launch_config = teacher.launch_config
        if launch_config is not None:
            local_teacher_requests.append((teacher, launch_config.num_workers))
    local_teacher_requests.sort(key=lambda request: request[1], reverse=True)

    placements: dict[str, tuple[int, list[int]]] = {}
    for teacher, num_workers in local_teacher_requests:
        if num_workers > gpus_per_node:
            raise ValueError(
                f"Teacher {teacher.name!r} requests {num_workers} workers, "
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
                f"Teacher {teacher.name!r} requests {num_workers} workers, "
                "but no single node has enough remaining GPUs; "
                f"{remaining_devices} GPUs remain across the cluster"
            )

        free_devices = free_devices_by_node[target_node_rank]
        local_devices = free_devices[-num_workers:]
        del free_devices[-num_workers:]
        placements[teacher.name] = (target_node_rank, local_devices)

    if not any(free_devices_by_node):
        raise ValueError("Teacher allocation leaves no GPUs for Student workers")
    student_local_num_workers = len(free_devices_by_node[node_rank])
    return placements, student_local_num_workers


def _get_teacher_server_paths(
    backend: Literal["sglang", "lmdeploy"],
) -> tuple[str, str]:
    if backend == "sglang":
        return "health_generate", "get_model_info"
    return "health", "v1/models"


def _build_sglang_command(teacher: OPDTeacherConfig) -> list[str]:
    config = teacher.launch_config
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
        str(config.server_port),
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


def _build_lmdeploy_command(teacher: OPDTeacherConfig) -> list[str]:
    config = teacher.launch_config
    data_parallel_size = config.expert_parallel_size if config.expert_parallel_size > 1 else 1
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
        str(config.server_port),
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
    endpoint_map: dict[str, str],
    student_num_workers: int,
    student_local_num_workers: int,
    records: list[list[str]],
) -> None:
    endpoint_map_json = json.dumps(endpoint_map, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
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

"""Build executable Teacher server commands from an OPD config.

The NUL-delimited output starts with the Teacher count. Each Teacher record
contains its name, safe name, endpoint, health URL, model-info URL,
command-argument count, and executable command arguments. An externally
managed Teacher has a command-argument count of zero.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from xtuner.v1.rl.on_policy_distillation import OPDTeacherConfig  # noqa: E402
from xtuner.v1.utils.config import Config  # noqa: E402


def build_teacher_server_command(
    teacher: OPDTeacherConfig,
    backend: Literal["sglang", "lmdeploy"],
) -> list[str]:
    """Build one executable Teacher server command."""
    if backend == "sglang":
        return _build_sglang_command(teacher)
    return _build_lmdeploy_command(teacher)


def build_teacher_server_commands(
    config_path: str,
    backend: Literal["sglang", "lmdeploy"],
) -> list[list[str]]:
    """Build executable Teacher server command records.

    Args:
        config_path (str): Path to the XTuner Python config containing
            ``opd_config``.
        backend (Literal["sglang", "lmdeploy"]): Teacher serving backend.

    Returns:
        list[list[str]]: Teacher metadata and executable argv records.
    """
    config = Config.fromfile(config_path)
    health_path, model_info_path = _get_teacher_server_paths(backend)
    records: list[list[str]] = []
    for teacher in config.opd_config.teachers:
        command = build_teacher_server_command(teacher, backend)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", teacher.name).strip("_") or "teacher"
        endpoint = teacher.endpoint.rstrip("/")
        records.append(
            [
                teacher.name,
                safe_name,
                endpoint,
                f"{endpoint}/{health_path}",
                f"{endpoint}/{model_info_path}",
                str(len(command)),
                *command,
            ]
        )
    return records


def _get_teacher_server_paths(
    backend: Literal["sglang", "lmdeploy"],
) -> tuple[str, str]:
    if backend == "sglang":
        return "health_generate", "get_model_info"
    return "health", "v1/models"


def _parse_teacher_endpoint(teacher: OPDTeacherConfig) -> tuple[str, int]:
    parsed_endpoint = urlparse(teacher.endpoint)
    if parsed_endpoint.scheme not in {"http", "https"}:
        raise ValueError(f"Teacher {teacher.name!r} endpoint must use http or https: {teacher.endpoint}")
    if parsed_endpoint.hostname is None or parsed_endpoint.port is None:
        raise ValueError(
            f"Teacher {teacher.name!r} endpoint must contain an explicit host and port: {teacher.endpoint}"
        )
    if parsed_endpoint.path not in {"", "/"} or parsed_endpoint.params or parsed_endpoint.query:
        raise ValueError(
            f"Teacher {teacher.name!r} endpoint must be a server base URL without a path or query: {teacher.endpoint}"
        )
    return parsed_endpoint.hostname, parsed_endpoint.port


def _build_sglang_command(teacher: OPDTeacherConfig) -> list[str]:
    host, port = _parse_teacher_endpoint(teacher)
    config = teacher.launch_config
    if config is None:
        return []

    tensor_parallel_size = config.tensor_parallel_size
    if config.expert_parallel_size > 1:
        tensor_parallel_size = config.expert_parallel_size

    command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={config.cuda_visible_devices}",
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(config.model_path),
        "--host",
        host,
        "--port",
        str(port),
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
    host, port = _parse_teacher_endpoint(teacher)
    config = teacher.launch_config
    if config is None:
        return []

    data_parallel_size = config.expert_parallel_size if config.expert_parallel_size > 1 else 1
    command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={config.cuda_visible_devices}",
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
        host,
        "--server-port",
        str(port),
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


def _write_teacher_records(records: list[list[str]]) -> None:
    fields = [str(len(records))]
    for record in records:
        fields.extend(record)

    payload = "\0".join(fields) + "\0"
    sys.stdout.buffer.write(payload.encode("utf-8"))


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("backend", choices=("sglang", "lmdeploy"))
    args = parser.parse_args()
    _write_teacher_records(build_teacher_server_commands(args.config_path, args.backend))


if __name__ == "__main__":
    _main()

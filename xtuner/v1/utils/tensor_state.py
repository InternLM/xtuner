from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist


def check_tensor_storage_dev(obj: Any, prefix: str = ""):
    cpu_count = 0
    gpu_count = 0
    other_device_counts: dict[str, int] = defaultdict(int)
    sample_cpu_path = ""
    sample_gpu_path = ""
    sample_other_paths: dict[str, str] = {}
    visited_ids: set[int] = set()

    if dist.is_available() and dist.is_initialized():
        rank_tag = f"rank{dist.get_rank()}"
    else:
        rank_tag = "rank?"

    def _walk(value: Any, path: str) -> None:
        nonlocal cpu_count, gpu_count, sample_cpu_path, sample_gpu_path
        value_id = id(value)
        if value_id in visited_ids:
            return

        if isinstance(value, torch.Tensor):
            device_type = value.device.type
            if device_type == "cpu":
                cpu_count += 1
                if not sample_cpu_path:
                    sample_cpu_path = path
            elif device_type == "cuda":
                gpu_count += 1
                if not sample_gpu_path:
                    sample_gpu_path = path
            else:
                device_name = str(value.device)
                other_device_counts[device_name] += 1
                sample_other_paths.setdefault(device_name, path)
            return

        if isinstance(value, Mapping):
            visited_ids.add(value_id)
            for key, nested_value in value.items():
                _walk(nested_value, f"{path}[{key!r}]")
            return

        if isinstance(value, tuple) and hasattr(value, "_fields"):
            visited_ids.add(value_id)
            for field_name, nested_value in zip(value._fields, value):
                _walk(nested_value, f"{path}.{field_name}")
            return

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            visited_ids.add(value_id)
            for index, nested_value in enumerate(value):
                _walk(nested_value, f"{path}[{index}]")

    root_name = prefix or "state"
    _walk(obj, root_name)

    other_count = sum(other_device_counts.values())
    total_tensor_count = cpu_count + gpu_count + other_count

    print(f"------------{rank_tag}-----{root_name} total_tensors:{total_tensor_count}, cpu_count:{cpu_count}, gpu_count:{gpu_count}\n")

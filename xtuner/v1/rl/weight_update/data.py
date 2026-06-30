from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, TypeAlias

import torch
from torch.distributed.device_mesh import DeviceMesh


DeviceMeshRaw: TypeAlias = List[List[int]]  # A list of lists representing device mesh indices.
ServiceUrlMap: TypeAlias = Dict[int, str]  # A dictionary mapping rollout ranks to their server URLs.
RolloutEngineInfo: TypeAlias = list[tuple[int, str, int]]  # (rollout rank, server url, engine gpu count)
TrainRolloutMode: TypeAlias = Literal["colocate", "disaggregated"]  # Train and rollout deployment mode.
RolloutBackend: TypeAlias = Literal["sglang", "vllm", "pytorch", "turbomind"]  # Rollout inference backend.
WeightTransportType: TypeAlias = Literal["ipc", "nccl"]  # Supported weight transport types.


@dataclass
class RolloutWeightUpdateInfo:
    # Common rollout metadata.
    api_key: list[str] | str | None = None
    rollout_url: str | None = None
    backend: RolloutBackend | None = None
    tp: int = 1
    ep: int = 1
    train_rollout_mode: TrainRolloutMode | None = None
    transport_type: WeightTransportType | None = None
    rollout_cfg_info: dict = field(default_factory=dict)
    endpoints: dict[str, str] = field(default_factory=lambda: {"update_weights": "update_weights"})

    # Colocated rollout metadata.
    rollout_device_mesh: DeviceMesh | None = None
    rollout_engine_rank_mesh_array: DeviceMeshRaw = field(default_factory=list)

    # Disaggregated rollout metadata.
    rollout_server_url_dict: ServiceUrlMap = field(default_factory=dict)
    worker_server_urls_status: dict[str, bool] = field(default_factory=dict)
    weight_update_host: str | None = None
    weight_update_port: int | None = None


@dataclass
class WeightUpdateBatch:
    """A single bucket of weights to send to rollout workers."""

    state_dict: dict[str, torch.Tensor]
    train_enable_ep: bool = False
    finished: bool = False

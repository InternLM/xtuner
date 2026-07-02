from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import torch


if TYPE_CHECKING:
    from xtuner.v1.rl.rollout.worker import RolloutConfig


RolloutBackend: TypeAlias = Literal["sglang", "vllm", "pytorch", "turbomind"]  # Rollout inference backend.
WeightTransportType: TypeAlias = Literal["ipc", "nccl"]  # Supported weight transport types.


def _resolve_rollout_backend(rollout_config: RolloutConfig) -> RolloutBackend:
    # Backend selection follows rollout launcher precedence.
    if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
        backend = "sglang"
    elif os.environ.get("XTUNER_USE_VLLM", "0") == "1":
        backend = "vllm"
    else:
        backend = (rollout_config.extra_rollout_config or dict()).get("lmdeploy_backend", "pytorch")

    backend = backend.lower()
    if backend not in ("sglang", "vllm", "pytorch", "turbomind"):
        raise ValueError(
            f"Unsupported rollout backend: {backend!r}. Expected 'sglang', 'vllm', 'pytorch' or 'turbomind'."
        )
    return cast(RolloutBackend, backend)


def _validate_transport_type(
    *,
    weight_transport_type: WeightTransportType | str,
    backend: RolloutBackend,
) -> WeightTransportType:
    assert weight_transport_type is not None, "bind_rollout_weight_update() must set weight_transport_type."

    transport_type = weight_transport_type.lower()
    if transport_type not in ("ipc", "nccl"):
        raise ValueError(f"Unsupported weight_transport_type: {weight_transport_type!r}. Expected 'ipc' or 'nccl'.")
    transport_type = cast(WeightTransportType, transport_type)
    if transport_type == "nccl" and backend in ("vllm", "turbomind"):
        raise NotImplementedError(f"NCCL weight transport is not supported for {backend} backend.")
    return transport_type


@dataclass(frozen=True)
class RolloutWeightUpdateTarget:
    """Runtime weight-update endpoint resolved from rollout registry state."""

    # Server-process worker rank that receives weight update requests.
    endpoint_rank: int
    # Rollout ranks updated through this endpoint.
    update_ranks: tuple[int, ...]
    # Runtime rollout server URL resolved from WorkerSnapshot.
    server_url: str
    # Registry lifecycle state value for this endpoint.
    lifecycle_state: str

    @property
    def is_active(self) -> bool:
        return self.lifecycle_state == "active"

    @property
    def engine_size(self) -> int:
        return len(self.update_ranks)


@dataclass(frozen=True)
class RolloutWeightUpdateInfo:
    # Rollout config owns api_key, backend choice, TP/EP, and default update host/port.
    rollout_config: RolloutConfig
    # Registry-resolved rollout update targets visible to every train worker.
    weight_update_targets: tuple[RolloutWeightUpdateTarget, ...]
    # Current train worker rank; used to derive the local weight update target.
    train_rank: int
    # Weight transport protocol; also determines rollout weight export strategy.
    transport_type: WeightTransportType
    # Resolved rollout backend used by transports and iterators.
    backend: RolloutBackend
    # Optional host used by NCCL external weight update groups.
    weight_update_host: str | None = None
    # Optional port used by NCCL external weight update groups.
    weight_update_port: int | None = None

    @classmethod
    def from_targets(
        cls,
        *,
        rollout_config: RolloutConfig,
        weight_update_targets: tuple[RolloutWeightUpdateTarget, ...],
        train_rank: int,
        weight_transport_type: WeightTransportType | str,
        weight_update_host: str | None = None,
        weight_update_port: int | None = None,
    ) -> RolloutWeightUpdateInfo:
        backend = _resolve_rollout_backend(rollout_config)
        tp = rollout_config.tensor_parallel_size
        ep = rollout_config.expert_parallel_size
        assert tp == 1 or ep == 1, "Either tensor parallel size or engine parallel size must be 1."
        transport_type = _validate_transport_type(
            weight_transport_type=weight_transport_type,
            backend=backend,
        )
        return cls(
            rollout_config=rollout_config,
            weight_update_targets=weight_update_targets,
            train_rank=train_rank,
            transport_type=transport_type,
            backend=backend,
            weight_update_host=weight_update_host,
            weight_update_port=weight_update_port if weight_update_port is not None else 30000,
        )

    @property
    def local_update_target(self) -> RolloutWeightUpdateTarget | None:
        return next(
            (target for target in self.weight_update_targets if self.train_rank == target.endpoint_rank),
            None,
        )

    @property
    def rollout_url(self) -> str | None:
        target = self.local_update_target
        if target is None or not target.is_active:
            return None
        return target.server_url

    @property
    def ipc_rank_mesh(self) -> tuple[tuple[int, ...], ...]:
        return tuple(target.update_ranks for target in self.weight_update_targets)

    @property
    def _ipc_update_target(self) -> RolloutWeightUpdateTarget | None:
        return next(
            (target for target in self.weight_update_targets if self.train_rank in target.update_ranks),
            None,
        )

    @property
    def ipc_engine_parallel_rank(self) -> int | None:
        target = self._ipc_update_target
        if target is None:
            return None
        return target.update_ranks.index(self.train_rank)

    @property
    def ipc_engine_parallel_size(self) -> int | None:
        target = self._ipc_update_target
        if target is None:
            return None
        return target.engine_size

    @property
    def active_update_targets(self) -> tuple[RolloutWeightUpdateTarget, ...]:
        return tuple(target for target in self.weight_update_targets if target.is_active)

    @property
    def nccl_engine_infos(self) -> tuple[tuple[int, str, int], ...]:
        return tuple(
            (target.endpoint_rank, target.server_url, target.engine_size) for target in self.active_update_targets
        )

    @property
    def transport_signature(self) -> tuple[Any, ...]:
        target_signature = tuple(
            (
                target.endpoint_rank,
                tuple(int(rank) for rank in target.update_ranks),
                target.server_url,
                target.lifecycle_state,
            )
            for target in self.weight_update_targets
        )
        frozen_api_key = tuple(self.api_key) if isinstance(self.api_key, list) else self.api_key
        return (
            self.transport_type,
            self.backend,
            self.tp,
            self.ep,
            frozen_api_key,
            self.weight_update_host,
            self.weight_update_port,
            target_signature,
        )

    @property
    def api_key(self) -> list[str] | str | None:
        return self.rollout_config.api_key

    @property
    def tp(self) -> int:
        return self.rollout_config.tensor_parallel_size

    @property
    def ep(self) -> int:
        return self.rollout_config.expert_parallel_size


@dataclass
class WeightUpdateBatch:
    """A single bucket of weights to send to rollout workers."""

    # HF-style named tensors or backend-specific tensors for one update bucket.
    state_dict: dict[str, torch.Tensor]
    # Whether the train model uses EP and may need rollout EP slicing.
    train_enable_ep: bool = False
    # Whether this is the final bucket in the current update stream.
    finished: bool = False

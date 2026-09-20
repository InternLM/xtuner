from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import torch


if TYPE_CHECKING:
    from xtuner.v1.rl.rollout.worker import RolloutConfig


RolloutBackend: TypeAlias = Literal["sglang", "vllm", "pytorch", "turbomind"]  # Rollout inference backend.
WeightTransportType: TypeAlias = Literal["ipc", "nccl", "checkpoint_engine"]  # Supported weight transport types.


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
    if transport_type not in ("ipc", "nccl", "checkpoint_engine"):
        raise ValueError(
            f"Unsupported weight_transport_type: {weight_transport_type!r}. "
            "Expected 'ipc', 'nccl' or 'checkpoint_engine'."
        )
    transport_type = cast(WeightTransportType, transport_type)
    if transport_type == "nccl" and backend in ("vllm", "turbomind"):
        raise NotImplementedError(f"NCCL weight transport is not supported for {backend} backend.")
    if transport_type == "checkpoint_engine" and backend != "sglang":
        raise NotImplementedError(
            f"Checkpoint Engine weight transport currently only supports sglang, got backend={backend!r}."
        )
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
    # All rollout ranks belonging to the logical inference engine.
    inference_engine_ranks: tuple[int, ...]

    @property
    def engine_size(self) -> int:
        return len(self.inference_engine_ranks)

    @property
    def update_size(self) -> int:
        return len(self.update_ranks)

    @property
    def inference_engine_rank(self) -> int:
        return self.inference_engine_ranks.index(self.endpoint_rank)


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
    # Optional prefix used by checkpoint-engine
    checkpoint_name_prefix: str | None = None
    # Optional timeout used by checkpoint-engine
    checkpoint_engine_timeout: float | None = None
    # Whether to explicitly synchronize after registering checkpoint-engine tensors.
    checkpoint_engine_sync_after_register: bool = True
    _ipc_update_target: RolloutWeightUpdateTarget | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Only for IPC transport type. But need to set _ipc_update_target to None for other transport type.
        # Checkpoint Engine 做P2P恢复时，正常运行的worker if分支永远不满足，需要给ipc_target设置默认值
        ipc_target = next(
            (target for target in self.weight_update_targets if self.train_rank in target.update_ranks),
            None,
        )
        object.__setattr__(self, "_ipc_update_target", ipc_target)

    @classmethod
    def from_targets(
        cls,
        *,
        rollout_config: RolloutConfig,
        weight_update_targets: tuple[RolloutWeightUpdateTarget, ...],
        train_rank: int,
    ) -> RolloutWeightUpdateInfo:
        backend = _resolve_rollout_backend(rollout_config)
        tp = rollout_config.tensor_parallel_size
        ep = rollout_config.expert_parallel_size
        assert tp == 1 or ep == 1, "Either tensor parallel size or engine parallel size must be 1."
        transport_type = rollout_config.weight_transport_type
        if transport_type is None:
            raise ValueError("rollout_config.weight_transport_type should be set in RL training")
        transport_type = _validate_transport_type(
            weight_transport_type=transport_type,
            backend=backend,
        )
        return cls(
            rollout_config=rollout_config,
            weight_update_targets=weight_update_targets,
            train_rank=train_rank,
            transport_type=transport_type,
            backend=backend,
            weight_update_host=rollout_config.weight_update_host,
            weight_update_port=rollout_config.weight_update_port
            if rollout_config.weight_update_port is not None
            else 30000,
            checkpoint_name_prefix=rollout_config.checkpoint_name_prefix,
            checkpoint_engine_timeout=rollout_config.checkpoint_engine_timeout,
            checkpoint_engine_sync_after_register=rollout_config.checkpoint_engine_sync_after_register,
        )

    @property
    def ipc_rank_mesh(self) -> tuple[tuple[int, ...], ...]:
        return tuple(target.update_ranks for target in self.weight_update_targets)

    @property
    def inference_engine_parallel_rank(self) -> int | None:
        target = self._ipc_update_target
        if target is None:
            return None
        return target.inference_engine_ranks.index(self.train_rank)

    @property
    def inference_engine_parallel_size(self) -> int | None:
        target = self._ipc_update_target
        return None if target is None else target.engine_size

    @property
    def update_targets(self) -> tuple[RolloutWeightUpdateTarget, ...]:
        return tuple(target for target in self.weight_update_targets)

    @property
    def update_target_infos(self) -> list[dict[str, Any]]:
        return [
            {
                "endpoint_rank": target.endpoint_rank,
                "server_url": target.server_url,
                "lifecycle_state": target.lifecycle_state,
                "update_ranks": target.update_ranks,
                "update_size": target.update_size,
                "inference_engine_size": target.engine_size,
            }
            for target in self.update_targets
        ]

    @property
    def nccl_engine_infos(self) -> tuple[tuple[int, str, int], ...]:
        return tuple((target.endpoint_rank, target.server_url, target.update_size) for target in self.update_targets)

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

from __future__ import annotations

import os
from typing import Any, cast

from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.rl.rollout.worker import RolloutConfig

from .data import (
    DeviceMeshRaw,
    RolloutBackend,
    RolloutWeightUpdateInfo,
    ServiceUrlMap,
    TrainRolloutMode,
)
from .transport import IPCWeightTransport, NCCLWeightTransport, WeightTransport
from .weight_iterator import WeightIterator


class UpdateWeighter:
    def __init__(self, *, rank: int, logger: Any, config: Any, engine: Any):
        self.rank = rank
        self.logger = logger
        self.config = config
        self._engine = engine
        # Used to update weight to rollout engine.
        self.rollout_info = RolloutWeightUpdateInfo()
        self._global_hf_keys_mapping_cache: dict[str, list[str]] = {}
        # Transport is initialized after update_rollout_info() is called.
        self._transport: WeightTransport | None = None
        # Used to detect changes in rollout metadata that require resetting the transport.
        self._transport_signature: tuple[Any, ...] | None = None

    @staticmethod
    def _normalize_rollout_backend(rollout_config: RolloutConfig) -> RolloutBackend:
        # Backend selection follows rollout launcher precedence: explicit SGLang/vLLM env vars win,
        # otherwise the LMDeploy backend decides between pytorch and turbomind.
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

    def update_rollout_info(
        self,
        engine_rank_mesh_array: DeviceMeshRaw,
        server_url_dict: ServiceUrlMap,
        rollout_config: RolloutConfig,
        worker_server_urls_status: dict[str, bool],
        train_rollout_mode: TrainRolloutMode,
        weight_update_host: str | None = None,
        weight_update_port: int | None = None,
        worker_session_url_dict: ServiceUrlMap | None = None,
    ):
        """Update the rollout information for the training worker."""

        self.rollout_info.backend = self._normalize_rollout_backend(rollout_config)
        self.set_train_rollout_mode(train_rollout_mode=train_rollout_mode)

        # Common rollout metadata.
        tp = rollout_config.tensor_parallel_size
        ep = rollout_config.expert_parallel_size
        assert tp == 1 or ep == 1, "Either tensor parallel size or engine parallel size must be 1."
        self.rollout_info.tp = tp
        self.rollout_info.ep = ep
        self.rollout_info.api_key = rollout_config.api_key
        rollout_server_url = server_url_dict.get(self.rank, "")
        if not worker_server_urls_status.get(rollout_server_url, False):
            self.logger.error(f"Rollout server url {rollout_server_url} is not available.")
            self.rollout_info.rollout_url = None
        else:
            self.rollout_info.rollout_url = rollout_server_url

        if self.rollout_info.transport_type == "ipc":
            # Colocated rollout metadata.
            # rollout_device_mesh is created after train_rollout_mode is set.
            self.rollout_info.rollout_engine_rank_mesh_array = [
                [int(rank) for rank in ranks] for ranks in engine_rank_mesh_array
            ]

        elif self.rollout_info.transport_type == "nccl":
            # Disaggregated rollout metadata.
            self.rollout_info.rollout_server_url_dict = {int(rank): url for rank, url in server_url_dict.items()}
            self.rollout_info.worker_server_urls_status = worker_server_urls_status
            self.rollout_info.weight_update_host = weight_update_host
            self.rollout_info.weight_update_port = weight_update_port if weight_update_port is not None else 30000

        new_transport_signature = self._build_transport_signature(
            engine_rank_mesh_array=engine_rank_mesh_array,
            server_url_dict=server_url_dict,
            worker_server_urls_status=worker_server_urls_status,
            train_rollout_mode=train_rollout_mode,
            backend=self.rollout_info.backend,
            tp=tp,
            ep=ep,
        )
        # Weight transports may cache resources derived from rollout metadata.
        # Since rollout workers can fail and recover with new URL/status/mesh metadata,
        # reset the cached transport whenever that metadata changes.
        if self._transport_signature is not None and new_transport_signature != self._transport_signature:
            self.logger.info("Rollout metadata changed, reset weight transport.")
            self._reset_transport()
        self._transport_signature = new_transport_signature

        self.weight_iterator = WeightIterator(
            config=self.config,
            engine=self._engine,
            rollout_info=self.rollout_info,
            global_hf_keys_mapping_cache=self._global_hf_keys_mapping_cache,
        )
        if self._transport is None:
            self._set_transport()

    def _ensure_rollout_device_mesh(self):
        if self.rollout_info.rollout_device_mesh is None:
            # 非共卡 SGLang 不使用这个 mesh；只有共卡/旧权重同步路径需要
            # 用 rollout rank 构造 torch DeviceMesh。
            self.rollout_info.rollout_device_mesh = DeviceMesh(
                "cpu",
                mesh=self.rollout_info.rollout_engine_rank_mesh_array,
                mesh_dim_names=("engine_instance", "engine_parallel"),
            )

    def set_train_rollout_mode(self, train_rollout_mode: TrainRolloutMode | str):
        assert train_rollout_mode is not None, "update_rollout_info() must set train_rollout_mode."

        if self.rollout_info.backend is None:
            raise RuntimeError("rollout backend is not set. Please set rollout backend in update_rollout_info().")

        mode = train_rollout_mode.lower()
        if mode not in ("colocate", "disaggregated"):
            raise ValueError(
                f"Unsupported train_rollout_mode: {train_rollout_mode!r}. Expected 'colocate' or 'disaggregated'."
            )
        mode = cast(TrainRolloutMode, mode)
        self.rollout_info.train_rollout_mode = mode
        if mode == "colocate":
            self.rollout_info.transport_type = "ipc"
            self._ensure_rollout_device_mesh()
        elif mode == "disaggregated":
            self.rollout_info.transport_type = "nccl"

            backend = self.rollout_info.backend
            if backend == "vllm" or backend == "turbomind":
                raise NotImplementedError(f"Disaggregated train-rollout mode is not supported for {backend} backend.")

    def update_weights(self):
        """Update the model weights."""

        assert self._transport is not None, (
            f"Weight transport is not initialized. transport_type={self.rollout_info.transport_type!r}, "
            f"backend={self.rollout_info.backend!r}."
        )
        self._transport.update(self.weight_iterator)

    def _set_transport(self) -> None:
        if self.rollout_info.transport_type == "ipc":
            self._transport = IPCWeightTransport(
                rank=self.rank,
                logger=self.logger,
                config=self.config,
                rollout_info=self.rollout_info,
            )
        elif self.rollout_info.transport_type == "nccl":
            self._transport = NCCLWeightTransport(rank=self.rank, logger=self.logger, rollout_info=self.rollout_info)
        else:
            raise NotImplementedError

    def _build_transport_signature(
        self,
        *,
        engine_rank_mesh_array: DeviceMeshRaw,
        server_url_dict: ServiceUrlMap,
        worker_server_urls_status: dict[str, bool],
        train_rollout_mode: TrainRolloutMode,
        backend: RolloutBackend,
        tp: int,
        ep: int,
    ) -> tuple[Any, ...]:
        mesh = tuple(tuple(int(rank) for rank in ranks) for ranks in engine_rank_mesh_array)

        active_urls = tuple(
            sorted(
                (int(rank), url)
                for rank, url in server_url_dict.items()
                if url and worker_server_urls_status.get(url, False)
            )
        )

        return (
            train_rollout_mode,
            backend,
            tp,
            ep,
            mesh,
            active_urls,
        )

    def _reset_transport(self) -> None:
        if self._transport is not None:
            self._transport.teardown()
            self._transport = None

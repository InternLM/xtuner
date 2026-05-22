from __future__ import annotations

import os
from typing import Any, cast

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.utils import get_torch_device_module

from .data import DeviceMeshRaw, RolloutBackend, RolloutWeightUpdateInfo, ServiceUrlMap, TrainRolloutMode
from .exporter import WeightExporter
from .transport import IPCWeightTransport, NCCLWeightTransport, WeightTransport


DEVICE_MODULE = get_torch_device_module()


class UpdateWeighter:
    def __init__(self, *, rank: int, logger: Any, config: Any, engine: Any):
        self.rank = rank
        self.logger = logger
        self.config = config
        self._engine = engine
        # Used to update weight to rollout engine.
        self.rollout_info = RolloutWeightUpdateInfo()
        self._global_hf_keys_mapping_cache: dict[str, list[str]] = {}
        self.is_train_rollout_colocated: bool | None = None
        # Only used by currently unsupported LMDeploy disaggregated modes.
        self.use_fake_weight_update = False
        self._transport: WeightTransport | None = None

    @staticmethod
    def _normalize_rollout_backend(backend: str) -> RolloutBackend:
        backend = backend.lower()
        if backend not in ("sglang", "vllm", "pytorch", "turbomind"):
            raise ValueError(
                f"Unsupported rollout backend: {backend!r}. Expected 'sglang', 'vllm', 'pytorch' or 'turbomind'."
            )
        return cast(RolloutBackend, backend)

    @staticmethod
    def _normalize_train_rollout_mode(train_rollout_mode: str) -> TrainRolloutMode:
        mode = train_rollout_mode.lower()
        if mode not in ("colocate", "disaggregated"):
            raise ValueError(
                f"Unsupported train_rollout_mode: {train_rollout_mode!r}. Expected 'colocate' or 'disaggregated'."
            )
        return cast(TrainRolloutMode, mode)

    def _hook_compare_test_sent_and_received_weight_hash(
        self,
        result: dict[str, Any],
        *,
        bucket_idx: int | None = None,
        names: list[str] | None = None,
    ) -> None:
        """Test hook for comparing sent and received weight hashes.

        This hook is intentionally a no-op in production code and is expected to be overridden in unit tests that need
        to compare training-side sent hashes with rollout-side received hashes returned by SGLang.
        """
        return

    def update_rollout_info(
        self,
        engine_rank_mesh_array: DeviceMeshRaw,
        server_url_dict: ServiceUrlMap,
        rollout_config: RolloutConfig,
        worker_server_urls_status: dict[str, bool],
        api_server_url: str | None = None,
    ):
        """Update the rollout information for the training worker."""
        tp = rollout_config.tensor_parallel_size
        ep = rollout_config.expert_parallel_size
        assert tp == 1 or ep == 1, "Either tensor parallel size or engine parallel size must be 1."
        if self.rollout_info.rollout_device_mesh is None:
            self.rollout_info.rollout_device_mesh = DeviceMesh(
                "cpu",
                mesh=engine_rank_mesh_array,
                mesh_dim_names=("engine_instance", "engine_parallel"),
            )
        rollout_server_url = server_url_dict.get(self.rank, "")
        if worker_server_urls_status.get(rollout_server_url, "False") is False:
            self.logger.error(f"Rollout server url {rollout_server_url} is not available.")
            self.rollout_info.rollout_url = None
        else:
            self.rollout_info.rollout_url = rollout_server_url

        self.rollout_info.rollout_engine_rank_mesh_array = [
            [int(rank) for rank in ranks] for ranks in engine_rank_mesh_array
        ]
        self.rollout_info.rollout_server_url_dict = {int(rank): url for rank, url in server_url_dict.items()}
        self.rollout_info.worker_server_urls_status = worker_server_urls_status

        # Backend selection follows rollout launcher precedence: explicit SGLang/vLLM env vars win,
        # otherwise the LMDeploy backend decides between pytorch and turbomind.
        if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
            backend = "sglang"
        elif os.environ.get("XTUNER_USE_VLLM", "0") == "1":
            backend = "vllm"
        else:
            backend = (rollout_config.extra_rollout_config or dict()).get("lmdeploy_backend", "pytorch")

        self.rollout_info.tp = tp
        self.rollout_info.ep = ep
        self.rollout_info.api_key = rollout_config.api_key
        self.rollout_info.backend = self._normalize_rollout_backend(backend)

        # Keep the legacy dict synchronized while downstream code migrates to typed fields.
        self.rollout_info.rollout_cfg_info["tp"] = self.rollout_info.tp
        self.rollout_info.rollout_cfg_info["ep"] = self.rollout_info.ep
        self.rollout_info.rollout_cfg_info["api_key"] = self.rollout_info.api_key
        self.rollout_info.rollout_cfg_info["backend"] = self.rollout_info.backend

    def set_train_rollout_mode(self, train_rollout_mode: TrainRolloutMode | str):
        mode = self._normalize_train_rollout_mode(train_rollout_mode)
        self.rollout_info.train_rollout_mode = mode
        if mode == "colocate":
            self.is_train_rollout_colocated = True
            self.use_fake_weight_update = False
            self.rollout_info.transport_type = "ipc"
        elif mode == "disaggregated":
            self.is_train_rollout_colocated = False
            self.rollout_info.transport_type = "nccl"

            backend = self.rollout_info.backend
            if backend == "vllm":
                raise NotImplementedError("Disaggregated train-rollout mode is not supported for vLLM backend.")
            if backend == "pytorch" or backend == "turbomind":
                self.logger.warning(
                    "Disaggregated train-rollout mode for lmdeploy backend is not fully supported yet. "
                    "A fake no-op interface will be used temporarily.",
                )
                # Fake update lets the training loop skip real synchronization for unsupported modes.
                self.use_fake_weight_update = True
            elif backend == "sglang":
                self.use_fake_weight_update = False
            else:
                raise ValueError(
                    f"Unsupported rollout backend for disaggregated mode: {backend!r}. "
                    "Expected 'vllm', 'pytorch', 'turbomind' or 'sglang'."
                )

        # IPC transports are per-update and cheap to recreate, while NCCL transports keep an
        # external process group alive for disaggregated updates.
        if self.is_train_rollout_colocated:
            self._reset_transport()

    def update_weights(self):
        """Update the model weights."""
        if self.is_train_rollout_colocated is None:
            raise RuntimeError(
                "train/rollout mode is not set. Please call set_train_rollout_mode() before update_weights()."
            )

        if self.use_fake_weight_update:
            train_rollout_mode = self.rollout_info.train_rollout_mode or (
                "colocate" if self.is_train_rollout_colocated else "disaggregated"
            )
            backend = self.rollout_info.backend or "unknown"
            self.logger.warning(
                "Using fake weight update interface, no actual weight synchronization will happen. "
                "This is only for testing purposes and should not be used in production. "
                f"train_rollout_mode={train_rollout_mode}, backend={backend}."
            )
            return

        transport = self._get_transport()
        exporter = WeightExporter(
            config=self.config,
            engine=self._engine,
            rollout_info=self.rollout_info,
            global_hf_keys_mapping_cache=self._global_hf_keys_mapping_cache,
        )

        transport.before_update()
        DEVICE_MODULE.empty_cache()
        try:
            for batches, sync_group in self._iter_export_batch_groups(exporter):
                self._send_exported_batches(transport, batches, sync_group=sync_group)
        finally:
            transport.after_update()
            DEVICE_MODULE.empty_cache()

    def _iter_export_batch_groups(self, exporter: WeightExporter):
        # Export path depends on rollout protocol: turbomind consumes layer-wise batches,
        # compose models update submodules in order, and plain models use HF-style batches.
        if self.is_train_rollout_colocated and self.rollout_info.backend == "turbomind":
            yield exporter.iter_layer_batches(), "colocated"
            return

        if isinstance(self.config.model_cfg, BaseComposeConfig):
            # Only the last compose submodule sends the final update marker.
            submodules = (
                ("language_model", False),
                ("vision_tower", False),
                ("multi_modal_projector", True),
            )
            for submodule, final_update in submodules:
                yield exporter.iter_hf_batches(submodule=submodule, final_update=final_update), "current"
            return

        yield exporter.iter_hf_batches(final_update=True), "current"

    def _send_exported_batches(self, transport: WeightTransport, batches, *, sync_group: str) -> None:
        for batch in batches:
            transport.send(batch)
        self._barrier_after_export(transport, sync_group=sync_group)
        DEVICE_MODULE.empty_cache()

    def _barrier_after_export(self, transport: WeightTransport, *, sync_group: str) -> None:
        # Colocated IPC synchronizes all training ranks, while disaggregated NCCL uses a
        # dedicated CPU sync group to avoid coupling with the external NCCL group.
        if self.is_train_rollout_colocated or sync_group == "colocated":
            dist.barrier()
            return
        if isinstance(transport, NCCLWeightTransport):
            dist.barrier(group=transport.get_train_update_sync_group())
            return
        dist.barrier()

    def _get_transport(self) -> WeightTransport:
        if self.rollout_info.transport_type == "ipc":
            return IPCWeightTransport(
                rank=self.rank,
                logger=self.logger,
                config=self.config,
                rollout_info=self.rollout_info,
            )

        if self.rollout_info.transport_type == "nccl" and self._transport is None:
            transport = NCCLWeightTransport(rank=self.rank, logger=self.logger, rollout_info=self.rollout_info)
            transport.hook_compare_test_sent_and_received_weight_hash = (
                self._hook_compare_test_sent_and_received_weight_hash
            )
            self._transport = transport
        if self._transport is None:
            raise RuntimeError(
                f"Weight transport is not initialized. transport_type={self.rollout_info.transport_type!r}."
            )
        return self._transport

    def _reset_transport(self):
        if self._transport is not None:
            self._transport.teardown()
        self._transport = None

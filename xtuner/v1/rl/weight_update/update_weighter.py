from __future__ import annotations

from typing import Any

from xtuner.v1.rl.rollout.worker import RolloutConfig

from .data import (
    RolloutWeightUpdateInfo,
    RolloutWeightUpdateTarget,
)
from .transport import CheckpointEngineWeightTransport, IPCWeightTransport, NCCLWeightTransport
from .weight_iterator import WeightIterator


class WeightUpdater:
    def __init__(self, *, rank: int, logger: Any, config: Any, engine: Any):
        self.rank = rank
        self.logger = logger
        self.config = config
        self._engine = engine
        # Bound rollout weight-update metadata, available after bind_rollout_weight_update().
        self.rollout_info: RolloutWeightUpdateInfo | None = None
        # Lazily constructed iterator bound to the current rollout_info.
        self.weight_iterator: WeightIterator | None = None
        self._global_hf_keys_mapping_cache: dict[str, list[str]] = {}
        # Transport is initialized after bind_rollout_weight_update() is called.
        self._transport: Any | None = None
        # Used to detect changes in rollout metadata that require resetting the transport.
        self._transport_signature: tuple[Any, ...] | None = None

    def bind_rollout_weight_update(
        self,
        *,
        targets: tuple[RolloutWeightUpdateTarget, ...],
        rollout_config: RolloutConfig,
    ):
        """Bind this train worker to rollout weight-update targets."""

        self.rollout_info = RolloutWeightUpdateInfo.from_targets(
            rollout_config=rollout_config,
            weight_update_targets=targets,
            train_rank=self.rank,
        )

        if self.rollout_info.transport_type == "checkpoint_engine" and self._transport is not None:
            # When using Checkpoint Engine, the ParameterServer must not be reinitialized
            # when rollout info changes. And only update rollout_info.
            self._transport.reset_rollout_info(self.rollout_info)
        else:
            new_transport_signature = self.rollout_info.transport_signature
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

    def weight_update(self, **kwargs: Any) -> None:
        """Update the model weights."""

        assert self.rollout_info is not None, "bind_rollout_weight_update() must be called before weight_update()."
        assert self._transport is not None, (
            f"Weight transport is not initialized. transport_type={self.rollout_info.transport_type!r}, "
            f"backend={self.rollout_info.backend!r}."
        )
        assert self.weight_iterator is not None, "Weight iterator is not initialized."
        self._transport.update(self.weight_iterator, **kwargs)

    def has_registered_weight_checkpoint(self) -> bool:
        transport = self._transport
        if transport is None:
            return False
        has_registered = getattr(transport, "has_registered_checkpoint", None)
        if has_registered is None:
            return False
        return bool(has_registered())

    def _set_transport(self) -> None:
        rollout_info = self.rollout_info
        assert rollout_info is not None, "bind_rollout_weight_update() must be called before setting transport."
        if rollout_info.transport_type == "ipc":
            self._transport = IPCWeightTransport(
                rank=self.rank,
                logger=self.logger,
                config=self.config,
                rollout_info=rollout_info,
            )
        elif rollout_info.transport_type == "nccl":
            self._transport = NCCLWeightTransport(rank=self.rank, logger=self.logger, rollout_info=rollout_info)
        elif rollout_info.transport_type == "checkpoint_engine":
            self._transport = CheckpointEngineWeightTransport(
                rank=self.rank,
                logger=self.logger,
                rollout_info=rollout_info,
            )
        else:
            raise NotImplementedError

    def _reset_transport(self) -> None:
        if self._transport is not None:
            self._transport.teardown()
            self._transport = None

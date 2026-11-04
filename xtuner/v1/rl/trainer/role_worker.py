"""Single-role training lifecycle shared by the actor and critic workers."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import torch
import torch.distributed as dist

from xtuner.v1.model.base import ModelItem
from xtuner.v1.rl.loss import BaseRLLossConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import get_device, get_torch_device_module

from .train_batch import resolve_mm_inputs


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

if TYPE_CHECKING:
    from logging import Logger

    from torch.distributed.device_mesh import DeviceMesh

    from xtuner.v1.data_proto.sequence_context import SequenceContext
    from xtuner.v1.engine.train_engine import TrainEngine
    from xtuner.v1.loss import BaseLossContext

    from .train_batch import TrainBatch
    from .worker import WorkerLogItem, WorkerTrainLogItem

RoleName = Literal["actor", "critic"]

# One prepared (seq_ctx, loss_ctx dict) pair per pack; roles build the dict
# (e.g. ``{"lm": ctx}`` or ``{"lm": ctx, "mtp": [depth ctxs]}``) before handing
# the step to ``_train_update``.
_PackItem = tuple["SequenceContext", dict[str, "BaseLossContext"]]


@dataclass(frozen=True)
class RoleContext:
    """Process-local dependencies handed to a role at construction.

    Args:
        data_mesh (DeviceMesh): Data/sequence-parallel mesh owned by the host.
        device (torch.device): Compute device of the role engine.
        rank (int): Global process rank.
        seed (int): Shared training seed for deterministic pass ordering.
        logger (Logger): Process logger.
        log_dir (Path | None): Experiment log directory used by profiling.
    """

    data_mesh: DeviceMesh
    device: torch.device
    rank: int
    seed: int
    logger: Logger
    log_dir: Path | None


class RoleWorker(ABC):
    """One role's engine, scheduler, update counters and placement facts.

    A role owns only its own engine. It does not hold the host or read the
    host's batch cache: every fit/collect call receives the ``TrainBatch``
    explicitly. Model/optimizer placement is recorded per role and only moved
    through the onload/offload methods below.
    """

    role_name: ClassVar[RoleName]

    engine: TrainEngine
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    # ``finalize_metrics`` is only defined on RL loss configs.
    loss_cfg: BaseRLLossConfig | None = None

    def __init__(self, config: Any, ctx: RoleContext) -> None:
        self.config = config
        self.data_mesh = ctx.data_mesh
        self.sp_mesh = ctx.data_mesh["sp"]
        self.device = ctx.device
        self.rank = ctx.rank
        self.seed = ctx.seed
        self.logger = ctx.logger
        self.log_dir = ctx.log_dir
        # Placement facts updated by the granular move methods below.
        self._model_on_device = True
        self._optimizer_on_device = True
        self._global_train_step = 0
        self.attempted_updates = 0
        self.completed_updates = 0

    @property
    def model_on_device(self) -> bool:
        """Whether the role model currently resides on the compute device."""
        return self._model_on_device

    @property
    def optimizer_on_device(self) -> bool:
        """Whether the role optimizer state currently resides on the compute
        device."""
        return self._optimizer_on_device

    def require_model_residency(self) -> None:
        """Fail fast when the role model is not resident for a forward pass."""
        if not self._model_on_device:
            raise RuntimeError(f"{self.role_name} model is not on the compute device")

    def require_train_residency(self) -> None:
        """Fail fast when either the model or optimizer is off the compute
        device."""
        self.require_model_residency()
        if not self._optimizer_on_device:
            raise RuntimeError(f"{self.role_name} optimizer is not on the compute device")

    def onload(self) -> None:
        """Move model + optimizer to the compute device as a whole."""
        self.onload_model()
        self.onload_optimizer()

    def offload(self) -> None:
        """Move model + optimizer to the CPU as a whole."""
        self.offload_optimizer()
        self.offload_model()

    def onload_model(self) -> bool:
        """Move the model to the compute device; return whether it moved."""
        moved = self.engine.put_model_to_device(self.device)
        if not moved:
            self.logger.info("Skip model onload because model placement is unchanged.")
        self._model_on_device = True
        return moved

    def onload_optimizer(self) -> bool:
        """Move the optimizer to the compute device; return whether it
        moved."""
        moved = self.engine.put_optimizer_to_device(self.device)
        if not moved:
            self.logger.info("Skip optimizer onload because no optimizer state needs to be moved.")
        self._optimizer_on_device = True
        return moved

    def offload_model(self) -> bool:
        """Move the model to the CPU; return whether it moved."""
        moved = self.engine.put_model_to_device("cpu")
        self._model_on_device = False
        if not moved:
            self.logger.info("Skip model offload because model placement is unchanged.")
            return False
        self.logger.info(
            f"Offloaded model to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, "
            f"reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )
        return True

    def offload_optimizer(self) -> bool:
        """Move the optimizer to the CPU; return whether it moved."""
        moved = self.engine.put_optimizer_to_device("cpu")
        self._optimizer_on_device = False
        if not moved:
            self.logger.info("Skip optimizer offload because no optimizer state needs to be moved.")
            return False
        self.logger.info(
            f"Offloaded optimizer to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, "
            f"reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )
        return True

    @abstractmethod
    def fit(self, batch: TrainBatch) -> WorkerLogItem:
        """Train one round on the complete local batch.

        Args:
            batch (TrainBatch): The batch installed by the worker.

        Returns:
            WorkerLogItem: Role training metrics.
        """

    def save(self, path: Path, no_save_optimizer: bool = False) -> None:
        """Save the role checkpoint under ``path``.

        Args:
            path (Path): Role directory chosen by the host (``actor``/``critic``).
            no_save_optimizer (bool): Skip optimizer state.

        Returns:
            None
        """
        self.engine.save_dcp(weights_dir=path, save_optimizer=not no_save_optimizer)
        path.mkdir(parents=True, exist_ok=True)
        rng_state: dict[str, object] = {"torch": torch.get_rng_state()}
        if torch.cuda.is_initialized():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, path / f"rng_state_rank{self.rank}.pt")
        if self.rank == 0:
            torch.save(
                {
                    "attempted_updates": self.attempted_updates,
                    "completed_updates": self.completed_updates,
                    "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                    # LambdaLR.state_dict does not store the lambda closure; persist its definition too.
                    "scheduler_config": self._scheduler_config(),
                },
                path / "role_state.pt",
            )

    def resume(self, path: Path, cfg: LoadCheckpointConfig) -> None:
        """Restore the role checkpoint under ``path``.

        Args:
            path (Path): Role directory chosen by the host.
            cfg (LoadCheckpointConfig): Optimizer/scheduler loading switches.

        Returns:
            None
        """
        if not (path / ".metadata").is_file():
            raise FileNotFoundError(f"Checkpoint at {path} has no DCP metadata.")
        initial_lrs = self.lr_scheduler.get_last_lr() if self.lr_scheduler is not None else None
        self.engine.load_dcp(
            weights_dir=path,
            load_states=cfg.load_optimizer_states,
            load_args=cfg.load_optimizer_args,
        )
        state_file = path / "role_state.pt"
        if not state_file.is_file():
            raise FileNotFoundError(f"Role checkpoint at {path} has no role_state.pt.")
        state = torch.load(state_file, weights_only=True)
        if cfg.load_scheduler and self.lr_scheduler is not None:
            if state["scheduler_config"] != self._scheduler_config():
                raise ValueError("Scheduler configuration changed; explicitly choose load_scheduler=False")
            self.lr_scheduler.load_state_dict(state["scheduler"])
        self.attempted_updates = state["attempted_updates"]
        self.completed_updates = state["completed_updates"]
        if self.lr_scheduler is not None:
            lrs = cast(list[float], self.lr_scheduler.get_last_lr() if cfg.load_scheduler else initial_lrs)
            for group, lr in zip(self.engine.optimizer.param_groups, lrs):
                group["lr"] = lr

        rng_file = path / f"rng_state_rank{self.rank}.pt"
        if not rng_file.is_file():
            raise FileNotFoundError(f"Role checkpoint at {path} has no rng_state_rank{self.rank}.pt.")
        rng_state = torch.load(rng_file, weights_only=True)
        torch.set_rng_state(rng_state["torch"])
        if "cuda" in rng_state and torch.cuda.is_initialized():
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    def save_hf(self, hf_dir: Path, save_dtype: torch.dtype) -> None:
        """Save the role model in Hugging Face format under ``hf_dir``.

        Args:
            hf_dir (Path): Output directory.
            save_dtype (torch.dtype): Export dtype.

        Returns:
            None
        """
        self.engine.save_hf(hf_dir=hf_dir, save_dtype=save_dtype)

    def _scheduler_config(self) -> dict | None:
        if self.lr_scheduler is None:
            return None
        return {"lr_cfg": self.config.lr_cfg.model_dump(), "scheduler_steps": self.config.scheduler_steps}

    def _resolve_refs(self, seq_ctx: SequenceContext) -> None:
        """Materialize object refs on the CPU side before device transfer.

        Args:
            seq_ctx (SequenceContext): Packed CPU context.

        Returns:
            None
        """
        resolve_mm_inputs(seq_ctx)

    def _to_device_seq(self, seq_ctx: SequenceContext) -> SequenceContext:
        """Copy a packed CPU context to the device and split it for SP.

        Args:
            seq_ctx (SequenceContext): Packed CPU context.

        Returns:
            SequenceContext: The device context of this rank (SP-split when configured).
        """
        seq_ctx.offload_rollout_routed_experts = getattr(self.config, "offload_rollout_routed_experts", False)
        device_seq = seq_ctx.copy().to(self.device)
        if self.sp_mesh.size() > 1:
            device_seq = device_seq.split(sequence_parallel_mesh=self.sp_mesh)
        return device_seq

    def _train_update(self, items: list[ModelItem]) -> WorkerTrainLogItem:
        """Run one optimizer update for an already calibrated step.

        Args:
            items (list[ModelItem]): Engine inputs of one optimizer step.

        Returns:
            WorkerTrainLogItem: Per-step training log item.
        """
        self._global_train_step += 1
        global_train_step = self._global_train_step
        self.attempted_updates += 1

        # Make max_memory/reserved_memory below this step's peak instead of the
        # process-lifetime peak (~2us, no sync).
        DEVICE_MODULE.reset_peak_memory_stats()
        with self._maybe_profiling(global_train_step):
            train_step_info = self.engine.train_step(data_batches=items)
        grad_norm = self.engine.clip_grad_norm()
        applied = self.engine.step_optimizer(grad_norm)
        if applied:
            self.completed_updates += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        engine_logs_info = dict(train_step_info.pop("logs_info"))  # type: ignore[misc]
        engine_extra_info = train_step_info.pop("extra_info")  # type: ignore[misc]
        train_step_info.pop("total_loss", None)  # type: ignore[misc]
        if hasattr(engine_extra_info, "get"):
            extra_info_dict = engine_extra_info.get()
        else:
            extra_info_dict = engine_extra_info
        extra_info_dict = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in extra_info_dict.items()
            if isinstance(v, (torch.Tensor, int, float))
        }
        if self.loss_cfg is not None:
            extra_info_dict = self.loss_cfg.finalize_metrics(extra_info_dict, DEVICE)
        extra_info_dict.update(self._extra_step_metrics())
        max_memory = DEVICE_MODULE.max_memory_allocated() / (1024**3)
        reserved_memory = DEVICE_MODULE.max_memory_reserved() / (1024**3)

        train_log_item = cast(
            "WorkerTrainLogItem",
            {
                **engine_logs_info,
                **train_step_info,
                **extra_info_dict,
                "grad_norm": grad_norm.item(),
                "max_memory": max_memory,
                "reserved_memory": reserved_memory,
            },
        )
        log_str = ", ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in train_log_item.items()
            if not key.startswith("reduced_train_policy_") and key != "max_ratio"
        )
        self.logger.info(f"Rank{self.rank} GlobalStep {global_train_step}: {log_str}")
        return train_log_item

    def _maybe_profiling(self, global_train_step: int) -> Any:
        """Profile matching steps when the role enables profiling; no-op
        default."""
        return contextlib.nullcontext()

    def _extra_step_metrics(self) -> dict[str, float]:
        """Role-specific per-step metrics merged into the training log."""
        return {}

    def _masked_token_count(self, batch: TrainBatch, packs: list[list[int]]) -> int:
        """Count trainable tokens across all ranks for a set of packs.

        Args:
            batch (TrainBatch): The local batch.
            packs (list[list[int]]): Local pack indices.

        Returns:
            int: Global masked-token count.
        """
        local_tokens = batch.local_masked_token_count(packs)
        # NCCL-only process groups reject CPU tensors; keep it on the device.
        total = torch.tensor(float(local_tokens), device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return int(total.item())

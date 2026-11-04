"""The critic role: PPO value collection and clipped value-head training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, cast

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, model_validator

from xtuner.v1.config.optim import LRConfig, OptimConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.utils import sp_gather
from xtuner.v1.model.base import ModelItem, TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.advantage.gae import DecoupledGAEEstimator, GAEEstimator, normalize_with_moments
from xtuner.v1.rl.advantage.stats import TokenStats
from xtuner.v1.rl.loss import CriticLossConfig
from xtuner.v1.utils import get_device

from .pack import TokenBatchProto
from .role_worker import RoleContext, RoleWorker
from .schedule import build_lr_scheduler, partition_indices, pass_order
from .train_batch import TrainBatch, iter_cpu_packs, resolve_mm_inputs


if TYPE_CHECKING:
    from xtuner.v1.data_proto.sequence_context import SequenceContext
    from xtuner.v1.rl.advantage.base import TokenAdvantageEstimator

    from .worker import WorkerLogItem


DEVICE = get_device()


class CriticWorkerConfig(BaseModel):
    """Configuration for an attached PPO critic worker.

    The critic is hosted by the actor Ray worker instead of being spawned as a
    second Ray actor. It shares the host process' distributed process group but
    owns its own value-head model and optimizer.

    FSDP and sequence-parallel topology is owned by the host ``WorkerConfig`` and
    is intentionally not duplicated here.

    Args:
        model_cfg (TransformerConfig | BaseComposeConfig): Value-head model config.
        optim_cfg (OptimConfig): Critic optimizer config.
        loss_cfg (CriticLossConfig): Clipped value loss config.
        lr_cfg (LRConfig): Critic learning-rate config.
        load_from (str | Path): Checkpoint to load (actor or critic).
        num_passes (int): Number of full passes over the local packs.
        optimizer_steps_per_pass (int): Maximum optimizer updates per pass; 1 means
            all packs of the pass form one update.
        scheduler_steps (int): Total scheduler steps.
    """

    model_config = ConfigDict(title="Critic worker config", extra="forbid", arbitrary_types_allowed=True)
    model_cfg: TransformerConfig | BaseComposeConfig
    optim_cfg: OptimConfig
    loss_cfg: CriticLossConfig = CriticLossConfig()
    lr_cfg: LRConfig
    load_from: str | Path
    num_passes: int = 1
    optimizer_steps_per_pass: int = 1
    scheduler_steps: int = 1_000_000

    @model_validator(mode="after")
    def _validate_schedule(self) -> CriticWorkerConfig:
        if self.num_passes <= 0:
            raise ValueError("num_passes must be positive.")
        if self.optimizer_steps_per_pass <= 0:
            raise ValueError("optimizer_steps_per_pass must be positive.")
        return self


class CriticWorker(RoleWorker):
    """Value worker used by PPO.

    The critic owns an independent value-head engine while reusing the host's sequence-parallel mesh and process
    lifetime.
    """

    role_name = "critic"

    def __init__(self, worker_cfg: CriticWorkerConfig, ctx: RoleContext, fsdp_cfg) -> None:
        super().__init__(worker_cfg, ctx)
        model_cfg = worker_cfg.model_cfg
        head_cfg = model_cfg.text_config if isinstance(model_cfg, BaseComposeConfig) else model_cfg
        if head_cfg.head_config.type != "value_head":
            raise ValueError("CriticWorker requires model_cfg.head_config.type='value_head'")
        self.engine = TrainEngine(
            optim_cfg=worker_cfg.optim_cfg,
            fsdp_cfg=fsdp_cfg,
            model_cfg=worker_cfg.model_cfg,
        )
        # Actor checkpoints have no value_head.weight; the model handles the
        # missing-head exception and initializes it.
        self.engine.from_hf(worker_cfg.load_from, strict=True)
        self.lr_scheduler = build_lr_scheduler(self.engine.optimizer, worker_cfg.lr_cfg, worker_cfg.scheduler_steps)
        self.loss_cfg = None  # CriticLossConfig has no finalize_metrics

    def _resolve_refs(self, seq_ctx: SequenceContext) -> None:
        """Multimodal refs only: the critic routes experts on its own."""
        resolve_mm_inputs(seq_ctx)
        seq_ctx.rollout_routed_experts = None

    def forward_only(self, seq_ctx: SequenceContext) -> torch.Tensor:
        """Return raw value-head predictions of one sequence context.

        Args:
            seq_ctx (SequenceContext): Device sequence context.

        Returns:
            torch.Tensor: Value predictions with shape ``[B, S, 1]``.
        """
        with torch.no_grad():
            # Value collection is inference: loss_ctx=None returns the value-head logits.
            output = self.engine.model(seq_ctx=seq_ctx, loss_ctx=None)
        values = output.logits
        if values is None:
            raise RuntimeError("critic forward did not return value predictions")
        return values.float()

    def collect_values(self, batch: TrainBatch, *, return_values: bool) -> TokenBatchProto | None:
        """Forward every pack and cache complete per-sample value rows.

        Args:
            batch (TrainBatch): Local batch.
            return_values (bool): Return the representative rows for session GAE.

        Returns:
            TokenBatchProto | None: Per-sample value rows when requested.
        """
        pack_indices = [pack for step in batch.plan.steps for pack in step]
        rows: list[torch.Tensor | None] = [None] * len(batch.plan.global_indices)
        self.engine._maybe_precompute_float8_dynamic_scale_for_fsdp()
        for full_pack in iter_cpu_packs(batch, pack_indices, self._resolve_refs):
            full_seq = full_pack.full_data["seq_ctx"]
            pack_length = full_seq.input_ids.shape[1]
            seq_ctx = self._to_device_seq(full_seq)
            local = self.forward_only(seq_ctx).squeeze(-1)
            full = sp_gather(local, self.sp_mesh, dim=1) if self.sp_mesh.size() > 1 else local
            full = full[:, :pack_length]  # remove SP padding; unpack removes pack padding
            for sample_index, value in batch.unpack_rows(full_pack, full):
                rows[sample_index] = value
            # Empty packs still run the same forward/gather and unpack to [].
        if any(row is None for row in rows):
            raise RuntimeError("pack plan did not cover every local sample")
        resolved = [cast(torch.Tensor, row) for row in rows]
        batch.install_old_values(resolved)
        return TokenBatchProto.concat([row.flatten() for row in resolved]) if return_values else None

    def compute_segment_targets(
        self,
        batch: TrainBatch,
        estimator: TokenAdvantageEstimator,
        *,
        contribute: bool,
    ) -> TokenStats:
        """Compute segment GAE targets from the cached full values.

        Every rank joins the normalization reduction; only one replica per DP rank
        contributes moments and logging statistics.

        Args:
            batch (TrainBatch): Local batch with complete old values.
            estimator (TokenAdvantageEstimator): Segment GAE estimator.
            contribute (bool): Whether this worker is the DP representative.

        Returns:
            TokenStats: Normalized action-token statistics (empty for non-representatives).
        """
        assert isinstance(estimator, (GAEEstimator, DecoupledGAEEstimator))
        trajectory = batch.trajectory_data()
        advantages, returns = estimator.compute_segment_gae(**trajectory)
        mask = trajectory["mask"]
        moments = torch.zeros(3, dtype=torch.float64, device=self.device)
        if contribute:
            valid = advantages[mask]
            moments[:] = torch.tensor(
                [valid.double().sum(), valid.double().square().sum(), valid.numel()],
                device=self.device,
                dtype=torch.float64,
            )
        if estimator.normalize_actor_advantage:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(moments, op=dist.ReduceOp.SUM)
            advantages = normalize_with_moments([advantages], [mask], moments.cpu())[0]

        lengths = (trajectory["cu_seq_lens"][1:] - trajectory["cu_seq_lens"][:-1]).tolist()
        advantage_rows = list(advantages.split(lengths)) if sum(lengths) else []
        return_rows = list(returns.split(lengths)) if sum(lengths) else []
        batch.install_ppo_targets(
            TokenBatchProto.concat(advantage_rows),
            TokenBatchProto.concat(return_rows),
        )
        return TokenStats.from_masked(advantages, mask) if contribute else TokenStats.empty()

    def fit(self, batch: TrainBatch) -> WorkerLogItem:
        """Train the clipped value head over the configured passes.

        Args:
            batch (TrainBatch): Local batch with complete returns and old values.

        Returns:
            WorkerLogItem: Critic training metrics.
        """
        batch.require_targets("critic")
        worker_log_item: WorkerLogItem = {"train_entropy": 0.0, "train_metrics": []}
        for update_packs in self._updates(batch):
            packs = [list(pack) for pack in update_packs]
            if self._masked_token_count(batch, packs) == 0:
                # All ranks observe the same global count and skip together.
                continue
            items: list[ModelItem] = []
            for full_pack in iter_cpu_packs(batch, update_packs, self._resolve_refs):
                seq_ctx = self._to_device_seq(full_pack.full_data["seq_ctx"])
                ctx = self.config.loss_cfg.build(full_pack.full_data, sp_mesh=self.sp_mesh, device=self.device)
                assert ctx is not None
                items.append(ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": ctx}))
            self.config.loss_cfg.loss_ctx_cls.build_batches([item["loss_ctx"]["lm"] for item in items])  # type: ignore[index]
            worker_log_item["train_metrics"].append(self._train_update(items))
        return worker_log_item

    def _updates(self, batch: TrainBatch) -> Iterator[list[tuple[int, ...]]]:
        """Yield one optimizer update's packs for every notified pass.

        Args:
            batch (TrainBatch): Local batch.

        Returns:
            Iterator[list[tuple[int, ...]]]: Pack groups of each optimizer update.
        """
        packs = [pack for step in batch.plan.steps for pack in step]
        rollout_idx = batch.attrs["rollout_idx"]
        for pass_index in range(self.config.num_passes):
            order = pass_order(len(packs), rollout_idx, pass_index, self.seed)
            for group in partition_indices(order, self.config.optimizer_steps_per_pass):
                yield [packs[index] for index in group]

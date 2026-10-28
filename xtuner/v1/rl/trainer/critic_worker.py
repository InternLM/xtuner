"""The critic role worker: PPO value head inside a ``TrainingWorker`` host."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, cast

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict

from xtuner.v1.config.optim import LRConfig, OptimConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.utils import sp_split
from xtuner.v1.model.base import ModelItem, TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.advantage.gae import DecoupledGAEEstimator, GAEEstimator, normalize_with_moments
from xtuner.v1.rl.loss import CriticLossConfig, CriticLossContext
from xtuner.v1.utils import get_device

from .pack import DPRankPackIndices, TokenBatchProto
from .role_worker import RoleWorker, _PackItem, resolve_mm_inputs
from .schedule import build_lr_scheduler, partition_indices, pass_order


if TYPE_CHECKING:
    from xtuner.v1.data_proto.sequence_context import SequenceContext
    from xtuner.v1.loss import BaseLossContext

    from .worker import TrainingWorker, TrainSample, WorkerLogItem

DEVICE = get_device()


class CriticWorkerConfig(BaseModel):
    """Configuration for an attached PPO critic worker.

    A critic is hosted by the actor Ray worker instead of being spawned as a second Ray actor. It shares the host
    process' distributed process group, but owns its own value-head model and optimizer.

    FSDP and sequence-parallel topology is owned by the host ``WorkerConfig`` and is
    intentionally not duplicated here.
    """

    model_config = ConfigDict(title="Critic worker config", extra="forbid", arbitrary_types_allowed=True)
    model_cfg: TransformerConfig | BaseComposeConfig
    optim_cfg: OptimConfig
    loss_cfg: CriticLossConfig = CriticLossConfig()
    lr_cfg: LRConfig
    load_from: str | Path
    optimizer_steps: int = 1
    num_passes: int = 1
    optimizer_steps_per_pass: int | None = None
    scheduler_steps: int = 1_000_000


class CriticWorker(RoleWorker):
    """Value worker used by PPO.

    The class deliberately does not inherit ``SingleAcceleratorWorker``: its
    host ``TrainingWorker`` has already initialized the process group. The
    critic owns an independent value-head engine, while reusing the host's
    sequence-parallel mesh, pack plan, and process lifetime.
    """

    role_name = "critic"

    def __init__(self, worker_cfg: CriticWorkerConfig, host: TrainingWorker) -> None:
        super().__init__(host)
        self.config = worker_cfg
        self.logger = host.logger
        self.rank = host.rank
        if worker_cfg.num_passes <= 0:
            raise ValueError("num_passes must be positive.")
        if worker_cfg.optimizer_steps_per_pass is not None and worker_cfg.optimizer_steps_per_pass <= 0:
            raise ValueError("optimizer_steps_per_pass must be positive.")
        model_cfg = worker_cfg.model_cfg
        head_cfg = model_cfg.text_config if isinstance(model_cfg, BaseComposeConfig) else model_cfg
        if getattr(head_cfg, "head_type", "lm_head") != "value_head":
            raise ValueError("CriticWorker requires model_cfg.head_type='value_head'")
        fsdp_cfg = host.config.fsdp_cfg
        self.engine = TrainEngine(
            optim_cfg=worker_cfg.optim_cfg,
            fsdp_cfg=fsdp_cfg,
            model_cfg=worker_cfg.model_cfg,
        )
        # Actor checkpoints do not contain value_head.weight. Load the backbone with
        # strict=False; the missing critic head is initialized after checkpoint loading.
        self.engine.from_hf(worker_cfg.load_from, strict=False)
        self._optimizer_steps = worker_cfg.optimizer_steps
        self._num_passes = worker_cfg.num_passes
        self._optimizer_steps_per_pass = worker_cfg.optimizer_steps_per_pass
        self.lr_scheduler = build_lr_scheduler(self.engine.optimizer, worker_cfg.lr_cfg, worker_cfg.scheduler_steps)
        self.loss_cfg = None  # CriticLossConfig has no finalize_metrics

    def _resolve_refs(self, seq_ctx: SequenceContext) -> None:
        """Multimodal refs only: the critic routes experts on its own, so
        routed-experts refs are dropped instead of resolved."""
        resolve_mm_inputs(seq_ctx)
        seq_ctx.rollout_routed_experts = None

    def forward_only(self, seq_ctx: SequenceContext) -> torch.Tensor:
        with torch.no_grad():
            # Value collection is inference: loss_ctx=None returns the value-head
            # logits. TrainEngine.forward_only always wraps a LogProbContext and
            # would compute token logprobs instead.
            output = self.engine.model(seq_ctx=seq_ctx, loss_ctx=None)
        values = output.logits
        if values is None:
            raise RuntimeError("critic forward did not return value predictions")
        return values.float()

    def collect_values(self, return_values: bool) -> TokenBatchProto | None:
        """Forward the value head over every pack and cache per-sample rows.

        The one path that deliberately skips the iter-pack pipeline: collect
        packs carry only seq-level data (CPU copies are input_ids scale) and
        the barrier aligns "every rank finished CPU packing" against "entering
        the FSDP forward sequence", preventing a rank still packing on CPU
        from desynchronizing the collectives. The forward runs without SP
        split so every sp rank sees the full sequence and can slice complete
        per-sample value rows.
        """
        samples, plan, _batch_attr = self._cached()
        flat_packs = [pack for step in plan for pack in step]
        all_packs = [self.host.pack_seq_ctx(pack) for pack in flat_packs]
        dist.barrier()
        values: list[torch.Tensor | None] = [None] * len(samples)
        for pack, seq_ctx in zip(flat_packs, all_packs):
            self._resolve_refs(seq_ctx)
            seq_ctx = seq_ctx.to(DEVICE)
            raw = self.forward_only(seq_ctx).detach().float().reshape(-1)
            offset = 0
            for index in pack:
                length = samples[index].shifted_labels.size(1)
                values[index] = raw[offset : offset + length].detach().float().cpu()
                offset += length
        collected: list[torch.Tensor] = []
        for value in values:
            if value is None:
                raise RuntimeError("pack plan did not cover every local sample")
            collected.append(value)
        batch = TokenBatchProto.concat(collected)
        for item, row in zip(samples, batch.chunk()):
            item.values = row.unsqueeze(0)
        return batch if return_values else None

    def compute_segment_gae(
        self,
        *,
        estimator: GAEEstimator | DecoupledGAEEstimator,
        contribute_to_global_stats: bool,
    ) -> dict[str, float]:
        """Compute independent segment GAE targets in this worker.

        ``estimator`` supplies the discount parameters (coupled or decoupled) through
        ``compute_segment_gae``. ``contribute_to_global_stats`` is true for one worker
        per data replica. Replica-zero moments preserve the controller's previous
        global-normalization statistics while all replicas can consume their locally
        cached values for actor/critic updates.
        """
        samples = self._cached()[0]
        if any(sample.values is None for sample in samples):
            raise RuntimeError("Segment GAE requires cached critic values")

        moments = torch.zeros(3, dtype=torch.float64, device=DEVICE)
        for sample in samples:
            assert sample.values is not None
            reward_score = sample.reward_score
            if reward_score is None:
                raise ValueError("Segment GAE requires a reward score for every sample")
            values = sample.values.detach().float().reshape(-1)
            action_mask = sample.shifted_labels.reshape(-1).ne(-100)
            advantages, returns = estimator.compute_segment_gae(
                values,
                torch.tensor([reward_score], dtype=torch.float32),
                action_mask,
                torch.tensor([0, values.numel()], dtype=torch.int32),
            )
            sample.advantages = advantages.reshape(1, -1)
            sample.returns = returns.reshape(1, -1)
            if contribute_to_global_stats:
                valid_advantages = advantages[action_mask]
                moments[0] += valid_advantages.double().sum()
                moments[1] += valid_advantages.double().square().sum()
                moments[2] += valid_advantages.numel()

        if estimator.normalize_actor_advantage:
            # The worker only accumulates moments and all-reduces them across replicas;
            # the normalization math itself lives in the estimator.
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(moments, op=dist.ReduceOp.SUM)
            moments = moments.cpu()
            staged: list[tuple[TrainSample, torch.Tensor, torch.Tensor]] = []
            for sample in samples:
                sample_advantages = sample.advantages
                if sample_advantages is None:
                    continue
                staged.append((sample, sample_advantages, sample.shifted_labels.ne(-100)))
            normalized = normalize_with_moments(
                [sample_advantages for _, sample_advantages, _ in staged],
                [mask for _, _, mask in staged],
                moments,
            )
            for (sample, _, _), normalized_advantages in zip(staged, normalized):
                sample.advantages = normalized_advantages

        if not contribute_to_global_stats:
            return {}

        flat_advantages = torch.cat(
            [
                sample.advantages.reshape(-1)[sample.shifted_labels.reshape(-1).ne(-100)]
                for sample in samples
                if sample.advantages is not None
            ]
        )
        if flat_advantages.numel() == 0:
            return {"sum": 0.0, "count": 0.0, "min": 0.0, "max": 0.0}
        return {
            "sum": float(flat_advantages.sum().item()),
            "count": float(flat_advantages.numel()),
            "min": float(flat_advantages.min().item()),
            "max": float(flat_advantages.max().item()),
        }

    def _build_step_input(self, items: list[_PackItem]) -> list[ModelItem]:
        batched = cast(
            list[CriticLossContext],
            CriticLossContext.build_batches([item[1]["lm"] for item in items]),  # type: ignore[misc]
        )
        return [
            ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx}) for (seq_ctx, _), loss_ctx in zip(items, batched)
        ]

    def pack_loss_ctx(self, pack: list[int], seq_ctx: SequenceContext) -> dict[str, BaseLossContext]:
        """Build the critic's packed loss context for one pack.

        ``old_values`` fills zeros when missing for the whole pack: a forward
        here would be rank-conditional and only some ranks would enter FSDP
        all-gather, deadlocking the others. This is the critic's own concern
        and stays out of the host.
        """
        samples = self._cached()[0]
        if pack:
            rows: dict[str, list[torch.Tensor] | None] = {
                "shifted_labels": [samples[index].shifted_labels for index in pack],
                "returns": [cast(torch.Tensor, samples[index].returns) for index in pack],
            }
            old_value_rows = [samples[index].values for index in pack]
            # old_values is pack-wide all-or-none.
            if any(row is not None for row in old_value_rows) and not all(row is not None for row in old_value_rows):
                raise ValueError("old_values must be all-present or all-absent within a pack")
            rows["old_values"] = (
                [cast(torch.Tensor, row) for row in old_value_rows]
                if all(row is not None for row in old_value_rows)
                else None
            )
        else:
            # Empty pack: empty rows mark the fields present.
            rows = {"shifted_labels": [], "returns": [], "old_values": []}
        packed = self._pack_loss_rows(samples, pack, rows)
        shifted_labels = cast(torch.Tensor, packed["shifted_labels"]).to(DEVICE)
        returns = cast(torch.Tensor, packed["returns"]).to(DEVICE)
        old_values = packed["old_values"]
        sp_mesh = self.host.sp_mesh
        if sp_mesh.size() > 1:
            shifted_labels = sp_split(shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100)
            returns = sp_split(returns, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        if old_values is None:
            old_values = torch.zeros_like(returns)
        else:
            old_values = old_values.to(DEVICE)
            if sp_mesh.size() > 1:
                old_values = sp_split(old_values, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        loss_ctx = self.config.loss_cfg.build(
            data={"shifted_labels": shifted_labels, "returns": returns, "old_values": old_values},
            sp_mesh=None,
        )
        if loss_ctx is None:
            raise ValueError("critic loss context could not be built from returns")
        return {"lm": loss_ctx}

    def fit(self) -> WorkerLogItem:
        """Train the clipped value head for ``num_passes`` over the cached
        plan.

        Multi-pass expansion is index-level; materialization is deferred to
        each update's ``_iter_packs`` stream (re-cat per pass), and empty
        updates are dropped before materialization using the global masked
        token count so the filter stays rank-symmetric.
        """
        samples = self._cached()[0]
        if any(sample.returns is None for sample in samples):
            raise ValueError("PPO critic fit requires per-token returns")
        update_plans = self._critic_updates(self._plan())

        def optimizer_step_packs() -> Iterator[list[tuple[SequenceContext, dict[str, BaseLossContext]]]]:
            for packs in update_plans:
                # Each update runs its own prefetch stream; a repeated pass
                # re-materializes (CPU re-cat) instead of staying resident.
                if self._masked_token_count(packs) == 0:
                    continue
                yield [
                    (seq_ctx, self.pack_loss_ctx(pack, seq_ctx))
                    for pack, seq_ctx in self._iter_packs(packs, split=True)
                ]

        return self._train_optimizer_steps(optimizer_step_packs())

    def _critic_updates(self, plan: DPRankPackIndices) -> list[list[list[int]]]:
        """Repeat the local pack indices for each critic pass.

        Without ``optimizer_steps_per_pass``, one pass follows the actor pack plan.
        Otherwise every pass flattens the pack indices, shuffles after the first
        pass, and splits them into ``optimizer_steps_per_pass`` optimizer updates.
        """
        if self._optimizer_steps_per_pass is None:
            if len(plan) > self._optimizer_steps:
                raise ValueError(
                    f"Pack plan optimizer steps {len(plan)} exceed configured optimizer_steps {self._optimizer_steps}"
                )
            return list(plan)
        packs = [pack for step in plan for pack in step]
        seed = int(self.host.config.seed or 0)
        rollout_idx = self._cached()[2]["rollout_idx"]
        updates: list[list[list[int]]] = []
        for pass_index in range(self._num_passes):
            order = pass_order(len(packs), rollout_idx, pass_index, seed)
            for indices in partition_indices(order, self._optimizer_steps_per_pass):
                updates.append([packs[index] for index in indices])
        return updates

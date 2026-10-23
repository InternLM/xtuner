from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import ray
import torch
from pydantic import BaseModel, ConfigDict

from xtuner.v1.config.optim import LRConfig, OptimConfig
from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.utils import sp_split
from xtuner.v1.model.base import ModelItem, TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.advantage.gae import compute_ppo_targets
from xtuner.v1.rl.loss import CriticLossConfig, CriticLossContext
from xtuner.v1.utils import get_device, get_torch_device_module

from .pack import DPRankPackIndices, TokenTensorBatch
from .schedule import build_lr_scheduler, optimizer_step_succeeds, partition_indices, pass_order


if TYPE_CHECKING:
    from .worker import TrainBatchAttr, TrainingWorker, WorkerLogItem, WorkerTrainLogItem

DEVICE_MODULE = get_torch_device_module()

DEVICE = get_device()


class CriticWorkerConfig(BaseModel):
    """Configuration for an attached PPO critic worker.

    A critic is hosted by the actor Ray worker instead of being spawned as a second Ray actor. It shares the actor
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


class CriticWorker:
    """Attached value worker used by PPO.

    The class deliberately does not inherit ``SingleAcceleratorWorker``: its
    host ``TrainingWorker`` has already initialized the process group. A critic
    owns an independent value-head engine, while reusing the host's
    sequence-parallel mesh, pack plan, and process lifetime.
    """

    def __init__(self, worker_cfg: CriticWorkerConfig, host: TrainingWorker) -> None:
        if host is None:
            raise ValueError("CriticWorker must be attached to a TrainingWorker")

        self.config = worker_cfg
        self.host = host
        self.sp_mesh = host.sp_mesh
        if worker_cfg.num_passes <= 0:
            raise ValueError("num_passes must be positive.")
        if worker_cfg.optimizer_steps_per_pass is not None and worker_cfg.optimizer_steps_per_pass <= 0:
            raise ValueError("optimizer_steps_per_pass must be positive.")
        model_cfg = worker_cfg.model_cfg
        head_cfg = model_cfg.text_config if isinstance(model_cfg, BaseComposeConfig) else model_cfg
        if getattr(head_cfg, "head_type", "lm_head") != "value_head":
            raise ValueError("CriticWorker requires model_cfg.head_type='value_head'")
        fsdp_cfg = host.config.fsdp_cfg
        if not fsdp_cfg.torch_compile:
            worker_cfg.model_cfg.compile_cfg = False
        self._engine = TrainEngine(
            optim_cfg=worker_cfg.optim_cfg,
            fsdp_cfg=fsdp_cfg,
            model_cfg=worker_cfg.model_cfg,
        )
        # Actor checkpoints do not contain value_head.weight. Load the backbone with
        # strict=False; the missing critic head is initialized after checkpoint loading.
        self._engine.from_hf(worker_cfg.load_from, strict=False)
        self._optimizer_steps = worker_cfg.optimizer_steps
        self._num_passes = worker_cfg.num_passes
        self._optimizer_steps_per_pass = worker_cfg.optimizer_steps_per_pass
        self._scheduler = build_lr_scheduler(self._engine.optimizer, worker_cfg.lr_cfg, worker_cfg.scheduler_steps)

    def _require_host(self) -> TrainingWorker:
        if self.host is None:
            raise RuntimeError("CriticWorker must be attached to a TrainingWorker")
        return self.host

    def _token_tensor(self, state: RolloutState, key: str, shifted_len: int) -> torch.Tensor | None:
        raw = state.extra_fields.get(key)
        if raw is None:
            return None
        values = raw.detach().cpu().tolist() if isinstance(raw, torch.Tensor) else list(raw)
        if len(values) == shifted_len + 1:
            values = values[1:]
        if len(values) != shifted_len:
            raise ValueError(
                f"RolloutState.extra_fields[{key!r}] length {len(values)} does not match "
                f"shifted labels length {shifted_len}"
            )
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    def _pack_one_batch(
        self,
        samples: list[tuple[SequenceContext, torch.Tensor, torch.Tensor, torch.Tensor | None]],
        batch_attr: TrainBatchAttr,
    ) -> tuple[SequenceContext, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        host = self._require_host()
        real_tokens = 0
        for seq_ctx, _, _, _ in samples:
            assert seq_ctx.input_ids is not None
            real_tokens += seq_ctx.input_ids.shape[1]
        padding_len = host.config.pack_max_length - real_tokens
        if padding_len < 0:
            raise ValueError(f"Pack tokens {real_tokens} exceed pack_max_length {host.config.pack_max_length}")

        if samples:
            # old_values is pack-wide all-or-none.
            if any(sample[3] is not None for sample in samples) and not all(
                sample[3] is not None for sample in samples
            ):
                raise ValueError("old_values must be all-present or all-absent within a pack")
            packed = host.cat_pack_fields(
                {
                    "shifted_labels": [sample[1] for sample in samples],
                    "returns": [sample[2] for sample in samples],
                    "old_values": (
                        [cast(torch.Tensor, sample[3]) for sample in samples]
                        if all(sample[3] is not None for sample in samples)
                        else None
                    ),
                },
                padding_len,
            )
            shifted_labels = cast(torch.Tensor, packed["shifted_labels"])
            returns = cast(torch.Tensor, packed["returns"])
            old_values = cast(torch.Tensor | None, packed["old_values"])
            seq_ctx_list = [sample[0] for sample in samples]
            if padding_len > 0:
                # Pack is not full: append the padding seq_ctx; padded rows carry no gradient.
                seq_ctx_list.append(
                    host.pack_padding_seq_ctx(
                        padding_len,
                        use_3d_position_ids=batch_attr["use_3d_position_ids"],
                        has_routed_experts=batch_attr["has_routed_experts"],
                    )
                )
        else:
            # Empty pack: the whole pack materializes as padding.
            shifted_labels = host.pack_padding_loss_input("shifted_labels", padding_len)
            returns = host.pack_padding_loss_input("returns", padding_len)
            old_values = host.pack_padding_loss_input("old_values", padding_len)
            seq_ctx_list = [
                host.pack_padding_seq_ctx(
                    padding_len,
                    use_3d_position_ids=batch_attr["use_3d_position_ids"],
                    has_routed_experts=batch_attr["has_routed_experts"],
                )
            ]
        # Cat only when the pack holds multiple contexts.
        seq_ctx = seq_ctx_list[0] if len(seq_ctx_list) == 1 else SequenceContext.cat(seq_ctx_list)
        return seq_ctx, shifted_labels, returns, old_values

    def _prepare_loss(
        self,
        seq_ctx: SequenceContext,
        shifted_labels: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor | None,
    ) -> tuple[SequenceContext, CriticLossContext]:
        host = self._require_host()
        seq_ctx = host.prepare_forward_seq_ctx(seq_ctx, need_routed_experts=False)
        shifted_labels = shifted_labels.to(DEVICE)
        returns = returns.to(DEVICE)
        if self.sp_mesh is not None and self.sp_mesh.size() > 1:
            shifted_labels = sp_split(shifted_labels, sp_mesh=self.sp_mesh, split_dim=1, padding_value=-100)
            returns = sp_split(returns, sp_mesh=self.sp_mesh, split_dim=1, padding_value=0.0)
        if old_values is None:
            # Collected values are missing for this whole pack. Fill zeros locally.
            # A forward here would be rank-conditional: only ranks that own such a
            # pack would enter FSDP all-gather, and the others would deadlock.
            old_values = torch.zeros_like(returns, device=DEVICE)
        else:
            old_values = old_values.to(DEVICE)
            if self.sp_mesh is not None and self.sp_mesh.size() > 1:
                old_values = sp_split(old_values, sp_mesh=self.sp_mesh, split_dim=1, padding_value=0.0)
        loss_ctx = self.config.loss_cfg.build(
            data={"shifted_labels": shifted_labels, "returns": returns, "old_values": old_values},
            sp_mesh=None,
        )
        if loss_ctx is None:
            raise ValueError("critic loss context could not be built from returns")
        return seq_ctx, loss_ctx

    def forward_only(self, seq_ctx: SequenceContext) -> torch.Tensor:
        with torch.no_grad():
            # Value collection is inference: loss_ctx=None returns the value-head
            # logits. TrainEngine.forward_only always wraps a LogProbContext and
            # would compute token logprobs instead.
            output = self._engine.model(seq_ctx=seq_ctx, loss_ctx=None)
        values = output.logits
        if values is None:
            raise RuntimeError("critic forward did not return value predictions")
        return values.float()

    def collect_values(self, return_values: bool = True) -> TokenTensorBatch | None:
        """Return packed values and keep the per-sample rows on this worker."""
        host = self._require_host()
        train_samples, pack_plan, batch_attr = host.get_cached_batch()
        samples: list[tuple[SequenceContext, torch.Tensor, torch.Tensor, None]] = []
        for item in train_samples:
            labels = item.shifted_labels
            samples.append((item.seq_ctx, labels, torch.zeros_like(labels, dtype=torch.float32), None))
        packed: list[list[tuple[list[int], SequenceContext, torch.Tensor]]] = []
        for step in pack_plan:
            packed.append([(pack, *self._pack_one_batch([samples[i] for i in pack], batch_attr)[:2]) for pack in step])
        torch.distributed.barrier()
        values: list[torch.Tensor | None] = [None] * len(samples)
        for packed_step in packed:
            for indices, forward_ctx, _shifted_labels in packed_step:
                forward_ctx = host.prepare_forward_seq_ctx(
                    forward_ctx, sequence_parallel=False, need_routed_experts=False
                )
                raw = self.forward_only(forward_ctx).detach().float().reshape(-1)
                offset = 0
                for index in indices:
                    length = samples[index][1].size(1)
                    values[index] = raw[offset : offset + length].detach().float().cpu()
                    offset += length
        collected: list[torch.Tensor] = []
        for value in values:
            if value is None:
                raise RuntimeError("pack plan did not cover every local sample")
            collected.append(value)
        batch = TokenTensorBatch.cat_samples(collected)
        for item, row in zip(train_samples, batch.split_samples()):
            item.values = row.unsqueeze(0)
        return batch if return_values else None

    def compute_segment_gae(
        self,
        *,
        gae_gamma: float,
        gae_lambda: float,
        normalize_actor_advantage: bool,
        contribute_to_global_stats: bool,
    ) -> dict[str, float]:
        """Compute independent segment GAE targets in this worker.

        ``contribute_to_global_stats`` is true for one worker per data replica. Replica-zero
        moments preserve the controller's previous global-normalization statistics while all
        replicas can consume their locally cached values for actor/critic updates.
        """
        host = self._require_host()
        samples, _pack_plan, _batch_attr = host.get_cached_batch()
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
            advantages, returns = compute_ppo_targets(
                values,
                torch.tensor([reward_score], dtype=torch.float32),
                action_mask,
                torch.tensor([0, values.numel()], dtype=torch.int32),
                gae_gamma=gae_gamma,
                gae_lambda=gae_lambda,
            )
            sample.advantages = advantages.reshape(1, -1)
            sample.returns = returns.reshape(1, -1)
            if contribute_to_global_stats:
                valid_advantages = advantages[action_mask]
                moments[0] += valid_advantages.double().sum()
                moments[1] += valid_advantages.double().square().sum()
                moments[2] += valid_advantages.numel()

        if normalize_actor_advantage:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(moments, op=torch.distributed.ReduceOp.SUM)
            moments = moments.cpu()
            count = float(moments[2].item())
            if count > 0:
                mean = moments[0] / moments[2]
                variance = torch.clamp(moments[1] / moments[2] - mean.square(), min=0.0)
                scale = (variance.sqrt() + 1e-8).float()
                mean_value = mean.float()
                for sample in samples:
                    if sample.advantages is not None:
                        mask = sample.shifted_labels.ne(-100)
                        sample.advantages = torch.where(
                            mask,
                            (sample.advantages.float() - mean_value) / scale,
                            torch.zeros_like(sample.advantages.float()),
                        )

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

    def _materialize(
        self,
        rollout_item_refs: list[ray.ObjectRef],
        pack_plan: DPRankPackIndices,
        batch_attr: TrainBatchAttr,
        token_returns: TokenTensorBatch | None = None,
    ) -> list[list[tuple[SequenceContext, torch.Tensor, torch.Tensor, torch.Tensor | None]]]:
        host = self._require_host()
        train_samples = host.cached_samples
        rollout_items: list[RolloutState] | None = None
        values_ready = train_samples is not None and all(item.values is not None for item in train_samples)
        if train_samples is None:
            if not rollout_item_refs:
                raise RuntimeError("critic materialize requires a loaded training batch")
            host.load_train_batch(rollout_item_refs, pack_plan, batch_attr)
            train_samples = host.cached_samples
            if train_samples is None:
                raise RuntimeError("critic materialize requires a loaded training batch")

        # Fetch rollout states only when cached samples are incomplete: missing returns must
        # be restored from extra fields, and missing old values are needed by the clipped loss.
        if (token_returns is None and any(item.returns is None for item in train_samples)) or not values_ready:
            if not rollout_item_refs:
                raise RuntimeError("critic materialize requires a loaded training batch")
            rollout_items = ray.get(rollout_item_refs[0])
        return_rows = None if token_returns is None else token_returns.split_samples()
        if return_rows is not None and len(return_rows) != len(train_samples):
            raise ValueError(f"token return rows {len(return_rows)} do not match samples {len(train_samples)}")
        samples: list[tuple[SequenceContext, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        for index, item in enumerate(train_samples):
            shifted_labels = item.shifted_labels
            shifted_len = shifted_labels.size(1)
            if return_rows is not None:
                item.returns = return_rows[index].unsqueeze(0)
                if item.returns.size(1) != shifted_len:
                    raise ValueError(
                        f"token returns length {item.returns.size(1)} does not match "
                        f"shifted labels length {shifted_len}"
                    )
            elif item.returns is None:
                if rollout_items is None:
                    raise RuntimeError("critic materialize is missing rollout states for extra-field returns")
                item.returns = self._token_tensor(rollout_items[index], "returns", shifted_len)
            if item.returns is None:
                raise ValueError("PPO critic fit requires per-token returns")
            if item.values is None and rollout_items is not None:
                item.values = self._token_tensor(rollout_items[index], "old_values", shifted_len)
            if item.values is not None and item.values.size(1) != shifted_len:
                raise ValueError(
                    f"cached value length {item.values.size(1)} does not match shifted labels length {shifted_len}"
                )
            samples.append((item.seq_ctx, shifted_labels, item.returns, item.values))
        return [
            [self._pack_one_batch([samples[index] for index in pack], batch_attr) for pack in step]
            for step in pack_plan
        ]

    def fit(self, token_returns: TokenTensorBatch | None = None, **kwargs) -> WorkerLogItem:
        """Train the clipped value head for ``num_passes`` over the loaded pack
        plan.

        Old values and returns are stored on the host ``TrainSample`` cache.
        """
        del kwargs
        host = self._require_host()
        _samples, pack_plan, batch_attr = host.get_cached_batch()
        step_batches = self._materialize(
            [],
            pack_plan,
            batch_attr,
            token_returns=token_returns,
        )
        updates = self._critic_updates(step_batches, batch_attr["rollout_idx"])
        worker_log_item: WorkerLogItem = {"train_entropy": 0.0, "train_metrics": []}

        for step in updates:
            if self._masked_token_count(step) == 0:
                continue
            prepared = [self._prepare_loss(*pack) for pack in step]
            batch_loss_ctx = cast(
                list[CriticLossContext],
                CriticLossContext.build_batches([loss_ctx for _, loss_ctx in prepared]),
            )
            engine_input = [
                ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
                for (seq_ctx, _), loss_ctx in zip(prepared, batch_loss_ctx)
            ]
            train_step_info = self._engine.train_step(engine_input)
            grad_norm = self._engine.clip_grad_norm()
            stepped = optimizer_step_succeeds(self._engine, grad_norm)
            self._engine.step_optimizer(grad_norm)
            if stepped:
                self._scheduler.step()
            logs_info = cast(dict[str, float], train_step_info["logs_info"])
            worker_log_item["train_metrics"].append(
                cast("WorkerTrainLogItem", {**logs_info, "grad_norm": grad_norm.item()})
            )
        return worker_log_item

    def _critic_updates(
        self,
        step_batches: list[list[tuple[SequenceContext, torch.Tensor, torch.Tensor, torch.Tensor | None]]],
        rollout_idx: int,
    ) -> list[list[tuple[SequenceContext, torch.Tensor, torch.Tensor, torch.Tensor | None]]]:
        """Repeat the local packs for each critic pass.

        Without ``optimizer_steps_per_pass``, one pass follows the actor pack plan.
        Otherwise every pass flattens those packs, shuffles after the first pass,
        and splits them into ``optimizer_steps_per_pass`` optimizer updates.
        """
        if self._optimizer_steps_per_pass is None:
            if len(step_batches) > self._optimizer_steps:
                raise ValueError(
                    f"Pack plan optimizer steps {len(step_batches)} exceed configured "
                    f"optimizer_steps {self._optimizer_steps}"
                )
            return step_batches

        packs = [pack for step in step_batches for pack in step]
        host = self._require_host()
        seed = int(host.config.seed or 0)
        updates: list[list[tuple[SequenceContext, torch.Tensor, torch.Tensor, torch.Tensor | None]]] = []
        for pass_index in range(self._num_passes):
            order = pass_order(len(packs), rollout_idx, pass_index, seed)
            for indices in partition_indices(order, self._optimizer_steps_per_pass):
                updates.append([packs[index] for index in indices])
        return updates

    def _masked_token_count(
        self,
        step: list[tuple[SequenceContext, torch.Tensor, torch.Tensor, torch.Tensor | None]],
    ) -> int:
        local_tokens = sum(int((labels != -100).sum().item()) for _, labels, _, _ in step)
        total = torch.tensor(float(local_tokens), device=DEVICE)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return int(total.item())

    def onload(self) -> None:
        self._engine.put_optimizer_to_device(DEVICE)
        self._engine.put_model_to_device(DEVICE)

    def offload(self) -> None:
        self._engine.put_optimizer_to_device("cpu")
        self._engine.put_model_to_device("cpu")

    def save(self, checkpoint_path: Path | str, no_save_optimizer: bool = False) -> None:
        """Save the critic engine under its role-specific checkpoint path."""
        checkpoint_path = Path(checkpoint_path)
        self._engine.save_dcp(
            weights_dir=checkpoint_path,
            save_optimizer=not no_save_optimizer,
        )

    def save_hf(self, hf_dir: Path | str, save_dtype: torch.dtype = torch.bfloat16) -> None:
        """Save the critic engine in Hugging Face format."""
        self._engine.save_hf(hf_dir=hf_dir, save_dtype=save_dtype)

    def resume(
        self,
        checkpoint_path: Path | str,
        *,
        load_optimizer_states: bool = True,
        load_optimizer_args: bool = True,
    ) -> None:
        """Restore the critic engine from its role-specific checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not (checkpoint_path / ".metadata").is_file():
            raise FileNotFoundError(f"Critic checkpoint at {checkpoint_path} has no DCP metadata.")
        self._engine.load_dcp(
            weights_dir=checkpoint_path,
            load_states=load_optimizer_states,
            load_args=load_optimizer_args,
        )

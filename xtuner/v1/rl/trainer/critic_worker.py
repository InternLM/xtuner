from pathlib import Path
from typing import TYPE_CHECKING, cast

import ray
import torch
from pydantic import BaseModel, ConfigDict

from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.config.optim import LRConfig, OptimConfig
from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.utils import sp_split
from xtuner.v1.model.base import ModelItem, TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.loss import CriticLossConfig, CriticLossContext
from xtuner.v1.utils import get_device, get_torch_device_module

from .pack import DPRankPackIndices, TokenTensorBatch
from .schedule import build_lr_scheduler, optimizer_step_succeeds, partition_indices, pass_order
from .worker import TrainBatchAttr, WorkerLogItem, WorkerTrainLogItem


if TYPE_CHECKING:
    from .worker import TrainingWorker
DEVICE_MODULE = get_torch_device_module()

DEVICE = get_device()
_SAVE_WEIGHTS_DIR = "weights"


class CriticWorkerConfig(BaseModel):
    """Configuration for an attached PPO critic worker.

    A critic is hosted by the actor Ray worker instead of being spawned as a second Ray actor. It shares the actor
    process' distributed process group, but owns its own value-head model and optimizer.
    """

    model_config = ConfigDict(title="Critic worker config", extra="forbid", arbitrary_types_allowed=True)
    model_cfg: TransformerConfig | BaseComposeConfig
    optim_cfg: OptimConfig
    loss_cfg: CriticLossConfig = CriticLossConfig()
    lr_cfg: LRConfig
    fsdp_cfg: FSDPConfig
    load_from: str | Path
    optimizer_steps: int = 1
    num_passes: int = 1
    optimizer_steps_per_pass: int | None = None
    scheduler_steps: int = 1_000_000
    sp_size: int = 1
    seed: int | None = None


class CriticWorker:
    """Attached value worker used by PPO.

    The class deliberately does not inherit ``SingleAcceleratorWorker``: its
    host ``TrainingWorker`` has already initialized the process group. A critic
    owns an independent value-head engine, while reusing the host's
    sequence-parallel mesh, pack plan, and process lifetime.
    """

    def __init__(self, worker_cfg: CriticWorkerConfig, host: "TrainingWorker | None" = None) -> None:
        self.config = worker_cfg
        self.host = host
        self.sp_mesh = host.sp_mesh if host is not None else None
        if worker_cfg.num_passes <= 0:
            raise ValueError("num_passes must be positive.")
        if worker_cfg.optimizer_steps_per_pass is not None and worker_cfg.optimizer_steps_per_pass <= 0:
            raise ValueError("optimizer_steps_per_pass must be positive.")
        model_cfg = worker_cfg.model_cfg
        head_cfg = model_cfg.text_config if isinstance(model_cfg, BaseComposeConfig) else model_cfg
        if getattr(head_cfg, "head_type", "lm_head") != "value_head":
            raise ValueError("CriticWorker requires model_cfg.head_type='value_head'")
        if not worker_cfg.fsdp_cfg.torch_compile:
            worker_cfg.model_cfg.compile_cfg = False
        self._engine = TrainEngine(
            optim_cfg=worker_cfg.optim_cfg,
            fsdp_cfg=worker_cfg.fsdp_cfg,
            model_cfg=worker_cfg.model_cfg,
        )
        self._engine.from_hf(worker_cfg.load_from)
        self._optimizer_steps = worker_cfg.optimizer_steps
        self._num_passes = worker_cfg.num_passes
        self._optimizer_steps_per_pass = worker_cfg.optimizer_steps_per_pass
        self._scheduler = build_lr_scheduler(self._engine.optimizer, worker_cfg.lr_cfg, worker_cfg.scheduler_steps)

    def _require_host(self) -> "TrainingWorker":
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

    def _pack_one(
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
            shifted_labels = torch.cat([sample[1] for sample in samples], dim=1)
            returns = torch.cat([sample[2] for sample in samples], dim=1)
            old_values_present = [sample[3] is not None for sample in samples]
            if any(old_values_present) and not all(old_values_present):
                raise ValueError("old_values must be all-present or all-absent within a pack")
            old_values = (
                torch.cat([cast(torch.Tensor, sample[3]) for sample in samples], dim=1)
                if all(old_values_present)
                else None
            )
            seq_ctx_list = [sample[0] for sample in samples]
        else:
            shifted_labels = host._pack_padding_loss_input("shifted_labels", padding_len)
            returns = torch.zeros(1, padding_len, dtype=torch.float32)
            # Padding slots are not on every rank. old_values must be a local tensor so
            # loss prep does not run a model forward that the other ranks skip.
            old_values = torch.zeros(1, padding_len, dtype=torch.float32)
            seq_ctx_list = []

        if padding_len > 0 and samples:
            shifted_labels = torch.cat(
                [shifted_labels, host._pack_padding_loss_input("shifted_labels", padding_len)], dim=1
            )
            returns = torch.cat([returns, torch.zeros(1, padding_len, dtype=torch.float32)], dim=1)
            if old_values is not None:
                old_values = torch.cat([old_values, torch.zeros(1, padding_len, dtype=torch.float32)], dim=1)
        if padding_len > 0:
            seq_ctx_list.append(
                host._pack_padding_seq_ctx(
                    padding_len,
                    use_3d_position_ids=batch_attr["use_3d_position_ids"],
                    has_routed_experts=batch_attr["has_routed_experts"],
                )
            )
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
        seq_ctx = host._prepare_forward_seq_ctx(seq_ctx, need_routed_experts=False)
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

    def collect_values(self) -> TokenTensorBatch:
        """Return packed values and keep the per-sample rows on this worker."""
        host = self._require_host()
        train_samples, pack_plan, batch_attr = host._get_cached_batch()
        samples: list[tuple[SequenceContext, torch.Tensor, torch.Tensor, None]] = []
        for item in train_samples:
            labels = item.shifted_labels
            samples.append((item.seq_ctx, labels, torch.zeros_like(labels, dtype=torch.float32), None))
        packed: list[list[tuple[list[int], SequenceContext, torch.Tensor]]] = []
        for step in pack_plan:
            packed.append([(pack, *self._pack_one([samples[i] for i in pack], batch_attr)[:2]) for pack in step])
        torch.distributed.barrier()
        values: list[torch.Tensor | None] = [None] * len(samples)
        for packed_step in packed:
            for indices, forward_ctx, _shifted_labels in packed_step:
                forward_ctx = host._prepare_forward_seq_ctx(
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
        return batch

    def _materialize(
        self,
        rollout_item_refs: list[ray.ObjectRef],
        pack_plan: DPRankPackIndices,
        batch_attr: TrainBatchAttr,
        token_returns: TokenTensorBatch | None = None,
    ) -> list[list[tuple[SequenceContext, torch.Tensor, torch.Tensor, torch.Tensor | None]]]:
        host = self._require_host()
        train_samples = host._cached_samples
        rollout_items: list[RolloutState] | None = None
        values_ready = train_samples is not None and all(item.values is not None for item in train_samples)
        if train_samples is None:
            if not rollout_item_refs:
                raise RuntimeError("critic materialize requires a loaded training batch")
            host.load_train_batch(rollout_item_refs, pack_plan, batch_attr)
            train_samples = host._cached_samples
            if train_samples is None:
                raise RuntimeError("critic materialize requires a loaded training batch")
        if token_returns is None or not values_ready:
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
            [self._pack_one([samples[index] for index in pack], batch_attr) for pack in step] for step in pack_plan
        ]

    def fit(self, token_returns: TokenTensorBatch | None = None, **kwargs) -> WorkerLogItem:
        """Train the clipped value head for ``num_passes`` over the loaded pack
        plan.

        Old values and returns are stored on the host ``TrainSample`` cache.
        """
        del kwargs
        host = self._require_host()
        _samples, pack_plan, batch_attr = host._get_cached_batch()
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
                cast(WorkerTrainLogItem, {**logs_info, "grad_norm": grad_norm.item()})
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
            weights_dir=checkpoint_path / _SAVE_WEIGHTS_DIR,
            save_optimizer=not no_save_optimizer,
        )

    def resume(
        self,
        checkpoint_path: Path | str,
        *,
        load_optimizer_states: bool = True,
        load_optimizer_args: bool = True,
    ) -> None:
        """Restore the critic engine from its role-specific checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        weights_path = checkpoint_path / _SAVE_WEIGHTS_DIR
        if not weights_path.exists():
            raise FileNotFoundError(f"Critic checkpoint at {checkpoint_path} has no '{_SAVE_WEIGHTS_DIR}/' directory.")
        self._engine.load_dcp(
            weights_dir=weights_path,
            load_states=load_optimizer_states,
            load_args=load_optimizer_args,
        )

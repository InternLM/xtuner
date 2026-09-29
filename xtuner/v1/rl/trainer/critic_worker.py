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

from .pack import DPRankPackIndices
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
        model_cfg = worker_cfg.model_cfg
        head_cfg = model_cfg.text_config if isinstance(model_cfg, BaseComposeConfig) else model_cfg
        if getattr(head_cfg, "head_type", "lm_head") != "value_head":
            raise ValueError("CriticWorker requires model_cfg.head_type='value_head'")
        self._engine = TrainEngine(
            optim_cfg=worker_cfg.optim_cfg,
            fsdp_cfg=worker_cfg.fsdp_cfg,
            model_cfg=worker_cfg.model_cfg,
        )
        self._engine.from_hf(worker_cfg.load_from)
        self._optimizer_steps = worker_cfg.optimizer_steps

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
        seq_ctx = host._prepare_forward_seq_ctx(seq_ctx)
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

    def _forward_values(self, seq_ctx: SequenceContext) -> torch.Tensor:
        with torch.no_grad():
            # Value collection is inference: loss_ctx=None returns the value-head
            # logits. TrainEngine.forward_only always wraps a LogProbContext and
            # would compute token logprobs instead.
            output = self._engine.model(seq_ctx=seq_ctx, loss_ctx=None)
        values = output.logits
        if values is None:
            raise RuntimeError("critic forward did not return value predictions")
        return values.float()

    def collect_values(
        self,
        rollout_item_refs: list[ray.ObjectRef],
        batch_attr: TrainBatchAttr,
        pack_plan: DPRankPackIndices,
    ) -> list[list[float]]:
        """Return one value vector per local sample, aligned with shifted
        labels."""
        host = self._require_host()
        rollout_items: list[RolloutState] = ray.get(rollout_item_refs[0])
        samples: list[tuple[SequenceContext, torch.Tensor, torch.Tensor, None]] = []
        for state in rollout_items:
            seq_ctx, loss_ctx = host._convert_one_rollout_state(state, 0.0, batch_attr["use_3d_position_ids"])
            labels = loss_ctx.loss_kwargs.shifted_labels
            samples.append((seq_ctx, labels, torch.zeros_like(labels, dtype=torch.float32), None))
        packed: list[list[tuple[list[int], SequenceContext, torch.Tensor]]] = []
        for step in pack_plan:
            packed.append([(pack, *self._pack_one([samples[i] for i in pack], batch_attr)[:2]) for pack in step])
        torch.distributed.barrier()
        self._engine.put_model_to_device(DEVICE)
        values: list[list[float] | None] = [None] * len(samples)
        try:
            for packed_step in packed:
                for indices, forward_ctx, _shifted_labels in packed_step:
                    forward_ctx = host._prepare_forward_seq_ctx(forward_ctx, sequence_parallel=False)
                    raw = self._forward_values(forward_ctx).detach().float().reshape(-1)
                    offset = 0
                    for index in indices:
                        length = samples[index][1].size(1)
                        values[index] = raw[offset : offset + length].cpu().tolist()
                        offset += length
            collected: list[list[float]] = []
            for item in values:
                if item is None:
                    raise RuntimeError("pack plan did not cover every local sample")
                collected.append(item)
            return collected
        finally:
            self._engine.put_model_to_device("cpu")

    def forward_only(
        self,
        rollout_item_refs: list[ray.ObjectRef],
        advantages: list[float],
        pack_plan: DPRankPackIndices,
        batch_attr: TrainBatchAttr,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Return old value predictions for the packed plan without a loss."""
        del advantages, kwargs
        packed = self._materialize(rollout_item_refs, pack_plan, batch_attr)
        self._engine.put_model_to_device(DEVICE)
        try:
            values: list[torch.Tensor] = []
            for step in packed:
                for seq_ctx, _, _, _ in step:
                    values.append(self._forward_values(self._require_host()._prepare_forward_seq_ctx(seq_ctx)))
            return values
        finally:
            self._engine.put_model_to_device("cpu")

    def _materialize(
        self,
        rollout_item_refs: list[ray.ObjectRef],
        pack_plan: DPRankPackIndices,
        batch_attr: TrainBatchAttr,
        token_returns: list[list[float]] | None = None,
        token_values: list[list[float]] | None = None,
    ) -> list[list[tuple[SequenceContext, torch.Tensor, torch.Tensor, torch.Tensor | None]]]:
        host = self._require_host()
        rollout_items: list[RolloutState] = ray.get(rollout_item_refs[0])
        samples: list[tuple[SequenceContext, torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        for index, state in enumerate(rollout_items):
            seq_ctx, loss_ctx = host._convert_one_rollout_state(state, 0.0, batch_attr["use_3d_position_ids"])
            shifted_labels = loss_ctx.loss_kwargs.shifted_labels
            shifted_len = shifted_labels.size(1)
            returns: torch.Tensor | None
            if token_returns is not None:
                returns = torch.tensor(token_returns[index], dtype=torch.float32).unsqueeze(0)
                if returns.size(1) != shifted_len:
                    raise ValueError(
                        f"token returns length {returns.size(1)} does not match shifted labels length {shifted_len}"
                    )
            else:
                returns = self._token_tensor(state, "returns", shifted_len)
            if returns is None:
                raise ValueError("PPO critic fit requires per-token returns")
            old_values: torch.Tensor | None
            if token_values is not None:
                old_values = torch.tensor(token_values[index], dtype=torch.float32).unsqueeze(0)
            else:
                old_values = self._token_tensor(state, "old_values", shifted_len)
            samples.append((seq_ctx, shifted_labels, returns, old_values))
        return [
            [self._pack_one([samples[index] for index in pack], batch_attr) for pack in step] for step in pack_plan
        ]

    def fit(
        self,
        rollout_item_refs: list[ray.ObjectRef],
        advantages: list[float],
        pack_plan: DPRankPackIndices,
        batch_attr: TrainBatchAttr,
        rollout_idx: int = 0,
        token_returns: list[list[float]] | None = None,
        token_values: list[list[float]] | None = None,
        **kwargs,
    ) -> WorkerLogItem:
        """Train the clipped value head on the controller's packed plan."""
        del advantages, kwargs, rollout_idx
        step_batches = self._materialize(
            rollout_item_refs,
            pack_plan,
            batch_attr,
            token_returns=token_returns,
            token_values=token_values,
        )
        num_optimizer_steps = len(step_batches)
        if num_optimizer_steps > self._optimizer_steps:
            raise ValueError(
                f"Pack plan optimizer steps {num_optimizer_steps} exceed configured "
                f"optimizer_steps {self._optimizer_steps}"
            )

        self._engine.put_model_to_device(DEVICE)
        self._engine.put_optimizer_to_device(DEVICE)
        worker_log_item: WorkerLogItem = {"train_entropy": 0.0, "train_metrics": []}
        try:
            for step in step_batches:
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
                self._engine.step_optimizer(grad_norm)
                logs_info = cast(dict[str, float], train_step_info["logs_info"])
                worker_log_item["train_metrics"].append(
                    cast(WorkerTrainLogItem, {**logs_info, "grad_norm": grad_norm.item()})
                )
        finally:
            self._engine.put_optimizer_to_device("cpu")
            self._engine.put_model_to_device("cpu")
        return worker_log_item

    def onload(self) -> None:
        self._engine.put_optimizer_to_device(DEVICE)
        self._engine.put_model_to_device(DEVICE)

        # model_moved = self._engine.put_model_to_device(DEVICE)
        # if not model_moved:
        #     self.logger.info("Skip model onload because model placement is unchanged.")

        # optimizer_moved = self._engine.put_optimizer_to_device(DEVICE)
        # if not optimizer_moved:
        #     if getattr(self.config.optim_cfg, "swap_optimizer", False):
        #         self.logger.info(
        #             "Skip optimizer onload because swap_optimizer=True; optimizer states stay on CPU and are swapped per step."
        #         )
        #     else:
        #         self.logger.info("Skip optimizer onload because optimizer state is empty.")

    def offload(self) -> None:
        # model_moved = self._engine.put_model_to_device("cpu")
        # DEVICE_MODULE.empty_cache()
        # self._clear_cublas_workspaces()
        # if not model_moved:
        #     self.logger.info("Skip model offload because model placement is unchanged.")
        #     return
        # self.logger.info(
        #     f"Offloaded model to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        # )
        # """Offload the optimizer of the training worker."""
        # optimizer_moved = self._engine.put_optimizer_to_device("cpu")
        # DEVICE_MODULE.empty_cache()
        # if not optimizer_moved:
        #     if getattr(self.config.optim_cfg, "swap_optimizer", False):
        #         self.logger.info(
        #             "Skip optimizer offload because swap_optimizer=True; optimizer states are already CPU-resident."
        #         )
        #     else:
        #         self.logger.info("Skip optimizer offload because optimizer state is empty.")
        #     return
        # self.logger.info(
        #     f"Offloaded optimizer to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, "
        #     f"reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        # )

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

"""The actor role: policy training, ref/teacher stages, MTP and SFT tail."""

from __future__ import annotations

import contextlib
import json
import os
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, cast

import torch
import torch.distributed as dist

from transformers import AutoTokenizer
from xtuner.v1.datasets.dataloader import Dataloader
from xtuner.v1.engine.train_engine import TrainEngine, TrainStepInfo
from xtuner.v1.loss import CELossConfig, LogProbConfig
from xtuner.v1.loss.ce_loss import CELossContext
from xtuner.v1.loss.mtp_loss import MTPLossConfig, MTPLossContext
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.profiler import profiling_memory, profiling_time
from xtuner.v1.rl.distillation import TrainTeacherManager, TrainTeacherTimings
from xtuner.v1.rl.loss import BaseRLLossConfig, BaseRLLossContext, kl_penalty
from xtuner.v1.rl.model_utils import build_frozen_model
from xtuner.v1.rl.weight_update import WeightUpdater
from xtuner.v1.utils import get_device, get_torch_device_module

from ..rollout_is import merge_rollout_is_metrics
from .role_worker import RoleContext, RoleWorker
from .schedule import build_lr_scheduler
from .train_batch import FullPack, TrainBatch, iter_cpu_packs, resolve_pack_refs


if TYPE_CHECKING:
    from xtuner.v1.data_proto.sequence_context import SequenceContext

    from .worker import WorkerConfig, WorkerLogItem


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

# SFT state files, written at the checkpoint root.
_SFT_DATALOADER_FILE = "sft_dataloader"
_SFT_TRAIN_STATE_FILE = "sft_train_state.json"


@dataclass(frozen=True)
class ActorSnapshot:
    """Frozen loss inputs of one pack, all tensors sequence-parallel local CPU.

    Args:
        shifted_labels (torch.Tensor): Labels after rollout importance sampling.
        old_logprobs (torch.Tensor): Frozen old-policy log probabilities.
        is_weights (torch.Tensor | None): Rollout importance-sampling weights.
        ref_logprobs (torch.Tensor | None): Reference log probabilities.
        teacher_logprobs (torch.Tensor | None): Train-teacher log probabilities.
        target_token_ids (torch.Tensor | None): Train-teacher top-k target ids.
    """

    shifted_labels: torch.Tensor
    old_logprobs: torch.Tensor
    is_weights: torch.Tensor | None
    ref_logprobs: torch.Tensor | None
    teacher_logprobs: torch.Tensor | None
    target_token_ids: torch.Tensor | None


class ActorWorker(RoleWorker):
    """The policy role: ref model, distillation, MTP, SFT tail and weight sync."""

    role_name = "actor"
    loss_cfg: BaseRLLossConfig

    def __init__(self, worker_cfg: WorkerConfig, ctx: RoleContext) -> None:
        super().__init__(worker_cfg, ctx)
        self.engine = self._build_engine(worker_cfg)
        self.lr_scheduler = None
        if worker_cfg.scheduler_steps is not None:
            self.lr_scheduler = build_lr_scheduler(
                self.engine.optimizer, worker_cfg.lr_cfg, worker_cfg.scheduler_steps
            )

        self._has_ref = False
        if worker_cfg.loss_cfg.use_kl_loss:
            self._has_ref = True
            if worker_cfg.ref_load_from is None:
                worker_cfg.ref_load_from = worker_cfg.load_from
            self._ref_model = build_frozen_model(
                worker_cfg.model_cfg, worker_cfg.ref_load_from, worker_cfg.ref_model_fsdp_cfg
            )

        self._train_teacher_manager: TrainTeacherManager | None = None
        if worker_cfg.train_teacher_manager_config is not None:
            self._train_teacher_manager = worker_cfg.train_teacher_manager_config.build(
                chunk_size=worker_cfg.loss_cfg.chunk_size,
            )

        profile_step = worker_cfg.profile_step
        if isinstance(profile_step, int):
            profile_step = [profile_step]
        self._profile_step = set(profile_step or [])
        self._profile_time = worker_cfg.profile_time
        self._profile_memory = worker_cfg.profile_memory

        if worker_cfg.loss_cfg.chunk_size is not None:
            mode = "chunk"
        else:
            mode = "eager"
        self.logprob_cfg = LogProbConfig(chunk_size=worker_cfg.loss_cfg.chunk_size, mode=mode)

        self.mtp_config = None
        if isinstance(worker_cfg.model_cfg, BaseComposeConfig):
            if hasattr(worker_cfg.model_cfg.text_config, "mtp_config"):
                self.mtp_config = worker_cfg.model_cfg.text_config.mtp_config
        elif hasattr(worker_cfg.model_cfg, "mtp_config"):
            self.mtp_config = worker_cfg.model_cfg.mtp_config
        self.mtp_cfgs: list[MTPLossConfig] = []
        if self.mtp_config is not None:
            for depth in range(self.mtp_config.num_layers):
                self.mtp_cfgs.append(
                    MTPLossConfig(
                        **worker_cfg.loss_cfg.model_dump(include={"mode", "chunk_size"}),
                        mtp_depth=depth + 1,
                        detach_mtp_lm_head_weight=self.mtp_config.detach_mtp_lm_head_weight,
                    )
                )
        self._mtp_trainable_parameters = (
            [parameter for name, parameter in self.engine.model.trainable_parameters() if "mtp_block" in name]
            if self.mtp_config is not None
            else []
        )

        self.update_weighter = WeightUpdater(
            rank=self.rank,
            logger=self.logger,
            config=self.config,
            engine=self.engine,
        )

        self._init_sft(worker_cfg)
        self.loss_cfg: BaseRLLossConfig | None = worker_cfg.loss_cfg

    def teacher_loss_keys(self) -> list[str]:
        """Return loss fields required from configured teachers.

        Returns:
            list[str]: Extra per-pack loss keys.
        """
        distillation_config = self.config.distillation_config
        if distillation_config is None:
            return []
        keys: list[str] = []
        if distillation_config.rollout_teachers:
            keys.append("teacher_logprobs")
            if not distillation_config.loss_config.uses_sampled_token_targets:
                keys.append("target_token_ids")
        if distillation_config.train_teachers:
            keys.append("teacher_indices")
        return keys

    def fit(self, batch: TrainBatch) -> WorkerLogItem:
        """Run old/ref/teacher stages and policy updates on ``batch``.

        Args:
            batch (TrainBatch): The installed local batch with actor targets.

        Returns:
            WorkerLogItem: Actor training metrics.
        """
        batch.require_targets("actor")
        worker_log_item: WorkerLogItem = {
            "train_entropy": 0.0,
            "train_metrics": [],
            "sft_train_metrics": {},
        }
        snapshots_by_pack = self._collect_old_policy_snapshots(batch, worker_log_item)
        try:
            self._stage_ref_logprobs(batch, snapshots_by_pack)
            teacher_timings = self._stage_teacher_outputs(batch, snapshots_by_pack)
            if teacher_timings is not None:
                worker_log_item["teacher_timings"] = teacher_timings.to_dict()
            return self._fit_policy_then_sft(batch, snapshots_by_pack, worker_log_item)
        finally:
            # Controller closes the local batch; RLTrainer releases remote refs.
            snapshots_by_pack.clear()

    def _build_engine(self, worker_cfg: WorkerConfig) -> TrainEngine:
        engine = TrainEngine(  # type: ignore[call-arg]
            optim_cfg=worker_cfg.optim_cfg,
            fsdp_cfg=worker_cfg.fsdp_cfg,
            model_cfg=worker_cfg.model_cfg,
        )
        if worker_cfg.load_from is not None:
            engine.from_hf(worker_cfg.load_from)
        if engine.model.compile_cfg is not None and self.rank == 0:
            self.logger.info(f"The `compile_cfg` of model is {json.dumps(engine.model.compile_cfg, indent=4)}")
        return engine

    def _init_sft(self, worker_cfg: WorkerConfig) -> None:
        self._sft_dataloader: Dataloader | None = None
        self._sft_dataloader_iter: Iterator | None = None
        self._sft_loss_cfg: CELossConfig | None = None
        self._rollout_steps_per_sft = worker_cfg.rollout_steps_per_sft

        self._rollout_step = 0
        self._sft_cur_epoch = 0
        self._sft_total_consumed_tokens = 0

        if worker_cfg.sft_dataloader_cfg is not None:
            assert worker_cfg.sft_global_batch_size > 0, "sft_global_batch_size must be greater than 0"
            assert worker_cfg.seed is not None, "seed must be set when sft_dataloader_config is not None"
            tokenizer = AutoTokenizer.from_pretrained(worker_cfg.load_from, trust_remote_code=True)
            self._sft_dataloader = worker_cfg.sft_dataloader_cfg.build(
                tokenizer=tokenizer,
                dp_mesh=self.data_mesh["dp"],
                global_batch_size=worker_cfg.sft_global_batch_size,
                micro_batch_size=1,
                seed=worker_cfg.seed,
            )
            self.logger.info(f"Sft Dataloader len: {len(self._sft_dataloader)}")
            self._sft_loss_cfg = worker_cfg.sft_loss_cfg or CELossConfig()

    def _maybe_profiling(self, global_train_step: int) -> AbstractContextManager[Any]:
        if global_train_step not in self._profile_step:
            return contextlib.nullcontext()

        if self.log_dir is not None:
            profile_home = self.log_dir.parent
        else:
            profile_home = Path(os.environ.get("WORK_DIR", "."))

        with contextlib.ExitStack() as stack:
            if self._profile_time:
                time_dir = profile_home / "profiling_time" / "train_step" / f"global-step-{global_train_step}"
                stack.enter_context(profiling_time(time_dir))
            if self._profile_memory:
                memory_dir = profile_home / "profiling_memory" / "train_step" / f"global-step-{global_train_step}"
                stack.enter_context(profiling_memory(memory_dir))
            return stack.pop_all()

    def _extra_step_metrics(self) -> dict[str, float]:
        if not self._mtp_trainable_parameters:
            return {}
        # `grad is None` is a structural signal: unlike a zero-valued grad,
        # it means the MTP loss never reached that trainable parameter.
        present_grad_count = sum(parameter.grad is not None for parameter in self._mtp_trainable_parameters)
        return {"mtp_grad_parameter_coverage": present_grad_count / len(self._mtp_trainable_parameters)}

    def _resolve_refs(self, seq_ctx: SequenceContext) -> None:
        resolve_pack_refs(
            seq_ctx,
            model_cfg=self.config.model_cfg,
            pack_max_length=self.config.pack_max_length,
        )

    def _forward_ref_logprobs(self, seq_ctx: SequenceContext, shifted_labels: torch.Tensor) -> torch.Tensor:
        assert self._has_ref
        with torch.no_grad():
            loss_ctx = self.logprob_cfg.build(data={"shifted_labels": shifted_labels}, device=self.device)
            assert loss_ctx is not None
            ref_output = self._ref_model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
        return ref_output["loss"]

    def _collect_old_policy_snapshots(
        self, batch: TrainBatch, worker_log_item: WorkerLogItem
    ) -> dict[tuple[int, ...], ActorSnapshot]:
        """Recompute old logprobs and freeze one snapshot per pack."""
        self.engine._maybe_precompute_float8_dynamic_scale_for_fsdp()
        entropy_sum: torch.Tensor | None = None
        rollout_entropy_sum: torch.Tensor | None = None
        grad_tokens: torch.Tensor | None = None
        all_rollout_is_metrics: list[dict[str, float]] = []
        all_mismatch_metrics: list[dict[str, float]] = []
        snapshots: dict[tuple[int, ...], ActorSnapshot] = {}
        pack_indices = [pack for step in batch.plan.steps for pack in step]
        pack_time_begin = time.perf_counter()

        for full_pack in iter_cpu_packs(batch, pack_indices, self._resolve_refs):
            data = full_pack.full_data
            seq_ctx = self._to_device_seq(data["seq_ctx"])
            ctx = self.loss_cfg.build(data, sp_mesh=self.sp_mesh, device=self.device)
            assert ctx is not None
            read_ctx = self.logprob_cfg.build(data, sp_mesh=self.sp_mesh, device=self.device)
            assert read_ctx is not None
            old_logprobs = self.engine.forward_only(seq_ctx=seq_ctx, loss_ctx=read_ctx)["loss"]
            ctx.loss_kwargs.old_logprobs = old_logprobs

            labels = ctx.loss_kwargs.shifted_labels
            mask = labels != self.config.loss_cfg.ignore_idx
            entropy = -(old_logprobs * mask).sum()
            entropy_sum = entropy if entropy_sum is None else entropy_sum + entropy
            grad_tokens = mask.sum() if grad_tokens is None else grad_tokens + mask.sum()
            if ctx.loss_kwargs.rollout_logprobs is not None:
                rollout_entropy = -(ctx.loss_kwargs.rollout_logprobs * mask).sum()
                rollout_entropy_sum = (
                    rollout_entropy if rollout_entropy_sum is None else rollout_entropy_sum + rollout_entropy
                )
                mismatch_metrics, rollout_is_metrics = ctx.compute_rollout_is(self.sp_mesh, seq_ctx.seq_lens_q)
                all_rollout_is_metrics.append(rollout_is_metrics)
                all_mismatch_metrics.append(mismatch_metrics)

            snapshots[full_pack.indices] = self._capture_snapshot(ctx)

        worker_log_item["packing_pack_time_s"] = time.perf_counter() - pack_time_begin
        worker_log_item["packing_conversion_time_s"] = 0.0

        entropy_sum = cast(torch.Tensor, entropy_sum)
        grad_tokens = cast(torch.Tensor, grad_tokens)
        dist.all_reduce(entropy_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(grad_tokens, op=dist.ReduceOp.SUM)
        avg_sum_entropy = entropy_sum / grad_tokens if grad_tokens > 0 else torch.tensor(0.0)
        worker_log_item["train_entropy"] = avg_sum_entropy.item()

        logger_msg = f"Rollout {batch.attrs['rollout_idx']}: avg entropy: {avg_sum_entropy:.4f}"
        if rollout_entropy_sum is not None:
            dist.all_reduce(rollout_entropy_sum, op=dist.ReduceOp.SUM)
            avg_rollout_entropy = rollout_entropy_sum / grad_tokens if grad_tokens > 0 else torch.tensor(0.0)
            worker_log_item["rollout_entropy"] = avg_rollout_entropy.item()
            logger_msg += f", avg rollout entropy: {avg_rollout_entropy:.4f}"

        if all_mismatch_metrics:
            mismatch_metrics = merge_rollout_is_metrics(all_mismatch_metrics, self.device)
            if mismatch_metrics:
                worker_log_item["mismatch_metrics"] = mismatch_metrics
                logger_msg += f"\n rollout mismatch metrics:\n{json.dumps(mismatch_metrics, indent=4)}"
        if all_rollout_is_metrics:
            rollout_is_metrics = merge_rollout_is_metrics(all_rollout_is_metrics, self.device)
            if rollout_is_metrics:
                worker_log_item["rollout_is_metrics"] = rollout_is_metrics
                logger_msg += f"\n rollout importance sampling metrics:\n{json.dumps(rollout_is_metrics, indent=4)}"
        if self.rank == 0:
            self.logger.info(logger_msg)

        self._global_grad_tokens = grad_tokens
        return snapshots

    def _stage_ref_logprobs(
        self,
        batch: TrainBatch,
        snapshots_by_pack: dict[tuple[int, ...], ActorSnapshot],
    ) -> None:
        """Stage reference logprobs and KL metrics for every pack."""
        if not self._has_ref:
            return
        kl_div_sum: torch.Tensor | None = None
        self._ref_model.to_device(self.device)
        pack_indices = [pack for step in batch.plan.steps for pack in step]
        try:
            for full_pack in iter_cpu_packs(batch, pack_indices, self._resolve_refs):
                data = full_pack.full_data
                seq_ctx = self._to_device_seq(data["seq_ctx"])
                ctx = self.loss_cfg.build(data, sp_mesh=self.sp_mesh, device=self.device)
                assert ctx is not None
                snapshot = snapshots_by_pack[full_pack.indices]
                ctx.bind_actor_snapshot(snapshot)
                ref_logprobs = self._forward_ref_logprobs(seq_ctx, ctx.loss_kwargs.shifted_labels)
                mask = ctx.loss_kwargs.shifted_labels != self.config.loss_cfg.ignore_idx
                kl_div = kl_penalty(
                    snapshot.old_logprobs.to(self.device),
                    ref_logprobs,
                    loss_weights=mask,
                    kl_penalty="low_var_kl",
                )
                kl_div_sum = kl_div if kl_div_sum is None else kl_div_sum + kl_div
                snapshots_by_pack[full_pack.indices] = self._capture_snapshot(
                    ctx, overrides={"ref_logprobs": ref_logprobs}
                )
        finally:
            self._ref_model.to_device("cpu")

        kl_div_sum = cast(torch.Tensor, kl_div_sum)
        dist.all_reduce(kl_div_sum, op=dist.ReduceOp.SUM)
        grad_tokens = self._global_grad_tokens
        avg_kl_div = kl_div_sum / grad_tokens if grad_tokens > 0 else 0
        self.logger.info(f"avg KL divergence: {avg_kl_div:.4f}")

    def _stage_teacher_outputs(
        self,
        batch: TrainBatch,
        snapshots_by_pack: dict[tuple[int, ...], ActorSnapshot],
    ) -> TrainTeacherTimings | None:
        """Run the whole-batch teacher pass while the actor engine steps
        aside."""
        if self._train_teacher_manager is None:
            return None
        self.offload_optimizer()
        self.offload_model()
        DEVICE_MODULE.empty_cache()

        pack_indices = [pack for step in batch.plan.steps for pack in step]
        seq_ctx_list: list[SequenceContext] = []
        loss_ctx_list: list[BaseRLLossContext] = []
        try:
            for full_pack in iter_cpu_packs(batch, pack_indices, self._resolve_refs):
                data = full_pack.full_data
                seq_ctx = self._to_device_seq(data["seq_ctx"])
                ctx = self.loss_cfg.build(data, sp_mesh=self.sp_mesh, device=self.device)
                assert ctx is not None
                ctx.bind_actor_snapshot(snapshots_by_pack[full_pack.indices])
                seq_ctx_list.append(seq_ctx)
                loss_ctx_list.append(ctx)

            # Keep the teacher's whole-batch interface: no per-pack on/offload.
            timings = self._train_teacher_manager.compute_teacher_outputs(seq_ctx_list, loss_ctx_list)
            for pack, ctx in zip(pack_indices, loss_ctx_list):
                snapshots_by_pack[pack] = self._capture_snapshot(ctx)
        finally:
            # Drop temporary device contexts before restoring the actor engine.
            seq_ctx_list.clear()
            loss_ctx_list.clear()
            del seq_ctx, ctx

        # Restore only on success; failures propagate and terminate the round.
        self.onload_model()
        self.onload_optimizer()
        return timings

    def _capture_snapshot(
        self,
        ctx: BaseRLLossContext,
        *,
        overrides: dict[str, torch.Tensor] | None = None,
    ) -> ActorSnapshot:
        """Freeze the current context's loss inputs into one CPU record."""
        kwargs = ctx.loss_kwargs
        overrides = overrides or {}
        values: dict[str, torch.Tensor | None] = {}
        for name in ActorSnapshot.__dataclass_fields__:
            if name in ("teacher_logprobs", "target_token_ids") and self._train_teacher_manager is None:
                values[name] = None
                continue
            tensor = overrides.get(name, getattr(kwargs, name, None))
            values[name] = None if tensor is None else tensor.detach().cpu().clone()
        return ActorSnapshot(**cast(dict[str, Any], values))

    def _fit_policy_then_sft(
        self,
        batch: TrainBatch,
        snapshots_by_pack: dict[tuple[int, ...], ActorSnapshot],
        worker_log_item: WorkerLogItem,
    ) -> WorkerLogItem:
        """Run every policy update, then the SFT tail at its configured
        cadence."""
        for step_indices in batch.plan.steps:
            packs = list(iter_cpu_packs(batch, step_indices, self._resolve_refs))
            items = self._build_step_input(packs, snapshots_by_pack)
            worker_log_item["train_metrics"].append(self._train_update(items))
            del items

        self._rollout_step += 1
        if self._sft_dataloader is not None and self._rollout_step % self._rollout_steps_per_sft == 0:
            train_step_info = self._fit_sft()
            worker_log_item["sft_train_metrics"] = {
                **train_step_info["logs_info"],
                **train_step_info["extra_info"].get(),
                "efficient_attn_ratio": train_step_info["efficient_attn_ratio"],
            }
        return worker_log_item

    def _build_step_input(
        self,
        full_packs: list[FullPack],
        snapshots_by_pack: dict[tuple[int, ...], ActorSnapshot],
    ) -> list[ModelItem]:
        """Build and calibrate one optimizer step's model inputs."""
        items: list[ModelItem] = []
        depth_ctx_lists: list[list[MTPLossContext]] = []
        for pack in full_packs:
            data = pack.full_data
            seq_ctx = self._to_device_seq(data["seq_ctx"])
            main_ctx = self.loss_cfg.build(data, sp_mesh=self.sp_mesh, device=self.device)
            assert main_ctx is not None
            mtp_ctxs = [
                cfg.build(
                    data={**data, "logprobs": data.get("rollout_logprobs")},
                    sp_mesh=self.sp_mesh,
                    device=self.device,
                )
                for cfg in self.mtp_cfgs
            ]
            main_ctx.bind_actor_snapshot(snapshots_by_pack[pack.indices])
            loss_ctx: dict[str, Any] = {"lm": main_ctx}
            if self.mtp_cfgs:
                loss_ctx["mtp"] = mtp_ctxs
                depth_ctx_lists.append(cast(list[MTPLossContext], mtp_ctxs))
            items.append(ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx))

        self.loss_cfg.loss_ctx_cls.build_batches(
            cast(list[CELossContext], [item["loss_ctx"]["lm"] for item in items])  # type: ignore[index]
        )
        if self.mtp_cfgs:
            cu_seq_lens_list = [item["seq_ctx"].cu_seq_lens_q for item in items]
            for depth, cfg in enumerate(self.mtp_cfgs):
                cfg.loss_ctx_cls.build_batches(  # type: ignore[arg-type]
                    [pack_ctxs[depth] for pack_ctxs in depth_ctx_lists],
                    cu_seq_lens_list=cu_seq_lens_list,
                    sp_mesh=self.sp_mesh,
                )
        return items

    def _fit_sft(self) -> TrainStepInfo:
        self.logger.info(f"Train SFT after {self._rollout_step} RL steps")
        assert self._sft_dataloader is not None, "SFT tail requires a configured dataloader"
        if self._sft_dataloader_iter is None:
            self._sft_dataloader_iter = iter(self._sft_dataloader)

        time_before_get_data = time.time()
        data_batch = self._next_sft_data_batch()
        time_before_train_step = time.time()
        data_time = time_before_train_step - time_before_get_data
        DEVICE_MODULE.reset_peak_memory_stats()

        train_step_info, grad_norm = self._train_one_step_sft(data_batch)

        time_after_train_step = time.time()
        step_time = time_after_train_step - time_before_train_step
        step_consumed_tokens = train_step_info["step_consumed_tokens"]

        reduced_step_consumed_tokens = self._reduce_number_across_rank(step_consumed_tokens)
        self._sft_total_consumed_tokens += reduced_step_consumed_tokens

        self._sft_log_step(
            train_step_info=train_step_info,
            local_step_consumed_tokens=step_consumed_tokens,
            step_consumed_tokens=reduced_step_consumed_tokens,
            total_consumed_tokens=self._sft_total_consumed_tokens,
            data_time=data_time,
            step_time=step_time,
            grad_norm=grad_norm,
        )

        return train_step_info

    def _next_sft_data_batch(self) -> list[dict[str, Any]]:
        assert self._sft_dataloader is not None and self._sft_dataloader_iter is not None
        try:
            data = next(self._sft_dataloader_iter)
        except StopIteration:
            self._sft_cur_epoch += 1
            self._sft_dataloader.set_epoch(self._sft_cur_epoch)
            self._sft_dataloader_iter = iter(self._sft_dataloader)
            data = next(self._sft_dataloader_iter)
        return data

    def _train_one_step_sft(self, data_batch: list[dict[str, Any]]) -> tuple[TrainStepInfo, torch.Tensor]:
        seq_ctx_list: list[SequenceContext] = []
        loss_cfg: CELossConfig = self._sft_loss_cfg  # type: ignore[assignment]
        loss_ctx_list: list[CELossContext] = []
        for data in data_batch:
            seq_ctx = data["seq_ctx"].to(self.device)
            if self.sp_mesh.size() > 1:
                seq_ctx = seq_ctx.split(sequence_parallel_mesh=self.sp_mesh)
            seq_ctx_list.append(seq_ctx)
            loss_ctx = loss_cfg.build(data=data, sp_mesh=self.sp_mesh, device=self.device)
            assert loss_ctx is not None
            loss_ctx_list.append(loss_ctx)

        del data_batch

        cu_seq_lens_list = [seq_ctx.cu_seq_lens_q for seq_ctx in seq_ctx_list]
        loss_ctx_list = CELossContext.build_batches(
            loss_ctx_list, cu_seq_lens_list=cu_seq_lens_list, sp_mesh=self.sp_mesh
        )

        engine_input = [
            ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
            for seq_ctx, loss_ctx in zip(seq_ctx_list, loss_ctx_list)
        ]

        train_step_info = self.engine.train_step(engine_input)
        grad_norm = self.engine.clip_grad_norm()
        self.engine.step_optimizer(grad_norm)
        return train_step_info, grad_norm

    def _sft_log_step(
        self,
        train_step_info: TrainStepInfo,
        local_step_consumed_tokens: int,
        step_consumed_tokens: int,
        total_consumed_tokens: int,
        data_time: float,
        step_time: float,
        grad_norm: torch.Tensor,
    ) -> None:
        tgs = local_step_consumed_tokens / step_time
        logs_info = train_step_info.get("logs_info", {})
        log_items = [f"{k}: {v:.8f}" for k, v in logs_info.items() if "loss" in k]
        log_items.append(f"total_loss: {train_step_info['total_loss']:.8f}")
        loss_log_str = ", ".join(log_items)

        max_memory = DEVICE_MODULE.max_memory_allocated()
        reserved_memory = DEVICE_MODULE.max_memory_reserved()

        self.logger.info(
            f"Rank{self.rank} Step {self._rollout_step}: data_time: {data_time:.4f} time: {step_time:.4f} "
            f"text_tokens: {local_step_consumed_tokens} "
            f"step_consumed_tokens: {step_consumed_tokens} "
            f"total_consumed_tokens: {total_consumed_tokens} "
            f"efficient_attn_ratio: {train_step_info['efficient_attn_ratio']:.4f} "
            f"{loss_log_str} "
            f"grad_norm: {grad_norm:.8f} "
            f"max_memory: {max_memory / (1024**3):.2f} GB "
            f"reserved_memory: {reserved_memory / (1024**3):.2f} GB "
            f"tgs: {tgs:.4f}"
        )

    def _reduce_number_across_rank(self, rank_number: int) -> int:
        _gathered_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(_gathered_list, rank_number)
        reduced_number = sum(_gathered_list)  # type: ignore[arg-type]
        return reduced_number

    def save_sft_state(self, checkpoint_path: Path) -> None:
        """Persist the SFT dataloader and train state under
        ``checkpoint_path``."""
        sft_dataloader = self._sft_dataloader
        if sft_dataloader is None:
            return
        sft_dataloader_path = checkpoint_path / _SFT_DATALOADER_FILE
        dataloader_state = sft_dataloader.get_state_dict()
        total_consumed_samples = dataloader_state["total_consumed_samples"]
        if self.rank != 0:
            return

        torch.save(dataloader_state, sft_dataloader_path)

        train_state_path = checkpoint_path / _SFT_TRAIN_STATE_FILE
        with train_state_path.open("w") as f:
            f.write(
                json.dumps(
                    {
                        "cur_step": self._rollout_step,
                        "cur_epoch": self._sft_cur_epoch,
                        "total_consumed_samples": total_consumed_samples,
                        "total_consumed_tokens": self._sft_total_consumed_tokens,
                    }
                )
            )

    def resume_sft_state(self, resume_from: Path) -> None:
        """Restore the SFT dataloader and train state from ``resume_from``."""
        sft_dataloader = self._sft_dataloader
        if sft_dataloader is None:
            return
        train_state_path = resume_from / _SFT_TRAIN_STATE_FILE
        if not train_state_path.exists():
            raise FileNotFoundError(f"Train state path {train_state_path} does not exist.")
        with train_state_path.open("r") as f:
            train_state = json.loads(f.read())
        self._rollout_step = train_state["cur_step"]
        self._sft_cur_epoch = train_state["cur_epoch"]
        self._sft_total_consumed_tokens = train_state["total_consumed_tokens"]
        self.logger.info(f"Resume sft train state from {train_state_path}")

        sft_dataloader_path = resume_from / _SFT_DATALOADER_FILE
        if not sft_dataloader_path.exists():
            raise FileNotFoundError(f"Dataloader path {sft_dataloader_path} does not exist.")
        dataloader_state = torch.load(sft_dataloader_path, map_location=self.device)
        sft_dataloader.load_state_dict(dataloader_state)
        self.logger.info(f"Resume sft dataloader from {sft_dataloader_path}")

"""The actor role worker: policy training inside a ``TrainingWorker`` host."""

from __future__ import annotations

import contextlib
import json
import os
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, cast

import torch
import torch.distributed as dist

from transformers import AutoTokenizer
from xtuner.v1.datasets.dataloader import Dataloader
from xtuner.v1.engine.train_engine import TrainEngine, TrainStepInfo
from xtuner.v1.loss import CELossConfig, LogProbConfig
from xtuner.v1.loss.ce_loss import CELossContext, LMHeadLossContext
from xtuner.v1.loss.mtp_loss import MTPLossConfig, MTPLossContext
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.profiler import profiling_memory, profiling_time
from xtuner.v1.rl.distillation import (
    DistillationTrainerAdapter,
    TrainTeacherManager,
    TrainTeacherTimings,
)
from xtuner.v1.rl.loss import BaseRLLossConfig, BaseRLLossContext, kl_penalty
from xtuner.v1.rl.model_utils import build_frozen_model
from xtuner.v1.rl.weight_update import WeightUpdater
from xtuner.v1.utils import get_device, get_torch_device_module

from ..rollout_is import merge_rollout_is_metrics
from .pack import DPRankPackIndices
from .role_worker import (
    RoleWorker,
    _PackItem,
    free_rollout_routed_expert_refs,
    resolve_mm_inputs,
    resolve_routed_experts,
)
from .schedule import build_lr_scheduler


if TYPE_CHECKING:
    from collections.abc import Sequence

    from xtuner.v1.data_proto.rl_data import RolloutState
    from xtuner.v1.data_proto.sequence_context import SequenceContext
    from xtuner.v1.loss import BaseLossContext

    from .worker import TrainBatchAttr, TrainingWorker, TrainSample, WorkerConfig, WorkerLogItem

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

# SFT state files, written at the checkpoint root for backward compatibility.
_SFT_DATALOADER_FILE = "sft_dataloader"
_SFT_TRAIN_STATE_FILE = "sft_train_state.json"


class ActorWorker(RoleWorker):
    """The policy role: ref model, distillation, MTP, SFT tail, and rollout
    weight sync all live here. The engine is the actor's own; the process
    group used for weight sync comes from the host."""

    role_name = "actor"

    def __init__(self, worker_cfg: WorkerConfig, host: TrainingWorker) -> None:
        super().__init__(host)
        self.config = worker_cfg
        self.logger = host.logger
        self.rank = host.rank
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

        self._distillation = DistillationTrainerAdapter(worker_cfg.distillation_config)
        self._optimizer_steps = worker_cfg.optimizer_steps

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
            # 非 compose 模型的 mtp_config 直接放在了 model_cfg 中
            self.mtp_config = worker_cfg.model_cfg.mtp_config
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

        # Per-pack stage-A products (old/ref logprobs, teacher outputs), keyed
        # by the pack's sample-index tuple. Rows are SP-split shaped and CPU
        # resident between the two stages.
        self._staged: dict[tuple[int, ...], dict[str, torch.Tensor | None]] = {}

        self.loss_cfg: BaseRLLossConfig | None = worker_cfg.loss_cfg

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
                dp_mesh=self.host.data_mesh["dp"],
                global_batch_size=worker_cfg.sft_global_batch_size,
                micro_batch_size=1,
                seed=worker_cfg.seed,
            )
            self.logger.info(f"Sft Dataloader len: {len(self._sft_dataloader)}")
            self._sft_loss_cfg = worker_cfg.sft_loss_cfg or CELossConfig()

    def _maybe_profiling(self, global_train_step: int) -> AbstractContextManager[Any]:
        if global_train_step not in self._profile_step:
            return contextlib.nullcontext()

        if self.host.log_dir is not None:
            profile_home = self.host.log_dir.parent
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
        resolve_mm_inputs(seq_ctx)
        resolve_routed_experts(
            seq_ctx,
            model_cfg=self.config.model_cfg,
            pack_max_length=self.config.pack_max_length,
        )

    def _forward_logprobs(self, seq_ctx: SequenceContext, shifted_labels: torch.Tensor) -> torch.Tensor:
        loss_ctx = self.logprob_cfg.build(data={"shifted_labels": shifted_labels})
        assert loss_ctx is not None
        output = self.engine.forward_only(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
        return output["loss"]

    def _forward_ref_logprobs(self, seq_ctx: SequenceContext, shifted_labels: torch.Tensor) -> torch.Tensor:
        assert self._has_ref
        with torch.no_grad():
            loss_ctx = self.logprob_cfg.build(data={"shifted_labels": shifted_labels})
            assert loss_ctx is not None
            ref_output = self._ref_model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
        return ref_output["loss"]

    def _configured_teacher_loss_keys(self) -> list[str]:
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

    def _pack_loss_fields(
        self, samples: list[TrainSample], pack: list[int], batch_attr: TrainBatchAttr
    ) -> dict[str, torch.Tensor | None]:
        """Concatenate one pack's loss inputs on CPU, tail-padded."""
        template_keys = list(batch_attr["pack_loss_keys"])
        for key in self._configured_teacher_loss_keys():
            if key not in template_keys:
                template_keys.append(key)

        if pack:
            rollout_rows = [samples[index].rollout_logprobs for index in pack]
            rows: dict[str, list[torch.Tensor] | None] = {
                "shifted_labels": [samples[index].shifted_labels for index in pack],
                "advantages": [cast(torch.Tensor, samples[index].advantages) for index in pack],
                "rollout_logprobs": (
                    [cast(torch.Tensor, row) for row in rollout_rows]
                    if all(row is not None for row in rollout_rows)
                    else None
                ),
            }

            for key in self._configured_teacher_loss_keys():
                if key not in rows:
                    values = [samples[index].teacher_targets.get(key) for index in pack]
                    present = [value is not None for value in values]
                    if not all(present):
                        raise ValueError(f"Required pack field {key} must be present on every sample: {present}")
                    rows[key] = [cast(torch.Tensor, value) for value in values]
        else:
            # Scheduling-only padding slot: empty rows mark the fields present.
            rows = {
                "shifted_labels": [],
                "advantages": [],
                "rollout_logprobs": [] if "rollout_logprobs" in template_keys else None,
            }
            for key in template_keys:
                rows[key] = []
        return self._pack_loss_rows(samples, pack, rows)

    def _stage_old_logprobs(self, rollout_idx: int) -> tuple[WorkerLogItem, float]:
        """Stage-A pass 1: recompute old logprobs and rollout-level metrics.

        Runs the iter-pack prefetch pipeline once; per pack the forward yields
        old logprobs while entropy/grad-token sums and IS metrics accumulate.
        Sums are all-reduced and normalized once at the end so communication
        does not scale with pack count.
        """
        # precompute float8 dynamic scale only once
        self.engine._maybe_precompute_float8_dynamic_scale_for_fsdp()

        # ``sft_train_metrics`` must exist even without the SFT
        worker_log_item: WorkerLogItem = {"train_entropy": 0.0, "train_metrics": [], "sft_train_metrics": {}}
        entropy_sum: torch.Tensor | None = None
        rollout_entropy_sum: torch.Tensor | None = None
        grad_tokens: torch.Tensor | None = None
        all_rollout_is_metrics: list[dict[str, float]] = []
        all_mismatch_metrics: list[dict[str, float]] = []
        pack_time_begin = time.perf_counter()

        for step_items in self._iter_steps():
            for pack, seq_ctx in step_items:
                ctx_dict = self.pack_loss_ctx(pack, seq_ctx, build_mtp=False)
                loss_ctx = cast(BaseRLLossContext, ctx_dict["lm"])
                kwargs = loss_ctx.loss_kwargs
                shifted_labels = kwargs.shifted_labels
                old_logprobs = self._forward_logprobs(seq_ctx, shifted_labels)
                kwargs.old_logprobs = old_logprobs

                mask = shifted_labels != -100
                entropy = -(old_logprobs * mask).sum()
                entropy_sum = entropy if entropy_sum is None else entropy_sum + entropy
                grad_tokens = mask.sum() if grad_tokens is None else grad_tokens + mask.sum()
                if kwargs.rollout_logprobs is not None:
                    rollout_entropy = -(kwargs.rollout_logprobs * mask).sum()
                    rollout_entropy_sum = (
                        rollout_entropy if rollout_entropy_sum is None else rollout_entropy_sum + rollout_entropy
                    )
                if kwargs.rollout_logprobs is not None:
                    num_tokens = seq_ctx.seq_lens_q
                    mismatch_metrics, rollout_is_metrics = loss_ctx.compute_rollout_is(self.host.sp_mesh, num_tokens)
                    all_rollout_is_metrics.append(rollout_is_metrics)
                    all_mismatch_metrics.append(mismatch_metrics)

                self._staged.setdefault(tuple(pack), {})["old_logprobs"] = old_logprobs.detach().cpu()
        pack_time_s = time.perf_counter() - pack_time_begin

        entropy_sum = cast(torch.Tensor, entropy_sum)
        grad_tokens = cast(torch.Tensor, grad_tokens)
        dist.all_reduce(entropy_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(grad_tokens, op=dist.ReduceOp.SUM)
        avg_sum_entropy = entropy_sum / grad_tokens if grad_tokens > 0 else torch.tensor(0.0)
        worker_log_item["train_entropy"] = avg_sum_entropy.item()

        logger_msg = f"Rollout {rollout_idx}: avg entropy: {avg_sum_entropy:.4f}"
        if rollout_entropy_sum is not None:
            dist.all_reduce(rollout_entropy_sum, op=dist.ReduceOp.SUM)
            avg_rollout_entropy = rollout_entropy_sum / grad_tokens if grad_tokens > 0 else torch.tensor(0.0)
            worker_log_item["rollout_entropy"] = avg_rollout_entropy.item()
            logger_msg += f", avg rollout entropy: {avg_rollout_entropy:.4f}"

        if len(all_mismatch_metrics) > 0:
            mismatch_metrics = merge_rollout_is_metrics(all_mismatch_metrics, DEVICE)
            if len(mismatch_metrics) > 0:
                worker_log_item["mismatch_metrics"] = mismatch_metrics
                logger_msg += f"\n rollout mismatch metrics:\n{json.dumps(mismatch_metrics, indent=4)}"
        if len(all_rollout_is_metrics) > 0:
            rollout_is_metrics = merge_rollout_is_metrics(all_rollout_is_metrics, DEVICE)
            if len(rollout_is_metrics) > 0:
                worker_log_item["rollout_is_metrics"] = rollout_is_metrics
                logger_msg += f"\n rollout importance sampling metrics:\n{json.dumps(rollout_is_metrics, indent=4)}"
        if self.rank == 0:
            self.logger.info(logger_msg)

        self._global_grad_tokens = grad_tokens
        return worker_log_item, pack_time_s

    def _stage_ref_logprobs(self) -> None:
        """Stage-A pass 2 (only with a ref model): ref logprobs + KL metric.

        The ref model is onloaded once for the whole pass and offloaded at the stage boundary; per-pack bouncing of the
        second model is not acceptable.
        """
        kl_div_sum: torch.Tensor | None = None
        self._ref_model.to_device(DEVICE)
        try:
            for step_items in self._iter_steps():
                for pack, seq_ctx in step_items:
                    staged = self._staged.setdefault(tuple(pack), {})
                    old_logprobs = staged["old_logprobs"]
                    assert old_logprobs is not None
                    ctx_dict = self.pack_loss_ctx(pack, seq_ctx, build_mtp=False)
                    loss_ctx = cast(BaseRLLossContext, ctx_dict["lm"])
                    kwargs = loss_ctx.loss_kwargs
                    ref_logprobs = self._forward_ref_logprobs(seq_ctx, kwargs.shifted_labels)
                    mask = kwargs.shifted_labels != -100
                    kl_div = kl_penalty(
                        old_logprobs.to(DEVICE),
                        ref_logprobs,
                        loss_weights=mask,
                        kl_penalty="low_var_kl",
                    )
                    kl_div_sum = kl_div if kl_div_sum is None else kl_div_sum + kl_div
                    staged["ref_logprobs"] = ref_logprobs.detach().cpu()
        finally:
            self._ref_model.to_device("cpu")
        kl_div_sum = cast(torch.Tensor, kl_div_sum)
        dist.all_reduce(kl_div_sum, op=dist.ReduceOp.SUM)
        grad_tokens = self._global_grad_tokens
        avg_kl_div = kl_div_sum / grad_tokens if grad_tokens > 0 else 0
        self.logger.info(f"avg KL divergence: {avg_kl_div:.4f}")

    def _stage_teacher_outputs(self) -> TrainTeacherTimings | None:
        """Stage-A pass 3 (only with train teachers): one whole-batch call.

        The teacher model is a second model: it goes on/off the GPU once at
        the stage boundary while the actor engine steps aside. The pass needs
        the full batch of prepared contexts alive at once, mirroring the
        pre-refactor behavior.
        """
        if self._train_teacher_manager is None:
            return None
        # Go through the granular role methods so the residency flag stays
        # correct (never touch engine.put_*_to_device directly).
        self.offload_model()
        self.offload_optimizer()
        if hasattr(DEVICE_MODULE, "empty_cache"):
            DEVICE_MODULE.empty_cache()
        try:
            seq_ctx_list: list[SequenceContext] = []
            loss_ctx_list: list[BaseRLLossContext] = []
            flat_packs = [p for step in self._plan() for p in step]
            for pack, seq_ctx in self._iter_packs(flat_packs):
                ctx_dict = self.pack_loss_ctx(pack, seq_ctx, build_mtp=False)
                loss_ctx_list.append(cast(BaseRLLossContext, ctx_dict["lm"]))
                seq_ctx_list.append(seq_ctx)
            teacher_timings = self._train_teacher_manager.compute_teacher_outputs(
                seq_ctx_list,
                loss_ctx_list,
            )
            for pack, loss_ctx in zip(flat_packs, loss_ctx_list):
                kwargs = loss_ctx.loss_kwargs
                teacher_logprobs = getattr(kwargs, "teacher_logprobs", None)
                target_token_ids = getattr(kwargs, "target_token_ids", None)
                staged = self._staged.setdefault(tuple(pack), {})
                staged["teacher_logprobs"] = teacher_logprobs.detach().cpu() if teacher_logprobs is not None else None
                staged["target_token_ids"] = target_token_ids.detach().cpu() if target_token_ids is not None else None
            return teacher_timings
        finally:
            self.onload_model()
            self.onload_optimizer()

    def _build_step_input(self, items: list[_PackItem]) -> list[ModelItem]:
        loss_cfg = self.config.loss_cfg
        LossContext = loss_cfg.loss_ctx_cls
        batched_lm = LossContext.build_batches([item[1]["lm"] for item in items])  # type: ignore[arg-type,misc]

        mtp_items: list[list[MTPLossContext]] = [
            item[1].get("mtp", [])  # type: ignore[assignment,misc]
            for item in items
        ]
        has_mtp = any(mtp_items)
        batched_mtp: list[list[MTPLossContext]] = [[] for _ in items]
        if has_mtp:
            cu_seq_lens_list = [item[0].cu_seq_lens_q for item in items]
            num_mtp_depths = len(mtp_items[0]) if mtp_items else 0
            for mtp_idx in range(num_mtp_depths):
                depth_mtp_loss_ctxs: list[LMHeadLossContext] = [depths[mtp_idx] for depths in mtp_items]
                batched_mtp_depth_ctxs = cast(
                    list[MTPLossContext],
                    MTPLossContext.build_batches(
                        depth_mtp_loss_ctxs,
                        cu_seq_lens_list=cu_seq_lens_list,
                        sp_mesh=self.host.sp_mesh,
                    ),
                )
                for batch_offset, mtp_ctx in enumerate(batched_mtp_depth_ctxs):
                    batched_mtp[batch_offset].append(mtp_ctx)

        engine_input: list[ModelItem] = []
        for item, lm_ctx, mtp_depths in zip(items, batched_lm, batched_mtp):
            loss_ctx_map: dict[str, BaseLossContext] = {"lm": lm_ctx}
            if has_mtp:
                loss_ctx_map["mtp"] = mtp_depths  # type: ignore[assignment]
            engine_input.append(ModelItem(seq_ctx=item[0], loss_ctx=loss_ctx_map))
        return engine_input

    def pack_loss_ctx(
        self,
        pack: list[int],
        seq_ctx: SequenceContext,
        *,
        build_mtp: bool = True,
    ) -> dict[str, BaseLossContext]:
        """Build the actor's packed loss context for one pack.

        Composes cached labels/advantages/rollout logprobs/rollout teacher
        targets with the stage-A rows (old/ref logprobs, train-teacher
        outputs) re-attached positionally. ``SequenceContext.to``/``split``
        are destructive, so the staged rows keep their stage-A SP-split shape
        and are only moved back to the device here, never re-split.
        """
        samples, _plan, batch_attr = self._cached()
        loss_inputs = self._pack_loss_fields(samples, pack, batch_attr)
        loss_cfg = self.config.loss_cfg
        loss_ctx = loss_cfg.build(data=loss_inputs, device="cpu")
        assert loss_ctx is not None
        kwargs = loss_ctx.loss_kwargs
        kwargs.to(DEVICE)
        # seq/loss alignment is by construction (same pack indices, same tail
        # padding rule); this is the belt-and-braces check. All loss fields are
        # pre-split here, so compare against the packed shifted-labels length,
        # not the (possibly SP-split) seq_ctx length.
        packed_len = cast(torch.Tensor, loss_inputs["shifted_labels"]).shape[1]
        for key, tensor in loss_inputs.items():
            assert tensor is None or tensor.shape[1] == packed_len, f"{key}: {tensor.shape[1]} vs {packed_len}"

        mtp_ctxs: list[MTPLossContext] = []
        if build_mtp and self.mtp_config is not None:
            for mtp_idx in range(self.mtp_config.num_layers):
                mtp_loss_cfg = MTPLossConfig(
                    **loss_cfg.model_dump(include={"mode", "chunk_size"}),
                    mtp_depth=mtp_idx + 1,
                    detach_mtp_lm_head_weight=self.mtp_config.detach_mtp_lm_head_weight,
                )
                mtp_ctx = mtp_loss_cfg.build(
                    data={
                        "shifted_labels": kwargs.shifted_labels,
                        "seq_ctx": seq_ctx,
                        "logprobs": kwargs.rollout_logprobs,
                    },
                    sp_mesh=self.host.sp_mesh,
                )
                if mtp_ctx is not None:
                    mtp_ctxs.append(mtp_ctx)

        if self.host.sp_mesh.size() > 1:
            kwargs.sp_split(self.host.sp_mesh)

        # Stage-A rows are already SP-split shaped (they were produced by
        # forwards on split contexts) and several loss-kwargs sp_split
        # implementations would re-split them, so they are re-attached only
        # after sp_split.
        staged = self._staged.get(tuple(pack), {})
        for field in ("old_logprobs", "ref_logprobs", "teacher_logprobs", "target_token_ids"):
            row = staged.get(field)
            if row is not None:
                setattr(kwargs, field, row.to(DEVICE))

        ctx_dict: dict[str, BaseLossContext] = {"lm": loss_ctx}
        if mtp_ctxs:
            ctx_dict["mtp"] = mtp_ctxs  # type: ignore[assignment]
        return ctx_dict

    def fit(self) -> WorkerLogItem:
        """Self-contained actor training: stage A then stage B.

        Stage A (GPU peak = 1 pack): the iter-pack pipeline recomputes old
        logprobs; the ref model (if any) runs the same pipeline once at a
        stage boundary; train teachers run one whole-batch call. Entropy /
        grad-token / KL accumulate raw sums and normalize once at the end;
        IS/mismatch are computed per pack and merged once. The
        ``ONLY_CALC_MISMATCH_RATIO`` debug path returns right after stage A.

        Stage B (GPU peak = 1 optimizer step): a fresh iter-pack stream
        re-cats each pack (to/split are destructive), re-attaches the staged
        rows, and feeds ``_train_optimizer_steps``. The SFT tail is actor-specific.
        """
        samples, plan, batch_attr = self._cached()
        rollout_idx = batch_attr["rollout_idx"]
        if any(sample.advantages is None for sample in samples):
            raise ValueError("actor fit requires staged advantages on every cached sample")
        num_optimizer_steps = len(plan)
        assert num_optimizer_steps <= self._optimizer_steps, (
            f"Pack plan optimizer steps {num_optimizer_steps} exceed configured "
            f"optimizer_steps {self._optimizer_steps}"
        )
        if num_optimizer_steps < self._optimizer_steps:
            self.logger.info(
                f"Optimizer only steps {num_optimizer_steps} times because pack plan steps "
                f"{num_optimizer_steps} < optimizer_steps {self._optimizer_steps}."
            )
        self._staged = {}
        try:
            return self._fit_stages(samples, plan, batch_attr, rollout_idx)
        finally:
            # The actor is the batch's last consumer: release the shared
            # routed-experts refs exactly once, on every exit path. Each stage
            # re-resolves from the pristine cache, so freeing earlier would
            # poison the later passes.
            self._staged = {}
            if self.config.free_rollout_routed_experts_in_worker:
                free_rollout_routed_expert_refs(samples, sp_mesh=self.host.sp_mesh)

    def _fit_stages(
        self,
        samples: list[TrainSample],
        plan: DPRankPackIndices,
        batch_attr: TrainBatchAttr,
        rollout_idx: int,
    ) -> WorkerLogItem:
        # ---- stage A ----
        worker_log_item, pack_time_s = self._stage_old_logprobs(rollout_idx)

        only_calc_mismatch_ratio = os.environ.get("ONLY_CALC_MISMATCH_RATIO", "0") == "1"
        if only_calc_mismatch_ratio:
            # This early-return path skips the MTP forward. FSDP may have
            # prefetched the MTP module already, leaving its all-gather result
            # cached until we release it explicitly.
            self._release_deferred_fsdp_all_gathers(f"rollout_{rollout_idx}/only_calc_mismatch")
            DEVICE_MODULE.empty_cache()
            return worker_log_item

        if self._has_ref:
            self._stage_ref_logprobs()

        teacher_timings = self._stage_teacher_outputs()
        if teacher_timings is not None:
            worker_log_item.update({"teacher_timings": teacher_timings.to_dict()})

        # ---- stage B ----
        def optimizer_step_packs() -> Iterator[list[_PackItem]]:
            for step_items in self._iter_steps():
                yield [(seq_ctx, self.pack_loss_ctx(pack, seq_ctx)) for pack, seq_ctx in step_items]

        train_log_item = self._train_optimizer_steps(optimizer_step_packs())
        worker_log_item["train_metrics"] = train_log_item["train_metrics"]
        worker_log_item["packing_pack_time_s"] = pack_time_s

        self._rollout_step += 1
        if self._sft_dataloader is not None and self._rollout_step % self._rollout_steps_per_sft == 0:
            train_step_info = self._fit_sft()
            engine_logs_info = train_step_info["logs_info"]
            worker_log_item["sft_train_metrics"] = {
                **engine_logs_info,
                **train_step_info["extra_info"].get(),
                "efficient_attn_ratio": train_step_info["efficient_attn_ratio"],
            }
        return worker_log_item

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
            seq_ctx = data["seq_ctx"].to(DEVICE)
            if self.host.sp_mesh.size() > 1:
                seq_ctx = seq_ctx.split(sequence_parallel_mesh=self.host.sp_mesh)
            seq_ctx_list.append(seq_ctx)
            loss_ctx = loss_cfg.build(data={"shifted_labels": data["shifted_labels"]}, sp_mesh=self.host.sp_mesh)
            assert loss_ctx is not None
            loss_ctx_list.append(loss_ctx)

        del data_batch

        cu_seq_lens_list = [seq_ctx.cu_seq_lens_q for seq_ctx in seq_ctx_list]
        loss_ctx_list = CELossContext.build_batches(
            loss_ctx_list, cu_seq_lens_list=cu_seq_lens_list, sp_mesh=self.host.sp_mesh
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

    def _release_deferred_fsdp_all_gathers(self, log_tag: str) -> None:
        from xtuner.v1.utils.fsdp import release_deferred_fsdp_all_gathers

        comm_states, param_groups = release_deferred_fsdp_all_gathers(self.engine.model)
        if comm_states or param_groups:
            self.logger.debug(
                f"[{log_tag}] released deferred FSDP all-gathers: "
                f"comm_states={comm_states}, param_groups={param_groups}"
            )

    def get_rollout_teacher_targets(
        self,
        state: RolloutState,
        *,
        shifted_labels: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        """Extract one rollout sample's distillation teacher target fields."""
        return self._distillation.rollout_teacher_targets(state, shifted_labels=shifted_labels)

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
        dataloader_state = torch.load(sft_dataloader_path, map_location=DEVICE)
        sft_dataloader.load_state_dict(dataloader_state)
        self.logger.info(f"Resume sft dataloader from {sft_dataloader_path}")

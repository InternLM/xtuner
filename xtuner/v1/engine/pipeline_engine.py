"""Pipeline-parallel training engine.

This engine is intentionally independent of FSDP. A model is split across ``pp_size`` pipeline stages
(each stage owns a contiguous range of decoder layers) and combined with expert parallel as
``world_size == pp_size * ep_size``. There is no parameter sharding (FSDP/HSDP); each stage holds the
full parameters of its layers, with MoE experts sharded across the expert-parallel dimension as usual.

The stage forward/backward is driven by ``torch.distributed.pipelining`` (``PipelineStage`` +
``Schedule1F1B``). xtuner's forward takes a structured ``SequenceContext`` rather than a single tensor,
and the loss is computed inside the model, so the schedule is driven through the lower-level
``_step_microbatches`` API: the per-microbatch ``seq_ctx`` / ``loss_ctx`` are passed as (non-chunked)
keyword arguments, and ``loss_fn`` is an identity passthrough because the last stage already returns
the total loss tensor.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    Schedule1F1B,
    ScheduleInterleaved1F1B,
    _PipelineSchedule,
)
from torch.distributed.tensor import DTensor
from torch.nn.utils.clip_grad import _no_grad
from torch.utils._foreach_utils import _device_has_foreach_support

from xtuner.v1.config import OptimConfig, PipelineParallelConfig
from xtuner.v1.engine.train_engine import TrainStepInfo
from xtuner.v1.model.base import BaseModel, BatchForwardInfo, ModelItem, ModelOutputs, XTunerBaseModelConfig
from xtuner.v1.model.utils.misc import ModelForwardExtraLogInfo
from xtuner.v1.utils import get_device, log_rank0


DEVICE = get_device()


class _PipelineStageModule(nn.Module):
    """Adapt one (virtual) pipeline stage to the tensor-in/tensor-out contract of ``PipelineStage``.

    A stage runs only its chunk of decoder layers (``layer_keys``). The first (global) stage ignores its
    positional input and embeds ``seq_ctx``; later stages consume the previous stage's ``hidden_states``.
    The last (global) stage returns the total loss (sum of all loss terms, mirroring the FSDP engine's
    aggregation) so the schedule can drive backward from it; other stages return the ``hidden_states``
    tensor to send downstream. With one virtual stage per rank ``layer_keys`` covers all owned layers;
    under interleaving each virtual stage gets its own chunk's keys.
    """

    def __init__(self, model: BaseModel, layer_keys: list[str], is_first: bool, is_last: bool):
        super().__init__()
        self.model = model
        self.layer_keys = layer_keys
        self.is_first = is_first
        self.is_last = is_last

    def forward(self, stage_input, seq_ctx=None, loss_ctx=None):
        x = None if self.is_first else stage_input
        out = self.model.pipeline_forward(
            x, seq_ctx, loss_ctx, is_first=self.is_first, is_last=self.is_last, layer_indices=self.layer_keys
        )
        if not self.is_last:
            return out
        return _total_loss(out)


def _total_loss(model_outputs: ModelOutputs) -> torch.Tensor:
    # Mirror TrainEngine._get_total_loss: sum every loss-named tensor field (CE + MoE balancing/z + mtp)
    # so backward covers all of them.
    total: torch.Tensor | None = None
    for key in type(model_outputs).model_fields:
        value = getattr(model_outputs, key)
        if "loss" in key and isinstance(value, torch.Tensor):
            total = value if total is None else total + value
    assert total is not None, "last pipeline stage produced no loss field"
    return total


class PPEngine:
    """Training engine for pure pipeline parallel (optionally combined with expert parallel)."""

    model: BaseModel
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        model_cfg: XTunerBaseModelConfig,
        optim_cfg: OptimConfig,
        pp_cfg: PipelineParallelConfig,
        ep_size: int = 1,
        param_dtype: torch.dtype = torch.bfloat16,
        recompute_ratio: float = 0.0,
    ) -> None:
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.pp_cfg = pp_cfg
        self.ep_size = ep_size
        self.param_dtype = param_dtype
        self.recompute_ratio = recompute_ratio

        world_size = dist.get_world_size()
        if world_size != pp_cfg.pp_size * ep_size:
            raise ValueError(f"world_size ({world_size}) must equal pp_size ({pp_cfg.pp_size}) * ep_size ({ep_size})")
        if pp_cfg.pp_size < 2:
            raise ValueError(f"PPEngine requires pp_size >= 2, got {pp_cfg.pp_size}")

        prefix = model_cfg.mesh_prefix
        # (pp, ep) mesh. The pp group send/recv activations between stages; the ep group runs MoE
        # all2all *and* serves as the data-parallel group for loss / grad reduction (the ranks sharing
        # this rank's pipeline stage). They coincide in v1 because ep is the only non-pp dimension.
        self.mesh = init_device_mesh(
            str(DEVICE), (pp_cfg.pp_size, ep_size), mesh_dim_names=(f"{prefix}.pp", f"{prefix}.ep")
        )
        self.pp_mesh = self.mesh[f"{prefix}.pp"]
        self.ep_mesh = self.mesh[f"{prefix}.ep"]
        self.pp_rank = self.pp_mesh.get_local_rank()
        self.num_stages = pp_cfg.pp_size
        self.num_virtual_stages = pp_cfg.num_virtual_stages
        # Total virtual stages across the whole pipeline. Global stage 0 lives on rank 0 and the last
        # global stage on rank num_stages-1 regardless of interleaving, so first/last-rank tests are
        # unchanged. ``self._owned_stages`` (set in build_model) lists this rank's (global_idx, keys).
        self.total_stages = self.num_stages * self.num_virtual_stages
        self.is_first = self.pp_rank == 0
        self.is_last = self.pp_rank == self.num_stages - 1

        if self.num_virtual_stages > 1 and (
            getattr(model_cfg, "balancing_loss_cfg", None) is not None
            or getattr(model_cfg, "z_loss_cfg", None) is not None
        ):
            # The MoE aux (balancing / z) loss accumulates per-layer stats in model-global state that is
            # drained once per forward at finalize. Interleaving runs several forwards per finalize
            # (across microbatches and virtual stages), so that accumulation mixes microbatches and the
            # finalize shapes stop matching. Block it clearly until aux accumulation is made
            # per-microbatch (and finalized per stage) for interleaved schedules.
            raise NotImplementedError(
                "Interleaved pipeline parallel (num_virtual_stages > 1) does not support the MoE "
                "balancing / z auxiliary loss yet. Set balancing_loss_cfg=None and z_loss_cfg=None, or "
                "use num_virtual_stages=1."
            )

        self.model = self.build_model()
        self.optimizer = self.build_optimizer(optim_cfg)

        self._stages: list[PipelineStage] | None = None
        self._schedule: _PipelineSchedule | None = None
        self._stage_modules = [
            _PipelineStageModule(
                self.model,
                layer_keys=keys,
                is_first=(global_idx == 0),
                is_last=(global_idx == self.total_stages - 1),
            )
            for global_idx, keys in self._owned_stages
        ]

    def build_model(self) -> BaseModel:
        model = self.model_cfg.build()
        self._owned_stages = model.split_for_pipeline(
            self.pp_rank,
            self.num_stages,
            layer_split=self.pp_cfg.layer_split,
            num_virtual_stages=self.num_virtual_stages,
        )
        # Activation checkpointing is orthogonal to FSDP; apply the same per-layer recompute policy the
        # FSDP path uses (only this stage's owned layers are wrapped, by global layer index).
        if self.recompute_ratio > 0:
            model.apply_activation_checkpointing(self.recompute_ratio)
        model = model.to(self.param_dtype).to(DEVICE)
        # Set parallel attributes after .to() so they live on the final module instance.
        # Enable expert parallel without FSDP: the model is built EP-aware (experts sharded across the
        # ep dimension), so we only need to bind the ep mesh used by the MoE all2all.
        if self.ep_size > 1:
            model.ep_mesh = self.ep_mesh  # type: ignore[attr-defined]
        # Data-parallel group for loss / aux reductions (= this rank's pipeline-stage group).
        model.dp_group = self.ep_mesh.get_group()
        return model

    def build_optimizer(self, optim_cfg: OptimConfig) -> torch.optim.Optimizer:
        optimizer = optim_cfg.build(self.model)
        # Without FSDP the model mixes plain (replicated) parameters with DTensor expert shards. The
        # fused/foreach optimizer kernels cannot operate on a group that mixes plain tensors and
        # DTensors, so fall back to the per-parameter path when both kinds are present. (With ep_size==1
        # every parameter is plain, so foreach stays enabled.)
        has_plain = any(p.requires_grad and not isinstance(p, DTensor) for p in self.model.parameters())
        has_dtensor = any(p.requires_grad and isinstance(p, DTensor) for p in self.model.parameters())
        if has_plain and has_dtensor:
            for group in optimizer.param_groups:
                group["foreach"] = False
                group["fused"] = False
        return optimizer

    def init_model_weights(self):
        self.model.init_weights()

    def from_hf(self, hf_path, strict: bool = False):
        self.model.from_hf(hf_path=hf_path, strict=strict)

    def save_hf(self, hf_dir, save_dtype: torch.dtype = torch.bfloat16):
        """Save the model in HuggingFace format.

        Each pipeline stage writes its own parameters; the per-stage shards are merged into one
        ``model.safetensors.index.json`` covering the full model (the model save path gathers the
        weight map across all ranks).

        Args:
            hf_dir (str | Path): Destination directory.
            save_dtype (torch.dtype): Dtype to cast parameters to when saving.
        """
        self.model.save_hf(hf_dir, save_dtype=save_dtype)

    def train_step(self, data_batches: list[ModelItem]) -> "TrainStepInfo":
        """Run one optimizer-step worth of microbatches through the pipeline schedule.

        Args:
            data_batches (list[ModelItem]): The microbatches for this step. Their length is the number
                of pipeline microbatches and must be >= num_stages.

        Returns:
            TrainStepInfo: The data-batch statistics plus ``total_loss`` reduced across the pipeline.
                ``logs_info`` / ``extra_info`` are left empty because the per-microbatch outputs only
                exist on the last stage and their reduction collectives run over WORLD, which would
                deadlock the stages that never produce outputs.
        """
        n_microbatches = len(data_batches)
        if n_microbatches < self.num_stages:
            raise ValueError(
                f"number of microbatches ({n_microbatches}) must be >= num_stages ({self.num_stages}); "
                "increase the gradient-accumulation / global batch size."
            )

        seq_ctx_list = [item["seq_ctx"] for item in data_batches]
        loss_ctx_list = [item["loss_ctx"] for item in data_batches]
        schedule = self._get_schedule(n_microbatches, seq_ctx_list[0])

        # Drive the schedule at the microbatch level so seq_ctx/loss_ctx ride along as per-microbatch
        # kwargs instead of being chunked. The first stage's positional input is a placeholder (it
        # embeds from seq_ctx); later stages receive their input from the previous stage.
        example = self._example_hidden(seq_ctx_list[0])
        if self.is_first:
            arg_mbs = [(example,) for _ in range(n_microbatches)]
        else:
            arg_mbs = [() for _ in range(n_microbatches)]
        kwarg_mbs = [{"seq_ctx": seq_ctx_list[i], "loss_ctx": loss_ctx_list[i]} for i in range(n_microbatches)]

        # Replicate what Schedule.step() does before driving microbatches (we bypass step() to keep
        # seq_ctx un-chunked): set has_backward and clear runtime state on every owned (virtual) stage.
        assert self._stages is not None
        for stage in self._stages:
            stage.has_backward = schedule._has_backward  # type: ignore[attr-defined]
            stage.clear_runtime_states()

        losses: list[torch.Tensor] = []
        if self.is_last:
            schedule._step_microbatches(
                arg_mbs=arg_mbs, kwarg_mbs=kwarg_mbs, target_mbs=[None] * n_microbatches, losses=losses
            )
        else:
            schedule._step_microbatches(arg_mbs=arg_mbs, kwarg_mbs=kwarg_mbs)

        total_loss = self._reduce_step_loss(losses)
        # ``pre_micro_batch_forward`` derives its stats purely from seq_ctx (no collectives), so it is
        # safe on every stage; the output-derived ``batch_forward_info`` is intentionally left empty.
        data_batch_info = self.model.pre_micro_batch_forward(data_batches)
        batch_forward_info = BatchForwardInfo(logs_info={}, extra_info=ModelForwardExtraLogInfo())
        return TrainStepInfo(total_loss=total_loss, **data_batch_info, **batch_forward_info)

    def destroy_async_checkpoint_pg(self) -> None:
        # Pipeline parallel has no async-checkpoint process group; provided so the trainer can call it
        # uniformly across engines.
        return None

    @_no_grad
    def clip_grad_norm(self, do_clip: bool = True, dtype=torch.float32) -> torch.Tensor:
        """Reduce gradients across the data-parallel group, compute the global grad norm across
        pipeline stages and the expert-parallel dimension, and optionally clip.

        Unlike the FSDP engine this computes the norm directly rather than via ``cal_grad_norm``: most
        parameters are plain (replicated) tensors, while MoE expert parameters are ``DTensor`` shards on
        the expert-parallel mesh, which need different reduction scopes (see below).
        """
        self._reduce_grads_over_dp()

        # Expert grads are DTensor shards on the ep mesh; everything else is a plain replicated tensor.
        replicated_grads: list[torch.Tensor] = []
        expert_grads: list[torch.Tensor] = []
        for _, param in self.model.trainable_parameters():
            if param.grad is None:
                continue
            (expert_grads if isinstance(param.grad, DTensor) else replicated_grads).append(param.grad)

        # Replicated grads are identical across the ep group, so their squared norms are summed over the
        # pipeline group only (different stages own different parameters) to avoid ep double-counting.
        # Expert grads are sharded over ep and split across stages, so their local-shard squared norms
        # are summed over the whole pp x ep world (== WORLD here) so each shard counts exactly once.
        replicated_sq = self._sum_squared_norm(replicated_grads, dtype)
        expert_sq = self._sum_squared_norm([g.to_local() for g in expert_grads], dtype)
        if self.pp_mesh.size() > 1:
            dist.all_reduce(replicated_sq, op=dist.ReduceOp.SUM, group=self.pp_mesh.get_group())
        if self.ep_mesh.size() > 1:
            dist.all_reduce(expert_sq, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        grad_norm = (replicated_sq + expert_sq).sqrt()

        if do_clip:
            clip_coef = self.optim_cfg.max_grad_norm / (grad_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            # DTensor grads cannot be mixed into a foreach call with plain tensors, so clip each list
            # with the kernel it supports.
            if replicated_grads and _device_has_foreach_support(replicated_grads[0].device):
                torch._foreach_mul_(replicated_grads, clip_coef_clamped.to(replicated_grads[0].device))
            else:
                for grad in replicated_grads:
                    grad.mul_(clip_coef_clamped.to(grad.device))
            for grad in expert_grads:
                grad.mul_(clip_coef_clamped.to(grad.device))
        return grad_norm

    @staticmethod
    def _sum_squared_norm(grads: list[torch.Tensor], dtype) -> torch.Tensor:
        if not grads:
            return torch.zeros((), device=DEVICE, dtype=dtype)
        return torch.stack([g.detach().to(dtype).norm(2) ** 2 for g in grads]).sum()

    def step_optimizer(self, grad_norm: torch.Tensor) -> torch.Tensor:
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            log_rank0.warning(f"Gradient norm {grad_norm} is invalid, skipping optimizer step.")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return grad_norm

    def _get_schedule(self, n_microbatches: int, seq_ctx) -> _PipelineSchedule:
        if self._schedule is not None:
            return self._schedule
        example = self._example_hidden(seq_ctx)
        loss_example = torch.zeros((), dtype=torch.float32, device=DEVICE)
        stages: list[PipelineStage] = []
        for (global_idx, _keys), module in zip(self._owned_stages, self._stage_modules):
            # Provide output_args so PipelineStage skips its forward-based shape inference, which would
            # call the stage without our seq_ctx kwarg. The last global stage outputs the (float32)
            # scalar loss; every other stage outputs hidden_states.
            output_args: tuple = (loss_example,) if global_idx == self.total_stages - 1 else (example,)
            stages.append(
                PipelineStage(
                    module,
                    global_idx,
                    self.total_stages,
                    torch.device(DEVICE),
                    input_args=(example,),
                    output_args=output_args,
                    group=self.pp_mesh.get_group(),
                )
            )
        self._stages = stages
        if self.num_virtual_stages == 1:
            self._schedule = Schedule1F1B(stages[0], n_microbatches=n_microbatches, loss_fn=_passthrough_loss)
        else:
            self._schedule = ScheduleInterleaved1F1B(stages, n_microbatches=n_microbatches, loss_fn=_passthrough_loss)
        return self._schedule

    def _example_hidden(self, seq_ctx) -> torch.Tensor:
        seq_len = seq_ctx.input_ids.shape[1] if seq_ctx.input_ids is not None else seq_ctx.inputs_embeds.shape[1]
        return torch.zeros(1, seq_len, self.model_cfg.hidden_size, dtype=self.param_dtype, device=DEVICE)

    def _reduce_step_loss(self, losses: list[torch.Tensor]) -> float:
        # Only the last stage computes the loss; broadcast it across the pipeline for logging.
        loss_t = torch.zeros((), device=DEVICE, dtype=torch.float32)
        if self.is_last and losses:
            loss_t = torch.stack([loss.detach().float() for loss in losses]).sum()
        if self.pp_mesh.size() > 1:
            src = int(self.pp_mesh.mesh[-1].item())  # global rank of the last pipeline stage
            dist.broadcast(loss_t, src=src, group=self.pp_mesh.get_group())
        return loss_t.item()

    @_no_grad
    def _reduce_grads_over_dp(self) -> None:
        # Pure-PP analogue of MoE.scale_and_reduce_grad for plain (non-DTensor) parameters: replicated
        # (non-expert) grads are averaged across the data-parallel group; expert grads, which live on a
        # single ep rank, are only rescaled.
        ep_size = self.ep_mesh.size()
        dp_size = self.ep_mesh.size()
        if dp_size == 1:
            return
        dp_group = self.ep_mesh.get_group()
        replicated_grads: list[torch.Tensor] = []
        for name, param in self.model.trainable_parameters():
            if param.grad is None:
                continue
            if ".experts" in name:
                param.grad.div_(ep_size)
                continue
            param.grad.div_(dp_size)
            replicated_grads.append(param.grad)
        if replicated_grads:
            with dist._coalescing_manager(group=dp_group):
                for grad in replicated_grads:
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=dp_group)


def _passthrough_loss(output: torch.Tensor, target) -> torch.Tensor:
    # The last stage already returns the total loss tensor; the schedule only needs to start backward
    # from it, so the loss function is the identity.
    return output

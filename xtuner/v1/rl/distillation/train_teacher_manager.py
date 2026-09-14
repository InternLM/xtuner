from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, cast

import torch

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.loss import LogProbConfig, LogProbContext, TopKLogProbConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.trainer.model_utils import FrozenModel, build_frozen_model
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module

from .config import TrainTeacherManagerConfig


if TYPE_CHECKING:
    from xtuner.v1.rl.loss.base_loss import BaseRLLossContext


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


@dataclass
class TeacherTiming:
    """Wall-clock seconds spent in one frozen Teacher lifecycle."""

    compute: float = 0.0
    onload: float = 0.0
    offload: float = 0.0


@dataclass
class TrainTeacherTimings(TeacherTiming):
    """Wall-clock seconds spent in each frozen Teacher lifecycle."""

    by_teacher: dict[str, TeacherTiming] = field(default_factory=dict)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Return per-Teacher lifecycle timings.

        Returns:
            dict[str, dict[str, float]]: Teacher names mapped to compute, onload, and offload seconds.
        """
        return {
            teacher_name: {
                "compute": timing.compute,
                "onload": timing.onload,
                "offload": timing.offload,
            }
            for teacher_name, timing in self.by_teacher.items()
        }


@dataclass
class TrainTeacherOutputs:
    teacher_logprobs: list[torch.Tensor]
    target_token_ids: list[torch.Tensor] | None = None
    timings: TrainTeacherTimings = field(default_factory=TrainTeacherTimings)


class TrainTeacherManager:
    """Execute training-side Teachers in one TrainingWorker process.

    The manager owns Teacher model construction, deterministic Teacher-major scheduling, CPU/device residency, and
    sampled-token or top-k output calculation. The caller remains responsible for swapping the Actor and optimizer
    around the Teacher phase.
    """

    def __init__(self, config: TrainTeacherManagerConfig, *, chunk_size: int | None) -> None:
        self.target_config = config.target_config
        mode = "chunk" if chunk_size is not None else "eager"
        self.logprob_config = LogProbConfig(chunk_size=chunk_size, mode=mode)
        self.topk_logprob_config: TopKLogProbConfig | None = None
        if self.target_config.uses_topk_targets:
            self.topk_logprob_config = TopKLogProbConfig(
                top_k=cast(int, self.target_config.top_k),
                chunk_size=chunk_size,
                mode=mode,
            )

        # Build every frozen Teacher before the Actor so checkpoint/config
        # errors fail during worker initialization. Each Teacher is offloaded
        # immediately, preventing multiple full models from co-residing on GPU.
        self._teachers: list[FrozenModel] = []
        for teacher_config in config.teachers:
            teacher = build_frozen_model(
                teacher_config.model_cfg,
                teacher_config.model_path,
                teacher_config.fsdp_cfg,
            )
            self._teachers.append(teacher)

        self._teacher_is_composed = [isinstance(teacher.model_cfg, BaseComposeConfig) for teacher in config.teachers]
        self._teacher_index_by_name = {
            teacher_config.name: teacher_index for teacher_index, teacher_config in enumerate(config.teachers)
        }
        self._teacher_names = [teacher_config.name for teacher_config in config.teachers]

    def compute_logprobs(
        self,
        *,
        seq_ctx_list: list[SequenceContext],
        shifted_labels_list: list[torch.Tensor],
        teacher_indices_list: list[torch.Tensor],
    ) -> TrainTeacherOutputs:
        timings = TrainTeacherTimings()
        if self.target_config.uses_sampled_token_targets:
            return TrainTeacherOutputs(
                teacher_logprobs=self._compute_sampled_logprobs(
                    seq_ctx_list,
                    shifted_labels_list,
                    teacher_indices_list,
                    timings,
                ),
                timings=timings,
            )

        target_token_ids, teacher_logprobs = self._compute_topk_targets(
            seq_ctx_list,
            teacher_indices_list,
            timings,
        )
        return TrainTeacherOutputs(
            teacher_logprobs=teacher_logprobs,
            target_token_ids=target_token_ids,
            timings=timings,
        )

    def compute_teacher_outputs(
        self,
        seq_ctx_list: list[SequenceContext],
        loss_ctx_list: list[BaseRLLossContext],
    ) -> TrainTeacherTimings:
        """Compute Teacher outputs for the given loss contexts."""
        teacher_indices_list: list[torch.Tensor] = []
        shifted_labels_list: list[torch.Tensor] = []
        for loss_ctx in loss_ctx_list:
            loss_kwargs = cast(Any, loss_ctx.loss_kwargs)
            teacher_indices = loss_kwargs.teacher_indices
            if teacher_indices is None:
                raise ValueError("teacher_indices are required when the training Teacher manager is enabled")
            teacher_indices_list.append(teacher_indices)
            shifted_labels_list.append(loss_kwargs.shifted_labels)

        try:
            outputs = self.compute_logprobs(
                seq_ctx_list=seq_ctx_list,
                shifted_labels_list=shifted_labels_list,
                teacher_indices_list=teacher_indices_list,
            )
            target_token_ids_list: list[torch.Tensor | None] = (
                list(outputs.target_token_ids)
                if outputs.target_token_ids is not None
                else [None] * len(outputs.teacher_logprobs)
            )
            for loss_ctx, teacher_logprobs, target_token_ids in zip(
                loss_ctx_list,
                outputs.teacher_logprobs,
                target_token_ids_list,
            ):
                loss_kwargs = cast(Any, loss_ctx.loss_kwargs)
                loss_kwargs.teacher_logprobs = teacher_logprobs
                if target_token_ids is not None:
                    loss_kwargs.target_token_ids = target_token_ids
            return outputs.timings
        finally:
            self.offload_all_to_cpu()

    def offload_all_to_cpu(self) -> None:
        for teacher in self._teachers:
            self._offload_to_cpu(teacher)

    def offload_to_disk(self, teacher_name: str) -> None:
        """Reserve the disk-offload lifecycle boundary for a later backend."""
        if teacher_name not in self._teacher_index_by_name:
            raise KeyError(f"Unknown training Teacher: {teacher_name!r}")
        raise NotImplementedError("Train Teacher disk offload is not implemented")

    @staticmethod
    def _offload_to_cpu(teacher: FrozenModel) -> None:
        teacher.to_device("cpu")
        if hasattr(DEVICE_MODULE, "empty_cache"):
            DEVICE_MODULE.empty_cache()

    @staticmethod
    def _synchronize_device() -> None:
        if str(DEVICE) != "cpu" and hasattr(DEVICE_MODULE, "synchronize"):
            DEVICE_MODULE.synchronize()

    @contextmanager
    def _teacher_on_device(
        self,
        teacher: FrozenModel,
        timings: TrainTeacherTimings,
        teacher_name: str | None = None,
    ) -> Iterator[None]:
        teacher_timing = (
            timings.by_teacher.setdefault(teacher_name, TeacherTiming()) if teacher_name is not None else None
        )

        onload_begin = time.perf_counter()
        teacher.to_device(DEVICE)
        onload_elapsed = time.perf_counter() - onload_begin
        timings.onload += onload_elapsed
        if teacher_timing is not None:
            teacher_timing.onload += onload_elapsed

        compute_begin = time.perf_counter()
        try:
            yield
            self._synchronize_device()
        finally:
            compute_elapsed = time.perf_counter() - compute_begin
            timings.compute += compute_elapsed
            if teacher_timing is not None:
                teacher_timing.compute += compute_elapsed
            offload_begin = time.perf_counter()
            self._offload_to_cpu(teacher)
            offload_elapsed = time.perf_counter() - offload_begin
            timings.offload += offload_elapsed
            if teacher_timing is not None:
                teacher_timing.offload += offload_elapsed

    @staticmethod
    def construct_teacher_seq_ctx(seq_ctx: SequenceContext, *, is_composed: bool) -> SequenceContext:
        """Construct a Teacher context from the Student rollout context.

        Rollout-only expert-routing metadata must not be reused by the Teacher. A plain language-model Teacher can only
        score text tokens. Composed/VLM Teachers retain their visual fields and M-RoPE positions; plain-text Teachers
        require text-only contexts with 2D packed positions.
        """
        has_visual_inputs = seq_ctx.pixel_values is not None
        if not is_composed and has_visual_inputs:
            logger.warning(
                "A plain-text Teacher received a multimodal rollout context; "
                "the sample cannot be scored by this Teacher."
            )
            raise ValueError("Plain-text Teacher cannot score a rollout context containing visual inputs")

        position_ids = seq_ctx.position_ids
        if not is_composed and position_ids is not None and position_ids.ndim == 3:
            # SequenceContext rebuilds standard 2D packed positions.
            position_ids = None

        return SequenceContext(
            input_ids=seq_ctx.input_ids,
            cu_seq_lens_q=seq_ctx.cu_seq_lens_q,
            cu_seq_lens_k=seq_ctx.cu_seq_lens_k,
            max_length_q=seq_ctx.max_length_q,
            max_length_k=seq_ctx.max_length_k,
            num_padding=seq_ctx.num_padding,
            sequence_parallel_mesh=seq_ctx.sequence_parallel_mesh,
            block_table=seq_ctx.block_table,
            device=seq_ctx.device,
            position_ids=position_ids,
            image_grid_thw=seq_ctx.image_grid_thw if is_composed else None,
            deepstack_visual_embeds=seq_ctx.deepstack_visual_embeds if is_composed else None,
            visual_pos_masks=seq_ctx.visual_pos_masks if is_composed else None,
            pixel_values=seq_ctx.pixel_values if is_composed else None,
            inputs_embeds=seq_ctx.inputs_embeds if is_composed else None,
            num_img_tokens=seq_ctx.num_img_tokens if is_composed else None,
            rollout_routed_experts=None,
            offload_rollout_routed_experts=False,
            raw_input_ids=seq_ctx._raw_input_ids,
            raw_inputs_embeds=seq_ctx._raw_inputs_embeds,
            shard_start=seq_ctx._shard_start,
            shard_size=seq_ctx._shard_size,
        )

    def _compute_topk_targets(
        self,
        seq_ctx_list: list[SequenceContext],
        teacher_indices_list: list[torch.Tensor],
        timings: TrainTeacherTimings,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        top_k = cast(int, self.target_config.top_k)
        topk_logprob_config = cast(TopKLogProbConfig, self.topk_logprob_config)
        target_ids = [
            torch.zeros((*teacher_indices.shape, top_k), dtype=torch.long, device=DEVICE)
            for teacher_indices in teacher_indices_list
        ]
        target_logprobs = [
            torch.zeros((*teacher_indices.shape, top_k), dtype=torch.float32, device=DEVICE)
            for teacher_indices in teacher_indices_list
        ]

        # All ranks iterate Teachers and local packs in the same order so every
        # FSDP rank enters the same collective sequence, even when a rank has no
        # tokens routed to a particular Teacher.
        for teacher_index, (teacher_name, teacher) in enumerate(zip(self._teacher_names, self._teachers)):
            with self._teacher_on_device(teacher, timings, teacher_name):
                # Every rank forwards every local pack before selecting routed
                # tokens so all ranks enter the same FSDP collective sequence.
                # TODO(perf): Use a collective-safe compacted-token path to
                # avoid computing teacher logprobs for tokens routed to other
                # Teachers.
                for batch_index, (seq_ctx, teacher_indices) in enumerate(zip(seq_ctx_list, teacher_indices_list)):
                    loss_ctx = topk_logprob_config.build(data={})
                    assert loss_ctx is not None
                    with torch.no_grad():
                        output = teacher(
                            seq_ctx=self.construct_teacher_seq_ctx(
                                seq_ctx,
                                is_composed=self._teacher_is_composed[teacher_index],
                            ),
                            loss_ctx={"lm": loss_ctx},
                        )
                    selected = teacher_indices == teacher_index
                    # Top-k teacher outputs reuse ModelOutputs.logits for token ids
                    # and ModelOutputs.loss for selected logprobs.
                    topk_token_ids = cast(torch.Tensor, output.logits)
                    topk_logprobs = cast(torch.Tensor, output.loss)
                    target_ids[batch_index][selected] = topk_token_ids[selected]
                    target_logprobs[batch_index][selected] = topk_logprobs[selected]
        return target_ids, target_logprobs

    def _compute_sampled_logprobs(
        self,
        seq_ctx_list: list[SequenceContext],
        shifted_labels_list: list[torch.Tensor],
        teacher_indices_list: list[torch.Tensor],
        timings: TrainTeacherTimings,
    ) -> list[torch.Tensor]:
        target_logprobs = [
            torch.zeros_like(shifted_labels, dtype=torch.float32) for shifted_labels in shifted_labels_list
        ]

        # Keep the Teacher-major schedule identical across ranks for FSDP.
        for teacher_index, (teacher_name, teacher) in enumerate(zip(self._teacher_names, self._teachers)):
            with self._teacher_on_device(teacher, timings, teacher_name):
                # TODO(perf): Forward only tokens routed to this Teacher once
                # a collective-safe compacted-token path is available; the
                # current full-pack forward computes logprobs for other Teachers
                # as well.
                for batch_index, (seq_ctx, shifted_labels, teacher_indices) in enumerate(
                    zip(seq_ctx_list, shifted_labels_list, teacher_indices_list)
                ):
                    loss_ctx = cast(
                        LogProbContext,
                        self.logprob_config.build(data={"shifted_labels": shifted_labels}),
                    )
                    with torch.no_grad():
                        output = teacher(
                            seq_ctx=self.construct_teacher_seq_ctx(
                                seq_ctx,
                                is_composed=self._teacher_is_composed[teacher_index],
                            ),
                            loss_ctx={"lm": loss_ctx},
                        )
                    selected = teacher_indices == teacher_index
                    target_logprobs[batch_index][selected] = cast(torch.Tensor, output.loss)[selected]
        return target_logprobs

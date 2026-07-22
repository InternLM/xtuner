from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, cast

import torch

from xtuner.v1.data_proto.sequence_context import DSATopKCacheState, SequenceContext
from xtuner.v1.loss import LogProbConfig, LogProbContext, TopKLogProbConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.loss import DistillationLossConfig
from xtuner.v1.rl.model_utils import FrozenModel, build_frozen_model
from xtuner.v1.utils import get_device, get_torch_device_module

from .config import DistillationConfig


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


@dataclass
class TrainTeacherTimings:
    """Wall-clock seconds spent in the frozen Teacher lifecycle."""

    compute: float = 0.0
    onload: float = 0.0
    offload: float = 0.0


@dataclass
class TrainTeacherOutputs:
    teacher_logprobs: list[torch.Tensor]
    target_token_ids: list[torch.Tensor] | None = None
    timings: TrainTeacherTimings = field(default_factory=TrainTeacherTimings)


class TrainTeacherManager:
    """Execute training-side Teachers in one TrainingWorker process.

    The manager owns Teacher model construction, deterministic Teacher-major scheduling, CPU/device residency, and
    sampled-token or top-k output calculation. The caller remains responsible for preparing distributed inputs and
    swapping the Actor and optimizer around the Teacher phase.
    """

    def __init__(self, distillation_config: DistillationConfig, *, chunk_size: int | None) -> None:
        self.loss_config = cast(DistillationLossConfig, distillation_config.loss_config)
        mode = "chunk" if chunk_size is not None else "eager"
        self.logprob_config = LogProbConfig(chunk_size=chunk_size, mode=mode)
        self.topk_logprob_config: TopKLogProbConfig | None = None
        if self.loss_config.uses_topk_targets:
            self.topk_logprob_config = TopKLogProbConfig(
                top_k=cast(int, self.loss_config.top_k),
                chunk_size=chunk_size,
                mode=mode,
            )

        self._teachers: list[FrozenModel] = [
            build_frozen_model(teacher.model_cfg, teacher.model_path, teacher.fsdp_cfg)
            for teacher in distillation_config.train_teachers
        ]
        self._teacher_is_composed = [
            isinstance(teacher.model_cfg, BaseComposeConfig) for teacher in distillation_config.train_teachers
        ]
        self._teachers_by_name = {
            teacher_config.name: teacher
            for teacher_config, teacher in zip(distillation_config.train_teachers, self._teachers)
        }

    def compute_logprobs(
        self,
        *,
        seq_ctx_list: list[SequenceContext],
        shifted_labels_list: list[torch.Tensor],
        teacher_indices_list: list[torch.Tensor],
    ) -> TrainTeacherOutputs:
        timings = TrainTeacherTimings()
        if self.loss_config.uses_sampled_token_targets:
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

    def offload_all_to_cpu(self) -> None:
        for teacher in self._teachers:
            self._offload_to_cpu(teacher)

    def offload_to_disk(self, teacher_name: str) -> None:
        """Reserve the disk-offload lifecycle boundary for a later backend."""
        if teacher_name not in self._teachers_by_name:
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
    ) -> Iterator[None]:
        onload_begin = time.perf_counter()
        teacher.to_device(DEVICE)
        timings.onload += time.perf_counter() - onload_begin

        compute_begin = time.perf_counter()
        try:
            yield
            self._synchronize_device()
        finally:
            timings.compute += time.perf_counter() - compute_begin
            offload_begin = time.perf_counter()
            self._offload_to_cpu(teacher)
            timings.offload += time.perf_counter() - offload_begin

    @staticmethod
    def _teacher_seq_ctx(seq_ctx: SequenceContext, *, is_composed: bool) -> SequenceContext:
        overrides = {
            "rollout_routed_experts": None,
            "offload_rollout_routed_experts": False,
            "dsa_topk_cache": DSATopKCacheState(),
        }
        if not is_composed:
            # A VLM tokenizer supplies 3D M-RoPE position ids for every sample,
            # including text-only samples. Plain language-model Teachers need
            # standard 2D packed positions. Passing ``None`` makes
            # SequenceContext rebuild them from the packed sequence lengths.
            # Visual fields are irrelevant to the text Teacher and may refer to
            # a pack routed to another Teacher, so do not retain them here.
            overrides.update(
                position_ids=None,
                image_grid_thw=None,
                deepstack_visual_embeds=None,
                visual_pos_masks=None,
                pixel_values=None,
                inputs_embeds=None,
                num_img_tokens=None,
            )
        return seq_ctx.copy(**overrides)

    def _compute_topk_targets(
        self,
        seq_ctx_list: list[SequenceContext],
        teacher_indices_list: list[torch.Tensor],
        timings: TrainTeacherTimings,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        top_k = cast(int, self.loss_config.top_k)
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
        for teacher_index, teacher in enumerate(self._teachers):
            with self._teacher_on_device(teacher, timings):
                # Every rank forwards every local pack before selecting routed
                # tokens so all ranks enter the same FSDP collective sequence.
                for batch_index, (seq_ctx, teacher_indices) in enumerate(zip(seq_ctx_list, teacher_indices_list)):
                    loss_ctx = topk_logprob_config.build(data={})
                    assert loss_ctx is not None
                    with torch.no_grad():
                        output = teacher(
                            seq_ctx=self._teacher_seq_ctx(
                                seq_ctx,
                                is_composed=self._teacher_is_composed[teacher_index],
                            ),
                            loss_ctx={"lm": loss_ctx},
                        )
                    selected = teacher_indices == teacher_index
                    target_ids[batch_index][selected] = cast(torch.Tensor, output.logits)[selected]
                    target_logprobs[batch_index][selected] = cast(torch.Tensor, output.loss)[selected]
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
        for teacher_index, teacher in enumerate(self._teachers):
            with self._teacher_on_device(teacher, timings):
                for batch_index, (seq_ctx, shifted_labels, teacher_indices) in enumerate(
                    zip(seq_ctx_list, shifted_labels_list, teacher_indices_list)
                ):
                    loss_ctx = cast(
                        LogProbContext,
                        self.logprob_config.build(data={"shifted_labels": shifted_labels}),
                    )
                    with torch.no_grad():
                        output = teacher(
                            seq_ctx=self._teacher_seq_ctx(
                                seq_ctx,
                                is_composed=self._teacher_is_composed[teacher_index],
                            ),
                            loss_ctx={"lm": loss_ctx},
                        )
                    selected = teacher_indices == teacher_index
                    target_logprobs[batch_index][selected] = cast(torch.Tensor, output.loss)[selected]
        return target_logprobs

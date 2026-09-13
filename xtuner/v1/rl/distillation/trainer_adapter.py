"""Trainer-side adapter for on-policy distillation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import torch

from xtuner.v1.data_proto.rl_data import RolloutState, TeacherTargets

from .config import DistillationConfig, RolloutTeacherScorerConfig, TrainTeacherManagerConfig


if TYPE_CHECKING:
    from xtuner.v1.rl.trainer.worker import WorkerLogItem


class DistillationTrainerAdapter:
    """Adapt distillation semantics to the Trainer integration points.

    The adapter is intentionally stateless.  It does not own Teacher or model
    lifecycle and does not keep cross-batch mutable state.  A ``None`` config
    is represented by the same object and produces the non-distillation
    defaults.
    """

    def __init__(self, config: DistillationConfig | None):
        self._config = config

    @property
    def rollout_teacher_scorer_config(self) -> RolloutTeacherScorerConfig | None:
        if self._config is None:
            return None
        return self._config.rollout_teacher_scorer_config

    @property
    def train_teacher_manager_config(self) -> TrainTeacherManagerConfig | None:
        if self._config is None:
            return None
        return self._config.train_teacher_manager_config

    @property
    def task_adv_weight(self) -> float:
        if self._config is None:
            return 1.0
        return self._config.loss_config.task_adv_weight

    @property
    def teacher_index_by_data_source(self) -> Mapping[str, int] | None:
        if self._config is None or not self._config.train_teachers:
            return None
        return self._config.teacher_index_by_data_source

    def rollout_teacher_targets(
        self,
        state: RolloutState,
        *,
        shifted_labels: Sequence[int],
        target_start: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return aligned rollout Teacher targets, or no fields when
        disabled."""

        if self._config is None or not self._config.rollout_teachers:
            return None, None
        if state.teacher_targets is None:
            raise ValueError(f"Rollout state has no Teacher targets: rollout_id={state.rollout_id}")
        expected_kind = "sampled" if self._config.loss_config.uses_sampled_token_targets else "topk"
        if state.teacher_targets.kind != expected_kind:
            raise ValueError(
                f"Expected {expected_kind} Teacher targets, got {state.teacher_targets.kind}: "
                f"rollout_id={state.rollout_id}"
            )
        return self._align_rollout_teacher_targets(
            state.teacher_targets,
            shifted_labels=shifted_labels,
            target_start=target_start,
            ignore_idx=self._config.loss_config.ignore_idx,
        )

    @staticmethod
    def _align_rollout_teacher_targets(
        targets: TeacherTargets,
        *,
        shifted_labels: Sequence[int],
        target_start: int,
        ignore_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        sequence_length = len(shifted_labels)
        if target_start < 0 or target_start > sequence_length:
            raise ValueError(
                f"Teacher target_start must be within the shifted sequence: {target_start} vs {sequence_length}"
            )
        if any(label != ignore_idx for label in shifted_labels[:target_start]):
            raise ValueError("Teacher target prefix must contain only ignored labels")

        expected_rows = sequence_length - target_start
        if len(targets.logprobs) != expected_rows:
            raise ValueError(
                "Teacher logprobs must align with the target suffix: "
                f"expected {expected_rows} rows, got {len(targets.logprobs)}"
            )

        if targets.kind == "sampled":
            sampled_logprobs = cast(list[float], targets.logprobs)
            return (
                torch.tensor([0.0] * target_start + sampled_logprobs, dtype=torch.float32).unsqueeze(0),
                None,
            )

        topk_tokens = cast(list[list[int]], targets.tokens)
        topk_logprobs = cast(list[list[float]], targets.logprobs)
        top_k = len(topk_logprobs[0])
        prompt_tokens = [[0] * top_k for _ in range(target_start)]
        prompt_logprobs = [[0.0] * top_k for _ in range(target_start)]
        return (
            torch.tensor(prompt_logprobs + topk_logprobs, dtype=torch.float32).unsqueeze(0),
            torch.tensor(prompt_tokens + topk_tokens, dtype=torch.int64).unsqueeze(0),
        )

    def reward_scalars(
        self,
        observations: Sequence[tuple[RolloutState, float]],
    ) -> dict[str, float]:
        """Aggregate representative session rewards by their configured
        Teacher."""

        if self._config is None:
            return {}

        teacher_rewards: dict[str, list[float]] = {}
        for representative, reward in observations:
            data_source = representative.extra_fields.get("origin_data_source")
            if not isinstance(data_source, str):
                continue
            teacher_name = self._config.data_source_teacher_map.get(data_source)
            if teacher_name is not None:
                teacher_rewards.setdefault(teacher_name, []).append(float(reward))

        return {
            f"rewards/{teacher_name}/mean": float(torch.tensor(rewards, dtype=torch.float32).mean().item())
            for teacher_name, rewards in teacher_rewards.items()
            if rewards
        }

    def timing_scalars(self, worker_log_items: Sequence[WorkerLogItem]) -> dict[str, float]:
        """Aggregate train Teacher timings across workers on the critical
        path."""

        phases = ("compute", "onload", "offload")
        teacher_timings_by_worker = [log_item.get("teacher_timings", {}) for log_item in worker_log_items]
        teacher_names = sorted(
            {teacher_name for teacher_timings in teacher_timings_by_worker for teacher_name in teacher_timings}
        )
        scalars: dict[str, float] = {}
        for teacher_name in teacher_names:
            for phase in phases:
                phase_values = [
                    float(teacher_timings[teacher_name][phase])
                    for teacher_timings in teacher_timings_by_worker
                    if teacher_name in teacher_timings
                ]
                scalars[f"time/train_teacher/{teacher_name}/{phase}"] = max(phase_values)

        # Training waits for the slowest worker, so report per-worker Teacher
        # totals and then take the maximum as the step-level critical-path time.
        for phase in phases:
            worker_totals = [
                sum(float(teacher_timing[phase]) for teacher_timing in teacher_timings.values())
                for teacher_timings in teacher_timings_by_worker
            ]
            if not teacher_names:
                continue
            total = max(worker_totals)
            scalars[f"time/train_teacher/total/{phase}"] = total
            scalars[f"time/train_teacher_{phase}"] = total
        return scalars

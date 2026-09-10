from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.loss.distillation_loss import DistillationLossConfig


if TYPE_CHECKING:
    from .rollout_teacher_manager import RolloutTeacherScorer
    from .train_teacher_manager import TrainTeacherManager


class RolloutTeacherLaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str | Path
    num_workers: int = Field(default=1, gt=0)
    server_port: int = Field(gt=0, le=65535)
    dtype: Literal["auto", "float16", "bfloat16"] = "bfloat16"
    tensor_parallel_size: int = Field(default=1, gt=0)
    expert_parallel_size: int = Field(default=1, gt=0)
    context_length: int | None = Field(default=None, gt=0)
    max_batch_size: int | None = Field(default=None, gt=0)
    log_level: Literal["critical", "error", "warning", "info", "debug"] | None = None
    chunked_prefill_size: int | None = Field(default=4096, gt=0)
    max_prefill_token_num: int | None = Field(default=4096, gt=0)
    gpu_memory_utilization: float = Field(default=0.6, gt=0.0, le=1.0)


class RolloutTeacherConfig(BaseModel):
    """Teacher served by an external rollout/inference engine."""

    model_config = ConfigDict(extra="forbid")

    name: str
    num_replicas: int = Field(default=1, gt=0)
    endpoints: list[str] = Field(default_factory=list)
    api_key: str | None = None
    request_timeout_s: float = Field(default=1200.0, gt=0.0)
    max_retry_per_sample: int = Field(default=2, ge=0)
    max_concurrency: int = Field(default=128, gt=0)
    enable_prefix_caching: bool = False
    launch_config: RolloutTeacherLaunchConfig | None = None


class TrainTeacherConfig(BaseModel):
    """Frozen Teacher loaded and executed by each training worker."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    model_path: str | Path
    model_cfg: TransformerConfig | BaseComposeConfig
    fsdp_cfg: FSDPConfig | None = None


TeacherConfig = RolloutTeacherConfig | TrainTeacherConfig


class TeacherTargetConfig(BaseModel):
    """Configuration for the output protocol requested from a Teacher."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["sampled_token", "topk"]
    top_k: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_target_config(self) -> TeacherTargetConfig:
        if self.mode == "topk" and self.top_k is None:
            raise ValueError("top_k is required for top-k Teacher targets")
        if self.mode == "sampled_token" and self.top_k is not None:
            raise ValueError("top_k is only valid for top-k Teacher targets")
        return self

    @property
    def uses_sampled_token_targets(self) -> bool:
        return self.mode == "sampled_token"

    @property
    def uses_topk_targets(self) -> bool:
        return self.mode == "topk"


class RolloutTeacherScorerConfig(BaseModel):
    """Serializable configuration used to build a rollout Teacher scorer."""

    model_config = ConfigDict(extra="forbid")

    teachers: list[RolloutTeacherConfig] = Field(min_length=1)
    data_source_teacher_map: dict[str, str] = Field(min_length=1)
    target_config: TeacherTargetConfig

    @model_validator(mode="after")
    def validate_config(self) -> RolloutTeacherScorerConfig:
        teacher_names_list = [teacher.name for teacher in self.teachers]
        if len(teacher_names_list) != len(set(teacher_names_list)):
            raise ValueError("Rollout Teacher names must be unique")
        teacher_names = set(teacher_names_list)
        unknown_teachers = set(self.data_source_teacher_map.values()) - teacher_names
        if unknown_teachers:
            raise ValueError(f"data_source_teacher_map references unknown teachers: {sorted(unknown_teachers)}")
        return self

    def build(self, *, defers_to_filter: bool) -> RolloutTeacherScorer:
        from .rollout_teacher_manager import RolloutTeacherScorer

        return RolloutTeacherScorer.from_config(self, defers_to_filter=defers_to_filter)


class TrainTeacherManagerConfig(BaseModel):
    """Serializable configuration used to build a training Teacher manager."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    teachers: list[TrainTeacherConfig] = Field(min_length=1)
    data_source_teacher_map: dict[str, str] = Field(min_length=1)
    target_config: TeacherTargetConfig

    @model_validator(mode="after")
    def validate_config(self) -> TrainTeacherManagerConfig:
        teacher_names_list = [teacher.name for teacher in self.teachers]
        if len(teacher_names_list) != len(set(teacher_names_list)):
            raise ValueError("Train Teacher names must be unique")
        teacher_names = set(teacher_names_list)
        unknown_teachers = set(self.data_source_teacher_map.values()) - teacher_names
        if unknown_teachers:
            raise ValueError(f"data_source_teacher_map references unknown teachers: {sorted(unknown_teachers)}")
        return self

    @property
    def teacher_index_by_data_source(self) -> dict[str, int]:
        teacher_index_by_name = {teacher.name: index for index, teacher in enumerate(self.teachers)}
        return {
            data_source: teacher_index_by_name[teacher_name]
            for data_source, teacher_name in self.data_source_teacher_map.items()
        }

    def build(self, *, chunk_size: int | None) -> TrainTeacherManager:
        from .train_teacher_manager import TrainTeacherManager

        return TrainTeacherManager(self, chunk_size=chunk_size)


class DistillationConfig(BaseModel):
    """Distillation objective, Teacher runtimes, and data-source routing."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    loss_config: DistillationLossConfig
    teachers: list[TeacherConfig] = Field(min_length=1)
    data_source_teacher_map: dict[str, str] = Field(min_length=1)
    _teacher_index_by_data_source: dict[str, int] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def validate_config(self) -> DistillationConfig:
        teacher_names = [teacher.name for teacher in self.teachers]
        if len(teacher_names) != len(set(teacher_names)):
            raise ValueError("Distillation teacher names must be unique")
        unknown_teachers = set(self.data_source_teacher_map.values()) - set(teacher_names)
        if unknown_teachers:
            raise ValueError(f"data_source_teacher_map references unknown teachers: {sorted(unknown_teachers)}")

        teacher_types = {type(teacher) for teacher in self.teachers}
        if len(teacher_types) != 1:
            raise ValueError(
                "Distillation requires all teachers to use the same runtime type; "
                "mixing RolloutTeacherConfig and TrainTeacherConfig is not supported"
            )
        teacher_index_by_name = {teacher.name: index for index, teacher in enumerate(self.teachers)}
        self._teacher_index_by_data_source = {
            data_source: teacher_index_by_name[teacher_name]
            for data_source, teacher_name in self.data_source_teacher_map.items()
        }
        return self

    @property
    def rollout_teachers(self) -> list[RolloutTeacherConfig]:
        return [teacher for teacher in self.teachers if isinstance(teacher, RolloutTeacherConfig)]

    @property
    def train_teachers(self) -> list[TrainTeacherConfig]:
        return [teacher for teacher in self.teachers if isinstance(teacher, TrainTeacherConfig)]

    @property
    def teacher_index_by_data_source(self) -> dict[str, int]:
        return self._teacher_index_by_data_source

    @property
    def teacher_target_config(self) -> TeacherTargetConfig:
        return TeacherTargetConfig(
            mode="topk" if self.loss_config.uses_topk_targets else "sampled_token",
            top_k=self.loss_config.top_k if self.loss_config.uses_topk_targets else None,
        )

    @property
    def rollout_teacher_scorer_config(self) -> RolloutTeacherScorerConfig | None:
        if not self.rollout_teachers:
            return None
        return RolloutTeacherScorerConfig(
            teachers=list(self.rollout_teachers),
            data_source_teacher_map=dict(self.data_source_teacher_map),
            target_config=self.teacher_target_config,
        )

    @property
    def train_teacher_manager_config(self) -> TrainTeacherManagerConfig | None:
        if not self.train_teachers:
            return None
        return TrainTeacherManagerConfig(
            teachers=list(self.train_teachers),
            data_source_teacher_map=dict(self.data_source_teacher_map),
            target_config=self.teacher_target_config,
        )

    def validate_student_model(self, student_model_cfg: TransformerConfig | BaseComposeConfig) -> None:
        """Validate contracts that require both Student and TrainTeacher
        configs."""
        if not self.train_teachers:
            return
        student_lm_cfg = (
            student_model_cfg.text_config if isinstance(student_model_cfg, BaseComposeConfig) else student_model_cfg
        )
        student_vocab_size = cast(TransformerConfig, student_lm_cfg).vocab_size
        for teacher in self.train_teachers:
            teacher_lm_cfg = (
                teacher.model_cfg.text_config
                if isinstance(teacher.model_cfg, BaseComposeConfig)
                else teacher.model_cfg
            )
            teacher_vocab_size = cast(TransformerConfig, teacher_lm_cfg).vocab_size
            if teacher_vocab_size != student_vocab_size:
                raise ValueError(
                    f"distillation teacher {teacher.name!r} vocab_size={teacher_vocab_size} does not match "
                    f"student vocab_size={student_vocab_size}"
                )

    def resolve_teacher_endpoints(
        self,
        endpoint_map: dict[str, list[str]],
    ) -> DistillationConfig:
        teachers = [
            teacher
            if not isinstance(teacher, RolloutTeacherConfig) or teacher.launch_config is None
            else teacher.model_copy(update={"endpoints": endpoint_map[teacher.name]})
            for teacher in self.teachers
        ]
        return self.model_copy(update={"teachers": teachers})

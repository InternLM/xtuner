from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MediaPipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    materialization: Literal["eager", "lazy"] = "eager"
    per_image: bool = False
    decode_threads: int = Field(default=1, ge=1, le=16)
    distribution: Literal["local", "sp_leader"] = "local"
    prefetch_batches: Literal[0, 1] = 0
    max_inflight_bytes: int | None = Field(default=None, gt=0)
    max_pending_tasks: int | None = Field(default=None, gt=0)
    max_transport_bytes: int | None = Field(default=None, gt=0)
    max_tensor_fields: int = Field(default=32, ge=1, le=128)
    max_tensor_rank: int = Field(default=8, ge=1, le=16)
    max_header_bytes: int = Field(default=65536, ge=1024, le=1048576)
    materialization_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    shutdown_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    unsupported_policy: Literal["error", "eager"] = "error"
    source_manifest: str | None = None

    @model_validator(mode="after")
    def validate_combination(self):
        requested = (
            self.materialization != "eager"
            or self.per_image
            or self.decode_threads != 1
            or self.distribution != "local"
            or self.prefetch_batches != 0
        )
        if not self.enabled:
            if requested:
                raise ValueError("media features require media_pipeline.enabled=True")
            return self
        if self.max_inflight_bytes is None or self.max_pending_tasks is None:
            raise ValueError("enabled media pipeline requires explicit byte and task limits")
        if self.max_pending_tasks < self.decode_threads:
            raise ValueError("max_pending_tasks must cover decode_threads")
        if (self.distribution == "sp_leader" or self.prefetch_batches) and self.materialization != "lazy":
            raise ValueError("SP distribution and prefetch require lazy materialization")
        if self.distribution == "sp_leader" and self.max_transport_bytes is None:
            raise ValueError("SP distribution requires an explicit per-rank transport byte limit")
        if self.materialization == "lazy" and not self.source_manifest:
            raise ValueError("lazy media requires a frozen source/geometry/decode-eligibility manifest")
        return self

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Mapping


if TYPE_CHECKING:
    import torch


DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def validate_json_object(value: str) -> None:
    if not isinstance(value, str) or not isinstance(json.loads(value), dict):
        raise ValueError("metadata/config must be a serialized JSON object")
    canonical_json(json.loads(value))


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    concat_dim: int | None = 0

    def __post_init__(self):
        if not self.name or self.dtype not in DTYPE_BYTES:
            raise ValueError("invalid tensor name or dtype")
        if not isinstance(self.shape, tuple) or any(type(size) is not int or size < 0 for size in self.shape):
            raise ValueError("tensor shape must contain nonnegative integer dimensions")
        if self.concat_dim is not None and not 0 <= self.concat_dim < len(self.shape):
            raise ValueError("concat_dim must identify an existing dimension")
        if self.nbytes > 2**63 - 1:
            raise ValueError("tensor byte count exceeds signed 64-bit range")

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * DTYPE_BYTES[self.dtype]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TensorSpec":
        return cls(**{**value, "shape": tuple(value["shape"])})


@dataclass(frozen=True)
class MediaSource:
    uri: str
    revision: str
    width: int
    height: int
    orientation: int = 1
    source_bytes: int = 0
    content_sha256: str | None = None
    metadata_json: str = "{}"

    def __post_init__(self):
        if not self.uri or not self.revision:
            raise ValueError("media identity requires URI and immutable source revision")
        if min(self.width, self.height) <= 0 or self.source_bytes < 0 or not 1 <= self.orientation <= 8:
            raise ValueError("invalid effective media geometry or source size")
        if self.content_sha256 is not None:
            if len(self.content_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in self.content_sha256
            ):
                raise ValueError("content_sha256 must be a lowercase SHA256")
        validate_json_object(self.metadata_json)


@dataclass(frozen=True)
class MediaEntry:
    occurrence_id: str
    sources: tuple[MediaSource, ...]
    output_specs: tuple[TensorSpec, ...]
    estimated_peak_bytes: int
    metadata_json: str = "{}"

    def __post_init__(self):
        if not self.occurrence_id or not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("a semantic media unit needs an occurrence and source tuple")
        if not all(isinstance(source, MediaSource) for source in self.sources):
            raise TypeError("media sources must be data-only MediaSource descriptors")
        if not isinstance(self.output_specs, tuple) or not all(
            isinstance(spec, TensorSpec) for spec in self.output_specs
        ):
            raise TypeError("media output specs must be an immutable TensorSpec tuple")
        if len({spec.name for spec in self.output_specs}) != len(self.output_specs):
            raise ValueError("duplicate output field")
        if self.estimated_peak_bytes < sum(spec.nbytes for spec in self.output_specs):
            raise ValueError("peak reservation cannot be smaller than final unit tensors")
        validate_json_object(self.metadata_json)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MediaEntry":
        return cls(
            **{
                **value,
                "sources": tuple(MediaSource(**source) for source in value["sources"]),
                "output_specs": tuple(TensorSpec.from_dict(spec) for spec in value["output_specs"]),
            }
        )


@dataclass(frozen=True)
class MediaPlan:
    provider: str
    provider_version: str
    processor_config_json: str
    processor_fingerprint: str
    source_manifest_fingerprint: str
    entries: tuple[MediaEntry, ...]
    output_specs: tuple[TensorSpec, ...]
    sample_ids: tuple[str, ...] = ()
    decisions_json: str = "{}"
    protocol_version: int = 1

    def __post_init__(self):
        if self.protocol_version != 1 or not all(
            (self.provider, self.provider_version, self.processor_fingerprint, self.source_manifest_fingerprint)
        ):
            raise ValueError("media plan needs supported protocol and frozen provider/processor/source identities")
        if not isinstance(self.entries, tuple) or not all(isinstance(entry, MediaEntry) for entry in self.entries):
            raise TypeError("plan entries must be an immutable MediaEntry tuple")
        if not isinstance(self.output_specs, tuple) or not all(
            isinstance(spec, TensorSpec) for spec in self.output_specs
        ):
            raise TypeError("plan output specs must be an immutable TensorSpec tuple")
        if not isinstance(self.sample_ids, tuple) or not all(isinstance(sample, str) for sample in self.sample_ids):
            raise TypeError("sample_ids must be an immutable string tuple")
        if len({spec.name for spec in self.output_specs}) != len(self.output_specs):
            raise ValueError("duplicate plan output field")
        validate_json_object(self.processor_config_json)
        validate_json_object(self.decisions_json)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @property
    def output_bytes(self) -> int:
        return sum(spec.nbytes for spec in self.output_specs)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MediaPlan":
        return cls(
            **{
                **value,
                "entries": tuple(MediaEntry.from_dict(entry) for entry in value["entries"]),
                "output_specs": tuple(TensorSpec.from_dict(spec) for spec in value["output_specs"]),
                "sample_ids": tuple(value.get("sample_ids", ())),
            }
        )

    @classmethod
    def from_json(cls, value: str) -> "MediaPlan":
        return cls.from_dict(json.loads(value))


@dataclass
class MaterializedMedia:
    tensors: dict[str, "torch.Tensor"]
    metrics: dict[str, int | float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors.values())

    @property
    def storage_bytes(self) -> int:
        storages = {}
        for tensor in self.tensors.values():
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        return sum(storages.values())

    def validate(self, specs: tuple[TensorSpec, ...]) -> None:
        if set(self.tensors) != {spec.name for spec in specs}:
            raise ValueError("materialized field presence differs from plan")
        for spec in specs:
            tensor = self.tensors[spec.name]
            if tuple(tensor.shape) != spec.shape or str(tensor.dtype).removeprefix("torch.") != spec.dtype:
                raise ValueError(f"materialized shape/dtype differs from plan for {spec.name}")


@dataclass(frozen=True)
class MediaDeliveryProof:
    consumption_id: str
    plan_fingerprint: str
    output_specs: tuple[TensorSpec, ...]
    mode: str = "materialized"

    def __post_init__(self):
        if not self.consumption_id or not self.plan_fingerprint or self.mode not in ("materialized", "empty", "eager"):
            raise ValueError("invalid media delivery proof")
        if not isinstance(self.output_specs, tuple) or not all(
            isinstance(spec, TensorSpec) for spec in self.output_specs
        ):
            raise TypeError("delivery proof specs must be immutable")

    def validate_context(self, context: Any) -> None:
        for spec in self.output_specs:
            tensor = getattr(context, spec.name, None)
            if tensor is None or tuple(tensor.shape) != spec.shape:
                raise ValueError(f"resolved media field missing or changed: {spec.name}")
            if str(tensor.dtype).removeprefix("torch.") != spec.dtype:
                raise ValueError(f"resolved media dtype changed: {spec.name}")

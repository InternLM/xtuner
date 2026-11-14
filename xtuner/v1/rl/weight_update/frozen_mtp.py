"""Content-bound, external Qwen3.5 MoE MTP weights for actor-only training.

This source owns only dedicated ``mtp.*`` tensors. Shared embeddings and the
output head continue to come from the updated actor through the native iterator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, Field
from safetensors import safe_open
from safetensors.torch import save_file

from .data import WeightUpdateBatch


SOURCE_RECORD = "frozen_mtp_source.json"
READY = "READY"
_SHARD = "frozen-mtp.safetensors"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class FrozenMTPConfig(BaseModel):
    """Explicit frozen source; unset means the original native lifecycle.

    ``manifest_path`` contains per-tensor shape, dtype, nbytes and SHA256 entries
    under ``tensors``. ``content_sha256`` pins the canonical JSON digest of that
    mapping, independent of filesystem paths and shard names.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_dir: Path
    manifest_path: Path
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def metadata(self) -> tuple[dict, dict, dict]:
        manifest = json.loads(self.manifest_path.read_text())
        tensors = manifest["tensors"]
        if not tensors or any(not key.startswith("mtp.") for key in tensors):
            raise ValueError("Frozen source manifest must contain dedicated mtp.* tensors only")
        if _digest(tensors) != self.content_sha256:
            raise ValueError("Frozen MTP manifest content identity changed")
        config_file = self.source_dir / "config.json"
        if hashlib.sha256(config_file.read_bytes()).hexdigest() != manifest["config_sha256"]:
            raise ValueError("Frozen source model config identity changed")
        config = json.loads(config_file.read_text())
        if config.get("model_type") != "qwen3_5_moe":
            raise ValueError("External frozen MTP currently supports Qwen3.5 MoE sources only")
        index = json.loads((self.source_dir / "model.safetensors.index.json").read_text())["weight_map"]
        if {key for key in index if key.startswith("mtp.")} != set(tensors):
            raise ValueError("Frozen source MTP keys do not match the bound manifest")
        for shard in sorted({index[key] for key in tensors}):
            shard_path = (self.source_dir / shard).resolve()
            if not shard_path.is_relative_to(self.source_dir.resolve()):
                raise ValueError("Frozen MTP shard must be inside source_dir")
            with safe_open(shard_path, framework="pt", device="cpu") as stream:
                for key, spec in tensors.items():
                    if index[key] != shard:
                        continue
                    view = stream.get_slice(key)
                    # Shape/key checks read headers only; contents are verified
                    # again before actor construction, transfer or export.
                    if view.get_shape() != spec["shape"]:
                        raise ValueError(f"Frozen MTP shape changed: {key}")
        return tensors, config.get("text_config", config), index

    def validate_model(self, model_cfg: Any) -> None:
        text = getattr(model_cfg, "text_config", model_cfg)
        if not type(text).__name__.startswith("Qwen3_5_"):
            raise ValueError("External frozen MTP requires a Qwen3.5 MoE actor")
        if getattr(text, "mtp_config", None) is not None:
            raise ValueError("External frozen MTP requires actor mtp_config=None; MTP training is unsupported")
        _, source, _ = self.metadata()
        for actor_field, source_field in (
            ("hidden_size", "hidden_size"),
            ("vocab_size", "vocab_size"),
            ("n_routed_experts", "num_experts"),
            ("moe_intermediate_size", "moe_intermediate_size"),
        ):
            if getattr(text, actor_field, None) != source.get(source_field):
                raise ValueError(f"Frozen source and actor disagree on {actor_field}")
        for field in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            if getattr(text.attention, field, None) != source.get(field):
                raise ValueError(f"Frozen source and actor disagree on attention.{field}")

    def validate_rollout(self, rollout: Any) -> None:
        # Resolve using the same backend precedence as native weight updates.
        from .data import _resolve_rollout_backend

        if (
            rollout.rollout_backend != "lmdeploy"
            or _resolve_rollout_backend(rollout) != "pytorch"
            or rollout.tensor_parallel_size != 1
            or rollout.expert_parallel_size != 1
            or rollout.weight_transport_type not in ("ipc", "nccl")
        ):
            raise ValueError("External frozen MTP requires LMDeploy pytorch, TP1/EP1 and IPC/NCCL")
        extra = rollout.extra_rollout_config or {}
        length = extra.get("lmdeploy_speculative_num_draft_tokens")
        if (
            extra.get("lmdeploy_speculative_algorithm") != "qwen3_5_mtp"
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length < 1
        ):
            raise ValueError("External frozen MTP requires qwen3_5_mtp and a positive draft length")


class FrozenMTPSource:
    """Own one verified CPU snapshot; consumers receive independent tensors."""

    def __init__(self, config: FrozenMTPConfig):
        self.config = config
        self.specs, _, index = config.metadata()
        self._config_sha256 = hashlib.sha256((config.source_dir / "config.json").read_bytes()).hexdigest()
        self._tensors = {}
        for shard in sorted({index[key] for key in self.specs}):
            with safe_open(config.source_dir / shard, framework="pt", device="cpu") as stream:
                for key, spec in self.specs.items():
                    if index[key] != shard:
                        continue
                    tensor = stream.get_tensor(key).clone().contiguous()
                    actual = dict(
                        shape=list(tensor.shape),
                        dtype=str(tensor.dtype),
                        nbytes=tensor.numel() * tensor.element_size(),
                        sha256=hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy()).cast("B")).hexdigest(),
                    )
                    if actual != spec:
                        raise ValueError(f"Frozen MTP tensor content changed: {key}")
                    self._tensors[key] = tensor

    @property
    def identity(self) -> dict:
        return {
            "format_version": 1,
            "content_sha256": self.config.content_sha256,
            "source_config_sha256": self._config_sha256,
            "tensors": json.loads(json.dumps(self.specs)),
        }

    def iter_batches(self, device: str, bucket_bytes: int = 256 * 1024**2):
        if bucket_bytes <= 0:
            raise ValueError("Frozen MTP bucket_bytes must be positive")
        bucket, size, dtype = {}, 0, None
        for key in sorted(self._tensors):
            tensor = self._tensors[key]
            nbytes = tensor.numel() * tensor.element_size()
            if bucket and (dtype != tensor.dtype or size + nbytes > bucket_bytes):
                yield WeightUpdateBatch(bucket, train_enable_ep=False, finished=False)
                bucket, size = {}, 0
            bucket[key] = tensor.to(device=device, copy=True)
            size += nbytes
            dtype = tensor.dtype
        if bucket:
            yield WeightUpdateBatch(bucket, train_enable_ep=False, finished=False)

    @staticmethod
    def _rank_zero(rank: int, operation: Callable[[], None]) -> None:
        result: list[str | None] = [None]
        if rank == 0:
            try:
                operation()
            except Exception as exc:
                result[0] = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            dist.broadcast_object_list(result, src=0)
        if result[0] is not None:
            raise RuntimeError(f"Frozen MTP checkpoint operation failed: {result[0]}")

    def save_binding(self, directory: Path, *, rank: int | None = None) -> None:
        if rank is None:
            _atomic_json(directory / SOURCE_RECORD, self.identity)
        else:
            self._rank_zero(rank, lambda: _atomic_json(directory / SOURCE_RECORD, self.identity))

    @staticmethod
    def check_binding(directory: Path, source: FrozenMTPSource | None) -> None:
        path = directory / SOURCE_RECORD
        if source is None:
            if path.exists():
                raise ValueError("Checkpoint requires its external frozen MTP source")
        elif not path.is_file() or json.loads(path.read_text()) != source.identity:
            raise ValueError("DCP frozen MTP binding is missing or differs from the configured source")

    @staticmethod
    def check_export_destination(directory: Path) -> None:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Frozen-MTP HF export requires an empty actor directory; isolate partial exports")

    def export_hf(self, directory: Path, export_actor: Callable[[], None], *, rank: int) -> None:
        self._rank_zero(rank, lambda: self.check_export_destination(directory))
        try:
            export_actor()
        finally:
            # Native HF saving copies non-safetensors files from the loaded HF.
            # A warm-start may therefore copy an old READY into this new, owned
            # directory. Clear it even if native export fails after the copy.
            # Preflight above protects an already populated/valid destination.
            self._rank_zero(rank, lambda: (directory / READY).unlink(missing_ok=True))
        self._rank_zero(rank, lambda: self.complete_hf(directory))

    def complete_hf(self, directory: Path) -> None:
        index = directory / "model.safetensors.index.json"
        data = json.loads(index.read_text())
        target = directory / _SHARD
        if (
            (directory / READY).exists()
            or target.exists()
            or any(key.startswith("mtp.") for key in data["weight_map"])
        ):
            raise FileExistsError("HF export already contains frozen MTP artifacts; repeated export is unsupported")
        for shard in set(data["weight_map"].values()):
            if not (directory / shard).is_file():
                raise FileNotFoundError(f"Actor HF shard is missing: {shard}")
        temporary = target.with_suffix(".tmp")
        if temporary.exists():
            raise FileExistsError("Partial frozen MTP shard exists; isolate the failed export")
        save_file(self._tensors, temporary, metadata={"format": "pt"})
        temporary.replace(target)
        data["weight_map"].update({key: _SHARD for key in self._tensors})
        meta = data.setdefault("metadata", {})
        if "total_size" in meta:
            meta["total_size"] += sum(spec["nbytes"] for spec in self.specs.values())
        _atomic_json(index, data)
        self.save_binding(directory)
        # No marker is written if shard/index/binding completion raises.
        _atomic_json(directory / READY, self.identity)

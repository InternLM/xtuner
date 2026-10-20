import hashlib
import io
import json
import math
import os
import stat
import sys
import tempfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from .schema import MaterializedMedia, MediaSource, TensorSpec, canonical_json


_SCRATCH_BYTES = 32 * 1024**2
_CACHE_VERSION = 1


def _positive(value, name):
    if type(value) is not int or not 0 < value < sys.maxsize:
        raise ValueError(f"{name} must be an exact positive bounded integer")
    return value


def _read_source(uri, max_source_bytes, max_inflight_bytes):
    from .providers.source_io import UnsupportedSourceIO, require_local_source

    require_local_source(uri)
    descriptor = os.open(uri, os.O_RDONLY | os.O_NONBLOCK)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise UnsupportedSourceIO("manifest preparation requires a local regular file")
        size = _positive(metadata.st_size, "source byte count")
        if size > max_source_bytes or 4 * size + _SCRATCH_BYTES > max_inflight_bytes:
            raise ValueError("source exceeds manifest preparation admission")
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as stream:
            content = stream.read(size + 1)
    finally:
        os.close(descriptor)
    if len(content) != size:
        raise ValueError("source size changed during manifest preparation")
    return content


def _reservation(source_bytes, width, height, patches, patch_size, pixel_bytes):
    decoded_rgb = width * height * 3
    resized_rgb = patches * patch_size**2 * 3
    return 4 * source_bytes + 64 * decoded_rgb + 64 * resized_rgb + 8 * pixel_bytes + _SCRATCH_BYTES


def _validate_metadata(uri, metadata, identity, profile_key, digest, source_bytes, max_inflight_bytes):
    if (
        metadata.get("revision") != f"sha256:{digest}"
        or metadata.get("content_sha256") != digest
        or type(metadata.get("source_bytes")) is not int
        or metadata["source_bytes"] != source_bytes
        or metadata.get("decode_verified") is not True
        or metadata.get("eligible") is not True
        or set(metadata.get("profiles", {})) != {profile_key}
    ):
        raise ValueError("cached image identity or eligibility differs")
    width = _positive(metadata["width"], "image width")
    height = _positive(metadata["height"], "image height")
    profile = metadata["profiles"][profile_key]
    grid = profile["image_grid_thw"]
    config = identity["processor_config"]
    patch_size = config["patch_size"]
    temporal = config["temporal_patch_size"]
    merge = config["merge_size"]
    if (
        not isinstance(grid, list)
        or len(grid) != 3
        or any(type(size) is not int or size <= 0 for size in grid)
        or grid[0] != 1
        or grid[1] % merge
        or grid[2] % merge
    ):
        raise ValueError("native image grid is incompatible with this provider")
    shape = (math.prod(grid), 3 * temporal * patch_size**2)
    if profile["pixel_values"] != {"dtype": "float32", "shape": list(shape)}:
        raise ValueError("native image pixel schema differs")
    specs = (TensorSpec("pixel_values", "float32", shape), TensorSpec("image_grid_thw", "int64", (1, 3)))
    peak = _reservation(source_bytes, width, height, math.prod(grid), patch_size, specs[0].nbytes)
    if type(profile.get("estimated_peak_bytes")) is not int or profile["estimated_peak_bytes"] != peak:
        raise ValueError("cached image reservation differs")
    if peak > max_inflight_bytes:
        raise ValueError("image exceeds manifest preparation admission")
    certificate = {"uri": uri, **metadata}
    MediaSource(
        uri=uri,
        revision=metadata["revision"],
        width=width,
        height=height,
        orientation=metadata["orientation"],
        source_bytes=source_bytes,
        content_sha256=digest,
        metadata_json=canonical_json(certificate),
    )
    return certificate


def _measure(content, digest, preprocess, processor, identity, profile_key, max_inflight_bytes):
    import torch
    from PIL import Image

    from ..utils import apply_exif_orientation

    image = None
    try:
        with io.BytesIO(content) as buffer, Image.open(buffer) as original:
            width = _positive(original.width, "image width")
            height = _positive(original.height, "image height")
            if 4 * len(content) + 64 * width * height * 3 + _SCRATCH_BYTES > max_inflight_bytes:
                raise ValueError("image header exceeds decode admission")
            orientation = original.getexif().get(274, 1)
            if type(orientation) is not int or not 1 <= orientation <= 8:
                raise ValueError("unsupported native EXIF orientation")
            if orientation in (5, 6, 7, 8):
                width, height = height, width
            patches = _positive(
                processor.get_number_of_image_patches(height=height, width=width), "native patch count"
            )
            config = identity["processor_config"]
            shape = (patches, 3 * config["temporal_patch_size"] * config["patch_size"] ** 2)
            pixels_spec = TensorSpec("pixel_values", "float32", shape)
            peak = _reservation(len(content), width, height, patches, config["patch_size"], pixels_spec.nbytes)
            if peak > max_inflight_bytes:
                raise ValueError("native image output exceeds preparation admission before processing")
            image = original.convert("RGB")
        if image.getexif().get(274, 1) != orientation:
            raise ValueError("native RGB conversion changed EXIF orientation")
        oriented = apply_exif_orientation(image)
        if oriented is not image:
            image.close()
            image = oriented
        if image.size != (width, height):
            raise ValueError("native effective geometry differs from admitted header")
        processed = preprocess([image])
        tensors = {name: processed[name] for name in ("pixel_values", "image_grid_thw")}
        for tensor in tensors.values():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.layout != torch.strided
                or tensor.requires_grad
                or not tensor.is_contiguous()
                or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
            ):
                raise ValueError("native preparation requires compact, non-autograd CPU tensors")
        MaterializedMedia(tensors).validate((pixels_spec, TensorSpec("image_grid_thw", "int64", (1, 3))))
        return {
            "revision": f"sha256:{digest}",
            "content_sha256": digest,
            "source_bytes": len(content),
            "width": width,
            "height": height,
            "orientation": orientation,
            "decode_verified": True,
            "eligible": True,
            "eligibility_basis": "native RGB/EXIF and exact per-image processor output; record filtering stays native",
            "profiles": {
                profile_key: {
                    "image_grid_thw": tensors["image_grid_thw"][0].tolist(),
                    "pixel_values": {"dtype": "float32", "shape": list(shape)},
                    "estimated_peak_bytes": peak,
                }
            },
        }
    finally:
        if image is not None:
            image.close()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_image_manifest(
    sources: Iterable[str | os.PathLike],
    *,
    provider: str,
    processor,
    cache_dir: str | os.PathLike,
    max_source_bytes: int,
    max_inflight_bytes: int,
) -> Path:
    from .providers.glm_image import GlmImageProvider
    from .providers.qwen_image import QwenImageProvider
    from .providers.shared import image_processor_identity, image_profile_fingerprint

    if isinstance(sources, (str, bytes, os.PathLike)):
        raise TypeError("sources must be an iterable of paths, not one path")
    _positive(max_source_bytes, "max_source_bytes")
    _positive(max_inflight_bytes, "max_inflight_bytes")
    providers = {"glm_image": GlmImageProvider, "qwen_image": QwenImageProvider}
    if provider not in providers:
        raise ValueError("manifest preparation currently supports only glm_image and qwen_image")
    provider_type = providers[provider]
    provider_version = provider_type.version
    if not isinstance(provider_version, str) or not provider_version:
        raise ValueError("manifest preparation requires an explicit provider version")
    preprocess = partial(provider_type._preprocess, SimpleNamespace(processor=processor))
    identity = image_processor_identity(provider, processor)
    if not identity["processor_module"].startswith(provider_type.processor_module_prefix):
        raise ValueError("processor class is outside this provider family")
    for field in ("patch_size", "temporal_patch_size", "merge_size"):
        _positive(identity["processor_config"][field], field)
    profile_key = image_profile_fingerprint(identity)
    cache_dir = Path(cache_dir).absolute()
    certificates = {}
    source_order = []
    for source in sources:
        if image_processor_identity(provider, processor) != identity or provider_type.version != provider_version:
            raise ValueError("processor identity changed during manifest preparation")
        uri = os.fspath(source)
        content = _read_source(uri, max_source_bytes, max_inflight_bytes)
        digest = hashlib.sha256(content).hexdigest()
        key = hashlib.sha256(
            canonical_json(
                {
                    "cache_version": _CACHE_VERSION,
                    "provider_version": provider_version,
                    "processor_profile": profile_key,
                    "content_sha256": digest,
                    "source_bytes": len(content),
                }
            ).encode()
        ).hexdigest()
        cache_path = cache_dir / "entries" / f"{key}.json"
        cache_hit = cache_path.exists()
        if cache_hit:
            with cache_path.open("rb") as stream:
                cached_bytes = stream.read(1024**2 + 1)
            if len(cached_bytes) > 1024**2:
                raise ValueError("cached image metadata exceeds its bound")
            cached = json.loads(cached_bytes)
            if not isinstance(cached, dict) or not isinstance(cached.get("metadata"), dict):
                raise ValueError("cached image metadata is corrupt")
            metadata = cached["metadata"]
            if (
                cached.get("key") != key
                or cached.get("metadata_sha256") != hashlib.sha256(canonical_json(metadata).encode()).hexdigest()
            ):
                raise ValueError("cached image metadata is corrupt")
        else:
            metadata = _measure(content, digest, preprocess, processor, identity, profile_key, max_inflight_bytes)
        certificate = _validate_metadata(
            uri, metadata, identity, profile_key, digest, len(content), max_inflight_bytes
        )
        if image_processor_identity(provider, processor) != identity or provider_type.version != provider_version:
            raise ValueError("processor identity changed during native image processing")
        if uri in certificates and certificates[uri] != certificate:
            raise ValueError("one URI changed contents during manifest preparation")
        if not cache_hit:
            _write_json(
                cache_path,
                {
                    "key": key,
                    "metadata_sha256": hashlib.sha256(canonical_json(metadata).encode()).hexdigest(),
                    "metadata": metadata,
                },
            )
        certificates[uri] = certificate
        source_order.append(uri)
        del content
    if image_processor_identity(provider, processor) != identity or provider_type.version != provider_version:
        raise ValueError("processor identity changed before manifest publication")
    manifest = {
        "schema_version": 1,
        "provider": provider,
        "provider_version": provider_version,
        "processor_identity": identity,
        "source_order": source_order,
        "sources": list(certificates.values()),
    }
    name = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    output = cache_dir / "manifests" / f"{name}.json"
    _write_json(output, manifest)
    return output

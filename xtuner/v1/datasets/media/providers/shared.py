import hashlib
import io
import json
import math
from importlib import import_module
from pathlib import Path
from typing import Sequence

from ..provider import ProviderCapabilities, create_provider
from ..schema import MaterializedMedia, MediaEntry, MediaPlan, MediaSource, TensorSpec, canonical_json
from .source_io import UnsupportedSourceIO, read_verified_local_source, require_local_source


class UnsupportedMediaSource(ValueError):
    pass


def image_processor_identity(provider: str, processor) -> dict:
    import transformers

    return {
        "provider": provider,
        "processor_module": type(processor).__module__,
        "processor_class": type(processor).__name__,
        "processor_config": json.loads(processor.to_json_string()),
        "transformers_version": transformers.__version__,
    }


def image_profile_fingerprint(identity: dict) -> str:
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


def make_image_provider_config(provider: str, processor, source_manifest: str | None, oss_loader_config=None) -> str:
    if source_manifest is None:
        raise UnsupportedMediaSource("optimized images require a certified source_manifest")
    manifest_path = str(Path(source_manifest).resolve())
    return canonical_json(
        {
            "identity": image_processor_identity(provider, processor),
            "source_manifest": manifest_path,
            "source_manifest_fingerprint": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            "oss_loader_config": oss_loader_config,
        }
    )


def _combined_specs(spec_groups: Sequence[tuple[TensorSpec, ...]]) -> tuple[TensorSpec, ...]:
    if not spec_groups:
        raise ValueError("cannot assemble an empty image plan")
    combined = []
    if any(len(group) != len(spec_groups[0]) for group in spec_groups):
        raise ValueError("image tensor field count differs")
    for field_index, first in enumerate(spec_groups[0]):
        fields = [group[field_index] for group in spec_groups]
        if any(
            (field.name, field.dtype, field.shape[1:], field.concat_dim)
            != (first.name, first.dtype, first.shape[1:], 0)
            for field in fields
        ):
            raise ValueError("incompatible image tensor schemas")
        combined.append(
            TensorSpec(first.name, first.dtype, (sum(field.shape[0] for field in fields), *first.shape[1:]))
        )
    return tuple(combined)


class CertifiedImageProvider:
    capabilities = ProviderCapabilities(metadata_exact=True, independent_images=True)
    version = "1"
    name: str
    processor_module_prefix: str
    native_per_image = False

    def __init__(self, processor_config_json: str):
        self.config_json = canonical_json(json.loads(processor_config_json))
        self.config = json.loads(self.config_json)
        self.identity = self.config["identity"]
        if self.identity["provider"] != self.name:
            raise ValueError("image provider identity mismatch")
        if not self.identity["processor_module"].startswith(self.processor_module_prefix):
            raise ValueError("processor class is outside this provider family")
        self.profile_fingerprint = image_profile_fingerprint(self.identity)
        self.processor_fingerprint = hashlib.sha256(self.config_json.encode()).hexdigest()
        manifest_bytes = Path(self.config["source_manifest"]).read_bytes()
        self.manifest_fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
        if self.manifest_fingerprint != self.config["source_manifest_fingerprint"]:
            raise ValueError("source certificate manifest changed after freezing")
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("sources"), list):
            raise ValueError("unsupported image certificate schema")
        self.certificates = {}
        for certificate in manifest["sources"]:
            uri = certificate["uri"]
            if uri in self.certificates:
                raise ValueError("duplicate certificate URI")
            self.certificates[uri] = certificate
        self._processor = None
        self._oss_loader = None

    @property
    def processor(self):
        if self._processor is None:
            import transformers

            if transformers.__version__ != self.identity["transformers_version"]:
                raise ValueError("native processor version changed")
            processor_class = getattr(
                import_module(self.identity["processor_module"]), self.identity["processor_class"]
            )
            self._processor = processor_class.from_dict(self.identity["processor_config"])
            if image_processor_identity(self.name, self._processor) != self.identity:
                raise ValueError("native processor did not reconstruct the frozen configuration")
        return self._processor

    def sources_for_uris(self, uris: Sequence[str]) -> tuple[MediaSource, ...]:
        sources = []
        for uri in uris:
            certificate = self.sources_for_certificate(uri)
            source = MediaSource(
                uri=uri,
                revision=certificate["revision"],
                width=certificate["width"],
                height=certificate["height"],
                orientation=certificate["orientation"],
                source_bytes=certificate["source_bytes"],
                content_sha256=certificate["content_sha256"],
                metadata_json=canonical_json(certificate),
            )
            content_revision = f"sha256:{source.content_sha256}"
            object_revision = source.revision.startswith("object-version:") and len(source.revision) > len(
                "object-version:"
            )
            if (
                source.content_sha256 is None
                or source.source_bytes <= 0
                or not (source.revision == content_revision or object_revision)
            ):
                raise ValueError("image certificate needs content hash, byte size and immutable revision")
            self._source_specs(source)
            sources.append(source)
        return tuple(sources)

    def _profile(self, source: MediaSource) -> dict:
        expected = self.sources_for_certificate(source.uri)
        if canonical_json(expected) != source.metadata_json:
            raise ValueError("source descriptor differs from frozen certificate")
        if any(
            getattr(source, field) != expected[field]
            for field in ("revision", "width", "height", "orientation", "source_bytes", "content_sha256")
        ):
            raise ValueError("source identity/geometry differs from certificate")
        return expected["profiles"][self.profile_fingerprint]

    def sources_for_certificate(self, uri: str) -> dict:
        try:
            require_local_source(uri)
        except UnsupportedSourceIO as error:
            raise UnsupportedMediaSource(str(error)) from error
        certificate = self.certificates.get(uri)
        if (
            certificate is None
            or certificate.get("decode_verified") is not True
            or certificate.get("eligible") is not True
        ):
            raise UnsupportedMediaSource(f"uncertified image: {uri}")
        if self.profile_fingerprint not in certificate.get("profiles", {}):
            raise UnsupportedMediaSource(f"uncertified image processor profile: {uri}")
        return certificate

    def _source_specs(self, source: MediaSource) -> tuple[TensorSpec, ...]:
        profile = self._profile(source)
        grid = profile["image_grid_thw"]
        pixel = profile["pixel_values"]
        if len(grid) != 3 or any(type(dimension) is not int or dimension <= 0 for dimension in grid) or grid[0] != 1:
            raise ValueError("image certificate must contain exact positive [1,h,w] grid")
        shape = tuple(pixel["shape"])
        patch = self.identity["processor_config"]["patch_size"]
        temporal = self.identity["processor_config"]["temporal_patch_size"]
        merge = self.identity["processor_config"]["merge_size"]
        if shape != (math.prod(grid), 3 * temporal * patch**2) or grid[1] % merge or grid[2] % merge:
            raise ValueError("certified patch shape/grid disagrees with native processor configuration")
        specs = (TensorSpec("pixel_values", pixel["dtype"], shape), TensorSpec("image_grid_thw", "int64", (1, 3)))
        minimum_peak = sum(spec.nbytes for spec in specs) + source.source_bytes + source.width * source.height * 3
        if type(profile["estimated_peak_bytes"]) is not int or profile["estimated_peak_bytes"] < minimum_peak:
            raise ValueError("certificate peak reservation is smaller than decoded RGB and output")
        return specs

    def describe(
        self,
        sources: Sequence[MediaSource],
        *,
        occurrence_id: str,
        per_image: bool,
        source_manifest_fingerprint: str,
        decisions_json: str = "{}",
    ) -> MediaPlan:
        if source_manifest_fingerprint != self.manifest_fingerprint:
            raise ValueError("source manifest fingerprint mismatch")
        groups = [(source,) for source in sources] if per_image or self.native_per_image else [tuple(sources)]
        entries = []
        for group_index, group in enumerate(groups):
            if not group:
                raise ValueError("image describe needs at least one source")
            specs = _combined_specs([self._source_specs(source) for source in group])
            peaks = [self._profile(source)["estimated_peak_bytes"] for source in group]
            peak = sum(peaks) + sum(spec.nbytes for spec in specs)
            entries.append(
                MediaEntry(
                    occurrence_id=f"{occurrence_id}:{group_index}",
                    sources=tuple(group),
                    output_specs=specs,
                    estimated_peak_bytes=peak,
                    metadata_json=canonical_json(
                        {"image_grid_thw": [self._profile(source)["image_grid_thw"] for source in group]}
                    ),
                )
            )
        return MediaPlan(
            provider=self.name,
            provider_version=self.version,
            processor_config_json=self.config_json,
            processor_fingerprint=self.processor_fingerprint,
            source_manifest_fingerprint=self.manifest_fingerprint,
            entries=tuple(entries),
            output_specs=_combined_specs([entry.output_specs for entry in entries]),
            sample_ids=(occurrence_id,),
            decisions_json=decisions_json,
        )

    def _read_source(self, source: MediaSource):
        self._source_specs(source)
        try:
            source_bytes = read_verified_local_source(source.uri, source.source_bytes, source.content_sha256)
        except UnsupportedSourceIO as error:
            raise UnsupportedMediaSource(str(error)) from error
        from PIL import Image

        from ...utils import apply_exif_orientation

        with io.BytesIO(source_bytes) as buffer, Image.open(buffer) as original:
            image = original.convert("RGB")
        try:
            orientation = image.getexif().get(274, 1)
            if orientation != source.orientation:
                raise ValueError("image EXIF orientation differs from certificate")
            oriented = apply_exif_orientation(image)
            if oriented is not image:
                image.close()
                image = oriented
            if image.size != (source.width, source.height):
                raise ValueError("effective image geometry differs from certificate")
            return image
        except BaseException:
            image.close()
            raise

    def process(self, entry: MediaEntry) -> MaterializedMedia:
        expected_specs = _combined_specs([self._source_specs(source) for source in entry.sources])
        expected_peak = sum(self._profile(source)["estimated_peak_bytes"] for source in entry.sources)
        expected_peak += sum(spec.nbytes for spec in expected_specs)
        if expected_specs != entry.output_specs or entry.estimated_peak_bytes < expected_peak:
            raise ValueError("entry schema/reservation differs from certified processor output")
        images = []
        try:
            for source in entry.sources:
                images.append(self._read_source(source))
            processed = self._preprocess(images)
            result = MaterializedMedia(
                {name: processed[name] for name in ("pixel_values", "image_grid_thw")},
                metadata={"occurrence_id": entry.occurrence_id},
            )
            result.validate(entry.output_specs)
            if result.tensors["image_grid_thw"].tolist() != [
                self._profile(source)["image_grid_thw"] for source in entry.sources
            ]:
                raise ValueError("actual native grid differs from certified grid")
            return result
        finally:
            for image in images:
                image.close()

    def _validate_assembly(self, plan: MediaPlan, ordered_results: Sequence[MaterializedMedia]) -> None:
        self._validate_plan(plan)
        if len(ordered_results) != len(plan.entries):
            raise ValueError("materialized image count/order differs from plan")
        for entry, result in zip(plan.entries, ordered_results):
            if result.metadata.get("occurrence_id") != entry.occurrence_id:
                raise ValueError("materialized image occurrence order differs from plan")
            result.validate(entry.output_specs)
            if result.tensors["image_grid_thw"].tolist() != json.loads(entry.metadata_json)["image_grid_thw"]:
                raise ValueError("materialized image grid/order differs from plan")

    def allocate_assembly(self, plan: MediaPlan, ordered_results: Sequence[MaterializedMedia]) -> MaterializedMedia:
        """Opt into ordered axis-zero copying into fresh, compact CPU outputs.

        The materializer owns scheduling and joins; this hook performs no copies
        and must preserve the metadata and metrics of the provider's assembly.
        """
        import torch

        self._validate_assembly(plan, ordered_results)
        return MaterializedMedia(
            {
                spec.name: torch.empty(spec.shape, dtype=getattr(torch, spec.dtype), device="cpu")
                for spec in plan.output_specs
            }
        )

    def assemble(self, plan: MediaPlan, ordered_results: Sequence[MaterializedMedia]) -> MaterializedMedia:
        import torch

        self._validate_assembly(plan, ordered_results)
        result = MaterializedMedia(
            {
                spec.name: torch.cat([output.tensors[spec.name] for output in ordered_results], dim=0)
                for spec in plan.output_specs
            }
        )
        result.validate(plan.output_specs)
        return result

    def _validate_plan(self, plan: MediaPlan):
        if (
            plan.provider,
            plan.provider_version,
            plan.processor_config_json,
            plan.processor_fingerprint,
            plan.source_manifest_fingerprint,
        ) != (self.name, self.version, self.config_json, self.processor_fingerprint, self.manifest_fingerprint):
            raise ValueError("cannot mix image provider/configuration/manifest identities")
        if plan.output_specs != _combined_specs([entry.output_specs for entry in plan.entries]):
            raise ValueError("plan output schema differs from its ordered semantic units")

    def merge_plans(self, plans: Sequence[MediaPlan]) -> MediaPlan:
        if not plans:
            raise ValueError("cannot merge an empty image plan list")
        for plan in plans:
            self._validate_plan(plan)
        return MediaPlan(
            provider=self.name,
            provider_version=self.version,
            processor_config_json=self.config_json,
            processor_fingerprint=self.processor_fingerprint,
            source_manifest_fingerprint=self.manifest_fingerprint,
            entries=tuple(entry for plan in plans for entry in plan.entries),
            output_specs=_combined_specs([plan.output_specs for plan in plans]),
            sample_ids=tuple(sample_id for plan in plans for sample_id in plan.sample_ids),
            decisions_json=canonical_json({"merged_decisions": [json.loads(plan.decisions_json) for plan in plans]}),
        )


def _materialize_plans(plans, config):
    from ..materializer import MediaMaterializer

    materializer = MediaMaterializer(config)
    try:
        return [materializer.materialize(plan) for plan in plans]
    finally:
        materializer.close()


def initialize_image_provider(tokenize_fn, provider_name: str):
    config = getattr(tokenize_fn, "media_pipeline", None)
    if config is None or not config.enabled:
        return None
    try:
        provider_config = make_image_provider_config(
            provider_name,
            tokenize_fn.image_processor,
            config.source_manifest,
            getattr(tokenize_fn, "_media_oss_loader_config", None),
        )
    except UnsupportedMediaSource:
        if config.unsupported_policy == "eager":
            return None
        raise
    tokenize_fn._media_provider = create_provider(provider_name, provider_config)
    return tokenize_fn._media_provider


def prepare_image_inputs(tokenize_fn, data_item: dict, media_root: str, provider_name: str):
    config = getattr(tokenize_fn, "media_pipeline", None)
    if config is None or not config.enabled:
        return None
    from ...mllm_tokenize_fn.base_mllm_tokenize_fn import get_image_path

    try:
        if getattr(tokenize_fn, "_media_provider", None) is None:
            if initialize_image_provider(tokenize_fn, provider_name) is None:
                return None
        provider = tokenize_fn._media_provider
        if image_processor_identity(provider_name, tokenize_fn.image_processor) != provider.identity:
            raise ValueError("tokenizer image processor changed after freezing its certificate profile")
        uris = [get_image_path(image, media_root) for image in tokenize_fn._image_path]
        sources = provider.sources_for_uris(uris)
        occurrence_id = hashlib.sha256(
            canonical_json(
                {
                    "data_name": tokenize_fn.data_name,
                    "record": data_item,
                    "uris": uris,
                }
            ).encode()
        ).hexdigest()
        plan = provider.describe(
            sources,
            occurrence_id=occurrence_id,
            per_image=config.per_image,
            source_manifest_fingerprint=provider.manifest_fingerprint,
        )
    except UnsupportedMediaSource:
        if config.unsupported_policy == "eager":
            return None
        raise
    import torch

    grids = [grid for entry in plan.entries for grid in json.loads(entry.metadata_json)["image_grid_thw"]]
    grid_tensor = torch.tensor(grids, dtype=torch.long)
    if config.materialization == "lazy":
        return None, grid_tensor, plan
    materialized = _materialize_plans([plan], config)[0]
    return materialized.tensors["pixel_values"], materialized.tensors["image_grid_thw"], None


def check_video_policy(tokenize_fn):
    config = getattr(tokenize_fn, "media_pipeline", None)
    if config is not None and config.enabled and config.unsupported_policy != "eager":
        raise UnsupportedMediaSource("video is native-eager only; choose unsupported_policy='eager' before collation")


def resolve_mixed_media_items(items: Sequence[dict], config) -> list[dict]:
    planned = [item for item in items if item.get("media_plan") is not None]
    native = [
        item
        for item in items
        if item.get("media_plan") is None
        and any(item.get(field) is not None for field in ("pixel_values", "pixel_values_videos"))
    ]
    if not planned or not native:
        return list(items)
    if config is None or not config.enabled or config.unsupported_policy != "eager":
        raise UnsupportedMediaSource("mixed planned/native media pack must resolve before collation/broadcast")
    materialized = iter(_materialize_plans([item["media_plan"] for item in planned], config))
    resolved = []
    for item in items:
        copied = dict(item)
        if copied.pop("media_plan", None) is not None:
            copied.update(next(materialized).tensors)
        resolved.append(copied)
    return resolved

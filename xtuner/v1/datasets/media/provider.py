from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Protocol, Sequence

from .schema import MaterializedMedia, MediaEntry, MediaPlan, MediaSource


@dataclass(frozen=True)
class ProviderCapabilities:
    metadata_exact: bool = False
    independent_images: bool = False
    thread_safe: bool = False
    worker_local: bool = True
    supported_modalities: tuple[str, ...] = ("image",)
    native_error_policy: str = "requires_certified_sources"


class MediaProvider(Protocol):
    capabilities: ProviderCapabilities
    version: str

    def describe(
        self,
        sources: Sequence[MediaSource],
        *,
        occurrence_id: str,
        per_image: bool,
        source_manifest_fingerprint: str,
        decisions_json: str = "{}",
    ) -> MediaPlan: ...

    def process(self, entry: MediaEntry) -> MaterializedMedia: ...

    def assemble(self, plan: MediaPlan, ordered_results: Sequence[MaterializedMedia]) -> MaterializedMedia: ...

    def merge_plans(self, plans: Sequence[MediaPlan]) -> MediaPlan: ...


_REGISTRY: dict[str, Callable[[str], MediaProvider] | str] = {}


def register_provider(name: str, factory: Callable[[str], MediaProvider] | str) -> None:
    if not name or name in _REGISTRY:
        raise ValueError(f"duplicate or empty media provider registration: {name}")
    _REGISTRY[name] = factory


def create_provider(name: str, processor_config_json: str) -> MediaProvider:
    if name not in _REGISTRY:
        import_module("xtuner.v1.datasets.media.providers")
    try:
        factory = _REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unregistered media provider: {name}") from error
    if isinstance(factory, str):
        module_name, attribute = factory.split(":", 1)
        factory = getattr(import_module(module_name), attribute)
    return factory(processor_config_json)

from __future__ import annotations

from typing import Any, Protocol

from ..core.models import (
    BackendHealth,
    CanonicalGenerateRequest,
    CanonicalGenerateResponse,
    ModelCapabilities,
    ModelCard,
)


class GatewayBackend(Protocol):
    async def generate(self, request: CanonicalGenerateRequest) -> CanonicalGenerateResponse: ...

    async def health(self) -> BackendHealth: ...

    async def list_models(self) -> list[ModelCard]: ...

    async def get_capabilities(self) -> ModelCapabilities: ...

    async def cancel(self, request_id: str) -> dict[str, Any]: ...

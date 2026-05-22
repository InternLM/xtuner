from __future__ import annotations

from typing import Any

import requests


class RolloutWeightUpdateClient:
    def __init__(self, api_key: list[str] | str | None):
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def collective_rpc(self, url: str, payload: dict[str, Any]):
        return requests.post(f"{url}/collective_rpc", headers=self._headers(), json=payload)

    def update_weights(self, url: str, endpoint: str, payload: dict[str, Any]):
        return requests.post(f"{url}/{endpoint}", headers=self._headers(), json=payload)

    def update_weights_from_tensor(self, url: str, payload: dict[str, Any]):
        return requests.post(f"{url}/update_weights_from_tensor", json=payload or {})

    def init_weights_update_group(self, url: str, payload: dict[str, Any]):
        return requests.post(f"{url}/init_weights_update_group", json=payload)

    def update_weights_from_distributed(self, url: str, payload: dict[str, Any]):
        return requests.post(f"{url}/update_weights_from_distributed", json=payload)

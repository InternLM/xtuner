from __future__ import annotations

from dataclasses import dataclass

from xtuner.v1.rl.utils.misc import check_chat_completions, delete_from_routedapiproxy, register_to_routedapiproxy
from xtuner.v1.utils import get_logger

from .session_worker_selector import RolloutWorkerHandle, RolloutWorkerUrlSource
from ..worker import RolloutConfig


@dataclass
class ExternalRolloutHttpEntryConfig:
    """Control-plane config for an external routedapiproxy HTTP entry.

    This class does not proxy user requests. It registers rollout worker
    generation URLs into an externally managed router, then AgentLoop uses
    ``base_url`` as the HTTP generation entry.
    """

    # The external router URL used by AgentLoop for generation.
    base_url: str

    # Which per-worker URL should be registered into the external router.
    worker_url_source: RolloutWorkerUrlSource = "backend"

    # Existing routedapiproxy semantics: delete old model registration before
    # registering the current worker URLs.
    delete_existing: bool = True

    # Optional smoke checks. They are useful for debug rollout, but can be
    # disabled when registration happens before the external router is ready.
    check_worker_urls: bool = True
    check_base_url: bool = True


class ExternalRolloutHttpEntry:
    def __init__(
        self,
        worker_handles: list[RolloutWorkerHandle],
        rollout_config: RolloutConfig,
        config: ExternalRolloutHttpEntryConfig,
        *,
        log_dir: str | None = None,
    ) -> None:
        self.worker_handles = worker_handles
        self.rollout_config = rollout_config
        self.config = config
        self.logger = get_logger(log_dir=log_dir, tag="ExternalRolloutHttpEntry")
        self._registered_urls: list[str] = []

    def start(self) -> None:
        model_name = self.rollout_config.model_name
        if self.config.delete_existing:
            delete_from_routedapiproxy(model_name)
            self.logger.info(f"Deleted existing routedapiproxy registrations for model {model_name}.")

        self.logger.info("Registering rollout worker URLs to routedapiproxy.")
        for worker in sorted(self.worker_handles, key=lambda item: item.rank):
            worker_url = worker.get_generate_url(self.config.worker_url_source)
            register_to_routedapiproxy(model_name, worker_url)
            self._registered_urls.append(worker_url)
            self.logger.info(f"Registered rollout worker {worker.rank} to routedapiproxy: {worker_url}")

            if self.config.check_worker_urls and not check_chat_completions(worker_url, model_name):
                raise RuntimeError(f"check chat completions failed for rollout worker URL {worker_url}")

        if self.config.check_base_url and not check_chat_completions(self.config.base_url, model_name):
            raise RuntimeError(f"check chat completions failed for external router URL {self.config.base_url}")

        self.logger.info("Registered rollout worker URLs to routedapiproxy.")

    def stop(self) -> None:
        self._registered_urls.clear()

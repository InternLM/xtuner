import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from xtuner.v1.rl.utils.misc import check_chat_completions, delete_from_routedapiproxy, register_to_routedapiproxy
from xtuner.v1.utils import get_logger


if TYPE_CHECKING:
    from .worker import RolloutConfig


class RolloutProxyManager:
    _CHECK_ATTEMPTS = 5
    _CHECK_INTERVAL_SECONDS = 3.0

    def __init__(self, config: "RolloutConfig"):
        self._model_name = config.model_name
        self._registered_session_urls: set[str] = set()
        self._logger = get_logger(log_dir=config.worker_log_dir, tag="RolloutProxyManager")

    def replace_registered_session_urls(self, session_urls: Iterable[str]) -> None:
        model_name = self._get_model_name()
        delete_from_routedapiproxy(model_name)
        self._registered_session_urls.clear()
        self._logger.info(f"Deleted routed API proxy registrations for model {model_name}.")

        unique_session_urls = sorted(set(session_urls))
        for session_url in unique_session_urls:
            self.register_session_url(session_url, raise_on_error=True)
        self._logger.info(f"Registered {len(unique_session_urls)} rollout session servers to routed API proxy.")

    def register_session_url(self, session_url: str | None, *, raise_on_error: bool = False) -> None:
        if session_url is None or session_url in self._registered_session_urls:
            return
        model_name = self._get_model_name()
        registered_remotely = False
        try:
            register_to_routedapiproxy(model_name, session_url)
            registered_remotely = True
            self._validate_session_url(session_url)
            self._registered_session_urls.add(session_url)
            self._logger.info(f"Registered rollout session server to routed API proxy: {session_url}")
        except Exception:
            self._logger.exception(f"Failed to register rollout session server to routed API proxy: {session_url}")
            if registered_remotely:
                self._rollback_session_url_registration(model_name, session_url)
            if raise_on_error:
                raise

    def delete_session_url(self, session_url: str | None, *, raise_on_error: bool = False) -> None:
        if session_url is None or session_url not in self._registered_session_urls:
            return
        try:
            delete_from_routedapiproxy(self._get_model_name(), session_url)
            self._registered_session_urls.discard(session_url)
            self._logger.info(f"Deleted rollout session server from routed API proxy: {session_url}")
        except Exception:
            self._logger.exception(f"Failed to delete rollout session server from routed API proxy: {session_url}")
            if raise_on_error:
                raise

    def _get_model_name(self) -> str:
        if self._model_name is None:
            raise ValueError("RolloutConfig.model_name must be set when enable_proxy=True.")
        return self._model_name

    def _validate_session_url(self, session_url: str) -> None:
        model_name = self._get_model_name()
        for attempt in range(1, self._CHECK_ATTEMPTS + 1):
            if check_chat_completions(session_url, model_name):
                return
            if attempt < self._CHECK_ATTEMPTS:
                self._logger.warning(
                    f"check chat completions failed for {session_url}, "
                    f"retrying {attempt}/{self._CHECK_ATTEMPTS - 1} after {self._CHECK_INTERVAL_SECONDS}s"
                )
                time.sleep(self._CHECK_INTERVAL_SECONDS)
        raise RuntimeError(f"check chat completions failed for registered rollout session server: {session_url}")

    def _rollback_session_url_registration(self, model_name: str, session_url: str) -> None:
        try:
            delete_from_routedapiproxy(model_name, session_url)
        except Exception:
            self._logger.exception(f"Failed to rollback routed API proxy registration: {session_url}")

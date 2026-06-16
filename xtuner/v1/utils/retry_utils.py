import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
    wait_incrementing,
)


_ResultT = TypeVar("_ResultT")

XTUNER_ROLLOUT_RETRY_WAIT_SECONDS = 0.1
XTUNER_SANDBOX_CREATE_RETRY_WAIT_MULTIPLIER = 2.0
XTUNER_SANDBOX_CREATE_RETRY_WAIT_MIN_SECONDS = 2.0
XTUNER_SANDBOX_CREATE_RETRY_WAIT_MAX_SECONDS = 8.0
XTUNER_TRACE_STORE_RETRY_ATTEMPTS = 10
XTUNER_TRACE_STORE_RETRY_WAIT_START_SECONDS = 0.2
XTUNER_TRACE_STORE_RETRY_WAIT_INCREMENT_SECONDS = 0.2
XTUNER_TRACE_STORE_RETRY_WAIT_MAX_SECONDS = 2.0


def _retry_before_sleep_with_logging(
    *,
    operation: str,
    logger: Any | None = None,
    context: Mapping[str, Any] | None = None,
    before_retry: Callable[[RetryCallState], None] | None = None,
) -> Callable[[RetryCallState], None]:
    def _before_sleep(retry_state: RetryCallState) -> None:
        if logger is not None:
            assert retry_state.outcome is not None
            exception = retry_state.outcome.exception()
            next_action = getattr(retry_state, "next_action", None)
            next_sleep = getattr(next_action, "sleep", None)
            retry_in = f", retrying in {next_sleep:.2f}s" if next_sleep is not None else ", retrying"
            if exception is None:
                reason = f"result={retry_state.outcome.result()!r}"
            else:
                reason = f"{type(exception).__name__}: {exception}"
            context_suffix = ""
            if context:
                context_suffix = " " + " ".join(f"{key}={value}" for key, value in context.items())
            logger.warning(
                f"{operation} failed on attempt {retry_state.attempt_number}{context_suffix} with {reason}{retry_in}."
            )
        if before_retry is not None:
            before_retry(retry_state)

    return _before_sleep


def _build_async_exception_retryer(
    *,
    operation: str,
    attempts: int,
    retry_exceptions: type[BaseException] | tuple[type[BaseException], ...],
    wait_strategy: Any,
    logger: Any | None = None,
    context: Mapping[str, Any] | None = None,
    before_retry: Callable[[RetryCallState], None] | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_strategy,
        retry=retry_if_exception_type(retry_exceptions),
        before_sleep=_retry_before_sleep_with_logging(
            operation=operation,
            logger=logger,
            context=context,
            before_retry=before_retry,
        ),
        sleep=sleep,
        reraise=True,
    )


def retry_rollout_request(
    *,
    attempts: int,
    retry_exceptions: type[BaseException] | tuple[type[BaseException], ...],
    logger: Any | None = None,
    request_uid: Any | None = None,
    wait_seconds: float | None = None,
    before_retry: Callable[[RetryCallState], None] | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> AsyncRetrying:
    wait_seconds = wait_seconds if wait_seconds is not None else XTUNER_ROLLOUT_RETRY_WAIT_SECONDS
    context = {"uid": request_uid} if request_uid is not None else None
    return _build_async_exception_retryer(
        operation="rollout_request",
        attempts=attempts,
        retry_exceptions=retry_exceptions,
        wait_strategy=wait_fixed(wait_seconds),
        logger=logger,
        context=context,
        before_retry=before_retry,
        sleep=sleep,
    )


def retry_sandbox_acquire(
    *,
    attempts: int,
    unhealthy_exceptions: type[BaseException] | tuple[type[BaseException], ...],
    logger: Any | None = None,
    sandbox_name: str | None = None,
    create_wait_multiplier: float | None = None,
    create_wait_min_seconds: float | None = None,
    create_wait_max_seconds: float | None = None,
    before_retry: Callable[[RetryCallState], None] | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> AsyncRetrying:
    create_wait_multiplier = (
        create_wait_multiplier if create_wait_multiplier is not None else XTUNER_SANDBOX_CREATE_RETRY_WAIT_MULTIPLIER
    )
    create_wait_min_seconds = (
        create_wait_min_seconds
        if create_wait_min_seconds is not None
        else XTUNER_SANDBOX_CREATE_RETRY_WAIT_MIN_SECONDS
    )
    create_wait_max_seconds = (
        create_wait_max_seconds
        if create_wait_max_seconds is not None
        else XTUNER_SANDBOX_CREATE_RETRY_WAIT_MAX_SECONDS
    )
    context = {"sandbox": sandbox_name} if sandbox_name else None
    create_wait = wait_exponential(
        multiplier=create_wait_multiplier,
        min=create_wait_min_seconds,
        max=create_wait_max_seconds,
    )

    def _wait_strategy(retry_state: RetryCallState) -> float:
        assert retry_state.outcome is not None
        exception = retry_state.outcome.exception()
        if exception is not None and isinstance(exception, unhealthy_exceptions):
            return 0.0
        return create_wait(retry_state)

    return _build_async_exception_retryer(
        operation="sandbox_acquire",
        attempts=attempts,
        retry_exceptions=(Exception,),
        wait_strategy=_wait_strategy,
        logger=logger,
        context=context,
        before_retry=before_retry,
        sleep=sleep,
    )


def poll_sandbox_health(
    *,
    max_wait_seconds: float | None = None,
    stop_strategy: Any | None = None,
    wait_seconds: float,
    logger: Any | None = None,
    sandbox_name: str | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> AsyncRetrying:
    if stop_strategy is None:
        if max_wait_seconds is None:
            raise ValueError("max_wait_seconds or stop_strategy must be provided")
        stop_strategy = stop_after_delay(max_wait_seconds)
    elif max_wait_seconds is not None:
        raise ValueError("Only one of max_wait_seconds and stop_strategy may be provided")
    context = {"sandbox": sandbox_name} if sandbox_name else None
    return AsyncRetrying(
        stop=stop_strategy,
        wait=wait_fixed(wait_seconds),
        retry=retry_if_result(lambda healthy: not healthy),
        retry_error_callback=lambda retry_state: retry_state.outcome.result(),
        before_sleep=_retry_before_sleep_with_logging(
            operation="sandbox_health_poll",
            logger=logger,
            context=context,
        ),
        sleep=sleep,
    )


def retry_trace_store_bootstrap(
    *,
    attempts: int | None = None,
    wait_start_seconds: float = XTUNER_TRACE_STORE_RETRY_WAIT_START_SECONDS,
    wait_increment_seconds: float = XTUNER_TRACE_STORE_RETRY_WAIT_INCREMENT_SECONDS,
    wait_max_seconds: float | None = None,
    logger: Any | None = None,
    before_retry: Callable[[RetryCallState], None] | None = None,
    sleep: Callable[[float], Any] = time.sleep,
) -> Retrying:
    attempts = attempts if attempts is not None else XTUNER_TRACE_STORE_RETRY_ATTEMPTS
    wait_max_seconds = wait_max_seconds if wait_max_seconds is not None else XTUNER_TRACE_STORE_RETRY_WAIT_MAX_SECONDS
    return Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait_incrementing(
            start=wait_start_seconds,
            increment=wait_increment_seconds,
            max=wait_max_seconds,
        ),
        retry=retry_if_exception_type((ValueError,)),
        before_sleep=_retry_before_sleep_with_logging(
            operation="trace_store_bootstrap",
            logger=logger,
            before_retry=before_retry,
        ),
        sleep=sleep,
        reraise=True,
    )

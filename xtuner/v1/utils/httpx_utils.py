import copy
import traceback
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional

import httpx

from xtuner.v1.data_proto.rl_data import RLRolloutResponseItem


class HttpRequestErrorType(IntEnum):
    """An enumeration for HTTP status codes and client-side request errors.
    Inherits from IntEnum for direct integer comparison.

    Custom codes are used for client-side exceptions that do not have
    an HTTP status code.

    Example:
        if error_code == HttpRequestErrorType.BAD_REQUEST:
            print("Bad request from server!")
        elif error_code == HttpRequestErrorType.TIMEOUT_ERROR:
            print("Client-side request timed out!")
    """

    # --- Custom Codes for Client-Side and Unhandled Errors ---
    UNKNOWN_ERROR = -1
    TIMEOUT_ERROR = 0
    REQUEST_ERROR = 1
    # --- Standard HTTP Status Codes ---
    SUCCESS = 200
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    REQUEST_TIMEOUT = 408
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504

    @classmethod
    def from_exception(cls, e: Exception) -> "HttpRequestErrorType":
        """Factory method to determine the RequestErrorType from a given
        exception."""
        if isinstance(e, httpx.TimeoutException):
            return cls.TIMEOUT_ERROR

        if isinstance(e, httpx.HTTPStatusError):
            # Try to match the status code to an existing enum member.
            # If not found, it's an unknown HTTP error, but we can still categorize it.
            # For simplicity here, we'll just return the known ones or fall back.
            try:
                return cls(e.response.status_code)
            except ValueError:
                # The status code is not a defined member of our enum.
                # We can decide to return UNKNOWN_ERROR or handle it differently.
                return cls.UNKNOWN_ERROR

        if isinstance(e, httpx.RequestError):
            # This check comes after its subclass (TimeoutException)
            return cls.REQUEST_ERROR

        # For any other standard Python exception
        return cls.UNKNOWN_ERROR


@dataclass
class HttpRequestResult:
    response: Optional[httpx.Response] = None
    error_type: HttpRequestErrorType = HttpRequestErrorType.SUCCESS
    error_msg: Optional[str] = None

    exception: Optional[Exception] = field(default=None, repr=False)
    url: Optional[str] = field(default=None, repr=False)
    payload: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __post_init__(self):
        """Generate error_msg automatically if an exception is provided and
        error_msg is not already set."""
        # Only generate a message if one hasn't been provided already.
        if self.error_msg is None and self.error_type != HttpRequestErrorType.SUCCESS:
            log_payload = {}
            if self.payload is not None and "input_ids" in self.payload:
                log_payload = copy.deepcopy(self.payload)
                log_payload["input_ids"] = str(log_payload["input_ids"])

            default_messages = {
                HttpRequestErrorType.UNKNOWN_ERROR: f"An unknown error {self.exception} occurred, Traceback: {traceback.format_exc()}",
                HttpRequestErrorType.TIMEOUT_ERROR: "The request timed out.",
                HttpRequestErrorType.REQUEST_ERROR: f"A network request error occurred. ExceptionType: {type(self.exception)}",
                HttpRequestErrorType.BAD_REQUEST: f"Bad Request (400): The server could not process the request {log_payload}",
                HttpRequestErrorType.UNAUTHORIZED: "Unauthorized (401): Authentication failed or is required.",
                HttpRequestErrorType.FORBIDDEN: "Forbidden (403): Access is denied.",
                HttpRequestErrorType.NOT_FOUND: "Not Found (404): The resource was not found.",
                HttpRequestErrorType.REQUEST_TIMEOUT: f"Request Timeout (408): The server timed out waiting for the request {log_payload}.",
                HttpRequestErrorType.TOO_MANY_REQUESTS: "Too Many Requests (429): Rate limit exceeded.",
                HttpRequestErrorType.INTERNAL_SERVER_ERROR: f"Internal Server Error (500) {self.exception} occurred in {self.url}, Traceback: {traceback.format_exc()}",
                HttpRequestErrorType.BAD_GATEWAY: f"Bad Gateway (502) in {self.url}.",
                HttpRequestErrorType.SERVICE_UNAVAILABLE: f"Service Unavailable (503) in {self.url}.",
                HttpRequestErrorType.GATEWAY_TIMEOUT: f"Gateway Timeout (504) in {self.url}.",
            }

            # Get the message from the map, or provide a generic fallback.
            self.error_msg = default_messages.get(
                self.error_type, f"An error occurred with status code: {self.error_type.value}"
            )
            if self.error_type == HttpRequestErrorType.REQUEST_ERROR and self.exception:
                if hasattr(self.exception, "__cause__") and self.exception.__cause__:
                    self.error_msg += f" __cause__: {self.exception.__cause__}"

    @property
    def is_success(self) -> bool:
        return self.error_type == HttpRequestErrorType.SUCCESS

    @property
    def is_retryable(self) -> bool:
        return self.error_type in {
            HttpRequestErrorType.TIMEOUT_ERROR,
            HttpRequestErrorType.REQUEST_ERROR,
        }

    @property
    def is_unknown_error(self) -> bool:
        return self.error_type == HttpRequestErrorType.UNKNOWN_ERROR

    @property
    def is_client_error(self) -> bool:
        return 400 <= self.error_type < 500

    @property
    def is_server_error(self) -> bool:
        return 500 <= self.error_type < 600


def set_rollout_response_status(http_result: HttpRequestResult, response: RLRolloutResponseItem, server_url=None):
    if http_result.is_retryable:
        response.finish_reason = "failed"
    elif http_result.is_client_error:
        response.finish_reason = "skipped"
    elif http_result.is_server_error:
        response.finish_reason = "failed"
        if server_url:
            response.extra_info.update({"url": server_url})

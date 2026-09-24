"""The error codes of contract section 9 and the one body every error has."""

import json

STATUS = {
    "invalid_request": 400,
    "unsupported_field": 400,
    "limit_exceeded": 400,
    "context_limit": 400,
    "unauthorized": 401,
    "class_not_allowed": 403,
    "scope_not_allowed": 403,
    "forbidden": 403,
    "not_found": 404,
    "stale_boot": 409,
    "stale_generation": 409,
    "sleep_pending": 409,
    "body_too_large": 413,
    "queue_full": 429,
    "internal_error": 500,
    "starting": 503,
    "draining": 503,
    "drained": 503,
    "engine_unavailable": 503,
    "timeout": 504,
}


class ServiceError(Exception):
    """A refusal with a code of section 9. It carries the code only, never request data."""

    def __init__(self, code: str) -> None:
        if code not in STATUS:
            raise ValueError("unknown error code")
        super().__init__(code)
        self.code = code

    @property
    def status(self) -> int:
        return STATUS[self.code]


def error_body(code: str) -> bytes:
    return json.dumps({"error": {"code": code}}).encode()

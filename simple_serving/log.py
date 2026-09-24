"""Log rows: one JSON object per line on standard error, with named fields only.

Contract section 10: a row about a request may hold the time, route, key label, class, scope kind, status, error
code, token counts, the measurements, whether the request was cancelled, and the finish reason. Rows about the service
itself add its boot, status and ports. A field outside `FIELDS` is dropped. Records of other libraries keep their
logger, level and the class of an exception only, because their messages and tracebacks may quote a request. That
includes uvicorn's own: its messages are mostly fixed texts, but nothing checks that each of them is.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from typing import IO, Any

LOGGER = logging.getLogger("simple_serving")
FIELDS = frozenset({
    "event", "listener", "route", "method", "key", "class", "scope", "status", "code", "cancelled", "finish",
    "input_tokens", "output_tokens", "cached_tokens", "count_matches", "wait_ms", "first_token_ms", "total_ms",
    "exception", "boot_id", "service_status", "drain_generation", "context_tokens", "port",
})


@dataclass
class RequestRecord:
    """The row of one request, filled in as the request goes."""

    listener: str
    route: str | None = None  # only a route of section 3, never a path the client made up
    method: str | None = None
    key: str | None = None  # the key's label
    cls: str | None = None
    scope: str | None = None  # the scope kind
    status: int | None = None
    code: str | None = None
    cancelled: bool = False
    finish: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    count_matches: bool | None = None  # the engine's prompt_tokens equals the gateway's count
    wait_ms: int | None = None
    first_token_ms: int | None = None
    total_ms: int | None = None
    exception: str | None = None  # the class of an unexpected exception, never its text

    def emit(self) -> None:
        fields = {("class" if name == "cls" else name): value for name, value in asdict(self).items()
                  if value is not None}
        row("request", **fields)


def row(event: str, **fields: Any) -> None:
    LOGGER.info(event, extra={"fields": {"event": event, **fields}})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {"time": _timestamp(record.created)}
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            line.update((name, value) for name, value in fields.items() if name in FIELDS)
        else:
            line.update(logger=record.name, level=record.levelname)
        if record.exc_info and record.exc_info[1] is not None:
            line["exception"] = type(record.exc_info[1]).__name__
        return json.dumps(line, ensure_ascii=False)


def setup(stream: IO[str] | None = None) -> None:
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def loop_exception(loop: Any, context: dict[str, Any]) -> None:
    """The event loop's handler for errors no task retrieved: their text and repr may quote a request."""
    error = context.get("exception")
    LOGGER.error("loop", extra={"fields": {"event": "loop_error",
                                           "exception": type(error).__name__ if error else None}})


def _timestamp(created: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(created)) + f".{int(created % 1 * 1000):03d}Z"

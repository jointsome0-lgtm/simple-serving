"""Raw ASGI plumbing: reading a body under a limit, the gateway's own responses, and noticing a client that leaves.

An inference route owns `receive` and `send` completely. Only one consumer may read `receive`: first the body is read
from it, counting bytes, and then it is watched for the client's disconnect, which is the only way to learn that a
client left while nothing is being sent to it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from starlette.types import Receive, Scope, Send

from .errors import STATUS, ServiceError
from .log import RequestRecord

RECORD = "simple_serving.record"  # the scope key of the request's log row


class ClientGone(Exception):
    """The client closed the connection."""


def header_map(scope: Scope) -> dict[str, bytes]:
    """Request headers by lowercase name. A header sent more than once is joined with commas, as HTTP combines it."""
    headers: dict[str, bytes] = {}
    for name, value in scope["headers"]:
        key = name.decode("latin-1").lower()
        headers[key] = headers[key] + b"," + value if key in headers else value
    return headers


def json_response(status: int, payload: Any, headers: Iterable[tuple[bytes, bytes]] = ()) -> list[dict[str, Any]]:
    """The two ASGI messages of a complete JSON response."""
    body = json.dumps(payload, ensure_ascii=False).encode()
    start = {"type": "http.response.start", "status": status,
             "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                         *headers]}
    return [start, {"type": "http.response.body", "body": body, "more_body": False}]


class Exchange:
    """One request and its response, as the gateway's inference and control routes handle them."""

    def __init__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self._receive = receive
        self._send = send
        self.headers = header_map(scope)
        self.record: RequestRecord = scope[RECORD]
        self.started = False  # the status and headers are sent
        self.finished = False  # the terminal message is sent

    def header(self, name: str) -> str | None:
        value = self.headers.get(name)
        return None if value is None else value.decode("latin-1")

    async def read_body(self, limit: int) -> bytes:
        """The whole body, counted while it is read, chunked or not. Past `limit` it stops with 413."""
        length = self.headers.get("content-length", b"")
        if length.isdigit() and int(length) > limit:
            raise ServiceError("body_too_large")
        chunks, size = [], 0
        while True:
            message = await self._receive()
            if message["type"] == "http.disconnect":
                raise ClientGone
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > limit:
                raise ServiceError("body_too_large")
            chunks.append(chunk)
            if not message.get("more_body", False):
                return b"".join(chunks)

    async def disconnected(self) -> None:
        """Return once the client has left. Call it only after the body was read."""
        while (await self._receive())["type"] != "http.disconnect":
            pass

    async def send_json(self, status: int, payload: Any) -> None:
        self.started = self.finished = True  # set first: a send cut short by a cancellation is not repeated
        for message in json_response(status, payload):
            await self._send(message)

    async def send_error(self, code: str) -> None:
        """An error of section 9: a plain HTTP error before the stream started, the error event after it. Nothing is
        sent after the terminal message."""
        if self.finished:
            return
        self.record.code = code
        if self.started:
            await self.send_event({"error": {"code": code}}, last=True)
        else:
            await self.send_json(STATUS[code], {"error": {"code": code}})

    async def start_stream(self) -> None:
        await self._send({"type": "http.response.start", "status": 200,
                          "headers": [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache"),
                                      (b"x-accel-buffering", b"no")]})
        self.started = True

    async def send_event(self, data: dict[str, Any] | str, *, last: bool = False) -> None:
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        self.finished = last
        await self._send({"type": "http.response.body", "body": f"data: {text}\n\n".encode(), "more_body": not last})

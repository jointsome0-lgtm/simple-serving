"""The two listeners' applications (contract section 3).

The public listener serves inference, models and state. The control listener, on loopback, serves state, drain and
open. They are two applications on two sockets, so the gateway tells them apart by the listener that received a
request, never by the client's address. Every route is a raw ASGI endpoint (`routes.py`); FastAPI routes the requests
and answers the rest with the gateway's own handlers, so its default answers, which may echo input, never go out.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import routes
from .asgi import RECORD, Exchange, json_response
from .errors import STATUS
from .log import RequestRecord
from .service import Service

Handler = Callable[[Service, Exchange], Awaitable[None]]
PUBLIC_ROUTES: list[tuple[str, str, Handler]] = [
    ("/v1/chat/completions", "POST", routes.generation),
    ("/v1/chat/completions/input_tokens", "POST", routes.count),
    ("/v1/models", "GET", routes.models),
    ("/v1/state", "GET", routes.state),
]
CONTROL_ROUTES: list[tuple[str, str, Handler]] = [
    ("/v1/state", "GET", routes.state),
    ("/v1/control/drain", "POST", routes.drain),
    ("/v1/control/open", "POST", routes.open_),
]


def public_app(service: Service) -> Guard:
    return Guard(_application(service, PUBLIC_ROUTES), "public", PUBLIC_ROUTES,
                 max_connections=service.config.max_connections)


def control_app(service: Service) -> Guard:
    return Guard(_application(service, CONTROL_ROUTES), "control", CONTROL_ROUTES)


class Endpoint:
    """A route handled on raw ASGI. Starlette passes requests to a class instance as they are."""

    def __init__(self, service: Service, handler: Handler) -> None:
        self._service = service
        self._handler = handler

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._handler(self._service, Exchange(scope, receive, send))


class Guard:
    """The outermost layer of a listener: the log row of every request, the public listener's cap on open
    connections, and a last answer to an error that nothing else caught. No exception reaches the ASGI server,
    whose log would print its text."""

    def __init__(self, app: ASGIApp, listener: str, table: list[tuple[str, str, Handler]], *,
                 max_connections: int | None = None) -> None:
        self._app = app
        self._listener = listener
        self._paths = frozenset(path for path, _, _ in table)
        self._max_connections = max_connections
        self.open_connections: Callable[[], int] = lambda: 0  # set by the server that runs the listener

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # no lifespan and no websockets
            return
        record = RequestRecord(listener=self._listener,
                               route=scope["path"] if scope["path"] in self._paths else None,
                               method=scope["method"] if scope["method"] in ("GET", "POST") else "other")
        scope[RECORD] = record

        async def send_noting_status(message: Message) -> None:
            if message["type"] == "http.response.start":
                record.status = message["status"]
            await send(message)

        try:
            if self._max_connections is not None and self.open_connections() > self._max_connections:
                await self._refuse(record, send_noting_status, "queue_full", close=True)
            else:
                await self._app(scope, receive, send_noting_status)
        except Exception as error:
            record.exception = type(error).__name__
            if record.status is None:
                with contextlib.suppress(Exception):
                    await self._refuse(record, send_noting_status, "internal_error")
        finally:
            record.emit()

    @staticmethod
    async def _refuse(record: RequestRecord, send: Send, code: str, *, close: bool = False) -> None:
        record.code = code
        headers = [(b"connection", b"close")] if close else []
        for message in json_response(STATUS[code], {"error": {"code": code}}, headers):
            await send(message)


def _application(service: Service, table: list[tuple[str, str, Handler]]) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, redirect_slashes=False)
    for path, method, handler in table:
        app.router.routes.append(Route(path, Endpoint(service, handler), methods=[method]))
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected_error)
    return app


async def _http_error(request: Request, error: Exception) -> Response:
    status = error.status_code if isinstance(error, HTTPException) else 500
    # A method the route does not serve is outside section 3, like an unknown path.
    code = "not_found" if status in (404, 405) else "invalid_request" if status < 500 else "internal_error"
    return _error(request, code)


async def _validation_error(request: Request, error: Exception) -> Response:
    return _error(request, "invalid_request")


async def _unexpected_error(request: Request, error: Exception) -> Response:
    request.scope[RECORD].exception = type(error).__name__
    return _error(request, "internal_error")


def _error(request: Request, code: str) -> Response:
    request.scope[RECORD].code = code
    return JSONResponse({"error": {"code": code}}, status_code=STATUS[code])

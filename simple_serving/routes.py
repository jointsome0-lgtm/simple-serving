"""The routes of contract section 3. Each handles one request on raw ASGI, with the checks in the contract's order.

The router has already matched the route; each handler checks the key next. The inference routes then check the
service status, the class and the scope, and only then read the body, which is refused past its limit before it is
parsed as JSON. From the arrival to the acceptance, a refusal's send included, they have PREPARE_S in all.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from .asgi import ClientGone, Exchange
from .errors import ServiceError
from .inference import Count, Generation
from .service import Service
from .validation import check_boot, check_chat, check_open, parse_json

CONTROL_BODY_LIMIT = 4096  # a drain, a sleep or an open is a few dozen bytes
PREPARE_S = 30  # a fixed bound, not a setting (contract section 4)


async def generation(service: Service, exchange: Exchange) -> None:
    await _inference(service, exchange, Generation)


async def count(service: Service, exchange: Exchange) -> None:
    # The same body and headers as a generation; no class limits and no context (contract section 5).
    await _inference(service, exchange, Count)


async def _inference(service: Service, exchange: Exchange, kind: type[Generation | Count]) -> None:
    """Check a request, accept it and serve it. A request of ours keeps the service awake from its authorization to
    its end (section 8), so what comes before acceptance is bounded: a client that sends its body slowly, or does not
    read a refusal, is given up at PREPARE_S."""
    deadline = asyncio.get_running_loop().time() + PREPARE_S
    try:
        caller = service.authorize(exchange)
    except ServiceError as error:
        await _refuse(exchange, error.code, deadline)
        return
    with service.ours(caller):
        try:
            async with asyncio.timeout_at(deadline):
                body = await exchange.read_body(service.config.body_limit_bytes)
            request = check_chat(parse_json(body), service.config.alias)
            if kind is Generation:
                service.check_max_tokens(caller, request)
            work = kind(service, exchange, caller, request)
            service.accept(work)  # a drain that began while the body was read refuses it here
        except ServiceError as error:
            await _refuse(exchange, error.code, deadline)
        except TimeoutError:
            await _refuse(exchange, "timeout", deadline)
        except ClientGone:
            exchange.record.cancelled = True
        else:
            await work.serve()


async def _refuse(exchange: Exchange, code: str, deadline: float) -> None:
    """A refusal before acceptance. Past the deadline it goes out only if the connection takes it at once."""
    try:
        async with asyncio.timeout_at(deadline):
            await exchange.send_error(code)
    except TimeoutError:
        exchange.record.cancelled = True


async def models(service: Service, exchange: Exchange) -> None:
    try:
        service.authenticate(exchange)
        await exchange.send_json(200, service.models())
    except ServiceError as error:
        await exchange.send_error(error.code)


async def state(service: Service, exchange: Exchange) -> None:
    try:
        key = service.authenticate(exchange)
        await exchange.send_json(200, service.state(control=key.control))
    except ServiceError as error:
        await exchange.send_error(error.code)


async def drain(service: Service, exchange: Exchange) -> None:
    await _control(service, exchange, lambda body: (202, service.drain(check_boot(body))))


async def sleep(service: Service, exchange: Exchange) -> None:
    await _control(service, exchange, lambda body: (202, service.sleep(check_boot(body))))


async def open_(service: Service, exchange: Exchange) -> None:
    await _control(service, exchange, lambda body: (200, service.open(*check_open(body))))


async def _control(service: Service, exchange: Exchange, act: Callable[[Any], tuple[int, dict[str, Any]]]) -> None:
    """A control route: the control key, the body, then the action. The action takes effect before its answer is
    sent, so a client that loses the answer finds the action done when it asks again."""
    try:
        if not service.authenticate(exchange).control:
            raise ServiceError("forbidden")
        status, answer = act(parse_json(await exchange.read_body(CONTROL_BODY_LIMIT)))
        await exchange.send_json(status, answer)
    except ServiceError as error:
        await exchange.send_error(error.code)
    except ClientGone:
        exchange.record.cancelled = True

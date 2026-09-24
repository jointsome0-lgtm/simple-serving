"""The routes of contract section 3. Each handles one request on raw ASGI, with the checks in the contract's order.

The router has already matched the route; each handler checks the key next. The inference routes then check the
service status, the class and the scope, and only then read the body, which is refused past its limit before it is
parsed as JSON.
"""

from __future__ import annotations

from .asgi import ClientGone, Exchange
from .errors import ServiceError
from .inference import Count, Generation
from .service import Service
from .validation import check_chat, check_drain, check_open, parse_json

CONTROL_BODY_LIMIT = 4096  # a drain or an open is a few dozen bytes


async def generation(service: Service, exchange: Exchange) -> None:
    try:
        caller = service.authorize(exchange)
        request = check_chat(parse_json(await exchange.read_body(service.config.body_limit_bytes)),
                             service.config.alias)
        service.check_max_tokens(caller, request)
    except ServiceError as error:
        await exchange.send_error(error.code)
        return
    except ClientGone:
        exchange.record.cancelled = True
        return
    await Generation(service, exchange, caller, request).serve()


async def count(service: Service, exchange: Exchange) -> None:
    # The same body and headers as a generation; no class limits and no context (contract section 5).
    try:
        caller = service.authorize(exchange)
        request = check_chat(parse_json(await exchange.read_body(service.config.body_limit_bytes)),
                             service.config.alias)
    except ServiceError as error:
        await exchange.send_error(error.code)
        return
    except ClientGone:
        exchange.record.cancelled = True
        return
    await Count(service, exchange, caller, request).serve()


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
    try:
        _authorize_control(service, exchange)
        boot_id = check_drain(parse_json(await exchange.read_body(CONTROL_BODY_LIMIT)))
        await exchange.send_json(202, service.drain(boot_id))
    except ServiceError as error:
        await exchange.send_error(error.code)
    except ClientGone:
        exchange.record.cancelled = True


async def open_(service: Service, exchange: Exchange) -> None:
    try:
        _authorize_control(service, exchange)
        boot_id, drain_generation = check_open(parse_json(await exchange.read_body(CONTROL_BODY_LIMIT)))
        await exchange.send_json(200, service.open(boot_id, drain_generation))
    except ServiceError as error:
        await exchange.send_error(error.code)
    except ClientGone:
        exchange.record.cancelled = True


def _authorize_control(service: Service, exchange: Exchange) -> None:
    if not service.authenticate(exchange).control:
        raise ServiceError("forbidden")

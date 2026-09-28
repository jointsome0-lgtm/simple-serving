"""The picture lane's routes (docs/pictures.md, sections 3 to 6), each on raw ASGI with the contract's checks in its
order: the key, the status, the class and the scope, then the body under its limit, then the lane's own checks. A
job's stop, an upload's erasure and the engine's facts answer in every status."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from ..asgi import ClientGone, Exchange
from ..errors import ServiceError
from ..policy import CLASS_HEADER, SCOPE_HEADER, Caller, resolve
from ..routes import PREPARE_S, _refuse
from ..validation import parse_json
from . import graphs
from .comfy import ComfyError
from .jobs import Picture
from .service import PictureService
from .uploads import SUBFOLDER

PICTURE_FIELDS = frozenset({"graph", "copies", "require_socket"})


async def picture(service: PictureService, exchange: Exchange) -> None:
    """`POST /v1/pictures`: checked, accepted, then served as its stream. What comes before the acceptance is bounded
    by PREPARE_S, as for the text lane's requests."""
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
            work = _picture(service, exchange, caller, parse_json(body))
            service.accept(work)  # a drain that began while the body was read refuses it here
        except ServiceError as error:
            await _refuse(exchange, error.code, deadline)
        except TimeoutError:
            await _refuse(exchange, "timeout", deadline)
        except ClientGone:
            exchange.record.cancelled = True
        else:
            await work.serve()


def _picture(service: PictureService, exchange: Exchange, caller: Caller, body: Any) -> Picture:
    if not isinstance(body, dict):
        raise ServiceError("invalid_request")
    if not body.keys() <= PICTURE_FIELDS:
        raise ServiceError("unsupported_field")
    require_socket = body.get("require_socket", False)
    if "graph" not in body or not isinstance(require_socket, bool):
        raise ServiceError("invalid_request")
    config = service.config
    graph, copies, references = graphs.check(body["graph"], body.get("copies"), cls=caller.cls,
                                             reader_nodes=config.reader_nodes, max_nodes=config.max_nodes,
                                             owns=service.uploads.owner(caller))
    exchange.record.nodes, exchange.record.references = len(graph), references
    return Picture(service, exchange, caller, graph, copies, references, require_socket)


async def stop(service: PictureService, exchange: Exchange) -> None:
    """`POST /v1/pictures/{job}/stop`: a job of the caller's own key and scope, stopped as the gateway stops any, and
    whether that is confirmed. A job the gateway no longer knows is not found, which the client takes for stopped."""
    try:
        caller = _caller(service, exchange)
        job_id = exchange.path_params.get("job", "")
        job, unresolved = service.jobs.get(job_id), service.unresolved.get(job_id)
        if job is not None and job.owned_by(caller):
            with service.ours(caller):
                stopped = await job.stop_by_client()
        elif unresolved is not None and unresolved.owned_by(caller):
            stopped = False
        else:
            raise ServiceError("not_found")
        await exchange.send_json(200, {"stopped": stopped})
    except ServiceError as error:
        await exchange.send_error(error.code)


async def put_upload(service: PictureService, exchange: Exchange) -> None:
    """`PUT /v1/pictures/uploads/{name}`: a reader's reference into ComfyUI's temp directory (section 5)."""
    deadline = asyncio.get_running_loop().time() + PREPARE_S
    try:
        caller = service.authorize(exchange)
    except ServiceError as error:
        await _refuse(exchange, error.code, deadline)
        return
    with service.ours(caller):
        try:
            name = exchange.path_params.get("name", "")
            found, kind = graphs.UPLOAD_NAME.fullmatch(name), _content_type(exchange)
            if found is None or kind != graphs.UPLOAD_TYPES[found[1]]:
                raise ServiceError("invalid_request")
            async with asyncio.timeout_at(deadline):
                data = await exchange.read_body(service.config.upload_limit_bytes)
            if graphs.image_type(data) != kind:
                raise ServiceError("invalid_request")
            exchange.record.upload_bytes = len(data)
            service.check_serving()  # a drain that began while the body was read refuses it here
            await service.uploads.put(caller, name, data, kind, exchange.disconnected)
        except ServiceError as error:
            await _refuse(exchange, error.code, deadline)
        except TimeoutError:
            await _refuse(exchange, "timeout", deadline)
        except ClientGone:
            exchange.record.cancelled = True
        else:
            await exchange.send_json(200, {"name": f"{SUBFOLDER}/{name} [temp]"})


async def erase_upload(service: PictureService, exchange: Exchange) -> None:
    """`DELETE /v1/pictures/uploads/{name}`: the caller's own upload, overwritten with a blank."""
    try:
        caller = _caller(service, exchange)
        name = exchange.path_params.get("name", "")
        if not graphs.UPLOAD_NAME.fullmatch(name):
            raise ServiceError("not_found")
        with service.ours(caller):
            await service.uploads.erase(caller, name)
        await exchange.send_empty(204)
    except ServiceError as error:
        await exchange.send_error(error.code)


async def put_input(service: PictureService, exchange: Exchange) -> None:
    """`PUT /v1/pictures/inputs/{name}`: an experiment's reference into ComfyUI's input directory, for `internal`
    alone, under the name its stripped bytes give it (section 5)."""
    deadline = asyncio.get_running_loop().time() + PREPARE_S
    try:
        caller = service.authorize(exchange)
        if caller.cls != "internal":
            raise ServiceError("class_not_allowed")
    except ServiceError as error:
        await _refuse(exchange, error.code, deadline)
        return
    with service.ours(caller):
        try:
            name = exchange.path_params.get("name", "")
            if not graphs.INPUT_NAME.fullmatch(name) or _content_type(exchange) != "image/png":
                raise ServiceError("invalid_request")
            async with asyncio.timeout_at(deadline):
                data = await exchange.read_body(service.config.upload_limit_bytes)
            try:
                stripped = graphs.strip_png(data)
            except graphs.NotPng:
                raise ServiceError("invalid_request") from None
            if hashlib.sha256(stripped).hexdigest()[:16] != name[4:20]:
                raise ServiceError("invalid_request")
            exchange.record.upload_bytes = len(stripped)
            service.check_serving()
            try:
                await service.comfy.upload(name, stripped, "image/png", kind="input", subfolder="")
            except ComfyError:
                raise ServiceError("engine_unavailable") from None
        except ServiceError as error:
            await _refuse(exchange, error.code, deadline)
        except TimeoutError:
            await _refuse(exchange, "timeout", deadline)
        except ClientGone:
            exchange.record.cancelled = True
        else:
            await exchange.send_json(200, {"name": name})


async def engine(service: PictureService, exchange: Exchange) -> None:
    """`GET /v1/pictures/engine`: the engine's facts as the service last checked them (section 6)."""
    try:
        _caller(service, exchange)
        await exchange.send_json(200, service.engine_facts())
    except ServiceError as error:
        await exchange.send_error(error.code)


def _caller(service: PictureService, exchange: Exchange) -> Caller:
    """The key, the class and the scope of a request that is answered in every status."""
    key = service.authenticate(exchange)
    caller = resolve(key, exchange.header(CLASS_HEADER), exchange.header(SCOPE_HEADER))
    exchange.record.cls, exchange.record.scope = caller.cls, caller.scope_kind
    return caller


def _content_type(exchange: Exchange) -> str:
    return (exchange.header("content-type") or "").partition(";")[0].strip().lower()

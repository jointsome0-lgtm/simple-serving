"""A reader's references on the card (docs/pictures.md, section 5): files in ComfyUI's temp directory under the name
the client chose, each owned by the first scope that sends it, and erased on the client's word, on a failure, and ten
minutes after its last send.

ComfyUI has no route that deletes a file, so an erasure overwrites the file with a blank PNG of one pixel, as the bot
does today; the card's sweeper removes whatever is left in the folder after the same ten minutes. What the gateway
knows of a name, its scope and its state, it keeps in memory alone, and no name is logged.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..asgi import ClientGone
from ..errors import ServiceError
from ..policy import Caller
from .comfy import Comfy, ComfyError
from .config import PictureLimits

TTL_S = 600  # the sweeper's cap on the folder (card/pictures/image-sweeper.py)
SUBFOLDER = "bot-references"
# simple-story-chat's blank (local/picture-references.ts), a grey PNG of one pixel, written over a reference.
BLANK = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")


@dataclass(eq=False)
class Upload:
    """One name. `state`: "sending" until ComfyUI first answers for it, "live" once it holds the bytes, "stale" when
    it may hold them and the gateway has not erased them yet, "erased" once the blank is in their place. A send and an
    erasure of the name take turns under `lock`."""

    name: str
    scope: str
    expires: float  # the event loop's time
    state: str = "sending"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Uploads:
    def __init__(self, comfy: Comfy, limits: Mapping[str, PictureLimits],
                 track: Callable[[Coroutine[Any, Any, None]], None]) -> None:
        self._comfy = comfy
        self._limits = limits
        self._track = track  # runs an erasure that nobody waits for, which a stop of the service waits for
        self._uploads: dict[str, Upload] = {}

    def owner(self, caller: Caller) -> Callable[[str], bool]:
        """Whether a name is a live upload of the caller's scope, for the checks of a graph."""
        now = asyncio.get_running_loop().time()

        def owns(name: str) -> bool:
            upload = self._uploads.get(name)
            return upload is not None and upload.scope == caller.scope and upload.state == "live" \
                and now < upload.expires

        return owns

    async def put(self, caller: Caller, name: str, data: bytes, content_type: str,
                  gone: Callable[[], Awaitable[None]]) -> None:
        """Send the caller's upload and return once ComfyUI holds it. The name is the caller's scope's from here, before
        anything is sent. An upload that fails, or whose client leaves before the answer, is erased at once in the
        background, whatever the client does next."""
        upload = self._claim(caller, name)
        send = asyncio.ensure_future(self._send(upload, data, content_type))
        left = asyncio.ensure_future(gone())
        kept = False
        try:
            await asyncio.wait({send, left}, return_when=asyncio.FIRST_COMPLETED)
            if left.done():
                raise ClientGone
            send.result()  # the send's failure, if it failed
            kept = True
        finally:
            left.cancel()
            if left.done() and not left.cancelled():
                left.exception()
            if not kept:
                self._track(self._erase_after(upload, send))

    async def erase(self, caller: Caller, name: str) -> None:
        """The caller's own upload, erased. An erased name may be erased again until its ten minutes are over."""
        upload = self._uploads.get(name)
        if upload is None or upload.scope != caller.scope:
            raise ServiceError("not_found")
        if not await self._erase(upload):
            raise ServiceError("engine_unavailable")

    async def sweep(self) -> None:
        """Erase and forget the names whose ten minutes are over. One that cannot be erased is the sweeper's."""
        loop = asyncio.get_running_loop()
        for upload in [one for one in self._uploads.values() if one.expires <= loop.time()]:
            await self._erase(upload)
            if self._uploads.get(upload.name) is upload and upload.expires <= loop.time():
                del self._uploads[upload.name]

    def _claim(self, caller: Caller, name: str) -> Upload:
        """The name for the caller's scope: another scope's name is not found, and a scope past its class's cap of live
        uploads is refused. The same scope may send a name again, which starts its ten minutes again."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        assert caller.scope is not None  # the picture lane has no outside keys
        upload = self._uploads.get(name)
        if upload is not None and upload.expires <= now and not upload.lock.locked():
            del self._uploads[name]  # over its ten minutes: the sweep's, whoever asks now
            upload = None
        if upload is not None:
            if upload.scope != caller.scope:
                raise ServiceError("not_found")
            upload.expires = now + TTL_S
            return upload
        live = sum(1 for one in self._uploads.values()
                   if one.scope == caller.scope and one.state != "erased" and now < one.expires)
        if live >= self._limits[caller.cls].uploads:
            raise ServiceError("queue_full")
        upload = self._uploads[name] = Upload(name=name, scope=caller.scope, expires=now + TTL_S)
        return upload

    async def _send(self, upload: Upload, data: bytes, content_type: str) -> None:
        async with upload.lock:
            try:
                await self._comfy.upload(upload.name, data, content_type, kind="temp", subfolder=SUBFOLDER)
            except ComfyError:
                upload.state = "stale"
                raise ServiceError("engine_unavailable") from None
            upload.state = "live"

    async def _erase_after(self, upload: Upload, send: asyncio.Future[None]) -> None:
        await asyncio.wait({send})
        if not send.cancelled():
            send.exception()  # retrieved: the client has its answer already, or has gone
        await self._erase(upload)

    async def _erase(self, upload: Upload) -> bool:
        """Whether the name is erased: the blank written over it now, or before."""
        async with upload.lock:
            if upload.state == "erased":
                return True
            try:
                await self._comfy.upload(upload.name, BLANK, "image/png", kind="temp", subfolder=SUBFOLDER)
            except ComfyError:
                upload.state = "stale"
                return False
            upload.state = "erased"
            return True

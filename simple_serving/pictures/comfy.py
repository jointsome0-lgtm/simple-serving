"""A thin client of the pinned ComfyUI (v0.37.0 at 73c9bad4) on the card's loopback, and its websocket.

It uses POST /prompt, GET /queue, GET /history/{id}, POST /history (delete), POST /api/jobs/{id}/cancel, POST
/upload/image, GET /view, GET /system_stats, GET /object_info/{node} and GET /internal/logs/raw, and the websocket
/ws. Of an answer it reads the fields named here and nothing else: ComfyUI's errors quote a graph's inputs, and its
records and its log carry prompts. A failure is a `ComfyError` with a kind, never ComfyUI's text, and no line of the
log is kept past the count made of it.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

CONNECT_TIMEOUT_S = 5.0
REQUEST_TIMEOUT_S = 30.0  # one request on loopback, never a drawing: the drawing is followed, not waited on
SOCKET_MESSAGE_BYTES = 16 << 20  # far above a message of ComfyUI's; a preview frame is dropped unread
RECONNECT_S = 0.5
STOP_POLLS, STOP_POLL_S = 10, 0.5  # the confirmation of a stop (docs/pictures.md, section 4, "Stop")
MIB = 1024 * 1024
PARTIAL_LOAD = re.compile(r"loaded partially|Unloaded partially")  # simple-story-chat's `partialLoadsSince`
ATTENTION_NODE, KITCHEN_ATTENTION = "ModelAttentionBackend", "comfy kitchen attention"


class ComfyError(Exception):
    """ComfyUI did not answer as asked. `unreachable`: the connection was refused, so nothing was sent; `failed`: any
    other failure, a status or a body that does not read. It carries the kind only."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


class Rejected(ComfyError):
    """ComfyUI refused a graph with 400: it did not validate. Its answer, which quotes the graph, is not read."""

    def __init__(self) -> None:
        super().__init__("rejected")


@dataclass(frozen=True)
class Queue:
    """A read of ComfyUI's queue: the waiting jobs as (number, id), and the ids being drawn. Only the first two
    elements of an entry are read; the rest is the job's graph."""

    pending: tuple[tuple[float, str], ...]
    running: tuple[str, ...]

    def holds(self, prompt_id: str) -> bool:
        return prompt_id in self.running or any(one == prompt_id for _, one in self.pending)


class Comfy:
    """ComfyUI's HTTP routes."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._http = httpx.AsyncClient(base_url=base_url, trust_env=False, follow_redirects=False,
                                       timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, **options: Any) -> httpx.Response:
        try:
            return await self._http.request(method, path, **options)
        except httpx.ConnectError:
            raise ComfyError("unreachable") from None
        except httpx.HTTPError:
            raise ComfyError("failed") from None

    async def _json(self, method: str, path: str, **options: Any) -> Any:
        response = await self._request(method, path, **options)
        if response.status_code != 200:
            raise ComfyError("failed")
        try:
            return response.json()
        except ValueError:
            raise ComfyError("failed") from None

    async def system_stats(self) -> dict[str, Any]:
        stats = await self._json("GET", "/system_stats")
        if not isinstance(stats, dict):
            raise ComfyError("failed")
        return stats

    async def attention_offered(self) -> bool:
        """Whether the kitchen's attention is among ModelAttentionBackend's options, as simple-story-chat's
        `attentionOffered` reads them: `["COMBO", {"options": [...]}]` or a list of options first. A server without
        the node answers with nothing for it, which is no."""
        info = await self._json("GET", f"/object_info/{ATTENTION_NODE}")
        node = info.get(ATTENTION_NODE) if isinstance(info, dict) else None
        required = node.get("input", {}).get("required", {}) if isinstance(node, dict) else {}
        attention = required.get("attention") if isinstance(required, dict) else None
        options: Any = None
        if isinstance(attention, list) and attention:
            first = attention[0]
            second = attention[1] if len(attention) > 1 else None
            options = first if isinstance(first, list) else second.get("options") if isinstance(second, dict) else None
        return isinstance(options, list) and KITCHEN_ATTENTION in options

    async def submit(self, graph: dict[str, Any], *, prompt_id: str, number: int, client_id: str) -> None:
        """Queue a graph under an id and a number of ours (server.py:1081-1088): the lowest number is drawn first."""
        response = await self._request("POST", "/prompt", json={"prompt": graph, "client_id": client_id,
                                                                 "prompt_id": prompt_id, "number": number})
        if response.status_code == 400:
            raise Rejected
        if response.status_code != 200:
            raise ComfyError("failed")
        try:
            answer = response.json()
        except ValueError:
            raise ComfyError("failed") from None
        if not isinstance(answer, dict) or answer.get("prompt_id") != prompt_id:
            raise ComfyError("failed")

    async def queue(self) -> Queue:
        seen = await self._json("GET", "/queue")
        running, pending = (seen.get(name) if isinstance(seen, dict) else None
                            for name in ("queue_running", "queue_pending"))
        if not isinstance(running, list) or not isinstance(pending, list):
            raise ComfyError("failed")
        return Queue(
            pending=tuple((float(one[0]), one[1]) for one in pending
                          if isinstance(one, list) and len(one) > 1 and _number(one[0]) and isinstance(one[1], str)),
            running=tuple(one[1] for one in running if isinstance(one, list) and len(one) > 1
                          and isinstance(one[1], str)))

    async def history(self, prompt_id: str) -> dict[str, Any] | None:
        """The job's record, or None while it has none."""
        seen = await self._json("GET", f"/history/{prompt_id}")
        if not isinstance(seen, dict):
            raise ComfyError("failed")
        record = seen.get(prompt_id)
        return record if isinstance(record, dict) else None

    async def forget(self, prompt_id: str) -> None:
        """Delete the job's record, which holds its graph and names its pictures."""
        await self._json("POST", "/history", json={"delete": [prompt_id]})

    async def cancel(self, prompt_id: str) -> None:
        """Take the job out of the queue if it waits, or interrupt it if it is the one being drawn, atomically, and
        nothing else (server.py:947-988, execution.py `interrupt_if_running`)."""
        await self._json("POST", f"/api/jobs/{prompt_id}/cancel")

    async def view(self, image: dict[str, Any]) -> bytes:
        params = {"filename": str(image.get("filename", "")), "subfolder": str(image.get("subfolder") or ""),
                  "type": str(image.get("type") or "output")}
        response = await self._request("GET", "/view", params=params)
        if response.status_code != 200:
            raise ComfyError("failed")
        return response.content

    async def upload(self, name: str, data: bytes, content_type: str, *, kind: str, subfolder: str) -> None:
        """A file into ComfyUI's `kind` directory ("temp" or "input") under `name`, overwriting one of that name, and
        ComfyUI's answer held to it."""
        files = {"image": (name, data, content_type)}
        form = {"overwrite": "true", "type": kind, "subfolder": subfolder}
        answer = await self._json("POST", "/upload/image", files=files, data=form)
        if not isinstance(answer, dict) or (answer.get("name"), answer.get("subfolder") or "",
                                            answer.get("type")) != (name, subfolder, kind):
            raise ComfyError("failed")

    async def log_lines(self) -> list[str]:
        """ComfyUI's log ring as `time\\0message` lines, to be counted and dropped."""
        seen = await self._json("GET", "/internal/logs/raw")
        entries = seen.get("entries") if isinstance(seen, dict) else None
        if not isinstance(entries, list):
            raise ComfyError("failed")
        return [f"{entry.get('t')}\0{entry.get('m')}" for entry in entries if isinstance(entry, dict)]


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


async def stop_job(comfy: Comfy, prompt_id: str) -> bool:
    """Take a job off the card, and whether that is confirmed: ComfyUI's cancel of the job by its id, then, within ten
    polls of half a second, its record, or two reads of the queue in a row that do not hold it (the queue read before
    the record each time). A card that answers neither confirms nothing."""
    with contextlib.suppress(ComfyError):
        await comfy.cancel(prompt_id)
    missing = 0
    for poll in range(STOP_POLLS):
        with contextlib.suppress(ComfyError):
            if (await comfy.queue()).holds(prompt_id):
                missing = 0
            elif await comfy.history(prompt_id) is not None:
                return True
            else:
                missing += 1
                if missing > 1:
                    return True
        if poll < STOP_POLLS - 1:
            await asyncio.sleep(STOP_POLL_S)
    return False


# ComfyUI's log, counted and dropped (simple-story-chat's `logCursor` and `partialLoadsAfter`).

def log_cursor(lines: list[str]) -> str:
    """Where a read of the log ended, kept without a word of it: the hash of the last line, or "" for an empty ring."""
    return hashlib.sha256(lines[-1].encode()).hexdigest() if lines else ""


def partial_loads_after(cursor: str, lines: list[str]) -> int:
    """The lines past the cursor that say a model was loaded onto the card in part. A cursor whose line has left the
    ring counts the whole ring."""
    start = 0
    if cursor:
        for at in range(len(lines) - 1, -1, -1):
            if hashlib.sha256(lines[at].encode()).hexdigest() == cursor:
                start = at + 1
                break
    return sum(1 for line in lines[start:] if PARTIAL_LOAD.search(line))


# The card's memory while a job draws (simple-story-chat's `readStats` and `mergeStats`).

def _mib(amount: float) -> int:
    return math.floor(amount / MIB + 0.5)  # JavaScript's Math.round, which the harness's numbers were made with


@dataclass
class Memory:
    """The video memory by device, as its maxima, the samples taken while the job drew, and the RAM in use."""

    vram: dict[int, dict[str, int]] = field(default_factory=dict)
    samples: int = 0
    ram: tuple[int, int] | None = None  # the least and the most in use

    def merge(self, stats: dict[str, Any], *, during: bool) -> None:
        """Add a read of /system_stats. A read taken while the job drew counts as a sample; the last one, at its
        over, adds to the maxima alone."""
        devices = stats.get("devices")
        seen: list[dict[str, int]] = []
        for order, device in enumerate(devices if isinstance(devices, list) else []):
            if not isinstance(device, dict) or not _number(device.get("vram_total")) \
                    or not _number(device.get("vram_free")):
                continue
            used = device["vram_total"] - device["vram_free"]
            index = device.get("index")
            one = {"index": index if isinstance(index, int) and not isinstance(index, bool) else order,
                   "total_mib": _mib(device["vram_total"]), "used_mib_max": _mib(used)}
            if _number(device.get("torch_vram_free")):
                one["occupied_mib_max"] = _mib(used + device["torch_vram_free"])
            seen.append(one)
        if seen and during:
            self.samples += 1
        for one in seen:
            known = self.vram.get(one["index"])
            if known is None:
                self.vram[one["index"]] = dict(one)
                continue
            known["used_mib_max"] = max(known["used_mib_max"], one["used_mib_max"])
            if "occupied_mib_max" in one:
                known["occupied_mib_max"] = max(known.get("occupied_mib_max", 0), one["occupied_mib_max"])
        system = stats.get("system")
        if isinstance(system, dict) and _number(system.get("ram_total")) and _number(system.get("ram_free")):
            ram = _mib(system["ram_total"] - system["ram_free"])
            self.ram = (ram, ram) if self.ram is None else (min(self.ram[0], ram), max(self.ram[1], ram))

    def event(self) -> dict[str, Any]:
        memory: dict[str, Any] = {"vram": list(self.vram.values()), "samples": self.samples}
        if self.ram is not None:
            memory["ram_mib"] = {"min": self.ram[0], "max": self.ram[1]}
        return memory


@dataclass(frozen=True)
class Fingerprint:
    """What the health check holds the engine to (docs/pictures.md, section 6)."""

    comfyui: str | None
    pytorch: str | None
    triton: bool  # comfy-kitchen's Triton backend asked for and not overridden, as the pinned server reads its flags
    card: str | None  # device 0's name

    @classmethod
    def of(cls, stats: dict[str, Any]) -> Fingerprint:
        seen, devices = stats.get("system"), stats.get("devices")
        system: dict[str, Any] = seen if isinstance(seen, dict) else {}
        first: dict[str, Any] = devices[0] if isinstance(devices, list) and devices and isinstance(devices[0], dict) \
            else {}
        argv = system.get("argv")
        flags = argv if isinstance(argv, list) else []

        def text(value: Any) -> str | None:
            return value if isinstance(value, str) and value else None

        return cls(comfyui=text(system.get("comfyui_version")), pytorch=text(system.get("pytorch_version")),
                   triton="--enable-triton-backend" in flags and "--disable-triton-backend" not in flags,
                   card=text(first.get("name")))


class Socket:
    """The gateway's one websocket to ComfyUI, for all its jobs, opened again under a new client id whenever it drops:
    the pinned server files a socket under its id alone, and the old handler's end takes the id's socket with it
    (server.py:274-276, 324-326). ComfyUI tells a job's news only to the socket of the client id the job was submitted
    under, while it is connected, so a job heard by one socket is not heard by the next. `epoch` counts the sockets
    that opened. Text messages go to `hear` as JSON objects with the epoch they came on; binary frames, the previews
    of a drawing, are dropped unread."""

    def __init__(self, base_url: str, hear: Callable[[dict[str, Any], int], None],
                 opened: Callable[[], None]) -> None:
        self._url = "ws" + base_url.removeprefix("http") + "/ws?clientId="
        self._hear = hear
        self._opened = opened
        self.client_id = ""
        self.epoch = 0
        self.is_open = False
        self._open = asyncio.Event()

    async def run(self) -> None:
        """Keep a socket open until cancelled."""
        while True:
            client_id = str(uuid.uuid4())
            try:
                async with connect(self._url + client_id, proxy=None, compression=None, user_agent_header=None,
                                   open_timeout=CONNECT_TIMEOUT_S, close_timeout=2,
                                   max_size=SOCKET_MESSAGE_BYTES) as socket:
                    self.client_id, self.epoch, self.is_open = client_id, self.epoch + 1, True
                    self._open.set()
                    self._opened()
                    async for message in socket:
                        if isinstance(message, str):
                            self._message(message)
            except (OSError, TimeoutError, WebSocketException):
                pass
            finally:
                self.is_open = False
                self._open.clear()
            await asyncio.sleep(RECONNECT_S)

    def _message(self, text: str) -> None:
        try:
            message = json.loads(text)
        except ValueError:
            return
        if isinstance(message, dict):
            self._hear(message, self.epoch)

    async def wait_open(self, timeout: float) -> bool:
        """Whether the socket is open now or opens within `timeout`."""
        if not self.is_open:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(timeout):
                    await self._open.wait()
        return self.is_open

"""One synthetic picture through an already-running loopback gateway, with an independent cleanup observation.

Run with --fake or --real and explicit public, control and ComfyUI observer ports. --wall-s bounds preflight and
one generation; --cleanup-s reserves additional time for this job's cleanup. No card lifecycle operation is used.
Only fixed codes, booleans and counts reach stdout. The graph, images, keys and observed identifiers stay in memory.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextlib
import json
import math
import re
import signal
import struct
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import httpx

from .. import card, cli
from ..policy import CLASS_HEADER

FOLDER = card.CODE / "card/pictures"
JSON_BYTES, PNG_BYTES, EVENT_BYTES, STREAM_BYTES = 1 << 20, 2 << 20, 3 << 20, 8 << 20
EVENTS, CALL_S, POLL_S = 4096, 5.0, 0.05
PNG = b"\x89PNG\r\n\x1a\n"
NODES = frozenset(str(i) for i in range(1, 10))
CODES = frozenset({"ok", "invalid_options", "config_invalid", "assets_invalid", "http_status", "transport",
                   "deadline", "state_mismatch", "versions_mismatch", "engine_mismatch", "observer_invalid",
                   "observer_uncorrelated", "sse_invalid", "stream_limit", "png_invalid", "png_limit",
                   "generation_error", "cleanup_unconfirmed", "cancelled"})
WORDS = CODES | {"fake", "real", "unknown", "preflight", "generation", "cleanup", "complete", "not_verifiable",
                 "not_measured"}
FIELDS = frozenset({"ok", "mode", "phase", "code", "gpu", "ready", "versions_match", "engine_match", "accepted",
                    "observer_correlated", "stream_complete", "png_received", "images", "image_bytes", "events",
                    "stop_requested", "stop_confirmed", "history_deleted", "cleanup_confirmed", "elapsed_ms"})


class Failure(Exception):
    def __init__(self, code: str) -> None:
        self.code = code if code in CODES else "sse_invalid"
        super().__init__(self.code)


@dataclass(frozen=True)
class Target:
    public_port: int
    control_port: int
    engine_port: int
    client_key: str = field(repr=False)
    control_key: str = field(repr=False)

    def url(self, lane: str, path: str) -> str:
        port = {"public": self.public_port, "control": self.control_port, "engine": self.engine_port}[lane]
        return f"http://127.0.0.1:{port}{path}"


@dataclass(frozen=True)
class Options:
    fake: bool
    wall_s: float = 300
    cleanup_s: float = 20


def expected_versions() -> dict[str, str]:
    pins = card.read_manifest(folder="card/pictures", name="image-manifest.env")
    source = re.search(r"^commit ([0-9a-f]{40})$", (FOLDER / "SOURCE").read_text(), re.MULTILINE)
    if source is None:
        raise Failure("assets_invalid")
    return {"comfyui_revision": pins["COMFYUI_REVISION"], "qwen_revision": pins["IMAGE_QWEN_REVISION"],
            "scripts_commit": source[1]}


def graph() -> dict[str, Any]:
    """The pinned Qwen graph with a small synthetic request and preview-only output."""
    found: dict[str, Any] = json.loads((FOLDER / "image-workflow-qwen.json").read_text())
    found["4"]["inputs"] |= {"prompt": "A plain blue circle centered on a white background.", "negative_prompt": ""}
    found["5"]["inputs"] |= {"width": 512, "height": 512, "batch_size": 1}
    found["6"]["inputs"] |= {"seed": 7, "steps": 4, "model": ["9", 0]}
    found["8"] = {"class_type": "PreviewImage", "inputs": {"images": ["7", 0]}}
    found["9"] = {"class_type": "ModelAttentionBackend", "inputs": {
        "model": ["1", 0], "attention": "comfy kitchen attention"}}
    return found


def report(fake: bool | None = None) -> dict[str, Any]:
    return {"ok": False, "mode": "unknown" if fake is None else "fake" if fake else "real", "phase": "preflight",
            "code": "ok", "gpu": "not_measured" if fake is False else "not_verifiable",
            **dict.fromkeys(("ready", "versions_match", "engine_match", "accepted", "observer_correlated",
                            "stream_complete", "png_received", "stop_requested", "stop_confirmed", "history_deleted",
                            "cleanup_confirmed"), False), "images": 0, "image_bytes": 0, "events": 0, "elapsed_ms": 0}


def checked(result: dict[str, Any]) -> str:
    if result.keys() != FIELDS or any(not (type(value) is bool or (type(value) is int and 0 <= value <= 10**12)
                                          or (isinstance(value, str) and value in WORDS)) for value in result.values()):
        raise Failure("sse_invalid")
    return json.dumps(result, separators=(",", ":"))


def count(value: Any) -> bool:
    return type(value) is int and 0 <= value <= 10**12


def number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, int | float)
            and 0 <= value <= 10**15 and math.isfinite(value))


def parse(data: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        found = dict(items)
        if len(found) != len(items):
            raise Failure("sse_invalid")
        return found

    def constant(value: str) -> Any:
        raise Failure("sse_invalid")

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError):
        raise Failure("sse_invalid") from None


def png_size(encoded: Any) -> int:
    if not isinstance(encoded, str) or len(encoded) > 4 * ((PNG_BYTES + 2) // 3):
        raise Failure("png_limit")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise Failure("png_invalid") from None
    if len(data) > PNG_BYTES:
        raise Failure("png_limit")
    if not data.startswith(PNG):
        raise Failure("png_invalid")
    at, compressed = 8, False
    kinds: list[bytes] = []
    while at + 12 <= len(data):
        size = int.from_bytes(data[at:at + 4], "big")
        kind, end = data[at + 4:at + 8], at + 12 + size
        if end > len(data) or kind not in {b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND"}:
            raise Failure("png_invalid")
        if zlib.crc32(data[at + 4:end - 4]) != int.from_bytes(data[end - 4:end], "big"):
            raise Failure("png_invalid")
        if not kinds:
            if kind != b"IHDR" or size != 13:
                raise Failure("png_invalid")
            width, height = struct.unpack(">II", data[at + 8:at + 16])
            depth, color, compression, filtering, interlace = data[at + 16:at + 21]
            depths = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
            if (width == 0 or height == 0 or depth not in depths.get(color, set())
                    or compression or filtering or interlace not in (0, 1)):
                raise Failure("png_invalid")
        elif kind == b"IHDR":
            raise Failure("png_invalid")
        kinds.append(kind)
        if len(kinds) > 1024:
            raise Failure("png_invalid")
        compressed = compressed or (kind == b"IDAT" and size > 0)
        at = end
        if kind == b"IEND":
            if size or at != len(data) or not compressed:
                raise Failure("png_invalid")
            return len(data)
    raise Failure("png_invalid")


async def json_call(client: httpx.AsyncClient, target: Target, lane: str, path: str,
                    headers: dict[str, str] | None = None, *, method: str = "GET") -> tuple[int, Any]:
    async with asyncio.timeout(CALL_S), client.stream(method, target.url(lane, path), headers=headers) as response:
        if response.headers.get("content-encoding", "identity") != "identity":
            raise Failure("http_status")
        data = bytearray()
        async for chunk in response.aiter_raw():
            data.extend(chunk)
            if len(data) > JSON_BYTES:
                raise Failure("stream_limit")
        return response.status_code, parse(bytes(data))


@dataclass
class Course:
    result: dict[str, Any]
    job: str | None = field(default=None, repr=False)
    started: bool = False
    over: bool = False
    done: bool = False

    def event(self, data: bytes) -> bool:
        if self.done:
            raise Failure("sse_invalid")
        self.result["events"] += 1
        if self.result["events"] > EVENTS:
            raise Failure("stream_limit")
        if data == b"[DONE]":
            if not self.over or self.result["images"] != 1:
                raise Failure("sse_invalid")
            self.done = True
            return False
        event = parse(data)
        if not isinstance(event, dict):
            raise Failure("sse_invalid")
        if "error" in event:
            raise Failure("generation_error")
        name = event.get("event")
        if not isinstance(name, str):
            raise Failure("sse_invalid")
        if name == "accepted":
            if self.job is not None:
                raise Failure("sse_invalid")
            job = event.get("job")
            try:
                if not isinstance(job, str) or str(uuid.UUID(job)) != job:
                    raise ValueError
            except ValueError:
                raise Failure("sse_invalid") from None
            self.job = job
            self.result["accepted"] = True
            if event.keys() != {"event", "job", "socket"} or event["socket"] is not True:
                raise Failure("sse_invalid")
            return True
        if self.job is None:
            raise Failure("sse_invalid")
        if name == "started":
            if self.started or self.over or not event.keys() <= {"event", "stamp"}:
                raise Failure("sse_invalid")
            if "stamp" in event and not number(event["stamp"]):
                raise Failure("sse_invalid")
            self.started = True
        elif name == "over":
            if (not self.started or self.over
                    or not event.keys() <= {"event", "heard", "memory", "stamps", "partial_loads"}
                    or type(event.get("heard")) is not bool or not isinstance(event.get("memory"), dict)):
                raise Failure("sse_invalid")
            self.over = True
        elif name == "image":
            if (not self.over or self.result["images"] or event.keys() != {"event", "node", "png"}
                    or event["node"] != "8"):
                raise Failure("sse_invalid")
            self.result["image_bytes"] = png_size(event["png"])
            self.result["images"], self.result["png_received"] = 1, True
        elif not self.over and name in {"node", "cached", "progress", "place"}:
            valid = False
            if name == "node":
                valid = (event.keys() == {"event", "node"} and isinstance(event["node"], str)
                         and event["node"] in NODES)
            elif name == "cached":
                nodes = event.get("nodes")
                valid = (event.keys() == {"event", "nodes"} and isinstance(nodes, list) and len(nodes) <= len(NODES)
                         and all(isinstance(node, str) and node in NODES for node in nodes))
            elif name == "progress":
                valid = (event.keys() == {"event", "value", "max"} and count(event["value"]) and count(event["max"])
                         and event["max"] > 0 and event["value"] <= event["max"])
            elif name == "place":
                valid = (event.keys() == {"event", "before", "drawing", "drawing_ms"} and count(event["before"])
                         and count(event["drawing_ms"]) and type(event["drawing"]) is bool)
            if not valid:
                raise Failure("sse_invalid")
        else:
            raise Failure("sse_invalid")
        return False


async def observation(client: httpx.AsyncClient, target: Target, job: str) -> tuple[bool, bool]:
    status, queue = await json_call(client, target, "engine", "/queue")
    if status != 200 or not isinstance(queue, dict):
        raise Failure("observer_invalid")
    held = False
    for name in ("queue_running", "queue_pending"):
        entries = queue.get(name)
        if not isinstance(entries, list):
            raise Failure("observer_invalid")
        for entry in entries:
            if not isinstance(entry, list) or len(entry) < 2 or not isinstance(entry[1], str):
                raise Failure("observer_invalid")
            held = held or entry[1] == job
    status, history = await json_call(client, target, "engine", f"/history/{job}")
    if status != 200 or not isinstance(history, dict) or not history.keys() <= {job}:
        raise Failure("observer_invalid")
    if job in history and not isinstance(history[job], dict):
        raise Failure("observer_invalid")
    return held, job in history


async def preflight(client: httpx.AsyncClient, target: Target, headers: dict[str, str], result: dict[str, Any]) -> None:
    manifest = card.read_manifest(folder="card/pictures")
    pins = card.read_manifest(folder="card/pictures", name="image-manifest.env")
    status, state = await json_call(client, target, "control", "/v1/state",
                                    {"Authorization": f"Bearer {target.control_key}"})
    if status != 200 or not isinstance(state, dict):
        raise Failure("http_status")
    wanted_state = {"contract": "2", "lane": "pictures", "status": "ready", "model": manifest["MODEL_ALIAS"]}
    if (any(state.get(key) != value for key, value in wanted_state.items())
            or state.get("sleep_requested") is not False):
        raise Failure("state_mismatch")
    boot_id = state.get("boot_id")
    if not isinstance(boot_id, str) or re.fullmatch(r"[0-9a-f]{32}", boot_id) is None:
        raise Failure("state_mismatch")
    status, public_state = await json_call(client, target, "public", "/v1/state", headers)
    if (status != 200 or not isinstance(public_state, dict)
            or any(public_state.get(key) != value for key, value in wanted_state.items())
            or public_state.get("boot_id") != boot_id):
        raise Failure("state_mismatch")
    result["ready"] = True
    versions = state.get("versions")
    if not isinstance(versions, dict) or any(versions.get(key) != value for key, value in expected_versions().items()):
        raise Failure("versions_mismatch")
    result["versions_match"] = True
    status, facts = await json_call(client, target, "public", "/v1/pictures/engine", headers)
    wanted = {"comfyui": pins["COMFYUI_VERSION"], "pytorch": pins["TORCH_CU130_VERSION"],
              "triton": True, "kitchen_attention": True}
    if (status != 200 or not isinstance(facts, dict)
            or any(facts.get(key) is not value if type(value) is bool else facts.get(key) != value
                   for key, value in wanted.items())):
        raise Failure("engine_mismatch")
    result["engine_match"] = True


async def generate(client: httpx.AsyncClient, target: Target, headers: dict[str, str], course: Course) -> None:
    async with client.stream("POST", target.url("public", "/v1/pictures"), headers=headers,
                             json={"graph": graph(), "require_socket": True}) as response:
        if (response.status_code != 200
                or response.headers.get("content-type", "").partition(";")[0] != "text/event-stream"
                or response.headers.get("content-encoding", "identity") != "identity"):
            raise Failure("http_status")
        pending, event, size = bytearray(), None, 0
        async for chunk in response.aiter_raw():
            size += len(chunk)
            if size > STREAM_BYTES:
                raise Failure("stream_limit")
            pending.extend(chunk)
            while (at := pending.find(b"\n")) >= 0:
                line = bytes(pending[:at]).removesuffix(b"\r")
                del pending[:at + 1]
                if len(line) > EVENT_BYTES:
                    raise Failure("stream_limit")
                if not line:
                    if event is not None:
                        if course.event(event):
                            held, recorded = await observation(client, target, course.job or "")
                            course.result["observer_correlated"] = held or recorded
                        event = None
                elif course.done or not line.startswith(b"data: ") or event is not None:
                    raise Failure("sse_invalid")
                else:
                    event = line[6:]
            if len(pending) > EVENT_BYTES:
                raise Failure("stream_limit")
        if pending or event is not None or not course.done:
            raise Failure("sse_invalid")
        course.result["stream_complete"] = True


async def cleanup(client: httpx.AsyncClient, target: Target, headers: dict[str, str], course: Course) -> None:
    result, job = course.result, course.job
    if job is None:
        return
    # If acceptance was malformed, make one correlation attempt before a stop could remove the evidence.
    if not result["observer_correlated"]:
        with contextlib.suppress(Failure, httpx.HTTPError, TimeoutError):
            held, recorded = await observation(client, target, job)
            result["observer_correlated"] = held or recorded
    if not result["stream_complete"]:
        result["stop_requested"] = True
        status, stopped = await json_call(client, target, "public", f"/v1/pictures/{job}/stop", headers, method="POST")
        result["stop_confirmed"] = status == 200 and isinstance(stopped, dict) and stopped.get("stopped") is True
        # A 404 says only that the gateway forgot the job; observer confirmation is still required below.
        if status not in (200, 404) or (status == 200 and not result["stop_confirmed"]):
            raise Failure("cleanup_unconfirmed")
    while True:
        held, recorded = await observation(client, target, job)
        if not held and not recorded:
            if not result["observer_correlated"]:
                raise Failure("observer_uncorrelated")
            result["history_deleted"] = result["cleanup_confirmed"] = True
            return
        await asyncio.sleep(POLL_S)


async def run(target: Target, options: Options) -> dict[str, Any]:
    result, course = report(options.fake), None
    began = asyncio.get_running_loop().time()
    try:
        ports = (target.public_port, target.control_port, target.engine_port)
        if (type(options.fake) is not bool or any(type(port) is not int or not 1 <= port <= 65535 for port in ports)
                or len(set(ports)) != 3
                or any(not number(value) or value <= 0 for value in (options.wall_s, options.cleanup_s))):
            raise Failure("invalid_options")
        if any(not isinstance(key, str) or not key or len(key) > 4096
               or any(not 33 <= ord(char) <= 126 for char in key) for key in (target.client_key, target.control_key)):
            raise Failure("config_invalid")
        headers = {"Authorization": f"Bearer {target.client_key}", CLASS_HEADER: "internal"}
        course = Course(result)
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=CALL_S),
                                     trust_env=False, follow_redirects=False) as client:
            try:
                async with asyncio.timeout(options.wall_s):
                    await preflight(client, target, headers, result)
                    result["phase"] = "generation"
                    await generate(client, target, headers, course)
            except Failure as error:
                result["code"] = error.code
            except TimeoutError:
                result["code"] = "deadline"
            except httpx.HTTPError:
                result["code"] = "transport"
            except asyncio.CancelledError:
                result["code"] = "cancelled"
            finally:
                if course.job is not None:
                    try:
                        async with asyncio.timeout(options.cleanup_s):
                            await cleanup(client, target, headers, course)
                    except asyncio.CancelledError:
                        result["code"] = "cancelled"
                    except (Failure, httpx.HTTPError, TimeoutError) as error:
                        if result["code"] == "ok":
                            result["phase"] = "cleanup"
                            result["code"] = error.code if isinstance(error, Failure) else "cleanup_unconfirmed"
            result["ok"] = result["code"] == "ok" and result["stream_complete"] and result["cleanup_confirmed"]
            if result["ok"]:
                result["phase"] = "complete"
    except Failure as error:
        result["code"] = error.code
    except (OSError, ValueError, KeyError, TypeError, RecursionError, OverflowError):
        result["code"] = "assets_invalid"
    result["elapsed_ms"] = round((asyncio.get_running_loop().time() - began) * 1000)
    return result


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise Failure("invalid_options")


async def execute(target: Target, options: Options) -> dict[str, Any]:
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    if task is not None:
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await run(target, options)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main(argv: list[str] | None = None) -> int:
    result = report()
    try:
        parser = Parser(description=__doc__.split("\n\n")[0])
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--fake", action="store_true")
        mode.add_argument("--real", action="store_true")
        parser.add_argument("--config", type=Path, default=cli.CONFIG)
        for name in ("public", "control", "engine"):
            parser.add_argument(f"--{name}-port", type=int, required=True)
        parser.add_argument("--wall-s", type=float, default=300)
        parser.add_argument("--cleanup-s", type=float, default=20)
        args = parser.parse_args(argv)
        result = report(args.fake)
        if args.config.stat().st_size > JSON_BYTES:
            raise Failure("config_invalid")
        keys = cli.read_config(args.config)
        target = Target(args.public_port, args.control_port, args.engine_port, keys["client_key"], keys["control_key"])
        result = asyncio.run(execute(target, Options(args.fake, args.wall_s, args.cleanup_s)))
    except Failure as error:
        result["code"] = error.code
    except (cli.Refusal, OSError, ValueError, KeyError, TypeError):
        result["code"] = "config_invalid"
    except KeyboardInterrupt:
        result["code"] = "cancelled"
    print(checked(result), flush=True)
    return 0 if result["ok"] else 130 if result["code"] == "cancelled" else 2 if result["code"] in {
        "invalid_options", "config_invalid"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

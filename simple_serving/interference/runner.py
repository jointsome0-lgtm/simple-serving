"""Bounded client-side interference trials. Chunk timing does not establish GPU prefill overlap."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from . import workload

CHAT = "/v1/chat/completions"
COUNT = CHAT + "/input_tokens"
MAX_STREAM_BYTES = 8 * 1024 * 1024
MAX_EVENT_BYTES = 128 * 1024
MAX_EVENTS = 65536
Diagnostics = Callable[[], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Target:
    public: str
    key: str
    alias: str


@dataclass(frozen=True)
class Options:
    rounds: int = 3
    long_tokens: int = 16000
    victim_tokens: int = 1024
    aggressor_tokens: int = 1
    concurrency: int = 2
    trigger_chunks: int = 4
    start_timeout_s: float = 30
    trial_timeout_s: float = 180
    aggressor_class: str = "internal"


class TrialError(Exception):
    """Only module-owned fixed codes may cross the report boundary."""


def _headers(target: Target, cls: str, scope: str) -> dict[str, str]:
    return {"Authorization": "Bearer " + target.key, "X-Simple-Serving-Class": cls,
            "X-Simple-Serving-Scope": scope}


def _count(value: Any) -> bool:
    return type(value) is int and 0 <= value <= 10000000


def _json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        raise TrialError("invalid_json") from None


async def _input_tokens(client: httpx.AsyncClient, target: Target, body: dict[str, Any], timeout: float) -> int:
    async with (asyncio.timeout(timeout), client.stream("POST", target.public.rstrip("/") + COUNT,
               json=body, headers=_headers(target, "internal", workload.scope())) as response):
        if response.status_code != 200:
            raise TrialError("count_http_error")
        raw = bytearray()
        async for chunk in response.aiter_bytes():
            raw.extend(chunk)
            if len(raw) > 4096:
                raise TrialError("count_too_large")
        value = _json(bytes(raw))
        if not isinstance(value, dict) or not _count(value.get("input_tokens")) or value["input_tokens"] < 1:
            raise TrialError("count_invalid")
        return int(value["input_tokens"])


async def _size(client: httpx.AsyncClient, target: Target, options: Options) -> tuple[dict[str, Any], dict[str, int]]:
    """Count every candidate, then choose the largest tested prompt within the token target."""
    seen: dict[int, int] = {}

    async def measure(units: int) -> int:
        count = await _input_tokens(client, target, workload.aggressor(target.alias, options.aggressor_tokens, units),
                                    options.start_timeout_s)
        if any((u < units and n > count) or (u > units and n < count) for u, n in seen.items()):
            raise TrialError("count_nonmonotonic")
        seen[units] = count
        return count

    low, high = 1, 1
    if await measure(1) > options.long_tokens:
        raise TrialError("count_target_too_small")
    while seen[high] < options.long_tokens and high < workload.MAX_UNITS:
        low, high = high, min(high * 2, workload.MAX_UNITS)
        await measure(high)
    if seen[high] < options.long_tokens:
        raise TrialError("count_target_unreachable")
    for _ in range(16):
        if high - low <= 1:
            break
        middle = (low + high) // 2
        if await measure(middle) <= options.long_tokens:
            low = middle
        else:
            high = middle
    units = high if seen[high] <= options.long_tokens else low
    return workload.aggressor(target.alias, options.aggressor_tokens, units), {
        "prompt_tokens": seen[units], "count_requests": len(seen), "units": units}


def _gaps(values: list[float]) -> dict[str, int | float | None]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)

    return {"samples": len(values), "p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99),
            "max": round(max(values), 3) if values else None}


@dataclass
class Reading:
    started: float = 0
    ended: float = 0
    chunks: list[float] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    finished: bool = False
    done: bool = False

    def report(self, injection: float | None, origin: float) -> dict[str, Any]:
        gaps = [(right - left) * 1000 for left, right in zip(self.chunks, self.chunks[1:], strict=False)]
        after = [(right - left) * 1000 for left, right in zip(self.chunks, self.chunks[1:], strict=False)
                 if injection is not None and right >= injection]
        return {"started_ms": round((self.started - origin) * 1000, 3) if self.started else None,
                "ended_ms": round((self.ended - origin) * 1000, 3) if self.ended else None,
                "ttft_ms": round((self.chunks[0] - self.started) * 1000, 3) if self.chunks else None,
                "total_ms": round((self.ended - self.started) * 1000, 3) if self.ended else None,
                "chunks": len(self.chunks), "gap_ms": _gaps(gaps), "post_injection_gap_ms": _gaps(after),
                "usage": self.usage}


def _usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(_count(value.get(k)) for k in ("prompt_tokens", "completion_tokens")):
        raise TrialError("usage_invalid")
    details = value.get("prompt_tokens_details")
    if details is not None and not isinstance(details, dict):
        raise TrialError("usage_invalid")
    cached = details.get("cached_tokens") if details else None
    if cached is not None and (not _count(cached) or cached > value["prompt_tokens"]):
        raise TrialError("usage_invalid")
    measured = value.get("simple_serving")
    if (not isinstance(measured, dict)
            or not all(_count(measured.get(k)) for k in ("wait_ms", "first_token_ms", "total_ms"))):
        raise TrialError("usage_invalid")
    return {"prompt_tokens": value["prompt_tokens"], "completion_tokens": value["completion_tokens"],
            "cached_tokens": cached, **{k: measured[k] for k in ("wait_ms", "first_token_ms", "total_ms")}}


def _event(raw: bytes, reading: Reading, alias: str, ready: asyncio.Event, trigger: int) -> None:
    if reading.done:
        raise TrialError("after_done")
    if raw == b"[DONE]":
        reading.done = True
        return
    event = _json(raw)
    if not isinstance(event, dict) or "error" in event:
        raise TrialError("stream_error")
    if event.get("object") != "chat.completion.chunk" or event.get("model") != alias:
        raise TrialError("event_invalid")
    if event.get("usage") is not None:
        if reading.usage is not None:
            raise TrialError("usage_duplicate")
        reading.usage = _usage(event["usage"])
    choices = event.get("choices")
    if not isinstance(choices, list) or len(choices) > 1:
        raise TrialError("event_invalid")
    if not choices:
        return
    choice = choices[0]
    if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0:
        raise TrialError("event_invalid")
    delta, finish = choice.get("delta"), choice.get("finish_reason")
    if not isinstance(delta, dict) or not delta.keys() <= {"role", "content", "reasoning_content"}:
        raise TrialError("delta_invalid")
    if delta.get("role", "assistant") != "assistant" or not all(isinstance(v, str) for v in delta.values()):
        raise TrialError("delta_invalid")
    if reading.finished:
        raise TrialError("after_finish")
    if delta.get("content") or delta.get("reasoning_content"):
        reading.chunks.append(time.monotonic())
        if len(reading.chunks) >= trigger:
            ready.set()
    if finish is not None:
        if finish not in ("stop", "length"):
            raise TrialError("finish_invalid")
        reading.finished = True


async def _stream(client: httpx.AsyncClient, target: Target, body: dict[str, Any], cls: str, scope: str,
                  reading: Reading, ready: asyncio.Event, trigger: int, start_timeout_s: float) -> None:
    reading.started = time.monotonic()
    try:
        async with (asyncio.timeout(start_timeout_s) as start_deadline,
                    client.stream("POST", target.public.rstrip("/") + CHAT, json=body,
                                  headers=_headers(target, cls, scope)) as response):
            if response.status_code != 200:
                raise TrialError("http_error")
            if response.headers.get("content-type", "").split(";")[0] != "text/event-stream":
                raise TrialError("content_type")
            pending, data, size, events = b"", [], 0, 0
            async for piece in response.aiter_bytes():
                size += len(piece)
                if size > MAX_STREAM_BYTES:
                    raise TrialError("stream_too_large")
                pending += piece
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    line = line.removesuffix(b"\r")
                    if line.startswith(b"data:"):
                        data.append(line[5:].removeprefix(b" "))
                        if sum(map(len, data)) > MAX_EVENT_BYTES:
                            raise TrialError("event_too_large")
                    elif line.startswith(b":"):
                        continue
                    elif line:
                        raise TrialError("event_invalid")
                    elif data:
                        events += 1
                        if events > MAX_EVENTS:
                            raise TrialError("too_many_events")
                        _event(b"\n".join(data), reading, target.alias, ready, trigger)
                        if reading.chunks:
                            start_deadline.reschedule(None)
                        data = []
                if len(pending) > MAX_EVENT_BYTES:
                    raise TrialError("event_too_large")
            if pending or data or not reading.done:
                raise TrialError("missing_done")
            if not reading.finished or reading.usage is None:
                raise TrialError("missing_finish_or_usage")
            if reading.usage["completion_tokens"] > body["max_tokens"]:
                raise TrialError("completion_exceeds_limit")
    finally:
        reading.ended = time.monotonic()


async def _trial(client: httpx.AsyncClient, target: Target, options: Options, case: str, round_index: int,
                 long_body: dict[str, Any], expected_count: int, origin: float) -> dict[str, Any]:
    reader, long = Reading(), []
    tasks: list[asyncio.Task[Any]] = []
    injection = None
    status, code = "ok", "ok"
    scopes = [workload.scope() for _ in range(options.concurrency - 1)]
    try:
        async with asyncio.timeout(options.trial_timeout_s):
            if case == "warm":
                for scope in scopes:
                    await _stream(client, target, long_body, options.aggressor_class, scope,
                                  Reading(), asyncio.Event(), 1, options.start_timeout_s)
            ready = asyncio.Event()
            victim = asyncio.create_task(_stream(client, target,
                workload.victim(target.alias, options.victim_tokens, round_index), "reader", workload.scope(),
                reader, ready, options.trigger_chunks, options.start_timeout_s))
            tasks.append(victim)
            if case != "alone":
                gate = asyncio.create_task(ready.wait())
                tasks.append(gate)
                async with asyncio.timeout(options.start_timeout_s):
                    await asyncio.wait((victim, gate), return_when=asyncio.FIRST_COMPLETED)
                if victim.done():
                    await victim
                    gate.cancel()
                    await asyncio.gather(gate, return_exceptions=True)
                    tasks.remove(gate)
                    status, code = "inconclusive", "victim_ended_before_injection"
                else:
                    injection = time.monotonic()
                    for scope in scopes:
                        reading = Reading()
                        long.append(reading)
                        tasks.append(asyncio.create_task(_stream(client, target, long_body, options.aggressor_class,
                            scope, reading, asyncio.Event(), 1, options.start_timeout_s)))
            await asyncio.gather(*tasks)
    except TrialError as error:
        status, code = "failed", str(error)
    except TimeoutError:
        status, code = "failed", "timeout"
    except httpx.HTTPError:
        status, code = "failed", "transport_error"
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    overlap = [round(max(0., min(reader.ended, item.ended) - max(reader.started, item.started)) * 1000, 3)
               for item in long]
    cached = [item.usage["cached_tokens"] if item.usage else None for item in long]
    warm = all(n is not None and n > 0 for n in cached) if cached and None not in cached and case == "warm" else None
    overlapping_chunks = [sum(item.started <= stamp <= item.ended for stamp in reader.chunks) for item in long]
    # A full freeze can span the entire aggressor request. Its first resumed victim chunk arrives after that request.
    continued_chunks = [sum(stamp > item.started for stamp in reader.chunks) for item in long]
    if status == "ok" and case != "alone":
        if any(item.usage is None or item.usage["prompt_tokens"] != expected_count for item in long):
            status, code = "failed", "count_mismatch"
        elif not overlap or min(overlap) <= 0:
            status, code = "inconclusive", "no_http_overlap"
        elif min(continued_chunks) == 0:
            status, code = "inconclusive", "victim_did_not_continue"
        elif any(n is None for n in cached):
            status, code = "inconclusive", "cache_unreported"
        elif case == "warm" and not warm:
            status, code = "inconclusive", "warm_cache_unconfirmed"
        elif case == "cold" and any(n != 0 for n in cached):
            status, code = "inconclusive", "cold_cache_hit"
    return {"case": case, "round": round_index, "status": status, "code": code,
            "injection_ms": round((injection - origin) * 1000, 3) if injection is not None else None,
            "victim": reader.report(injection, origin), "aggressors": [item.report(None, origin) for item in long],
            "http_overlap_ms": overlap, "victim_chunks_during_aggressor": overlapping_chunks,
            "victim_chunks_after_aggressor_start": continued_chunks,
            "warm_cache_confirmed": warm}


async def run(client: httpx.AsyncClient, target: Target, options: Options,
              diagnostics: Diagnostics | None = None) -> dict[str, Any]:
    origin = time.monotonic()
    report: dict[str, Any] = {"schema_version": 1, "status": "ok", "code": "ok",
                              "options": None, "sizing": None, "trials": []}
    try:
        integers = (options.rounds, options.long_tokens, options.victim_tokens, options.aggressor_tokens,
                    options.concurrency, options.trigger_chunks)
        if (any(type(n) is not int or n < 1 for n in integers) or options.rounds > 100
                or options.concurrency not in (2, 4) or options.aggressor_class not in ("reader", "agent", "internal")
                or max(options.long_tokens, options.victim_tokens, options.aggressor_tokens) > 1000000
                or not all(math.isfinite(n) and 0 < n <= 3600
                           for n in (options.start_timeout_s, options.trial_timeout_s))):
            raise TrialError("options_invalid")
        report["options"] = asdict(options)
        long_body, sizing = await _size(client, target, options)
        report["sizing"] = sizing
        cases = ("alone", "cold", "warm")
        for index in range(options.rounds):
            for case in cases[index % 3:] + cases[:index % 3]:
                before = None
                if diagnostics is not None:
                    async with asyncio.timeout(options.start_timeout_s):
                        before = await diagnostics()
                trial = await _trial(client, target, options, case, index, long_body, sizing["prompt_tokens"], origin)
                report["trials"].append(trial)
                if diagnostics is not None:
                    trial["diagnostic_before"] = before
                    async with asyncio.timeout(options.start_timeout_s):
                        trial["diagnostic_after"] = await diagnostics()
                if trial["status"] != "ok":
                    report["status"], report["code"] = trial["status"], trial["code"]
                if trial["status"] == "failed":
                    return report
    except TrialError as error:
        report["status"], report["code"] = "failed", str(error)
    except TimeoutError:
        report["status"], report["code"] = "failed", "timeout"
    except httpx.HTTPError:
        report["status"], report["code"] = "failed", "transport_error"
    return report

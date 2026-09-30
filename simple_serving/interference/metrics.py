"""Bounded, numeric diagnostics from the existing control listener.

The counter selection follows held-criterion's throughput/metrics.py at a898060. Label values and raw text never
enter the report. A snapshot is aggregate evidence, not ownership proof or a reason to stop a card.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
import time
from collections.abc import Mapping
from typing import Any

import httpx

from ..config import CLASSES

CHAT = "/v1/chat/completions"
MAX_BYTES = 8_000_000
SAMPLE = re.compile(r'([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)(?:\s+-?\d+)?')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
CLASS_GAUGES = frozenset({"generations_active", "generations_waiting", "counts_active", "counts_waiting",
                          "generation_active_limit", "generation_waiting_limit"})
SCALARS = frozenset({"start_time_seconds", "ours_in_flight", "generation_shared_limit", "count_active_limit"})
STAGES = ("count_place_wait", "count", "generation_place_wait", "wait")
ENGINE_COUNTERS = frozenset({"prompt_tokens_total", "generation_tokens_total", "prefix_cache_queries_total",
                             "prefix_cache_hits_total", "num_preemptions_total", "request_success_total"})
ENGINE_GAUGES = frozenset({"num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"})
ENGINE_STAGES = ("request_queue_time", "request_prefill_time", "request_decode_time")


class DiagnosticError(Exception):
    """A fixed code only; never an HTTP answer or exception's text."""


def parse(text: str, *, engine: bool = False) -> dict[str, float]:
    """Keep only numeric families needed to distinguish admission, preemption and cache reuse."""
    values: dict[str, float] = {}
    prefix = "vllm:" if engine else "simple_serving_"
    for line in text.splitlines():
        match = SAMPLE.fullmatch(line.strip())
        if match is None or not match[1].startswith(prefix):
            continue
        name, labels = match[1].removeprefix(prefix), dict(LABEL.findall(match[2] or ""))
        key = name
        if engine:
            if name not in ENGINE_COUNTERS | ENGINE_GAUGES and not any(
                name == stage + "_seconds" + suffix for stage in ENGINE_STAGES for suffix in ("_sum", "_count")
            ):
                continue
        elif name in CLASS_GAUGES:
            if labels.get("class") not in CLASSES:
                continue
            key += ":" + labels["class"]
        elif name == "requests_total":
            if labels.get("route") != CHAT or labels.get("class") not in (*CLASSES, "none"):
                continue
            key += ":" + labels["class"]
        elif name not in SCALARS:
            if not any(name == stage + "_seconds" + suffix for stage in STAGES for suffix in ("_sum", "_count")):
                continue
            if labels.get("class") not in CLASSES or labels.get("route", CHAT) != CHAT:
                continue
            key += ":" + labels["class"]
        try:
            value = float(match[3])
        except ValueError:
            continue
        if math.isfinite(value) and 0 <= value <= 1e18:
            values[key] = values.get(key, 0.0) + value
    return values


async def read(client: httpx.AsyncClient, base: str, path: str, key: str) -> bytes:
    """Read at most MAX_BYTES and five seconds, with no redirects or raw errors."""
    try:
        async with asyncio.timeout(5), client.stream(
            "GET", base + path, headers={"Authorization": f"Bearer {key}"}, timeout=5,
        ) as response:
            if response.status_code != 200:
                raise DiagnosticError("diagnostics_unavailable")
            data = bytearray()
            async for part in response.aiter_bytes():
                data.extend(part)
                if len(data) > MAX_BYTES:
                    raise DiagnosticError("diagnostics_invalid")
    except (httpx.HTTPError, TimeoutError):
        raise DiagnosticError("diagnostics_unavailable") from None
    return bytes(data)


class Monitor:
    """An experiment's snapshots, relative times and restart checks; it never controls the card."""

    def __init__(self, client: httpx.AsyncClient, base: str, key: str, boot: str) -> None:
        self.client, self.base, self.key, self.boot = client, base, key, boot
        self.started = time.monotonic()
        self.samples: list[dict[str, Any]] = []
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self.gateway_start: float | None = None
        self.boot_changed = False
        self.sampling_capped = False

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            start = time.monotonic()
            entry: dict[str, Any] = {"at_ms": round((start - self.started) * 1000, 3), "gateway": {}, "engine": {},
                                      "missing": [], "reset_suspected": False}
            for engine, path in ((False, "/metrics"), (True, "/metrics/engine")):
                lane = "engine" if engine else "gateway"
                try:
                    raw = await read(self.client, self.base, path, self.key)
                    values = parse(raw.decode("utf-8", errors="replace"), engine=engine)
                    entry[lane] = values
                    if not values:
                        entry["missing"].append(lane)
                except DiagnosticError:
                    entry["missing"].append(lane)
            gateway_start = entry["gateway"].get("start_time_seconds")
            if self.gateway_start is None:
                self.gateway_start = gateway_start
            elif gateway_start is not None and gateway_start != self.gateway_start:
                entry["reset_suspected"] = True
            entry["scrape_ms"] = round((time.monotonic() - start) * 1000, 3)
            self.samples.append(entry)
            return entry

    def start(self) -> None:
        self._task = asyncio.create_task(self._sample())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _sample(self) -> None:
        while len(self.samples) < 1200:
            await self.snapshot()
            await asyncio.sleep(0.5)
        self.sampling_capped = True

    async def check_boot(self) -> None:
        raw = await read(self.client, self.base, "/v1/state", self.key)
        try:
            state = httpx.Response(200, content=raw).json()
            self.boot_changed = not isinstance(state, dict) or state.get("boot_id") != self.boot
        except (ValueError, RecursionError):
            self.boot_changed = True


def changes(before: Mapping[str, float], after: Mapping[str, float]) -> tuple[dict[str, float], bool]:
    """Missing baselines and decreasing counters cannot support a valid delta."""
    moved = {key: value - before[key] for key, value in after.items() if key in before}
    return moved, any(value < 0 for value in moved.values())


def capacity(snapshot: Mapping[str, Any], concurrency: int, aggressor: str) -> None:
    gateway = snapshot["gateway"]
    if (any(gauge + ":" + cls not in gateway for cls in CLASSES
            for gauge in ("generations_active", "generations_waiting", "counts_active", "counts_waiting"))
            or any(gauge not in snapshot["engine"] for gauge in ("num_requests_running", "num_requests_waiting"))):
        raise DiagnosticError("diagnostics_unavailable")
    needed = {"reader": 1}
    needed[aggressor] = needed.get(aggressor, 0) + concurrency - 1
    for cls, jobs in needed.items():
        limit = gateway.get("generation_active_limit:" + cls)
        if limit is None:
            raise DiagnosticError("limits_unknown")
        if jobs > limit:
            raise DiagnosticError("admission_limited")
    shared = sum(jobs for cls, jobs in needed.items() if cls != "reader")
    limit = gateway.get("generation_shared_limit")
    if limit is None:
        raise DiagnosticError("limits_unknown")
    if shared > limit:
        raise DiagnosticError("admission_limited")
    if any(value > 0 for key, value in gateway.items() if key.startswith(
        ("generations_active:", "generations_waiting:", "counts_active:", "counts_waiting:")
    )) or any(snapshot["engine"].get(key, 0) > 0 for key in ("num_requests_running", "num_requests_waiting")):
        raise DiagnosticError("target_busy")

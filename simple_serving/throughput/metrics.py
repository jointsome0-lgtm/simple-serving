"""The engine's /metrics, read before and after every pass and sampled during it (docs/speed.md, item 1).

vLLM 0.30.0 exposes Prometheus text (`vllm/v1/metrics/loggers.py:499-712`, `vllm/v1/spec_decode/metrics.py:229-255`).
Its counters are cumulative, where the periodic log line resets, so a pass's counts are the difference of two reads.
The gauges say what holds at one moment, so a sampler keeps the largest value seen while a pass runs. Only the names
below are kept, each an integer; a label value outside the fixed ones is dropped, so nothing the engine printed passes
through but numbers.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Mapping
from typing import Self

import httpx

SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")  # vllm/v1/metrics/stats.py:394-398
REASONS = ("capacity", "deferred")  # vllm:num_requests_waiting_by_reason
POSITIONS = tuple(str(position) for position in range(16))  # the drafter's positions, far above any count here
# vLLM's name without `vllm:` and a counter's `_total`, and the label it is split by with that label's values.
COUNTERS: dict[str, tuple[str, tuple[str, ...]] | None] = {
    "prompt_tokens": None, "generation_tokens": None, "prompt_tokens_cached": None,
    "prompt_tokens_by_source": ("source", SOURCES),
    "prefix_cache_queries": None, "prefix_cache_hits": None,
    "external_prefix_cache_queries": None, "external_prefix_cache_hits": None,
    "num_preemptions": None,
    "spec_decode_num_drafts": None, "spec_decode_num_draft_tokens": None, "spec_decode_num_accepted_tokens": None,
    "spec_decode_num_accepted_tokens_per_pos": ("position", POSITIONS),
}
GAUGES: dict[str, tuple[str, tuple[str, ...]] | None] = {
    "num_requests_running": None, "num_requests_waiting": None,
    "num_requests_waiting_by_reason": ("reason", REASONS),
    "kv_cache_usage_perc": None,  # a fraction, kept as thousandths
}
SAMPLE = re.compile(r"vllm:(?P<name>[a-z_]+?)(?:_total)?(?:\{(?P<labels>[^}]*)\})? (?P<value>\S+)(?: -?\d+)?")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
METRICS_BYTES = 4 << 20  # far above vLLM's text for one engine
SAMPLE_S = 0.5


def parse(text: str) -> dict[str, int]:
    """The kept counters and gauges of one read, summed over engines: `name` or `name_<label value>`. A gauge that is
    a fraction becomes thousandths."""
    found: dict[str, float] = {}
    for line in text.splitlines():
        sample = SAMPLE.fullmatch(line.strip())
        if sample is None:
            continue
        name = sample["name"]
        if name not in COUNTERS and name not in GAUGES:
            continue
        kept = COUNTERS[name] if name in COUNTERS else GAUGES[name]
        try:
            value = float(sample["value"])
        except ValueError:
            continue
        key = name
        if kept is not None:
            label, values = kept
            labels = dict(LABEL.findall(sample["labels"] or ""))
            if labels.get(label) not in values:
                continue
            key = f"{name}_{labels[label]}"
        found[key] = found.get(key, 0.0) + value
    return {key: round(value * 1000) if key == "kv_cache_usage_perc" else round(value) for key, value in found.items()}


def is_gauge(key: str) -> bool:
    return any(key == name or (key.startswith(f"{name}_") and GAUGES[name] is not None) for name in GAUGES)


def counters(read: Mapping[str, int]) -> dict[str, int]:
    return {key: value for key, value in read.items() if not is_gauge(key)}


def gauges(read: Mapping[str, int]) -> dict[str, int]:
    return {key: value for key, value in read.items() if is_gauge(key)}


def delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    """What the counters counted between two reads. A counter that went down means the engine started again between
    them, and is reported as -1."""
    moved = {}
    for key, value in counters(after).items():
        change = value - before.get(key, 0)
        moved[key] = change if change >= 0 else -1
    return moved


def acceptance(moved: Mapping[str, int]) -> list[float]:
    """The drafter's acceptance by position over a pass, unconditional as vLLM's synthetic rates are
    (`vllm/config/speculative.py:525-531`): position i's accepted tokens over the drafts."""
    drafts = moved.get("spec_decode_num_drafts", 0)
    if drafts <= 0:
        return []
    rates = []
    for position in POSITIONS:
        accepted = moved.get(f"spec_decode_num_accepted_tokens_per_pos_{position}")
        if accepted is None:
            break
        rates.append(round(accepted / drafts, 4))
    return rates


async def read(client: httpx.AsyncClient, base_url: str) -> dict[str, int]:
    """One read of the engine's /metrics. An answer that is not 200 or is too long reads as nothing."""
    async with client.stream("GET", f"{base_url}/metrics") as response:
        if response.status_code != 200:
            return {}
        data = b""
        async for piece in response.aiter_bytes():
            data += piece
            if len(data) > METRICS_BYTES:
                return {}
    return parse(data.decode("utf-8", errors="replace"))


class Sampler:
    """The largest value of each gauge while a pass runs, read every SAMPLE_S."""

    def __init__(self, client: httpx.AsyncClient, base_url: str, interval_s: float = SAMPLE_S) -> None:
        self.client, self.base_url, self.interval_s = client, base_url, interval_s
        self.peaks: dict[str, int] = {}
        self.samples = 0
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._task is not None
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def _run(self) -> None:
        while True:
            with contextlib.suppress(httpx.HTTPError):
                self.take(gauges(await read(self.client, self.base_url)))
            await asyncio.sleep(self.interval_s)

    def take(self, values: Mapping[str, int]) -> None:
        self.samples += 1
        for key, value in values.items():
            self.peaks[key] = max(value, self.peaks.get(key, value))

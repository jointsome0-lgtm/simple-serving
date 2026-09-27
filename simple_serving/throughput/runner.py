"""One pass of a workload through the gateway, and what it measured (docs/speed.md, "The metric").

Before a pass the harness counts each request's prompt with the gateway's count route (contract section 5). The pass
then runs its lanes, `at` at once, each lane's requests one after another with the lane's gaps, and reads the engine's
/metrics before and after it while a sampler keeps the gauges' peaks. A request's numbers are its usage chunk's counts
and the gateway's measurements (section 4), wait and first token included, and the harness's own time. Its text stays
in memory and is dropped; a failure keeps a fixed code.

The pass's numbers: output tokens a second (decode) and prompt tokens read a second (prefill: the prompt less what the
cache held), each summed over the pass and divided by its elapsed time, and both together; per request, the median
rate of decoding after the first token and of reading before it, and the first token's time (median, 90th percentile
and largest), the gateway's wait included.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from .. import smoke
from . import metrics
from .workloads import Request

CLASS = "internal"


@dataclass(frozen=True)
class Target:
    public: str  # the gateway's public listener
    engine: str  # the engine itself, for /metrics
    key: str  # the harness's key, never written anywhere
    alias: str

    def headers(self, scope: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}", "X-Simple-Serving-Class": CLASS,
                "X-Simple-Serving-Scope": scope}


@dataclass
class Outcome:
    """One request's numbers. `code` is None when it streamed whole; otherwise a fixed code."""

    lane: int
    turn: int
    counted: int | None = None  # the count route's input tokens, before the pass
    code: str | None = None
    finish: str | None = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    wait_ms: int = 0
    first_token_ms: int = 0  # from acceptance, the wait included
    total_ms: int = 0
    client_ms: int = 0  # the harness's own, from sending to the end of the stream
    started_ms: int = 0  # from the pass's start

    @property
    def read_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_tokens)


def ms(seconds: float) -> int:
    return max(0, round(seconds * 1000))


async def count(client: httpx.AsyncClient, target: Target, request: Request) -> int | None:
    """The gateway's count of a request's prompt, or None."""
    try:
        async with client.stream("POST", target.public + smoke.COUNT, json=request.body,
                                 headers=target.headers(request.scope)) as response:
            answer = smoke.Answer(response.status_code, await smoke.read_json(response))
    except httpx.HTTPError:
        return None
    tokens = answer.field("input_tokens")
    return tokens if answer.status == 200 and smoke.is_count(tokens) else None


async def generate(client: httpx.AsyncClient, target: Target, request: Request) -> Outcome:
    outcome, begun = Outcome(request.lane, request.turn), time.monotonic()
    try:
        async with client.stream("POST", target.public + smoke.CHAT, json=request.body,
                                 headers=target.headers(request.scope)) as response:
            stream = await smoke.read_stream(response, target.alias)
    except httpx.HTTPError:
        stream = None
    outcome.client_ms = ms(time.monotonic() - begun)
    if stream is None:
        outcome.code = "connection"
    elif not stream.ok or stream.usage is None:
        outcome.code = stream.code or (stream.broken[0] if stream.broken else "stream")
    else:
        usage = stream.usage
        outcome.finish = stream.finish
        outcome.input_tokens, outcome.output_tokens = usage["prompt_tokens"], usage["completion_tokens"]
        outcome.cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        measured = usage["simple_serving"]
        outcome.wait_ms, outcome.first_token_ms = measured["wait_ms"], measured["first_token_ms"]
        outcome.total_ms = measured["total_ms"]
    return outcome


async def counted(client: httpx.AsyncClient, target: Target, lanes: Sequence[Sequence[Request]]) -> list[int | None]:
    """Every request's count, lane after lane, one at a time."""
    return [await count(client, target, request) for lane in lanes for request in lane]


async def run_pass(client: httpx.AsyncClient, target: Target, lanes: Sequence[Sequence[Request]], at: int, *,
                   pause: Callable[[float], Awaitable[None]] = asyncio.sleep,
                   sample_s: float = metrics.SAMPLE_S) -> dict[str, Any]:
    """Count, then run the lanes `at` at once, and return the pass's numbers."""
    counts = await counted(client, target, lanes)
    before = await metrics.read(client, target.engine)
    pending, outcomes = list(enumerate(lanes)), []
    offsets = [sum(len(lane) for lane in lanes[:index]) for index in range(len(lanes))]
    start, began = time.monotonic(), time.time()

    async def worker() -> None:
        while pending:
            index, lane = pending.pop(0)
            for position, request in enumerate(lane):
                if request.gap_s:
                    await pause(request.gap_s)
                started = ms(time.monotonic() - start)
                outcome = await generate(client, target, request)
                outcome.counted, outcome.started_ms = counts[offsets[index] + position], started
                outcomes.append(outcome)

    async with metrics.Sampler(client, target.engine, sample_s) as sampler:
        await asyncio.gather(*(worker() for _ in range(max(1, min(at, len(lanes))))))
    elapsed_ms, window = ms(time.monotonic() - start), [round(began, 3), round(time.time(), 3)]
    after = await metrics.read(client, target.engine)
    moved = metrics.delta(before, after)
    outcomes.sort(key=lambda outcome: (outcome.started_ms, outcome.lane, outcome.turn))
    return {"at": at, "lanes": len(lanes), "window": window, **summary(outcomes, elapsed_ms),
            "metrics": moved, "acceptance": metrics.acceptance(moved), "peaks": sampler.peaks,
            "samples": sampler.samples, "outcomes": [asdict(outcome) for outcome in outcomes]}


def summary(outcomes: Sequence[Outcome], elapsed_ms: int) -> dict[str, Any]:
    """The pass's counts and rates. Rates are tokens a second, rounded to one decimal."""
    done = [outcome for outcome in outcomes if outcome.code is None]
    failed: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.code is not None:
            failed[outcome.code] = failed.get(outcome.code, 0) + 1
    output = sum(outcome.output_tokens for outcome in done)
    read = sum(outcome.read_tokens for outcome in done)
    seconds = elapsed_ms / 1000
    decoding = [(outcome.output_tokens - 1) / ((outcome.total_ms - outcome.first_token_ms) / 1000) for outcome in done
                if outcome.output_tokens > 1 and outcome.total_ms > outcome.first_token_ms]
    reading = [outcome.read_tokens / ((outcome.first_token_ms - outcome.wait_ms) / 1000) for outcome in done
               if outcome.read_tokens and outcome.first_token_ms > outcome.wait_ms]
    first = [outcome.first_token_ms for outcome in done]
    return {
        "requests": len(outcomes), "done": len(done), "failed": failed, "elapsed_ms": elapsed_ms,
        "input_tokens": sum(outcome.input_tokens for outcome in done),
        "cached_tokens": sum(outcome.cached_tokens for outcome in done), "read_tokens": read, "output_tokens": output,
        "count_mismatches": sum(1 for outcome in done if outcome.counted != outcome.input_tokens),
        "finished_length": sum(1 for outcome in done if outcome.finish == "length"),
        "output_tokens_per_s": rate(output, seconds), "read_tokens_per_s": rate(read, seconds),
        "total_tokens_per_s": rate(output + read, seconds),
        "decode_tokens_per_s_median": round(statistics.median(decoding), 1) if decoding else None,
        "prefill_tokens_per_s_median": round(statistics.median(reading), 1) if reading else None,
        "first_token_ms": percentiles(first), "wait_ms_max": max((outcome.wait_ms for outcome in done), default=None),
    }


def rate(tokens: int, seconds: float) -> float | None:
    return round(tokens / seconds, 1) if seconds > 0 else None


def percentiles(values: Sequence[int]) -> dict[str, int] | None:
    """The median, the 90th percentile (nearest rank) and the largest."""
    if not values:
        return None
    ordered = sorted(values)
    return {"median": round(statistics.median(ordered)), "p90": ordered[max(0, -(-9 * len(ordered) // 10) - 1)],
            "max": ordered[-1]}


async def chars_per_token(client: httpx.AsyncClient, target: Target, request: Request) -> float | None:
    """Characters of a request's messages a token, by the gateway's count: what sizes a target context."""
    tokens = await count(client, target, request)
    chars = sum(len(message["content"]) for message in request.body["messages"])
    return round(chars / tokens, 3) if tokens else None

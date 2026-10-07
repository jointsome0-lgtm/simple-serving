"""The worlds measurement counts what the engine reports, through the real gateway in front of the fake engine."""

from __future__ import annotations

import itertools
import json

import httpx
import pytest

from simple_serving.interference import workload, worlds
from simple_serving.interference.runner import Target

SHAPE = worlds.Shape(worlds=2, sessions=3, turns=3, system_tokens=600, first_tokens=1200, last_tokens=2400,
                     growth_tokens=300, answer_tokens=200, request_timeout_s=10)


async def measured(capacity: int) -> tuple[dict[str, object], worlds.PrefixCache]:
    async with (worlds.dry_gateway(capacity, SHAPE.worlds) as gateway,
                httpx.AsyncClient(timeout=None, trust_env=False) as client):
        report = await worlds.measure(client, Target(gateway.public, gateway.key, gateway.alias), SHAPE)
    encoded = json.dumps(report, allow_nan=False)
    assert gateway.key not in encoded and workload.UNIT.strip() not in encoded and "keeper" not in encoded
    return report, gateway.cache


@pytest.mark.anyio
async def test_the_report_counts_the_engines_own_tokens_and_a_cache_too_small_shows_in_them() -> None:
    large, cache = await measured(worlds.LARGE)
    assert (large["status"], large["code"]) == ("ok", "ok")
    rows, totals = large["requests"], large["totals"]
    assert isinstance(rows, list) and isinstance(totals, dict) and len(rows) == 2 * 3 * 3 == totals["requests"]
    # Every request's prompt and cached tokens are the fake engine's, and the sums are theirs.
    assert sorted((row["prompt_tokens"], row["cached_tokens"]) for row in rows) == sorted(cache.seen)
    assert totals["prompt_tokens"] == sum(prompt for prompt, _ in cache.seen)
    assert totals["cached_tokens"] == sum(cached for _, cached in cache.seen) > 0
    answer = -(-len("".join(worlds.ANSWER)) // 4)
    assert all(row["completion_tokens"] == answer for row in rows) and totals["completion_tokens"] == answer * 18
    assert all(row["cached_tokens"] == 0 for row in rows if row["turn"] == 0)  # nothing is shared between sessions
    # A later turn has its prefix behind it, but for the last block and a session that started over.
    later = [row for row in rows if row["turn"]]
    assert sum(row["cached_tokens"] >= row["prompt_tokens"] - 2 * worlds.BLOCK_TOKENS - 300 for row in later) >= 9
    for world in range(2):  # the sessions of a world take turns, one request at a time
        mine = sorted((row for row in rows if row["world"] == world), key=lambda row: row["started_ms"])
        assert all(left["started_ms"] + left["total_ms"] <= right["started_ms"] + 1
                   for left, right in itertools.pairwise(mine))
    small, _ = await measured(1024)
    assert isinstance(small["totals"], dict) and small["totals"]["cached_tokens"] < totals["cached_tokens"] / 4

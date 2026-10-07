"""The probes tell a prefix from the GPU's cache, one from the copy and one read again apart, and a new schema."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from simple_serving.interference import probes, workload, worlds


async def reloaded(copy: int) -> dict[str, Any]:
    async with (worlds.dry_gateway(5000, 1, copy) as gateway,
                httpx.AsyncClient(timeout=None, trust_env=False) as client):
        to, _ = await worlds.target(client, gateway.public, gateway.control, gateway.key, gateway.control_key)
        report = await probes.reload(client, to, gateway.control, gateway.control_key, tokens=2000, rounds=2,
                                     fillers=3)
    encoded = json.dumps(report, allow_nan=False)
    assert gateway.key not in encoded and gateway.control_key not in encoded and workload.UNIT.strip() not in encoded
    return report


@pytest.mark.anyio
async def test_a_prefix_that_fell_out_of_the_gpus_cache_is_seen_to_come_from_the_copy_or_to_be_read_again() -> None:
    without, copied = await reloaded(0), await reloaded(1_000_000)
    for report in (without, copied):
        rows = report["rows"]
        assert report["status"] == "ok"
        assert [row["kind"] for row in rows] == ["cold", "gpu", "filler", "filler", "filler", "back"] * 2
        for row in rows:
            hit = (row["gpu_hit_tokens"], row["copy_hit_tokens"])
            if row["kind"] in ("cold", "filler"):
                assert hit == (0, 0) and row["cached_tokens"] == 0
            elif row["kind"] == "gpu":
                assert hit == (row["cached_tokens"], 0) and row["cached_tokens"] > 1500
    assert without["back_from"] == "read_again" and copied["back_from"] == "copy"
    assert all(row["copy_hit_tokens"] == row["cached_tokens"] > 1500 and row["gpu_hit_tokens"] == 0
               for row in copied["rows"] if row["kind"] == "back")
    assert all(row["copy_hit_tokens"] == row["gpu_hit_tokens"] == 0
               for row in without["rows"] if row["kind"] == "back")


@pytest.mark.anyio
async def test_a_new_schema_costs_the_stand_ins_compilation_and_a_repeated_one_nothing() -> None:
    assert json.dumps(probes.world_schema(8)) != json.dumps(probes.world_schema(8))
    async with (worlds.dry_gateway(worlds.LARGE, 1) as gateway,
                httpx.AsyncClient(timeout=None, trust_env=False) as client):
        to, _ = await worlds.target(client, gateway.public, gateway.control, gateway.key, gateway.control_key)
        report = await probes.schemas(client, to, tokens=800, pairs=4, labels=8)
    assert report["status"] == "ok"
    assert [row["kind"] for row in report["rows"]] == ["warm"] * 2 + ["repeated", "new"] * 4
    assert all(row["cached_tokens"] > 500 for row in report["rows"][1:])  # the schema is no part of the prompt
    assert worlds.COMPILE_S * 500 < report["compile_ms"] < worlds.COMPILE_S * 3000

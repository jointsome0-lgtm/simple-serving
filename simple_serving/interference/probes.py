"""Two small measurements through a text gateway: what a prefix costs to bring back, and what a new schema costs.

    uv run python -m simple_serving.interference.probes reload
    uv run python -m simple_serving.interference.probes schemas
    uv run python -m simple_serving.interference.probes reload --run --config ~/.config/simple-serving/cli.json \
        --out reload.json

`reload` asks one session for a turn (`cold`: its whole prompt is read), at once for the next (`gpu`: its prefix is
in the GPU's cache), then sends other sessions' prompts until the GPU's cache has dropped it, and asks for a third
(`back`). With a copy of the KV cache in RAM the third comes from the copy, and without one it is read again. Each row
has the time to the first token and how many prompt tokens the engine served from the GPU's cache and from the copy,
by its own counters, so a `back` row says by itself where its prefix came from. Requests go one at a time, since the
counters belong to the whole engine.

`schemas` sends one prompt again and again under a strict schema of the shape sagents' world uses: about ten fields,
nullable strings, an enum with null, arrays of objects whose fields are enums of labels. Every second request has the
same schema as before and every other one new labels, so new text. The prompt is the same and cached, so the
difference of their first tokens is what one compilation of a grammar costs.

Without --run each is a dry run through this repository's gateway in front of the fake engine, whose cache, copy and
compile time are stand-ins (worlds.py): it shows the report and that the client tells the cases apart, and measures
nothing. Reports hold counts and times, never a prompt, a key or an answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import secrets
import statistics
from pathlib import Path
from typing import Any

import httpx

from ..cli import Refusal, read_config
from . import metrics, workload
from .cli import RunError, _base
from .runner import Reading, Target, TrialError, _input_tokens, _stream
from .worlds import SAMPLE_UNITS, SCHEMA, dry_gateway, gained, hits, lines, target

TIMEOUT_S = 180


async def ask(client: httpx.AsyncClient, to: Target, body: dict[str, Any], scope: str) -> dict[str, Any]:
    """One request, as a row: its status, first token and whole time, and the tokens of its usage."""
    reading, row = Reading(), {"status": "ok"}
    try:
        async with asyncio.timeout(TIMEOUT_S):
            await _stream(client, to, body, "reader", scope, reading, asyncio.Event(), 1, TIMEOUT_S)
    except TrialError as error:
        row["status"] = str(error)
    except TimeoutError:
        row["status"] = "timeout"
    except httpx.HTTPError:
        row["status"] = "transport_error"
    usage = reading.usage or {}
    return row | {"ttft_ms": round((reading.chunks[0] - reading.started) * 1000, 3) if reading.chunks else None,
                  "total_ms": round((reading.ended - reading.started) * 1000, 3),
                  **{name: usage.get(name) for name in ("prompt_tokens", "cached_tokens", "completion_tokens")}}


def body_of(alias: str, text: str, schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    body = workload.body(alias, [{"role": "user", "content": text + "\nAnswer with what happens next."}],
                         max_tokens, 0)
    body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "strict": True,
                                                                      "schema": schema}}
    return body


async def per_line(client: httpx.AsyncClient, to: Target) -> float:
    sample = workload.body(to.alias, [{"role": "user", "content": lines("note 00000000", 0, SAMPLE_UNITS)}], 1, 0)
    return await _input_tokens(client, to, sample, TIMEOUT_S) / SAMPLE_UNITS


def median(rows: list[dict[str, Any]], kind: str) -> float | None:
    values = [row["ttft_ms"] for row in rows if row["kind"] == kind and row["ttft_ms"] is not None]
    return round(statistics.median(values), 3) if values else None


async def reload(client: httpx.AsyncClient, to: Target, control: str, control_key: str, *, tokens: int, rounds: int,
                 fillers: int) -> dict[str, Any]:
    """`rounds` times: a session's cold turn, its next from the GPU's cache, `fillers` other sessions, its third."""
    line = await per_line(client, to)
    count, grow, rows = max(1, round(tokens / line)), max(1, round(300 / line)), []

    async def step(kind: str, text: str, scope: str, round_index: int) -> None:
        before = await hits(client, control, control_key)
        row = await ask(client, to, body_of(to.alias, text, SCHEMA, 120), scope)
        served = gained(before, await hits(client, control, control_key))
        rows.append({"round": round_index, "kind": kind, **row, "gpu_hit_tokens": served["local"],
                     "copy_hit_tokens": served["external"], "preempted": served["preempted"]})

    for index in range(rounds):
        tag, scope = "note " + secrets.token_hex(4), workload.scope()
        await step("cold", lines(tag, 0, count), scope, index)
        await step("gpu", lines(tag, 0, count + grow), scope, index)
        for _ in range(fillers):
            await step("filler", lines("note " + secrets.token_hex(4), 0, count), workload.scope(), index)
        await step("back", lines(tag, 0, count + 2 * grow), scope, index)
    back = [row for row in rows if row["kind"] == "back" and row["status"] == "ok"]
    failed = sum(row["status"] != "ok" for row in rows)
    return {"schema_version": 1, "status": "failed" if failed else "ok", "code": "requests_failed" if failed else "ok",
            "tokens": tokens, "rounds": rounds, "fillers": fillers, "rows": rows,
            "ttft_ms": {kind: median(rows, kind) for kind in ("cold", "gpu", "back")},
            # Where the third turns' prefixes came from: all from the copy, all read again, or neither clearly.
            "back_from": ("copy" if back and all(row["copy_hit_tokens"] >= row["prompt_tokens"] // 2 for row in back)
                          else "gpu" if back and all(row["gpu_hit_tokens"] >= row["prompt_tokens"] // 2 for row in back)
                          else "read_again" if back and not any(row["cached_tokens"] for row in back) else "mixed")}


def world_schema(labels: int) -> dict[str, Any]:
    """A schema of the shape sagents' world sends (its src/live.ts at 5aacbd5, `resultSchemaOf`), with `labels` new
    labels in each list, so that its text is new."""
    names = [f"thing-{secrets.token_hex(3)}" for _ in range(labels)]
    text = {"type": ["string", "null"]}
    one, said = {"type": "string", "enum": names}, {"type": "string"}

    def entries(properties: dict[str, Any]) -> dict[str, Any]:
        return {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                           "required": list(properties), "properties": properties}}

    properties = {"search": {"type": "boolean"}, "finds": {"type": "array", "items": one},
                  "moves": entries({"what": one, "n": {"type": "integer"}, "to": one}),
                  "sets": entries({"what": one, "state": one}), "poses": entries({"of": one, "text": said}),
                  "wakes": {"type": "array", "items": one}, "feels": entries({"of": one, "text": said}),
                  "place": {"type": ["string", "null"], "enum": [*names, None]}, "beyond": text, "result": text}
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


async def schemas(client: httpx.AsyncClient, to: Target, *, tokens: int, pairs: int, labels: int) -> dict[str, Any]:
    """`pairs` times: the first schema again, then one with new labels, at one prompt."""
    text = lines("note " + secrets.token_hex(4), 0, max(1, round(tokens / await per_line(client, to))))
    scope, first, rows = workload.scope(), world_schema(labels), []
    order = [("warm", first), ("warm", first)]
    for _ in range(pairs):
        order += [("repeated", first), ("new", world_schema(labels))]
    for kind, schema in order:
        rows.append({"kind": kind, **await ask(client, to, body_of(to.alias, text, schema, 96), scope)})
    repeated, new = median(rows, "repeated"), median(rows, "new")
    failed = sum(row["status"] != "ok" for row in rows)
    return {"schema_version": 1, "status": "failed" if failed else "ok", "code": "requests_failed" if failed else "ok",
            "tokens": tokens, "pairs": pairs, "labels": labels, "schema_bytes": len(json.dumps(first)), "rows": rows,
            "ttft_ms": {"repeated": repeated, "new": new},
            "compile_ms": round(new - repeated, 3) if new is not None and repeated is not None else None}


async def _measure(args: argparse.Namespace, public: str, control: str, key: str, control_key: str) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": 1, "status": "failed", "code": "not_started"}
    try:
        async with (httpx.AsyncClient(timeout=None, follow_redirects=False, trust_env=False) as client,
                    asyncio.timeout(args.max_seconds)):
            to, context = await target(client, public, control, key, control_key)
            if args.tokens + 1024 > context:
                raise RunError("context_limit")
            if args.command == "reload":
                report = await reload(client, to, control, control_key, tokens=args.tokens, rounds=args.rounds,
                                      fillers=args.fillers)
            else:
                report = await schemas(client, to, tokens=args.tokens, pairs=args.pairs, labels=args.labels)
    except (RunError, metrics.DiagnosticError, TrialError) as error:
        report |= {"status": "failed", "code": str(error)}
    except TimeoutError:
        report |= {"status": "failed", "code": "budget_expired"}
    except httpx.HTTPError:
        report |= {"status": "failed", "code": "transport_error"}
    return report


async def _dry(args: argparse.Namespace) -> dict[str, Any]:
    """`schemas` once; `reload` twice, without a copy and with one, on a cache that holds two of its prompts."""
    runs = []
    for name, copy in ((("no_copy", 0), ("copy", 2_000_000)) if args.command == "reload" else (("stand_in", 0),)):
        async with dry_gateway(2 * args.tokens + 1024, 1, copy) as gateway:
            run = await _measure(args, gateway.public, gateway.control, gateway.key, gateway.control_key)
        runs.append({"engine": name, **{field: value for field, value in run.items() if field != "rows"}})
    failed = [run["code"] for run in runs if run["status"] != "ok"]
    return {"schema_version": 1, "synthetic": True, "status": "failed" if failed else "ok",
            "code": failed[0] if failed else "ok", "runs": runs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m simple_serving.interference.probes",
                                     description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("command", choices=("reload", "schemas"))
    parser.add_argument("--run", action="store_true", help="against a gateway; without it, the dry run")
    parser.add_argument("--config", type=Path, help="--run: owner-only CLI configuration with client/control keys")
    parser.add_argument("--out", type=Path, help="new JSON report file; required with --run, never overwritten")
    parser.add_argument("--public", default="http://127.0.0.1:8080")
    parser.add_argument("--control", default="http://127.0.0.1:8081")
    parser.add_argument("--tokens", type=int, help="the prompt's tokens: 8000 for reload, 1500 for schemas")
    parser.add_argument("--rounds", type=int, default=3, help="reload: sessions tried")
    parser.add_argument("--fillers", type=int, default=12, help="reload: other sessions between a session's turns")
    parser.add_argument("--pairs", type=int, default=12, help="schemas: repeated and new schemas, each")
    parser.add_argument("--labels", type=int, default=8, help="schemas: labels in each list")
    parser.add_argument("--max-seconds", type=float, default=900, help="whole run budget, at most 3600 seconds")
    args = parser.parse_args(argv)
    if args.run != (args.config is not None and args.out is not None):
        parser.error("--run goes with --config and --out, and they with it")
    args.tokens = args.tokens or (8000 if args.command == "reload" else 1500)
    if not args.run and args.command == "reload":
        args.fillers = min(args.fillers, 4)
    if (not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 3600 or not 64 <= args.tokens <= 200000
            or not 1 <= args.rounds <= 20 or not 1 <= args.fillers <= 200 or not 1 <= args.pairs <= 100
            or not 1 <= args.labels <= 512):
        parser.error("dimensions outside bounds")
    output = None
    report: dict[str, Any]
    try:
        # Reserve a new owner-only report before any request could start.
        if args.out is not None:
            output = os.fdopen(os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")
        if not args.run:
            report = asyncio.run(_dry(args))
        else:
            public, control = _base(args.public), _base(args.control)
            config = read_config(args.config)
            if not all(isinstance(config.get(name), str) and config[name] for name in ("client_key", "control_key")):
                raise RunError("config_keys_missing")
            report = asyncio.run(_measure(args, public, control, config["client_key"], config["control_key"]))
    except Refusal:
        report = {"status": "failed", "code": "config_invalid"}
    except RunError as error:
        report = {"status": "failed", "code": str(error)}
    except OSError:
        report = {"status": "failed", "code": "output_or_config_unavailable"}
    except KeyboardInterrupt:
        report = {"status": "failed", "code": "interrupted"}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output is not None:
        with output:
            output.write(text)
    else:
        print(text, end="")
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

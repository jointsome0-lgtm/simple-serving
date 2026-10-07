"""Many worlds at once through a text gateway: what a turn costs, and how much of each prompt came from the cache.

    uv run python -m simple_serving.interference.worlds
    uv run python -m simple_serving.interference.worlds run --config ~/.config/simple-serving/cli.json \
        --out worlds.json --worlds 4 --turns 8

A world is some sessions, 5 unless told otherwise, each with a system text of its own and a request that grows by a
few lines a turn: a growing prefix, as a resident's or a world's conversation is. The sessions of a world take turns,
one request at a time, and the worlds run at once. A request that grows past --last-tokens starts over with new text,
as after a rewrite of its memory. Each answer is short and under a strict JSON schema, with thinking off. All of a
world's sessions share one opaque reader scope, so the gateway gives them one cache salt.

The report holds, for every request, its prompt tokens, how many of them the engine read from its cache, the time to
the first token and the whole time, and for the run its requests and output tokens a second. It holds no prompt, key
or answer. The text is invented lines, sized by one count through the gateway.

With no arguments, the dry run: the real gateway in front of the fake engine, whose cache is the stand-in below, once
small, once small with a copy behind it and once large. It shows that the client counts what the engine reports, and
what a cache too small for the worlds looks like in these numbers, with a copy and without one. It is not a GPU, and
its times and its sizes predict nothing. `totals.engine_hit_tokens` are the engine's own counters over the run: prompt
tokens served from the GPU's cache (`local`) and from the copy (`external`).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from ..cli import Refusal, read_config
from . import metrics, workload
from .cli import RunError, _base
from .runner import Reading, Target, TrialError, _gaps, _input_tokens, _stream

# The answer's shape: one string long enough for about 80 tokens and short enough for about 200, and a choice.
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["deed", "mood"],
          "properties": {"deed": {"type": "string", "minLength": 240, "maxLength": 600},
                         "mood": {"type": "string", "enum": ["calm", "busy", "tired"]}}}
SAMPLE_UNITS = 64  # the lines of the one count that sizes the text


@dataclass(frozen=True)
class Shape:
    worlds: int = 4
    sessions: int = 5
    turns: int = 8
    system_tokens: int = 2500
    first_tokens: int = 3000
    last_tokens: int = 16000
    growth_tokens: int = 300
    answer_tokens: int = 200
    cls: str = "reader"
    request_timeout_s: float = 180


def lines(tag: str, start: int, count: int) -> str:
    return "".join(f"{tag} {index}. {workload.UNIT}" for index in range(start, start + count))


class Session:
    """One growing conversation: its text is kept, and each turn adds lines to its end."""

    def __init__(self, shape: Shape, tokens_per_line: float, offset: int) -> None:
        self.shape, self.per_line = shape, tokens_per_line
        self.system = lines("rule " + secrets.token_hex(4), 0, self._lines(shape.system_tokens))
        self.offset = offset
        self._start()

    def _lines(self, tokens: int) -> int:
        return max(1, round(tokens / self.per_line))

    def _start(self) -> None:
        self.tag, self.count, self.text = "note " + secrets.token_hex(4), 0, ""
        self._grow(self._lines(self.shape.first_tokens + self.offset))
        self.offset = 0

    def _grow(self, count: int) -> None:
        self.text += lines(self.tag, self.count, count)
        self.count += count

    def body(self, alias: str, seed: int) -> dict[str, Any]:
        if (self.count + 1) * self.per_line > self.shape.last_tokens:
            self._start()
        body = workload.body(alias, [{"role": "system", "content": self.system},
                                     {"role": "user", "content": self.text + "\nAnswer with what happens next."}],
                             self.shape.answer_tokens, seed)
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "turn", "strict": True, "schema": SCHEMA}}
        self._grow(self._lines(self.shape.growth_tokens))
        return body


async def _world(client: httpx.AsyncClient, target: Target, shape: Shape, index: int, per_line: float,
                 origin: float, rows: list[dict[str, Any]]) -> None:
    scope = workload.scope()
    spread = (shape.last_tokens - shape.first_tokens) // shape.sessions
    sessions = [Session(shape, per_line, spread * number) for number in range(shape.sessions)]
    for turn in range(shape.turns):
        for number, session in enumerate(sessions):
            reading, row = Reading(), {"world": index, "session": number, "turn": turn, "status": "ok"}
            try:
                async with asyncio.timeout(shape.request_timeout_s):
                    await _stream(client, target, session.body(target.alias, turn), shape.cls, scope, reading,
                                  asyncio.Event(), 1, shape.request_timeout_s)
            except TrialError as error:
                row["status"] = str(error)
            except TimeoutError:
                row["status"] = "timeout"
            except httpx.HTTPError:
                row["status"] = "transport_error"
            usage = reading.usage or {}
            row |= {"started_ms": round((reading.started - origin) * 1000, 3),
                    "ttft_ms": round((reading.chunks[0] - reading.started) * 1000, 3) if reading.chunks else None,
                    "total_ms": round((reading.ended - reading.started) * 1000, 3),
                    **{name: usage.get(name) for name in ("prompt_tokens", "cached_tokens", "completion_tokens",
                                                          "wait_ms")}}
            rows.append(row)


def totals(rows: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    done = [row for row in rows if row["status"] == "ok"]
    prompt = sum(row["prompt_tokens"] for row in done)
    cached = sum(row["cached_tokens"] or 0 for row in done)
    output = sum(row["completion_tokens"] for row in done)
    later = [row for row in done if row["turn"]]
    return {"requests": len(rows), "failed": len(rows) - len(done), "wall_s": round(wall_s, 3),
            "requests_per_s": round(len(done) / wall_s, 3), "output_tokens_per_s": round(output / wall_s, 3),
            "prompt_tokens": prompt, "cached_tokens": cached, "completion_tokens": output,
            "cached_share": round(cached / prompt, 4) if prompt else None,
            # A turn after a session's first has its prefix behind it: with room in the cache, nearly all of it.
            "later_turns_cached_share": (round(sum(row["cached_tokens"] or 0 for row in later)
                                               / sum(row["prompt_tokens"] for row in later), 4) if later else None),
            "ttft_ms": _gaps([row["ttft_ms"] for row in done if row["ttft_ms"] is not None]),
            "total_ms": _gaps([row["total_ms"] for row in done])}


async def measure(client: httpx.AsyncClient, target: Target, shape: Shape) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": 1, "status": "failed", "code": "not_started", "shape": asdict(shape)}
    try:
        sample = workload.body(target.alias, [{"role": "user", "content": lines("note 00000000", 0, SAMPLE_UNITS)}],
                               1, 0)
        per_line = await _input_tokens(client, target, sample, shape.request_timeout_s) / SAMPLE_UNITS
    except TrialError as error:
        return report | {"code": str(error)}
    rows: list[dict[str, Any]] = []
    origin = time.monotonic()
    await asyncio.gather(*(_world(client, target, shape, index, per_line, origin, rows)
                           for index in range(shape.worlds)))
    rows.sort(key=lambda row: row["started_ms"])
    report |= {"tokens_per_line": round(per_line, 3), "requests": rows,
               "totals": totals(rows, time.monotonic() - origin)}
    failed = report["totals"]["failed"]
    return report | {"status": "failed" if failed else "ok", "code": "requests_failed" if failed else "ok"}


COUNTERS = {"prefix_cache_hits": "local", "external_prefix_cache_hits": "external", "num_preemptions": "preempted"}
HITS = re.compile(rf"^vllm:({'|'.join(COUNTERS)})_total(?:\{{[^}}]*\}})?\s+(\S+)", re.MULTILINE)


async def hits(client: httpx.AsyncClient, control: str, control_key: str) -> dict[str, int]:
    """The prompt tokens the engine has served from the GPU's cache (`local`) and from the copy in RAM (`external`)
    since its start, by vLLM 0.30.0's two counters (vllm/v1/metrics/loggers.py:601,626) through the gateway's
    relay, and the requests it has put back for want of room in the GPU's cache (`preempted`, :667). A request's
    own `cached_tokens` does not say which of the two stores it was."""
    found = dict.fromkeys(COUNTERS.values(), 0.0)
    text = (await metrics.read(client, control, "/metrics/engine", control_key)).decode("utf-8", "replace")
    for name, value in HITS.findall(text):
        with contextlib.suppress(ValueError):
            found[COUNTERS[name]] += float(value)
    return {name: int(value) for name, value in found.items()}


def gained(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {name: after[name] - before[name] for name in before}


async def target(client: httpx.AsyncClient, public: str, control: str, key: str,
                 control_key: str) -> tuple[Target, int]:
    """The gateway's model and context, once it says it is ready."""
    try:
        state = json.loads(await metrics.read(client, control, "/v1/state", control_key))
    except (ValueError, RecursionError):
        raise RunError("state_invalid") from None
    if not isinstance(state, dict) or state.get("status") != "ready":
        raise RunError("target_not_ready")
    alias, context = state.get("model"), state.get("context_tokens")
    if not isinstance(alias, str) or not alias or len(alias) > 256 or type(context) is not int:
        raise RunError("state_invalid")
    return Target(public, key, alias), context


async def _against(public: str, control: str, key: str, control_key: str, shape: Shape,
                   max_seconds: float) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": 1, "status": "failed", "code": "not_started"}
    try:
        async with (httpx.AsyncClient(timeout=None, follow_redirects=False, trust_env=False) as client,
                    asyncio.timeout(max_seconds)):
            to, context = await target(client, public, control, key, control_key)
            if shape.system_tokens + shape.last_tokens + shape.answer_tokens > context:
                raise RunError("context_limit")
            before = await hits(client, control, control_key)
            report = await measure(client, to, shape)
            if "totals" in report:
                engine = gained(before, await hits(client, control, control_key))
                report["totals"] |= {"engine_preemptions": engine.pop("preempted"), "engine_hit_tokens": engine}
    except (RunError, metrics.DiagnosticError) as error:
        report |= {"status": "failed", "code": str(error)}
    except TimeoutError:
        report |= {"status": "failed", "code": "budget_expired"}
    except httpx.HTTPError:
        report |= {"status": "failed", "code": "transport_error"}
    return report


# The dry run's engine. Everything below stands in for a GPU and measures none.

BLOCK_TOKENS = 256  # the stand-in's cache block: 1024 characters of a prompt, as the fake engine counts tokens
PREFILL_S_PER_TOKEN, STEP_S = 1 / 50000, 0.001  # the stand-in's pace for a token it had to read, and for a chunk
ANSWER = ['{"deed": "', *("the keeper walks the shore " for _ in range(12)), '", "mood": "calm"}']
SMALL, LARGE = 100_000, 2_000_000  # the dry run's two caches, in tokens: too small for its worlds, and large enough


LOAD_S_PER_TOKEN, COMPILE_S = 1 / 500000, 0.05  # a token back from the copy, and a schema's text not seen before


class PrefixCache:
    """A prefix cache of `capacity` tokens for the fake engine: whole blocks, chained from the cache salt, the least
    recently used dropped first. With `copy` above 0 a block it drops goes to a second store of that many tokens, as
    to a copy in RAM, and a prefix found there comes back at LOAD_S_PER_TOKEN. A schema whose text is new costs
    COMPILE_S once. It keeps hashes and counts, never text. `seen` is what it told each generation, `local` and
    `external` the tokens it has served from each store."""

    def __init__(self, capacity: int, copy: int = 0) -> None:
        self.room, self.copy_room = capacity // BLOCK_TOKENS, copy // BLOCK_TOKENS
        self.blocks: OrderedDict[bytes, None] = OrderedDict()
        self.copy: OrderedDict[bytes, None] = OrderedDict()
        self.schemas: set[str] = set()
        self.seen: list[tuple[int, int]] = []
        self.local = self.external = 0

    def __call__(self, body: Any) -> Any:
        from ..fake_engine import Script, _prompt_text  # noqa: PLC0415 - only the dry run and its test use the fake

        if not isinstance(body, dict) or "cache_salt" not in body:
            return Script()  # a count
        text = _prompt_text(body)
        prompt, chain, local, loaded, missed = max(1, math.ceil(len(text) / 4)), [], 0, 0, False
        link = hashlib.sha256(str(body["cache_salt"]).encode()).digest()
        for start in range(0, len(text) - BLOCK_TOKENS * 4 + 1, BLOCK_TOKENS * 4):
            link = hashlib.sha256(link + text[start:start + BLOCK_TOKENS * 4].encode()).digest()
            missed = missed or (link not in self.blocks and link not in self.copy)
            if not missed and (loaded or link not in self.blocks):
                loaded += 1  # from the first block the GPU lacks, the rest of the prefix comes from the copy
            elif not missed:
                local += 1
            chain.append(link)
        for link in chain:
            self.copy.pop(link, None)
            self.blocks[link] = None
            self.blocks.move_to_end(link)
        while len(self.blocks) > self.room:
            dropped, _ = self.blocks.popitem(last=False)
            if self.copy_room:
                self.copy[dropped] = None
        while len(self.copy) > self.copy_room:
            self.copy.popitem(last=False)
        cached = (local + loaded) * BLOCK_TOKENS
        self.local, self.external = self.local + local * BLOCK_TOKENS, self.external + loaded * BLOCK_TOKENS
        self.seen.append((prompt, cached))
        schema = json.dumps(body.get("response_format"))
        compile_s = 0.0 if schema in self.schemas else COMPILE_S
        self.schemas.add(schema)
        return Script(events=[{"role": "assistant"}, *({"content": part} for part in ANSWER),
                              {"finish_reason": "stop"}], usage={"cached_tokens": cached},
                      first_delay_s=((prompt - cached) * PREFILL_S_PER_TOKEN + loaded * BLOCK_TOKENS * LOAD_S_PER_TOKEN
                                     + compile_s), delay_s=STEP_S)

    def metrics(self) -> str:
        """The two of vLLM 0.30.0's counters that tell a hit in the GPU's cache from a load from the copy."""
        return ("vllm:num_preemptions_total 0.0\n"
                f"vllm:prefix_cache_hits_total {float(self.local)}\n"
                f"vllm:external_prefix_cache_hits_total {float(self.external)}\n")


@dataclass(frozen=True)
class Dry:
    public: str
    control: str
    key: str
    control_key: str
    alias: str
    cache: PrefixCache


@contextlib.asynccontextmanager
async def dry_gateway(capacity: int, worlds: int, copy: int = 0) -> AsyncIterator[Dry]:
    """The gateway of this repository in front of the fake engine with a cache of `capacity` tokens, and behind it
    a copy of `copy` tokens."""
    from .. import vast  # noqa: PLC0415 - only the dry run starts a gateway
    from ..config import PROVISIONAL, Listener, from_service_block  # noqa: PLC0415
    from ..fake_engine import Call, FakeEngine, _respond  # noqa: PLC0415
    from ..server import Gateway, Servers, bind  # noqa: PLC0415

    class Engine(FakeEngine):
        async def _metrics(self, call: Call, send: Any) -> None:
            await _respond(send, 200, cache.metrics().encode(), b"text/plain; version=0.0.4; charset=utf-8")

    key, control_key = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    limits = PROVISIONAL["limits"] | {"reader": PROVISIONAL["limits"]["reader"] | {"waiting": max(8, worlds)}}
    block = PROVISIONAL | {"alias": "synthetic-worlds", "context_tokens": 65536, "limits": limits, "keys": {
        key: {"label": "worlds", "classes": ["reader", "internal"], "default": "internal", "scopes": True,
              "control": False},
        control_key: {"label": "control", "classes": [], "default": None, "scopes": False, "control": True}}}
    fake, cache = Engine(block["alias"], block["context_tokens"]), PrefixCache(capacity, copy)
    fake.script = cache
    sock = bind(Listener("127.0.0.1", 0))
    engine = Servers([(fake, sock)], graceful_s=1)
    await engine.start()
    try:
        config = from_service_block(block, engine_url=f"http://127.0.0.1:{sock.getsockname()[1]}")
        async with Gateway(config, stop=vast.unconfigured, graceful_s=1) as gateway:
            while gateway.service.status != "ready":
                await asyncio.sleep(0.01)
            yield Dry(f"http://127.0.0.1:{gateway.ports['public']}", f"http://127.0.0.1:{gateway.ports['control']}",
                      key, control_key, block["alias"], cache)
    finally:
        await engine.stop()


async def dry(shape: Shape, max_seconds: float) -> dict[str, Any]:
    """The same worlds three times, each on a new gateway: a cache too small for them, the same cache with a copy
    behind it, and a cache that holds them all."""
    report: dict[str, Any] = {"schema_version": 1, "synthetic": True, "status": "ok", "code": "ok",
                              "shape": asdict(shape), "runs": []}
    for name, capacity, copy in (("small_cache", SMALL, 0), ("small_cache_with_copy", SMALL, LARGE),
                                 ("large_cache", LARGE, 0)):
        async with dry_gateway(capacity, shape.worlds, copy) as gateway:
            run = await _against(gateway.public, gateway.control, gateway.key, gateway.control_key, shape,
                                 max_seconds)
        report["runs"].append({"cache": name, "cache_tokens": capacity, "copy_tokens": copy, "status": run["status"],
                               "code": run["code"], "totals": run.get("totals")})
        if run["status"] != "ok":
            report |= {"status": "failed", "code": run["code"]}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m simple_serving.interference.worlds",
                                     description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("command", nargs="?", choices=("dry-run", "run"), default="dry-run")
    parser.add_argument("--config", type=Path, help="run: owner-only CLI configuration with client/control keys")
    parser.add_argument("--out", type=Path, help="new JSON report file; required for run, never overwritten")
    parser.add_argument("--public", default="http://127.0.0.1:8080")
    parser.add_argument("--control", default="http://127.0.0.1:8081")
    defaults = Shape()
    parser.add_argument("--worlds", type=int, default=defaults.worlds)
    parser.add_argument("--sessions", type=int, default=defaults.sessions)
    parser.add_argument("--turns", type=int, default=defaults.turns, help="turns of every session")
    parser.add_argument("--system-tokens", type=int, default=defaults.system_tokens)
    parser.add_argument("--first-tokens", type=int, default=defaults.first_tokens)
    parser.add_argument("--last-tokens", type=int, default=defaults.last_tokens)
    parser.add_argument("--growth-tokens", type=int, default=defaults.growth_tokens)
    parser.add_argument("--answer-tokens", type=int, default=defaults.answer_tokens)
    parser.add_argument("--class", dest="cls", choices=("reader", "agent", "internal"), default=defaults.cls)
    parser.add_argument("--max-seconds", type=float, default=900, help="whole run budget, at most 3600 seconds")
    args = parser.parse_args(argv)
    if args.command == "run" and (args.config is None or args.out is None):
        parser.error("run requires --config and --out")
    if args.command == "dry-run" and args.config is not None:
        parser.error("--config goes with run")
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 3600:
        parser.error("--max-seconds must be finite, positive and at most 3600")
    if (not 1 <= args.worlds <= 64 or not 1 <= args.sessions <= 16 or not 1 <= args.turns <= 200
            or not 64 <= args.first_tokens <= args.last_tokens <= 200000 or not 64 <= args.system_tokens <= 50000
            or not 1 <= args.growth_tokens <= 8192 or not 1 <= args.answer_tokens <= 8192):
        parser.error("dimensions outside bounds")
    shape = Shape(args.worlds, args.sessions, args.turns, args.system_tokens, args.first_tokens, args.last_tokens,
                  args.growth_tokens, args.answer_tokens, args.cls)
    output = None
    report: dict[str, Any]
    try:
        # Reserve a new owner-only report before any request could start.
        if args.out is not None:
            output = os.fdopen(os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")
        if args.command == "dry-run":
            report = asyncio.run(dry(shape, args.max_seconds))
        else:
            public, control = _base(args.public), _base(args.control)
            config = read_config(args.config)
            if not all(isinstance(config.get(name), str) and config[name] for name in ("client_key", "control_key")):
                raise RunError("config_keys_missing")
            report = asyncio.run(_against(public, control, config["client_key"], config["control_key"], shape,
                                          args.max_seconds))
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

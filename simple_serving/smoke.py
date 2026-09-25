"""The smoke probes and the count matrix of the first rental (contract section 15): a client of the gateway.

    python -m simple_serving.smoke [--config PATH] [--only NAME[,NAME]] [--before FILE | --after FILE]
                                   [--public-port N] [--control-port N] [--fake [--card-dir DIR]]

It calls the gateway through the forward that `cli up` holds, http://127.0.0.1:8080 to the public listener and 8081 to
the control one, with the client key and the control key of the command's configuration, which `--config` names as it
does for the command. Every request is of class `internal`. What the gateway does not say, the card's own report does
(`card.py --inspect`), which the smoke runs over SSH as `up` reaches the card, with the configuration's `ssh_host`.
The probes run in this order, each within a time bound of its own, and the first that fails ends the smoke:

- lifecycle, with --before or --after alone, around the stop and the resume of section 15's step 1: exactly one pair
  runs, and the trial guard's deadline is there. --before writes the deadline and a hash of the boot into a new file,
  and --after checks that the boot is new and the deadline unchanged.
- state: /v1/models and /v1/state serve contract 2, ready, with the manifest's alias and context, and the control key
  sees the pinned versions: the manifest's, and those of the gateway's lock for its packages.
- completion: one short completion with the least body, and its stream as section 4 has it: the alias in every chunk,
  one finish, one usage chunk after it, and [DONE].
- fields: each optional field of section 4 alone, at a legal value. The answer with `response_format` ends in `stop`
  and follows its schema, and the one with thinking off has no reasoning. Beside them an observation that passes
  nothing: two answers at temperature 0.8 with one seed, compared by hash.
- reasoning: thinking on gives reasoning and an answer, thinking off an answer alone.
- finish: `length` at a small max_tokens, after exactly that many tokens, and `stop`.
- refusal: a schema that the engine cannot compile is 400 invalid_request, before any stream.
- abort: a stream too long to end within the probe, closed after its first event, frees its place, and the gateway's
  log row of it says that it was cancelled. A request that ended by itself shows nothing of the cancel, and fails as
  inconclusive. Then a drain finds no work of ours left, counts included, and an open undoes it.
- schemas: the bot's own JSON schemas in strict mode (bot_schemas.json), each answer checked against its schema.
- counts: the count matrix. For each cell the count, then a generation with a small max_tokens, and the count must
  equal usage.prompt_tokens: plain, system and user, turns, a schema, thinking on and off, and near the context.
- privacy: one completion whose prompt holds a fresh random marker, which no log that the gateway or the engine's
  filter writes on the card may hold (section 15, step 2).

Each probe prints one JSON object on a line of its own: its name, `ok`, the gateway's code where one came, the checks
that failed, and numbers. A probe of parts stops at its first part that fails. Strings in it are only names from this
file and the bot's schema copy, codes and statuses of the contract, finish reasons, versions (the pins of this
checkout, and Python's in digits and dots), and the class of an error inside the smoke: never a key, a prompt, the
marker, the model's text or its reasoning. The exit code is 0 when every probe passed, 1 when one failed, and 2 when
the smoke cannot run.

The dry run points it at the dev launcher instead: --public-port and --control-port name its listeners, --config a
file with its test keys, and --card-dir a directory on this machine that stands for the card's state, /root and
/proc. --fake is for that run alone. The fake engine cannot answer some checks truthfully: an answer that follows a
schema, and a count that differs from usage, since it computes both the same way. --fake leaves those unchecked and
names them in `not_verifiable`, it refuses the forward's ports, and it reads the card's files only from --card-dir,
so it never runs against a card.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any, NoReturn

import httpx

from . import CONTRACT, bot_schemas, card, cli
from .errors import STATUS
from .service import PINNED_PACKAGES

CHAT, COUNT = "/v1/chat/completions", "/v1/chat/completions/input_tokens"
HEADERS = {"X-Simple-Serving-Class": "internal"}  # without a scope header, the class's own cache scope
# Each probe's time bound on the card, generous: the first requests after a start may wait for kernels to compile.
BOUND_S = {"lifecycle": 60, "state": 30, "completion": 300, "fields": 600, "reasoning": 300, "finish": 180,
           "refusal": 120, "abort": 120, "schemas": 900, "counts": 900, "privacy": 120}
UNDO_S = {"abort": 30}  # of a probe's bound, what it keeps for undoing its change: three control calls
CARD_PROBES = ("lifecycle", "abort", "privacy")  # the probes that read the card's report
# What the fake engine cannot answer truthfully, and --fake leaves unchecked. The seed's repeat is no check anyway.
NOT_VERIFIABLE = {"fields": ["answer", "seed_repeats"], "schemas": ["answer"], "counts": ["equal"]}
CALL_S = 10  # a call to the control listener, from connecting to the end of its answer
ANSWER_BYTES = 16384  # the most of an answer that is not a stream
STREAM_BYTES = 2_000_000  # the most of a stream, as the bot reads one
TEXT_CHARS = 100_000  # the most text of a stream, reasoning included, as the bot reads one
FREE_S = 10  # after a client left, for the gateway to free its place
ROWS_S = 10  # after a request ended, for its log row to reach the card's file
INSPECT = cli.HOLD.removesuffix("--hold") + "--inspect"  # the card's report, run as up runs --hold
INSPECT_S = 30  # one report of the card, from starting ssh to the end of its line
REPORT_BYTES = 65536  # the most of a report that the smoke reads

# Synthetic prompts, never printed.
SYSTEM = "You narrate a short synthetic story about a lighthouse keeper."
SHORT = "In one sentence, the keeper lights the lamp."
WORD = "Give one word for the colour of the sea at night."
QUESTION = "A lighthouse has two towers with three lamps in each. How many lamps are there? Answer with the number."
LONG = "Tell a long story of the keeper's night, hour by hour."
COUNTING = "Count from one to ten thousand in words, one number per line, and write nothing else."
ABORT_TOKENS = 8192  # the internal class's most, so that the count cannot end within the abort probe
ECHO = "Repeat this code exactly, and nothing else:"  # before the privacy marker
YES = "Reply with the single word yes."
TURNS = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "The keeper opens the door."},
         {"role": "assistant", "content": "Wind fills the stairwell."}, {"role": "user", "content": "He climbs up."}]
FILLER = "The keeper climbs the stair, trims the wick and lights the lamp.\n"  # the near-context cell's unit
ENDING = "How does the night end? One sentence."
WORD_SCHEMA = {"type": "object", "properties": {"word": {"type": "string", "minLength": 1, "maxLength": 20}},
               "required": ["word"], "additionalProperties": False}
# The refusal: a pattern with an unclosed group, which no dialect of regular expressions accepts. vLLM 0.30.0 checks
# a structured request in its API server before it creates the stream. With its default backend, `auto`, it builds
# the grammar with xgrammar (0.2.8 in the card's lock) and, when that fails, with llguidance (1.7.6); both compile
# `pattern` as a regular expression, so both fail, and vLLM answers 400, which the gateway passes on as 400
# invalid_request (section 4). This rests on vLLM's earlier sources; the probe is what checks it on the pin.
UNCOMPILABLE = {"type": "object", "properties": {"word": {"type": "string", "pattern": "(unclosed"}},
                "required": ["word"], "additionalProperties": False}
FIELDS: dict[str, Any] = {
    "temperature": 0.2, "top_p": 0.95, "top_k": 64, "min_p": 0.05, "repetition_penalty": 1.05, "seed": 1234,
    "response_format": {"type": "json_schema", "json_schema": {"name": "word", "strict": True, "schema": WORD_SCHEMA}},
    "chat_template_kwargs": {"enable_thinking": False},
}
SEED = 20260925
CELL_TOKENS = 8  # a count cell's generation
NEAR = 16  # the near-context cell aims this many tokens below the context...
NEAR_BAND = 64  # ...and takes a count up to this far below it
SCHEMA_TOKENS = 2048  # the most any bot schema's answer may take here; the bot's own caps go up to 16384


class NoAnswer(Exception):
    """A call to the gateway that got no answer."""


class NoReport(Exception):
    """The card's report did not come, or came in another form."""


def user(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def system(text: str) -> dict[str, str]:
    return {"role": "system", "content": text}


def error_code(data: Any) -> str:
    """The code of an error body of section 9, or `error` for anything else, which is never printed as it came."""
    error = data.get("error") if isinstance(data, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and code in STATUS else "error"


def is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class Answer:
    status: int
    data: Any  # the answer's JSON, or None

    @property
    def code(self) -> str | None:
        return None if 200 <= self.status < 300 else error_code(self.data)

    def field(self, name: str) -> Any:
        return self.data.get(name) if isinstance(self.data, dict) else None


@dataclass
class Stream:
    """A generation as the smoke read it. Its text stays in memory for the checks and is never printed."""

    status: int
    code: str | None = None  # of a refusal before the stream, or of an error event in it
    chunks: int = 0
    content: str = ""
    reasoning: str = ""
    finish: str | None = None
    usage: dict[str, Any] | None = None  # a usage object of section 4's shape
    done: bool = False
    broken: list[str] = field(default_factory=list)  # the rules of section 4 that the stream broke

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.code is None and self.done and not self.broken

    def fail(self, rule: str) -> None:
        if rule not in self.broken:
            self.broken.append(rule)

    def numbers(self) -> dict[str, int]:
        numbers = {"chunks": self.chunks, "content_chars": len(self.content), "reasoning_chars": len(self.reasoning)}
        if self.usage is not None:
            numbers.update(prompt_tokens=self.usage["prompt_tokens"],
                           completion_tokens=self.usage["completion_tokens"], **self.usage["simple_serving"])
            if "prompt_tokens_details" in self.usage:
                numbers["cached_tokens"] = self.usage["prompt_tokens_details"]["cached_tokens"]
        return numbers

    def take(self, data: bytes, alias: str) -> None:
        """One event of the stream: a chunk, an error event or [DONE]."""
        if self.done or self.code is not None:
            self.fail("after_end")
            return
        if data == b"[DONE]":
            self.done = True
            return
        try:
            event = json.loads(data)
        except ValueError:
            event = None
        if not isinstance(event, dict):
            self.fail("json")
            return
        if "error" in event:
            self.code = error_code(event)
            return
        self.chunks += 1
        if event.get("object") != "chat.completion.chunk":
            self.fail("object")
        if event.get("model") != alias:
            self.fail("model")
        if self.usage is not None:
            self.fail("after_usage")
        choices = event.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            self.fail("choices")
        elif event.get("usage") is not None:
            if choices or self.finish is None:
                self.fail("usage_chunk")
            if usage_shaped(event["usage"]):
                self.usage = event["usage"]
            else:
                self.fail("usage_fields")
        elif choices:
            self._choice(choices[0])

    def _choice(self, choice: Any) -> None:
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0:
            self.fail("choices")
        elif (not isinstance(delta, dict) or len(delta) > 1
              or not delta.keys() <= {"role", "content", "reasoning_content"}
              or delta.get("role", "assistant") != "assistant"
              or not all(isinstance(delta[name], str) for name in delta.keys() - {"role"})):
            self.fail("delta")
        elif self.finish is not None:
            self.fail("after_finish")  # a second finish included
        else:
            self.content += delta.get("content", "")
            self.reasoning += delta.get("reasoning_content", "")
            if len(self.content) + len(self.reasoning) > TEXT_CHARS:
                self.fail("too_long")
            reason = choice.get("finish_reason")
            if reason in ("stop", "length"):
                self.finish = reason
            elif reason is not None:
                self.fail("finish")

    def end(self) -> None:
        """The checks of a stream that has ended."""
        if self.code is None and not self.done:
            self.fail("done")
        if self.done and (self.finish is None or self.usage is None):
            self.fail("finish" if self.finish is None else "usage")


def usage_shaped(usage: Any) -> bool:
    """Whether a usage object has section 4's fields, each a count, and nothing else."""
    if not isinstance(usage, dict) or not usage.keys() <= {"prompt_tokens", "completion_tokens",
                                                           "prompt_tokens_details", "simple_serving"}:
        return False
    details, measured = usage.get("prompt_tokens_details", {"cached_tokens": 0}), usage.get("simple_serving")
    return (is_count(usage.get("prompt_tokens")) and usage["prompt_tokens"] >= 1
            and is_count(usage.get("completion_tokens"))
            and isinstance(details, dict) and details.keys() == {"cached_tokens"} and is_count(details["cached_tokens"])
            and isinstance(measured, dict) and measured.keys() == {"wait_ms", "first_token_ms", "total_ms"}
            and all(is_count(value) for value in measured.values()))


async def read_stream(response: httpx.Response, alias: str) -> Stream:
    stream = Stream(response.status_code)
    if response.status_code != 200:
        stream.code = error_code(await read_json(response))
        return stream
    if response.headers.get("content-type", "").split(";")[0] != "text/event-stream":
        stream.fail("content_type")
    size, pending, data = 0, b"", []
    try:
        async for piece in response.aiter_bytes():
            size += len(piece)
            if size > STREAM_BYTES:
                stream.fail("too_long")
                return stream
            pending += piece
            *lines, pending = pending.split(b"\n")
            for line in lines:
                if line.startswith(b"data: "):
                    data.append(line[6:])
                elif line:
                    stream.fail("event_form")
                elif data:
                    stream.take(b"\n".join(data), alias)
                    data = []
    except httpx.HTTPError:
        stream.fail("connection")
    if pending or data:
        stream.fail("event_form")
    stream.end()
    return stream


async def read_json(response: httpx.Response) -> Any:
    data = b""
    async for piece in response.aiter_bytes():
        data += piece
        if len(data) > ANSWER_BYTES:
            return None
    try:
        return json.loads(data)
    except ValueError:
        return None


@dataclass(frozen=True)
class CardFiles:
    """The card's own report (`card.py --inspect`): over SSH as `up` reaches the card, or in the dry run on this
    machine, over a directory that stands for the card's state, /root and /proc."""

    host: str | None = None
    directory: Path | None = None

    def command(self) -> tuple[list[str], dict[str, str] | None]:
        if self.directory is not None:
            places = {"DIR": self.directory, "ROOT": self.directory, "PROC": self.directory / "proc"}
            return ([sys.executable, "-m", "simple_serving.card", "--inspect"],
                    os.environ | {f"SIMPLE_SERVING_CARD_{name}": str(path) for name, path in places.items()})
        if self.host is None:
            raise NoReport
        return [*cli.SSH, self.host, INSPECT], None

    async def inspect(self, marker: str | None = None, since: int | None = None) -> dict[str, Any]:
        """The report, as a request with `marker` and `since` asks for it: numbers and flags alone, of the form that
        `card.inspect` gives, or NoReport."""
        argv, env = self.command()
        request = {name: value for name, value in (("marker", marker), ("since", since)) if value is not None}
        process = await asyncio.create_subprocess_exec(*argv, env=env, cwd=card.CODE, stdin=asyncio.subprocess.PIPE,
                                                       stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.DEVNULL)
        assert process.stdin is not None and process.stdout is not None
        out = b""
        try:
            async with asyncio.timeout(INSPECT_S):
                process.stdin.write(json.dumps(request).encode() + b"\n")
                await process.stdin.drain()
                process.stdin.close()
                while len(out) <= REPORT_BYTES and (piece := await process.stdout.read(REPORT_BYTES)):
                    out += piece
                code = await process.wait()
        except (TimeoutError, OSError):
            raise NoReport from None
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        if code != 0 or len(out) > REPORT_BYTES:
            raise NoReport
        return checked_report(out, marker is not None, since is not None)


def checked_report(data: bytes, marker: bool, since: bool) -> dict[str, Any]:
    """A report of the card as `card.inspect` makes it, or NoReport: nothing else of the card is ever read."""
    try:
        report = json.loads(data)
    except ValueError:
        raise NoReport from None
    logs = report.get("logs") if isinstance(report, dict) else None
    entry = {"rows", "found"} if marker else {"rows"}
    cancelled = report.get("cancelled", []) if isinstance(report, dict) else None
    if not (isinstance(report, dict) and report.keys() == {"launchers", "children", "deadline", "now", "logs",
                                                           *(["cancelled"] if since else [])}
            and all(is_count(report[name]) for name in ("launchers", "children", "now"))
            and (report["deadline"] is None or is_count(report["deadline"]))
            and isinstance(logs, dict) and logs.keys() == {*card.LOGS, "other"}
            and all(isinstance(counts, dict) and counts.keys() == entry and all(map(is_count, counts.values()))
                    for counts in logs.values())
            and (cancelled is None or (isinstance(cancelled, list) and all(type(flag) is bool for flag in cancelled)))):
        raise NoReport
    return report


@dataclass(frozen=True)
class Record:
    """The lifecycle's file: --before writes it, and --after reads it into `before`."""

    path: Path
    before: dict[str, Any] | None = None


@dataclass
class Smoke:
    client: httpx.AsyncClient
    public: str
    control: str
    client_key: str = field(repr=False)
    control_key: str = field(repr=False)
    alias: str
    context: int
    fake: bool = False
    card: CardFiles = field(default_factory=CardFiles)
    record: Record | None = None
    undo: Callable[[], Awaitable[bool]] | None = None  # what the running probe changed and run() undoes

    def headers(self, listener: str) -> dict[str, str]:
        if listener == "control":
            return {"Authorization": f"Bearer {self.control_key}"}
        return {"Authorization": f"Bearer {self.client_key}", **HEADERS}

    def body(self, messages: list[dict[str, str]], max_tokens: int, **fields: Any) -> dict[str, Any]:
        return {"model": self.alias, "messages": messages, "max_tokens": max_tokens, "stream": True,
                "stream_options": {"include_usage": True}, **fields}

    async def call(self, listener: str, method: str, path: str, body: Any = None) -> Answer:
        base = self.control if listener == "control" else self.public
        try:
            # A control call within an absolute deadline of its own; a public one within the probe's bound.
            async with (asyncio.timeout(CALL_S if listener == "control" else None),
                        self.client.stream(method, base + path, json=body, headers=self.headers(listener)) as response):
                return Answer(response.status_code, await read_json(response))
        except (httpx.HTTPError, TimeoutError):
            raise NoAnswer from None

    async def view(self) -> dict[str, Any]:
        """/v1/state as the control key sees it; a failed read fails the probe."""
        answer = await self.call("control", "GET", "/v1/state")
        if answer.status != 200 or not isinstance(answer.data, dict):
            raise NoAnswer
        return answer.data

    async def count(self, body: dict[str, Any]) -> Answer:
        return await self.call("public", "POST", COUNT, body)

    async def generate(self, body: dict[str, Any]) -> Stream:
        try:
            async with self.client.stream("POST", self.public + CHAT, json=body,
                                          headers=self.headers("public")) as response:
                return await read_stream(response, self.alias)
        except httpx.HTTPError:
            raise NoAnswer from None


def verdict(stream: Stream, failed: list[str] | None = None, **numbers: Any) -> dict[str, Any]:
    """A probe's or a part's line: `ok`, the code, the rules and checks that failed, and numbers."""
    part: dict[str, Any] = {"ok": stream.ok and not failed, "status": stream.status}
    if stream.code is not None:
        part["code"] = stream.code
    if problems := [*stream.broken, *(failed or [])]:
        part["failed"] = problems
    return part | stream.numbers() | numbers


def answered(stream: Stream, failed: list[str], answer: list[str]) -> dict[str, Any]:
    """A part whose answer should follow a schema. `answer` holds what is wrong with it, empty when nothing is."""
    return verdict(stream, [*failed, "answer"], answer=answer) if answer else verdict(stream, failed)


def combined(parts: dict[str, dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"ok": all(part["ok"] for part in parts.values()), **extra, **parts}


def answer_problems(content: str, schema: dict[str, Any]) -> list[str]:
    """What is wrong with an answer that should follow the schema: `json`, or the keywords it breaks. Empty when it
    follows it. Its numbers are read as decimals, exactly: 1.0000000000000001 is no integer, though a float would
    round it to one."""
    try:
        instance = json.loads(content, parse_float=Decimal, parse_constant=no_constant)
    except (ValueError, RecursionError):
        return ["json"]
    return sorted(bot_schemas.problems(instance, schema))


def no_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not JSON")


def total(counts: Any) -> int | None:
    """The sum of counts by class, as /v1/state has them, or None for another shape."""
    if not isinstance(counts, dict) or not all(is_count(n) for n in counts.values()):
        return None
    return sum(counts.values())


def locked(names: tuple[str, ...]) -> dict[str, str]:
    """The versions that the gateway's lock on the card pins, by package."""
    lock = (card.CODE / "card" / "gateway-requirements.txt").read_text(encoding="utf-8")
    pins = dict(re.findall(r"^([A-Za-z0-9._-]+)==(\S+)", lock, re.MULTILINE))
    return {name: pins[name] for name in names}


async def probe_lifecycle(smoke: Smoke) -> dict[str, Any]:
    if smoke.record is None:
        raise ValueError("the lifecycle runs with --before or --after")
    boot_id = (await smoke.view()).get("boot_id")
    boot = digest(boot_id) if isinstance(boot_id, str) and boot_id else None
    report = await smoke.card.inspect()
    failed = []
    line: dict[str, Any] = {"launchers": report["launchers"], "children": report["children"]}
    if (report["launchers"], report["children"]) != (1, 2):
        failed.append("pair")
    if boot is None:
        failed.append("boot")
    deadline = report["deadline"]
    if deadline is not None:
        line["deadline_left_s"] = deadline - report["now"]
    before = smoke.record.before
    if before is None:
        if deadline is None:
            failed.append("deadline")
        if not failed:
            write_record(smoke.record.path, {"boot": boot, "deadline": deadline})
            line["recorded"] = True
    else:
        line["boot_new"] = boot is not None and boot != before["boot"]
        line["deadline_unchanged"] = deadline == before["deadline"]
        failed += [check for check, name in (("boot", "boot_new"), ("deadline", "deadline_unchanged"))
                   if not line[name] and check not in failed]
    return {"ok": not failed, **({"failed": failed} if failed else {}), **line}


def write_record(path: Path, record: dict[str, Any]) -> None:
    """A new file, open to its owner alone."""
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as file:
        json.dump(record, file)


async def probe_state(smoke: Smoke) -> dict[str, Any]:
    failed = []
    models = await smoke.call("public", "GET", "/v1/models")
    cards = models.field("data")
    entry = next((c for c in cards if isinstance(c, dict) and c.get("id") == smoke.alias), None) \
        if models.status == 200 and isinstance(cards, list) else None
    if entry is None or entry.get("max_model_len") != smoke.context:
        failed.append("models")
    views = [await smoke.call("public", "GET", "/v1/state"), await smoke.call("control", "GET", "/v1/state")]
    expected = {"contract": ("contract", CONTRACT), "ready": ("status", "ready"), "alias": ("model", smoke.alias),
                "context": ("context_tokens", smoke.context)}
    for answer in views:
        view = answer.data if answer.status == 200 and isinstance(answer.data, dict) else {}
        failed += [check for check, (name, value) in expected.items() if view.get(name) != value
                   and check not in failed]
    control = views[1].data if isinstance(views[1].data, dict) else {}
    if control.get("sleep_requested") is not False:
        failed.append("sleep_requested")
    seen: dict[str, Any] = control["versions"] if isinstance(control.get("versions"), dict) else {}
    pinned = card.pinned_versions(card.read_manifest()) | locked(PINNED_PACKAGES)
    # A pinned version is printed as the pin, and one that differs as false; Python's, which the card does not pin,
    # as its numbers.
    versions: dict[str, Any] = {name: pin if seen.get(name) == pin else False for name, pin in pinned.items()}
    python = seen.get("python")
    versions["python"] = python if isinstance(python, str) and re.fullmatch(r"\d+(\.\d+){0,3}", python) else False
    if not all(versions[name] for name in pinned):
        failed.append("versions")
    line: dict[str, Any] = {"ok": not failed}
    if codes := [answer.code for answer in (models, *views) if answer.code is not None]:
        line["code"] = codes[0]
    if failed:
        line["failed"] = failed
    for name in ("active", "waiting"):
        if (n := total(control.get(name))) is not None:
            line[name] = n
    return line | {"versions": versions}


async def probe_completion(smoke: Smoke) -> dict[str, Any]:
    stream = await smoke.generate(smoke.body([user(SHORT)], 32))
    failed = []
    if stream.ok and not stream.content:
        failed.append("content")
    if stream.ok and stream.usage is not None and stream.usage["completion_tokens"] > 32:
        failed.append("completion_tokens")
    return verdict(stream, failed, finish=stream.finish)


async def probe_fields(smoke: Smoke) -> dict[str, Any]:
    parts = {}
    for name, value in FIELDS.items():
        schema = name == "response_format"
        stream = await smoke.generate(smoke.body([user(WORD if schema else SHORT)], 64 if schema else 32,
                                                 **{name: value}))
        failed, answer = [], []
        if stream.ok and schema and stream.finish != "stop":
            failed.append("finish")  # an answer cut short cannot follow its schema
        elif stream.ok and schema and not smoke.fake:
            answer = answer_problems(stream.content, WORD_SCHEMA)
        if stream.ok and name == "chat_template_kwargs" and stream.reasoning:
            failed.append("reasoning")
        parts[name] = answered(stream, failed, answer)
        if not parts[name]["ok"]:
            return combined(parts)
    # An observation, not a check: the engine may batch the two answers differently.
    first, second = [await smoke.generate(smoke.body([user(SHORT)], 32, temperature=0.8, seed=SEED))
                     for _ in range(2)]
    seed: dict[str, Any] = {}
    if first.ok and second.ok:
        seed["seed_repeats"] = digest(first.content) == digest(second.content)
    return combined(parts, **seed)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def probe_reasoning(smoke: Smoke) -> dict[str, Any]:
    parts = {}
    for name, thinking in (("on", True), ("off", False)):
        stream = await smoke.generate(smoke.body([user(QUESTION)], 2048,
                                                 chat_template_kwargs={"enable_thinking": thinking}))
        failed = []
        if stream.ok and bool(stream.reasoning) != thinking:
            failed.append("reasoning")
        if stream.ok and not stream.content:
            failed.append("content")
        parts[name] = verdict(stream, failed, finish=stream.finish)
        if not parts[name]["ok"]:
            break
    return combined(parts)


async def probe_finish(smoke: Smoke) -> dict[str, Any]:
    cut = await smoke.generate(smoke.body([user(LONG)], 8))
    failed = []
    if cut.ok and cut.finish != "length":
        failed.append("finish")
    elif cut.ok and cut.usage is not None and cut.usage["completion_tokens"] != 8:
        failed.append("completion_tokens")
    parts = {"length": verdict(cut, failed, finish=cut.finish)}
    if parts["length"]["ok"]:
        stop = await smoke.generate(smoke.body([user(YES)], 64))
        parts["stop"] = verdict(stop, ["finish"] if stop.ok and stop.finish != "stop" else [], finish=stop.finish)
    return combined(parts)


async def probe_refusal(smoke: Smoke) -> dict[str, Any]:
    stream = await smoke.generate(smoke.body([user(WORD)], 32, response_format={
        "type": "json_schema", "json_schema": {"name": "word", "strict": True, "schema": UNCOMPILABLE}}))
    line: dict[str, Any] = {"ok": stream.status == 400 and stream.code == "invalid_request", "status": stream.status}
    if stream.code is not None:
        line["code"] = stream.code
    if not line["ok"]:
        line["failed"] = ["streamed" if stream.status == 200 else "code"]
    return line


async def probe_abort(smoke: Smoke) -> dict[str, Any]:
    before = await smoke.view()
    if before.get("status") != "ready" or total(before.get("active")) is None:
        return {"ok": False, "failed": ["ready"]}
    since = (await smoke.card.inspect())["logs"]["gateway.jsonl"]["rows"]
    line: dict[str, Any] = {}
    body = smoke.body([user(COUNTING)], ABORT_TOKENS, chat_template_kwargs={"enable_thinking": False})
    # A client of its own, so that leaving the stream closes its connection, which is how a client cancels (section 9).
    try:
        async with (httpx.AsyncClient(timeout=httpx.Timeout(None, connect=CALL_S), trust_env=False) as own,
                    own.stream("POST", smoke.public + CHAT, json=body, headers=smoke.headers("public")) as response):
            if response.status_code != 200:
                return {"ok": False, "status": response.status_code,
                        "code": error_code(await read_json(response)), "failed": ["status"]}
            # The pieces stay referenced until the leave: a loop left with break would drop them, and the event
            # loop's finalizer of the dropped iterator would close the connection at a moment of its own.
            pieces, received = response.aiter_bytes(), b""
            while b"\n\n" not in received and (piece := await anext(pieces, b"")):
                received += piece
            if not received.startswith(b"data: {"):
                return {"ok": False, "failed": ["first_event"]}
            # While it streams, the request holds a place; without that, the rest of the probe would prove nothing.
            line["held"] = (internal_active(await smoke.view()) or 0) - (internal_active(before) or 0)
    except httpx.HTTPError:
        raise NoAnswer from None
    left = time.monotonic()
    while True:
        after = await smoke.view()
        if total(after.get("active")) == total(before["active"]) and total(after.get("waiting")) == 0:
            line["freed_ms"] = round((time.monotonic() - left) * 1000)
            break
        if time.monotonic() - left > FREE_S:
            return {"ok": False, "failed": ["freed"], **line}
        await asyncio.sleep(0.05)
    # The gateway's own row of the request, of which only its cancelled flag is read, tells a cancel from an end: a
    # request that had ended by itself shows nothing of the cancel, and the probe cannot pass on it.
    cancelled = (await logged(smoke, since))["cancelled"]
    if cancelled is None or len(cancelled) != 1:
        return {"ok": False, "failed": ["unlogged"], **line}
    line["cancelled"] = cancelled[0]
    if not cancelled[0]:
        return {"ok": False, "failed": ["inconclusive"], **line}
    if line["held"] != 1:
        return {"ok": False, "failed": ["held"], **line}
    smoke.undo = partial(reopen, smoke)  # before the drain, which may take effect though its answer is lost
    drain = await smoke.call("control", "POST", "/v1/control/drain", {"boot_id": after.get("boot_id")})
    # With no accepted work left, counts included, a drain is complete before it answers (section 8).
    if drain.status != 202 or drain.field("status") != "drained":
        return {"ok": False, "failed": ["drained"], **line}
    return {"ok": True, **line}


def internal_active(view: dict[str, Any]) -> int | None:
    """The generations of class internal that the engine has, as /v1/state counts them."""
    active = view.get("active")
    n = active.get("internal", 0) if isinstance(active, dict) else None
    return n if is_count(n) else None


async def reopen(smoke: Smoke) -> bool:
    """Undo the abort probe's drain, whatever became of its answer, and read the state back: True once the service
    is ready. A service that is falling asleep stays as it is."""
    view = await smoke.view()
    if view.get("status") in ("draining", "drained") and view.get("sleep_requested") is False:
        await smoke.call("control", "POST", "/v1/control/open",
                         {"boot_id": view.get("boot_id"), "drain_generation": view.get("drain_generation")})
    return (await smoke.view()).get("status") == "ready"


async def logged(smoke: Smoke, since: int, marker: str | None = None) -> dict[str, Any]:
    """The card's report once gateway.jsonl holds a chat request after its first `since` rows, or ROWS_S later."""
    loop = asyncio.get_running_loop()
    until = loop.time() + ROWS_S
    while True:
        report = await smoke.card.inspect(marker=marker, since=since)
        if report["cancelled"] != [] or loop.time() >= until:
            return report
        await asyncio.sleep(0.2)


async def probe_schemas(smoke: Smoke) -> dict[str, Any]:
    parts = {}
    entries = bot_schemas.load()
    for entry in entries:
        # The bot's body around the schema (local/serving.ts at the copy's commit).
        stream = await smoke.generate(smoke.body(
            [system(entry.system), user(entry.user)], min(entry.bot_max_tokens, SCHEMA_TOKENS),
            temperature=0.2 if entry.memory else 0.8, top_p=0.95, top_k=64, min_p=0, repetition_penalty=1,
            chat_template_kwargs={"enable_thinking": False},
            response_format={"type": "json_schema", "json_schema": {"name": "reply", "strict": True,
                                                                    "schema": entry.schema}}))
        failed, answer = [], []
        if stream.ok and stream.finish != "stop":
            failed.append("finish")  # an answer cut short cannot follow its schema
        elif stream.ok and not smoke.fake:
            answer = answer_problems(stream.content, entry.schema)
        parts[entry.name] = answered(stream, failed, answer)
        if not parts[entry.name]["ok"]:
            break
    return combined(parts, passed=sum(part["ok"] for part in parts.values()), of=len(entries))


async def probe_counts(smoke: Smoke) -> dict[str, Any]:
    word = {"type": "json_schema", "json_schema": {"name": "word", "strict": True, "schema": WORD_SCHEMA}}
    pair = [system(SYSTEM), user(SHORT)]
    cells = {
        "plain": smoke.body([user(SHORT)], CELL_TOKENS),
        "system_user": smoke.body(pair, CELL_TOKENS),
        "turns": smoke.body(TURNS, CELL_TOKENS),
        "schema": smoke.body([system(SYSTEM), user(WORD)], CELL_TOKENS, response_format=word),
        "thinking_on": smoke.body(pair, CELL_TOKENS, chat_template_kwargs={"enable_thinking": True}),
        "thinking_off": smoke.body(pair, CELL_TOKENS, chat_template_kwargs={"enable_thinking": False}),
    }
    parts = {}
    for name, body in cells.items():
        parts[name] = await count_cell(smoke, body)
        if not parts[name]["ok"]:
            return combined(parts)
    parts["near_context"] = await near_context(smoke)
    return combined(parts)


async def count_cell(smoke: Smoke, body: dict[str, Any], **numbers: Any) -> dict[str, Any]:
    """The count of a body, then its generation, whose usage.prompt_tokens must be that count."""
    counted = await smoke.count(body)
    input_tokens = counted.field("input_tokens")
    if counted.status != 200 or not is_count(input_tokens):
        return {"ok": False, "status": counted.status, **({"code": counted.code} if counted.code else {}),
                "failed": ["count"]}
    stream = await smoke.generate(body)
    failed = []
    if stream.ok and stream.usage is not None and stream.usage["prompt_tokens"] != input_tokens and not smoke.fake:
        failed.append("equal")
    return verdict(stream, failed, input_tokens=input_tokens, **numbers)


async def near_context(smoke: Smoke) -> dict[str, Any]:
    """A prompt whose count is just below the context, found by counting, and a generation that fills the context
    to its last token: max_tokens is the context less the count."""

    def body(copies: int, max_tokens: int = 1) -> dict[str, Any]:
        return smoke.body([system(SYSTEM), user(FILLER * copies + ENDING)], max_tokens)

    counts = []

    async def count(copies: int) -> int | None:
        n = (await smoke.count(body(copies))).field("input_tokens")
        counts.append(n)
        return n if is_count(n) else None

    one, many = await count(1), await count(257)
    if one is None or many is None or many <= one:
        return {"ok": False, "failed": ["count"], "counts": len(counts)}
    per_copy = (many - one) / 256
    copies = 1 + max(0, round((smoke.context - NEAR - one) / per_copy))
    for _ in range(8):
        n = await count(copies)
        if n is None:
            return {"ok": False, "failed": ["count"], "counts": len(counts)}
        if smoke.context - NEAR_BAND <= n < smoke.context:
            return await count_cell(smoke, body(copies, smoke.context - n), max_tokens=smoke.context - n,
                                    counts=len(counts))
        copies = max(1, copies + (round((smoke.context - NEAR - n) / per_copy) or (1 if n < smoke.context else -1)))
    return {"ok": False, "failed": ["near"], "counts": len(counts)}


def new_marker() -> str:
    return secrets.token_hex(16)


async def probe_privacy(smoke: Smoke) -> dict[str, Any]:
    since = (await smoke.card.inspect())["logs"]["gateway.jsonl"]["rows"]
    marker = new_marker()
    stream = await smoke.generate(smoke.body([user(f"{ECHO} {marker}")], 64,
                                             chat_template_kwargs={"enable_thinking": False}))
    report = await logged(smoke, since, marker)
    found = {name: counts["found"] for name, counts in report["logs"].items()}
    line: dict[str, Any] = {"status": stream.status, **({"code": stream.code} if stream.code is not None else {})}
    failed = [*stream.broken, *(["code"] if stream.code is not None else [])]
    if not report["cancelled"]:
        failed.append("unlogged")  # without the request's own row, a log without the marker proves nothing
    if any(found.values()):
        failed.append("marker")
    return {"ok": not failed, **line, **({"failed": failed} if failed else {}), "found": found,
            "logged": len(report["cancelled"] or []), "echoed": stream.content.count(marker)}


PROBES: dict[str, Callable[[Smoke], Awaitable[dict[str, Any]]]] = {
    "lifecycle": probe_lifecycle, "state": probe_state, "completion": probe_completion, "fields": probe_fields,
    "reasoning": probe_reasoning, "finish": probe_finish, "refusal": probe_refusal, "abort": probe_abort,
    "schemas": probe_schemas, "counts": probe_counts, "privacy": probe_privacy,
}
CHOSEN = tuple(name for name in PROBES if name != "lifecycle")  # what --only may name; --before and --after add it


async def run(smoke: Smoke, names: list[str]) -> int:
    for index, name in enumerate(names):
        line = await one(smoke, name)
        if smoke.fake and name in NOT_VERIFIABLE:
            line["not_verifiable"] = NOT_VERIFIABLE[name]
        print(json.dumps({"probe": name, **line}), flush=True)
        if not line["ok"]:
            print(f"simple-serving smoke: {index} of {len(names)} probes passed; {name} failed, and the smoke stopped "
                  "there", file=sys.stderr)
            return 1
    print(f"simple-serving smoke: {len(names)} of {len(names)} probes passed", file=sys.stderr)
    return 0


async def one(smoke: Smoke, name: str) -> dict[str, Any]:
    """A probe within its bound, then the undo of what it changed. The bound keeps UNDO_S of its time for the undo,
    which runs once the probe's own time has closed, so that the bound cannot cut it short, and within a deadline of
    its own. An undo that is not confirmed fails the probe."""
    line: dict[str, Any]
    try:
        async with asyncio.timeout(BOUND_S[name] - UNDO_S.get(name, 0)):
            line = await PROBES[name](smoke)
    except TimeoutError:
        line = {"ok": False, "failed": ["time_bound"], "bound_s": BOUND_S[name]}
    except NoAnswer:
        line = {"ok": False, "failed": ["no_answer"]}
    except NoReport:
        line = {"ok": False, "failed": ["no_report"]}
    except Exception as error:  # noqa: BLE001 - it fails its probe by its class: a traceback may quote text
        line = {"ok": False, "failed": ["smoke_error"], "error": type(error).__name__}
    undo, smoke.undo = smoke.undo, None
    if undo is not None and not await undone(undo, UNDO_S.get(name, CALL_S)):
        line = {**line, "ok": False, "failed": [*line.get("failed", []), "left_drained"]}
    return line


async def undone(undo: Callable[[], Awaitable[bool]], within_s: float) -> bool:
    try:
        async with asyncio.timeout(within_s):
            return await undo()
    except Exception:  # noqa: BLE001 - an undo that fails, however, is not confirmed
        return False


def probe_names(text: str) -> list[str]:
    names = text.split(",")
    if unknown := [name for name in names if name not in CHOSEN]:
        raise argparse.ArgumentTypeError(f"no probe {unknown[0]!r}; the probes are {', '.join(CHOSEN)}")
    return [name for name in CHOSEN if name in names]


def port(text: str) -> int:
    if not (text.isascii() and text.isdigit() and 1 <= int(text) <= 65535):
        raise argparse.ArgumentTypeError("must be a port, 1 to 65535")
    return int(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m simple_serving.smoke", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=cli.CONFIG,
                        help="the command's configuration, with client_key and control_key, and ssh_host for the "
                             "card's report (default: %(default)s)")
    parser.add_argument("--only", type=probe_names, default=list(CHOSEN), metavar="NAME[,NAME]",
                        help=f"run these probes alone: {', '.join(CHOSEN)}")
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--before", type=Path, metavar="FILE",
                           help="before the stop of section 15's step 1: first check the pair and the guard's "
                                "deadline, and write the deadline and a hash of the boot into FILE, a new file")
    lifecycle.add_argument("--after", type=Path, metavar="FILE",
                           help="after the resume: first check the pair, that the boot is new and that the guard's "
                                "deadline is the one in FILE")
    parser.add_argument("--public-port", type=port, default=cli.LOCAL["public"],
                        help="the public listener on 127.0.0.1 (default: %(default)s, the forward of up)")
    parser.add_argument("--control-port", type=port, default=cli.LOCAL["control"],
                        help="the control listener on 127.0.0.1 (default: %(default)s, the forward of up)")
    parser.add_argument("--fake", action="store_true",
                        help="the dry run in front of the dev launcher's fake engine: leave unchecked what it cannot "
                             "answer truthfully")
    parser.add_argument("--card-dir", type=Path, metavar="DIR",
                        help="with --fake: a directory on this machine that stands for the card's state, /root and "
                             "/proc, whose report replaces the card's")
    args = parser.parse_args(argv)
    names = (["lifecycle"] if args.before or args.after else []) + args.only
    reading = [name for name in names if name in CARD_PROBES]
    try:
        if args.fake and {args.public_port, args.control_port} & set(cli.LOCAL.values()):
            raise cli.Refusal("--fake is for the dev launcher, never for the forward of up")
        if args.card_dir is not None and not args.fake:
            raise cli.Refusal("--card-dir stands for the card in the dry run, with --fake alone")
        if args.fake and reading and args.card_dir is None:
            raise cli.Refusal(f"--fake reads the card's files from --card-dir alone, and {', '.join(reading)} "
                              "read them")
        config = cli.read_config(args.config)
        if missing := [name for name in cli.KEYS if not isinstance(config.get(name), str) or not config[name]]:
            raise cli.Refusal(f"{args.config} has no {', '.join(missing)}")
        host = config.get("ssh_host")
        if reading and args.card_dir is None and not (isinstance(host, str) and cli.SSH_HOST.fullmatch(host)):
            raise cli.Refusal(f"{args.config} has no ssh_host of the right form, and {', '.join(reading)} read the "
                              "card's files over SSH")
        record = lifecycle_record(args.before, args.after)
        manifest = card.read_manifest()
    except cli.Refusal as error:
        print(f"simple-serving smoke: {error}", file=sys.stderr)
        return 2
    files = CardFiles(host if reading and args.card_dir is None else None, args.card_dir)
    return asyncio.run(smoke_run(args, config, manifest, files, record, names))


def lifecycle_record(before: Path | None, after: Path | None) -> Record | None:
    """The lifecycle's file: one that --before may create, or one that --after reads."""
    if before is not None:
        if os.path.lexists(before):
            raise cli.Refusal(f"{before} exists, and --before writes a new file")
        return Record(before)
    if after is None:
        return None
    try:
        written = json.loads(after.read_text())
    except (OSError, ValueError):
        raise cli.Refusal(f"cannot read {after}, which --before writes") from None
    if not (isinstance(written, dict) and written.keys() == {"boot", "deadline"} and isinstance(written["boot"], str)
            and re.fullmatch(r"[0-9a-f]{64}", written["boot"]) and is_count(written["deadline"])):
        raise cli.Refusal(f"{after} is not a file of --before")
    return Record(after, written)


async def smoke_run(args: argparse.Namespace, config: dict[str, Any], manifest: dict[str, str], files: CardFiles,
                    record: Record | None, names: list[str]) -> int:
    # trust_env=False: the keys go to loopback only, never through a proxy of the environment.
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=CALL_S), trust_env=False) as client:
        smoke = Smoke(client, f"http://127.0.0.1:{args.public_port}", f"http://127.0.0.1:{args.control_port}",
                      config["client_key"], config["control_key"], manifest["MODEL_ALIAS"],
                      int(manifest["CONTEXT_TOKENS"]), args.fake, files, record)
        return await run(smoke, names)


if __name__ == "__main__":
    sys.exit(main())

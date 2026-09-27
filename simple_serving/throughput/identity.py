"""The identity set (docs/speed.md, "Quality"): 16 of the workloads' requests at temperature 0, one at a time, on both
sides of a change, and for each whether the tokens are the same and where they first differ.

The requests go straight to the engine, as the gateway would send them (`engine.generation_payload`) with vLLM's
`return_token_ids`, since the gateway passes no token ids on. Each has a fresh cache salt, so that no request reads
another's cache and both sides compute every prompt in full. The tokens stay in memory. Within one start, the
reference's two runs are compared token by token: its own floor. Across starts, a side keeps only digests of its
tokens, BLOCK at a time, each an HMAC under a key of the harness's own chained to the one before, so that the digests
hold no output and cannot be turned back into one; a later side then differs from the first block whose digest
differs, and the position it reports is that block's first token. The key and the digests stay on the card, 0600.
What is printed is a count, a position or a fixed word.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

from .. import card
from ..engine import EngineError, EngineStream, generation_payload
from ..errors import ServiceError
from ..validation import check_chat
from . import workloads

KINDS = ("turns", "compactions", "scenes", "unshared")  # four requests of each
CONTEXT = 4000  # tokens of the turns and of the prompts that share nothing, at CHARS_PER_TOKEN
CHARS_PER_TOKEN = 4.0  # fixed, not measured, so that every side builds the same 16 requests
MAX_TOKENS = 256
BLOCK = 16  # tokens a digest covers
VISIBLE_BELOW = 8  # fewer than half of the 16 the same: the change counts as visible
KEY_BYTES = 32
REFERENCE, KEY = "reference.json", "key"  # in the identity directory

Tokens = list[int] | None  # a request's generated ids, or None when it did not end well


class IdentityError(ValueError):
    """The set cannot be compared. The message is a fixed sentence."""


def payloads(stories: list[workloads.Story], alias: str) -> list[dict[str, Any]]:
    """The 16 requests as the engine gets them, at temperature 0 and at most MAX_TOKENS, without their cache salt."""
    shapes = {"turns": workloads.Shape(at=4, context=CONTEXT, turns=1, chars_per_token=CHARS_PER_TOKEN),
              "unshared": workloads.Shape(at=2, context=CONTEXT, chars_per_token=CHARS_PER_TOKEN)}
    bodies = [request.body for kind in KINDS
              for lane in workloads.build(kind, stories, alias, shapes.get(kind, workloads.Shape(at=2)))
              for request in lane[:1]]
    found = []
    for body in bodies:
        request = check_chat({**body, "temperature": 0, "max_tokens": min(MAX_TOKENS, body["max_tokens"])}, alias)
        payload = generation_payload(alias, request, priority=0, cache_salt="")
        del payload["cache_salt"]
        found.append({**payload, "return_token_ids": True})
    return found


async def tokens(client: httpx.AsyncClient, engine_url: str, payload: dict[str, Any]) -> Tokens:
    """The generated token ids of one request, or None if it did not end well. The prompt's ids, which vLLM sends in
    its first chunk, are not read."""
    ids: list[int] = []
    try:
        async with client.stream("POST", f"{engine_url}/v1/chat/completions", json=payload) as response:
            if response.status_code != 200:
                return None
            stream = EngineStream(response)
            while (event := await stream.next_event()) is not None:
                if "error" in event:
                    return None
                for choice in event.get("choices") or ():
                    delta = choice.get("token_ids") if isinstance(choice, dict) else None
                    if isinstance(delta, list) and all(type(token) is int for token in delta):
                        ids += delta
    except (httpx.HTTPError, EngineError, ServiceError):
        return None
    return ids


async def run(client: httpx.AsyncClient, engine_url: str, bodies: Sequence[dict[str, Any]],
              at: int = 1) -> list[Tokens]:
    """The set, `at` requests at once (one at a time unless item 3 asks for its batch), each with a salt of its own."""
    places = asyncio.Semaphore(max(1, at))

    async def one(body: dict[str, Any]) -> Tokens:
        async with places:
            return await tokens(client, engine_url, {**body, "cache_salt": secrets.token_urlsafe(32)})

    return list(await asyncio.gather(*(one(body) for body in bodies)))


def first_difference(a: Sequence[Any], b: Sequence[Any]) -> int | None:
    """The first position where two runs differ, a longer run included; None when they are the same."""
    for position, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return position
    return None if len(a) == len(b) else min(len(a), len(b))


def digests(ids: Sequence[int], key: bytes) -> list[str]:
    """One digest for every BLOCK tokens, the last block possibly shorter, each chained to the one before."""
    chain, found = b"", []
    for start in range(0, len(ids), BLOCK):
        block = b"".join(token.to_bytes(4, "big") for token in ids[start:start + BLOCK])
        chain = hmac.new(key, chain + block, hashlib.sha256).digest()
        found.append(chain.hex())
    return found


def fingerprint(bodies: Sequence[dict[str, Any]], key: bytes) -> str:
    """An HMAC of the 16 requests, so that a later side knows it sent the reference's."""
    return hmac.new(key, json.dumps(list(bodies), sort_keys=True).encode(), hashlib.sha256).hexdigest()


def compared(pairs: Sequence[tuple[Any, Any]], position: Callable[[Any, Any], int | None]) -> dict[str, Any]:
    """The set's verdict from each request's two sides: how many are the same, where each first differs, and whether
    the change counts as visible. A request that failed on either side is not the same, and its position is None."""
    firsts: list[int | None] = []
    same = failed = 0
    for a, b in pairs:
        if a is None or b is None:
            failed += 1
            firsts.append(None)
            continue
        first = position(a, b)
        firsts.append(first)
        same += first is None
    return {"same": same, "of": len(pairs), "failed": failed, "first_difference": firsts,
            "visible": same < VISIBLE_BELOW}


def floor(first: Sequence[Tokens], second: Sequence[Tokens]) -> dict[str, Any]:
    """The reference against itself, token by token."""
    return compared(list(zip(first, second, strict=True)), first_difference)


def against(reference: dict[str, Any], side: Sequence[Tokens], key: bytes) -> dict[str, Any]:
    """A side against the reference's kept digests. A first difference is the first token of the block that
    differs."""
    def position(kept: list[str], ids: list[int]) -> int | None:
        block = first_difference(kept, digests(ids, key))
        return None if block is None else block * BLOCK

    return compared(list(zip(reference["digests"], side, strict=True)), position)


def kept(side: Sequence[Tokens], key: bytes, bodies: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What the reference leaves on the card for later sides: digests, lengths and the set's fingerprint, never a
    token."""
    return {"block": BLOCK, "fingerprint": fingerprint(bodies, key),
            "digests": [None if ids is None else digests(ids, key) for ids in side],
            "tokens": [None if ids is None else len(ids) for ids in side]}


def write_private(path: Path, text: str) -> None:
    descriptor = card.private(path, os.O_WRONLY | os.O_TRUNC)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(text)


def key_at(directory: Path) -> bytes:
    """The harness's digest key on the card, made on first use, 0600."""
    path = directory / KEY
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        write_private(path, secrets.token_hex(KEY_BYTES) + "\n")
    return bytes.fromhex(path.read_text(encoding="utf-8").strip())


def load(directory: Path, bodies: Sequence[dict[str, Any]], key: bytes) -> dict[str, Any]:
    try:
        data = json.loads((directory / REFERENCE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise IdentityError("no reference's identity set on this card: run the reference first") from None
    if not isinstance(data, dict) or data.get("block") != BLOCK or not isinstance(data.get("digests"), list) \
            or len(data["digests"]) != len(bodies):
        raise IdentityError("the reference's identity file is not one this harness wrote")
    if data.get("fingerprint") != fingerprint(bodies, key):
        raise IdentityError("the identity set differs from the reference's: another pack or another harness")
    return data


async def measure(client: httpx.AsyncClient, engine_url: str, bodies: Sequence[dict[str, Any]], side: str,
                  directory: Path, at: int = 1) -> dict[str, Any]:
    """The set on one side. The reference runs it twice for its floor and keeps its first run's digests; any other
    side is compared with those."""
    key = key_at(directory)
    if side == "reference":
        first = await run(client, engine_url, bodies, at)
        second = await run(client, engine_url, bodies, at)
        write_private(directory / REFERENCE, json.dumps(kept(first, key, bodies)) + "\n")
        return {"side": side, "at": at, "block": BLOCK, **floor(first, second)}
    reference = load(directory, bodies, key)
    changed = await run(client, engine_url, bodies, at)
    return {"side": side, "at": at, "block": BLOCK, **against(reference, changed, key)}

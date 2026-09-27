"""One run's result and its short report (docs/speed.md, item 1): counts, times and fixed words only.

A run of the harness writes one JSON file, 0600, in its run directory: the change and how its start went, the proof
that it took effect, the engine's numbers, the identity set's verdict, each pass with its reader-overlap flag and each
cell of the sweep. Before it is written, every string in it is checked to be a code, a number's text, a pin or a time:
at most WORD_LENGTH characters of letters, digits and `_.:=,+/-`, without a space; and no list may hold more than
LIST_NUMBERS numbers. A prompt or an output is not such a word and a text's token ids would be such a list, so a
result that holds either is not written. The report is a few lines made of the result's numbers and words.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .. import card

WORD_LENGTH = 96
LIST_NUMBERS = 64  # a result's longest lists: the drafter's positions, the identity set's 16
WORD = re.compile(rf"[A-Za-z0-9_.:=,+/-]{{0,{WORD_LENGTH}}}")
RESULT = "result.json"
VERSION = 1


class ResultError(ValueError):
    """A result that holds more than counts, times and fixed words. The message names where, never what."""


def check(value: Any, where: str = "result") -> None:
    """Raise ResultError if `value` holds anything but numbers, booleans, None and fixed words."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not WORD.fullmatch(key):
                raise ResultError(f"a key in {where} is not a fixed word")
            check(item, f"{where}.{key}")
    elif isinstance(value, list | tuple):
        if sum(1 for item in value if isinstance(item, int | float)) > LIST_NUMBERS:
            raise ResultError(f"{where} is a list of numbers as long as a text's tokens")
        for index, item in enumerate(value):
            check(item, f"{where}[{index}]")
    elif isinstance(value, str):
        if not WORD.fullmatch(value):
            raise ResultError(f"{where} is not a fixed word")
    elif value is not None and not isinstance(value, bool | int | float):
        raise ResultError(f"{where} is not a number or a word")


def write(directory: Path, result: dict[str, Any]) -> Path:
    """The run's result file, 0600, after the check."""
    check(result)
    path = directory / RESULT
    descriptor = card.private(path, os.O_WRONLY | os.O_TRUNC)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=1)
        file.write("\n")
    return path


def read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("harness") != VERSION:
        raise ResultError("not a result of this harness")
    check(data)
    return data


def _number(value: Any, unit: str = "") -> str:
    return "-" if value is None else f"{value}{unit}"


def _first(first: dict[str, int] | None) -> str:
    return "-" if not first else f"first token {first['median']}/{first['p90']}/{first['max']} ms"


def _pass(entry: dict[str, Any]) -> str:
    flag = f", {entry['reader_requests']} reader requests beside it" if entry.get("reader_overlap") else ""
    failed = sum(entry["failed"].values())
    return (f"  {entry['workload']} at {entry['at']}: {entry['done']}/{entry['requests']} done"
            f"{f', {failed} failed' if failed else ''}; {_number(entry['output_tokens_per_s'])} out/s, "
            f"{_number(entry['read_tokens_per_s'])} read/s, {_number(entry['total_tokens_per_s'])} total/s; "
            f"{_first(entry['first_token_ms'])}; cached {entry['cached_tokens']} of {entry['input_tokens']}"
            f"{flag}")


def _cell(entry: dict[str, Any]) -> str:
    if entry.get("skipped"):
        return f"  {entry['cell']}: skipped, the cache did not hold the one before"
    bench = entry.get("bench") or {}
    held = "held" if entry["held"] else "not held"
    rates = entry.get("acceptance") or bench.get("spec_decode_per_position_acceptance_rates") or []
    return (f"  {entry['cell']}: {held}, exit {entry['exit_code']}; {_number(bench.get('output_throughput'))} out/s, "
            f"{_number(bench.get('total_token_throughput'))} total/s; TPOT median "
            f"{_number(bench.get('median_tpot_ms'), ' ms')}, TTFT median {_number(bench.get('median_ttft_ms'), ' ms')}"
            f"{'; acceptance ' + ','.join(str(rate) for rate in rates) if rates else ''}")


def _identity(verdict: dict[str, Any]) -> str:
    firsts = [position for position in verdict["first_difference"] if position is not None]
    where = f"; first differences at {','.join(str(position) for position in firsts)}" if firsts else ""
    against = "its own second run" if verdict["side"] == "reference" else "the reference"
    return (f"  identity ({verdict['side']}, {verdict['at']} at once): {verdict['same']} of {verdict['of']} the same "
            f"as {against}, {verdict['failed']} failed{where}{'; visible' if verdict['visible'] else ''}")


def lines(result: dict[str, Any]) -> Iterable[str]:
    proof = result.get("proof") or {}
    yield (f"change {result['change']}: {result['outcome']} after {_number(result.get('load_s'), ' s')}"
           f"{', timing only' if result.get('timing_only') else ''}")
    if proof:
        yield (f"  proof: {'ok' if proof['ok'] else 'not shown'}, {len(proof['codes'])} codes"
               f"{'; missing ' + ','.join(proof['missing']) if proof['missing'] else ''}"
               f"{'; unexpected ' + ','.join(proof['unexpected']) if proof['unexpected'] else ''}")
    if result.get("identity"):
        yield _identity(result["identity"])
    for entry in result.get("passes", ()):
        yield _pass(entry)
    for entry in result.get("sweep", ()):
        yield _cell(entry)
    if result.get("passes") or result.get("sweep"):
        yield f"  flagged for a reader's request: {result.get('flagged', 0)}"


def text(result: dict[str, Any]) -> str:
    return "\n".join(lines(result))

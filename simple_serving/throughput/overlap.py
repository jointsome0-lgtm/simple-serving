"""Whether a pass overlapped a reader's request (docs/speed.md, the runbook's "Before any card", step 2).

The harness's gateway serves the bot's client key beside its own, unless the run is timing only, so a reader may send
a request while a pass runs and share the card with it. Such a pass is flagged, and repeated if time allows. The
gateway writes one row per request when the request ends, with its class and, for a generation, its time from
acceptance to the end (`total_ms`); the run's log is read once the pair has ended, so that a request still running at
a pass's end has its row too. Only a row's time, class and duration are read.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import card

MARGIN_S = 1.0  # a reader's request this close to a pass counts as overlapping it
READER = "reader"


def moment(text: str) -> float | None:
    """Seconds since the epoch of a row's time, as log.py writes it (UTC, milliseconds)."""
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


def reader_requests(logs: Path) -> list[tuple[float, float]]:
    """From and to of each reader's request in the gateway's log of a run, the file moved aside first."""
    found = []
    for name in ("gateway.jsonl.1", "gateway.jsonl"):
        try:
            lines = (logs / name).read_bytes().splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            row = card.gateway_row(line)
            if row is None or row.get("event") != "request" or row.get("class") != READER:
                continue
            end = moment(row["time"]) if isinstance(row.get("time"), str) else None
            if end is None:
                continue
            total = row.get("total_ms")
            found.append((end - (total / 1000 if isinstance(total, int) and total > 0 else 0.0), end))
    return found


def flag(windows: Iterable[MutableMapping[str, Any]], requests: list[tuple[float, float]]) -> int:
    """Mark each pass, one with a `window` of two epoch seconds, by how many reader requests overlapped it, and return
    how many passes were flagged."""
    flagged = 0
    for entry in windows:
        start, end = entry["window"]
        overlapping = sum(1 for begun, ended in requests if begun <= end + MARGIN_S and ended >= start - MARGIN_S)
        entry["reader_requests"] = overlapping
        entry["reader_overlap"] = overlapping > 0
        flagged += overlapping > 0
    return flagged

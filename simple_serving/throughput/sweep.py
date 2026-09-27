"""Step 6's timing sweep (docs/speed.md, the runbook's "The 5090", and item 3): `vllm bench serve` from the card's
vLLM venv against the engine's port, at 1 to 8 requests at once.

Each cell is vLLM's random dataset: a prefix of the context less NEW tokens that every request of the cell shares, NEW
tokens of its own and OUT tokens out with `--ignore-eos`, at temperature 0, at 4000, 16000 and 40000 tokens; and
prompts that share nothing at 16000. The start it runs on is the A/B launcher's `synthetic=` change, rates by position
from step 5's acceptance pass, so the engine accepts drafts at those rates whatever the random prompts are: timing only,
never serving. `mtp=0` gives the count without the drafter.

A context's cells run from 1 request at once up, and stop at the first cell whose requests the cache did not hold: the
engine never ran them all at once, or one waited for the cache. The harness reads /metrics around each cell as around
a pass, and keeps of the bench's own result only the numbers named in KEPT; without `--save-detailed` vLLM leaves the
generated texts out of that file anyway (vllm/benchmarks/serve.py:2395-2407). The bench's output goes nowhere.

Every argument list is checked against FLAGS, the options of vLLM 0.30.0's `vllm bench serve` that the sweep uses
(vllm/benchmarks/serve.py:1603-2014, vllm/benchmarks/datasets/datasets.py:1600-1665,1932-1960), before it runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .. import card
from ..card import Card
from . import metrics

CONTEXTS = (4000, 16000, 40000)  # a shared prefix of the context less NEW
UNSHARED = (16000,)
AT = (1, 2, 3, 4, 6, 8)
NEW, OUT = 256, 512
ROUNDS = 3  # requests a cell sends for each one at once
WARMUPS = 2
SEED = 7
PERCENTILES = "50,90,99"
# vLLM 0.30.0's `vllm bench serve` options the sweep uses, each with its value's type; None is a switch.
FLAGS: dict[str, type | None] = {
    "--backend": str, "--host": str, "--port": int, "--endpoint": str, "--model": str, "--tokenizer": str,
    "--dataset-name": str, "--random-prefix-len": int, "--random-input-len": int, "--random-output-len": int,
    "--num-prompts": int, "--max-concurrency": int, "--ignore-eos": None, "--temperature": float,
    "--num-warmups": int, "--seed": int, "--percentile-metrics": str, "--metric-percentiles": str,
    "--save-result": None, "--result-dir": str, "--result-filename": str, "--disable-tqdm": None,
}
REQUIRED = ("--dataset-name", "--random-prefix-len", "--random-input-len", "--random-output-len", "--num-prompts",
            "--max-concurrency", "--ignore-eos", "--temperature", "--save-result", "--result-dir", "--result-filename")
# The numbers kept of the bench's result, beside the percentiles that PERCENTILE names.
KEPT = ("duration", "completed", "failed", "total_input_tokens", "total_output_tokens", "request_throughput",
        "output_throughput", "total_token_throughput", "max_output_tokens_per_s", "max_concurrent_requests",
        "spec_decode_acceptance_rate", "spec_decode_acceptance_length", "spec_decode_num_drafts",
        "spec_decode_draft_tokens", "spec_decode_accepted_tokens")
PERCENTILE = re.compile(r"(?:mean|median|std|p50|p90|p99)_(?:ttft|tpot|itl|e2el)_ms")
RESULT_BYTES = 1 << 20


class SweepError(ValueError):
    """An argument list the pinned bench would not take. The message names a flag of FLAGS or a position."""


@dataclass(frozen=True)
class Cell:
    context: int
    at: int
    shared: bool = True

    @property
    def name(self) -> str:
        return f"{'shared' if self.shared else 'unshared'}-{self.context}-at{self.at}"


def cells(contexts: Sequence[int] = CONTEXTS, unshared: Sequence[int] = UNSHARED,
          at: Sequence[int] = AT) -> list[Cell]:
    """The sweep's cells, context by context, each from 1 request at once up."""
    return [Cell(context, count, shared) for shared, lengths in ((True, contexts), (False, unshared))
            for context in lengths for count in sorted(at)]


def arguments(base: Card, cell: Cell, result_dir: Path, rounds: int = ROUNDS) -> list[str]:
    """The bench's command for one cell, against the engine of `base`'s manifest."""
    m = base.manifest
    return [
        str(base.state / "vllm/bin/vllm"), "bench", "serve",
        "--backend", "openai", "--host", "127.0.0.1", "--port", m["ENGINE_PORT"], "--endpoint", "/v1/completions",
        "--model", m["MODEL_ALIAS"], "--tokenizer", str(base.state / "tokenizer"),
        "--dataset-name", "random", "--random-prefix-len", str(cell.context - NEW if cell.shared else 0),
        "--random-input-len", str(NEW if cell.shared else cell.context), "--random-output-len", str(OUT),
        "--num-prompts", str(rounds * cell.at), "--max-concurrency", str(cell.at),
        "--ignore-eos", "--temperature", "0", "--num-warmups", str(WARMUPS), "--seed", str(SEED),
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", PERCENTILES,
        "--save-result", "--result-dir", str(result_dir), "--result-filename", f"{cell.name}.json", "--disable-tqdm",
    ]


def check(argv: Sequence[str]) -> None:
    """That an argument list is `vllm bench serve` with FLAGS only, each once, each value of its type, and every flag
    the sweep needs. Raises SweepError."""
    if len(argv) < 3 or Path(argv[0]).name != "vllm" or list(argv[1:3]) != ["bench", "serve"]:
        raise SweepError("not a vllm bench serve command")
    seen: set[str] = set()
    at = 3
    while at < len(argv):
        flag = argv[at]
        if flag not in FLAGS:
            raise SweepError(f"an option vllm bench serve does not know, at position {at}")
        if flag in seen:
            raise SweepError(f"{flag} given twice")
        seen.add(flag)
        kind = FLAGS[flag]
        if kind is None:
            at += 1
            continue
        if at + 1 >= len(argv) or argv[at + 1].startswith("--"):
            raise SweepError(f"{flag} without its value")
        try:
            kind(argv[at + 1])
        except ValueError:
            raise SweepError(f"{flag} with a value that is not {kind.__name__}") from None
        at += 2
    if missing := [flag for flag in REQUIRED if flag not in seen]:
        raise SweepError(f"{missing[0]} missing")


def read_result(path: Path) -> dict[str, Any] | None:
    """The kept numbers of the bench's result file, and the drafter's rates by position; None without a file."""
    try:
        with path.open("rb") as file:
            data = json.loads(file.read(RESULT_BYTES))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    kept: dict[str, Any] = {key: round(value, 3) if isinstance(value, float) else value
                            for key, value in data.items()
                            if (key in KEPT or PERCENTILE.fullmatch(key)) and type(value) in (int, float)}
    rates = data.get("spec_decode_per_position_acceptance_rates")
    if isinstance(rates, list) and all(type(rate) in (int, float) for rate in rates):
        kept["spec_decode_per_position_acceptance_rates"] = [round(rate, 4) for rate in rates]
    return kept


def environment() -> dict[str, str]:
    """The engine's environment, offline, without the instance's credential."""
    return {name: value for name, value in os.environ.items() if name != "CONTAINER_API_KEY"} | card.ENGINE_ENV


async def run_cell(client: httpx.AsyncClient, base: Card, cell: Cell, result_dir: Path, rounds: int = ROUNDS,
                   sample_s: float = metrics.SAMPLE_S) -> dict[str, Any]:
    """One cell: the bench's run with /metrics read around it, and whether the cache held its requests."""
    argv = arguments(base, cell, result_dir, rounds)
    check(argv)
    engine = f"http://127.0.0.1:{base.manifest['ENGINE_PORT']}"
    result_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 - one local directory, before the cell
    before = await metrics.read(client, engine)
    began = time.time()
    async with metrics.Sampler(client, engine, sample_s) as sampler:
        process = await asyncio.create_subprocess_exec(
            *argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=environment(), start_new_session=True)
        exit_code = await process.wait()
    window = [round(began, 3), round(time.time(), 3)]
    moved = metrics.delta(before, await metrics.read(client, engine))
    peaks = sampler.peaks
    held = (exit_code == 0 and peaks.get("num_requests_running", 0) >= cell.at
            and not peaks.get("num_requests_waiting_by_reason_capacity"))
    return {"cell": cell.name, "context": cell.context, "at": cell.at, "shared": cell.shared, "exit_code": exit_code,
            "held": held, "window": window, "bench": read_result(result_dir / f"{cell.name}.json"),
            "metrics": moved, "acceptance": metrics.acceptance(moved), "peaks": peaks, "samples": sampler.samples}


async def sweep(client: httpx.AsyncClient, base: Card, result_dir: Path, planned: Sequence[Cell],
                rounds: int = ROUNDS, sample_s: float = metrics.SAMPLE_S) -> list[dict[str, Any]]:
    """The cells in order. After a cell whose requests the cache did not hold, that context's larger cells are
    skipped."""
    found: list[dict[str, Any]] = []
    stopped: set[tuple[int, bool]] = set()
    for cell in planned:
        if (cell.context, cell.shared) in stopped:
            found.append({"cell": cell.name, "context": cell.context, "at": cell.at, "shared": cell.shared,
                          "skipped": "cache"})
            continue
        result = await run_cell(client, base, cell, result_dir, rounds, sample_s)
        if not result["held"]:
            stopped.add((cell.context, cell.shared))
        found.append(result)
    return found

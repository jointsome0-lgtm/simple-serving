"""The sweep of requests at once on a text card's tail (docs/speed.md, "The sweep on a text card's tail").

    python sweep.py --until EPOCH [--plan] [--state DIR] [--manifest FILE] [--result-dir DIR] [--bench-cmd CMD]
                    [--cache-tokens N]

It runs on the card once the card's own work is done, never beside a text run, against the engine as it runs: vLLM's
`vllm bench serve` from the card's vLLM venv on the engine's port, the random dataset only, at 1 to 8 requests at once
and 4000, 16000 and 40000 tokens of context. It needs the standard library alone, so the file is copied to the card by
itself and run with the vLLM venv's Python.

A level reads the engine's /metrics before and after, and every second in between for the peaks, and prints one JSON
line of numbers and fixed words: never a prompt, an output or a path. vLLM's JSON result and its output of each level
stay in the result directory. --until is a hard end: a level starts only if its estimate fits before it, and one that
still runs then is ended. A level starts only while the engine has no request, and one that meets another request
ends the sweep. --plan prints the levels with their estimates and runs nothing.

Exit codes: 0 when the sweep ran to its end or to --until, 1 when a level failed, 2 when the engine could not be read
or served other requests.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import math
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

STATE = Path("/workspace/simple-serving-card")  # the card's state: the vLLM venv, the tokenizer (README, "The card")
MANIFEST = Path("/workspace/simple-serving/card/manifest.env")  # the manifest the engine was started from
CACHE_TOKENS = 104492  # the fp8 cache with three drafts, measured on the fourth card (README, "The card")
FILL = 0.95  # of the cache, the most a level's requests may need if they shared nothing
NEW, OUT = 256, 512  # a prompt's tokens after the shared prefix, and a request's output tokens
ORDER = (2, 4, 3, 6, 8, 1)  # requests at once, the most informative first
SETS = (("shared", 16000), ("shared", 4000), ("shared", 40000), ("none", 16000))  # prefix, context tokens

# A level's time, a model of ours and not measured: `vllm bench serve` starts in START_S, reads prompt tokens at
# READ_TOK_S (docs/speed.md estimates 3000), and one request alone answers at DECODE_TOK_S with three drafts, less
# BATCH_COST a further request at once and CONTEXT_COST a token of context (the fourth card measured 123-171 each at
# two at once). A level that runs longer than its estimate makes the estimates after it longer by the same share.
START_S, READ_TOK_S, DECODE_TOK_S, BATCH_COST, CONTEXT_COST = 15.0, 2500.0, 170.0, 0.12, 1 / 150000

# The engine's counters and gauges, as vLLM 0.30.0 names them (vllm/v1/metrics/loggers.py, vllm/v1/spec_decode/
# metrics.py). prometheus_client adds `_total` to a counter's samples.
COUNTERS = {
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits",
    "vllm:num_preemptions_total": "preemptions",
    "vllm:spec_decode_num_drafts_total": "drafts",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted",
}
PER_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos_total"  # one sample per drafted position
GAUGES = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
          "vllm:kv_cache_usage_perc": "kv_usage"}
SAMPLE = re.compile(r"(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)")
POSITION = re.compile(r'position="(\d+)"')

# What a line may hold: these keys, and as values numbers, lists of numbers, null, or one of these words.
WORDS = frozenset({"shared", "none", "planned", "done", "skipped", "cache", "until", "cut", "failed", "busy",
                   "overlap", "unreachable"})
KEYS = frozenset({"level", "set", "context", "requests", "prompts", "status", "reason", "est_s", "wall_s",
                  "exit_code", "completed", "failed_requests", "duration_s", "out_tok_s", "out_tok_s_per_request",
                  "ttft_ms_median", "ttft_ms_p90", *COUNTERS.values(), "accepted_by_pos", "running_before",
                  "waiting_before", "running_peak", "waiting_peak", "kv_usage_peak", "end", "levels_run",
                  "first6_s"})
ENDS = {"cut": "until", "failed": "failed", "busy": "busy", "overlap": "busy"}  # a level's status that ends the sweep
EXIT = {"done": 0, "until": 0, "failed": 1, "busy": 2, "unreachable": 2}
DROPPED = ("OPENAI_API_KEY", "CONTAINER_API_KEY", "SAVE_TO_PYTORCH_BENCHMARK_FORMAT")  # never given to the benchmark
# Offline, and vLLM's Python benchmark: VLLM_USE_RUST_BENCH would hand it to a Rust binary
# (vllm/entrypoints/cli/benchmark/main.py:23-35).
BENCH_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
             "VLLM_USE_RUST_BENCH": "0"}
TERM_GRACE_S, POLL_S = 10.0, 1.0


@dataclass(frozen=True)
class Level:
    index: int
    share: str  # "shared": all prompts share the context less NEW tokens; "none": they share nothing
    context: int  # a prompt's tokens
    requests: int  # at once

    @property
    def prompts(self) -> int:
        return max(4, 2 * self.requests)

    @property
    def prefix(self) -> int:
        return self.context - NEW if self.share == "shared" else 0

    @property
    def name(self) -> str:
        return f"L{self.index:02d}-{self.share}-{self.context}-c{self.requests}"

    def fits(self, cache_tokens: int) -> bool:
        """Whether the level's requests fit in the cache even if they shared nothing."""
        return (self.context + OUT) * self.requests <= FILL * cache_tokens

    def estimate(self) -> float:
        """Seconds from the benchmark's start to its end: the start, the prompt tokens read, and the waves of output.
        A shared level's warm-up reads the prefix and answers alone, and its first sibling may read the prefix again,
        since vLLM keeps a sliding layer's blocks only at a replay boundary or a junction."""
        slowed = 1 - self.context * CONTEXT_COST
        speed = DECODE_TOK_S / (1 + BATCH_COST * (self.requests - 1)) * slowed
        waves = -(-self.prompts // self.requests)
        if self.share == "shared":
            read = 2 * self.context + self.prompts * NEW
            return START_S + read / READ_TOK_S + OUT / (DECODE_TOK_S * slowed) + waves * OUT / speed
        return START_S + self.prompts * self.context / READ_TOK_S + waves * OUT / speed

    def head(self) -> dict[str, Any]:
        return {"level": self.index, "set": self.share, "context": self.context, "requests": self.requests,
                "prompts": self.prompts}


def levels() -> list[Level]:
    """Every level in the order it runs: each set of SETS at the requests of ORDER."""
    shapes = [(share, context, requests) for share, context in SETS for requests in ORDER]
    return [Level(index, *shape) for index, shape in enumerate(shapes, start=1)]


@dataclass(frozen=True)
class Target:
    command: list[str]  # what runs `vllm bench serve`
    port: str  # the engine's
    alias: str  # the model's name there
    tokenizer: Path
    result_dir: Path

    @property
    def metrics_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/metrics"


def bench_args(level: Level, target: Target) -> list[str]:
    """`vllm bench serve`'s arguments for a level (vllm/benchmarks/serve.py and vllm/benchmarks/datasets/datasets.py
    of vLLM 0.30.0). A distinct seed gives every level prompts of its own, and a shared level's one warm-up request
    reads its prefix before the clock starts."""
    shared = level.share == "shared"
    return [
        "--backend", "openai", "--host", "127.0.0.1", "--port", target.port, "--model", target.alias,
        "--tokenizer", str(target.tokenizer), "--dataset-name", "random", "--random-prefix-len", str(level.prefix),
        "--random-input-len", str(NEW if shared else level.context), "--random-output-len", str(OUT),
        "--num-prompts", str(level.prompts), "--max-concurrency", str(level.requests), "--seed", str(level.index),
        "--num-warmups", "1" if shared else "0", "--ignore-eos", "--temperature", "0",
        "--percentile-metrics", "ttft,tpot", "--metric-percentiles", "50,90", "--disable-tqdm",
        "--save-result", "--result-dir", str(target.result_dir), "--result-filename", f"{level.name}.json",
    ]


def parse_metrics(text: str) -> dict[str, float]:
    """The counters and gauges kept, each summed over its engines, and the accepted drafts by position as
    `accepted_pos<k>`."""
    found: dict[str, float] = {}
    for line in text.splitlines():
        match = SAMPLE.match(line)
        if match is None:  # a comment, or a blank line
            continue
        name, key = match["name"], COUNTERS.get(match["name"]) or GAUGES.get(match["name"])
        if name == PER_POSITION and (position := POSITION.search(match["labels"] or "")):
            key = f"accepted_pos{position[1]}"
        if key is None:
            continue
        with contextlib.suppress(ValueError):
            found[key] = found.get(key, 0.0) + float(match["value"])
    return found


def read_metrics(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return parse_metrics(response.read().decode())


def deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    """What the counters gained over a level, the warm-up included."""
    gained: dict[str, Any] = {key: round(after.get(key, 0.0) - before.get(key, 0.0)) for key in COUNTERS.values()}
    positions = sorted(int(key.removeprefix("accepted_pos")) for key in after if key.startswith("accepted_pos"))
    gained["accepted_by_pos"] = [round(after[f"accepted_pos{p}"] - before.get(f"accepted_pos{p}", 0.0))
                                 for p in positions]
    return gained


class Peaks(threading.Thread):
    """The engine's gauges every POLL_S while a level runs: the most requests running, waiting and in flight, and
    the highest cache usage. A read that fails is skipped."""

    def __init__(self, url: str) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.stopped = threading.Event()
        self.running = self.waiting = self.inflight = 0
        self.kv_usage = 0.0

    def run(self) -> None:
        while True:
            with contextlib.suppress(OSError, ValueError, http.client.HTTPException):
                self.note(read_metrics(self.url))
            if self.stopped.wait(POLL_S):
                return

    def note(self, sample: dict[str, float]) -> None:
        running, waiting = round(sample.get("running", 0.0)), round(sample.get("waiting", 0.0))
        self.running, self.waiting = max(self.running, running), max(self.waiting, waiting)
        self.inflight = max(self.inflight, running + waiting)
        self.kv_usage = max(self.kv_usage, sample.get("kv_usage", 0.0))


def number(value: object, digits: int = 1) -> float | None:
    return round(value, digits) if isinstance(value, int | float) and math.isfinite(value) else None


def outcome(result: dict[str, Any]) -> dict[str, Any]:
    """The numbers of vLLM's result: output tokens a second in all and for one request while it answers (from the
    median time per output token), and the first token's time."""
    tpot = number(result.get("median_tpot_ms"), 3)
    return {
        "completed": result.get("completed") if isinstance(result.get("completed"), int) else None,
        "failed_requests": result.get("failed") if isinstance(result.get("failed"), int) else None,
        "duration_s": number(result.get("duration")),
        "out_tok_s": number(result.get("output_throughput")),
        "out_tok_s_per_request": round(1000 / tpot, 1) if tpot else None,
        "ttft_ms_median": number(result.get("median_ttft_ms")),
        "ttft_ms_p90": number(result.get("p90_ttft_ms")),
    }


def bench_env() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if name not in DROPPED} | BENCH_ENV


def end(process: subprocess.Popen[bytes]) -> None:
    """End the benchmark and its children: SIGTERM, then SIGKILL after TERM_GRACE_S. Its requests end with their
    connections."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signum)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(TERM_GRACE_S)
            return


def run_level(level: Level, until: float, *, target: Target, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """One level: the engine's counters before, `vllm bench serve` until it ends or `until` comes, the counters
    after, and what vLLM's result says. A level whose engine has requests before it starts nothing."""
    before = read_metrics(target.metrics_url)
    line: dict[str, Any] = {"running_before": round(before.get("running", 0.0)),
                            "waiting_before": round(before.get("waiting", 0.0))}
    if line["running_before"] or line["waiting_before"]:
        return {"status": "busy"} | line
    (target.result_dir / f"{level.name}.json").unlink(missing_ok=True)  # never an older run's result
    peaks = Peaks(target.metrics_url)
    peaks.start()
    status, code = "done", None
    with (target.result_dir / f"{level.name}.log").open("wb") as log:
        process = subprocess.Popen([*target.command, *bench_args(level, target)], stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, env=bench_env(), start_new_session=True)
        try:
            code = process.wait(max(until - clock(), 0.1))
        except subprocess.TimeoutExpired:
            end(process)
            status = "cut"
    peaks.stopped.set()
    peaks.join()
    after = read_metrics(target.metrics_url)
    line |= deltas(before, after) | {"running_peak": peaks.running, "waiting_peak": peaks.waiting,
                                     "kv_usage_peak": round(peaks.kv_usage, 3)}
    if status == "cut":
        return {"status": status} | line
    try:
        result = json.loads((target.result_dir / f"{level.name}.json").read_text())
    except (OSError, ValueError):
        result = None
    if code != 0 or not isinstance(result, dict):
        return {"status": "failed", "exit_code": code} | line
    # Our requests are at most `requests` in flight, the warm-up alone before them: more are another client's.
    return {"status": "overlap" if peaks.inflight > level.requests else "done"} | outcome(result) | line


def sweep(plan: list[Level], until: float, run: Callable[[Level, float], dict[str, Any]], *,
          clock: Callable[[], float] = time.time, cache_tokens: int = CACHE_TOKENS) -> Iterator[dict[str, Any]]:
    """The levels in order, each started only if it fits in the cache and its estimate before `until`; then a last
    line with how the sweep ended."""
    started, ran, spent, estimated, stretch, ended = clock(), 0, 0.0, 0.0, 1.0, "done"
    for level in plan:
        if not level.fits(cache_tokens):
            yield level.head() | {"status": "skipped", "reason": "cache"}
            continue
        estimate = round(level.estimate() * stretch, 1)
        if clock() + estimate > until:
            ended = "until"
            yield level.head() | {"status": "skipped", "reason": "until", "est_s": estimate}
            continue
        begun = clock()
        result = run(level, until)
        wall = clock() - begun
        yield level.head() | {"est_s": estimate, "wall_s": round(wall, 1)} | result
        if result["status"] in ENDS:
            ended = ENDS[result["status"]]
            break
        ran, spent, estimated = ran + 1, spent + wall, estimated + level.estimate()
        stretch = max(1.0, spent / estimated)
    yield {"end": ended, "levels_run": ran, "wall_s": round(clock() - started, 1)}


class Simulated:
    """A clock that a planned level moves on by its estimate."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def run(self, level: Level, until: float) -> dict[str, Any]:
        self.now += level.estimate()
        return {"status": "planned"}


def checked(line: dict[str, Any]) -> str:
    """The line as JSON, once every key is one of KEYS and every value a number, null or a word of WORDS."""
    for key, value in line.items():
        if key not in KEYS:
            raise ValueError("a key outside the fixed set")
        for item in value if isinstance(value, list) else [value]:
            if (isinstance(item, str) and item in WORDS) or item is None:
                continue
            if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
                raise ValueError(f"a value under {key} that is not a number or a fixed word")
    return json.dumps(line, separators=(",", ":"))


def read_manifest(path: Path) -> dict[str, str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return {name: value for name, _, value in (line.partition("=") for line in lines)
            if name and not name.startswith("#")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The sweep of requests at once on a text card's tail.")
    parser.add_argument("--until", type=float, help="the hard end, in seconds since the epoch")
    parser.add_argument("--plan", action="store_true", help="print the levels and their estimates, run nothing")
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--result-dir", type=Path, help="default: sweep/<UTC time> in the card's state")
    parser.add_argument("--bench-cmd", help="what runs `vllm bench serve`, default the vLLM venv's")
    parser.add_argument("--cache-tokens", type=int, default=CACHE_TOKENS)
    args = parser.parse_args(argv)
    plan = levels()
    if args.plan:
        simulated = Simulated()
        lines = list(sweep(plan, math.inf if args.until is None else args.until - time.time(), simulated.run,
                           clock=simulated, cache_tokens=args.cache_tokens))
        first = [line["est_s"] for line in lines if line.get("status") == "planned"][:6]
        lines[-1]["first6_s"] = round(sum(first), 1)
        for line in lines:
            print(checked(line), flush=True)
        return 0
    if args.until is None:
        parser.error("--until is required")
    manifest = read_manifest(args.manifest)
    command = shlex.split(args.bench_cmd) if args.bench_cmd else [str(args.state / "vllm/bin/vllm"), "bench", "serve"]
    result_dir = args.result_dir or args.state / "sweep" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    result_dir.mkdir(parents=True, exist_ok=True)
    target = Target(command, manifest["ENGINE_PORT"], manifest["MODEL_ALIAS"], args.state / "tokenizer", result_dir)
    ended = "unreachable"
    try:
        for line in sweep(plan, args.until, partial(run_level, target=target), cache_tokens=args.cache_tokens):
            print(checked(line), flush=True)
            ended = line.get("end", ended)
    except (OSError, http.client.HTTPException):  # the engine's /metrics could not be read
        print(checked({"end": "unreachable"}), flush=True)
    return EXIT[ended]


if __name__ == "__main__":
    raise SystemExit(main())

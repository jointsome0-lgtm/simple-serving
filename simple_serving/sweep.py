"""Standalone concurrency measurement against an already running text engine (docs/sweep.md).

    python sweep.py --until EPOCH [--plan] [--state DIR] [--manifest FILE] [--result-dir DIR] [--bench-cmd CMD]
                    --cache-tokens N

It runs on the card once the card's own work is done, never beside a text run, against the engine as it runs: vLLM's
`vllm bench serve` from the card's vLLM venv on the engine's port, the random dataset only, at 1 to 8 requests at once
and 4000, 16000 and 40000 tokens of context. It needs the standard library alone, so the file is copied to the card by
itself and run with the vLLM venv's Python.

A level reads the engine's /metrics before and after, and every second in between for the peaks, and prints one JSON
line of numbers and fixed words: never a prompt, an output or a path. vLLM's JSON result and its output of each level
stay in the private result directory. --until is the launch deadline: a level starts only if its estimate fits
before it, and one still running then is terminated with bounded cleanup. It does not bound provider billing.
A level starts only while the engine has no request, and one that meets another request
ends the sweep. --plan prints the levels with their estimates and runs nothing.

Exit codes: 0 when the sweep ran to its end or to --until, 1 when a level failed, 2 when the engine could not be read
or served other requests; 130 when interrupted. The engine itself is never restarted or stopped.
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
import socket
import subprocess
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

STATE = Path("/workspace/simple-serving-card")  # the card's state: the vLLM venv, the tokenizer (README, "The card")
MANIFEST = Path("/workspace/simple-serving/card/manifest.env")  # the manifest the engine was started from
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
                   "overlap", "unreachable", "cancelled"})
KEYS = frozenset({"level", "set", "context", "requests", "prompts", "status", "reason", "est_s", "wall_s",
                  "exit_code", "completed", "failed_requests", "duration_s", "out_tok_s", "out_tok_s_per_request",
                  "ttft_ms_median", "ttft_ms_p90", *COUNTERS.values(), "accepted_by_pos", "running_before",
                  "waiting_before", "running_peak", "waiting_peak", "kv_usage_peak", "end", "levels_run",
                  "first6_s", "cache_tokens"})
ENDS = {"cut": "until", "failed": "failed", "busy": "busy", "overlap": "busy", "unreachable": "unreachable"}
EXIT = {"done": 0, "until": 0, "failed": 1, "busy": 2, "unreachable": 2, "cancelled": 130}
DROPPED = ("OPENAI_API_KEY", "CONTAINER_API_KEY", "SAVE_TO_PYTORCH_BENCHMARK_FORMAT")  # never given to the benchmark
# Offline, and vLLM's Python benchmark: VLLM_USE_RUST_BENCH would hand it to a Rust binary
# (vllm/entrypoints/cli/benchmark/main.py:23-35).
BENCH_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
             "VLLM_USE_RUST_BENCH": "0", "NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1"}
TERM_GRACE_S, POLL_S = 10.0, 1.0
METRICS_TIMEOUT_S, METRICS_BYTES, RESULT_BYTES = 5.0, 4 << 20, 4 << 20


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
        return (self.context + OUT) * self.requests * 100 <= round(FILL * 100) * cache_tokens

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


def activity(sample: dict[str, float]) -> tuple[int, int]:
    """Require both request gauges; absence or malformed values cannot establish an idle engine."""
    counts = []
    for key in ("running", "waiting"):
        value = sample.get(key)
        if (isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value)
                or value < 0 or value != int(value)):
            raise ValueError("invalid request gauge")
        counts.append(int(value))
    return counts[0], counts[1]


def parse_metrics(text: str) -> dict[str, float]:
    """Sum known metrics over engines, requiring valid request gauges and finite nonnegative samples."""
    found: dict[str, float] = {}
    for line in text.splitlines():
        name = re.split(r"[{\s]", line, maxsplit=1)[0]
        key = COUNTERS.get(name) or GAUGES.get(name)
        if key is None and name != PER_POSITION:
            continue
        match = SAMPLE.fullmatch(line)
        if match is None:
            raise ValueError("malformed engine metric")
        if name == PER_POSITION:
            position = POSITION.search(match["labels"] or "")
            if position is None:
                raise ValueError("missing draft position")
            key = f"accepted_pos{position[1]}"
        assert key is not None
        value = float(match["value"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid engine metric")
        if key in ("running", "waiting") and value != int(value):
            raise ValueError("invalid request gauge")
        total = found.get(key, 0.0) + value
        if not math.isfinite(total):
            raise ValueError("invalid metric sum")
        found[key] = total
    activity(found)
    return found


def read_metrics(url: str) -> dict[str, float]:
    # A direct loopback connection never uses environment proxies or follows redirects. Shutdown enforces one wall
    # deadline, including a server that trickles headers or a body slowly enough to evade per-read socket timeouts.
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path != "/metrics":
        raise ValueError("invalid metrics endpoint")
    deadline = time.monotonic() + METRICS_TIMEOUT_S
    connection = http.client.HTTPConnection("127.0.0.1", parsed.port, timeout=METRICS_TIMEOUT_S)
    timer: threading.Timer | None = None
    try:
        connection.connect()
        connected = connection.sock
        assert connected is not None

        def expire() -> None:
            with contextlib.suppress(OSError):
                connected.shutdown(socket.SHUT_RDWR)
            connected.close()

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("metrics deadline")
        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        timer.start()
        connection.request("GET", "/metrics")
        with connection.getresponse() as response:
            if response.status != 200:
                raise ValueError("metrics response")
            body = response.read(METRICS_BYTES + 1)
        if len(body) > METRICS_BYTES or time.monotonic() >= deadline:
            raise ValueError("metrics limit")
        return parse_metrics(body.decode())
    finally:
        connection.close()
        if timer is not None:
            timer.cancel()
            timer.join(1)


def deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    """Counter changes, warm-up included. Missing counters are unknown; a reset invalidates the measurement."""
    def change(key: str) -> int | None:
        if key not in before or key not in after:
            return None
        value = after[key] - before[key]
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid counter delta")
        return round(value)

    gained: dict[str, Any] = {key: change(key) for key in COUNTERS.values() if key in before and key in after}
    positions = sorted(int(key.removeprefix("accepted_pos")) for key in before.keys() & after.keys()
                       if key.startswith("accepted_pos"))
    gained["accepted_by_pos"] = [change(f"accepted_pos{p}") for p in positions]
    return gained


class Peaks(threading.Thread):
    """Sample gauges while a level runs. An unreadable sample invalidates the measurement."""

    def __init__(self, url: str) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.stopped = threading.Event()
        self.failed = threading.Event()
        self.running = self.waiting = self.inflight = 0
        self.kv_usage: float | None = None

    def run(self) -> None:
        while not self.stopped.is_set():
            try:
                self.note(read_metrics(self.url))
            except (OSError, ValueError, http.client.HTTPException):
                self.failed.set()
                return
            self.stopped.wait(POLL_S)

    def note(self, sample: dict[str, float]) -> None:
        running, waiting = activity(sample)
        self.running, self.waiting = max(self.running, running), max(self.waiting, waiting)
        self.inflight = max(self.inflight, running + waiting)
        if "kv_usage" in sample:
            value = sample["kv_usage"]
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid cache gauge")
            self.kv_usage = max(self.kv_usage or 0.0, value)


def number(value: object, digits: int = 1) -> float | None:
    return (round(value, digits) if not isinstance(value, bool) and isinstance(value, int | float)
            and math.isfinite(value) and value >= 0 else None)


def outcome(result: dict[str, Any], expected: int) -> dict[str, Any]:
    """Require a complete successful benchmark before publishing its numeric measurements."""
    if (type(result.get("completed")) is not int or result["completed"] != expected
            or type(result.get("failed")) is not int or result["failed"] != 0):
        raise ValueError("incomplete benchmark")
    values = {key: number(result.get(key), 6) for key in
              ("duration", "output_throughput", "median_tpot_ms", "median_ttft_ms", "p90_ttft_ms")}
    if any(value is None for value in values.values()):
        raise ValueError("invalid benchmark result")
    duration, tpot = values["duration"], values["median_tpot_ms"]
    assert duration is not None and tpot is not None
    if duration <= 0 or tpot <= 0:
        raise ValueError("invalid benchmark timing")
    return {
        "completed": result["completed"], "failed_requests": result["failed"],
        "duration_s": number(duration), "out_tok_s": number(values["output_throughput"]),
        "out_tok_s_per_request": round(1000 / tpot, 1),
        "ttft_ms_median": number(values["median_ttft_ms"]), "ttft_ms_p90": number(values["p90_ttft_ms"]),
    }


def bench_env() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if name not in DROPPED} | BENCH_ENV


def end(process: subprocess.Popen[bytes]) -> None:
    """Terminate only the benchmark's process group, with bounded waits, including any surviving children."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(TERM_GRACE_S)
    # The parent can exit on TERM while a child ignores it; always kill the remaining group as well.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(TERM_GRACE_S)


def limits(until: float, cache_tokens: int = 1) -> None:
    if isinstance(until, bool) or not math.isfinite(until) or until <= 0:
        raise ValueError("until must be finite and positive")
    if type(cache_tokens) is not int or cache_tokens <= 0:
        raise ValueError("cache tokens must be a positive integer")


def run_level(level: Level, until: float, *, target: Target, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Run one benchmark against the existing engine; always clean up its child processes and sampler."""
    limits(until)
    line: dict[str, Any] = {}
    try:
        before = read_metrics(target.metrics_url)
        running, waiting = activity(before)
    except (OSError, ValueError, http.client.HTTPException):
        return {"status": "unreachable"}
    line |= {"running_before": running, "waiting_before": waiting}
    if running or waiting:
        return {"status": "busy"} | line
    if clock() >= until:
        return {"status": "cut"} | line
    process: subprocess.Popen[bytes] | None = None
    peaks = Peaks(target.metrics_url)
    status, code = "done", None
    result_path = target.result_dir / f"{level.name}.json"
    try:
        target.result_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.result_dir.is_symlink() or target.result_dir.stat().st_mode & 0o777 != 0o700:
            raise ValueError("result directory must be private")
        result_path.unlink(missing_ok=True)
        log_path = target.result_dir / f"{level.name}.log"
        descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as log:
            os.fchmod(log.fileno(), 0o600)
            process = subprocess.Popen([*target.command, *bench_args(level, target)], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, env=bench_env(),
                                       start_new_session=True, umask=0o077)
            peaks.note(before)
            peaks.start()
            while True:
                if peaks.failed.is_set():
                    status = "unreachable"
                    break
                if peaks.inflight > level.requests:
                    status = "overlap"
                    break
                remaining = until - clock()
                if remaining <= 0:
                    status = "cut"
                    break
                try:
                    code = process.wait(min(POLL_S, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
    except (OSError, ValueError):
        status = "failed"
    finally:
        peaks.stopped.set()
        try:
            if process is not None:
                try:
                    end(process)
                except (OSError, subprocess.TimeoutExpired):
                    status = "failed"
        finally:
            if peaks.ident is not None:
                peaks.join(2 * METRICS_TIMEOUT_S + 1)
                if peaks.is_alive():
                    status = "unreachable"
    if peaks.failed.is_set() and status == "done":
        status = "unreachable"
    line |= {"running_peak": peaks.running, "waiting_peak": peaks.waiting,
             "kv_usage_peak": number(peaks.kv_usage, 3)}
    if status != "done":
        return {"status": status} | line
    try:
        after = read_metrics(target.metrics_url)
        peaks.note(after)
        line |= deltas(before, after) | {"running_peak": peaks.running, "waiting_peak": peaks.waiting,
                                         "kv_usage_peak": number(peaks.kv_usage, 3)}
    except (OSError, ValueError, http.client.HTTPException):
        return {"status": "unreachable"} | line
    try:
        if code != 0 or result_path.stat().st_size > RESULT_BYTES:
            raise ValueError("failed benchmark")
        result_path.chmod(0o600)
        result = json.loads(result_path.read_text())
        if not isinstance(result, dict):
            raise TypeError("malformed benchmark result")
        summary = outcome(result, level.prompts)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        return {"status": "failed", "exit_code": code} | line
    return {"status": "overlap" if peaks.inflight > level.requests else "done"} | summary | line


def sweep(plan: list[Level], until: float, run: Callable[[Level, float], dict[str, Any]], *,
          cache_tokens: int, clock: Callable[[], float] = time.time) -> Iterator[dict[str, Any]]:
    """The levels in order, each started only if it fits in the cache and its estimate before `until`; then a last
    line with how the sweep ended."""
    limits(until, cache_tokens)
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
    yield {"end": ended, "levels_run": ran, "wall_s": round(clock() - started, 1), "cache_tokens": cache_tokens}


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
    parser.add_argument("--until", type=float, help="the benchmark cutoff, in seconds since the epoch")
    parser.add_argument("--plan", action="store_true", help="print the levels and their estimates, run nothing")
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--result-dir", type=Path, help="default: sweep/<UTC time> in the card's state")
    parser.add_argument("--bench-cmd", help="what runs `vllm bench serve`, default the vLLM venv's")
    parser.add_argument("--cache-tokens", type=int, required=True, help="measured positive KV cache capacity")
    args = parser.parse_args(argv)
    try:
        limits(args.until if args.until is not None else 1.0, args.cache_tokens)
    except ValueError as error:
        parser.error(str(error))
    plan = levels()
    if args.plan:
        simulated = Simulated()
        simulated.now = time.time()
        until = args.until if args.until is not None else simulated.now + sum(level.estimate() for level in plan) + 1
        lines = list(sweep(plan, until, simulated.run,
                           clock=simulated, cache_tokens=args.cache_tokens))
        first = [line["est_s"] for line in lines if line.get("status") == "planned"][:6]
        lines[-1]["first6_s"] = round(sum(first), 1)
        for line in lines:
            print(checked(line), flush=True)
        return 0
    if args.until is None:
        parser.error("--until is required")
    ended = "unreachable"
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        manifest = read_manifest(args.manifest)
        command = (shlex.split(args.bench_cmd) if args.bench_cmd else
                   [str(args.state / "vllm/bin/vllm"), "bench", "serve"])
        if not command:
            raise ValueError("empty benchmark command")
        result_dir = args.result_dir or args.state / "sweep" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        target = Target(command, manifest["ENGINE_PORT"], manifest["MODEL_ALIAS"], args.state / "tokenizer", result_dir)
        result_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        for line in sweep(plan, args.until, partial(run_level, target=target), cache_tokens=args.cache_tokens):
            print(checked(line), flush=True)
            ended = line.get("end", ended)
    except KeyboardInterrupt:
        ended = "cancelled"
        print(checked({"end": ended, "cache_tokens": args.cache_tokens}), flush=True)
    except (OSError, ValueError, KeyError, http.client.HTTPException):
        print(checked({"end": "unreachable", "cache_tokens": args.cache_tokens}), flush=True)
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    return EXIT[ended]


def interrupted(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    raise SystemExit(main())

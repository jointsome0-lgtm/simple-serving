"""The throughput harness's commands (docs/speed.md, item 1 and the runbook), on the card as its launcher runs:

    python -m simple_serving.throughput run --change CHANGE [--pack DIR --workloads W,... --at N,...]
        [--identity reference|change] [--sweep]
    python -m simple_serving.throughput report RESULT.json
    python -m simple_serving.throughput microbench gemm|attention [--kernels K,...] [--dry-run]
    python -m simple_serving.throughput dry-run

`run` ends the launcher's pair, holds the launcher's lock, and starts vLLM and the gateway with one change
(launcher.py). Once its engine_choice rows show the change, it runs the identity set, then each workload at each count
at once through the gateway, then the timing sweep, and ends the pair; the card runs no pair until
`python -m simple_serving.card` starts the launcher again. A run keeps its directory under the card's state
(throughput/<time>-<change>): its gateway configuration, its logs, the bench's results and result.json, whose report it
prints. `microbench` runs the cheap card's steps 2 and 3 with the card's vLLM venv, after ending the pair. `dry-run`
runs all of it on this machine against the fake engine (dry_run.py).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import os
import re
import resource
import secrets
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .. import card, log
from ..card import Card
from . import identity, launcher, metrics, microbench, overlap, report, runner, sweep, workloads

RUNS = "throughput"  # in the card's state, a directory a run
IDENTITY = "throughput/identity"  # in the card's state: the digest key and the reference's digests
NOT_READY, NOT_WRITTEN = 4, 5  # exit codes beside card.py's and launcher.STILL_RUNNING
REQUEST_TIMEOUT_S = 900.0  # a compaction of 4096 tokens at a batch of 8
PROBE_CONTEXT = 4000  # the request whose count gives the characters a token


class RunError(ValueError):
    """Options that do not make a run. The message is a fixed sentence."""


@dataclass(frozen=True)
class Options:
    change: str = "reference"
    pack: Path | None = None  # a copy of the public eval pack, simple-story-chat-eval
    workloads: tuple[str, ...] = ()
    at: tuple[int, ...] = (1, 2, 4, 8)
    context: int | None = None  # None: each workload's default
    turns: int = 4
    gap_s: float = 10.0
    identity: str | None = None  # "reference" or "change"
    identity_at: int = 1
    contexts: tuple[int, ...] = sweep.CONTEXTS
    unshared: tuple[int, ...] = sweep.UNSHARED
    sweep_at: tuple[int, ...] = sweep.AT
    rounds: int = sweep.ROUNDS
    sample_s: float = metrics.SAMPLE_S
    out: Path | None = None  # None: the card's state/throughput
    sweep: bool = False  # last, since the name hides the module's in the class


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_directory(root: Path, change: launcher.Change) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", change.slug.lower()).strip("-")
    path = root / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{slug}"
    path.mkdir(parents=True, mode=0o700)
    (path / "logs").mkdir(mode=0o700)
    return path


@contextlib.contextmanager
def launcher_lock(state: Path) -> Iterator[bool]:
    """The launcher's own lock, so that no launcher starts a pair beside the harness's; False if another holds it."""
    descriptor = os.open(state / "launcher.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        os.close(descriptor)


async def measure(client: httpx.AsyncClient, base: Card, plan: launcher.Plan, target: runner.Target,
                  options: Options, run_dir: Path, stories: list[workloads.Story]) -> dict[str, Any]:
    """The identity set, the passes and the sweep of one start whose change took effect."""
    found: dict[str, Any] = {}
    alias = plan.manifest["MODEL_ALIAS"]
    chars = workloads.Shape().chars_per_token
    if stories:
        probe = workloads.build("unshared", stories, alias, workloads.Shape(at=1, context=PROBE_CONTEXT))[0][0]
        chars = await runner.chars_per_token(client, target, probe) or chars
        found["chars_per_token"] = chars
    if options.identity:
        found["identity"] = await identity.measure(client, target.engine, identity.payloads(stories, alias),
                                                   options.identity, base.state / IDENTITY, options.identity_at)
    passes = []
    for workload in options.workloads:
        for at in (1,) if workload in ("reading", "interleaved") else options.at:
            shape = workloads.Shape(at=at, context=options.context, turns=options.turns, gap_s=options.gap_s,
                                    chars_per_token=chars)
            lanes = workloads.build(workload, stories, alias, shape)
            entry = await runner.run_pass(client, target, lanes, workloads.at_once(workload, shape),
                                          sample_s=options.sample_s)
            passes.append({"workload": workload, **entry})
    if passes:
        found["passes"] = passes
    if options.sweep:
        planned = sweep.cells(options.contexts, options.unshared, options.sweep_at)
        found["sweep"] = await sweep.sweep(client, replace(base, manifest=plan.manifest), run_dir / "bench", planned,
                                           options.rounds, options.sample_s)
    return found


async def run(base: Card, options: Options) -> tuple[int, dict[str, Any] | None]:
    """One run: the exit code and its result, which is also written in its directory."""
    change = launcher.parse_change(options.change)
    plan = launcher.plan_for(base, change)
    if plan.timing_only and (options.workloads or options.identity):
        raise RunError("a timing-only change runs the sweep alone: no workload and no identity set")
    if (options.workloads or options.identity) and options.pack is None:
        raise RunError("the workloads and the identity set are built from --pack, a copy of the public eval pack")
    if options.identity not in (None, "reference", "change"):
        raise RunError("--identity is reference or change")
    stories = workloads.load_pack(options.pack) if options.pack is not None else []
    if code := launcher.take_over(base):
        return code, None
    with launcher_lock(base.state) as held:
        if not held:
            return launcher.STILL_RUNNING, None
        return await started_run(base, plan, options, stories)


async def started_run(base: Card, plan: launcher.Plan, options: Options,
                      stories: list[workloads.Story]) -> tuple[int, dict[str, Any] | None]:
    run_dir = run_directory(options.out or base.state / RUNS, plan.change)
    log.setup(card.JsonLines(run_dir / "logs/card.jsonl", card.CARD_LOG_BYTES))
    key = secrets.token_urlsafe(32)  # the harness's own key: in memory, its hash in the gateway's configuration
    m = plan.manifest
    target = runner.Target(f"http://127.0.0.1:{m['PUBLIC_PORT']}", f"http://127.0.0.1:{m['ENGINE_PORT']}", key,
                           m["MODEL_ALIAS"])
    result: dict[str, Any] = {"harness": report.VERSION, "change": plan.change.slug, "timing_only": plan.timing_only,
                              "started": now(), "versions": card.pinned_versions(m)}
    shown = False
    async with launcher.started(base, plan, run_dir, key) as start:
        result |= {"outcome": start.outcome, "load_s": start.load_s}
        if start.pair.choices or start.outcome == "ready":
            result["proof"] = launcher.proof(plan, start.pair.choices)
            result["engine"] = dict(start.pair.measured)
        shown = start.outcome == "ready" and result["proof"]["ok"]
        if shown:
            async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=10)) as client:
                result |= await measure(client, base, plan, target, options, run_dir, stories)
    measured = [*result.get("passes", ()), *(cell for cell in result.get("sweep", ()) if "window" in cell)]
    result["flagged"] = overlap.flag(measured, overlap.reader_requests(run_dir / "logs"))  # the pair's log is whole
    result["ended"] = now()
    try:
        report.write(run_dir, result)
    except report.ResultError:
        return NOT_WRITTEN, None
    return (0 if shown else NOT_READY), result


def microbench_argv(base: Card, kind: str, kernels: str | None) -> list[str]:
    """The microbenchmark's command with the card's vLLM venv."""
    chosen = ("--kernels", kernels) if kernels else ()
    return [str(base.state / "vllm/bin/python"), microbench.__file__, kind, *chosen]


def run_microbench(base: Card, kind: str, kernels: str | None, dry_run: bool, out: Path | None = None) -> int:
    argv = microbench_argv(base, kind, kernels)
    if dry_run:
        return subprocess.run([*argv, "--dry-run"], env=sweep.environment(), stdin=subprocess.DEVNULL,
                              check=False).returncode
    if code := launcher.take_over(base):
        return code
    with launcher_lock(base.state) as held:
        if not held:
            return launcher.STILL_RUNNING
        run_dir = run_directory(out or base.state / RUNS, launcher.Change(f"microbench-{kind}"))
        return subprocess.run([*argv, "--out", str(run_dir / f"{kind}.json")], env=sweep.environment(),
                              stdin=subprocess.DEVNULL, check=False).returncode


def numbers(text: str) -> tuple[int, ...]:
    try:
        found = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("a comma-separated list of whole numbers") from None
    if not found or min(found) < 1:
        raise argparse.ArgumentTypeError("a comma-separated list of whole numbers from 1")
    return found


def names(text: str) -> tuple[str, ...]:
    found = tuple(text.split(","))
    if unknown := [name for name in found if name not in workloads.NAMES]:
        raise argparse.ArgumentTypeError(f"unknown workload {unknown[0]!r}; known: {', '.join(workloads.NAMES)}")
    return found


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m simple_serving.throughput", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="one start with one change, and what it measured")
    run_parser.add_argument("--change", default="reference", help=f"NAME or NAME=VALUE: {', '.join(launcher.CHANGES)}")
    run_parser.add_argument("--pack", type=Path, help="a copy of the public eval pack (simple-story-chat-eval)")
    run_parser.add_argument("--workloads", type=names, default=(), help=f"of {', '.join(workloads.NAMES)}")
    run_parser.add_argument("--at", type=numbers, default=(1, 2, 4, 8), help="requests at once, a pass each")
    run_parser.add_argument("--context", type=int, help="the target context in tokens, else each workload's own")
    run_parser.add_argument("--turns", type=int, default=4)
    run_parser.add_argument("--gap-s", type=float, default=10.0, help="the pause between a reader's turns")
    run_parser.add_argument("--identity", choices=("reference", "change"), help="run the identity set as this side")
    run_parser.add_argument("--identity-at", type=int, default=1, help="the identity set's requests at once")
    run_parser.add_argument("--sweep", action="store_true", help="the timing sweep with vllm bench serve")
    run_parser.add_argument("--contexts", type=numbers, default=sweep.CONTEXTS, help="the sweep's shared contexts")
    run_parser.add_argument("--unshared", type=numbers, default=sweep.UNSHARED, help="its contexts that share nothing")
    run_parser.add_argument("--sweep-at", type=numbers, default=sweep.AT, help="its requests at once")
    run_parser.add_argument("--rounds", type=int, default=sweep.ROUNDS, help="requests a cell sends for each at once")
    run_parser.add_argument("--out", type=Path, help="where the run's directory goes, else the card's state")
    report_parser = commands.add_parser("report", help="a result's short report")
    report_parser.add_argument("result", type=Path)
    bench_parser = commands.add_parser("microbench", help="the cheap card's FP4 GEMMs or attention")
    bench_parser.add_argument("kind", choices=tuple(microbench.KINDS))
    bench_parser.add_argument("--kernels", help="a comma-separated subset of the kind's kernels")
    bench_parser.add_argument("--dry-run", action="store_true", help="print the plan; needs no GPU")
    dry_parser = commands.add_parser("dry-run", help="all of it on this machine, against the fake engine")
    dry_parser.add_argument("--keep", type=Path, help="keep the dry run's card here instead of a temporary directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "report":
        try:
            print(report.text(report.read(args.result)))
        except (OSError, ValueError):
            print("not a result of this harness", file=sys.stderr)
            return 2
        return 0
    if args.command == "dry-run":
        from . import dry_run  # noqa: PLC0415 - the fake engine's stand-ins, never on a card

        return dry_run.main(args.keep)
    base = Card.load()
    if args.command == "microbench":
        return run_microbench(base, args.kind, args.kernels, args.dry_run)
    options = Options(change=args.change, pack=args.pack, workloads=args.workloads, at=args.at, context=args.context,
                      turns=args.turns, gap_s=args.gap_s, identity=args.identity, identity_at=args.identity_at,
                      sweep=args.sweep, contexts=args.contexts, unshared=args.unshared, sweep_at=args.sweep_at,
                      rounds=args.rounds, out=args.out)
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # a core dump of either process could hold prompts
    try:
        code, result = asyncio.run(run(base, options))
    except (launcher.ChangeError, RunError, identity.IdentityError) as error:
        print(f"throughput: {error}", file=sys.stderr)
        return 2
    if result is not None:
        print(report.text(result))
    return code

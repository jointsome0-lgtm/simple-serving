"""The A/B launcher (docs/speed.md, item 1 and the runbook's "Before any card", step 2).

It ends the launcher's pair (`card --stop`) and starts vLLM and the gateway itself: the card's own arguments from
`manifest.env` plus exactly one change, and the rental's gateway limits. manifest.env is never edited, so the card
stays prepared; a change of a manifest value is made on a copy in memory. The gateway gets no instance credential, so
neither it nor the harness can stop the instance: a start that fails leaves the card running, where `card` would stop
it. Before anything is measured, the engine's `engine_choice` rows (card.py) must show that the change took effect.

A change is `NAME` or `NAME=VALUE`, one of CHANGES. `synthetic` makes the engine accept drafts at the given rates
whatever they are, so its run is timing only: its gateway leaves out the bot's client key, and the harness runs no
workload through it, only the sweep. When the run ends, the pair it started ends too, and the card runs no pair until
`python -m simple_serving.card` starts the launcher again.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import os
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .. import card
from ..card import Card, Command

# The rental's gateway limits: up to the engine's MAX_NUM_SEQS of the harness's own `internal` requests at once, so
# that the gateway does not cap the batch a pass asks for (docs/speed.md, item 3). The rest stays provisional.
RENTAL_LIMITS = {"internal": {"active": 8, "waiting": 16}, "shared": {"active": 8}}
IDLE_S = 3 * 3600  # the harness's gateway: a sweep that bypasses it must not put it to sleep
HARNESS_LABEL = "harness"
DEFAULT_GPU_MEMORY = 0.92  # vLLM 0.30.0's, which `non-default args` leaves out (vllm/config/cache.py:103)
LINEAR_KERNELS = {"marlin": "marlin", "humming": "humming", "flashinfer_cutedsl": "flashinfer_cutedsl_w4a16"}
FLASHINFER_ENV = {"VLLM_HAS_FLASHINFER_CUBIN": "1"}  # item 13: no download of a cubin that sm_120 does not use
STILL_RUNNING = 1  # the launcher's pair did not end
MAX_DRAFTS = 8  # drafted tokens a change may ask for: 6 for the acceptance pass, 5 at most for the sweep


class ChangeError(ValueError):
    """A change the launcher does not know or cannot apply. The message names the change, never a value."""


@dataclass(frozen=True)
class Change:
    name: str
    value: str = ""

    @property
    def slug(self) -> str:
        return f"{self.name}={self.value}" if self.value else self.name


@dataclass
class Plan:
    """One start: the engine's command, the manifest as changed in memory, and what its engine_choice rows must
    show."""

    change: Change
    manifest: dict[str, str]
    engine: Command
    expect: list[tuple[str, int | None]] = field(default_factory=list)  # a code, with its value where it has one
    absent: list[str] = field(default_factory=list)
    timing_only: bool = False


def parse_change(text: str) -> Change:
    name, _, value = text.partition("=")
    if name not in CHANGES:
        raise ChangeError(f"unknown change {name!r}; known: {', '.join(CHANGES)}")
    return Change(name, value)


def plan_for(base: Card, change: Change) -> Plan:
    """The start of `change` on this card. Raises ChangeError when the value does not fit the change."""
    apply = CHANGES[change.name]
    manifest = dict(base.manifest)
    edits: list[Callable[[Plan], None]] = []
    expect: list[tuple[str, int | None]] = []
    timing_only = apply(change.value, manifest, edits, expect)
    changed = replace(base, manifest=manifest)
    plan = Plan(change, manifest, changed.engine(), timing_only=timing_only)
    plan.expect, plan.absent = reference_expectations(manifest)
    for edit in edits:
        edit(plan)
    plan.expect += [item for item in expect if item not in plan.expect]
    return plan


def reference_expectations(manifest: Mapping[str, str]) -> tuple[list[tuple[str, int | None]], list[str]]:
    """What `non-default args` and the offloading line show of the manifest's own values. A value equal to vLLM's
    default is not printed, and is not expected."""
    expect: list[tuple[str, int | None]] = [
        ("arg_max_model_len", int(manifest["CONTEXT_TOKENS"])), ("arg_max_num_seqs", int(manifest["MAX_NUM_SEQS"])),
        ("arg_max_num_batched_tokens", int(manifest["MAX_NUM_BATCHED_TOKENS"])),
    ]
    absent = []
    if (utilization := float(manifest["GPU_MEMORY_UTILIZATION"])) != DEFAULT_GPU_MEMORY:
        expect.append(("arg_gpu_memory_utilization_x1000", round(utilization * 1000)))
    if manifest["KV_CACHE_DTYPE"] != "auto":
        expect.append((f"arg_kv_cache_dtype_{manifest['KV_CACHE_DTYPE']}", None))
    if tokens := int(manifest["MTP_SPECULATIVE_TOKENS"]):
        expect.append(("arg_num_speculative_tokens", tokens))
    else:
        absent.append("arg_num_speculative_tokens")
    if gib := int(manifest["KV_OFFLOADING_GIB"]):
        expect += [("arg_kv_offloading_mib", gib * 1024), ("kv_offloading_cpu", None)]
    else:
        absent.append("kv_offloading_cpu")
    return expect, absent


def proof(plan: Plan, choices: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether the engine's engine_choice rows show the plan: each expected code, with its value, and none of the
    absent ones. `missing` and `unexpected` are codes, fixed words."""
    seen = {(row["code"], row.get("value")) for row in choices}
    codes = {row["code"] for row in choices}
    missing = [code if value is None else f"{code}={value}" for code, value in plan.expect
               if ((code, value) not in seen if value is not None else code not in codes)]
    unexpected = [code for code in plan.absent if code in codes]
    return {"ok": not missing and not unexpected, "missing": missing, "unexpected": unexpected,
            "codes": sorted(codes)}


# The changes, each a function of its value that edits the manifest's copy or queues an edit of the engine's command,
# adds what must show, and returns whether the run is timing only.
Apply = Callable[[str, dict[str, str], list[Callable[[Plan], None]], list[tuple[str, int | None]]], bool]


def _integer(value: str, low: int, high: int) -> int:
    if not re.fullmatch(r"\d{1,6}", value) or not low <= int(value) <= high:
        raise ChangeError(f"the value must be a whole number from {low} to {high}")
    return int(value)


def _fraction(value: str, low: float, high: float) -> float:
    if not re.fullmatch(r"0?\.\d{1,4}|1(?:\.0{1,4})?", value) or not low < float(value) <= high:
        raise ChangeError(f"the value must be a number above {low} and at most {high}")
    return float(value)


def _word(value: str, words: Mapping[str, str] | frozenset[str] | tuple[str, ...]) -> str:
    if value not in words:
        raise ChangeError(f"the value must be one of {', '.join(sorted(words))}")
    return value


def _reference(value: str, manifest: dict[str, str], edits: list[Callable[[Plan], None]],
               expect: list[tuple[str, int | None]]) -> bool:
    if value:
        raise ChangeError("reference takes no value")
    return False


def _manifest(key: str, parse: Callable[[str], object]) -> Apply:
    def apply(value: str, manifest: dict[str, str], edits: list[Callable[[Plan], None]],
              expect: list[tuple[str, int | None]]) -> bool:
        parse(value)
        if value == manifest[key]:
            raise ChangeError("the value is the manifest's own: that is the reference")
        manifest[key] = value
        return False
    return apply


def _arguments(*argv: str, env: Mapping[str, str] | None = None) -> Callable[[Plan], None]:
    def edit(plan: Plan) -> None:
        plan.engine = Command([*plan.engine.argv, *argv], {**plan.engine.env, **(env or {})}, plan.engine.cwd)
    return edit


def _speculation(**settings: Any) -> Callable[[Plan], None]:
    """An edit of --speculative-config's JSON, which the manifest's drafter must be on for."""
    def edit(plan: Plan) -> None:
        argv = list(plan.engine.argv)
        at = argv.index("--speculative-config") + 1
        argv[at] = json.dumps(json.loads(argv[at]) | settings)
        plan.engine = Command(argv, plan.engine.env, plan.engine.cwd)
    return edit


def _drafting(manifest: Mapping[str, str]) -> int:
    tokens = int(manifest["MTP_SPECULATIVE_TOKENS"])
    if not tokens:
        raise ChangeError("the change needs the drafter, and MTP_SPECULATIVE_TOKENS is 0")
    return tokens


def _linear(value: str, manifest: dict[str, str], edits: list[Callable[[Plan], None]],
            expect: list[tuple[str, int | None]]) -> bool:
    backend = _word(value, LINEAR_KERNELS)
    edits.append(_arguments("--kernel-config", json.dumps({"linear_backend_per_quant": {"nvfp4_w4a16": backend}})))
    expect += [(f"linear_override_nvfp4_w4a16_{backend}", None), (f"arg_linear_nvfp4_w4a16_{backend}", None),
               (f"nvfp4_gemm_{LINEAR_KERNELS[backend]}", None)]
    return False


def _retention(value: str, manifest: dict[str, str], edits: list[Callable[[Plan], None]],
               expect: list[tuple[str, int | None]]) -> bool:
    _word(value, ("dense",))
    edits.append(_arguments("--prefix-cache-retention-interval", "None"))
    expect.append(("arg_prefix_cache_retention_interval_none", None))
    return False


def _watermark(value: str, manifest: dict[str, str], edits: list[Callable[[Plan], None]],
               expect: list[tuple[str, int | None]]) -> bool:
    fraction = _fraction(value, 0.0, 0.5)
    edits.append(_arguments("--watermark", value))
    expect.append(("arg_watermark_x1000", round(fraction * 1000)))
    return False


def _draft(value: str, manifest: dict[str, str], edits: list[Callable[[Plan], None]],
           expect: list[tuple[str, int | None]]) -> bool:
    _drafting(manifest)
    _word(value, ("probabilistic",))
    edits.append(_speculation(draft_sample_method=value))
    expect.append(("arg_draft_sample_method_probabilistic", None))
    return False


def _verify(value: str, manifest: dict[str, str], edits: list[Callable[[Plan], None]],
            expect: list[tuple[str, int | None]]) -> bool:
    _drafting(manifest)
    _word(value, ("block",))
    edits.append(_speculation(rejection_sample_method=value))
    expect.append(("arg_rejection_sample_method_block", None))
    return False


def _per_batch(value: str, manifest: dict[str, str], edits: list[Callable[[Plan], None]],
               expect: list[tuple[str, int | None]]) -> bool:
    """`START-END:K,...`, vLLM's num_speculative_tokens_per_batch_size (vllm/config/speculative.py:479)."""
    drafted = _drafting(manifest)
    ranges = []
    for part in value.split(","):
        found = re.fullmatch(r"(\d{1,3})-(\d{1,3}):(\d)", part)
        if found is None or not 1 <= int(found[1]) <= int(found[2]) or not 1 <= int(found[3]) <= drafted:
            raise ChangeError("per-batch takes START-END:K,... with K from 1 to the manifest's count")
        ranges.append([int(found[1]), int(found[2]), int(found[3])])
    edits.append(_speculation(num_speculative_tokens_per_batch_size=ranges))
    expect.append(("arg_num_speculative_tokens_per_batch_size", None))
    return False


def _synthetic(value: str, manifest: dict[str, str], edits: list[Callable[[Plan], None]],
               expect: list[tuple[str, int | None]]) -> bool:
    """Rates by position, unconditional and not increasing, one for each drafted token (vllm/config/speculative.py:
    517-536): the engine drafts as many tokens as there are rates and accepts them at those rates whatever they are,
    for the timing sweep only. The count is part of the change, since the sweep times each count at its own rates."""
    _drafting(manifest)
    try:
        rates = [float(part) for part in value.split(",")]
    except ValueError:
        raise ChangeError("synthetic takes the rates by position, separated by commas") from None
    if not 1 <= len(rates) <= MAX_DRAFTS or any(not 0 <= rate <= 1 for rate in rates) \
            or rates != sorted(rates, reverse=True):
        raise ChangeError(f"synthetic takes 1 to {MAX_DRAFTS} rates from 0 to 1, none above the one before")
    manifest["MTP_SPECULATIVE_TOKENS"] = str(len(rates))
    edits.append(_speculation(rejection_sample_method="synthetic", synthetic_acceptance_rates=rates))
    expect += [("arg_rejection_sample_method_synthetic", None), ("arg_synthetic_acceptance_rates", None)]
    return True


def _attention(sliding: bool) -> Apply:
    def apply(value: str, manifest: dict[str, str], edits: list[Callable[[Plan], None]],
              expect: list[tuple[str, int | None]]) -> bool:
        _word(value, ("flashinfer",))
        if sliding:
            edits.append(_arguments("--attention-config",
                                    json.dumps({"backend_per_kind": {"sliding_window": "FLASHINFER"}}),
                                    env=FLASHINFER_ENV))
            expect.append(("arg_attention_sliding_window_flashinfer", None))
        else:
            edits.append(_arguments("--attention-backend", "FLASHINFER", env=FLASHINFER_ENV))
            expect.append(("arg_attention_backend_flashinfer", None))
        expect.append(("attention_selected_flashinfer", None))
        return False
    return apply


CHANGES: dict[str, Apply] = {
    "reference": _reference,
    "mtp": _manifest("MTP_SPECULATIVE_TOKENS", lambda value: _integer(value, 0, MAX_DRAFTS)),  # item 3
    "kv-offloading": _manifest("KV_OFFLOADING_GIB", lambda value: _integer(value, 0, 512)),  # item 2
    "gpu-memory": _manifest("GPU_MEMORY_UTILIZATION", lambda value: _fraction(value, 0.5, 0.99)),  # item 4
    "batched-tokens": _manifest("MAX_NUM_BATCHED_TOKENS", lambda value: _integer(value, 64, 65536)),  # item 5
    "linear": _linear,  # items 6 and 7
    "retention": _retention,  # item 8
    "draft": _draft,  # item 9
    "verify": _verify,  # item 9
    "per-batch": _per_batch,  # item 10
    "watermark": _watermark,  # item 11
    "kv-cache": _manifest("KV_CACHE_DTYPE", lambda value: _word(value, card.CACHE_DTYPES)),  # item 12
    "attention-sliding": _attention(sliding=True),  # item 13
    "attention": _attention(sliding=False),  # item 13
    "synthetic": _synthetic,  # the timing sweep of item 3
}


def take_over(base: Card, pause: Callable[[float], object] = time.sleep) -> int:
    """End the launcher's pair as `card --stop` does, and check that the card's ports are free for the harness's own:
    0, card.NOT_PREPARED, STILL_RUNNING when the launcher has not ended within card.STOP_WAIT_S, or card.PORT_TAKEN.
    The instance keeps running whatever the answer."""
    if not base.prepared():
        return card.NOT_PREPARED
    if card.stop_pair(base.state, pause):
        return STILL_RUNNING
    if any(card.taken(port) for port in base.ports.values()):
        return card.PORT_TAKEN
    return 0


def harness_config(base: Card, plan: Plan, key_sha256: str) -> dict[str, Any]:
    """The gateway's configuration for this run: the card's own, the rental's limits, and the harness's key beside
    the card's two. A timing-only run leaves the bot's client key out, so that no reader is ever served by it."""
    config = replace(base, manifest=plan.manifest).config()
    limits = copy.deepcopy(config["limits"])
    for name, values in RENTAL_LIMITS.items():
        limits[name] = {**limits[name], **values}
    keys = [key for key in config["keys"] if not (plan.timing_only and key["label"] == "client")]
    keys.append({"sha256": key_sha256, "label": HARNESS_LABEL, "classes": ["internal"], "default": "internal",
                 "scopes": True, "control": False})
    return config | {"limits": limits, "keys": keys, "idle_timeout_s": IDLE_S,
                     "versions": config["versions"] | {"harness_change": plan.change.slug}}


def harness_gateway(base: Card, config_path: Path) -> Command:
    """The card's gateway command without the instance's credential: whatever happens, it cannot stop the instance."""
    command = base.gateway()
    env = {name: value for name, value in command.env.items() if name not in ("CONTAINER_API_KEY", "CONTAINER_ID")}
    return Command(command.argv, env | {"SIMPLE_SERVING_CONFIG": str(config_path)}, command.cwd)


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass
class Started:
    """A pair the harness started: how its start went, and what the engine said of itself."""

    pair: card.Pair
    outcome: str  # "ready", or how the pair ended before it was ready
    load_s: float


@asynccontextmanager
async def started(base: Card, plan: Plan, run_dir: Path, key: str) -> AsyncIterator[Started]:
    """Start the plan's pair in `run_dir` and yield once the gateway is ready or the pair has ended. On leaving, the
    pair ends, the gateway first, as card.Pair ends it on a signal."""
    config_path = run_dir / "gateway.json"
    descriptor = card.private(config_path, os.O_WRONLY | os.O_TRUNC)
    with os.fdopen(descriptor, "w") as file:
        json.dump(harness_config(base, plan, key_digest(key)), file)
    (run_dir / "logs").mkdir(exist_ok=True)
    pair, ended = card.Pair(run_dir), asyncio.Event()
    loop = asyncio.get_running_loop()
    began = loop.time()
    task = asyncio.create_task(pair.run(plan.engine, harness_gateway(base, config_path),
                                        float(plan.manifest["LOAD_DEADLINE_S"]), ended))
    ready = asyncio.create_task(pair.ready.wait())
    try:
        await asyncio.wait({task, ready}, return_when=asyncio.FIRST_COMPLETED)
        outcome = "ready" if pair.ready.is_set() and not task.done() else (
            task.result() if task.done() and task.exception() is None else "spawn_failed")
        yield Started(pair, outcome, round(loop.time() - began, 1))
    finally:
        ready.cancel()
        ended.set()
        with contextlib.suppress(OSError):  # a process that could not start: nothing runs
            await task

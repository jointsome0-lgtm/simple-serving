"""The card's launcher and scripts (contract section 8, "Starting the card"): one pair with no restarts, how it ends,
what it keeps of the output, and a bootstrap and an onstart that change nothing when they run again. Stand-ins take
the place of vLLM, the gateway, pip, aria2c, curl and flock; the clock and Vast are fakes, and nothing reaches the
network."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from simple_serving import card, preflight
from simple_serving.card import Card
from simple_serving.config import load as load_config
from simple_serving.pictures.card import PictureCard
from simple_serving.vast import RETRY_S, VastError

from .support import FakeClock, FakeVast, until

ENGINE, GATEWAY = "vllm/bin/vllm", "gateway/bin/python"
SECRET = "synthetic-secret-7f3a9c"  # stands in for a prompt, which no log may hold
CREDENTIAL = {"CONTAINER_ID": "1234", "CONTAINER_API_KEY": "synthetic-container-key"}
CLIENT, CONTROL, METRICS = (hashlib.sha256(key).hexdigest() for key in (b"synthetic-client-key",
                                                                          b"synthetic-control-key",
                                                                          b"synthetic-metrics-key"))
READY = json.dumps({"time": "2026-09-24T12:00:00.000Z", "event": "engine", "service_status": "ready",
                    "context_tokens": 65536})
SLEEP = json.dumps({"time": "2026-09-24T12:13:00.000Z", "event": "sleep", "reason": "idle"})
STAND_IN = """#!{python}
import json, os, sys, time
with open(sys.argv[0] + ".plan") as file:
    plan = json.load(file)
with open(sys.argv[0] + ".ran", "w") as file:
    json.dump({{"argv": sys.argv[1:], "env": sorted(os.environ)}}, file)
for line in plan["lines"]:
    print(line, flush=True)
if plan["exit"] is None:
    time.sleep(600)
sys.exit(plan["exit"])
"""


def stand_in(path: Path, *lines: str, exit_code: int | None = None) -> None:
    """A stand-in for vLLM or the gateway's Python: it notes its arguments and the names in its environment, prints
    `lines`, then exits with `exit_code` or waits to be ended."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STAND_IN.format(python=sys.executable))
    path.chmod(0o700)
    Path(f"{path}.plan").write_text(json.dumps({"lines": lines, "exit": exit_code}))


def ran(path: Path) -> dict[str, Any]:
    return json.loads(Path(f"{path}.ran").read_text())


def rows(logged: io.StringIO | str, event: str) -> list[dict[str, Any]]:
    text = logged if isinstance(logged, str) else logged.getvalue()
    found = (json.loads(line) for line in text.splitlines())
    return [{name: value for name, value in row.items() if name != "time"}
            for row in found if row.get("event") == event]


def exits(logged: io.StringIO | str) -> dict[str, int]:
    return {row["process"]: row["exit_code"] for row in rows(logged, "exit")}


@pytest.fixture(autouse=True)
def no_vast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever path a test takes, it never reaches Vast: every stop that is not a fake fails at once."""
    async def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a test tried to reach Vast")

    monkeypatch.setattr("simple_serving.vast.stop", refuse)


@pytest.fixture
def state(tmp_path: Path) -> Path:
    (tmp_path / "state/logs").mkdir(parents=True)
    return tmp_path / "state"


def card_at(state: Path, **manifest: str) -> Card:
    loaded = Card.load(state=state, environ=CREDENTIAL, root=state)
    return replace(loaded, manifest={**loaded.manifest, **manifest})


def eventually(condition: Callable[[], bool]) -> None:
    """Wait for another process to get somewhere, which takes it milliseconds."""
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, "the condition did not come true in time"
        time.sleep(0.01)


@pytest.mark.anyio
async def test_a_signal_ends_the_pair_and_leaves_the_instance_running(state: Path, logged: io.StringIO,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in CREDENTIAL.items():  # as Vast gives them to the container, and so to onstart
        monkeypatch.setenv(name, value)
    stand_in(state / ENGINE)
    stand_in(state / GATEWAY, READY)
    ended, vast = asyncio.Event(), FakeVast()
    launch = asyncio.create_task(card.launch(card_at(state), ended, vast))
    await until(lambda: bool(rows(logged, "ready")))
    ended.set()
    assert await launch == "signal"
    assert (vast.attempts, (state / "given-up").exists()) == ([], False)
    assert exits(logged) == {"gateway": -signal.SIGTERM, "engine": -signal.SIGTERM}
    assert (state / "logs/gateway.jsonl").read_text() == READY + "\n"
    engine, gateway = ran(state / ENGINE), ran(state / GATEWAY)
    argv = engine["argv"]
    assert argv[:2] == ["serve", str(state / "models" / card_at(state).manifest["MODEL_REVISION"])]
    assert (argv[argv.index("--host") + 1], argv[argv.index("--port") + 1]) == ("127.0.0.1", "8092")
    assert argv[argv.index("--scheduling-policy") + 1] == "priority"
    assert "--enable-prefix-caching" in argv and argv[argv.index("--prefix-cache-retention-interval") + 1] == "None"
    assert argv[argv.index("--reasoning-parser") + 1] == "gemma4" and "--language-model-only" in argv
    assert "CONTAINER_API_KEY" not in engine["env"] and "VLLM_NO_USAGE_STATS" in engine["env"]
    assert gateway["argv"] == ["-m", "simple_serving"]
    assert {"CONTAINER_ID", "CONTAINER_API_KEY", "SIMPLE_SERVING_CONFIG"} <= set(gateway["env"])


def test_mtp_thinking_budget_and_vision_reach_vllm_and_the_pins_only_when_on(state: Path) -> None:
    off, on = card_at(state, MTP_SPECULATIVE_TOKENS="0"), card_at(state)
    assert on.manifest["MTP_SPECULATIVE_TOKENS"] == "3"  # on by default since the owner's choice of 2026-09-26
    assert "--speculative-config" not in off.engine().argv
    drafter = state / "drafters" / on.manifest["DRAFTER_REVISION"]
    assert on.engine().argv == [*off.engine().argv, "--speculative-config",
                                f'{{"method": "mtp", "model": "{drafter}", "num_speculative_tokens": 3}}']
    assert card.pinned_versions(on.manifest) == card.pinned_versions(off.manifest) | {
        "drafter_revision": on.manifest["DRAFTER_REVISION"], "mtp_speculative_tokens": "3"}
    assert on.manifest["THINKING_BUDGET"] == "0"
    assert "--reasoning-config" not in on.engine().argv and "thinking_budget" not in card.pinned_versions(on.manifest)
    budget = card_at(state, THINKING_BUDGET="1")
    assert budget.engine().argv == [*on.engine().argv, "--reasoning-config",
                                   '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}']
    assert card.pinned_versions(budget.manifest) == card.pinned_versions(on.manifest) | {"thinking_budget": "on"}
    argv = budget.engine().argv
    assert argv[argv.index("--prefix-cache-retention-interval") + 1] == "None"
    assert on.manifest["VISION"] == "0" and "vision" not in card.pinned_versions(on.manifest)
    vision = card_at(state, THINKING_BUDGET="1", VISION="1")
    args = ["--limit-mm-per-prompt", '{"image": 1, "video": 0, "audio": 0}', "--mm-processor-cache-gb", "0"]
    at = argv.index("--language-model-only")
    assert vision.engine().argv == [*argv[:at], *args, *argv[at + 1:]]
    assert card.pinned_versions(vision.manifest) == card.pinned_versions(budget.manifest) | {"vision": "1"}


def test_the_kv_cache_copy_in_ram_is_off_by_default_and_reaches_vllm_and_the_pins_only_when_on(state: Path) -> None:
    off, on = card_at(state), card_at(state, KV_OFFLOADING_GIB="12")
    assert off.manifest["KV_OFFLOADING_GIB"] == "0"  # pinned RAM a card may lack, until a rental measures it
    assert "--kv-offloading-size" not in off.engine().argv
    assert on.engine().argv == [*off.engine().argv, "--kv-offloading-size", "12"]
    assert card.pinned_versions(on.manifest) == card.pinned_versions(off.manifest) | {"kv_offloading_gib": "12"}


FAILED_LOAD = ("Traceback (most recent call last):", f'  File "{SECRET}.py", line 1, in load',
               "x" * 3 * card.LINE_BYTES + SECRET, f"torch.OutOfMemoryError: CUDA out of memory. {SECRET}")
WAITS = ((), None)  # a stand-in that prints nothing and waits to be ended


@pytest.mark.anyio
@pytest.mark.parametrize(("engine", "gateway", "outcome", "exit_codes"), [
    ((FAILED_LOAD, 1), WAITS, "engine_exit", {"gateway": -signal.SIGTERM, "engine": 1}),
    (WAITS, WAITS, "load_deadline", {"gateway": -signal.SIGTERM, "engine": -signal.SIGTERM}),
    (WAITS, None, "spawn_failed", {"engine": -signal.SIGTERM}),  # a gateway that cannot start
    (WAITS, ((READY, SLEEP), 0), "asleep", {"gateway": 0, "engine": -signal.SIGTERM}),  # the gateway's own stop began
    (WAITS, ((READY,), 1), "gateway_exit", {"gateway": 1, "engine": -signal.SIGTERM}),
])
async def test_a_pair_that_ends_stops_the_instance_and_gives_up_only_before_ready(
        state: Path, logged: io.StringIO, monkeypatch: pytest.MonkeyPatch, engine: tuple[tuple[str, ...], int | None],
        gateway: tuple[tuple[str, ...], int | None] | None, outcome: str, exit_codes: dict[str, int]) -> None:
    clock = FakeClock(monkeypatch)
    stand_in(state / ENGINE, *engine[0], exit_code=engine[1])
    if gateway is not None:
        stand_in(state / GATEWAY, *gateway[0], exit_code=gateway[1])
    vast = FakeVast()
    launch = asyncio.create_task(card.launch(card_at(state, LOAD_DEADLINE_S="900"), asyncio.Event(), vast))
    if outcome == "load_deadline":
        await until(lambda: Path(f"{state / GATEWAY}.ran").exists())
        await clock.advance(899)
        assert not launch.done()
        await clock.advance(1)
    assert await launch == outcome
    if outcome in ("asleep", "gateway_exit"):  # an end after ready gives nothing up
        assert not (state / "given-up").exists()
    else:
        assert (state / "given-up").read_text() == f"{outcome}\n"
    assert vast.stopped and exits(logged) == exit_codes
    failures = [row["code"] for row in rows(logged, "engine_failure")]
    assert failures == (["traceback", "out_of_memory"] if outcome == "engine_exit" else [])
    assert SECRET not in logged.getvalue()


@pytest.mark.anyio
async def test_a_card_that_gave_up_loads_nothing_at_a_resume_and_stops_again(state: Path, logged: io.StringIO,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    stand_in(state / ENGINE, exit_code=1)
    stand_in(state / GATEWAY)
    failed = FakeVast()
    assert await card.launch(card_at(state), asyncio.Event(), failed) == "engine_exit"
    assert failed.stopped
    resumed = FakeVast(VastError("forbidden", 403))  # the owner resumes the instance, and onstart starts the launcher
    launch = asyncio.create_task(card.launch(card_at(state), asyncio.Event(), resumed))
    await until(lambda: bool(rows(logged, "wait_for_retry")))
    await clock.advance(card.IDLE_TIMEOUT_S - 1)
    assert not launch.done() and resumed.attempts == []
    await clock.advance(1)
    await until(lambda: bool(rows(logged, "stop_unconfirmed")))  # Vast refuses the first stop
    assert (state / "stop-unconfirmed").read_text() == "forbidden\n"
    await clock.advance(RETRY_S)
    assert await launch == "given_up"
    assert resumed.stopped and (state / "given-up").read_text() == "engine_exit\n"
    assert rows(logged, "stop_unconfirmed") == [{"event": "stop_unconfirmed", "code": "forbidden", "status": 403}]
    assert [row["reason"] for row in rows(logged, "pair_end")] == ["engine_exit", "given_up"]
    assert len(rows(logged, "exit")) == 2  # the processes of the failed load, and none since


@pytest.mark.anyio
async def test_a_retry_within_the_interval_runs_the_pair_and_stops_nothing(state: Path, logged: io.StringIO,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    stand_in(state / ENGINE)
    stand_in(state / GATEWAY, READY)
    (state / "given-up").write_text("load_deadline\n")
    ended, vast = asyncio.Event(), FakeVast()
    launch = asyncio.create_task(card.launch(card_at(state), ended, vast))
    await until(lambda: bool(rows(logged, "wait_for_retry")))
    await clock.advance(card.IDLE_TIMEOUT_S - 60)
    (state / "given-up").unlink()  # what --retry does, and its start then finds this launcher running
    await clock.advance(card.POLL_S)
    await until(lambda: bool(rows(logged, "ready")))
    await clock.advance(60)  # past the end of the interval, which no longer counts
    ended.set()
    assert await launch == "signal"
    assert vast.attempts == []


def test_the_output_leaves_a_few_numbers_and_categories_of_vllm_and_the_gateways_own_rows(state: Path,
                                                                                          logged: io.StringIO) -> None:
    pair = card.Pair(state)
    for text in (
        f"Model loading took {'9' * 1000} GiB memory and {'9' * 1000} seconds",
        f"GPU KV cache size: {'9' * 1000} tokens",
        "GPU KV cache size: 1,,2 tokens",
        "Available KV cache memory: NaN GiB",
        "INFO 09-24 12:00:00 [gpu_model_runner.py:2007] Model loading took 23.5000 GiB memory and 41.250000 seconds",
        "INFO 09-24 12:00:10 [gpu_worker.py:298] Available KV cache memory: 4.25 GiB",
        ("(EngineCore pid=7) INFO 09-24 12:00:10 [kv_cache_utils.py:2395] GPU KV cache size: 123,456 tokens, Maximum "
         "concurrency for 65,536 tokens per request: 3.45x"),
        "INFO 09-24 12:00:10 [gpu_worker.py:298] Available KV cache memory: -1.25 GiB",
        f"INFO 09-24 12:01:00 [logger.py:43] Received request chatcmpl-1: prompt: '{SECRET}'",
        f"ERROR 09-24 12:02:00 [core.py:710] RuntimeError: CUDA error: an illegal memory access, {SECRET}",
        f"ERROR 09-24 12:02:00 [core.py:710] RuntimeError: CUDA error: {SECRET}",
        ("ValueError: To serve at least one request with the models's max seq len (65536), (8.00 GiB KV cache is "
         "needed, which is larger than the available KV cache memory (4.25 GiB)."),
        ("ValueError: Free memory on device cuda:0 (28.1/31.84 GiB) on startup is less than desired GPU memory "
         "utilization (0.92, 29.3 GiB)."),
        "AssertionError: Error in memory profiling. Initial free memory 29.1 GiB, current free memory 29.6 GiB.",
        "RuntimeError: Frontend process failed during engine core initialization. See root cause above.",
        ("(EngineCore pid=7) WARNING 09-24 12:00:05 [modelopt.py:1981] In NVFP4 linear, the global weight scale "
         "differs across parallel layers (e.g. q_proj, k_proj, v_proj). This will likely reduce accuracy."),
    ):
        pair.engine_line(text.encode() + b"\n")
    assert rows(logged, "engine_measure") == [
        {"event": "engine_measure", "weights_mib": 24064, "weights_load_ms": 41250},
        {"event": "engine_measure", "kv_cache_mib": 4352},
        {"event": "engine_measure", "kv_cache_tokens": 123456, "concurrency_x100": 345},
        {"event": "engine_measure", "kv_cache_mib": -1280},
    ]
    assert [row["code"] for row in rows(logged, "engine_failure")] == [
        "cuda_error", "kv_cache_too_small", "memory_taken", "memory_profiling_failed", "engine_dead"]
    assert rows(logged, "engine_warning") == [{"event": "engine_warning", "code": "fused_scales_differ"}]
    refused = json.dumps({"event": "stop_failed", "code": "forbidden", "status": 403})  # the gateway's own stop
    lines = [
        READY.encode(),
        refused.encode(),
        f"Traceback (most recent call last): {SECRET}".encode(),
        json.dumps({"event": "request", "prompt": SECRET}).encode(),  # a field outside log.FIELDS
        json.dumps({"event": "request", "code": [SECRET]}).encode(),  # a value that is not a scalar
        b"[" * 5000,
        b'{"event": "\xff"}',
        b"",  # a blank line
    ]
    for line in lines:
        pair.gateway_line(line + b"\n")
    assert (state / "logs/gateway.jsonl").read_text() == f"{READY}\n{refused}\n"
    assert len(rows(logged, "gateway_output")) == len(lines) - 2
    assert pair.ready.is_set() and not pair.asleep and card.marked(state) == card.STOP_UNCONFIRMED
    assert SECRET not in logged.getvalue()


def test_engine_choices_emit_fixed_codes_and_bounded_numbers_without_quoted_config_values(
        state: Path, logged: io.StringIO) -> None:
    pair = card.Pair(state)
    model = f"/workspace/{SECRET}/model"
    arguments = {
        "model": model, "tokenizer": model, "served_model_name": [SECRET], "max_model_len": 65536,
        "max_num_seqs": 8, "max_num_batched_tokens": 512, "gpu_memory_utilization": 0.94,
        "kv_cache_dtype": "int4_per_token_head", "prefix_cache_retention_interval": None, "watermark": 0.02,
        "kv_offloading_size": 12.0, "attention_backend": "FLASHINFER",
        # These settings belong only inside speculative_config; valid neighbors must still be reported.
        "num_speculative_tokens": 777, "rejection_sample_method": "standard",
        "num_speculative_tokens_per_batch_size": [(1, 8, 4)],
        "speculative_config": {"model": model, "method": "mtp", "num_speculative_tokens": 4,
                               "draft_sample_method": "probabilistic", "rejection_sample_method": "block",
                               "num_speculative_tokens_per_batch_size": [(1, 4, 4), (5, 8, 2)],
                               # These settings belong only at the top level.
                               "max_model_len": 777, "kv_cache_dtype": "fp8", "attention_backend": "TRITON_ATTN"},
        "kernel_config": {"linear_backend_per_quant": {"nvfp4_w4a16": "humming"}},
        "attention_config": {"backend_per_kind": {"sliding_window": "FLASHINFER"}},
    }
    unknown = {
        "kv_cache_dtype": SECRET + " / arbitrary 'quoted' value",
        "attention_backend": SECRET,
        "model": "'max_num_seqs': 777, 'synthetic_acceptance_rates': [1] " + SECRET,
        "ignored": {"max_num_seqs": 777, "kv_cache_dtype": "fp8"},
    }
    calls = ("{'kernel_config': KernelConfig(linear_backend_per_quant={'nvfp4_w4a16': 'humming'}, "
             f"note={SECRET!r}), 'attention_config': AttentionConfig(backend=None, "
             "backend_per_kind={'sliding_window': 'FLASHINFER'})}")
    # Pinned vLLM log formats; requested arguments and observed selections remain distinct codes.
    for source, message in (
        ("__init__.py:1180", "Using MarlinNvFp4LinearKernel for NVFP4 GEMM"),
        ("__init__.py:1136", f"Using {SECRET.replace('-', '_')} for NVFP4 GEMM"),
        ("__init__.py:245", "Applied linear backend override for 'nvfp4_w4a16': 'humming'"),
        ("cuda.py:478", "Using AttentionBackendEnum.TRITON_ATTN backend."),
        ("cuda.py:538", "Using FLASHINFER attention backend out of potential backends: ['TRITON_ATTN', 'FLASHINFER']."),
        ("flashinfer.py:838", ("FlashInfer XQA decode does not support head_dim=512; reverting this KV-cache "
                              "group to native FlashInfer decode.")),
        ("flashinfer.py:909", ("FlashInfer resolved query dtypes: prefill=torch.bfloat16, decode=torch.bfloat16, "
                              "decode_backend=xqa, kv_cache_dtype=fp8_e4m3, arch=sm120")),
        ("gpu_worker.py:441", "Using V2 Model Runner"),
        ("vllm.py:717", f"Model Runner V2 does not yet support {SECRET}; using the V1 model runner instead."),
        ("model_runner.py:1067", "Graph capturing finished in 12 secs, took 0.45 GiB"),
        ("factory.py:57", "Creating offloading spec with name: CPUOffloadingSpec"),
        ("api_utils.py:286", f"non-default args: {arguments}"),
        ("api_utils.py:286", f"non-default args: {unknown}"),
        ("api_utils.py:286", f"non-default args: {calls}"),
    ):
        pair.engine_line(f"(EngineCore pid=7) INFO 09-27 12:00:00 [{source}] {message}\n".encode())
    expected: list[dict[str, Any]] = [
        {"code": "nvfp4_gemm_marlin"}, {"code": "nvfp4_gemm_other"}, {"code": "linear_override_nvfp4_w4a16_humming"},
        {"code": "attention_selected_triton_attn"}, {"code": "attention_chosen_flashinfer"},
        {"code": "flashinfer_xqa_reverted", "head_dim": 512}, {"code": "flashinfer_decode_xqa"},
        {"code": "model_runner_v2"}, {"code": "model_runner_v1"},
        {"code": "graphs_captured", "capture_ms": 12000, "graph_mib": 461}, {"code": "kv_offloading_cpu"},
        {"code": "arg_max_model_len", "value": 65536}, {"code": "arg_max_num_seqs", "value": 8},
        {"code": "arg_max_num_batched_tokens", "value": 512},
        {"code": "arg_gpu_memory_utilization_x1000", "value": 940}, {"code": "arg_watermark_x1000", "value": 20},
        {"code": "arg_kv_offloading_mib", "value": 12288}, {"code": "arg_prefix_cache_retention_interval_none"},
        {"code": "arg_num_speculative_tokens", "value": 4}, {"code": "arg_kv_cache_dtype_int4_per_token_head"},
        {"code": "arg_attention_backend_flashinfer"}, {"code": "arg_draft_sample_method_probabilistic"},
        {"code": "arg_rejection_sample_method_block"}, {"code": "arg_num_speculative_tokens_per_batch_size"},
        {"code": "arg_linear_nvfp4_w4a16_humming"}, {"code": "arg_attention_sliding_window_flashinfer"},
        {"code": "arg_kv_cache_dtype_other"}, {"code": "arg_attention_backend_other"},
        {"code": "arg_linear_nvfp4_w4a16_humming"}, {"code": "arg_attention_sliding_window_flashinfer"},
    ]
    assert sorted(rows(logged, "engine_choice"), key=repr) == sorted(
        [{"event": "engine_choice", **row} for row in expected], key=repr)
    pair.engine_line("non-default args: {'attention_backend': 'FLA\u017fHINFER'}\n".encode())
    choices = rows(logged, "engine_choice")
    assert len(choices) == len(expected) + 1
    assert choices[-1]["code"] in {"arg_attention_backend_flashinfer", "arg_attention_backend_other"}

    # Quoted key-looking text is not configuration. Unsupported reprs and extreme numbers leave no extra rows.
    before = logged.getvalue()
    for value in ("nan", "inf", "True", "'12'", "1.2.3", "9" * 5000, "1e308"):
        pair.engine_line(f"non-default args: {{'max_num_seqs': {value}, 'kv_offloading_size': {value}}}\n".encode())
    for message in (
        f"FlashInfer XQA decode does not support head_dim={'9' * 5000}",
        f"Graph capturing finished in {'9' * 5000} secs, took 1 GiB",
        f"Graph capturing finished in 1 secs, took {'9' * 1000} GiB",
        "Graph capturing finished in 1.2.3 secs, took NaN GiB",
        "non-default args: " + repr({"note": "'max_num_seqs': 777"}),
        ("non-default args: {'attention_config': AttentionConfig(backend=None, backend_per_kind="
         "{'sliding_window': <AttentionBackendEnum.FLASHINFER: 'private-backend-path'>}), 'max_num_seqs': 777}"),
    ):
        pair.engine_line((message + "\n").encode())
    assert logged.getvalue() == before
    assert not rows(logged, "engine_measure") and not rows(logged, "engine_failure")
    assert all(type(value) is int for row in rows(logged, "engine_choice")
               for name, value in row.items() if name not in {"event", "code"})
    assert all(re.fullmatch(r"[a-z0-9_]+", row["code"]) for row in rows(logged, "engine_choice"))
    assert all(value not in logged.getvalue() for value in (SECRET, SECRET.replace("-", "_"), model,
                                                           "private-backend-path", "arbitrary", "777"))


@pytest.mark.anyio
async def test_of_a_line_over_line_bytes_only_the_head_is_read() -> None:
    output = asyncio.StreamReader(limit=card.LINE_BYTES)
    output.feed_data(b"x" * 3 * card.LINE_BYTES + b"CUDA error\nshort\nlast")
    output.feed_eof()
    lines: list[bytes] = []
    await card.read(cast(Any, SimpleNamespace(stdout=output)), lines.append)  # a process, as far as read() needs one
    assert lines == [b"x" * card.LINE_BYTES, b"short\n", b"last"]


def test_a_log_file_moves_aside_before_it_grows_past_its_size(tmp_path: Path) -> None:
    log_file = card.JsonLines(tmp_path / "card.jsonl", 100)
    for index in range(5):
        log_file.write(json.dumps({"event": "synthetic", "index": index, "padding": "x" * 20}) + "\n")
    current, older = tmp_path / "card.jsonl", tmp_path / "card.jsonl.1"
    assert current.stat().st_size <= 100 and older.stat().st_size <= 100
    assert json.loads(current.read_text().splitlines()[-1])["index"] == 4
    assert {current.stat().st_mode & 0o777, older.stat().st_mode & 0o777} == {0o600}


def test_the_launcher_runs_one_pair_until_sigterm(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL}))
    stand_in(state / ENGINE, "INFO 09-24 12:00:10 [kv_cache_utils.py:1087] GPU KV cache size: 123,456 tokens")
    stand_in(state / GATEWAY, READY)
    for older in (state / "gateway.json", state / "logs/card.jsonl"):  # left wide, as by another hand
        older.parent.mkdir(exist_ok=True)
        older.write_text('{"event": "synthetic"}\n')
        older.chmod(0o644)
    (state / "stop-unconfirmed").write_text("forbidden\n")  # of the start before
    environ = {name: value for name, value in os.environ.items() if not name.startswith("CONTAINER_")}
    environ |= {"SIMPLE_SERVING_CARD_DIR": str(state), "SIMPLE_SERVING_CARD_ROOT": str(tmp_path)}
    command = [sys.executable, "-m", "simple_serving.card", "--run"]
    launcher = subprocess.Popen(command, cwd=card.CODE, env=environ)
    card_log = state / "logs/card.jsonl"
    try:
        eventually(lambda: card_log.exists() and bool(rows(card_log.read_text(), "ready")))
        assert card.launcher(state) == launcher.pid
        assert subprocess.run(command, cwd=card.CODE, env=environ, timeout=30, check=False).returncode == 0  # the lock
    finally:
        launcher.terminate()  # a launcher that is killed leaves its pair running
        assert launcher.wait(timeout=30) == 0
    assert card.launcher(state) is None
    assert not (state / "given-up").exists() and not (state / "stop-unconfirmed").exists()
    text = card_log.read_text()
    assert rows(text, "engine_measure") == [{"event": "engine_measure", "kv_cache_tokens": 123456}]
    assert exits(text) == {"gateway": -signal.SIGTERM, "engine": -signal.SIGTERM}
    assert rows(text, "pair_end") == [{"event": "pair_end", "reason": "signal"}]
    config = json.loads((state / "gateway.json").read_text())
    assert [(key["label"], key["sha256"]) for key in config["keys"]] == [("client", CLIENT), ("control", CONTROL)]
    assert (config["engine_url"], config["listen"]["control"]) == ("http://127.0.0.1:8092",
                                                                   {"host": "127.0.0.1", "port": 8091})
    assert {path.stat().st_mode & 0o777 for path in (state / "gateway.json", card_log)} == {0o600}


def test_a_second_launcher_leaves_the_pair_to_the_first(state: Path) -> None:
    with open(state / "launcher.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert card.run(card_at(state)) == 0
    assert not (state / "launcher.pid").exists()


def prepared(tmp_path: Path) -> Card:
    """A card that bootstrap.sh has prepared, on free loopback ports."""
    code, state = tmp_path / "code", tmp_path / "state"
    (code / "card").mkdir(parents=True)
    state.mkdir()
    ports = []
    for name in ("PUBLIC", "CONTROL", "ENGINE"):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            ports.append(f"{name}_PORT={sock.getsockname()[1]}\n")
    (code / "card/manifest.env").write_text((card.CODE / "card/manifest.env").read_text() + "".join(ports))
    for name in ("gateway", "vllm"):
        (code / f"card/{name}-requirements.txt").write_text(f"{name}==0.0.1 \\\n    --hash=sha256:{'0' * 64}\n")
    stamp = hashlib.sha256(b"".join((code / "card" / name).read_bytes() for name in card.STAMPED))
    (state / "prepared").write_text(stamp.hexdigest() + "\n")
    (state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL}))
    return Card.load(code=code, state=state, environ=CREDENTIAL, root=state)


def test_start_says_what_stands_in_the_way(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    ready = prepared(tmp_path)
    assert card.start(ready, dry_run=True) == 0
    engine, gateway = capsys.readouterr().out.splitlines()
    assert engine.startswith(f"{ready.state}/vllm/bin/vllm serve {ready.state}/models/")
    assert gateway == f"{ready.state}/gateway/bin/python -m simple_serving"
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", ready.ports["control"]))
        busy.listen()
        assert card.start(ready, dry_run=True) == card.PORT_TAKEN
    assert card.start(replace(ready, credential={}), dry_run=True) == card.NOT_PREPARED
    manifest = ready.code / "card/manifest.env"
    manifest.write_text(manifest.read_text() + "MAX_NUM_SEQS=4\n")  # a change that bootstrap.sh has not prepared
    assert card.start(ready, dry_run=True) == card.NOT_PREPARED
    (ready.state / "given-up").write_text("load_deadline\n")
    assert card.start(ready, dry_run=True) == card.GAVE_UP
    spawned: list[list[str]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **options: spawned.append(argv))
    assert card.start(ready) == card.GAVE_UP
    assert spawned == [[sys.executable, "-m", "simple_serving.card", "--run"]]  # the launcher that will stop it


def launcher_stand_in(state: Path) -> subprocess.Popen[bytes]:
    """A process that passes for a running launcher."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "simple_serving.card", "--run"])
    (state / "launcher.pid").write_text(f"{process.pid}\n")
    return process


def test_hold_lasts_as_long_as_the_launcher(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ready = prepared(tmp_path)
    state = ready.state
    pauses: list[float] = []
    assert card.hold(ready, pauses.append) == 0  # none started: it waits a while for onstart
    assert sum(pauses) == card.HOLD_START_S
    assert capsys.readouterr().out == f"{card.HOLDING}\n"  # the command's sign that the tunnel's forwards are up
    assert card.hold(replace(ready, credential={}), pauses.append) == card.NOT_PREPARED  # and none will start
    assert capsys.readouterr().out == ""

    process, pauses = launcher_stand_in(state), []

    def end_on_the_third(seconds: float) -> None:
        pauses.append(seconds)
        if len(pauses) == 3:
            process.kill()
            process.wait()

    assert card.hold(ready, end_on_the_third) == 0
    assert len(pauses) == 3

    process = launcher_stand_in(state)
    try:
        assert card.hold(ready, lambda _: (state / "given-up").write_text("engine_exit\n")) == card.GAVE_UP
        capsys.readouterr()
        assert card.hold(ready, pauses.append) == card.GAVE_UP
        (state / "stop-unconfirmed").write_text("forbidden\n")
        assert card.hold(ready, pauses.append) == card.STOP_UNCONFIRMED  # the stop first
        assert capsys.readouterr().out == ""
    finally:
        process.kill()
        process.wait()


def test_a_retry_leaves_the_pair_to_a_running_launcher_and_stop_ends_it(state: Path, monkeypatch: pytest.MonkeyPatch,
                                                                         capsys: pytest.CaptureFixture[str]) -> None:
    loaded, process = card_at(state), launcher_stand_in(state)
    (state / "given-up").write_text("engine_exit\n")
    monkeypatch.setattr(Card, "load", lambda: loaded)
    assert card.main(["--retry"]) == 0 and not (state / "given-up").exists()
    assert "deferred to the next start" in capsys.readouterr().out  # a 0 that does not read as ready
    assert card.stop_pair(state, lambda _: process.wait()) == 0
    assert process.wait() == -signal.SIGTERM

    def unreadable(*args: Any, **kwargs: Any) -> Card:
        raise OSError("a checkout whose manifest cannot be read")

    monkeypatch.setattr(Card, "load", unreadable)
    monkeypatch.setattr(card, "stop_pair",
                        lambda at, running: 0 if (at, running) == (card.STATE, card.RUNNING) else 1)
    assert card.main(["--stop"]) == 0


def recorder(path: Path, calls: Path) -> None:
    """A stand-in command that notes each of its calls in `calls`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho "${{0##*/}} $*" >> {calls}\n')
    path.chmod(0o700)


def test_bootstrap_changes_nothing_when_it_runs_again_and_the_rentals_onstart_starts_the_card(tmp_path: Path) -> None:
    code, state, root, calls = tmp_path / "code", tmp_path / "state", tmp_path / "root", tmp_path / "calls"
    (code / "card").mkdir(parents=True)
    root.mkdir()
    environ = {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "SIMPLE_SERVING_CARD_DIR": str(state),
               "SIMPLE_SERVING_CARD_ROOT": str(root), **CREDENTIAL}
    # The rental's own onstart, which ends with the README's line. The platform may write it anew at every start.
    line = next(line for line in (card.CODE / "README.md").read_text().splitlines() if line.startswith("if [[ -f /"))
    rental = f"#!/usr/bin/env bash\n{line.replace('/workspace/simple-serving', str(code))}\n"

    def start_container() -> int:
        (root / "onstart.sh").write_text(rental)
        return subprocess.run(["bash", str(root / "onstart.sh")], env=environ, capture_output=True, timeout=60,
                              check=False).returncode

    assert start_container() == 0 and not calls.exists()  # the first start, before the checkout is on the disk
    for name in ("bootstrap.sh", "onstart.sh", "gateway-requirements.txt", "vllm-requirements.txt"):
        shutil.copy(card.CODE / "card" / name, code / "card" / name)
    manifest = (card.CODE / "card/manifest.env").read_text()
    (code / "card/manifest.env").write_text(manifest)
    weights, tokenizer = b"synthetic weights", b'{"synthetic": true}'
    model = state / "models" / Card.load(code=code, state=state, environ={}, root=root).manifest["MODEL_REVISION"]
    model.mkdir(parents=True)
    for name in ("model.safetensors", "other.safetensors"):  # the pinned one in place, so nothing is fetched
        (model / name).write_bytes(weights)
    (state / "tokenizer").mkdir()
    (state / "tokenizer/tokenizer.json").write_bytes(tokenizer)
    for path in (state / "gateway/bin/pip", state / "vllm/bin/pip", state / GATEWAY, tmp_path / "bin/aria2c",
                 tmp_path / "bin/curl", tmp_path / "bin/flock"):
        recorder(path, calls)

    def bootstrap(stdin: str) -> int:
        done = subprocess.run(["bash", str(code / "card/bootstrap.sh")], input=stdin, env=environ, capture_output=True,
                              text=True, timeout=60, check=False)
        assert not any(digest in done.stdout + done.stderr for digest in (CLIENT, CONTROL, METRICS))
        return done.returncode

    keys = f"{CLIENT}\n{CONTROL}\n"
    assert bootstrap("") == 4  # the real pins and locks pass, and the card holds no keys yet
    # The gateway's lock is uv.lock's non-dev set: the same packages, versions and hashes.
    lock = {package["name"]: package for package in tomllib.loads((card.CODE / "uv.lock").read_text())["package"]}
    runtime = ["simple-serving"]  # the project, then what it needs, and what that needs in turn
    for name in runtime:
        runtime += [need["name"] for need in lock[name].get("dependencies", []) if need["name"] not in runtime]
    text = (card.CODE / "card/gateway-requirements.txt").read_text()
    exported = re.findall(r"^(\S+)==(\S+) \\\n((?: +--hash=.*\n)+)", text, re.MULTILINE)
    assert {name: (version, set(re.findall(r"sha256:\w+", hashes))) for name, version, hashes in exported} == {
        name: (lock[name]["version"], {file["hash"] for file in [lock[name]["sdist"], *lock[name]["wheels"]]})
        for name in runtime[1:]}
    for name in ("gateway", "vllm"):
        (code / f"card/{name}-requirements.txt").write_text(f"{name}==0.0.1 \\\n    --hash=sha256:{'0' * 64}\n")
    assert bootstrap(keys) == 3  # the lock is not the pinned vLLM's
    assert not (state / "keys.json").exists()
    for older, text in ((state / "keys.json", json.dumps({"client": CLIENT, "control": CONTROL})),
                        (root / ".simple-chat-instance-api-key", "synthetic-older-key")):
        older.write_text(text + "\n")  # left wide, as by another hand
        older.chmod(0o644)
    pins = {"VLLM_VERSION": "0.0.1", "MODEL_FILES": f"model.safetensors:{hashlib.sha256(weights).hexdigest()}",
            "TOKENIZER_FILES": f"tokenizer.json:{hashlib.sha256(tokenizer).hexdigest()}",
            "MTP_SPECULATIVE_TOKENS": "0"}  # the drafter's fetch has a test of its own
    (code / "card/manifest.env").write_text(manifest + "".join(f"{name}={value}\n" for name, value in pins.items()))
    assert bootstrap(keys) == 0
    assert [path.name for path in model.iterdir()] == ["model.safetensors"]  # vLLM would load any other
    written = [state / "keys.json", state / "prepared", *root.glob(".simple-chat-*")]
    first = {path: (path.read_bytes(), path.stat().st_mode & 0o777) for path in written}
    assert bootstrap(keys) == 0
    assert bootstrap("") == 0  # the card holds the keys
    assert bootstrap(f"{hashlib.sha256(b'other').hexdigest()}\n{CONTROL}\n") == 5
    assert bootstrap(f"{CLIENT}\n") == 4
    assert bootstrap(f"{CLIENT}\n{CLIENT}\n") == 4
    assert {path: (path.read_bytes(), path.stat().st_mode & 0o777) for path in written} == first
    assert {mode for _, mode in first.values()} == {0o600}
    assert json.loads((state / "keys.json").read_text()) == {"client": CLIENT, "control": CONTROL}
    assert (root / "onstart.sh").read_text() == rental  # the preparation changes no onstart
    kept = [(root / f".simple-chat-instance-{name}").read_text() for name in ("id", "api-key")]
    assert kept == list(CREDENTIAL.values())
    assert start_container() == 0  # a resume: the rental's own onstart, as it was, starts the card
    noted = calls.read_text().splitlines()
    assert noted.count("python -m simple_serving.card") == 4  # the three preparations, then the resume
    assert not any(line.startswith(("aria2c", "curl", "flock")) for line in noted)  # nothing to fetch, and no guard
    assert not list(root.glob(".simple-chat-trial-*"))
    assert all("--require-hashes --only-binary :all:" in line for line in noted if line.startswith("pip install"))
    assert noted.count("pip check") == 6  # both venvs at each preparation
    assert Card.load(code=code, state=state, environ=environ, root=root).prepared()
    # The metrics key: a card that holds the first two keys alone takes it beside them, and then keeps all three.
    other = hashlib.sha256(b"other").hexdigest()
    for wrong in (CONTROL, CLIENT, "not-a-digest"):
        assert bootstrap(f"{keys}{wrong}\n") == 4
    assert bootstrap(f"{keys}{METRICS}\n") == 0
    three = {"client": CLIENT, "control": CONTROL, "metrics": METRICS}
    assert json.loads((state / "keys.json").read_text()) == three
    assert (state / "keys.json").stat().st_mode & 0o777 == 0o600
    assert bootstrap(keys) == 5  # the card holds other keys now, which stay
    assert bootstrap(f"{keys}{other}\n") == 5
    assert bootstrap(f"{keys}{METRICS}\n") == 0 and bootstrap("") == 0
    assert json.loads((state / "keys.json").read_text()) == three and not (state / "keys.json.new").exists()
    (state / "keys.json").unlink()  # a card prepared anew takes the three at once
    assert bootstrap(f"{keys}{METRICS}\n") == 0
    assert json.loads((state / "keys.json").read_text()) == three

    # Text switches change only the text preparation stamp. The picture lane needs neither field.
    picture_folder, picture_state = code / PictureCard.folder, tmp_path / "picture-state"
    picture_folder.mkdir(parents=True)
    picture_state.mkdir()
    for name in PictureCard.stamped:
        shutil.copy(card.CODE / PictureCard.folder / name, picture_folder / name)
    (picture_state / "keys.json").write_text(json.dumps(three))
    picture_stamp = hashlib.sha256(b"".join((picture_folder / name).read_bytes() for name in PictureCard.stamped))
    (picture_state / "prepared").write_text(picture_stamp.hexdigest() + "\n")
    pictures = PictureCard.load(code=code, state=picture_state, environ=environ, root=root)
    assert not {"THINKING_BUDGET", "VISION"} & pictures.manifest.keys()
    assert "--reasoning-config" not in pictures.engine().argv and "thinking_budget" not in pictures.config()["versions"]
    assert all("vision" not in key for key in pictures.config()["keys"])
    assert pictures.prepared()
    manifest_path = code / "card/manifest.env"
    base = manifest_path.read_text()
    before_calls, before_prepared = calls.read_bytes(), (state / "prepared").read_bytes()
    for switch in ("THINKING_BUDGET", "VISION"):
        for value in (None, "", "2", "01", "-1", "on"):
            changed = base.replace(f"{switch}=0\n", "") if value is None else base + f"{switch}={value}\n"
            manifest_path.write_text(changed)
            assert bootstrap("") == 3
            assert calls.read_bytes() == before_calls and (state / "prepared").read_bytes() == before_prepared
            assert json.loads((state / "keys.json").read_text()) == three and pictures.prepared()
    manifest_path.write_text(base + "VISION=1\n")  # the processor pins are required before any changes
    assert bootstrap("") == 3 and calls.read_bytes() == before_calls
    assert (state / "prepared").read_bytes() == before_prepared
    for value in ("1", "0"):
        manifest_path.write_text(base + f"THINKING_BUDGET={value}\n")
        loaded = Card.load(code=code, state=state, environ=environ, root=root)
        assert not loaded.prepared() and pictures.prepared()
        assert bootstrap("") == 0 and loaded.prepared() and pictures.prepared()
        assert loaded.config()["versions"].get("thinking_budget") == ("on" if value == "1" else None)
        assert json.loads((state / "keys.json").read_text()) == three
    processor = ("processor_config.json", "tokenizer.json", "tokenizer_config.json")
    for name in processor:
        (state / "tokenizer" / name).write_bytes(tokenizer)
    pins = {name: hashlib.sha256(tokenizer).hexdigest() for name in processor}
    base += "TOKENIZER_FILES=" + ",".join(f"{name}:{digest}" for name, digest in pins.items()) + "\n"
    before_calls, before_prepared = calls.read_bytes(), (state / "prepared").read_bytes()
    manifest_path.write_text(base.replace("processor_config.json:", "tokenizer_config.json:") + "VISION=1\n")
    assert bootstrap("") == 3  # three entries must name the three distinct processor assets
    assert calls.read_bytes() == before_calls and (state / "prepared").read_bytes() == before_prepared
    (model / ".hidden").write_text("keep")
    for value in ("1", "1", "0"):
        manifest_path.write_text(base + f"VISION={value}\n")
        loaded = Card.load(code=code, state=state, environ=environ, root=root)
        assert bootstrap("") == 0 and loaded.prepared() and pictures.prepared()
        assert json.loads((state / "keys.json").read_text()) == three
        assert loaded.config()["versions"].get("vision") == ("1" if value == "1" else None)
        assert sorted(path.name for path in model.iterdir()) == sorted(
            ["model.safetensors", ".hidden", *(processor if value == "1" else ())])
        if value == "1":
            assert all(hashlib.sha256((model / name).read_bytes()).hexdigest() == pins[name] for name in processor)
            (model / "processor_config.json").write_bytes(b"stale processor")  # next run repairs it from the pins
    assert all((state / "tokenizer" / name).read_bytes() == tokenizer for name in processor)
    assert not any(line.startswith(("aria2c", "curl")) for line in calls.read_text().splitlines())


def test_the_metrics_key_reads_the_metrics_and_has_no_other_right(state: Path) -> None:
    (state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL, "metrics": METRICS}))
    for value in ("0", "1"):
        (state / "gateway.json").write_text(json.dumps(card_at(state, VISION=value).config()))
        keys = {key.label: key for key in load_config(str(state / "gateway.json")).keys}
        assert (keys["metrics"].metrics, keys["metrics"].control, keys["metrics"].classes, keys["metrics"].scopes,
                keys["metrics"].vision) == (True, False, frozenset(), False, False)
        assert not keys["client"].metrics and keys["control"].control and not keys["control"].vision
        assert keys["client"].vision is (value == "1")
    (state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL}))  # prepared before it
    assert [key["label"] for key in card_at(state).config()["keys"]] == ["client", "control"]


def test_bootstrap_fetches_the_pinned_drafter_with_mtp_on_and_keeps_a_file_only_with_its_hash(tmp_path: Path) -> None:
    code, state, root, hub, calls = (tmp_path / name for name in ("code", "state", "root", "hub", "calls"))
    for path in (code / "card", root, hub, tmp_path / "bin"):
        path.mkdir(parents=True)
    for name in ("bootstrap.sh", "onstart.sh"):
        shutil.copy(card.CODE / "card" / name, code / "card" / name)
    for name in ("gateway", "vllm"):
        (code / f"card/{name}-requirements.txt").write_text(f"{name}==0.0.1 \\\n    --hash=sha256:{'0' * 64}\n")
    manifest = (card.CODE / "card/manifest.env").read_text()
    loaded = Card.load(state=state, environ={}, root=root).manifest
    model, drafter = state / "models" / loaded["MODEL_REVISION"], state / "drafters" / loaded["DRAFTER_REVISION"]
    for directory, name, data in ((model, "model.safetensors", b"synthetic weights"),  # in place, so not fetched
                                  (state / "tokenizer", "tokenizer.json", b"{}"), (drafter, "other.safetensors", b""),
                                  (hub, "model.safetensors", b"synthetic drafter"), (hub, "config.json", b"{}")):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)
    (state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL}) + "\n")
    for path in (state / "gateway/bin/pip", state / "vllm/bin/pip", state / GATEWAY):
        recorder(path, calls)
    aria2c = tmp_path / "bin/aria2c"  # notes its call and fetches the file of its URL from `hub`
    aria2c.write_text(f'#!/bin/sh\necho "aria2c $*" >> {calls}\nfor arg; do case $arg in --dir=*) dir=${{arg#*=}};; '
                      f'--out=*) out=${{arg#*=}};; esac; url=$arg; done\ncp "{hub}/${{url##*/}}" "$dir/$out"\n')
    aria2c.chmod(0o700)
    environ = {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "SIMPLE_SERVING_CARD_DIR": str(state),
               "SIMPLE_SERVING_CARD_ROOT": str(root), **CREDENTIAL}
    pins = {name: ",".join(f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()}" for path in files)
            for name, files in (("MODEL_FILES", [model / "model.safetensors"]),
                                ("TOKENIZER_FILES", [state / "tokenizer/tokenizer.json"]),
                                ("DRAFTER_FILES", [hub / "model.safetensors", hub / "config.json"]))}

    def bootstrap(tokens: str, vision: str = "0") -> int:
        lines = {"VLLM_VERSION": "0.0.1", "MTP_SPECULATIVE_TOKENS": tokens, "VISION": vision, **pins}.items()
        (code / "card/manifest.env").write_text(manifest + "".join(f"{name}={value}\n" for name, value in lines))
        return subprocess.run(["bash", str(code / "card/bootstrap.sh")], stdin=subprocess.DEVNULL, env=environ,
                              capture_output=True, timeout=60, check=False).returncode

    def fetched() -> list[str]:
        return [line.rsplit(" ", 1)[1] for line in calls.read_text().splitlines() if line.startswith("aria2c")]

    assert bootstrap("03") == 3 and bootstrap("x") == 3 and not calls.exists()  # a leading 0 is octal to bash
    assert bootstrap("0") == 0 and fetched() == []  # off: the drafter's directory stays as it was
    assert [path.name for path in drafter.iterdir()] == ["other.safetensors"]
    assert bootstrap("3") == 0
    assert fetched() == [f"https://huggingface.co/{loaded['DRAFTER_REPO']}/resolve/{loaded['DRAFTER_REVISION']}/{name}"
                         for name in ("model.safetensors", "config.json")]
    # vLLM would load any other *.safetensors
    assert sorted(path.name for path in drafter.iterdir()) == ["config.json", "model.safetensors"]
    assert (drafter / "model.safetensors").read_bytes() == b"synthetic drafter"
    assert bootstrap("3") == 0 and len(fetched()) == 2  # nothing more to fetch
    processor = ("processor_config.json", "tokenizer_config.json")
    for name in processor:
        (hub / name).write_bytes(b'{"synthetic_processor": true}')
        pins["TOKENIZER_FILES"] += f",{name}:{hashlib.sha256((hub / name).read_bytes()).hexdigest()}"
    assert bootstrap("3", "1") == 0
    assert fetched()[2:] == [f"https://huggingface.co/{loaded['TOKENIZER_REPO']}/resolve/"
                            f"{loaded['TOKENIZER_REVISION']}/{name}" for name in processor]
    assert all((model / name).read_bytes() == (state / "tokenizer" / name).read_bytes()
               for name in (*processor, "tokenizer.json"))
    started = calls.read_text().count("python -m simple_serving.card")
    before = (state / "prepared").read_bytes()
    (state / "tokenizer/processor_config.json").unlink()
    (hub / "processor_config.json").write_bytes(b'{"wrong": true}')
    assert bootstrap("3", "1") == 1  # an unpinned processor never reaches the model directory or starts the pair
    assert not (state / "tokenizer/processor_config.json").exists()
    assert not (state / "tokenizer/.processor_config.json.part").exists()
    assert (model / "processor_config.json").read_bytes() == b'{"synthetic_processor": true}'
    assert (state / "prepared").read_bytes() == before
    assert calls.read_text().count("python -m simple_serving.card") == started
    (hub / "processor_config.json").write_bytes(b'{"synthetic_processor": true}')
    assert bootstrap("3", "1") == 0
    assert bootstrap("3") == 0 and [path.name for path in model.iterdir()] == ["model.safetensors"]
    (drafter / "config.json").unlink()
    (hub / "config.json").write_bytes(b'{"other": true}')
    assert bootstrap("3") == 1  # the hub's bytes are not the pinned ones
    assert [path.name for path in drafter.iterdir()] == ["model.safetensors"]  # and no part of them stays


def test_a_copy_of_the_kv_cache_fits_only_with_the_shm_and_the_memory_for_it(tmp_path: Path) -> None:
    gib = 1024
    assert preflight.offload_reasons(0, None, None) == []  # off: nothing is asked of the container
    assert preflight.offload_reasons(12, 12 * gib + 512, 24 * gib) == []
    assert preflight.offload_reasons(12, 12 * gib + 511, 24 * gib) == ["shm_too_small"]
    assert preflight.offload_reasons(12, 30720, 24 * gib - 1) == ["memory_too_small"]
    # The 16 GiB that vLLM refused on a host of 251 GB whose /dev/shm was 15 GiB (docs/speed.md, item 5).
    assert preflight.offload_reasons(16, 15 * gib, 240 * gib) == ["shm_too_small"]
    assert preflight.offload_reasons(12, None, None, "expandable_segments:True") == [
        "shm_too_small", "memory_too_small", "expandable_segments"]  # a number that cannot be read refuses too
    assert preflight.largest_offload_gib(30720, 60 * gib) == 29
    assert preflight.largest_offload_gib(15 * gib, 16 * gib) == 4
    assert preflight.offer_needs(24) == {"offload_gib": 24, "min_memory_mib": 36 * gib, "min_ram_gb": 39,
                                         "min_shm_mib": 24 * gib + 512}
    # The size is what all the sessions write before one returns: E2B's twenty needed 2 GiB where 1 GiB gave nothing.
    assert preflight.copy_gib(20, 8300, 9) == 2 and preflight.copy_gib(4, 16000, preflight.DENSE_KIB_PER_TOKEN) == 27
    cgroup, proc = tmp_path / "cgroup", tmp_path / "proc"
    cgroup.mkdir()
    proc.mkdir()
    (proc / "meminfo").write_text(f"MemTotal:       {64 * gib * 1024} kB\nMemFree:        1 kB\n")
    assert preflight.memory_limit_mib(cgroup, proc) == 64 * gib  # no limit: the machine's RAM
    (cgroup / "memory.max").write_text("max\n")
    assert preflight.memory_limit_mib(cgroup, proc) == 64 * gib
    (cgroup / "memory.max").write_text(f"{20 * gib * 1024 * 1024}\n")
    assert preflight.memory_limit_mib(cgroup, proc) == 20 * gib  # the container's limit, below the machine's
    assert preflight.gpu_reasons((120, 32607), 120, 30000) == []
    assert preflight.gpu_reasons((89, 24564), 120, 30000) == ["gpu_compute_cap_too_low", "gpu_memory_too_small"]
    assert preflight.gpu_reasons(None, 120, 30000) == ["gpu_unknown"]


@pytest.mark.anyio
async def test_a_copy_the_container_cannot_hold_starts_nothing_gives_up_and_stops_the_instance(
        state: Path, logged: io.StringIO, monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in(state / ENGINE)
    stand_in(state / GATEWAY)
    monkeypatch.setattr(preflight, "measured", lambda environ: {"shm_free_mib": 15360, "memory_limit_mib": 20480})
    vast = FakeVast()
    assert await card.launch(card_at(state, KV_OFFLOADING_GIB="16"), asyncio.Event(), vast) == "unfit"
    assert vast.stopped and (state / "given-up").read_text() == "unfit\n"
    assert [row["code"] for row in rows(logged, "unfit")] == ["shm_too_small", "memory_too_small"]
    assert sorted(path.name for path in (state / "vllm/bin").iterdir()) == ["vllm", "vllm.plan"]  # never ran
    monkeypatch.setattr(preflight, "measured", lambda environ: pytest.fail("off, nothing is read"))
    assert card_at(state).unfit() == []


def test_the_qwen_profile_is_one_manifest_value_and_gemmas_arguments_stay_as_they_are(state: Path) -> None:
    (state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL}))
    gemma, qwen = card_at(state), card_at(state, MODEL_PROFILE="qwen")
    m = gemma.manifest
    assert m["MODEL_PROFILE"] == "gemma"  # off by default
    without = replace(gemma, manifest={name: value for name, value in m.items()
                                       if name != "MODEL_PROFILE" and not name.startswith("QWEN_")})
    assert gemma.engine().argv == without.engine().argv and gemma.config() == without.config()
    assert "model_profile" not in card.pinned_versions(m)
    assert not {"qwen3", m["QWEN_MODEL_ALIAS"], "--prefix-match-unit"} & set(gemma.engine().argv)
    argv = qwen.engine().argv
    assert argv[2] == str(state / "models" / m["QWEN_MODEL_REVISION"]) != gemma.engine().argv[2]

    def value(name: str) -> str:
        return argv[argv.index(name) + 1]

    assert value("--served-model-name") == m["QWEN_MODEL_ALIAS"] == qwen.config()["alias"]
    assert value("--tokenizer") == str(state / "tokenizers" / m["QWEN_TOKENIZER_REVISION"])
    assert value("--max-model-len") == m["QWEN_CONTEXT_TOKENS"] and qwen.config()["context_tokens"] == 32768
    assert value("--reasoning-parser") == "qwen3"
    # The drafter is a layer of the weights' own: no model is named, and nothing of a second repository is pinned.
    assert value("--speculative-config") == '{"method": "mtp", "num_speculative_tokens": 3}'
    # Nothing of the repository runs as code: vLLM 0.30.0 has the architecture (docs/qwen.md, fact 1).
    assert "--trust-remote-code" not in argv and "--language-model-only" in argv and "--prefix-match-unit" not in argv
    assert card.pinned_versions(qwen.manifest) == {
        "vllm": m["VLLM_VERSION"], "model_revision": m["QWEN_MODEL_REVISION"],
        "tokenizer_revision": m["QWEN_TOKENIZER_REVISION"], "model_profile": "qwen", "mtp_speculative_tokens": "3"}
    budget = card_at(state, MODEL_PROFILE="qwen", THINKING_BUDGET="1", QWEN_PREFIX_MATCH_UNIT="32",
                     QWEN_MTP_SPECULATIVE_TOKENS="0")
    assert budget.engine().argv[-4:] == ["--reasoning-config",
                                         '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}',
                                         "--prefix-match-unit", "32"]
    assert "--speculative-config" not in budget.engine().argv
    # Every pinned file has its size, which is what the bootstrap asks of the disk before it fetches.
    sizes = dict(entry.split(":") for entry in m["QWEN_MODEL_SIZES"].split(","))
    files = dict(entry.split(":") for entry in m["QWEN_MODEL_FILES"].split(","))
    assert sizes.keys() == files.keys() and all(re.fullmatch("[0-9a-f]{64}", digest) for digest in files.values())
    assert sum(map(int, sizes.values())) == 20_559_300_089


def test_the_small_profile_a_gpu_cache_size_and_more_generations_at_once_are_manifest_values(state: Path) -> None:
    (state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL}))
    gemma = card_at(state)
    m = gemma.manifest
    assert (m["KV_CACHE_MEMORY_MIB"], m["GENERATIONS_AT_ONCE"]) == ("0", "0")  # off by default
    without = replace(gemma, manifest={name: value for name, value in m.items() if not name.startswith(
        ("SMALL_", "KV_CACHE_MEMORY_MIB", "GENERATIONS_AT_ONCE"))})
    assert gemma.engine().argv == without.engine().argv and gemma.config() == without.config()
    assert "--kv-cache-memory-bytes" not in gemma.engine().argv
    assert not {"kv_cache_memory_mib", "generations_at_once"} & card.pinned_versions(m).keys()
    small = card_at(state, MODEL_PROFILE="small", KV_CACHE_MEMORY_MIB="384", KV_OFFLOADING_GIB="4",
                    GENERATIONS_AT_ONCE="8")
    argv = small.engine().argv

    def value(name: str) -> str:
        return argv[argv.index(name) + 1]

    assert argv[2] == str(state / "models" / m["SMALL_MODEL_REVISION"])
    assert value("--served-model-name") == m["SMALL_MODEL_ALIAS"] == small.config()["alias"] == "gemma-4-e2b-it"
    assert value("--tokenizer") == str(state / "tokenizers" / m["SMALL_TOKENIZER_REVISION"])
    assert value("--max-model-len") == "16384" and small.config()["context_tokens"] == 16384
    assert json.loads(value("--speculative-config")) == {
        "method": "mtp", "num_speculative_tokens": 3, "model": str(state / "drafters" / m["SMALL_DRAFTER_REVISION"])}
    # The GPU's cache is 384 MiB whatever the card has free, and the copy in RAM is the manifest's.
    assert value("--kv-cache-memory-bytes") == str(384 << 20) and value("--kv-offloading-size") == "4"
    assert "--language-model-only" in argv
    limits = small.config()["limits"]
    assert limits["shared"]["active"] == 8 and {limits[cls]["active"] for cls in ("reader", "agent", "internal")} == {8}
    assert limits["reader"]["waiting"] == gemma.config()["limits"]["reader"]["waiting"]  # only the count at once
    pinned = card.pinned_versions(small.manifest)
    assert (pinned["model_profile"], pinned["kv_cache_memory_mib"], pinned["generations_at_once"]) == (
        "small", "384", "8")


def test_bootstrap_refuses_an_unfit_container_before_it_installs_and_fetches_qwen_by_its_hashes(
        tmp_path: Path) -> None:
    code, state, root, hub, calls = (tmp_path / name for name in ("code", "state", "root", "hub", "calls"))
    for path in (code / "card", code / "simple_serving", root, hub, tmp_path / "bin"):
        path.mkdir(parents=True)
    for name in ("card/bootstrap.sh", "card/onstart.sh", "simple_serving/preflight.py"):
        shutil.copy(card.CODE / name, code / name)
    for name in ("gateway", "vllm"):
        (code / f"card/{name}-requirements.txt").write_text(f"{name}==0.0.1 \\\n    --hash=sha256:{'0' * 64}\n")
    manifest = (card.CODE / "card/manifest.env").read_text()
    loaded = Card.load(state=state, environ={}, root=root).manifest
    files = {"model.safetensors": b"synthetic qwen weights", "model-mtp-bf16.safetensors": b"synthetic mtp layer",
             "tokenizer.json": b'{"synthetic": "qwen"}'}
    for name, data in files.items():
        (hub / name).write_bytes(data)
    (state / "tokenizer").mkdir(parents=True)
    (state / "tokenizer/tokenizer.json").write_bytes(b"{}")  # Gemma's, which the qwen profile leaves alone
    (state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL}) + "\n")
    for path in (state / "gateway/bin/pip", state / "vllm/bin/pip", state / GATEWAY):
        recorder(path, calls)
    aria2c = tmp_path / "bin/aria2c"  # notes its call and fetches the file of its URL from `hub`
    aria2c.write_text(f'#!/bin/sh\necho "aria2c $*" >> {calls}\nfor arg; do case $arg in --dir=*) dir=${{arg#*=}};; '
                      f'--out=*) out=${{arg#*=}};; esac; url=$arg; done\ncp "{hub}/${{url##*/}}" "$dir/$out"\n')
    aria2c.chmod(0o700)
    environ = {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "SIMPLE_SERVING_CARD_DIR": str(state),
               "SIMPLE_SERVING_CARD_ROOT": str(root), **CREDENTIAL}

    def pinned(*names: str) -> str:
        return ",".join(f"{name}:{hashlib.sha256(files[name]).hexdigest()}" for name in names)

    weights = ("model.safetensors", "model-mtp-bf16.safetensors")
    pins = {"VLLM_VERSION": "0.0.1", "QWEN_MODEL_FILES": pinned(*weights),
            "QWEN_MODEL_SIZES": ",".join(f"{name}:{len(files[name])}" for name in weights),
            "QWEN_TOKENIZER_FILES": pinned("tokenizer.json")}

    def bootstrap(gpu: str = "12.0, 32607", **changes: str) -> int:
        smi = tmp_path / "bin/nvidia-smi"  # the card: its compute capability and its memory
        smi.write_text(f'#!/bin/sh\necho "{gpu}"\n')
        smi.chmod(0o700)
        lines = {**pins, "MODEL_PROFILE": "qwen", **changes}.items()
        (code / "card/manifest.env").write_text(manifest + "".join(f"{name}={value}\n" for name, value in lines))
        return subprocess.run(["bash", str(code / "card/bootstrap.sh")], stdin=subprocess.DEVNULL, env=environ,
                              capture_output=True, timeout=60, check=False).returncode

    assert bootstrap(MODEL_PROFILE="llama") == 3 and bootstrap(VISION="1") == 3  # no such profile; qwen is text only
    assert bootstrap(MODEL_PROFILE="gemma", KV_OFFLOADING_GIB="1000000") == 9  # a copy no container holds
    assert bootstrap("8.9, 24564") == 9  # a 4090: not the card NVFP4's kernels are for, and too small
    assert bootstrap("12.0, 16303") == 9  # a Blackwell card too small for the weights
    assert bootstrap(QWEN_MODEL_SIZES=f"model.safetensors:{1 << 60}") == 9  # more than the disk holds
    assert not calls.exists() and not (state / "models").exists()  # refused before pip and before any fetch
    model, tokenizer = state / "models" / loaded["QWEN_MODEL_REVISION"], state / "tokenizers"
    (hub / "model-mtp-bf16.safetensors").write_bytes(b"other bytes")
    assert bootstrap() == 1  # the hub's bytes are not the pinned ones
    assert [path.name for path in model.iterdir()] == ["model.safetensors"]  # and no part of them stays
    (hub / "model-mtp-bf16.safetensors").write_bytes(files["model-mtp-bf16.safetensors"])
    assert bootstrap() == 0
    fetched = [line.rsplit(" ", 1)[1] for line in calls.read_text().splitlines() if line.startswith("aria2c")]
    assert fetched == [f"https://huggingface.co/{loaded['QWEN_MODEL_REPO']}/resolve/{loaded['QWEN_MODEL_REVISION']}/"
                       f"{name}" for name in ("model.safetensors", "model-mtp-bf16.safetensors",
                                              "model-mtp-bf16.safetensors", "tokenizer.json")]
    assert {path.name: path.read_bytes() for path in model.iterdir()} == {name: files[name] for name in weights}
    assert (tokenizer / loaded["QWEN_TOKENIZER_REVISION"] / "tokenizer.json").read_bytes() == files["tokenizer.json"]
    assert (state / "tokenizer/tokenizer.json").read_bytes() == b"{}" and not (state / "drafters").exists()
    assert calls.read_text().count("python -m simple_serving.card") == 1
    # The small profile has a model, a tokenizer and a drafter of its own, each by its hash.
    files |= {"small.safetensors": b"synthetic small weights", "small-drafter.safetensors": b"synthetic drafter",
              "small-tokenizer.json": b'{"synthetic": "small"}'}
    for name in ("small.safetensors", "small-drafter.safetensors", "small-tokenizer.json"):
        (hub / name).write_bytes(files[name])
    small = {"MODEL_PROFILE": "small", "SMALL_MODEL_FILES": pinned("small.safetensors"),
             "SMALL_TOKENIZER_FILES": pinned("small-tokenizer.json"),
             "SMALL_DRAFTER_FILES": pinned("small-drafter.safetensors")}
    assert bootstrap(**small, GENERATIONS_AT_ONCE="9") == 3  # more than vLLM's batch holds
    assert bootstrap(**small, KV_CACHE_MEMORY_MIB="384", GENERATIONS_AT_ONCE="8") == 0
    kept = {str(path.relative_to(state)): path.read_bytes() for path in state.rglob("small*") if path.is_file()}
    assert kept == {f"models/{loaded['SMALL_MODEL_REVISION']}/small.safetensors": files["small.safetensors"],
                    f"tokenizers/{loaded['SMALL_TOKENIZER_REVISION']}/small-tokenizer.json":
                        files["small-tokenizer.json"],
                    f"drafters/{loaded['SMALL_DRAFTER_REVISION']}/small-drafter.safetensors":
                        files["small-drafter.safetensors"]}

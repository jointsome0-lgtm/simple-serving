"""The standalone concurrency sweep (docs/sweep.md): the order of
its levels, their admission by the cache and by --until, the arguments of every `vllm bench serve`, and one line of
numbers and fixed words a level. A stand-in takes the place of vLLM's benchmark and a loopback server that of the
engine's /metrics; all traffic stays on loopback."""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import stat
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from simple_serving import sweep
from simple_serving.sweep import Level, Simulated, Target

ROOT = Path(__file__).resolve().parent.parent
MARKER = "synthetic-prompt-5d1e9a"  # stands in for a prompt or an output, which no line may hold
ALIAS = sweep.read_manifest(ROOT / "card/manifest.env")["MODEL_ALIAS"]
STEP = {"prompt": 16256, "generation": 2048, "queries": 16000, "hits": 12000, "drafts": 300, "draft_tokens": 900,
        "accepted": 600}  # what the engine's counters gain while the stand-in runs once
METRICS = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{engine="0",model_name="{marker}"}} {running}
vllm:num_requests_waiting{{engine="0",model_name="{marker}"}} 0.0
vllm:num_requests_waiting_by_reason{{engine="0",model_name="{marker}",reason="capacity"}} 0.0
vllm:kv_cache_usage_perc{{engine="0",model_name="{marker}"}} {kv_usage}
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{{engine="0",model_name="{marker}"}} {prompt}
vllm:prompt_tokens_created{{engine="0",model_name="{marker}"}} 1.79e+09
vllm:prompt_tokens_by_source_total{{engine="0",model_name="{marker}",source="local_compute"}} {prompt}
vllm:prompt_tokens_cached_total{{engine="0",model_name="{marker}"}} {hits}
vllm:generation_tokens_total{{engine="0",model_name="{marker}"}} {generation}
vllm:prefix_cache_queries_total{{engine="0",model_name="{marker}"}} {queries}
vllm:prefix_cache_hits_total{{engine="0",model_name="{marker}"}} {hits}
vllm:num_preemptions_total{{engine="0",model_name="{marker}"}} 0.0
vllm:spec_decode_num_drafts_total{{engine="0",model_name="{marker}"}} {drafts}
vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="{marker}"}} {draft_tokens}
vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="{marker}"}} {accepted}
vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",model_name="{marker}",position="0"}} {pos0}
vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",model_name="{marker}",position="1"}} {pos1}
vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",model_name="{marker}",position="2"}} {pos2}
vllm:time_to_first_token_seconds_bucket{{engine="0",le="0.001",model_name="{marker}"}} 0.0
"""
# The stand-in for `vllm bench serve`: it notes its arguments, makes the engine's counters move, prints the marker and
# writes a result that holds it, as vLLM's holds the outputs with --save-detailed. With `sleep` it waits to be ended.
BENCH = '''import json, os, signal, subprocess, sys, time, urllib.request
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "argv.jsonl"), "a") as file:
    file.write(json.dumps(sys.argv[1:]) + "\\n")
with open(os.path.join(here, "env.json"), "w") as file:
    json.dump({"dropped": {name: name in os.environ for name in
              ("OPENAI_API_KEY", "CONTAINER_API_KEY", "SAVE_TO_PYTORCH_BENCHMARK_FORMAT")},
              "offline": {name: os.environ.get(name) for name in
              ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_USE_RUST_BENCH")}}, file)
mode_file = os.path.join(here, "mode")
mode = open(mode_file).read() if os.path.exists(mode_file) else "done"
if mode == "sleep":
    ready = os.path.join(here, "child-ready")
    child = subprocess.Popen([sys.executable, "-c",
        "import signal,time,pathlib,sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).touch(); time.sleep(600)", ready])
    while not os.path.exists(ready):
        time.sleep(0.005)
    with open(os.path.join(here, "pids.tmp"), "w") as file:
        json.dump([os.getpid(), child.pid], file)
    os.replace(os.path.join(here, "pids.tmp"), os.path.join(here, "pids.json"))
    time.sleep(600)
args = dict(zip(sys.argv[3::1], sys.argv[4::1]))
urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
    "http://127.0.0.1:" + args["--port"] + "/work").read()
print("$MARKER")
result = {"completed": int(args["--num-prompts"]), "failed": 0, "duration": 12.5, "output_throughput": 327.68,
          "median_tpot_ms": 7.25, "median_ttft_ms": 180.0, "p90_ttft_ms": 5400.0, "generated_texts": ["$MARKER"],
          "model_id": "$MARKER"}
if mode == "failed":
    result["failed"] = 1
elif mode == "incomplete":
    result["completed"] -= 1
elif mode == "nan":
    result["output_throughput"] = float("nan")
elif mode == "negative":
    result["duration"] = -1
with open(os.path.join(args["--result-dir"], args["--result-filename"]), "w") as file:
    if mode == "malformed":
        file.write("$MARKER")
    else:
        json.dump(result, file)
sys.exit(3 if mode == "exit" else 0)
'''


class Engine(ThreadingHTTPServer):
    """The engine's /metrics on loopback. GET /work moves its counters on by STEP, as a level's requests would."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.counters = dict.fromkeys(STEP, 0)
        self.running = 0
        self.metrics_override: str | None = None

    def text(self) -> str:
        if self.metrics_override is not None:
            return self.metrics_override
        c = self.counters
        return METRICS.format(marker=MARKER, running=float(self.running), kv_usage=0.25, **c,
                              pos0=c["accepted"] // 2, pos1=c["accepted"] // 3, pos2=c["accepted"] // 6)


class Handler(BaseHTTPRequestHandler):
    server: Engine

    def do_GET(self) -> None:
        if self.path == "/work":
            for name, step in STEP.items():
                self.server.counters[name] += step
        body = self.server.text().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def engine() -> Iterator[Engine]:
    server = Engine()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def bench(tmp_path: Path) -> Path:
    path = tmp_path / "bench/vllm.py"
    path.parent.mkdir()
    path.write_text(BENCH.replace("$MARKER", MARKER))
    return path


def test_levels_run_in_order_while_they_fit_the_cache_and_the_end(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = sweep.levels()
    simulated = Simulated()
    lines = list(sweep.sweep(plan, 10000, simulated.run, clock=simulated, cache_tokens=104492))
    ran = [(line["set"], line["context"], line["requests"]) for line in lines if line.get("status") == "planned"]
    assert ran == [*(("shared", 16000, r) for r in (2, 4, 3, 6, 1)), *(("shared", 4000, r) for r in (2, 4, 3, 6, 8, 1)),
                   ("shared", 40000, 2), ("shared", 40000, 1), *(("none", 16000, r) for r in (2, 4, 3, 6, 1))]
    cache = [(line["set"], line["context"], line["requests"]) for line in lines if line.get("reason") == "cache"]
    assert cache == [("shared", 16000, 8), *(("shared", 40000, r) for r in (4, 3, 6, 8)), ("none", 16000, 8)]
    assert lines[-1] == {"end": "done", "levels_run": 18, "wall_s": round(simulated.now, 1), "cache_tokens": 104492}
    assert 780 < simulated.now < 960  # the whole sweep, about 14.5 minutes of the card

    # A different explicitly measured capacity changes admission, without changing the order of requests.
    simulated = Simulated()
    larger = list(sweep.sweep(plan, 10000, simulated.run, clock=simulated, cache_tokens=400000))
    assert [line["level"] for line in larger if line.get("status") == "planned"] == list(range(1, 25))
    assert larger[-1]["levels_run"] == 24

    # --until: a level starts only if its estimate fits before it, in order, and later levels are skipped.
    simulated = Simulated()
    until = plan[0].estimate() + plan[1].estimate() + 1
    lines = list(sweep.sweep(plan, until, simulated.run, clock=simulated, cache_tokens=104492))
    assert [line["level"] for line in lines if line.get("status") == "planned"] == [1, 2]
    assert {line.get("reason") for line in lines[2:-1]} == {"until", "cache"}
    assert lines[-1]["end"] == "until"

    # A level that took twice its estimate makes the estimates after it twice as long; a cut or busy one ends it all.
    def slow(level: Level, _until: float) -> dict[str, Any]:
        simulated.now += 2 * level.estimate()
        return {"status": "cut" if level.index == 3 else "done"}

    simulated = Simulated()
    lines = list(sweep.sweep(plan, 10000, slow, clock=simulated, cache_tokens=104492))
    assert lines[1]["est_s"] == round(2 * plan[1].estimate(), 1)
    assert [line.get("status") for line in lines] == ["done", "done", "cut", None]
    assert lines[-1] == {"end": "until", "levels_run": 2, "wall_s": round(simulated.now, 1), "cache_tokens": 104492}

    # Planning needs capacity but no card files, result directory, process or engine connection.
    locations = ["--state", str(tmp_path / "missing-state"), "--manifest", str(tmp_path / "missing-manifest"),
                 "--result-dir", str(tmp_path / "missing-results"), "--bench-cmd", "/missing-benchmark"]
    assert sweep.main(["--plan", "--cache-tokens", "400000", *locations]) == 0
    shown = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert shown[-1]["levels_run"] == 24 and shown[-1]["end"] == "done"
    for args in ([], ["--cache-tokens", "0"], ["--cache-tokens", "-1"], ["--cache-tokens", "1.5"],
                 *(["--cache-tokens", "104492", "--until", value] for value in ("0", "-1", "nan", "inf"))):
        with pytest.raises(SystemExit) as error:
            sweep.main([*args, *locations])
        assert error.value.code == 2
        assert capsys.readouterr().out == ""
    assert list(tmp_path.iterdir()) == []


def test_a_dry_run_prints_numbers_and_fixed_words_only(tmp_path: Path, engine: Engine,
                                                     capsys: pytest.CaptureFixture[str],
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.env"
    port = str(engine.server_address[1])
    manifest.write_text((ROOT / "card/manifest.env").read_text().replace("ENGINE_PORT=8092", f"ENGINE_PORT={port}"))
    stand_in, results = bench(tmp_path), tmp_path / "results"
    for name in ("OPENAI_API_KEY", "CONTAINER_API_KEY", "SAVE_TO_PYTORCH_BENCHMARK_FORMAT"):
        monkeypatch.setenv(name, MARKER + "-env")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    # 90 s admits every level that fits the cache but the one of 6 requests that share nothing, estimated at 103 s.
    code = sweep.main(["--cache-tokens", "104492", "--until", str(time.time() + 90),
                       "--state", str(tmp_path / "state"), "--manifest",
                       str(manifest), "--result-dir", str(results), "--bench-cmd",
                       shlex.join([sys.executable, str(stand_in), "bench", "serve"])])
    captured = capsys.readouterr()
    out = captured.out
    lines = [json.loads(line) for line in out.splitlines()]

    assert code == 0 and captured.err == ""
    assert all(value not in out for value in (MARKER, str(tmp_path), "OPENAI_API_KEY", "CONTAINER_API_KEY"))
    environment = json.loads((stand_in.parent / "env.json").read_text())
    assert not any(environment["dropped"].values())
    assert environment["offline"] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_USE_RUST_BENCH": "0"}
    assert stat.S_IMODE(results.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in results.iterdir())
    assert MARKER in (results / "L01-shared-16000-c2.log").read_text()
    assert MARKER in (results / "L01-shared-16000-c2.json").read_text()
    for line in lines:
        assert sweep.checked(line)
    ran = [line for line in lines if line.get("status") == "done"]
    assert [line["level"] for line in ran] == [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 24]
    assert [line["level"] for line in lines if line.get("reason") == "until"] == [22]
    assert lines[-1] == {"end": "until", "levels_run": 17, "wall_s": lines[-1]["wall_s"], "cache_tokens": 104492}
    assert ran[0] == {
        "level": 1, "set": "shared", "context": 16000, "requests": 2, "prompts": 4, "est_s": 39.1,
        "wall_s": ran[0]["wall_s"], "status": "done", "completed": 4, "failed_requests": 0, "duration_s": 12.5,
        "out_tok_s": 327.7, "out_tok_s_per_request": 137.9, "ttft_ms_median": 180.0, "ttft_ms_p90": 5400.0,
        "running_before": 0, "waiting_before": 0, "prompt_tokens": 16256, "generation_tokens": 2048,
        "prefix_cache_queries": 16000, "prefix_cache_hits": 12000, "preemptions": 0, "drafts": 300,
        "draft_tokens": 900, "accepted": 600, "accepted_by_pos": [300, 200, 100], "running_peak": 0,
        "waiting_peak": 0, "kv_usage_peak": 0.25,
    }

    argvs = [json.loads(line) for line in (stand_in.parent / "argv.jsonl").read_text().splitlines()]
    tokenizer = str(tmp_path / "state/tokenizer")
    assert argvs[0] == [
        "bench", "serve", "--backend", "openai", "--host", "127.0.0.1", "--port", port, "--model", ALIAS,
        "--tokenizer", tokenizer, "--dataset-name", "random", "--random-prefix-len", "15744", "--random-input-len",
        "256", "--random-output-len", "512", "--num-prompts", "4", "--max-concurrency", "2", "--seed", "1",
        "--num-warmups", "1", "--ignore-eos", "--temperature", "0", "--percentile-metrics", "ttft,tpot",
        "--metric-percentiles", "50,90", "--disable-tqdm", "--save-result", "--result-dir", str(results),
        "--result-filename", "L01-shared-16000-c2.json",
    ]
    assert len(argvs) == len(ran)
    shapes = [(a[a.index("--random-prefix-len") + 1], a[a.index("--random-input-len") + 1],
               a[a.index("--num-prompts") + 1], a[a.index("--max-concurrency") + 1], a[a.index("--num-warmups") + 1])
              for a in argvs]
    assert shapes[5] == ("3744", "256", "4", "2", "1")  # 4000 tokens at 2 at once
    assert shapes[11] == ("39744", "256", "4", "2", "1")  # 40000 at 2
    assert shapes[13] == ("0", "16000", "4", "2", "0")  # 16000 that share nothing, at 2
    assert shapes[10] == ("3744", "256", "4", "1", "1")  # 4000 alone: at least 4 prompts
    assert shapes[8] == ("3744", "256", "12", "6", "1")
    assert len({a[a.index("--seed") + 1] for a in argvs}) == len(argvs)  # every level's prompts are its own


def test_the_end_and_other_requests_stop_a_level(
        tmp_path: Path, engine: Engine, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    results = tmp_path / "results"
    results.mkdir(mode=0o700)
    stand_in = bench(tmp_path)
    target = Target([sys.executable, str(stand_in), "bench", "serve"], str(engine.server_address[1]), ALIAS,
                    tmp_path / "tokenizer", results)
    level = sweep.levels()[0]
    seen: list[sweep.Peaks] = []
    start = sweep.Peaks.start

    def observe(peaks: sweep.Peaks) -> None:
        seen.append(peaks)
        start(peaks)

    monkeypatch.setattr(sweep.Peaks, "start", observe)
    monkeypatch.setattr(sweep, "POLL_S", 0.01)
    monkeypatch.setattr(sweep, "TERM_GRACE_S", 0.1)
    engine.running = 1  # A text run: the level starts nothing.
    assert sweep.run_level(level, time.time() + 60, target=target) == {
        "status": "busy", "running_before": 1, "waiting_before": 0}
    engine.running = 0
    normal = engine.text()
    running = next(line for line in normal.splitlines() if line.startswith("vllm:num_requests_running{"))
    waiting = next(line for line in normal.splitlines() if line.startswith("vllm:num_requests_waiting{"))
    counter = next(line for line in normal.splitlines() if line.startswith("vllm:prompt_tokens_total{"))
    engine.metrics_override = normal.replace(waiting, waiting.rsplit(" ", 1)[0] + " 1")
    assert sweep.run_level(level, time.time() + 60, target=target)["status"] == "busy"
    for malformed in (normal.replace(running, ""), normal.replace(waiting, ""),
                      normal.replace(running, running.rsplit(" ", 1)[0] + " 0.5"),
                      normal.replace(counter, counter.rsplit(" ", 1)[0] + " NaN"),
                      normal.replace(counter, counter.rsplit(" ", 1)[0] + " -1"),
                      normal.replace(counter, counter.rsplit(" ", 1)[0] + " broken")):
        engine.metrics_override = malformed
        assert sweep.run_level(level, time.time() + 60, target=target)["status"] == "unreachable"
        assert not (stand_in.parent / "argv.jsonl").exists()
        assert not seen
    engine.metrics_override = None

    # A successful CLI child can still report an unsuccessful or unreadable benchmark. Each stops the sweep.
    manifest = tmp_path / "manifest.env"
    manifest.write_text(f"ENGINE_PORT={target.port}\nMODEL_ALIAS={ALIAS}\n")
    argv = ["--cache-tokens", "104492", "--state", str(tmp_path / "state"), "--manifest", str(manifest),
            "--bench-cmd", shlex.join(target.command)]
    existing = tmp_path / "existing"
    existing.mkdir(mode=0o755)
    original_mode = stat.S_IMODE(existing.stat().st_mode)
    (existing / "keep").write_text(MARKER)
    linked = tmp_path / "linked"
    linked.symlink_to(existing, target_is_directory=True)
    for destination in (existing, linked):
        assert sweep.main([*argv, "--result-dir", str(destination), "--until", str(time.time() + 60)]) == 2
        captured = capsys.readouterr()
        assert MARKER not in captured.out + captured.err and str(tmp_path) not in captured.out + captured.err
        assert not (stand_in.parent / "argv.jsonl").exists()
        assert (existing / "keep").read_text() == MARKER and stat.S_IMODE(existing.stat().st_mode) == original_mode
    for index, mode in enumerate(("failed", "incomplete", "malformed", "nan", "negative", "exit"), start=1):
        (stand_in.parent / "mode").write_text(mode)
        assert sweep.main([*argv, "--result-dir", str(tmp_path / f"failed-{index}"),
                           "--until", str(time.time() + 60)]) == 1
        captured = capsys.readouterr()
        assert all(value not in captured.out + captured.err for value in (MARKER, str(tmp_path), "OPENAI_API_KEY"))
        lines = [json.loads(line) for line in captured.out.splitlines()]
        assert len(lines) == 2 and lines[0]["status"] == "failed"
        assert lines[-1]["end"] == "failed" and lines[-1]["levels_run"] == 0
        assert all(sweep.checked(line) for line in lines)
        assert len((stand_in.parent / "argv.jsonl").read_text().splitlines()) == index
        assert all(not thread.is_alive() for thread in seen)

    # Optional counters cannot silently become measured zeros when the engine does not publish them.
    (stand_in.parent / "mode").write_text("done")
    engine.metrics_override = f"{running}\n{waiting}"
    line = sweep.run_level(level, time.time() + 60, target=target)
    assert line["status"] == "done" and "prompt_tokens" not in line and "draft_tokens" not in line
    engine.metrics_override = None
    missing = Target([str(tmp_path / "missing-benchmark")], target.port, ALIAS, target.tokenizer, results)
    assert sweep.run_level(level, time.time() + 60, target=missing)["status"] == "failed"
    assert all(not thread.is_alive() for thread in seen)

    # Deadline and cancellation both end the benchmark's group, including a child that ignores SIGTERM.
    (stand_in.parent / "mode").write_text("sleep")
    pids = stand_in.parent / "pids.json"
    original_wait = sweep.subprocess.Popen.wait
    for cancel in (False, True):
        pids.unlink(missing_ok=True)
        (stand_in.parent / "child-ready").unlink(missing_ok=True)
        interrupted = False

        def interrupt_once(process: Any, timeout: float | None = None) -> int:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                ready_by = time.monotonic() + 2
                while not pids.exists():
                    assert time.monotonic() < ready_by, "synthetic benchmark did not start"
                    time.sleep(0.01)
                raise KeyboardInterrupt
            return original_wait(process, timeout)

        started = time.monotonic()
        try:
            if cancel:
                previous_handler = signal.getsignal(signal.SIGTERM)
                with monkeypatch.context() as patch:
                    patch.setattr(sweep.subprocess.Popen, "wait", interrupt_once)
                    assert sweep.main([*argv, "--result-dir", str(tmp_path / "cancelled"),
                                       "--until", str(time.time() + 60)]) == 130
                captured = capsys.readouterr()
                assert json.loads(captured.out) == {"end": "cancelled", "cache_tokens": 104492}
                assert signal.getsignal(signal.SIGTERM) == previous_handler
            else:
                line = sweep.run_level(level, time.time() + 1, target=target)
                assert line["status"] == "cut"
                assert "out_tok_s" not in line and sweep.checked(line)
            assert time.monotonic() - started < 3
            assert seen and all(not thread.is_alive() for thread in seen)
            parent, child = json.loads(pids.read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(parent, 0)  # The direct child must also have been reaped.
            deadline = time.monotonic() + 1
            while True:
                try:
                    child_state = Path(f"/proc/{child}/stat").read_text().split()[2]
                except FileNotFoundError:
                    break
                if child_state == "Z":
                    break
                assert time.monotonic() < deadline, "benchmark child survived its process group's deadline"
                time.sleep(0.01)
        finally:
            # Even a failed assertion must not leave this synthetic process group running.
            if pids.exists():
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(json.loads(pids.read_text())[0], signal.SIGKILL)

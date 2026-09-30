"""Synthetic interference is detectable; broken or unobserved overlap is never evidence of no stall."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from simple_serving.fake_engine import SENTENCE, Script
from simple_serving.interference import cli, metrics
from simple_serving.interference import fake as fake_module
from simple_serving.interference.fake import KEY, MARKER, FakeInterference, synthetic_gateway
from simple_serving.interference.runner import Options, Target, run

from .support import BOT, CONTROL, running, until

QUICK = Options(rounds=1, long_tokens=2048, victim_tokens=32, aggressor_tokens=1, trigger_chunks=4,
                start_timeout_s=2, trial_timeout_s=5)


async def measured(fake: FakeInterference, options: Options = QUICK) -> dict[str, Any]:
    async with synthetic_gateway(fake) as gateway, httpx.AsyncClient(timeout=None, trust_env=False) as client:
        report = await run(client, Target(gateway.url, gateway.key, gateway.model), options)
        await until(lambda: not fake.pending and not any(fake.active.values()))
    encoded = json.dumps(report, allow_nan=False)
    assert MARKER not in encoded and KEY not in encoded
    assert all(call.ended_at is not None for call in fake.calls)
    return report


@pytest.mark.anyio
async def test_synthetic_cold_prefill_stalls_decode_while_the_exact_warm_request_does_not(
        capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeInterference()
    report = await measured(fake)
    assert (report["status"], report["code"]) == ("ok", "ok")
    alone, cold, warm = report["trials"]
    assert [trial["case"] for trial in report["trials"]] == ["alone", "cold", "warm"]
    assert all(trial["victim"]["chunks"] == QUICK.victim_tokens for trial in report["trials"])
    assert all(trial["victim"]["gap_ms"]["samples"] == QUICK.victim_tokens - 1 for trial in report["trials"])
    cold_gap = cold["victim"]["post_injection_gap_ms"]["max"]
    assert cold_gap >= fake.stall_s * 1000 * 0.8
    assert cold_gap > 3 * max(alone["victim"]["gap_ms"]["max"], warm["victim"]["gap_ms"]["max"])
    assert cold["aggressors"][0]["usage"]["cached_tokens"] == 0
    assert warm["warm_cache_confirmed"] is True
    assert warm["aggressors"][0]["usage"]["cached_tokens"] == report["sizing"]["prompt_tokens"]
    assert all(min(trial["http_overlap_ms"]) > 0 and min(trial["victim_chunks_during_aggressor"]) > 0
               for trial in (cold, warm))
    assert alone["injection_ms"] is None
    for trial in (cold, warm):
        assert (trial["victim"]["started_ms"] < trial["injection_ms"]
                <= trial["aggressors"][0]["started_ms"] < trial["victim"]["ended_ms"])

    full_freeze = FakeInterference(post_prefill_tail=False, token_delay_s=0.01)
    frozen = await measured(full_freeze)
    assert (frozen["status"], frozen["code"]) == ("ok", "ok")
    cold = frozen["trials"][1]
    assert cold["victim_chunks_during_aggressor"] == [0]
    assert min(cold["victim_chunks_after_aggressor_start"]) > 0
    assert cold["victim"]["post_injection_gap_ms"]["max"] >= full_freeze.stall_s * 1000 * 0.8
    assert cold["aggressors"][0]["ended_ms"] < cold["victim"]["ended_ms"]
    assert MARKER not in capsys.readouterr().out


@pytest.mark.anyio
async def test_metadata_empty_content_and_usage_cannot_trigger_injection_and_utf8_fragments_count_once() -> None:
    gate = asyncio.Event()
    fake = FakeInterference(metadata_gate=gate, gate_decoder=2, early_usage=True, fragments=True)
    async with synthetic_gateway(fake) as gateway, httpx.AsyncClient(timeout=None, trust_env=False) as client:
        task = asyncio.create_task(run(client, Target(gateway.url, gateway.key, gateway.model), QUICK))
        try:
            await asyncio.wait_for(fake.metadata_sent.wait(), 2)
            await asyncio.sleep(fake.token_delay_s * 4)
            assert not fake.long_started.is_set() and not task.done()
            assert fake.calls[-1].role == "decoder" and fake.calls[-1].tokens == 0
            gate.set()
            report = await asyncio.wait_for(task, 5)
        finally:
            gate.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await until(lambda: not fake.pending)
    assert report["status"] == "ok"
    assert all(trial["victim"]["chunks"] == QUICK.victim_tokens for trial in report["trials"])
    cold_decoder = [call for call in fake.calls if call.role == "decoder"][1]
    first_long = next(call for call in fake.calls if call.role == "long")
    assert cold_decoder.first_token_at is not None and first_long.started_at > cold_decoder.first_token_at


@pytest.mark.anyio
async def test_incomplete_failed_and_timed_out_streams_report_no_measurement_and_close_their_peers(
        capsys: pytest.CaptureFixture[str]) -> None:
    cases = (
        ("http_error", "decoder", "failed", "http_error"),
        ("stream_error", "long", "failed", "stream_error"),
        ("missing_done", "decoder", "failed", "missing_done"),
        ("early_finish", "decoder", "inconclusive", "victim_ended_before_injection"),
        ("empty_content", "decoder", "inconclusive", "victim_ended_before_injection"),
        ("hang", "long", "failed", "timeout"),
    )
    for fault, role, status, code in cases:
        fake = FakeInterference(fault=fault, fault_on=role)
        options = replace(QUICK, victim_tokens=128, trial_timeout_s=0.35) if fault == "hang" else QUICK
        report = await measured(fake, options)
        assert (report["status"], report["code"]) == (status, code), fault
        if role == "long":
            assert any(call.disconnected for call in fake.calls), fault
        if status == "inconclusive":
            assert all(not trial["aggressors"] for trial in report["trials"])
    output = capsys.readouterr()
    assert all(secret not in output.out + output.err for secret in (MARKER, KEY))


def test_cli_defaults_to_synthetic_and_checks_live_bounds_capacity_and_private_output(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    instances: list[FakeInterference] = []

    def no_owner_config(_: Path) -> dict[str, Any]:
        raise AssertionError("No owner configuration should be opened by these invocations")

    @asynccontextmanager
    async def observed_gateway():
        fake = FakeInterference()
        instances.append(fake)
        async with synthetic_gateway(fake) as gateway:
            yield gateway
        assert not fake.pending and not any(fake.active.values())
        assert all(call.ended_at is not None for call in fake.calls)

    monkeypatch.setattr(cli, "read_config", no_owner_config)
    monkeypatch.setattr(fake_module, "synthetic_gateway", observed_gateway)
    assert cli.main([]) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["synthetic"] is True and report["status"] == "ok"
    assert len(report["trials"]) == 3 and all(not trial["diagnostic_flags"] for trial in report["trials"])
    assert all(secret not in output.out + output.err for secret in (MARKER, KEY))

    assert cli.main(["--concurrency", "4"]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "admission_limited"
    assert not instances[-1].calls
    assert cli.main(["--concurrency", "4", "--aggressor-class", "reader"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert all(len(trial["aggressors"]) == 3 for trial in report["trials"] if trial["case"] != "alone")
    assert report["trials"][-1]["warm_cache_confirmed"] is True

    count = len(instances)
    for args in (["run"], ["run", "--config", "not-opened.toml"], ["--rounds", "0"],
                 ["--rounds", "11"], ["--max-seconds", "nan"], ["--max-seconds", "1801"]):
        with pytest.raises(SystemExit) as error:
            cli.main(args)
        assert error.value.code == 2
    for index, public in enumerate(("http://[::1", "http://127.0.0.1:0")):
        destination = tmp_path / f"invalid-url-{index}.json"
        assert cli.main(["run", "--config", "not-opened.toml", "--public", public,
                         "--out", str(destination)]) == 2
        assert json.loads(destination.read_text())["code"] in ("loopback_url_required", "port_required")
    assert len(instances) == count
    existing = tmp_path / "existing.json"
    existing.write_text("keep this report")
    capsys.readouterr()
    assert cli.main(["run", "--config", "not-opened.toml", "--out", str(existing)]) == 2
    assert existing.read_text() == "keep this report"
    assert json.loads(capsys.readouterr().out)["code"] == "output_or_config_unavailable"

    parsed = metrics.parse(f'vllm:generation_tokens_total{{model_name="{MARKER}"}} 4\n'
                           f'vllm:unknown_private_metric{{label="{KEY}"}} 99\n', engine=True)
    assert parsed == {"generation_tokens_total": 4.0}


@pytest.mark.anyio
async def test_experiment_crosses_the_actual_gateway_contract_and_does_not_invent_warm_cache() -> None:
    options = replace(QUICK, long_tokens=128, trigger_chunks=1)
    async with running() as stack:
        stack.fake.script = Script(delay_s=0.01, usage={"cached_tokens": 0})
        report = await cli._experiment(stack.public, stack.control, BOT, CONTROL, options, 15, True)
        await until(lambda: not stack.service.work and stack.service.ours_inflight == 0)
        assert all(call.ended_at is not None for call in stack.fake.calls_of("generate"))
    assert len(report["trials"]) == 3
    assert all(trial["status"] != "failed" for trial in report["trials"])
    assert report["trials"][0]["status"] == "ok"
    assert report["trials"][-1]["warm_cache_confirmed"] is False
    assert (report["status"], report["code"]) == ("inconclusive", "warm_cache_unconfirmed")
    assert all(request["usage"]["prompt_tokens"] == report["sizing"]["prompt_tokens"]
               for trial in report["trials"] for request in trial["aggressors"])
    encoded = json.dumps(report, allow_nan=False)
    assert all(secret not in encoded for secret in (BOT, CONTROL, *SENTENCE))


@pytest.mark.anyio
async def test_diagnostics_require_complete_numeric_evidence_and_cancel_scrapes_before_closing_the_client(
        monkeypatch: pytest.MonkeyPatch) -> None:
    raw = (f'vllm:generation_tokens_total{{engine="0",model_name="{MARKER}"}} 2\n'
           f'vllm:generation_tokens_total{{engine="1",model_name="{KEY}"}} 3\n'
           'vllm:num_requests_running 0\n'
           'vllm:num_requests_waiting NaN\n'
           'vllm:prompt_tokens_total +Inf\n'
           'vllm:prefix_cache_hits_total -1\n'
           'vllm:prefix_cache_queries_total 1e30\n'
           f'vllm:unknown_private_metric{{label="{MARKER}"}} 99\n')
    assert metrics.parse(raw, engine=True) == {"generation_tokens_total": 5.0, "num_requests_running": 0.0}
    assert metrics.changes({"total": 5, "missing": 10}, {"total": 8, "new": 2}) == ({"total": 3}, False)
    assert metrics.changes({"total": 5}, {"total": 4}) == ({"total": -1}, True)

    fake = FakeInterference()
    gateway = metrics.parse(fake.metrics(engine=False))
    engine = metrics.parse(fake.metrics(engine=True), engine=True)
    required = [f"{family}:{cls}" for family in (
        "generations_active", "generations_waiting", "counts_active", "counts_waiting") for cls in fake_module.CLASSES]
    metrics.capacity({"gateway": gateway, "engine": engine}, 2, "internal")
    for lane, fields in (("gateway", required), ("engine", ["num_requests_running", "num_requests_waiting"])):
        for field in fields:
            snapshot = {"gateway": dict(gateway), "engine": dict(engine)}
            del snapshot[lane][field]
            with pytest.raises(metrics.DiagnosticError, match=r"^diagnostics_unavailable$"):
                metrics.capacity(snapshot, 2, "internal")

    scrape_started = asyncio.Event()

    class YieldingTransport(httpx.AsyncBaseTransport):
        metric_requests = 0
        pending = 0
        cancelled = False
        closed = False

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            assert not self.closed
            if request.url.path == "/v1/state":
                return httpx.Response(200, json={"status": "ready", "model": fake_module.MODEL,
                                                "context_tokens": fake_module.CONTEXT, "boot_id": "test-boot"})
            if request.url.path == "/metrics":
                self.metric_requests += 1
                if self.metric_requests == 2:
                    self.pending += 1
                    scrape_started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        self.pending -= 1
                        self.cancelled = True
                return httpx.Response(200, text=fake.metrics(engine=False))
            assert request.url.path == "/metrics/engine"
            return httpx.Response(200, text=fake.metrics(engine=True))

        async def aclose(self) -> None:
            await asyncio.sleep(0)
            assert self.cancelled and self.pending == 0
            self.closed = True

    transport = YieldingTransport()
    client_type = httpx.AsyncClient

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return client_type(transport=transport, **kwargs)

    async def finished_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        await asyncio.wait_for(scrape_started.wait(), 2)
        return {"status": "ok", "code": "ok", "trials": []}

    monkeypatch.setattr(cli.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(cli, "run", finished_run)
    report = await cli._experiment("http://synthetic.invalid", "http://synthetic.invalid", KEY, KEY, QUICK, 5, True)
    assert report["status"] == "ok" and transport.closed and transport.cancelled and transport.pending == 0
    assert all(secret not in json.dumps(report, allow_nan=False) for secret in (MARKER, KEY))

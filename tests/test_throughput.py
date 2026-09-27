"""The throughput harness of docs/speed.md's item 1, off the card: its reader of /metrics on canned text, the A/B
launcher's commands, the workloads built from a synthetic pack, the identity set's comparison, the reader-overlap
flag, the timing sweep's arguments, the microbenchmarks' plans, and the result and report, which hold counts, times
and fixed words only."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from simple_serving import bot_schemas, card, policy
from simple_serving.fake_engine import Script
from simple_serving.throughput import (
    cli,
    identity,
    launcher,
    metrics,
    microbench,
    overlap,
    report,
    runner,
    sweep,
    workloads,
)
from simple_serving.validation import check_chat

from . import support
from .support import until
from .test_card import (
    CLIENT,
    CONTROL,
    CREDENTIAL,
    ENGINE,
    GATEWAY,
    READY,
    card_at,
    exits,
    prepared,
    ran,
    stand_in,
    vllm_line,
)

SECRET = "synthetic-secret-7f3a9c"  # stands in for a prompt or an output, which nothing the harness writes may hold
ALIAS = "gemma-4-31b-heretic-nvfp4"


def canned_metrics(scale: int = 1, running: int = 3) -> str:
    """/metrics as vLLM 0.30.0's Prometheus client writes it, with `scale` times the counts."""
    labels = f'engine="0",model_name="{ALIAS}"'
    lines = [
        "# HELP vllm:num_requests_running Number of requests in model execution batches.",
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{{{labels}}} {running}.0",
        f"vllm:num_requests_waiting{{{labels}}} 1.0",
        f'vllm:num_requests_waiting_by_reason{{{labels},reason="capacity"}} 1.0',
        f'vllm:num_requests_waiting_by_reason{{{labels},reason="deferred"}} 0.0',
        f"vllm:kv_cache_usage_perc{{{labels}}} 0.8125",
        "# TYPE vllm:prefix_cache_queries counter",
        f"vllm:prefix_cache_queries_total{{{labels}}} {120000 * scale}.0",
        f"vllm:prefix_cache_queries_created{{{labels}}} 1.7908e+09",
        f"vllm:prefix_cache_hits_total{{{labels}}} {60000 * scale}.0",
        f"vllm:num_preemptions_total{{{labels}}} {2 * scale}.0",
        f"vllm:prompt_tokens_total{{{labels}}} {120000 * scale}.0",
        f'vllm:prompt_tokens_by_source_total{{{labels},source="local_compute"}} {60000 * scale}.0',
        f'vllm:prompt_tokens_by_source_total{{{labels},source="local_cache_hit"}} {60000 * scale}.0',
        f'vllm:prompt_tokens_by_source_total{{{labels},source="external_kv_transfer"}} 0.0',
        f'vllm:prompt_tokens_by_source_total{{{labels},source="{SECRET}"}} 7.0',  # a label value that is not vLLM's
        f"vllm:prompt_tokens_cached_total{{{labels}}} {60000 * scale}.0",
        f"vllm:generation_tokens_total{{{labels}}} {5000 * scale}.0",
        f"vllm:spec_decode_num_drafts_total{{{labels}}} {1000 * scale}.0",
        f"vllm:spec_decode_num_draft_tokens_total{{{labels}}} {3000 * scale}.0",
        f"vllm:spec_decode_num_accepted_tokens_total{{{labels}}} {1890 * scale}.0",
        *(f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{{labels},position="{position}"}} {count * scale}.0'
          for position, count in enumerate((780, 620, 490))),
        f'vllm:request_success_total{{engine="0",finished_reason="stop",model_name="{ALIAS}"}} 10.0',
        f'vllm:time_to_first_token_seconds_bucket{{engine="0",le="0.001",model_name="{ALIAS}"}} 0.0',
        f"vllm:{SECRET.replace('-', '_')}_total{{{labels}}} 1.0",
    ]
    return "\n".join(lines) + "\n"


def test_metrics_keep_the_named_counters_and_gauges_as_integers_and_no_other_value() -> None:
    read = metrics.parse(canned_metrics())
    assert read == {
        "num_requests_running": 3, "num_requests_waiting": 1, "num_requests_waiting_by_reason_capacity": 1,
        "num_requests_waiting_by_reason_deferred": 0, "kv_cache_usage_perc": 812,
        "prefix_cache_queries": 120000, "prefix_cache_hits": 60000, "num_preemptions": 2, "prompt_tokens": 120000,
        "prompt_tokens_by_source_local_compute": 60000, "prompt_tokens_by_source_local_cache_hit": 60000,
        "prompt_tokens_by_source_external_kv_transfer": 0, "prompt_tokens_cached": 60000,
        "generation_tokens": 5000, "spec_decode_num_drafts": 1000, "spec_decode_num_draft_tokens": 3000,
        "spec_decode_num_accepted_tokens": 1890, "spec_decode_num_accepted_tokens_per_pos_0": 780,
        "spec_decode_num_accepted_tokens_per_pos_1": 620, "spec_decode_num_accepted_tokens_per_pos_2": 490,
    }
    moved = metrics.delta(read, metrics.parse(canned_metrics(scale=3)))
    assert moved["prompt_tokens"] == 240000 and moved["generation_tokens"] == 10000
    assert "num_requests_running" not in moved and "kv_cache_usage_perc" not in moved
    assert metrics.acceptance(moved) == [0.78, 0.62, 0.49]
    assert metrics.delta(metrics.parse(canned_metrics(scale=3)), read)["prompt_tokens"] == -1  # a restart between
    sampler = metrics.Sampler(client=None, base_url="")  # type: ignore[arg-type]
    for running in (1, 4, 2):
        sampler.take(metrics.gauges(metrics.parse(canned_metrics(running=running))))
    assert sampler.peaks["num_requests_running"] == 4 and sampler.samples == 3
    assert metrics.parse("") == {} and metrics.acceptance({}) == []


# The A/B launcher.

@pytest.fixture(autouse=True)
def no_vast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing the harness starts may reach Vast: every stop that is not a fake fails the test."""
    async def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the harness tried to reach Vast")

    monkeypatch.setattr("simple_serving.vast.stop", refuse)


@pytest.fixture
def state(tmp_path: Path) -> Path:
    (tmp_path / "state/logs").mkdir(parents=True)
    (tmp_path / "state/keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL}))
    return tmp_path / "state"


def test_each_change_is_one_edit_of_the_cards_own_command_and_manifest_env_stays(state: Path) -> None:
    base = card_at(state)
    manifest_file = card.CODE / "card/manifest.env"
    before = manifest_file.read_bytes()
    reference = launcher.plan_for(base, launcher.parse_change("reference"))
    assert reference.engine.argv == base.engine().argv and not reference.timing_only
    assert set(reference.expect) >= {("arg_num_speculative_tokens", 3), ("arg_kv_cache_dtype_fp8", None),
                                     ("arg_max_num_batched_tokens", 1024)}
    assert reference.absent == ["kv_offloading_cpu"]

    def extra(change: str) -> list[str]:
        argv = launcher.plan_for(base, launcher.parse_change(change)).engine.argv
        return [part for part in argv if part not in reference.engine.argv]

    assert extra("linear=humming") == ["--kernel-config", '{"linear_backend_per_quant": {"nvfp4_w4a16": "humming"}}']
    assert extra("retention=dense") == ["--prefix-cache-retention-interval", "None"]
    assert extra("watermark=0.02") == ["--watermark", "0.02"]
    assert extra("kv-offloading=12") == ["--kv-offloading-size", "12"]
    assert extra("gpu-memory=0.95") == ["0.95"] and extra("batched-tokens=512") == ["512"]
    assert extra("kv-cache=int4_per_token_head") == ["int4_per_token_head"]
    assert extra("attention=flashinfer") == ["--attention-backend", "FLASHINFER"]
    speculation = json.loads(extra("verify=block")[0])
    assert speculation["rejection_sample_method"] == "block" and speculation["num_speculative_tokens"] == 3
    assert json.loads(extra("per-batch=1-4:3,5-8:2")[0])["num_speculative_tokens_per_batch_size"] == [
        [1, 4, 3], [5, 8, 2]]
    assert "--speculative-config" not in launcher.plan_for(base, launcher.parse_change("mtp=0")).engine.argv
    mtp4 = launcher.plan_for(base, launcher.parse_change("mtp=4"))
    assert ("arg_num_speculative_tokens", 4) in mtp4.expect and mtp4.manifest["MTP_SPECULATIVE_TOKENS"] == "4"
    synthetic = launcher.plan_for(base, launcher.parse_change("synthetic=0.78,0.62,0.49"))
    assert synthetic.timing_only and json.loads(synthetic.engine.argv[-1])["synthetic_acceptance_rates"] == [
        0.78, 0.62, 0.49]
    four = launcher.plan_for(base, launcher.parse_change("synthetic=0.78,0.62,0.49,0.37"))  # the count is the rates'
    assert json.loads(four.engine.argv[-1])["num_speculative_tokens"] == 4
    assert ("arg_num_speculative_tokens", 4) in four.expect and base.manifest["MTP_SPECULATIVE_TOKENS"] == "3"
    sliding = launcher.plan_for(base, launcher.parse_change("attention-sliding=flashinfer"))
    assert sliding.engine.env["VLLM_HAS_FLASHINFER_CUBIN"] == "1"
    for wrong in ("mtp=9", "mtp=3", "linear=cutlass", "synthetic=0.5,0.6,0.4", "synthetic=" + "0.5," * 8 + "0.4",
                  "synthetic=nan", "watermark=2",
                  "gpu-memory=0.92", "per-batch=4-1:3", f"kv-cache={SECRET}", "reference=1", "nothing"):
        with pytest.raises(launcher.ChangeError) as refused:
            launcher.plan_for(base, launcher.parse_change(wrong))
        assert SECRET not in str(refused.value)
    assert manifest_file.read_bytes() == before


def test_the_proof_needs_every_expected_code_and_none_of_the_absent_ones(state: Path) -> None:
    plan = launcher.plan_for(card_at(state), launcher.parse_change("linear=humming"))
    shown: list[dict[str, Any]] = [
        {"code": "arg_max_model_len", "value": 65536}, {"code": "arg_max_num_seqs", "value": 8},
        {"code": "arg_max_num_batched_tokens", "value": 1024}, {"code": "arg_kv_cache_dtype_fp8"},
        {"code": "arg_num_speculative_tokens", "value": 3}, {"code": "linear_override_nvfp4_w4a16_humming"},
        {"code": "arg_linear_nvfp4_w4a16_humming"}, {"code": "nvfp4_gemm_humming"}]
    assert launcher.proof(plan, shown)["ok"]
    marlin = [row for row in shown if row["code"] != "nvfp4_gemm_humming"] + [{"code": "nvfp4_gemm_marlin"}]
    assert launcher.proof(plan, marlin)["missing"] == ["nvfp4_gemm_humming"]
    three = [row for row in shown if row["code"] != "arg_num_speculative_tokens"]
    assert launcher.proof(plan, [*three, {"code": "arg_num_speculative_tokens", "value": 4}])["missing"] == [
        "arg_num_speculative_tokens=3"]
    assert launcher.proof(plan, [*shown, {"code": "kv_offloading_cpu"}])["unexpected"] == ["kv_offloading_cpu"]


def test_the_harness_gateway_has_its_own_key_the_rental_limits_and_no_credential(state: Path) -> None:
    base = card_at(state)
    plan = launcher.plan_for(base, launcher.parse_change("reference"))
    config = launcher.harness_config(base, plan, launcher.key_digest("synthetic-harness-key"))
    assert [key["label"] for key in config["keys"]] == ["client", "control", "harness"]
    harness = config["keys"][-1]
    assert harness == {"sha256": hashlib.sha256(b"synthetic-harness-key").hexdigest(), "label": "harness",
                       "classes": ["internal"], "default": "internal", "scopes": True, "control": False}
    assert config["limits"]["internal"]["active"] == 8 and config["limits"]["shared"]["active"] == 8
    assert card.PROVISIONAL["limits"]["internal"]["active"] == 2  # the card's own limits are left as they were
    assert config["versions"]["harness_change"] == "reference" and config["idle_timeout_s"] == launcher.IDLE_S
    timing = launcher.plan_for(base, launcher.parse_change("synthetic=0.8,0.6,0.5"))
    assert [key["label"] for key in launcher.harness_config(base, timing, "0" * 64)["keys"]] == ["control", "harness"]
    gateway = launcher.harness_gateway(base, state / "run/gateway.json")
    assert base.gateway().env["CONTAINER_API_KEY"] == CREDENTIAL["CONTAINER_API_KEY"]
    assert not {"CONTAINER_API_KEY", "CONTAINER_ID"} & set(gateway.env)
    assert gateway.env["SIMPLE_SERVING_CONFIG"] == str(state / "run/gateway.json")


@pytest.mark.anyio
async def test_a_start_that_fails_ends_the_pair_and_leaves_the_instance_running(state: Path,
                                                                                logged: io.StringIO) -> None:
    base = card_at(state)
    run_dir = state / "throughput/run"
    run_dir.mkdir(parents=True)
    stand_in(state / ENGINE, "ERROR 09-27 12:00:00 [core.py:710] RuntimeError: CUDA error: an illegal memory access",
             exit_code=1)
    stand_in(state / GATEWAY)
    plan = launcher.plan_for(base, launcher.parse_change("mtp=4"))
    async with launcher.started(base, plan, run_dir, "synthetic-harness-key") as start:
        assert start.outcome == "engine_exit" and not start.pair.ready.is_set()
    assert "cuda_error" in start.pair.noted and not (state / "given-up").exists()
    assert (run_dir / "gateway.json").stat().st_mode & 0o777 == 0o600
    assert not {"CONTAINER_API_KEY", "CONTAINER_ID"} & set(ran(state / GATEWAY)["env"])


@pytest.mark.anyio
async def test_a_start_that_is_ready_shows_its_change_took_effect(state: Path, logged: io.StringIO) -> None:
    base = card_at(state)
    run_dir = state / "throughput/run"
    run_dir.mkdir(parents=True)
    arguments = {"max_model_len": 65536, "max_num_seqs": 8, "max_num_batched_tokens": 1024, "kv_cache_dtype": "fp8",
                 "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
                 "prefix_cache_retention_interval": None}
    stand_in(state / ENGINE, vllm_line("api_utils.py:286", "non-default args: %s", arguments).decode().rstrip())
    stand_in(state / GATEWAY, READY)
    plan = launcher.plan_for(base, launcher.parse_change("retention=dense"))
    async with launcher.started(base, plan, run_dir, "synthetic-harness-key") as start:
        assert start.outcome == "ready"
        await until(lambda: bool(start.pair.choices))  # the stand-ins print at once; vLLM long before it is ready
        assert launcher.proof(plan, start.pair.choices)["ok"]
    assert exits(logged) == {"gateway": -signal.SIGTERM, "engine": -signal.SIGTERM}
    written = json.loads((run_dir / "gateway.json").read_text())
    assert "synthetic-harness-key" not in json.dumps(written)


def test_the_harness_takes_over_only_a_prepared_card_whose_pair_has_ended(tmp_path: Path) -> None:
    base = prepared(tmp_path)
    assert launcher.take_over(base, pause=lambda seconds: None) == 0  # no launcher runs: nothing to end
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", base.ports["engine"]))
        assert launcher.take_over(base, pause=lambda seconds: None) == card.PORT_TAKEN
    (base.state / "prepared").write_text("0" * 64 + "\n")
    assert launcher.take_over(base, pause=lambda seconds: None) == card.NOT_PREPARED


def synthetic_pack(root: Path, scenes: int = 16, chars: int = 800) -> Path:
    """The public eval pack's form with synthetic text: five scenarios, each a seed and a chain of scenes, whose nodes
    are stored in reverse so that only their parent links give the order."""
    for index, name in enumerate(workloads.SCENARIOS):
        nodes, parent = {}, None
        for number in range(scenes):
            node = f"x{index}{number:02d}"
            nodes[node] = {"id": node, "parent": parent, "input": f"{SECRET} input {number}",
                           "text": f"{SECRET} " + "word " * (chars // 5), "time": "", "truncated": False}
            parent = node
        state = {"seeds": {"s1": {"id": "s1", "title": f"Story {index}", "startTime": "evening",
                                  "text": f"{SECRET} seed"}},
                 "stories": {"h1": {"id": "h1", "seedId": "s1", "nodes": dict(reversed(nodes.items()))}}}
        (root / name).mkdir(parents=True)
        (root / name / "frozen.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    return root


def tokens(request: workloads.Request) -> float:
    return workloads.characters(request.body["messages"]) / 4


def test_the_workloads_are_the_bots_requests_built_from_the_pack(tmp_path: Path) -> None:
    stories = workloads.load_pack(synthetic_pack(tmp_path))
    assert [len(story.scenes) for story in stories] == [16] * 5
    assert stories[0].scenes[3].input.endswith("input 3")
    shape = workloads.Shape(at=3, context=2000, turns=4, gap_s=5.0)
    built = {name: workloads.build(name, stories, support.ALIAS, shape) for name in workloads.NAMES}
    for name, lanes in built.items():
        for request in (request for lane in lanes for request in lane):
            check_chat(request.body, support.ALIAS)  # the gateway takes it as it is
            assert policy.READER_SCOPE.fullmatch(request.scope)
            assert request.body["temperature"] == (0.2 if name == "compactions" else 0.8)
            assert request.body["max_tokens"] == workloads.MAX_TOKENS[name]
        assert workloads.at_once(name, shape) == (1 if name in ("reading", "interleaved") else 3)
    turns = built["turns"]
    assert [len(lane) for lane in turns] == [4, 4, 4] and len({lane[0].scope for lane in turns}) == 3
    for lane in turns:
        assert len({request.scope for request in lane}) == 1
        for before, after in itertools.pairwise(lane):  # each turn extends the one before, as a story's does
            assert after.body["messages"][:len(before.body["messages"]) - 1] == before.body["messages"][:-1]
            assert len(after.body["messages"]) == len(before.body["messages"]) + 2
        assert tokens(lane[0]) <= 2000 < tokens(lane[1])
    schema = built["compactions"][0][0].body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["facts"]["items"]["properties"]["source"]["items"]["enum"] == ["n1", "n2", "n3", "n4"]
    fact = {"kind": "event", "at": "evening", "text": "The keeper lights the lamp.", "source": ["n2", "n3"]}
    assert not bot_schemas.problems({"facts": [fact]}, schema)
    assert len(built["compactions"]) == len(built["scenes"]) == len(built["unshared"]) == 6
    assert len({lane[0].scope for lane in built["unshared"]}) == 6
    assert all(tokens(lane[0]) <= 2000 for lane in built["unshared"])
    [reading] = built["reading"]
    assert [request.gap_s for request in reading] == [0, 5, 5, 5] and len({r.scope for r in reading}) == 1
    [interleaved] = built["interleaved"]
    assert [(request.lane, request.turn) for request in interleaved[:5]] == [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1)]
    assert len(interleaved) == 16 and [request.gap_s for request in interleaved][:2] == [0, 5]
    assert len({request.scope for request in interleaved}) == 4
    firsts = [tokens(request) for request in interleaved[:4]]
    assert firsts == sorted(firsts) and all(first <= 2000 + 4000 * lane for lane, first in enumerate(firsts))


@pytest.mark.anyio
async def test_a_pass_counts_each_request_runs_them_at_once_and_keeps_numbers_only(tmp_path: Path) -> None:
    stories = workloads.load_pack(synthetic_pack(tmp_path, scenes=6, chars=400))
    block = support.service_with(context_tokens=65536, limits={"internal": {"active": 8, "max_tokens": 4096}})
    async with support.running(block) as stack:
        stack.fake.script = Script(first_delay_s=0.05, usage={"cached_tokens": 16})
        target = runner.Target(stack.public, stack.gateway.service.config.engine_url, support.BOT, support.ALIAS)
        lanes = workloads.build("scenes", stories, support.ALIAS, workloads.Shape(at=2))
        async with httpx.AsyncClient(timeout=10) as client:
            result = await runner.run_pass(client, target, lanes, 2, sample_s=0.01)
    generations = stack.fake.calls_of("generate")
    assert len(generations) == 4 and len(stack.fake.calls_of("tokenize")) == 8  # the harness's and the gateway's
    assert {call.body["priority"] for call in generations} == {block["engine_priority"]["internal"]}
    assert len({call.body["cache_salt"] for call in generations}) == 4  # a scope each
    most = max(sum(1 for other in generations if other.started_at <= call.started_at < (other.ended_at or 0))
               for call in generations)
    assert most == 2
    assert result["requests"] == result["done"] == 4 and result["failed"] == {}
    assert result["count_mismatches"] == 0 and result["cached_tokens"] == 64
    assert result["read_tokens"] == result["input_tokens"] - 64 and result["output_tokens"] == 4 * 13
    assert result["metrics"]["generation_tokens"] == 4 * 13 and result["metrics"]["prompt_tokens_cached"] == 64
    assert result["peaks"]["num_requests_running"] >= 1 and result["samples"] >= 1
    assert result["first_token_ms"]["median"] >= 50 and result["output_tokens_per_s"] > 0
    assert 0 < result["window"][1] - result["window"][0] < 10  # epoch seconds, for the reader-overlap flag
    assert all(outcome["counted"] == outcome["input_tokens"] for outcome in result["outcomes"])
    assert SECRET not in json.dumps(result)


# The identity set.

def test_the_identity_set_finds_the_first_difference_and_keeps_only_chained_digests() -> None:
    key, other = b"k" * 32, b"o" * 32
    ids = list(range(40))
    assert identity.first_difference(ids, ids) is None
    assert identity.first_difference(ids, [*ids[:17], 99, *ids[18:]]) == 17
    assert identity.first_difference(ids, ids[:30]) == 30
    kept = identity.digests(ids, key)
    assert len(kept) == 3 and kept != identity.digests(ids, other)  # blocks of 16, the last one shorter
    assert identity.digests([*ids[:5], 99, *ids[6:]], key)[1:] != kept[1:]  # each block chained to the one before
    reference = {"digests": [kept, kept, kept, None]}
    changed = [ids, [*ids[:20], 7, *ids[21:]], ids[:39], ids]
    verdict = identity.against(reference, changed, key)
    assert verdict == {"same": 1, "of": 4, "failed": 1, "first_difference": [None, 16, 32, None], "visible": True}
    floor = identity.floor([ids] * 16, [ids] * 8 + [ids[:39]] * 8)
    assert (floor["same"], floor["visible"], floor["first_difference"][-1]) == (8, False, 39)
    assert identity.floor([ids] * 16, [ids] * 7 + [None] * 9)["visible"]


@pytest.mark.anyio
async def test_the_identity_set_runs_on_the_engine_and_leaves_digests_only(tmp_path: Path) -> None:
    stories = workloads.load_pack(synthetic_pack(tmp_path / "pack", scenes=8, chars=400))
    bodies = identity.payloads(stories, support.ALIAS)
    assert len(bodies) == 16 and {body["temperature"] for body in bodies} == {0}
    assert all(body["return_token_ids"] and body["max_tokens"] <= identity.MAX_TOKENS and "cache_salt" not in body
               for body in bodies)
    assert sum(1 for body in bodies if "response_format" in body) == 4  # the compactions, under the bot's schema
    words = [{"role": "assistant"}, *({"content": f"w{SECRET[:3]}{n:04d}"} for n in range(40)),
             {"finish_reason": "stop"}]
    changed = [*words[:21], {"content": "x" * 8}, *words[22:]]  # the 21st word: tokens 40 and 41, in block 2
    altered = bodies[5]["messages"]
    directory = tmp_path / "identity"
    async with support.running() as stack:
        engine = stack.gateway.service.config.engine_url
        stack.fake.script = lambda body: Script(events=words)
        async with httpx.AsyncClient(timeout=10) as client:
            reference = await identity.measure(client, engine, bodies, "reference", directory)
            stack.fake.script = lambda body: Script(events=changed if body["messages"] == altered else words)
            change = await identity.measure(client, engine, bodies, "change", directory)
            with pytest.raises(identity.IdentityError):
                await identity.measure(client, engine, bodies[:15] + bodies[:1], "change", directory)
    generations = stack.fake.calls_of("generate")
    assert len(generations) == 48 and len({call.body["cache_salt"] for call in generations}) == 48
    assert (reference["same"], reference["failed"], reference["visible"]) == (16, 0, False)
    assert change["same"] == 15 and change["first_difference"][5] == 32 and not change["visible"]
    written = {path.name: path for path in directory.iterdir()}
    assert set(written) == {"key", "reference.json"}
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in written.values())
    kept = json.loads(written["reference.json"].read_text())
    assert kept["tokens"] == [80] * 16 and all(len(digests) == 5 for digests in kept["digests"])
    assert set(reference) == set(change) == {"side", "at", "block", "same", "of", "failed", "first_difference",
                                             "visible"}  # counts, positions and fixed words


# The reader-overlap flag.

def test_a_pass_that_overlapped_a_readers_request_is_flagged(tmp_path: Path) -> None:
    def row(at: float, cls: str, **fields: Any) -> str:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(at)) + f".{int(at % 1 * 1000):03d}Z"
        return json.dumps({"time": stamp, "event": "request", "route": "/v1/chat/completions", "class": cls,
                           "scope": "reader", **fields})

    t = 1_790_000_000.0
    (tmp_path / "gateway.jsonl.1").write_text(row(t + 5, "reader", total_ms=2000) + "\n")  # within the first pass
    (tmp_path / "gateway.jsonl").write_text("\n".join([
        row(t + 130, "reader", total_ms=40_000),  # began before the second pass ended
        row(t + 250, "internal", total_ms=9000),  # the harness's own
        row(t + 400, "reader"),  # a refusal, which has no duration: after every pass
        f"Traceback {SECRET}", json.dumps({"event": "request", "class": "reader", "note": SECRET}),
    ]) + "\n")
    requests = overlap.reader_requests(tmp_path)
    assert requests == [(t + 3, t + 5), (t + 90, t + 130), (t + 400, t + 400)]
    passes: list[dict[str, Any]] = [{"window": [t, t + 60]}, {"window": [t + 61, t + 100]},
                                    {"window": [t + 200, t + 300]}]
    assert overlap.flag(passes, requests) == 2
    assert [(entry["reader_overlap"], entry["reader_requests"]) for entry in passes] == [
        (True, 1), (True, 1), (False, 0)]


# The timing sweep.

BENCH = """#!{python}
import asyncio, json, sys
import httpx
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]
async def one(client):
    body = {{"model": "m", "messages": [{{"role": "user", "content": "synthetic"}}], "max_tokens": 8, "stream": True}}
    async with client.stream("POST", "http://127.0.0.1:" + value("--port") + "/v1/chat/completions", json=body) as r:
        async for _ in r.aiter_bytes():
            pass
async def main():
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client) for _ in range(min(2, int(value("--max-concurrency"))))))
asyncio.run(main())
result = {{"duration": 12.5, "completed": 6, "failed": 0, "output_throughput": 245.75, "median_ttft_ms": 812.4,
          "p99_itl_ms": 31.0, "spec_decode_per_position_acceptance_rates": [0.78, 0.62, 0.49],
          "generated_texts": ["{secret}"], "model_id": "/workspace/{secret}", "date": "20260927-120000",
          "{secret}_ms": 1.0}}
with open(value("--result-dir") + "/" + value("--result-filename"), "w") as file:
    json.dump(result, file)
"""


def test_the_sweeps_commands_are_the_runbooks_and_the_pinned_bench_takes_them(state: Path) -> None:
    base = card_at(state)
    planned = sweep.cells()
    assert len(planned) == 24 and [cell.name for cell in planned[:2]] == ["shared-4000-at1", "shared-4000-at2"]
    assert planned[-1] == sweep.Cell(16000, 8, shared=False)
    for cell in planned:
        argv = sweep.arguments(base, cell, state / "sweep")
        sweep.check(argv)

        def value(flag: str, argv: list[str] = argv) -> str:
            return argv[argv.index(flag) + 1]

        assert argv[:3] == [str(state / "vllm/bin/vllm"), "bench", "serve"] and value("--port") == "8092"
        assert int(value("--random-prefix-len")) + int(value("--random-input-len")) == cell.context
        assert value("--random-prefix-len") == (str(cell.context - 256) if cell.shared else "0")
        assert (value("--random-output-len"), value("--temperature"), value("--max-concurrency")) == (
            "512", "0", str(cell.at))
        assert int(value("--num-prompts")) == 3 * cell.at and "--ignore-eos" in argv
    argv = sweep.arguments(base, planned[0], state / "sweep")
    for wrong in ([*argv, "--save-detailed"], [*argv, "--seed", "8"], [*argv[:-3], "--result-filename"],
                  [part if part != "512" else "many" for part in argv],
                  [part for part in argv if part != "--ignore-eos"], ["python", *argv[1:]]):
        with pytest.raises(sweep.SweepError):
            sweep.check(wrong)


@pytest.mark.anyio
async def test_a_sweep_keeps_the_benchs_numbers_and_stops_a_context_the_cache_did_not_hold(state: Path) -> None:
    bench = state / "vllm/bin/vllm"
    bench.parent.mkdir(parents=True)
    bench.write_text(BENCH.format(python=sys.executable, secret=SECRET))
    bench.chmod(0o700)
    async with support.running() as stack:
        stack.fake.script = Script(first_delay_s=0.3)
        port = stack.gateway.service.config.engine_url.rsplit(":", 1)[1]
        base = card_at(state, ENGINE_PORT=port)
        planned = sweep.cells(contexts=(4000,), unshared=(), at=(1, 2, 3, 4))
        async with httpx.AsyncClient(timeout=10) as client:
            found = await sweep.sweep(client, base, state / "sweep", planned, sample_s=0.02)
    assert [(cell["cell"], cell.get("held"), cell.get("skipped")) for cell in found] == [
        ("shared-4000-at1", True, None), ("shared-4000-at2", True, None), ("shared-4000-at3", False, None),
        ("shared-4000-at4", None, "cache")]
    first = found[0]
    assert first["exit_code"] == 0 and first["metrics"]["generation_tokens"] > 0
    assert first["bench"] == {"duration": 12.5, "completed": 6, "failed": 0, "output_throughput": 245.75,
                              "median_ttft_ms": 812.4, "p99_itl_ms": 31.0,
                              "spec_decode_per_position_acceptance_rates": [0.78, 0.62, 0.49]}
    assert SECRET not in json.dumps(found)


def test_the_microbenchmarks_plan_the_31bs_shapes_without_torch(capsys: pytest.CaptureFixture[str]) -> None:
    assert microbench.main(["gemm", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    table = {"qkv_sliding": (16384, 5376, 50), "qkv_full": (20480, 5376, 10), "o_sliding": (5376, 8192, 50),
             "o_full": (5376, 16384, 10), "gate_up": (43008, 5376, 60), "down": (5376, 21504, 60)}  # docs/speed.md
    assert {shape["name"]: (shape["n"], shape["k"], shape["layers"]) for shape in plan["shapes"]} == table
    assert plan["kernels"] == ["marlin", "humming", "cutedsl"]
    verification = {requests * (drafts + 1) for requests in range(1, 9) for drafts in range(2, 6)}
    assert set(range(1, 9)) | verification | {64, 1024} <= set(plan["rows"]) and max(plan["rows"]) == 1024
    assert min(plan["large"]["values"]) >= 30000
    assert microbench.main(["attention", "--dry-run"]) == 0
    cases = json.loads(capsys.readouterr().out)["cases"]
    assert {(case["kernel"], case["layer"], case["q"]) for case in cases} == {
        *(("triton", "full", q) for q in (1, 4, 5, 6)), ("triton", "sliding", 4), ("triton", "sliding", 5),
        ("xqa", "sliding", 4), ("xqa", "sliding", 5), ("native", "full", 1)}
    assert {case["context"] for case in cases} == {4000, 16000, 40000} and max(case["batch"] for case in cases) == 8
    assert "torch" not in sys.modules and "vllm" not in sys.modules
    with pytest.raises(microbench.PlanError):
        microbench.check({"kind": "gemm", "shapes": [{"name": "odd", "n": 24, "k": 20, "parts": [24]}], "rows": [1]})
    with pytest.raises(SystemExit):
        microbench.main(["gemm", "--kernels", "cutlass", "--dry-run"])


# The result, its report and the commands.

def test_a_result_holds_counts_times_and_fixed_words_only(tmp_path: Path) -> None:
    result = {"harness": report.VERSION, "change": "synthetic=0.8,0.6", "timing_only": True, "outcome": "ready",
              "load_s": 201.5, "started": "2026-09-27T06:00:00Z", "versions": {"model_revision": "24ac337e"},
              "proof": {"ok": True, "missing": [], "unexpected": [], "codes": ["arg_max_num_seqs"]},
              "sweep": [{"cell": "shared-4000-at1", "held": True, "exit_code": 0, "acceptance": [0.8, 0.6],
                         "bench": {"output_throughput": 101.5, "median_tpot_ms": 9.5}},
                        {"cell": "shared-4000-at2", "skipped": "cache"}], "flagged": 0}
    path = report.write(tmp_path, result)
    assert path.stat().st_mode & 0o777 == 0o600 and report.read(path) == result
    lines = report.text(result).splitlines()
    assert lines[0] == "change synthetic=0.8,0.6: ready after 201.5 s, timing only"
    assert "acceptance 0.8,0.6" in lines[2] and "skipped" in lines[3]
    for leak in ({"outcomes": [{"text": f"{SECRET} and more words"}]}, {f"{SECRET} key": 1}, {"tokens": b"\x00"},
                 {"identity": {"first": list(range(100))}}):
        with pytest.raises(report.ResultError) as refused:
            report.write(tmp_path / "other", result | leak)
        assert SECRET not in str(refused.value)
    assert not (tmp_path / "other").exists()


@pytest.mark.anyio
async def test_a_timing_only_change_runs_no_workload_and_a_workload_needs_the_pack(state: Path) -> None:
    base = card_at(state)
    for options in (cli.Options(change="synthetic=0.5", workloads=("turns",), pack=state),
                    cli.Options(change="reference", identity="reference")):
        with pytest.raises(cli.RunError):
            await cli.run(base, options)
    assert not (state / cli.RUNS).exists()
    argv = cli.microbench_argv(base, "gemm", "marlin")
    assert argv[0] == str(state / "vllm/bin/python") and argv[2:] == ["gemm", "--kernels", "marlin"]


def test_the_cheap_cards_manifest_is_the_cards_with_e2b_and_its_assistant_pinned() -> None:
    def read(name: str) -> dict[str, str]:
        lines = (card.CODE / "card" / name).read_text().splitlines()
        return dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))

    cheap, manifest = read("cheap-card.env"), read("manifest.env")
    changed = {name for name in manifest if cheap[name] != manifest[name]}
    assert cheap.keys() == manifest.keys()
    assert changed == {"MODEL_REPO", "MODEL_REVISION", "MODEL_FILES", "TOKENIZER_REPO", "TOKENIZER_REVISION",
                       "TOKENIZER_FILES", "MODEL_ALIAS", "DRAFTER_REPO", "DRAFTER_REVISION", "DRAFTER_FILES"}
    assert cheap["MODEL_REPO"] == "google/gemma-4-E2B-it" and cheap["MODEL_REVISION"].startswith("3e22461f")
    assert cheap["DRAFTER_REPO"] == "google/gemma-4-E2B-it-assistant"
    assert cheap["DRAFTER_REVISION"].startswith("2d874ef7")
    for key in ("MODEL_FILES", "TOKENIZER_FILES", "DRAFTER_FILES"):
        pins = dict(pin.split(":") for pin in cheap[key].split(","))
        assert all(len(digest) == 64 and int(digest, 16) >= 0 for digest in pins.values())
        assert {"config.json"} <= pins.keys() and (key == "TOKENIZER_FILES") == ("tokenizer.json" in pins)

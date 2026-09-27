"""The throughput harness of docs/speed.md's item 1, off the card: its reader of /metrics on canned text, the A/B
launcher's commands, the workloads built from a synthetic pack, the identity set's comparison, the reader-overlap
flag, the timing sweep's arguments, the microbenchmarks' plans, and the result and report, which hold counts, times
and fixed words only."""

from __future__ import annotations

from simple_serving.throughput import metrics

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

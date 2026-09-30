# Concurrency sweep

The standalone tool from `sweep` at `b6d4919` is integrated on 2026-09-30. It measures an already running vLLM engine
with synthetic random prompts. It does not start, restart or configure the engine, and adds no runtime dependency.
The file can run as a package module or be copied alone to a card and run with the vLLM environment's Python.

## Plan without a card

```sh
uv run python -m simple_serving.sweep --plan --cache-tokens 100000
```

Here 100000 is a synthetic planning input. There is no historical capacity default. Every invocation requires a
positive `--cache-tokens`, including `--plan`. Planning reads no manifest or engine and launches no benchmark.
For a real run use the current engine's capacity, recorded as `kv_cache_tokens` in the launcher's `engine_measure`
rows. A previous boot's value may differ after enabling vision, changing MTP or changing cache memory settings.

The plan considers 24 levels: concurrency 2, 4, 3, 6, 8 and 1, for shared-prefix inputs of 16000, 4000 and 40000
tokens, then unshared 16000-token inputs. Each request asks for 512 output tokens. Each level uses at least four
prompts and twice its concurrency, with a distinct seed. Shared levels use one warmup request. A level is skipped
when its unshared occupancy estimate exceeds 95 percent of the supplied cache capacity.

Time estimates are a historical model, not measurements. Current dense prefix retention changes cache reuse and
warmup behavior. `--plan` neither establishes throughput nor proves the chosen levels fit the real engine's other
constraints. After a slower completed level, the remaining time estimates grow proportionally.

## Local regression check

```sh
uv run python -m pytest -q tests/test_sweep.py
```

The tests substitute a synthetic benchmark subprocess and a loopback metrics server. They check planning,
arguments, explicit capacity, deadlines, refusal to start on unknown or busy metrics, failures and child cleanup.
They do not run vLLM or measure GPU performance. The existing interference client remains separate: it exercises
long prefill during another request's decode through the gateway, rather than aggregate engine concurrency.

## An authorised engine measurement

Set `SWEEP_CACHE_TOKENS` to the current engine's recorded capacity and `SWEEP_END_EPOCH` to the already agreed
experiment deadline in Unix seconds. On a prepared card, a command has this form:

```sh
/workspace/simple-serving-card/vllm/bin/python /workspace/simple-serving/simple_serving/sweep.py \
  --cache-tokens "$SWEEP_CACHE_TOKENS" --until "$SWEEP_END_EPOCH"
```

`--manifest` defaults to the prepared text manifest, `--state` to the text card state. `--result-dir` selects the
fresh private artifact directory; an existing path is refused without overwriting or changing its permissions.
`--bench-cmd` substitutes the benchmark executable for local tests; the real default is that state's
`vllm/bin/vllm bench serve`. The dataset is always random, and no owner or tester story is supplied.
Only loopback engine requests are made. The pinned benchmark uses OpenAI completions, not the gateway's chat API.

Run only after other text work ends. Direct engine traffic bypasses gateway admission, cache salts and activity
tracking. It does not reset the gateway's 780-second idle timer or keep the card awake. Choose a deadline within the
remaining idle interval and external experiment deadline. The tool does not manage rentals, extend a trial guard
or stop billing.

Before a level, running and waiting gauges must be valid and zero. Peak in-flight requests above the requested
concurrency indicate overlap and stop further measurement. Sampling can miss brief or lower-load overlap; absence
of that signal does not establish exclusive access to the engine.

No new level starts unless its estimate fits before `--until`. A benchmark still running at that time is terminated,
with bounded process cleanup afterward. Only the benchmark's process group is terminated; the serving engine is
left running. The deadline is a benchmark cutoff, not a provider-side billing guarantee.

## Reports and privacy

The wrapper prints one JSON line per level and a final summary. Lines contain only fixed field names, fixed status
words, finite numbers and nulls. They report benchmark completion, duration, throughput, TTFT, token/cache/preemption
counter deltas, accepted drafts by position, peak running/waiting requests and cache usage. Cache usage is a fraction
from 0 to 1. The final summary includes the supplied `cache_tokens`. Missing optional counters are omitted rather
than reported as zero. Lines contain no prompts, generated text, metric label values, keys or private paths.

The private result directory holds the upstream benchmark's JSON and `.log` files. The wrapper's allowlist does
not filter those raw artifacts; an upstream error can include paths or request details. They must remain private
and must not be published as the numeric report. Benchmark requests remain synthetic. The child runs offline and
without the selected API credentials named in the script. Environment proxies are bypassed for loopback metrics
and benchmark requests.

Exit 0 means the planned sweep ended or reached its deadline. Exit 1 means a benchmark failed. Exit 2 means input
or engine state did not permit measurement; interruption exits 130 after cleanup. A successful local fake run
establishes execution and cleanup only.
GPU throughput, overlap under real traffic, capacity and the value of tuning remain unmeasured here.

The vLLM 0.30.0 interfaces checked for this integration are [benchmark options and results](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/benchmarks/serve.py),
[random datasets](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/benchmarks/datasets/datasets.py),
[scheduler metrics](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/metrics/loggers.py) and
[draft metrics](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/spec_decode/metrics.py).

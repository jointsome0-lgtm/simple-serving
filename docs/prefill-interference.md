# Prefill interference measurement

The first question for prefill/decode disaggregation is whether a long new prompt delays an answer already being
generated enough to justify another GPU and KV transfer. This client measures delivery pauses through the existing
text gateway. It uses synthetic prompts and fresh scopes; it does not change the engine or manage a rental.

## Local demonstration

From the repository root:

```sh
uv run python -m simple_serving.interference
```

With no arguments, the command starts a local fake, runs one round and prints JSON. The fake deliberately pauses
the first request while a cold long prompt arrives, then reuses the exact warmed prompt. This verifies the scenario
and report, without a model, owner configuration or GPU. The tests also cover a full freeze: no answer chunks arrive
while the second request runs, then the first answer resumes after it finishes.

Use `--out /tmp/prefill-demo.json` to save a new owner-only file. Existing files are never overwritten. Exit status
is 0 for a complete, unconfounded scenario and 2 for a failed or inconclusive run.

## Measure an existing text engine

Start a separate, quiet text gateway and its tunnels using the [runbook](runbook.md). The benchmark does not start,
resume, stop or restore serving. It refuses a busy initial snapshot. The client key must allow `reader` and
`internal`, including counting; the control key must read state and both metric routes. Configuration uses the
existing [CLI format](runbook.md#the-command), with owner-only permissions.

Run explicitly against loopback tunnel endpoints, choosing a new report filename each time:

```sh
uv run python -m simple_serving.interference run \
  --config ~/.config/simple-serving/config.json \
  --out /tmp/prefill-baseline.json
```

The defaults use public port 8080 and control port 8081. Override them with `--public http://127.0.0.1:PORT` and
`--control http://127.0.0.1:PORT`. Redirects and environment proxies are disabled.

The live defaults are three rounds, a long prompt targeted at 16000 counted input tokens, a 1024-token output
limit for the first request and a one-token output limit for the second. The second request starts after four
nonempty content or reasoning chunks from the first. Metadata and usage events do not trigger it. Each round
rotates these cases:

1. `alone`: a reader generates without an injected request.
2. `cold`: a long prompt arrives under a fresh scope during that reader's generation.
3. `warm`: each long request is first generated under its own scope, then repeated exactly during the reader.

Sizing uses the gateway's input-token endpoint, with at most 32 bounded count calls. Counting does not prefill
the KV cache. The chosen long prompt is the largest tested candidate within the target; its measured generation
usage must match that count. The long input target plus its output limit must fit the advertised context.
The first request's output limit is checked here; the gateway checks its prompt plus output budget during generation.
The command checks ready state and admission capacity before generating. Default concurrency 2 means one reader
plus one `internal` request. For four requests, the current internal limit is too small; use four reader requests
on a key that permits them:

```sh
uv run python -m simple_serving.interference run \
  --config ~/.config/simple-serving/config.json \
  --out /tmp/prefill-four-readers.json \
  --concurrency 4 --aggressor-class reader
```

The other dimensions are `--rounds`, `--long-tokens`, `--victim-tokens`, `--aggressor-tokens` and `--trigger-chunks`.
The whole run has a 600-second default budget, adjustable with `--max-seconds` up to 1800 seconds. A live request
has a 30-second first-content deadline; each case, including warm priming, has a 180-second deadline. Failures
cancel and await the other requests, without retrying. Streams, count replies and metric scrapes have byte limits.

Record the GPU, model/tokenizer revisions, vLLM version, attention backend, MTP, KV dtype, prefix-cache settings,
batching limits and gateway limits alongside every live result. Keep these conditions fixed when repeating a
baseline. Compare a changed batching or cache setting in a separate controlled run; this command changes none of
them. It has no automatic SLO verdict. Agree on the acceptable reader pause and extra GPU cost before using the
measurement to choose disaggregation.

## Read the report

The report contains options, actual prompt counts, per-request usage, client TTFT, total time, and gap summaries
with sample count, p50, p95, p99 and maximum. `post_injection_gap_ms` includes a gap spanning the injection instant.
HTTP overlap and victim chunk counts show whether requests overlapped and the first answer continued after the
second started. A freeze throughout the second request remains a valid observation if the first answer resumes.

`started_ms`, `ended_ms` and `injection_ms` are relative to the trial runner's clock. Add `run_offset_ms` to align
them with diagnostic sample `at_ms`. Scrapes also record their duration; their readings are not simultaneous.
The control metrics are sampled before and after each case and approximately every half second during the run.
Only selected numeric families and fixed class names enter the report. It stores no keys, scopes, model alias,
URLs, prompt text, answer text, raw metric labels or raw errors.

`warm_cache_confirmed` means the measured warmed requests each reported more than zero cached prompt tokens.
Inspect their actual `cached_tokens` and `prompt_tokens` to assess how much was reused. A missing cache field stays
unknown; it is never converted to zero. A cold cache hit, unconfirmed warm reuse, early completion or absence of
HTTP overlap makes that case inconclusive. Counter deltas for `warm` include its priming requests, so their totals
cannot be compared directly with the cold case's totals.

Diagnostic flags identify observed generation admission waiting, preemption, additional gateway requests, counter decreases,
gateway restart, missing evidence and the 1200-snapshot background sampling cap. A flagged successful case becomes
inconclusive. Extra gateway requests indicate contamination; aggregate counters cannot establish exclusive engine
ownership or detect all traffic sent directly to vLLM. Missing preemption evidence is flagged rather than treated
as zero. Missing active/waiting gauges prevent the initial quiet check and fail before generation.

These are gaps between delivered SSE chunks, which may contain several tokens, especially with MTP. They are not
per-token GPU ITL. Client TTFT includes upload, token counting, gateway admission, engine queuing and transport.
The gateway's count stage is tokenizer work; engine prefill histograms are separate aggregate evidence. Short runs
and their p99 values describe the observed samples, without establishing a stable latency distribution or a GPU
cause. Repeat controlled GPU measurements before drawing that conclusion.

## Scope and remaining work

This tool selectively reuses the synthetic workload, sizing and numeric metric-selection ideas from
`held-criterion` at `a898060`. That branch's A/B launcher and throughput work remain separate and retained. Its
launcher takes over the serving pair; this client has no such lifecycle operation.

Local fake checks exercise pauses, cache reporting, framing, failures, cleanup and the actual gateway contract.
No real GPU measurement has been made for this scenario. Splitwise-style placement, KV transport, separate
prefill/decode workers and connector compatibility remain unimplemented. The next decision needs a measured
reader-pause baseline and a comparison with batching and cache settings on one card.

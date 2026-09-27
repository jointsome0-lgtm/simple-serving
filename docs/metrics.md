# Metrics

The catalog of what the monitoring reads, as the owner asked on 2026-09-27: the gateway's own metrics, the names of
vLLM's that the dashboards and the rules use, and those of the optional exporters. It holds for any gateway that
implements the contract: one written in another language serves the same names, labels and meanings, and
`monitoring/` depends on nothing else. `tests/test_monitoring.py` checks it both ways. The gateway serves each family
of the gateway's tables below with the type and the labels given, and no other. The dashboards, the rules and
`monitoring/prometheus/prometheus.yml` use no name outside this file and select no label its name does not have.
vLLM's names are also checked against vLLM's source, with their types, when `SIMPLE_SERVING_VLLM_SOURCE` names a git
checkout of the version that `card/manifest.env` pins:

    SIMPLE_SERVING_VLLM_SOURCE=/path/to/vllm uv run pytest tests/test_monitoring.py

## Where they are served

- `GET /metrics` on the control listener: the gateway's own, in Prometheus's text format 0.0.4
  (`text/plain; version=0.0.4; charset=utf-8`).
- `GET /metrics/engine` on the control listener: the engine's own `/metrics`, relayed as it is, with its content
  type, when the engine answers 200 within 5 s and 8 MB; otherwise 503 `engine_unavailable`. The engine listens on
  the card's loopback and its port is not forwarded, since that would open its API without a key (contract section 1).
- Both take the metrics key, which reads the metrics and nothing else, or the control key. Without a known key they
  answer 401 `unauthorized`, to a key without either right 403 `forbidden`, and the public listener does not have
  them (404). The control listener listens on the card's loopback, and the owner's machine reaches it only through
  the tunnel of `cli up`, at 127.0.0.1:8081.
- A scrape is not work of ours. It does not reset the idle interval (contract section 8), holds nothing awake and
  starts no generation: the monitoring never probes the model. It writes one row to the gateway's log, as every
  request does.

## Labels

Every label value comes from a fixed set. No label holds a key, a key's label, a scope, a request's id or anything
of a body (contract section 10).

| Label | Values |
| --- | --- |
| `listener` | `public`, `control` |
| `route` | the paths of contract section 3: `/v1/chat/completions`, `/v1/chat/completions/input_tokens`, `/v1/models`, `/v1/state`, `/v1/control/drain`, `/v1/control/sleep`, `/v1/control/open`, `/metrics`, `/metrics/engine`; `other` for any other path |
| `class` | `reader`, `agent`, `internal`, `external`; `none` for a request refused before its class was known |
| `outcome` | the outcomes below |
| `code` | an error code of contract section 9, or `none` |
| `status` | in `simple_serving_http_responses_total`, the HTTP status sent (`200`, `202`, `400`, `401`, `403`, `404`, `409`, `413`, `429`, `500`, `503`, `504`), or `none` when none was sent; in `simple_serving_status`, a service status of `/v1/state` (`starting`, `ready`, `draining`, `drained`, `failed`) |

A request is refused before its class is known when its key is missing or unknown (401 `unauthorized`), or when the
service does not serve as its key is checked: 503 `starting`, `draining`, `drained`, or `engine_unavailable` while
the service has failed.

## Outcomes

Each generation (`POST /v1/chat/completions`) and each count (`POST /v1/chat/completions/input_tokens`) is counted
once in `simple_serving_requests_total`, as its log row is written, with one outcome and the code of contract section
9 that it answered or began to answer. The outcome comes from the row: the code, whether the request was accepted,
and which stop cancelled it. A stream that fails after its 200, with an error event, is an `engine_error`, not a
success.

| Outcome | Codes | What happened |
| --- | --- | --- |
| `success` | `none` | The answer went out whole: for a generation, its usage chunk and `[DONE]`. |
| `rejected` | `invalid_request`, `unsupported_field`, `limit_exceeded`, `context_limit`, `unauthorized`, `class_not_allowed`, `scope_not_allowed`, `body_too_large`, `queue_full`, `timeout` | The request, its key or a cap refused it. `timeout` here is a body that did not arrive before acceptance. |
| `unavailable` | `starting`, `engine_unavailable`, `none` | The service could not take it: it was starting, or the engine was unavailable before acceptance. `none` is an accepted request that the gateway's own shutdown cut. |
| `draining` | `draining`, `drained` | A drain or a sleep refused it, or stopped it while it waited. |
| `timeout` | `timeout` | An accepted request reached its class's wall time (contract section 7), whether it was waiting, streaming or sending its end. |
| `engine_error` | `engine_unavailable` | The engine failed an accepted request: an error event after the 200, an engine that went away, or a change of engine under it. |
| `internal_error` | `internal_error` | The gateway failed. |
| `client_disconnect` | `none` | The client closed its connection before the answer ended. |

The HTTP status is a separate counter, `simple_serving_http_responses_total`, over every request on both listeners:
there a stream that failed after its 200 is a 200.

## The gateway's metrics

Every name begins with `simple_serving_`. Counters and histograms start at 0 with each gateway process, which
`simple_serving_start_time_seconds` names; Prometheus's `rate()` and `increase()` take the reset in their stride.
Every series of the counters and the histograms is served from the start, at 0, so that `increase()` sees the first
event of each: the series of `simple_serving_requests_total` are the two routes, the five classes and the pairs of
outcome and code of the table above.

### Counters

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `simple_serving_requests_total` | counter | `route`, `class`, `outcome`, `code` | Generations and counts, each once, at its end, by its outcome. `route` is one of the two routes. |
| `simple_serving_http_responses_total` | counter | `listener`, `route`, `status` | Every request on either listener, by the HTTP status sent. |
| `simple_serving_input_tokens_total` | counter | `class` | Input tokens, as the gateway counted them, of generations whose engine stream ended well, whether or not the client took the end. |
| `simple_serving_output_tokens_total` | counter | `class` | Tokens generated, reasoning included, in those generations. |
| `simple_serving_cached_tokens_total` | counter | `class` | Input tokens that the engine read from its prefix cache, where it said so, in those generations. |
| `simple_serving_count_mismatches_total` | counter | `class` | Those generations whose `prompt_tokens` from the engine differed from the gateway's count (contract section 5). |

### Histograms

In seconds, with the upper bounds 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600
and 900, and `+Inf`: from the few milliseconds of a free place to the wall times of contract section 7. Each is served
as `_bucket`, `_sum` and `_count`, for the four classes; a request is accepted only once its class is known. A stage
is observed when it ends, and only if it ends. A request stopped before the end of a stage observes nothing for it.
The times start at acceptance, once the body has arrived and passed its checks (contract section 11).

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `simple_serving_count_place_wait_seconds` | histogram | `route`, `class` | From entering the count stage to a count place: the gateway's count queue. `route` tells a count from a generation's count stage. |
| `simple_serving_count_seconds` | histogram | `route`, `class` | The engine's count of the input, once it has a count place. |
| `simple_serving_generation_place_wait_seconds` | histogram | `class` | From asking for a generation place, after the count and the checks of the input, to having one: the gateway's generation queue. |
| `simple_serving_wait_seconds` | histogram | `class` | From acceptance to a generation place, when the gateway sends the generation to the engine: the log row's `wait_ms`, the count stage included. |
| `simple_serving_time_to_first_token_seconds` | histogram | `class` | From acceptance to the first token received from the engine, reasoning included: the log row's `first_token_ms`, observed only when a token arrived. |
| `simple_serving_total_seconds` | histogram | `class` | From acceptance to the end of the engine's stream, for streams that ended well: the log row's `total_ms`. |

The log row keeps its times as before, and differs from these histograms in one way: a request that ended before a stage
writes the stage's time as its end. A stream that ended without any token has a `first_token_ms` in its row, the moment
its finish arrived, and no observation here; a stopped request has all three times in its row.

### Gauges

Read from the service at each scrape.

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `simple_serving_start_time_seconds` | gauge | | When the gateway process started, in Unix time: a new value is a new boot. |
| `simple_serving_status` | gauge | `status` | 1 for the service's status of `/v1/state`, 0 for the four others. |
| `simple_serving_sleep_requested` | gauge | | 1 once a sleep has begun: the instance is being stopped. |
| `simple_serving_drain_generation` | gauge | | The drain generation of `/v1/state`. |
| `simple_serving_generations_active` | gauge | `class` | Generations that the gateway has sent to the engine and that have not ended: contract section 7's Active. |
| `simple_serving_generations_waiting` | gauge | `class` | Generations waiting in the gateway for a generation place. |
| `simple_serving_counts_active` | gauge | `class` | Counts in a count place, a generation's count stage included. |
| `simple_serving_counts_waiting` | gauge | `class` | Counts waiting in the gateway for a count place, a generation's count stage included. |
| `simple_serving_accepted_in_flight` | gauge | `route` | Accepted generations and counts that have not answered yet: what a drain waits for. |
| `simple_serving_ours_in_flight` | gauge | | Requests of reader, agent and internal between their authorization and their end: what holds the card awake. |
| `simple_serving_idle_seconds` | gauge | | Seconds without work of ours since the last ended; 0 while one runs, and before the first ready. |
| `simple_serving_idle_timeout_seconds` | gauge | | The idle interval of contract section 8: the service falls asleep when `simple_serving_idle_seconds` reaches it. |
| `simple_serving_generation_active_limit` | gauge | `class` | Contract section 7's Active cap of each class. |
| `simple_serving_generation_waiting_limit` | gauge | `class` | Contract section 7's Waiting cap of each class. |
| `simple_serving_generation_shared_limit` | gauge | | The places that agent, internal and external share. |
| `simple_serving_count_active_limit` | gauge | | The count places of contract section 5. |
| `simple_serving_wall_time_limit_seconds` | gauge | `class` | Contract section 7's wall time of each class. |

## The gateway's queues and the engine's

A generation passes three queues. It waits in the gateway for a count place (`simple_serving_counts_waiting`), is
counted (`simple_serving_counts_active`), then waits in the gateway for a generation place
(`simple_serving_generations_waiting`). With a place it is sent to the engine and is Active
(`simple_serving_generations_active`); there it may still wait in vLLM's own scheduler queue
(`vllm:num_requests_waiting`) before it runs (`vllm:num_requests_running`), for instance while the KV cache is full.
The first two queues are the gateway's and are bounded by contract sections 5 and 7; the third is vLLM's.

## The gateway's times, not the reader's

The gateway's times start at acceptance and end at an event in the gateway: a place, the first token received from
the engine, or the end of the engine's stream. A reader waits longer and sees something else. Before acceptance come
Telegram, the bot's own work, the bot's queue, the tunnel and the upload of the body; after the gateway's event come
the way back through the tunnel, the bot, its edits of the message and Telegram. The first token counts reasoning,
which the reader does not see. No panel or rule here measures what the reader sees.

vLLM's times start later still, when vLLM's server takes the request from the gateway, after the gateway's wait.

## vLLM's metrics

The card runs vLLM 0.30.0 (`VLLM_VERSION` in `card/manifest.env`), and the gateway relays its `/metrics` at
`/metrics/engine`. The dashboards and the rules read the names below, defined in `vllm/v1/metrics/loggers.py` and
`vllm/v1/spec_decode/metrics.py` of vLLM's tag v0.30.0, commit ced6857afa0ea7b2e3f0846a62e1394e90f15607. Each has
the labels `model_name`, the name vLLM serves the model under, and `engine`, the engine's index, and some have one
more. vLLM serves a counter with `_total`, as named here, and a histogram as `_bucket`, `_sum` and `_count`.

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `vllm:num_requests_running` | gauge | `model_name`, `engine` | Requests in the engine's running batches. |
| `vllm:num_requests_waiting` | gauge | `model_name`, `engine` | Requests in vLLM's own waiting queue: the engine's queue, not the gateway's. |
| `vllm:num_requests_waiting_by_reason` | gauge | `model_name`, `engine`, `reason` | The same by reason: `capacity` waits for room in the batch or the KV cache, `deferred` for a passing constraint. |
| `vllm:kv_cache_usage_perc` | gauge | `model_name`, `engine` | The fraction of the KV cache in use, 1 for all of it. |
| `vllm:num_preemptions_total` | counter | `model_name`, `engine` | Preemptions: requests the scheduler took out of the batch for lack of KV cache. |
| `vllm:prefix_cache_queries_total` | counter | `model_name`, `engine` | Tokens looked up in the prefix cache. |
| `vllm:prefix_cache_hits_total` | counter | `model_name`, `engine` | Of those, the tokens found. |
| `vllm:prompt_tokens_total` | counter | `model_name`, `engine` | Prefill tokens processed. |
| `vllm:generation_tokens_total` | counter | `model_name`, `engine` | Tokens generated. |
| `vllm:request_success_total` | counter | `model_name`, `engine`, `finished_reason` | Requests that finished, by reason: `stop`, `length`, `abort`, `error` or `repetition`. |
| `vllm:spec_decode_num_drafts_total` | counter | `model_name`, `engine` | Drafts of the multi-token prediction. |
| `vllm:spec_decode_num_draft_tokens_total` | counter | `model_name`, `engine` | Tokens drafted. |
| `vllm:spec_decode_num_accepted_tokens_total` | counter | `model_name`, `engine` | Drafted tokens accepted. |
| `vllm:time_to_first_token_seconds` | histogram | `model_name`, `engine` | vLLM's time to the first token. |
| `vllm:inter_token_latency_seconds` | histogram | `model_name`, `engine` | The time between tokens. |
| `vllm:request_time_per_output_token_seconds` | histogram | `model_name`, `engine` | The mean time per output token of each request. |
| `vllm:e2e_request_latency_seconds` | histogram | `model_name`, `engine` | vLLM's time of a whole request. |
| `vllm:request_queue_time_seconds` | histogram | `model_name`, `engine` | The time a request spent waiting in vLLM's queue. |
| `vllm:request_prefill_time_seconds` | histogram | `model_name`, `engine` | The time a request spent in prefill. |
| `vllm:request_decode_time_seconds` | histogram | `model_name`, `engine` | The time a request spent in decode. |
| `vllm:request_prompt_tokens` | histogram | `model_name`, `engine` | The prompt's length in tokens, a request each. |
| `vllm:request_generation_tokens` | histogram | `model_name`, `engine` | Tokens generated, a request each. |

The speculative counters exist only while the drafter is on (`MTP_SPECULATIVE_TOKENS` in `card/manifest.env`). The
acceptance rate is `rate(vllm:spec_decode_num_accepted_tokens_total[5m]) /
rate(vllm:spec_decode_num_draft_tokens_total[5m])`, and the mean acceptance length, the tokens each step yields,
`1 + rate(vllm:spec_decode_num_accepted_tokens_total[5m]) / rate(vllm:spec_decode_num_drafts_total[5m])`.

## Prometheus's own series

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `up` | gauge | `job`, `instance` | 1 when the last scrape of a target succeeded. 0 for the gateway and the engine whenever no tunnel runs: a stopped rental is 0 and is not an alert. |
| `ALERTS` | gauge | `alertname`, `alertstate`, `severity` | 1 for each alert pending or firing, which the rules write. |

## The exporters' metrics, optional

Not installed: nothing on the card serves these yet, and the tunnel does not forward them (README, "Monitoring").
The machine row of the overview names them so that it fills once they run; until then it shows no data and raises no
alert. The names are those that node_exporter and utkuozdemir's nvidia_gpu_exporter document, unchecked against a
running exporter here.

| Name | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `node_cpu_seconds_total` | counter | `cpu`, `mode` | CPU time by mode. In a container on a rented host, the host's CPUs. |
| `node_memory_MemTotal_bytes` | gauge | | The memory the kernel sees, the host's in a container. |
| `node_memory_MemAvailable_bytes` | gauge | | Of that, the memory available. |
| `node_filesystem_size_bytes` | gauge | `device`, `fstype`, `mountpoint` | A filesystem's size. |
| `node_filesystem_avail_bytes` | gauge | `device`, `fstype`, `mountpoint` | A filesystem's free space. |
| `node_network_receive_bytes_total` | counter | `device` | Bytes received. |
| `node_network_transmit_bytes_total` | counter | `device` | Bytes sent. |
| `nvidia_smi_utilization_gpu_ratio` | gauge | `uuid` | The GPU's utilization, 0 to 1. |
| `nvidia_smi_memory_used_bytes` | gauge | `uuid` | The GPU memory in use. |
| `nvidia_smi_memory_total_bytes` | gauge | `uuid` | The GPU memory. |
| `nvidia_smi_temperature_gpu` | gauge | `uuid` | The GPU's temperature in °C. |
| `nvidia_smi_power_draw_watts` | gauge | `uuid` | The GPU's power draw. |

# Contract v1 (draft)

The HTTP API of simple-serving, version 1. Status: draft, nothing is implemented. The bot's side is in
simple-story-chat (`local/model.ts`, `local/llama.ts`); section 11 lists what it sends to llama-server today.

## 1. Scope

- One text model behind a gateway. The gateway is FastAPI, the engine is vLLM, both on one rented card. Clients see
  only the gateway. The engine, its metrics and its own API listen on loopback.
- Version 1 does text chat only. No tools, no images, no audio, no logprobs, no `n > 1`, no LoRA adapters.
- Clients are simple-story-chat (readers' turns, agent turns, our eval and probes) and outside users with keys.

## 2. Keys and classes

Every request carries `Authorization: Bearer <key>`. The gateway's configuration maps each key to a label, used in logs
instead of the key, a set of allowed classes and a default class.

| Class      | Who                                   | Order   | Preempted by the engine | Keeps the card awake |
|------------|---------------------------------------|---------|-------------------------|----------------------|
| `reader`   | a person's turn in the bot            | first   | never                   | yes                  |
| `agent`    | a turn of the bot's agent interface   | second  | may be paused, not cut  | yes                  |
| `internal` | our eval and probes                   | third   | may be paused           | open question 1      |
| `external` | outside keys                          | last    | may be paused           | no                   |

- A request may name its class in `X-Simple-Serving-Class`. A class the key does not allow is refused with 403. An
  outside key allows `external` only, so an outside client can never ask for `reader`.
- The gateway turns the class into the engine's priority. Clients never set engine priority: the gateway removes any
  priority field from the body and any engine priority header before it forwards a request.
- The engine may pause a running request of a lower class to make room for a higher one and resume it later. It does
  not cancel it. Only limits (section 7), a drain (section 8) or the client cancel a request.
- The bot keeps its own scheduler: turns, holders, yielding work prepared ahead. The classes above only order the bot's
  calls against everyone else's.

## 3. Routes

| Route                                  | Method | Keys                  | Public |
|----------------------------------------|--------|-----------------------|--------|
| `/v1/chat/completions`                 | POST   | any                   | yes    |
| `/v1/chat/completions/input_tokens`    | POST   | any                   | yes    |
| `/v1/models`                           | GET    | any                   | yes    |
| `/v1/state`                            | GET    | any                   | yes    |
| `/v1/control/drain`, `/v1/control/open`| POST   | the control key only  | no     |

Public means reachable from the internet through HTTPS. The control routes are served on loopback and reached over the
SSH tunnel. Any other path is 404.

## 4. Generation: `POST /v1/chat/completions`

### Request

The request is OpenAI chat completions, streaming only. Accepted fields:

| Field                                  | Rule                                                              |
|----------------------------------------|-------------------------------------------------------------------|
| `model`                                | the exact alias from `/v1/models`                                 |
| `messages`                             | `system`, `user`, `assistant`; `content` is a string              |
| `max_tokens`                           | required, at most the class limit                                 |
| `stream`                               | must be `true`                                                    |
| `stream_options.include_usage`         | must be `true`                                                    |
| `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` | optional sampling              |
| `seed`                                 | optional                                                          |
| `response_format`                      | optional, `{"type": "json_schema", "json_schema": {...}}`         |
| `chat_template_kwargs.enable_thinking` | optional, the gateway's default is `false`                        |

Any other field is refused with 400 `unsupported_field`, so a client that drifts from the contract fails in tests, not
silently in production.

### Response

`Content-Type: text/event-stream`. Every event is one `data:` line with a JSON chunk, UTF-8, and the stream ends with
`data: [DONE]`.

- Content chunks have exactly one choice, `index: 0`, with `delta.content`. If the model still writes reasoning, it
  comes in `delta.reasoning_content` and is counted apart from the text.
- `finish_reason` is `stop` or `length` and arrives exactly once, in the last content chunk.
- The last chunk before `[DONE]` has `choices: []` and `usage`:
  - `prompt_tokens`: equal to what `input_tokens` returns for the same body.
  - `completion_tokens`.
  - `prompt_tokens_details.cached_tokens`: optional; when absent it is unknown, not zero.
  - `simple_serving`: the gateway's own measurements in milliseconds, `queue_ms`, `first_token_ms` and `total_ms`.
    These are measured at the gateway, not the engine's internal timings.
- `model` in every chunk is the alias.
- A stream that ends without `finish_reason` and `[DONE]` is broken. A client must not treat its text as a finished
  answer.

## 5. Counting: `POST /v1/chat/completions/input_tokens`

The body is the same as for generation. The gateway counts it with the tokenizer and chat template the engine uses
and answers `{"input_tokens": n}`, `n > 0`. For the same body, `n` equals the `prompt_tokens` that generation reports.
The test on the card checks this equality with the pinned template.

## 6. Models and state

`GET /v1/models` lists one model: `{"object": "list", "data": [{"id": "<alias>", "object": "model",
"max_model_len": <context tokens>}]}`.

`GET /v1/state`:

```json
{"contract": "1", "status": "ready", "model": "<alias>", "context_tokens": 65536}
```

`status` is `starting`, `ready`, `draining` or `drained`. For the control key the answer also has `in_flight` and
`queued`, each counted by class.

## 7. Limits

Each class has a cap on requests running at once, on requests queued, on input tokens, on `max_tokens` and on wall
time. An outside key also has its own caps on requests running and queued. Requests per minute alone do not protect
readers, because one long prompt costs more than many short ones.

| Class      | Running | Queued | Input tokens | Output tokens | Wall time |
|------------|---------|--------|--------------|---------------|-----------|
| `reader`   | measure | measure| context      | measure       | measure   |
| `agent`    | measure | measure| context      | measure       | measure   |
| `internal` | measure | measure| measure      | measure       | measure   |
| `external` | measure | measure| measure      | measure       | measure   |

The numbers come from the synthetic load test on the card. The rule is that the caps of `agent`, `internal` and
`external` together leave room for the readers' peak. A request over a cap is refused with 429 `queue_full` or 400
`context_limit`, and it is never queued behind the cap.

## 8. Stopping the card

One controller stops the card: the bot's GPU control (`local/gpu.ts` in simple-story-chat). A paused card runs
nothing, so the service cannot wake itself.

1. The controller calls `POST /v1/control/drain`. From that moment every new request gets 503 `draining`.
2. Requests of `reader`, `agent` and `internal` that are running finish. `external` requests that are running are
   cancelled at once, and their streams end without `[DONE]`.
3. The controller polls `/v1/state` until it reads `drained`, with nothing running and nothing queued.
4. The controller stops the instance through the Vast API and reads the status back.

A drain that must be undone is undone with `POST /v1/control/open`. A service that starts is `starting` until the
engine answers, then `ready`. The guard on the card deletes the instance at the rental's deadline whatever the bot and
the service do; clients then see connection errors.

Checking that the engine is idle and then stopping the card is not enough, because a request can arrive between the
check and the stop. The drain closes that gap.

## 9. Errors and cancellation

The status carries the meaning, and the body is `{"error": {"code": "<code>"}}` with no text copied from the request.

| Status | Code                                       | When                                       |
|--------|--------------------------------------------|--------------------------------------------|
| 400    | `invalid_request`, `unsupported_field`     | the body breaks this contract              |
| 400    | `context_limit`                            | input plus `max_tokens` exceeds the context|
| 401    | `unauthorized`                             | no key or an unknown key                   |
| 403    | `class_not_allowed`                        | the key may not use the class              |
| 404    | `not_found`                                | a route outside section 3                  |
| 413    | `body_too_large`                           | the request body is over 2 MB              |
| 429    | `queue_full`                               | a class or key cap is reached              |
| 503    | `starting`, `draining`, `drained`          | the service is not accepting               |

Closing the HTTP connection cancels the request in the engine. The gateway passes the cancellation on at once, and the
test measures how long the engine keeps generating after the client left. There is no retry inside the gateway, and no
`Retry-After`.

## 10. Privacy

- No log, trace, error report or metric label holds a request or a response body: not the messages, not the system
  prompt, not the output, not the schema. This covers the gateway, the engine and the web server.
- A log row may hold the time, route, key label, class, status, error code, token counts, the three durations of
  section 4, whether the request was cancelled and the finish reason.
- The engine's prefix cache is shared unless told otherwise, and shared caches leak prompts through timing. The gateway
  sets `cache_salt` on every request. All of our classes share one salt, and each outside key gets its own. The salts
  are random at every start and are never shown.
- Tests and load use synthetic stories only, never a real reader's story.
- Before any outside key is issued, check Vast's terms and the license of the weights.

## 11. What simple-story-chat sends llama-server today

Written from `local/llama.ts` at 80fd241. The bot's new adapter follows sections 2 to 9; this list is what changes.

| Today, llama-server                                          | Contract v1                                          |
|--------------------------------------------------------------|------------------------------------------------------|
| `GET /props`: `default_generation_settings.n_ctx` at least the configured context, `total_slots` equal to the configured slots | `GET /v1/state` and `/v1/models`; no slots |
| `id_slot` and `cache_prompt: true`, so a reader's prefix stays in their slot | dropped: the engine's prefix cache has no slots and gives no placement guarantee |
| `repeat_penalty`                                             | `repetition_penalty`                                 |
| `reasoning_format: "deepseek"`, `reasoning_effort: "none"`   | dropped; thinking is off through `chat_template_kwargs` |
| `response_format: {"type": "json_object", "schema": ...}`    | `{"type": "json_schema", "json_schema": ...}`         |
| `timings`: `cache_n`, `prompt_n`, `prompt_ms`, `predicted_n`, `predicted_ms`, `draft_n`, `draft_n_accepted` | `usage.simple_serving`: `queue_ms`, `first_token_ms`, `total_ms` |
| no class                                                     | `X-Simple-Serving-Class`                             |

Kept as they are:
- the model alias checked in `/v1/models` and in every chunk;
- `POST .../input_tokens` with the generation body;
- streaming with usage;
- sampling `temperature` (0.2 for memory, otherwise the configured value, 0.8 by default), `top_p` 0.95, `top_k` 64,
  `min_p` 0;
- the status mapping 401/403, 429, 503, 404 and anything else;
- the client's limits of 2 MB per response body and 100 000 characters of text.

The bot also merges neighbouring messages of the same role before it sends them.

Two changes follow in simple-story-chat. First, the address and the key of the service are configured apart from the
rental control; today `local/gpu-connection.ts` starts `gpu/ensure-server.sh` on the card and forwards port 8080.
Second, the scheduler runs over a provider without slots. It keeps its turns and its order, and it stops placing calls
in slots.

## 12. Versions and shared cases

- `/v1/state` reports `contract: "1"`. A change that breaks a client makes it `"2"`.
- The contract cases live in this repository, in `contract/cases/`. A case holds the request, a script for the fake
  engine, the expected status and stream from the fake, and the expected normalized result or error code. Expected
  streams are compared only for the fake. Real model output is never compared byte for byte.
- simple-story-chat keeps a pinned copy with the contract version, the source commit and a checksum, so that
  `npm test` needs no network. The copy is updated on purpose, never edited in place.
- The same cases run in two places: TypeScript tests of the adapter in simple-story-chat, Python tests of the gateway
  here. Cancellation, a dropped connection and a drain racing a new request are scenario tests in each implementation,
  because a static case cannot describe them.
- Before the first rental: one run of the real adapter against the real gateway over the fake engine. Two green test
  suites alone do not show that the two sides fit.

## 13. The first rental

Everything below is written and dry-run before the card is rented, and the rental rules of simple-story-chat
(`docs/gpu.md`, "While the cards are paid for") apply.

1. Smoke: vLLM loads the same GGUF Q6_K the bot uses with llama.cpp. The time for this is fixed in advance. If it does
   not load or is too slow, it is not fixed on the paid card.
2. The same weights on llama.cpp and on vLLM, with the tokenizer, chat template, thinking off, sampling, context and
   cache mode pinned. Cold and warm are measured apart. This compares two engines on one set of weights. It does not
   promise the same tokens.
3. Synthetic load: the readers' time to first token and speed while `internal` and `external` load runs, including a
   long outside prompt.
4. `npm run eval` in simple-story-chat.
5. Readers move to the service.
6. Outside keys, as a separate step.

Other quantizations, AWQ or GPTQ, are separate configurations measured later. They free memory for more requests, but
the gain in requests served at once still has to be measured.

## Open questions for the owner

1. Does our own `internal` work keep the card awake? Proposal: yes, until the rental's deadline.
2. Outside access in version 1: keys given by hand only? Proposal: yes.

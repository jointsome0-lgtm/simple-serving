# Contract v1 (draft)

The HTTP API of simple-serving, version 1. Status: draft. The gateway in this repository implements it and passes
the shared cases in front of a fake engine; nothing has run in front of vLLM yet (section 15). The bot's side is
`local/serving.ts` in simple-story-chat; section 13 is written from its `local/llama.ts` and `local/scheduler.ts` at
commit 80fd241.

## 1. Scope

- One text model behind a gateway on one rented card. The gateway is FastAPI, run as one worker process in version 1,
  because the drain state, the quotas and the boot identity live in its memory. The engine is vLLM.
- Clients reach only the gateway. The engine, its metrics and its own API listen on loopback.
- Version 1 is streamed text chat. There are no tools, images, audio, logprobs, LoRA adapters, `n > 1` or answers
  without streaming.
- The clients are simple-story-chat and outside keys. The bot sends readers' turns, agent turns and our eval and
  probes, all through its own scheduler.
- The bot's research batches stay on llama.cpp. They ask one server for several samples of one prompt in a single call
  without streaming (`generateMany` in `local/llama.ts`, used by `local/memory-probe.ts --lab`).

## 2. Keys, classes and cache scopes

### Keys

Every request carries `Authorization: Bearer <key>`, on loopback too. The gateway's configuration maps each key to:
- a label, which logs show instead of the key;
- the classes it may use and its default class;
- whether it may name cache scopes.

Outside keys are issued by hand in version 1. There is no sign-up.

### Classes

| Class      | Who                                  | Order | Keeps the card awake |
|------------|--------------------------------------|-------|----------------------|
| `reader`   | a person's turn in the bot           | 1     | yes                  |
| `agent`    | a turn of the bot's agent interface  | 2     | yes                  |
| `internal` | our eval and probes                  | 3     | yes                  |
| `external` | outside keys                         | 4     | no                   |

- A request names its class in `X-Simple-Serving-Class`, otherwise the key's default applies. A class the key may not
  use is refused with 403 `class_not_allowed`, and so is a class that does not exist. Outside keys may use `external`
  only.
- The class sets three things: its limits (section 7), the order in which waiting requests take a shared place
  (section 7), and the engine priority the gateway attaches. An engine priority in the body is refused as an unknown
  field. The gateway builds the engine's request itself, so no client header reaches the engine.
- The class does not protect a request from the engine's preemption. When the engine runs short of cache memory, its
  scheduler takes a running request off and continues it later, sometimes after recomputing its cache. This happens
  inside the engine. The HTTP request is not cancelled, and its tokens arrive later. The contract promises readers
  two things only: places of their own in the gateway (section 7) and a time to first token measured under load
  (section 15).

### Cache scopes

The engine reuses cached prompt prefixes across requests. A client that sends prompts and times the answers can tell
whether a prefix is already cached. So a shared cache leaks, through timing, whether someone else sent the same text.
The engine does not hand out other people's prompts; the channel is timing only.

The gateway sets `cache_salt` on every request from a scope:
- each outside key is one scope;
- a key allowed to name scopes sends one in `X-Simple-Serving-Scope`: `reader.<opaque>` for each reader, `agent` or
  `internal`. `<opaque>` is 8 to 64 characters of `A-Z`, `a-z`, `0-9`, `_` and `-`. A scope of any other form is
  refused with 400 `invalid_request`. Without the header the scope is the class name. A request of class `reader`
  must name a reader scope, otherwise it is refused with 400 `invalid_request`, so that readers never share a cache
  by accident. The bot derives the opaque part from the reader's identity with a secret of its own. It is never a
  Telegram ID. A scope from a key without that right is refused with 403 `scope_not_allowed`, whatever its form;
- the salt is an HMAC, under a secret that the gateway generates at every start and never shows, of the scope's kind
  and name: `scope:reader.<opaque>`, `scope:agent`, `scope:internal` or `key:<key id>`. So an outside key never shares
  a salt with one of our scopes, whatever its label. A salt in the request body is refused as an unknown field.

A reader's turns reuse that reader's own cached story. Readers share the cache neither with each other nor with
agents, eval or outside keys. The price is that each reader's first turn reads the shared system prompt again.

### Turns

The service knows single HTTP requests. A turn of the bot is several requests, such as compaction, repair and the
scene, and between them nothing else of the bot runs on its lane. Readers' turns go through the bot's scheduler, which
is the only controller of the card (section 8). Agent turns reach the model through the bot's socket
(`local/background.ts`) and the same scheduler when the bot serves that socket, which it does with its GPU control;
without the socket they call the service directly. `npm run memory:probe` needs the socket and fails without it.
`npm run eval` and every probe run with `--direct` always call the service directly. A direct call carries no class of
the scheduler, so it is `internal`, and the bot does not see it (section 8).

## 3. Routes

| Route                                   | Method | Keys             | Listener         |
|-----------------------------------------|--------|------------------|------------------|
| `/v1/chat/completions`                  | POST   | any              | public           |
| `/v1/chat/completions/input_tokens`     | POST   | any              | public           |
| `/v1/models`                            | GET    | any              | public           |
| `/v1/state`                             | GET    | any              | public, control  |
| `/v1/control/drain`, `/v1/control/open` | POST   | the control key  | control          |

The gateway has two listeners. The public one is reached over HTTPS. The control one listens on loopback and is
reached over the SSH tunnel. The gateway tells them apart by the listener, never by the client's address, because a
TLS proxy on the card also connects from loopback. The gateway starts only when the control listener and the engine
are on loopback IP addresses, 127.0.0.0/8 or ::1; a name, even `localhost`, is refused. On each listener any other
path or method is 404 `not_found`, and so is a path with a trailing slash: the gateway does not redirect. A key
without the control right gets 403 `forbidden` on a control route. A control body is at most 4096 bytes; a longer
one is refused with 413 `body_too_large`.

When the public listener handles a request, the gateway counts the connections open on it. Past 64 (provisional, see
section 7) it answers 429 `queue_full` and closes the connection. That is a check on each request, not a cap on TCP
connections: a connection whose headers have not all arrived never reaches it. So the TLS proxy in front of the
public listener limits the connections, the time to read headers and bodies, and the size of a body.

## 4. Generation: `POST /v1/chat/completions`

### Request

This is OpenAI chat completions with streaming. The body is at most 2 000 000 bytes. The gateway counts bytes as it
reads, chunked bodies included, and stops reading past the limit with 413 `body_too_large`, before it parses JSON.

| Field                  | Type and range                                   | Rule                               |
|------------------------|--------------------------------------------------|------------------------------------|
| `model`                | string                                           | the exact alias from `/v1/models`  |
| `messages`             | array of `{role, content}`                       | role `system`, `user` or `assistant`; content a string; no other keys |
| `max_tokens`           | integer, at least 1                              | required, at most the class limit  |
| `stream`               | `true`                                           | required                           |
| `stream_options`       | `{"include_usage": true}`                        | required                           |
| `temperature`          | number from 0 to 2                               | optional                           |
| `top_p`                | number above 0, at most 1                        | optional                           |
| `top_k`                | integer, at least 1                              | optional                           |
| `min_p`                | number from 0 to 1                               | optional                           |
| `repetition_penalty`   | number above 0, at most 2                        | optional                           |
| `seed`                 | integer                                          | optional                           |
| `response_format`      | `{"type": "json_schema", "json_schema": {"name": string, "strict": boolean, "schema": object}}` | optional |
| `chat_template_kwargs` | `{"enable_thinking": boolean}`                   | optional, default `false`          |

Any other field, at any depth, is refused with 400 `unsupported_field`. The rule does not look inside `schema`, which
the gateway passes on as it is. A value of the wrong type or out of range is refused with 400 `invalid_request`. So is
a body whose objects and arrays nest more than 64 deep, counting the body itself and the inside of `schema`.

### Before the stream starts

The gateway checks, in this order: the route, the key, the service status, the class, the scope, the body (its size,
then JSON, then the fields), `max_tokens` against the class limit. Then it accepts the request, checking the service
status once more, since a drain may have begun while the body was read. From acceptance on, the wall time and the
measurements of section 11 run, and a drain sees the request as work (section 8). The body is read before
acceptance, so the wall time does not bound that read: the TLS proxy's own timeout and body limit do (section 3).

An accepted generation passes two stages:
1. The count stage: the request waits for a count place and the engine counts the input, under the count limits of
   section 5. The gateway checks the count against the class limit and, together with `max_tokens`, against the
   context.
2. Generation admission: the request waits for its place under the limits of section 7.

The gateway hands the request to the engine and waits for the engine's first valid stream event that is not an error.
Only then does it send 200, the stream headers and that event. Every refusal up to that point is a plain HTTP error
from section 9. When the engine refuses the request with 400 or 422, the answer is 400 `invalid_request`: the engine
found the request itself wrong, for example a schema it cannot compile. Any other status from the engine, a failed
connection, an error event, an invalid event, or a stream that ends before its first event give 503
`engine_unavailable`. An engine that answers 401, 403 or 404 is misconfigured, and that is not the client's error.

An engine event is valid when it is a chunk of the served model: `object` is `chat.completion.chunk`, `model` is the
alias, and `choices` holds at most one choice, with `index` 0 and a delta that holds nothing but the role
`assistant`, text and reasoning. Version 1 is text only, so a tool call or a function call is invalid. Where a field
may be absent, such as `usage`, `finish_reason` or a field of the delta, null counts as absent.

### Stream

`Content-Type: text/event-stream`. Each event is a line `data: <json>` followed by a blank line.

- Every chunk has `model` equal to the alias.
- A chunk has either one choice with `index: 0` or `choices: []`.
- A choice's `delta` may be empty, may hold only `role`, or may hold `content` or `reasoning_content`. The gateway
  renames whatever field the pinned engine uses for reasoning to `reasoning_content`. Reasoning is not part of the
  text.
- Exactly one chunk has a `finish_reason`, `stop` or `length`. Its delta may be empty.
- After it comes exactly one usage chunk with `choices: []` and `usage`:
  - `prompt_tokens`;
  - `completion_tokens`;
  - `prompt_tokens_details.cached_tokens`, optional. Absent means unknown, not zero;
  - `simple_serving`, the measurements of section 11.
- The stream ends with `data: [DONE]`.

### Errors after the stream starts

The status is already 200 and cannot change. The gateway sends one event `data: {"error": {"code": "<code>"}}`
with a code from section 9. Then it ends the stream without a usage chunk or `[DONE]`. Chunks sent before the error,
a finish included, do not make the answer a success. A client treats every stream without `[DONE]` as failed and never
keeps its text as an answer.

The engine's own stream may break the rules after its first event: an invalid event, such as another model or a tool
call, a second finish, an unknown finish reason, no usage chunk, bytes that are not UTF-8, a line that is not an
event. The gateway then ends its stream with the error event `engine_unavailable`.

## 5. Counting: `POST /v1/chat/completions/input_tokens`

- The body is the same as for generation. The answer is `{"input_tokens": n}` with `n` at least 1.
- Counting and generation render the prompt through one code path, with the same chat template, template arguments
  and special tokens, and nothing is cut silently. For the same body, `n` equals `usage.prompt_tokens`. A test on the
  card checks the equality.
- The body limit of section 4 applies, and the headers are checked as for generation. A count is accepted as a
  generation is (section 4), and its class's wall time applies. At most 8 counts run at once (provisional). A count
  past that waits for a count place, in the class order of section 2 and within its wall time. An outside key has at
  most 2 counts at once, waiting ones included; a third is refused with 429 `queue_full`. The count stage of a
  generation shares these limits.
- The gateway sends the engine only what renders the prompt: the model alias, the messages and the template arguments,
  never the rest of the body.
- A count applies neither the class limits of section 7 nor the context. A body whose `n` plus `max_tokens` exceeds
  the context is still counted. The answer is the count.

## 6. Models and state

`GET /v1/models`:

```json
{"object": "list", "data": [{"id": "<alias>", "object": "model", "max_model_len": 65536}]}
```

`max_model_len` is the effective context. It is the engine's configured length, or the gateway's smaller limit if it
sets one. `/v1/state` shows the same number.

`GET /v1/state`:

```json
{"contract": "1", "boot_id": "<random at every start>", "status": "ready", "model": "<alias>",
 "context_tokens": 65536, "drain_generation": 0}
```

- The gateway reports `starting` until it has verified the engine: the engine answers, serves the alias and reports
  the context length. Then the status is `ready`.
- The status is `draining` or `drained` during a stop (section 8), and `failed` if the engine stops answering or
  changes after it was ready. While a drain is on, the status is `draining` or `drained` whatever the engine does.
- The gateway checks the engine every 5 seconds. A check that fails makes a `ready` service `failed`: new requests are
  refused, and the requests already accepted go on within their wall time, since the engine may only be busy. An
  error from the engine still ends its own request at once. The next check that passes makes the service `ready`
  again.
- An engine that answers a check with another model, or with another context length than the boot verified, has
  changed. The service becomes `failed` at once and cancels every request it has accepted, counts included, since
  they were checked against an engine that is gone. One that has not started its stream gets 503
  `engine_unavailable`. One whose stream has started ends with the error event `engine_unavailable`. The service is
  `ready` again only once the engine serves the verified model and context again. A restarted service verifies the
  engine anew.
- In any status except `ready`, the inference routes answer 503. The code is the status itself, except `failed`,
  which answers `engine_unavailable`.
- For the control key the answer also holds `active` and `waiting`, counted by class, and the pinned versions of
  section 12. Both cover generation admission only (section 7): a count, and a generation still in its count stage,
  are in neither. A controller that stops the card waits for the status `drained` (section 8), not for zero
  counters.
- `/v1/state` and the control routes answer in every status.

## 7. Limits

These limits govern generation admission, the second stage of section 4. Counts, and the count stage of a
generation, have the limits of section 5.

- **Waiting**: the generation has passed its count stage and waits for its place. This is the admission queue only.
- **Active**: the gateway handed the request to the engine and it has not finished. That includes the engine's own
  queue and its preemption. A request the engine paused still counts as active.
- **Wall time**: from acceptance (section 4) to the gateway's terminal event. For a generation it includes the wait
  for a count place, the count, the wait for a place and preemption. A count has its class's wall time as well.

Provisional values, used until the first measurement on the card:

| Class               | Active | Waiting | Input tokens | `max_tokens` | Wall time |
|---------------------|--------|---------|--------------|--------------|-----------|
| `reader`            | 4      | 8       | context      | 8192         | 300 s     |
| `agent`             | 1      | 2       | context      | 8192         | 900 s     |
| `internal`          | 2      | 8       | context      | 8192         | 900 s     |
| `external`, all keys| 2      | 4       | 8192         | 1024         | 120 s     |
| `external`, one key | 1      | 2       | 8192         | 1024         | 120 s     |

- A request that would go over Active waits if Waiting has room. Otherwise it is refused with 429 `queue_full`. The
  gateway never hands the engine a request past the cap.
- Readers have places of their own: their Active cap. `agent`, `internal` and `external` share 4 places (provisional),
  each class within its own cap. A request with room in its class but no free shared place waits as well. When a
  shared place frees, the first waiting request of the first class in the order of section 2 takes it, among the
  classes that have room. Within a class the order is arrival.
- These are limits on the requests the gateway admits, not the engine's measured capacity. Whether the engine holds
  the readers' places and the shared places at once, with long prompts, is measured (section 15).
- Input or `max_tokens` over the class limit is refused with 400 `limit_exceeded`. Input plus `max_tokens` over the
  context is refused with 400 `context_limit`. The first is our quota, the second the model's size.
- Past the wall time the gateway cancels the request with the code `timeout` (section 9).
- A limit on requests per minute alone would not protect readers: one long prompt costs more than many short ones.
- The measured values follow one rule: the engine holds the readers' places and the shared places at once.

## 8. Stopping the card

One controller stops the card: the bot's GPU control, `local/gpu.ts` in simple-story-chat. A paused card runs nothing,
so the service cannot wake itself.

1. The bot stops starting new turns and waits until its running turns end. It does not pause while work of `reader`,
   `agent` or `internal` runs or waits. Outside requests never keep the card awake. Only the guard's deadline stops
   the card while our work goes on. What counts as running work differs by class. A reader's job and an agent's turn
   hold the card from start to end, gaps between their requests included. A probe through the bot's socket holds it
   per request, from the moment the bot's scheduler accepts the request until the request has ended locally. The
   scheduler cancels a probe request that has waited 10 minutes or run 90 seconds by its next tick, and the hold ends
   once the cancelled request has ended locally. A whole probe run is not a turn: between two of its requests only
   the idle interval keeps the card up. The bot does not see work that calls the service directly (section 2), such
   as `npm run eval`, so in version 1 that work holds nothing, and the auto-pause can stop the card under it. The
   plan to close that gap is at the end of this section.
2. The bot calls `POST /v1/control/drain` with `{"boot_id": "<from /v1/state>"}`. The gateway increments
   `drain_generation` and answers 202 with `{"status": ..., "boot_id": ..., "drain_generation": n}`. The status is
   `drained` when the gateway had no accepted work, counts included, because the drain then completes before the
   answer, and `draining` otherwise. From then on every new generation or count gets 503 with the current status as
   its code, `draining` or `drained`. `/v1/state` and the control routes stay available.
3. The gateway removes waiting requests of every class and answers them 503 `draining`. That includes a generation
   still in its count stage and a count that waits for a count place. It cancels active outside requests at once.
   Active requests of our classes finish, a count the engine is computing included. Any of them still running after
   60 seconds is cancelled. A cancelled request that has not started its stream yet, because the engine is still
   reading its prompt, gets 503 `draining`. One whose stream has started ends with the error event `draining`.
4. The bot polls `/v1/state` until it reads `drained`. `drained` means the gateway has no accepted work left, counts
   included; the counters `active` and `waiting` do not show counts (section 6). It does not confirm that the GPU is
   idle: an aborted request may compute a moment longer inside the engine (section 9), and stopping the instance
   ends that too.
5. The bot stops the instance through the Vast API and reads the status back.

- A second drain with the same `boot_id` answers the current state and does not increment the generation.
- `POST /v1/control/open` with `{"boot_id": ..., "drain_generation": n}` undoes only that drain and answers 200 with
  the state. With another generation it answers 409 `stale_generation`, so a late open cannot undo a newer drain. An
  open during `draining` also stops the 60-second deadline. An open while no drain is on, with the current
  generation, changes nothing.
- A control call with a `boot_id` other than the current one answers 409 `stale_boot`. A restarted service is a new
  boot, and the controller reads `/v1/state` again.
- The check "nothing runs" followed by a stop is not enough, because a request can arrive between the two. The drain
  closes that gap.
- The guard on the card deletes the instance at the rental's deadline, whatever the bot and the service do. Clients
  then see connection errors.

Not in version 1 yet, the plan for work the bot does not see:

- Polling `active` and `waiting` is not enough. They leave out counts, so the card could sleep under a long count.
  A short request can start and end between two reads, so a sequential eval could look idle every time. And zero
  counters between an agent's calls do not mean its turn has ended.
- For the control key, `/v1/state` would report our accepted, unfinished work per class, counts included. It would
  also report a counter that grows at every change in our activity within the boot. Outside work counts in neither.
- The controller holds the card while that work is above zero. It starts the idle interval again when the counter
  has moved between two reads, and forgets both after a boot change. The bot's own holds for readers' jobs and agent
  turns stay.
- An idle drain carries the counter the controller last read. The gateway refuses it with 409 if our activity has
  changed since. Without that, work that arrives between the check and the drain would wait and be cancelled by
  step 3. A manual drain has no such condition.
- Tests, none of which needs a card: a long count, short calls between two reads, an agent turn between its calls,
  an outside stream with none of our work, and work that arrives between the idle check and the drain.

## 9. Errors and cancellation

Every error body is `{"error": {"code": "<code>"}}`. The gateway installs its own handlers for validation errors, HTTP
errors, unhandled exceptions and engine errors. None of them serializes an exception, its detail, the input or the
body. FastAPI's default validation answer echoes the input, so it is replaced.

| Status | Code                                              | When                                         |
|--------|---------------------------------------------------|----------------------------------------------|
| 400    | `invalid_request`, `unsupported_field`            | the body breaks section 4 or 8, or the scope breaks section 2 |
| 400    | `limit_exceeded`                                  | input or `max_tokens` over the class limit   |
| 400    | `context_limit`                                   | input plus `max_tokens` over the context     |
| 401    | `unauthorized`                                    | no key or an unknown key                     |
| 403    | `class_not_allowed`, `scope_not_allowed`          | the key may not use the class or the scope   |
| 403    | `forbidden`                                       | a key without the control right on a control route |
| 404    | `not_found`                                       | a path or method outside section 3           |
| 409    | `stale_boot`, `stale_generation`                  | control calls, section 8                     |
| 413    | `body_too_large`                                  | a body over 2 000 000 bytes, a control body over 4096 bytes |
| 429    | `queue_full`                                      | a cap of section 3, 5 or 7                   |
| 500    | `internal_error`                                  | an error in the gateway itself               |
| 503    | `starting`, `draining`, `drained`, `engine_unavailable` | the service is not serving             |
| 504    | `timeout`                                         | the wall time ran out before the stream started |

After the stream has started, the same codes arrive as the error event of section 4.

Cancellation:
- A client cancels by closing the connection.
- The gateway watches for the disconnect in every phase: while the request is counted or waits, while the engine
  reads the prompt, and while it streams. It sends nothing while the prompt is read, so it listens for the disconnect
  itself instead of waiting for a failed send.
- On a disconnect, a timeout, a drain or an engine change, the gateway aborts the engine request in a `finally`
  block. It frees the quota only after its own engine request has ended: the task that talks to the engine has
  finished and its connection is closed. That is a local end, not a confirmation from the engine. vLLM learns of the
  abort from the closed connection and may compute a little longer; that tail is measured on the card (section 15). A
  stronger guarantee would need the engine to report the end of that very request, and a closed socket is not such a
  report.
- The terminal message goes out after the engine request has ended and the place is free: the usage chunk and
  `[DONE]`, the count, or an error. So a client that stops reading holds no place. If the request is stopped while
  that message waits for the client, by its wall time, a drain or an engine change, the gateway gives the message up
  and closes the connection.
- Tests run the gateway in a real ASGI server over HTTP, not only through a test client. They measure how long the
  engine keeps generating after the client left.
- The gateway does not retry and sends no `Retry-After`.

## 10. Privacy

- No log, trace, error report or metric label holds a request or response body. That means no messages, no system
  prompt, no output and no schema. The rule covers the gateway, the web server and the engine, and the engine's
  request logging stays off.
- A log row may hold the time, route, key label, class, scope kind (`reader`, `agent`, `internal` or `external`, never
  the opaque part), status, error code, token counts, the measurements of section 11, whether the request was
  cancelled, and the finish reason.
- Cache scopes and salts are described in section 2. They close the timing channel between scopes. They do not limit
  memory.
- Tests and load use synthetic stories only, never a real reader's story.
- Before any outside key is issued, check Vast's terms and the license of the weights.

## 11. Measurements

The gateway measures in whole milliseconds, counting from the moment it accepted the request:
- `wait_ms`: until it handed the request to the engine to generate, its count stage included;
- `first_token_ms`: until it received the first generated token of any kind from the engine, reasoning included;
- `total_ms`: until it read the end of the engine's stream. The cleanup and the send of the terminal event come after
  and are not in it, because the usage chunk cannot know how long its own send will take.

These are the gateway's own numbers, not llama.cpp's timings. The engine's queue and preemption are inside
`first_token_ms` and `total_ms`. The numbers arrive in `usage.simple_serving`.

## 12. Pinned versions

Before any contract run on the card, a lock file in this repository pins:
- vLLM and its GGUF plugin;
- FastAPI, Starlette and the ASGI server;
- the model file, with its hash;
- the revision of the tokenizer and the chat template.

`/v1/state` shows them to the control key. A change to any of them makes a new configuration, which is measured again.

## 13. What changes in simple-story-chat

The new adapter is separate from `createLlama` and `createOpenAI`. `SIMPLE_CHAT_ALLOW_HOSTED` stays as it is.

| Today, llama-server                                          | Contract v1                                     |
|--------------------------------------------------------------|-------------------------------------------------|
| `GET /props`: `n_ctx` at least the configured context, `total_slots` equal to the configured slots | `/v1/models` and `/v1/state`, no slots |
| `id_slot` when the scheduler names a slot, and `cache_prompt: true` | dropped                                  |
| `repeat_penalty`                                             | `repetition_penalty`                            |
| `reasoning_format: "deepseek"`, `reasoning_effort: "none"`   | dropped; thinking is off through `chat_template_kwargs` |
| `response_format: {"type": "json_object", "schema": ...}`    | `{"type": "json_schema", "json_schema": {"name", "strict", "schema"}}` |
| `timings` from llama-server                                  | `usage.simple_serving` and new log fields       |
| the key is optional                                          | the key is required                             |
| `model` in a chunk is checked when present                   | `model` is required in every chunk              |
| `usage` is optional; with an exact count before generation, a stream may pass without it | the usage chunk is required |
| a status other than 401/403/429/503/404 is `provider_failed`; after a 400 on a trusted estimate the adapter counts the input and may report `context_limit` | the adapter reads `error.code`, and `context_limit` arrives as a code; the recount after a 400 stays until the adapter trusts the code |
| no class, no scope                                           | `X-Simple-Serving-Class`, `X-Simple-Serving-Scope` |

Unchanged:
- the count request carries the generation body, without `id_slot` as today;
- sampling: `temperature` 0.2 for memory, otherwise the configured value, 0.8 by default; `top_p` 0.95, `top_k` 64,
  `min_p` 0;
- the client's limits: 2 000 000 bytes per response body and 100 000 characters of text;
- neighbouring messages of the same role are merged before sending.

Also in simple-story-chat:
- The service's address and key are configured apart from the rental control. Today `local/gpu-connection.ts` starts
  `gpu/ensure-server.sh` on the card and forwards port 8080.
- Over the new adapter the scheduler runs one lane, so the bot's calls stay one at a time. The guarantees built on
  slots are off. Work marked `sharesPrefix` no longer has a slot where its prefix is sure to be cached, and the engine
  may evict any prefix. This is a limitation of version 1. Several lanes without placement come later.
- `usage.simple_serving` goes into new log fields, added to the whitelist in `local/model-error.ts` as non-negative
  integers. The adapter does not write these numbers into `promptMs` or `predictedMs`.
- `generateMany` stays on llama.cpp (section 1).
- The idle pause in `local/gpu.ts` counts readers' jobs, agent turns and probe requests through its queue as work that
  keeps the card awake. Before, only readers' jobs held the countdown, and an agent turn delayed the pause while the
  countdown ran on. Work that calls the service directly, such as `npm run eval`, waits for the plan at the end of
  section 8.

## 14. Shared cases

- `/v1/state` reports the contract version. A change that breaks a client makes it `"2"`.
- The canonical cases live here, in `contract/cases-v1.json`; `contract/README.md` describes the format. A case holds
  the request, a script for the fake engine, the expected status and stream from the fake, and the expected
  normalized result or error code. Streams are compared only for the fake. Real model output is never compared byte
  for byte.
- simple-story-chat keeps a pinned copy with the contract version, the source commit and a checksum, so that
  `npm test` needs no network. The copy is updated on purpose and never edited in place.
- The same cases run in two places: TypeScript tests of the adapter there, Python tests of the gateway here.
- Cancellation, a dropped connection, a drain racing a new request and a late open are scenario tests in each
  implementation. A static case cannot describe them.
- Before the first rental there is one run of the real adapter against the real gateway, in a real ASGI server, over
  the fake engine.

## 15. The first rental

Everything below is written and dry-run before the card is rented. The rental rules of simple-story-chat apply
(`docs/gpu.md`, "While the cards are paid for").

Before the rental, without a card:

- Pin vLLM (a version or an image), the weights, the tokenizer and the chat template. Check the API and the CLI of
  that pin, not the latest docs:
  - `--scheduling-policy priority`: without it vLLM refuses a non-zero priority, and every `agent`, `internal` and
    `external` request fails;
  - prefix caching with `cache_salt`, and `--enable-prompt-tokens-details` for `cached_tokens`;
  - request and output logging off;
  - the name of the reasoning field, the finish reasons, and the shape of an error in the middle of a stream;
  - the model name in every chunk, the usage chunk included: the gateway refuses a chunk that does not name the alias;
  - the fields the gateway sends: `add_special_tokens`, `chat_template_kwargs`, `top_k`, `min_p`,
    `repetition_penalty`, `cache_salt`, `priority`, and `max_tokens`, which upstream calls deprecated;
  - `max_model_len` and the served name in `/v1/models`.
- Write the launch script with those flags.
- Write the smoke probes. They cover every field section 4 accepts, reasoning, an engine error, an abort, the model
  name, and the bot's own JSON schemas, which use `minLength`, `maxLength`, `minItems`, `maxItems` and `pattern` in
  strict mode.
- Write the count matrix: the count against `usage.prompt_tokens` for plain text, for system, user and assistant
  turns, with a schema, with thinking on and off, and near the context. One `count_matches` in a log is an
  observation, not a passed check.
- Configure the TLS proxy: no SSE buffering, the upstream connection closed when the client leaves, timeouts for
  headers and bodies, a connection limit. Check it in front of the fake engine with a slow client.
- Write the load scenarios and what counts as a pass. Only their numbers are measured on the card.

On the card:

1. Smoke: vLLM loads the GGUF Q6_K that the bot uses with llama.cpp. The time for this is fixed in advance. If the
   file does not load or is too slow, nobody fixes that on the paid card, and other weights are measured later as a
   separate configuration. Then the smoke probes and the count matrix run, with the real template.
2. Isolation and privacy, before any real story or outside key: a synthetic series checks that cache scopes stay
   apart, and that the privacy marker shows up in no log of the engine, the proxy or the gateway.
3. The same weights on llama.cpp and on vLLM. The tokenizer, chat template, thinking, sampling, context and cache
   mode are pinned. Cold and warm runs are measured apart. This compares two engines on one set of weights. It does
   not promise the same tokens.
4. Synthetic load: the readers' time to first token and speed while `internal` and `external` load runs, including a
   long outside prompt. The measured limits replace the provisional ones of section 7. Also the tail of an abort: how
   long the engine keeps computing a request after the gateway closed it, measured apart while the prompt is read and
   while the answer streams, through the proxy.
5. `npm run eval` in simple-story-chat.
6. Readers move to the service.
7. Outside keys, as a separate step.

AWQ and GPTQ are separate configurations, measured later. They free memory, but whether more requests then run at
once has to be measured.

## Decided

- 2026-09-24, the owner: our `internal` work keeps the card awake until the rental's deadline, and outside keys
  are issued by hand only.

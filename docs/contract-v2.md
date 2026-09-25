# Contract v2 (draft)

The HTTP API of simple-serving, version 2. Status: draft. The gateway in this repository implements it and passes
the shared cases in front of a fake engine; nothing has run in front of vLLM yet (section 15). The bot's side is
`local/serving.ts` in simple-story-chat; section 13 is written from its `local/llama.ts` and `local/scheduler.ts` at
commit 80fd241.

Version 2 changes who stops the card (section 8). In version 1 the bot stopped it, and a reader's or an agent's turn
held it between calls. In version 2 the service stops its own card after 13 minutes without work of ours, only
requests hold it, and simple-serving's own command starts it. A client of version 1 counts on both things that went,
so the number changes.

## 1. Scope

- One text model behind a gateway on one rented card. The gateway is FastAPI, run as one worker process in version 2,
  because the drain state, the idle interval, the quotas and the boot identity live in its memory. The engine is
  vLLM.
- Clients reach only the gateway. The engine, its metrics and its own API listen on loopback.
- Version 2 is streamed text chat. There are no tools, images, audio, logprobs, LoRA adapters, `n > 1` or answers
  without streaming.
- The clients call the API with a key: simple-story-chat for readers' turns, its agent interface, our eval and probes,
  and outside keys. The story turn, its storage, the prompts and the reader's identity stay in simple-story-chat.
  Admission, the idle interval and the stop of the card are the gateway's (sections 7 and 8).
- The bot's research batches stay on llama.cpp. They ask one server for several samples of one prompt in a single call
  without streaming (`generateMany` in `local/llama.ts`, used by `local/memory-probe.ts --lab`).

## 2. Keys, classes and cache scopes

### Keys

Every request carries `Authorization: Bearer <key>`, on loopback too. The gateway's configuration maps each key to:
- a label, which logs show instead of the key;
- the classes it may use and its default class;
- whether it may name cache scopes.

Outside keys are issued by hand in version 2. There is no sign-up.

### Classes

| Class      | Who                                  | Order | Keeps the card awake |
|------------|--------------------------------------|-------|----------------------|
| `reader`   | a person's turn in the bot           | 1     | yes                  |
| `agent`    | a call of the bot's agent interface  | 2     | yes                  |
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
scene, and between them nothing else of the bot runs on its lane. Each client calls the service directly and names
its class: the bot's readers' turns `reader` with the reader's scope, the agent interface `agent`, eval and the probes
`internal`. The agent interface does not go through the bot's model socket (`local/background.ts`) on this path, and
`npm run memory:probe`, which needs that socket, does not run on it. A turn holds the card only while one of its
requests runs (section 8).

## 3. Routes

| Route                                   | Method | Keys             | Listener         |
|-----------------------------------------|--------|------------------|------------------|
| `/v1/chat/completions`                  | POST   | any              | public           |
| `/v1/chat/completions/input_tokens`     | POST   | any              | public           |
| `/v1/models`                            | GET    | any              | public           |
| `/v1/state`                             | GET    | any              | public, control  |
| `/v1/control/drain`                     | POST   | the control key  | control          |
| `/v1/control/sleep`                     | POST   | the control key  | control          |
| `/v1/control/open`                      | POST   | the control key  | control          |

The gateway has two listeners, both on loopback on the card: public on 8090, control on 8091. Our clients reach them
through one SSH tunnel that simple-serving's command holds (section 8), local port 8080 to the public listener and 8081
to the control one. The control listener is for that command; the bot holds no control key. Outside keys will reach
the public listener over HTTPS through a TLS proxy on the card, which also connects from loopback, so the gateway
tells the listeners apart by the listener, never by the client's address. The gateway starts only when the control
listener and the engine are on loopback IP addresses, 127.0.0.0/8 or ::1; a name, even `localhost`, is refused. On
each listener any other path or method is 404 `not_found`, and so is a path with a trailing slash: the gateway does
not redirect. A key without the control right gets 403 `forbidden` on a control route. A control body is at most 4096
bytes; a longer one is refused with 413 `body_too_large`.

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
measurements of section 11 run, and a drain sees the request as work (section 8). Before acceptance a fixed bound
applies instead, 30 seconds from the request's arrival, and it is not a setting: a body that has not all arrived by
then is refused with 504 `timeout`, and a refusal that its client does not take by then is given up with the
connection.

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
`assistant`, text and reasoning. Version 2 is text only, so a tool call or a function call is invalid. Where a field
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
{"contract": "2", "boot_id": "<random at every start>", "status": "ready", "model": "<alias>",
 "context_tokens": 65536, "drain_generation": 0}
```

- The gateway reports `starting` until it has verified the engine: the engine answers, serves the alias and reports
  the context length. Then the status is `ready`.
- The status is `draining` or `drained` during a drain or a sleep (section 8), and `failed` if the engine stops
  answering or changes after it was ready. While a drain is on, the status is `draining` or `drained` whatever the
  engine does.
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
- For the control key the answer also holds `sleep_requested`, true once a sleep has begun (section 8); `active` and
  `waiting`, counted by class; and the pinned versions of section 12. `active` and `waiting` cover generation
  admission only (section 7): a count, and a generation still in its count stage, are in neither. Whether the card
  may stop is the gateway's own decision (section 8), never a client's reading of these counters.
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

## 8. Sleep, stop and start

The service decides when its card stops: after 13 minutes without work of ours, or at once when simple-serving's
command asks it to sleep. The same command starts the card again. A stopped card runs nothing and nothing wakes it by
itself; a client that finds it asleep gets connection errors, or 503 while it falls asleep.

### What holds the card

- A request of class `reader`, `agent` or `internal` holds the card from the moment its key, class and scope have
  passed their checks, before its body is read, until it has ended, whether it was accepted, refused or cancelled.
- Nothing else holds it: not outside requests, `/v1/models`, `/v1/state`, the control routes, the engine's health
  checks, a request refused before its class is known, or an open tunnel. Nor does a turn between its calls, or a job
  that waits in a client and has not reached the service.
- `idle_timeout_s` is 780 seconds by default. The interval starts when the service is first `ready` in its boot, and
  runs again in full from the end of the last request of ours. While a request of ours runs, the service does not
  fall asleep.
- So a turn whose calls are more than 13 minutes apart may find the card asleep. A client or a tunnel that goes away
  never stops the card: only the interval and the command's sleep do.
- The interval lives in the gateway's memory. A new boot starts a new one at its first `ready`. A load that never
  becomes `ready` is bounded by the card's load deadline instead (below).

### Falling asleep

When the interval has run out, the gateway checks in one step, with nothing else running in between, that no request
of ours is in flight and that the interval has really run out. In the same step it sets `sleep_requested` and begins a
drain. A request that arrived before that step holds the card; one that arrives after it gets 503 `draining` or
`drained`, and the sleep goes on.

`POST /v1/control/sleep` with `{"boot_id": "<from /v1/state>"}` does the same at once, whatever the interval. It
answers 202 with `{"status": ..., "boot_id": ..., "drain_generation": n}`. A repeated sleep answers the current state
and starts nothing new, so a client that lost the answer asks again. A sleep for another boot answers 409 `stale_boot`
and changes nothing.

The drain of a sleep is the drain below. When a drain is already on, the sleep goes on with it and the generation
stays. Once the drain has ended, status `drained`, the gateway stops its instance. Nothing undoes a sleep: an open
answers 409 `sleep_pending`, because a stop already on its way would take down a service that had opened again.

If the drain has not ended 120 seconds after the sleep began, the gateway begins to stop the instance all the same.
The 120 seconds bound the wait for the drain, not the time to a confirmed stop: attempts that fail, by a 401 or 403 or
a network error, go on without end (below). This is the one case where the card stops under unfinished work of ours,
an exception to the rule that our work finishes. It happens only with a drain deadline above 120 seconds, or with a
request that fails to end when the drain cancels it. An idle sleep has no work of ours when it begins.

### The stop

- The gateway stops the instance it runs on with the credential Vast gives the container, `CONTAINER_API_KEY`, never
  the owner's account key: `PUT {"state": "stopped"}` for its own instance, `CONTAINER_ID`, to one fixed HTTPS
  endpoint, with the key in a header. The stop is accepted when Vast answers 2xx with `"success": true`.
- One attempt runs at a time, each for at most 20 seconds, and after a failure the next comes 30 seconds later, until
  Vast accepts one. A timeout, a network error, a 5xx, a 401 or 403 and an answer without success are all failures,
  which the log tells apart by a fixed category and the HTTP status, never by Vast's text. Stopping an instance that
  already stops changes nothing, so a repeated attempt does no harm.
- A failed attempt, the gateway's or the launcher's, leaves the card's marker `stop-unconfirmed` and the log row
  `stop_unconfirmed` with the failure's category and status, until the launcher's next start: the stop is not
  confirmed, and costs may go on. `up`, `sleep` and `status` say so. The attempts go on, but while Vast refuses the
  container's key they bound nothing, and the owner stops the instance in Vast's console (section 15).
- Stop, not delete: the disk with the weights stays, and Vast bills for it while the card is stopped.
- The gateway cannot see its own stop complete, since the stop ends it. Until then admission stays closed and it
  never reports `ready`. Its status is `drained`, or, after a stop that did not wait for the drain, `draining` until
  the work that held it ends. The command reads the stopped state back from Vast.

### The command

simple-serving's command on the owner's machine starts the card, holds the tunnel and asks for sleep. It uses the
owner's restricted Vast key (section 12) and never the card's.

- `up` reads the instance's state in Vast. It waits for a stop in flight to end rather than start against it, resumes
  the instance once if it is stopped, and waits for it to run and for SSH. It opens the tunnel, checks the gateway's
  identity, contract and model, waits for `ready`, and holds the tunnel in the foreground. It never starts a service
  over SSH; the card starts its own (below). Ctrl+C closes the tunnel and sends nothing, so the card stays up until
  its interval runs out. A lost tunnel is opened again a few times while the instance runs, never by a resume, and
  once the card has stopped `up` ends.
- `sleep` sends `POST /v1/control/sleep` through a short control-only forward of its own, never through `up`'s ports,
  whose lock does not show that the tunnel is bound, and then reads `stopped` back from Vast. Without the gateway it
  does not stop the instance directly, since that would skip the drain.
- `status` tells the instance's state in Vast apart from what the gateway answers, through the same kind of forward.
  Of an answer it prints only the statuses and codes of this contract, counts, and whether the contract and the model
  are the expected ones.
- `keys` makes the client key and the control key once, and prints only the SHA-256 of each, which the card's
  preparation reads (section 12).

### Starting the card

- At every start of the instance the last line of the rental's own onstart runs the card's `onstart.sh` from the
  persistent disk, and that starts the card's launcher. The preparation changes no onstart, since a start may restore
  it. The launcher runs one pair, vLLM and the gateway, under a lock and with no restarts. Each start is a new boot.
- A load that is not `ready` by the load deadline of the card's manifest, or a process of the pair that exits, ends
  the pair. The launcher then stops the instance with the same call as the gateway. When the pair never became
  `ready`, it first leaves a marker, and the card does not load the model again until the owner retries by hand. A
  load never stays `starting` without end.
- A start that finds the marker, a resume included, runs no pair. The launcher waits the idle interval for the owner's
  retry, which runs the pair, and without one stops the instance again as a sleep does, so that no card stays up with
  nothing to stop it. A retry that comes too late for the wait takes effect at the next start. `up` refuses such a
  card and says why, and `status` says it too.

### Drain and open

A sleep drains through the same steps. On their own, drain and open are for technical checks.

1. `POST /v1/control/drain` with `{"boot_id": "<from /v1/state>"}`. The gateway increments `drain_generation` and
   answers 202 with `{"status": ..., "boot_id": ..., "drain_generation": n}`. The status is `drained` when the gateway
   had no accepted work, counts included, because the drain then completes before the answer, and `draining`
   otherwise. From then on every new generation or count gets 503 with the current status as its code, `draining` or
   `drained`. `/v1/state` and the control routes stay available.
2. The gateway removes waiting requests of every class and answers them 503 `draining`. That includes a generation
   still in its count stage and a count that waits for a count place. It cancels active outside requests at once.
   Active requests of our classes finish, a count the engine is computing included. Any of them still running after
   60 seconds is cancelled. A cancelled request that has not started its stream yet, because the engine is still
   reading its prompt, gets 503 `draining`. One whose stream has started ends with the error event `draining`.
3. `drained` means the gateway has no accepted work left, counts included; the counters `active` and `waiting` do not
   show counts (section 6). It does not confirm that the GPU is idle: an aborted request may compute a moment longer
   inside the engine (section 9), and stopping the instance ends that too.

- A second drain with the same `boot_id` answers the current state and does not increment the generation.
- `POST /v1/control/open` with `{"boot_id": ..., "drain_generation": n}` undoes only that drain and answers 200 with
  the state. With another generation it answers 409 `stale_generation`, so a late open cannot undo a newer drain. An
  open during `draining` also stops the 60-second deadline. An open while no drain is on, with the current
  generation, changes nothing. Once a sleep has begun, an open answers 409 `sleep_pending`.
- A control call with a `boot_id` other than the current one answers 409 `stale_boot`. A restarted service is a new
  boot, and the command reads `/v1/state` again.

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
| 409    | `stale_boot`, `stale_generation`, `sleep_pending` | control calls, section 8                     |
| 413    | `body_too_large`                                  | a body over 2 000 000 bytes, a control body over 4096 bytes |
| 429    | `queue_full`                                      | a cap of section 3, 5 or 7                   |
| 500    | `internal_error`                                  | an error in the gateway itself               |
| 503    | `starting`, `draining`, `drained`, `engine_unavailable` | the service is not serving             |
| 504    | `timeout`                                         | the wall time ran out before the stream started, or the body did not arrive within 30 seconds (section 4) |

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
- On the card nothing of vLLM's own output is kept. A bounded filter reads it to the end and keeps the exit code, a
  fixed category of failure or warning and a few numbers: load time, memory and KV cache capacity. The gateway's rows
  are checked again on the way to their file. Request and access logging, core dumps and usage statistics are off.
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

## 12. Pinned versions, configuration and keys

Before any contract run on the card, this repository pins:
- vLLM, in a lock with hashes;
- FastAPI, Starlette, the ASGI server and the rest of the gateway, in `uv.lock`;
- the model's files at one revision, each with its hash;
- the revision of the tokenizer and the chat template.

`card/manifest.env` holds the card's pins and parameters. The card's preparation installs each lock into a venv of its
own, checks every hash and runs `pip check`, once per rental and never at a resume. `/v1/state` shows the versions to
the control key. A change to any of them makes a new configuration, which is measured again.

Configuration and keys live in simple-serving. The owner's machine keeps one private configuration of the command,
with the owner's restricted Vast key, allowed GET and PUT on the chosen instance only, and two gateway keys: a
client key and a control key. The card gets only the SHA-256 of each, once, through the stdin of SSH during the
preparation. The owner copies the client key and the address into the bot's model profile, so simple-story-chat holds
neither a Vast key nor the control key. The bot, its agent interface, eval and the probes share the client key for
now; their classes still differ (section 2).

## 13. What changes in simple-story-chat

The new adapter is separate from `createLlama` and `createOpenAI`. `SIMPLE_CHAT_ALLOW_HOSTED` stays as it is.

| Today, llama-server                                          | Contract v2                                     |
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
- The service's address and client key are the whole model profile. The bot has no card control on this path: no
  start or pause buttons, no drain and no stop. It starts while the service is down, and a request that finds the
  service unavailable fails with `model_unavailable`.
- The agent interface calls the service directly with class `agent` (section 2).
- Over the new adapter the scheduler runs one lane, so the bot's calls stay one at a time. The guarantees built on
  slots are off. Work marked `sharesPrefix` no longer has a slot where its prefix is sure to be cached, and the engine
  may evict any prefix. This is a limitation of version 2. Several lanes without placement come later.
- `usage.simple_serving` goes into new log fields, added to the whitelist in `local/model-error.ts` as non-negative
  integers. The adapter does not write these numbers into `promptMs` or `predictedMs`.
- `generateMany` stays on llama.cpp (section 1).

## 14. Shared cases

- `/v1/state` reports the contract version. A change that breaks a client makes it `"3"`.
- The canonical cases live here, in `contract/cases-v2.json`; `contract/README.md` describes the format. A case holds
  the request, a script for the fake engine, the expected status and stream from the fake, and the expected
  normalized result or error code. Streams are compared only for the fake. Real model output is never compared byte
  for byte.
- simple-story-chat keeps a pinned copy with the contract version, the source commit and a checksum, so that
  `npm test` needs no network. The copy is updated on purpose and never edited in place.
- The same cases run in two places: TypeScript tests of the adapter there, Python tests of the gateway here.
- Cancellation, a dropped connection, a drain racing a new request and a late open are scenario tests in each
  implementation. A static case cannot describe them.
- Three kinds of test stay apart:
  - the adapter's conformance to the public cases, in simple-story-chat;
  - the control routes, the idle interval, the sleep, the stop and the command, in simple-serving alone, on a fake
    clock, a fake Vast and a fake SSH;
  - an opt-in test in simple-story-chat that runs the real adapter against the real gateway, in a real ASGI server,
    over the fake engine, and records both commits.

## 15. The first rental

Everything below is written and dry-run before the card is rented. The rental rules of simple-story-chat apply
(`docs/gpu.md`, "While the cards are paid for").

The first rental is a disposable trial. Its own onstart, simple-story-chat's `gpu/trial-onstart.sh`, arms a guard
that deletes the instance three hours after the first start, and writes the instance's id and key, which the first
preparation needs; its last line runs the card's `onstart.sh` (section 8). The guard deletes with the container's key,
the key of the card's stop, so a key that Vast refuses or has revoked defeats both: nothing on the card then bounds
the costs. The owner is that bound, with the readback, the deadline and the console action of the README's "The first
rental", approved in advance. The service never deletes its card, and its `onstart.sh` arms no guard. An onstart for
a permanent rental, which writes the two files and ends with the same line but arms no guard, is decided before
permanent use. Unattended or permanent use also needs an independent budget path, which is open and not built.

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

1. Smoke: vLLM loads the weights of the manifest, route A: a 4-bit NVFP4 conversion of the heretic whose Q6_K GGUF
   the bot uses with llama.cpp, since no released vLLM loads that GGUF. The time for this is fixed in advance. If the
   weights do not load or are too slow, nobody fixes that on the paid card, and other weights are measured later as a
   separate configuration. Once it is `ready`, before anything long, the stop: `sleep`, and the command reads
   `stopped` back from Vast; then `up` resumes the instance, and the rental's own onstart, with no start over SSH,
   starts exactly one pair with a new boot, the pins, keys and weights still in place, and the trial guard's deadline
   unchanged. Then the smoke probes and the count matrix run, with the real template.
2. Isolation and privacy, before any real story or outside key: a synthetic series checks that cache scopes stay
   apart, and that the privacy marker shows up in no log of the engine, the proxy or the gateway.
3. llama.cpp with the bot's Q6_K, and vLLM with route A's 4-bit weights. The tokenizer, chat template, thinking,
   sampling, context and cache mode are pinned on each side. Cold and warm runs are measured apart. The two differ in
   engine and in weights at once, so this step measures both together and cannot tell the 4-bit loss from the
   engine. It does not promise the same tokens. Whether A's quality is enough is for eval to judge, against the
   llama.cpp baseline (step 6).
4. Synthetic load: the readers' time to first token and speed while `internal` and `external` load runs, including a
   long outside prompt. The measured limits replace the provisional ones of section 7. Also the tail of an abort: how
   long the engine keeps computing a request after the gateway closed it, measured apart while the prompt is read and
   while the answer streams, through the proxy.
5. The card's own lifecycle, with no bot running: eval requests hold the service, and once they end the service falls
   asleep on its own, and the command reads `stopped` back from Vast. The listeners stay on loopback, and no raw
   output is kept.
6. `npm run eval` in simple-story-chat: route A's quality against the llama.cpp baseline.
7. Readers move to the service.
8. Outside keys, as a separate step.

If eval finds A's quality short, route B follows: our own 4-bit conversion of the heretic's bfloat16 weights, AWQ,
GPTQ or NVFP4, a separate configuration, measured again.

## Decided

- 2026-09-24, the owner: our `internal` work keeps the card awake until the rental's deadline, and outside keys
  are issued by hand only.
- 2026-09-24, the owner: the service stops its own card after 13 minutes without work of ours, and stops rather
  than deletes it. The bot keeps the story flow only; everything about the card and the model service lives in
  simple-serving.
- 2026-09-25, the owner: the trial serves route A, a ready 4-bit NVFP4 conversion of the heretic that vLLM loads in
  tree. Route B, our own conversion, follows only if eval finds A's quality short.

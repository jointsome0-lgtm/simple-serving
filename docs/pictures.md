# Pictures (design)

How simple-story-chat's picture card moves here. Status: design, 2026-09-28, reviewed the same day by a GPT-6 Astra
session ("sound with changes", section 16); phase 1 (section 15) is written against a fake ComfyUI and has not run on
a card.

The owner decided on 2026-09-26 that pictures move here with ComfyUI inside and the engine unchanged (README), and
asked on 2026-09-28 to begin: «можем начать переход comfyui на simple-serving? Что бы мы все запускали через
simple-serving, а тут только доменная часть бы была и тгшная, ну и эксперименты наверное». When the move is over:

- simple-story-chat keeps the domain: what to draw and why, the graphs, the references, the recipes it saves with a
  picture, Telegram, and the experiments with their harnesses;
- simple-serving installs, starts, keeps, queues and serves the picture lane as it does the text lane.

The engine is the one simple-story-chat's bot draws on since 2026-09-27 (its `docs/gpu.md#bot-card`): ComfyUI
v0.37.0 at `73c9bad4`, torch 2.11.0+cu130 in `.venv-cu130`, comfy-kitchen's Triton backend on, Qwen-Image 2.1 alone
(`SIMPLE_CHAT_IMAGE_QWEN=only`), no custom nodes. Nothing here changes it or its pins: the scripts that install and
start it are copied byte for byte (section 9), and every route of ComfyUI's that the gateway calls is in the pinned
revision.

## 1. Scope

- One picture engine behind a gateway on one rented card, as contract v2 has one text engine. The gateway is this
  package's, a second service beside the text one (`simple_serving.pictures`): keys, classes, scopes, the status,
  drain, sleep and the stop of the card, the idle interval, the boot identity and the logs are the text lane's code
  and rules (contract sections 2, 6, 8, 9 and 10). What differs is the engine client, the routes, the checks and the
  limits, below.
- The lanes stay on separate cards: one 5090 does not hold the text model and the picture model at once. Each lane
  has its own manifest, ports, preparation, launcher, state directory and instance.
- Clients: the bot for its readers (`reader`), the bot's experiments and probes (`internal`). The agent interface
  draws nothing today; `agent` is allowed with small limits so that it can. Outside keys get no picture route in
  phase 1: the picture configuration holds no outside key, and the limits of `external` are zero.
- Version 2's text contract does not change. The picture lane answers `"contract": "2"` in `/v1/state` beside
  `"lane": "pictures"`, and its routes are new ones.

## 2. Where the graph is built

**Recommendation: the graph is built in simple-story-chat and sent through. The gateway checks it by class and hands
it to ComfyUI unchanged, apart from ComfyUI's per-job preview nonce. Experiments send raw graphs on the same route
under class `internal`, whose check is looser.** Named recipes on the gateway are rejected.

Why:

1. The graph is the domain. Which workflow, the prompt as the text encoder reads it, the references and their scales,
   the size, the seed, the kitchen's attention node: all of it is decided in simple-story-chat (`local/picture.ts`,
   `local/picture-references.ts`), from the story. A saved picture keeps its recipe, and a variant is drawn from that
   recipe later. Recipes on the gateway would split one decision over two repositories and make a change of the
   bot's picture a release of this one.
2. The experiments change their graphs every round: the reference stand, the action draws, the T probe, the figure
   test, the identity run. Each new graph would wait for a recipe here, and a recipe per experiment would bring the
   domain here after all.
3. ComfyUI checks a graph itself: it validates every node and input against its registry and refuses an unknown
   node with 400. The server starts with `--disable-all-custom-nodes`, so the registry is the pinned core. What is
   left for the gateway is what a graph may touch on the card: the node set, the files it reads, the files it writes.
   That check needs no recipe (section 4).

The cost: a new node in a bot graph needs the reader allowlist extended first. The list is in the picture manifest
(section 9), so extending it is a new preparation of the card, not a change of the bot.

## 3. Routes

| Route                                   | Method | Keys            | Listener        |
|-----------------------------------------|--------|-----------------|-----------------|
| `/v1/pictures`                          | POST   | ours            | public          |
| `/v1/pictures/{job}/stop`               | POST   | ours, the job's | public          |
| `/v1/pictures/uploads/{name}`           | PUT    | ours            | public          |
| `/v1/pictures/uploads/{name}`           | DELETE | ours, the name's| public          |
| `/v1/pictures/inputs/{name}`            | PUT    | ours, `internal`| public          |
| `/v1/pictures/engine`                   | GET    | ours            | public          |
| `/v1/state`                             | GET    | any             | public, control |
| `/v1/control/drain`, `sleep`, `open`    | POST   | the control key | control         |

On the card the public listener is 8094 and the control listener 8095, both on loopback; ComfyUI listens on
127.0.0.1:8188, where `image-serve.sh` puts it. The command's tunnel maps local 8084 and 8085 to them (section 10),
so that the text lane's 8080 and 8081 and the picture lane can be up at once on the owner's machine. "Ours" is a key
whose classes include `reader`, `agent` or `internal`; another key gets 403 `class_not_allowed`. The contract's
rules for unknown paths, trailing slashes, control bodies and the connection check (section 3) hold as they are. A
row about a route with a part of its own (`{job}`, `{name}`) logs the route as written in this table, never the path.

## 4. A picture: `POST /v1/pictures`

### Request

```json
{"graph": {"<node id>": {"class_type": "…", "inputs": {…}}, …}, "copies": ["<node id>", …], "require_socket": false}
```

- Headers as in the contract: `Authorization`, `X-Simple-Serving-Class`, `X-Simple-Serving-Scope` (contract
  section 2). A reader's picture names the reader's scope, as a reader's turn does; it decides whose uploads the
  graph may name (section 5).
- `graph` is ComfyUI's API format, an object of at most 64 nodes (provisional). `copies` (default none, at most 16)
  names output nodes whose pictures are checks of the frame, as the harness's `copies` does today; each must be a
  node of the graph. `require_socket` (default false) refuses the job before it is sent when the gateway cannot hear
  it from its start (below), as the harness's `requireSocket` does today. Any other field is refused as unknown. The
  body is at most 1 MiB.

### Checks before the job goes to the card

In the contract's order: route, key, status, class, scope, body. Then the graph:

- **`reader` and `agent`**: every node's `class_type` is on the reader allowlist, which has no node that saves; the
  `image` of every `LoadImage` is `bot-references/<name> [temp]` for a live upload of the caller's own scope
  (section 5). The loaders name model files, which ComfyUI checks against its own model folders. The allowlist in
  phase 1 is what the bot's graphs hold today, and a test holds it to the copied workflows (section 9):
  `UNETLoader`, `CLIPLoader`, `VAELoader`, `CheckpointLoaderSimple`, `CLIPTextEncode`, `ConditioningZeroOut`,
  `TextEncodeQwenImage21`, `QwenImage21Cache`, `EmptyLatentImage`, `KSampler`, `VAEDecode`, `PreviewImage`,
  `ModelAttentionBackend`, `LoadImage`, `ImageScale`.
- **`internal`**: any node the server has. The `image` of `LoadImage` is a plain name of the input directory,
  `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` without a path (which is how `uploadReference` names them), or an upload of the
  caller's own scope; and any string input that names a ` [temp]` file must be the caller's own upload, so that no
  harness reads a reader's reference or picture by mistake. ` [output]` names are allowed, since only synthetic
  pictures are ever saved there.
- A graph that fails a check is refused with 400 `graph_refused`, before anything reaches the card.
- Then the cap of the class (section 7): past it, 429 `queue_full`. From here the request is the service's work,
  seen by a drain and a stop (section 6).
- Then the gateway waits up to 2 seconds for its socket (below), sets one fresh `nonce` on every `PreviewImage`,
  which is what `freshPreviews` does in the bot today, mints the job's `prompt_id` (a UUID) and registers it with its
  socket, so that no news of the job is missed, and only then submits the graph to ComfyUI under the socket's client
  id, with the job's `number` (below). A submit once sent is never cut short: a stop that comes meanwhile waits for
  its answer and then stops the job by its id (section 4, "Stop").
- ComfyUI's 400 is answered 400 `graph_rejected`; nothing of ComfyUI's answer passes on, since its errors quote the
  graph's inputs. A ComfyUI that does not answer is 503 `engine_unavailable`, after the job is stopped by its id in
  case the submit reached it.
- The answer is 200, and the stream begins, only once ComfyUI has taken the job. So a graph ComfyUI refuses is
  refused before any wait, as today, and the bot's fallback to the plain graph (`local/picture.ts`: a refused
  reference graph before the submit) works as it does now. Every other refusal of the gateway's own (401, 403, 413,
  429, 503) comes before the submit too, and the client maps it to a code that starts no fallback (section 13).

The socket: the gateway keeps one websocket to ComfyUI open for all its jobs, under a client id of its own, and
opens it again with a new id when it drops (the pinned server files a socket under its id alone, and the old
handler's end takes the id's socket with it, server.py:274-276 and 324-326). ComfyUI tells a job's news only to a
socket connected when it tells it, so before a submit the gateway waits up to 2 seconds for its socket to be open. A
socket that is not open by then leaves the job to the polls (progress and phases lost, the picture not), unless
`require_socket` asked otherwise: then 503 `socket_unavailable`, and nothing is submitted.

### Order on the card

The gateway submits each job it accepts at once, as the bot does today, and ComfyUI's own queue orders them. The
pinned server draws the pending job with the lowest `(number, prompt_id)` first and takes a `number` from the submit
(server.py:1081-1088, execution.py:1297-1309; `front: true` only makes its own counter negative). So the gateway
numbers every job itself, from a counter `n` of its own:

- `reader`: −(10⁹ + n), below every other class, and the later reader's picture first, which is what `front` gives
  the bot's jobs today;
- `agent`: n, in arrival order;
- `internal`: 10⁹ + n, in arrival order.

The job being drawn is never interrupted for another. For readers and for the experiments this is today's order
(simple-story-chat's `docs/gpu.md#bot-card`), so what a waiting reader sees does not change; `agent` gets the place
the contract gives it, between them. A test holds the order with several waiting readers, an agent's job and an
experiment's already drawing.

Why not a queue of the gateway's own, handing ComfyUI one job at a time: ComfyUI validates a graph only when it is
submitted. A job held back by the gateway would learn that its graph is refused only after its wait, inside a stream
that has already begun, and the bot's plain-graph fallback would start behind the queue.

### Stream

`Content-Type: text/event-stream`, one `data:` line of JSON per event, each with its name in `event`, then
`data: [DONE]`:

| Event       | Fields                                   | When                                                          |
|-------------|------------------------------------------|---------------------------------------------------------------|
| `accepted`  | `job`, `socket`                          | first, once ComfyUI took the job                              |
| `place`     | `before`, `drawing`, `drawing_ms`        | every 2 s while the job waits, from 2 s after it was taken    |
| `started`   | `stamp` when the socket heard it         | once, when the job no longer waits                            |
| `cached`    | `nodes`                                  | the nodes ComfyUI answered from its cache                     |
| `node`      | `node`                                   | a node began                                                  |
| `progress`  | `value`, `max`                           | the running node's steps                                      |
| `over`      | `heard`, `stamps`, `memory`, `partial_loads` | the job's record is written and names its picture         |
| `image`     | `node`, `png`                            | the frame, then each copy                                     |

- `job` is the job's id for its stop route. `socket`: the gateway's socket was open at the submit, so it hears the
  job from its start unless it drops.
- `place`: `before` counts the pending jobs ComfyUI will draw first, comparing `(number, prompt_id)` as the server
  does; `drawing` says whether it is drawing another; `drawing_ms` how long this job's wait has seen that one drawing,
  as `readPlace` counts them today. The gateway reads ComfyUI's queue once at the submit, so that the first sight of
  the job being drawn is at the job's start, as the bot's first read is today, and then once a second while any job
  waits; all waiting jobs share that read.
- `started` comes when the socket hears the job start (`stamp`, ComfyUI's own clock), or when a read of the queue no
  longer holds it among the waiting, and at the latest before `over`. The client clears the reader's place with it,
  as `onPlace(null)` does today, and times `queueMs` from it only when it has a `stamp`, as today only a socket's
  word does.
- `node` and `progress` pass the socket's messages on as they come, without a time: the client times them on
  receipt, as it times the socket's messages today, and makes its phases from them only when `over` says `heard`,
  the socket having heard the job from its start to its end.
- `over`: `stamps` (`start`, `end`) are ComfyUI's clock at the job's start and success when the socket heard both.
  `memory` is the video memory by device (`index`, `total_mib`, `used_mib_max`, `occupied_mib_max`), `samples` and
  `ram_mib` (`min`, `max`), sampled from `/system_stats` every 500 ms while the job draws, with a last sample taken
  at the over that adds to the maxima and not to `samples`, as the harness's `lastSample` does. `partial_loads`, for
  `internal` alone: the lines of ComfyUI's log that say a model was loaded in part, between a read of the log right
  before the submit and one at the over, as `partialLoadsSince` counts them; `null` when the log could not be read.
  It counts lines of other jobs in that window too, as today. No line of the log leaves the gateway.
- `image`: the first picture of the first output node that is not a copy, then the first picture of each copy in
  the order of `copies`, a copy without a picture skipped, as `follow`'s `fetch` takes them today. `png` is base64 of
  the picture as ComfyUI's `/view` gave it, with every PNG chunk but `IHDR`, `PLTE`, `tRNS`, `IDAT` and `IEND`
  removed, as `stripPngMetadata` does.
- The job's record is deleted (`/history` delete) once the pictures are read, without waiting for its answer, as
  `forget` does today; the pictures do not wait for it, and a stop of the service waits for the deletes still out.

`over` is where the harness's pipeline turns (`local/action-draw.ts` `drawAhead`): the memory and the partial loads
are in it, so the harness submits its next job then, while this one's pictures come down, as it does today.

### Errors

After the stream has begun, an error is the event `{"error": {"code": "…"}}` without `[DONE]`, as in the contract:

| Code                 | Why                                                                          |
|----------------------|------------------------------------------------------------------------------|
| `picture_failed`     | the job's record says it failed, or names no picture; `"oom": true` beside the code when the card ran out of memory |
| `timeout`            | past the class's wall time                                                   |
| `draining`           | a drain or a sleep stopped the job                                           |
| `engine_unavailable` | ComfyUI stopped answering, changed, or lost the job                          |
| `stop_unconfirmed`   | the job had to be stopped and its stop is not confirmed: it may still be drawing |
| `internal_error`     | anything else                                                                |

### Stop

A job that ends without its picture for any reason but `picture_failed`, a client that goes away included, is
stopped as `stopJob` stops it, with ComfyUI's atomic cancel in place of the queue delete and the interrupt:
`/api/jobs/{prompt_id}/cancel` takes the job out of the queue if it waits and interrupts it only while it is the one
being drawn (server.py:947-988, comfy_execution/jobs.py:505-547, execution.py `interrupt_if_running`), where
`/interrupt` checks a snapshot and then stops whatever runs (server.py:1163-1191). Then the stop is confirmed as
today: the job's record is there, or two reads of the queue in a row do not hold it, within ten polls of 500 ms. Only
a confirmed stop deletes the record.

A stop that is not confirmed ends the stream with `stop_unconfirmed`, whatever the stop's reason. The job stays
counted against its class's cap and in `active` (section 6) until a later check finds its record, which is then
deleted, or finds it in neither the queue nor the history twice.

A client's own stop, at its wait's end or when its caller lets go, is `POST /v1/pictures/{job}/stop` with the job's
key and scope: it answers `{"stopped": true}` once the stop is confirmed or the job was over, and
`{"stopped": false}` when it is not confirmed, which the client takes for `comfy_stop_unconfirmed` as `follow` does
today. The job's stream, if its client still reads it, then ends without an event: the answer is the stop route's. A job that is in neither the queue nor the history twice in a row, the queue read first, is lost:
`engine_unavailable`.

## 5. References

### A reader's: `PUT /v1/pictures/uploads/{name}`

`name` is `<uuid>.<ext>`, a canonical lowercase UUID and `png`, `jpg` or `webp`, chosen by the client before it
sends, as `temporaryReferences` does today, so that a client whose answer is lost can still erase it. The body is
the picture's bytes, with `Content-Type` `image/png`, `image/jpeg` or `image/webp` matching both its magic bytes and
the extension, at most 10 MiB, which is `readReference`'s limit.

- The name belongs to the first scope that sends it, recorded before anything is sent to ComfyUI; another scope's
  PUT or DELETE of it is 404. The same scope may send it again.
- The gateway uploads it into ComfyUI's temp directory, subfolder `bot-references`, under that name, checks
  ComfyUI's answer as `temporaryReferences` does, and answers `{"name": "bot-references/<name> [temp]"}`, the exact
  string the bot writes into its graph today.
- An upload that fails on the way, or whose client goes away before the answer, is erased by the gateway at once,
  whatever the client does.
- The upload lives 10 minutes from its PUT, the sweeper's cap on that folder; then the gateway erases it and forgets
  it.

`DELETE /v1/pictures/uploads/{name}`: the caller's own upload is overwritten with a 1×1 blank PNG, which is how the
bot erases it today (ComfyUI has no route that deletes a file); 204. It may be asked again, and while the service
drains or sleeps: an erased name answers 204 until its 10 minutes are over. The sweeper's cap stays below as the
last guard.

### An experiment's: `PUT /v1/pictures/inputs/{name}`

For `internal` alone, into the input directory, as `uploadReference` sends them today: `name` is
`ref-<16 hex>.png`, and the gateway strips the picture's chunks as `stripPngMetadata` does and refuses it with 400
`invalid_request` unless the name is the first 16 hex of the SHA-256 of the stripped bytes. It answers
`{"name": "<name>"}`, and the file stays until the card goes, as today; only synthetic pictures are sent this way.

## 6. Engine facts, state and health

`GET /v1/pictures/engine`, for keys of ours:

```json
{"comfyui": "0.37.0", "pytorch": "2.11.0+cu130", "card": "<device 0's name>", "triton": true, "kitchen_attention": true}
```

The bot asks this where it asked ComfyUI's `/system_stats` and `/object_info/ModelAttentionBackend` before
(`kitchenAttention`, `serverPins`, `attentionOffered`). The answer is the gateway's reading from its last check,
never a request of its own to ComfyUI; `kitchen_attention` is null while the node has not answered. Before the first
check that passed it is 503 `starting`.

`/v1/state` is the contract's, with `"lane": "pictures"` and without `context_tokens`; `model` is the manifest's
`MODEL_ALIAS`. For the control key, `waiting` counts by class the jobs being submitted or waiting in ComfyUI's queue,
and `active` those being drawn, those whose pictures are on their way, and the stops not yet confirmed.

Health, every 5 seconds: ComfyUI's `/system_stats` answers with the engine's fingerprint, its `comfyui_version`, its
`pytorch_version`, whether comfy-kitchen's Triton backend is on (`--enable-triton-backend` in its command line and
`--disable-triton-backend` not, as `tritonOn` reads it) and device 0's name. Before the first `ready` the service is
`starting`, and a check passes when the fingerprint's first three are the manifest's; the gateway then reads
`/object_info/ModelAttentionBackend`, keeps the fingerprint, and is `ready`. A mismatch keeps it `starting`, and the
launcher's load deadline ends it. After `ready`, a check that fails makes it `failed` (accepted jobs go on within
their wall time); a check that answers another fingerprint is an engine that changed: `failed`, every job ends with
`engine_unavailable`, and the service is `ready` again only on the kept fingerprint. The attention offer is read
again whenever the socket opens anew.

ComfyUI cannot start again under a running gateway on the card: the launcher runs the pair once, and the engine's
exit ends it (section 9). A ComfyUI started again by hand, as the rollback does, loses the jobs it held, which end
as lost.

Drain (contract section 8): new jobs and uploads get 503 `draining`; jobs being submitted or waiting in ComfyUI's
queue are stopped at once with `draining`; a job being drawn has the drain's deadline (60 s) to end, and is then
stopped with `draining`; one whose pictures are on their way ends as it would. A stop that is not confirmed ends
with `stop_unconfirmed`. `/v1/state`, the control routes, a job's stop and an upload's erasure answer in every status.

## 7. Limits

Provisional:

| Class      | Jobs on the card | Wall time | Uploads live per scope |
|------------|------------------|-----------|------------------------|
| `reader`   | 32               | 1860 s    | 32                     |
| `agent`    | 4                | 1860 s    | 16                     |
| `internal` | 32               | 3600 s    | 32                     |
| `external` | 0                | —         | —                      |

- Jobs on the card are the class's accepted jobs and its stops not yet confirmed. A job past its class's cap is
  refused with 429 `queue_full`; there is no waiting line in the gateway.
- The wall time runs from the acceptance to the last event. A reader's is longer than the bot's longest wait
  (`SIMPLE_CHAT_IMAGE_WAIT_SECONDS`, at most 1800), so that the bot's own timer ends a reader's picture, as today.
- 64 connections at most on the public listener, as in the contract.

## 8. Sleep, stop and start

Contract section 8 as it stands: only requests of ours hold the card, a job for as long as it runs and an upload for
as long as its request; after 13 minutes without work the service drains, falls asleep and stops its instance through
Vast with the card's own key; the command starts it again. That is the owner's rule 3 of 2026-09-28 (simple-story-
chat's `docs/gpu.md`): no card waits for the tester. A reader who asks for a picture while the card sleeps gets the
failure line the bot shows today when the picture card is down.

## 9. The card

### Files

- `card/pictures/`, copied byte for byte from simple-story-chat's `gpu/` at a named commit: `image-bootstrap.sh`,
  `image-serve.sh`, `image-sweeper.py`, `image-manifest.env`, `image-workflow.json`, `image-workflow-qwen.json` and
  `image-workflow-qwen-edit.json`, the scripts and the files they read beside themselves. `card/pictures/SOURCE`
  names the commit and each file's SHA-256, and a test holds the copies to it, so a change comes only as a new copy
  of a new commit. In phase 3 this copy becomes the source.
- `card/pictures/manifest.env`: `MODEL_ALIAS`, `LOAD_DEADLINE_S` (300), `PUBLIC_PORT` 8094, `CONTROL_PORT` 8095,
  `ENGINE_PORT` 8188, `GPU_DIR` (`/workspace/simple-chat-gpu`, the install today's scripts use), `IMAGE_QWEN=only`,
  `IMAGE_TORCH=cu130`, `IMAGE_TRITON=1`, `IMAGE_GPU=0`, and `READER_NODES`, the reader allowlist. The versions the
  health check expects are the copied `image-manifest.env`'s `COMFYUI_VERSION` and `TORCH_CU130_VERSION`.
- `card/pictures/gateway-requirements.txt`: the gateway's packages with hashes, the text lane's plus `websockets`
  (section 12), exported by the command in its header.
- `card/pictures/bootstrap.sh` and `card/pictures/onstart.sh`, as the text lane's.

### Preparation

`card/pictures/bootstrap.sh` takes the two key hashes on standard input, as the text lane's does, and:

1. writes the keys and builds the gateway's environment from the hashed requirements, in the lane's state directory
   `/workspace/simple-serving-pictures-card`;
2. adopts an install the old scripts left when it is whole: `.venv-cu130` has its ready mark, `image-verified.txt`
   names the pinned ComfyUI revision and the three Qwen files with the manifest's hashes, and the two Qwen graphs are
   there. Nothing of it is touched then: the copied `image-bootstrap.sh` would remove the ready mark, upgrade pip and
   resolve ComfyUI's requirements again. Otherwise it runs the copied `image-bootstrap.sh` with
   `SIMPLE_CHAT_GPU_DIR` from the manifest, `SIMPLE_CHAT_IMAGE_QWEN=only`, `SIMPLE_CHAT_IMAGE_TORCH=cu130` and
   `SIMPLE_CHAT_IMAGE_VIGGLE=false`: the same downloads checked by size and SHA-256, the same hashed wheels, the same
   torch pins, the same check of sm_120, the same `image-verified.txt`;
3. stamps what it prepared: the copies, the manifest and the requirements.

### The launcher

`python -m simple_serving.pictures.card`, the text launcher with a picture engine:

- the engine is the copied `image-serve.sh` with `SIMPLE_CHAT_IMAGE_QWEN`, `_TORCH`, `_TRITON`, `_GPU`, `_PORT` and
  `SIMPLE_CHAT_GPU_DIR` from the manifest, `SIMPLE_CHAT_IMAGE_VIGGLE=false` whatever the environment says (an
  inherited `true` would load Viggle's node), and without the card's own key in its environment;
- ComfyUI runs under `flock -n /root/.simple-chat-comfy.lock`, the lock the old start takes, and holds it for as
  long as it runs, so the two cannot run at once: a start that finds port 8188 taken refuses with 7, and a ComfyUI
  that finds the lock held exits at once, which ends the pair as any engine that exits before ready;
- one pair, no restarts, the load deadline, the given-up and stop-unconfirmed markers, the same exit codes, `--hold`,
  `--stop` and `--inspect`, as for the text lane; its own marker (`simple_serving.pictures.card --run`) and its own
  state;
- ComfyUI's lines are reduced to categories (out of memory, CUDA error, a model loaded in part, the attention
  node's fallback to PyTorch's attention, exception, traceback, address taken, and `image-serve.sh`'s refusals to
  start on an install that is not whole or a temp directory off tmpfs), each logged once and never kept; the
  sweeper's rows (`sweeper_started`, `swept`, `records_undeleted`, `history_unreadable`, `history_readable`,
  `sweep_failed`, `sweeper_stopped`) pass as rows of their own, with only their counts as integers and their codes
  from the sweeper's closed set, whatever else a line holds.

`card/pictures/onstart.sh` starts the launcher when the lane is prepared and exits 0 otherwise.
simple-story-chat's `gpu/trial-onstart.sh` gets one line, before the text lane's, that runs it when it is there.

## 10. The command

`python -m simple_serving.cli --lane pictures up | sleep | status | trial --ssh-host HOST`. The keys are the same two
for both lanes. The lane's instance is `pictures` in the configuration: `{"pictures": {"instance_id", "vast_api_key",
"ssh_host"}}`, written by `trial --lane pictures`. `up` holds a lock of its own, opens local 8084 and 8085, and runs
the picture launcher's `--hold`. Without `--lane` everything is as before.

## 11. Privacy and logs

The rule is simple-story-chat's (`docs/gpu.md#story-text-on-a-rented-card`): no story text, prompt, graph, picture,
upload name, job id or line of ComfyUI in any log, answer to a stranger or file of ours. Pictures pass through the
gateway's memory only; ComfyUI's temp directory stays on tmpfs; the sweeper stays; the job's record is deleted as
soon as the pictures are read. Nothing of ComfyUI's answers passes into an error of the gateway's.

New fields of the request row, each an enum, a boolean or a count: `nodes`, `images`, `image_bytes`,
`upload_bytes`, `references`, `queue_ms`, `draw_ms`, `partial_loads`, `oom`, `stop_confirmed`, `socket`. The
launcher's rows add the sweeper's counts and codes and the engine's categories. Tests send a refused graph, a graph
ComfyUI rejects with its inputs quoted, a job whose engine raises, a malformed upload and a forged sweeper line, each
carrying a marker, and find the marker in no log row and no answer. On the card, a picture whose prompt holds a fresh
random marker is drawn, and the lane's `logs/` are searched for it, counts only, as the text lane's `privacy` probe
does (README, 'The smoke').

## 12. The new dependency

`websockets` 17.1, pinned in `uv.lock` with its hashes, in a dependency group `pictures` that `dev` includes. ComfyUI
tells a job's start, its nodes, its cache and its steps only over its websocket, and the reader's progress line and
the experiments' phases are made of them; the polls alone give the picture and nothing of its course. It has no
dependencies of its own, and a wheel for CPython 3.12 on manylinux x86_64. The text lane's
`card/gateway-requirements.txt` does not change, so no text card needs preparing again; the picture lane's
requirements are exported with `--group pictures`. Tests hold that the text requirements are still `uv.lock`'s
runtime packages, that the picture requirements are the text ones plus `websockets` alone, and that the text gateway
starts with `websockets` impossible to import.

## 13. What changes in simple-story-chat

- A switch, off by default: `SIMPLE_CHAT_IMAGE_PROVIDER=comfyui` (the default) or `simple-serving`, with
  `SIMPLE_CHAT_IMAGE_API_KEY`; `SIMPLE_CHAT_IMAGE_URL` is then the gateway, `http://127.0.0.1:8084`. Off, nothing the
  bot sends, logs or shows changes, byte for byte (section 14).
- On, the bot's `Comfy` carries `{key, class: reader, scope}` with the reader's scope, and `drawOne`,
  `temporaryReferences`, `kitchenAttention` and `serverPins` go through the routes above. The graphs are the same,
  the fallback to the plain graph is the same, and a waiting reader sees the same place and progress.
- The client keeps today's result: `bytes`, `totalMs` from before the request to the last picture (the harness's
  without the time between `over` and its own `fetch`, as today), `viewMs` from `over`, or the harness's `fetch`, to
  the frame, `queueMs` to a `started` with a `stamp`, the phases from the receipt of `node` and `over` when `heard`,
  `loaderCacheMiss` from `cached`, `stamps`, `memory` and `vram`, the frame and the copies. `onSubmitted` fires at
  `accepted`, `onPlace` at each `place` and with nothing at `started`, `onProgress` at each `progress`.
- Errors map to today's codes, so that what the bot does with them stays: `graph_rejected` and `graph_refused` to
  `comfy_rejected_prompt`, which is the only refusal that starts the plain-graph fallback; any other refusal before
  the stream to `serving_refused` with its HTTP status and the gateway's code in `servingCode`, which the fallback
  does not take and which stops a harness's run; `socket_unavailable` to `comfy_socket_unavailable`; `timeout` to
  `image_timeout`; `picture_failed` to `image_failed`, with `oom`; `stop_unconfirmed` to `comfy_stop_unconfirmed`;
  `draining`, `engine_unavailable` and `internal_error` in the stream, and a stream that ends without an error and
  without `[DONE]`, to `comfy_connection_lost`. A reference upload that fails is `reference_upload_failed`, as
  today. `serving_refused` and the new gateway codes go into the log field whitelist first.
- The experiments: `--comfy serving` in place of a URL draws through the gateway at `http://127.0.0.1:8084` under
  class `internal`, with the client key from simple-serving's configuration, as the eval does for text. The staged
  path (`submitOnStage`) keeps its pipeline over the stream (section 4, `over`), and `partialModelLoadEvents` comes
  from `over`'s `partial_loads`. A dropped connection to the gateway loses the cell as `comfy_connection_lost`, which
  stops the run (open question 3). The kitchen's startup lines (`kitchenOf`) and the log lines a pilot observes stay
  unread through the gateway. The engine-variant harnesses, `image-pilot` (Viggle's node) and `image-levers` (other
  torch, attention and flags), need a server started their way and refuse `serving`.

## 14. The dry run

No card. The bot runs against simple-story-chat's fake ComfyUI (`local/fake-comfy.ts`), extended for this: the
submit's `number` orders its queue, an upload's answer says its `type` and `subfolder`, `/api/jobs/{id}/cancel`
cancels as the pinned server does, and the versions and command line `/system_stats` reports are set by the test.
With a fake Telegram and a fake model, one synthetic story draws a frame, a portrait, a style sample, a variant and a
reference frame. Every request that reaches the fake ComfyUI is captured with its graph, and every Telegram call and
log row; ids, times and the nonce are normalized before the comparison:

1. the parent commit and this change with the switch off, both against the extended fake: the graphs ComfyUI
   received, every request to it, every Telegram call and every log row must be the same;
2. the switch on, through this gateway's dev launcher in front of the same fake: the same pictures reach Telegram,
   the graphs ComfyUI received are the same as in 1 apart from the nonce, and the rows say `ready`.

## 15. Phases

1. This change: the picture lane here (config, engine client, routes, checks, uploads, launcher, preparation,
   command) against a fake ComfyUI; the switch and the client in simple-story-chat, off by default; the dry run.
   Written on 2026-09-28 on the branches `pictures` here and `pictures-via-serving` there, with the copies of
   simple-story-chat's commit ddd2850 (`card/pictures/SOURCE`).
2. On a card: the card test (below); then, with the owner's word, the switch on for the owner's chat, then for the
   tester.
3. When the switch has been on for a while without a fallback: simple-story-chat's `gpu/image-*` scripts leave, and
   `card/pictures/` becomes their source. The switch's `comfyui` side stays as long as the experiments' engine
   variants need a server of their own.

### The card test

One 5090 rented for it, with a CUDA 13 driver, about 75 minutes (about $0.6–0.9 at today's offers), inside the
owner's $3 a day and the rules of simple-story-chat's `docs/gpu.md#while-the-cards-are-paid-for`, backed first by a
fresh GPT-6 Astra at `xhigh` as rule 4 asks. The rental has its external deadline set before anything else runs (the
trial guard), and ends with the instance destroyed and the destroy confirmed by Vast, after a failed step too.

| Step | What | Minutes |
|------|------|---------|
| 0 | the container: `memory.swap.max` is 0 and its RAM limit is what the RAM-pressure rule assumes (`docs/gpu.md` "What the card keeps of a picture"); otherwise the card is destroyed and no reader draws on it | 2 |
| 1 | `card/pictures/bootstrap.sh`: weights (17.3 GB), `.venv-cu130`, `image-verified.txt` | 15 |
| 2 | `up --lane pictures`; `/v1/state`, `/v1/pictures/engine` against the manifest, `kitchen_attention` true | 5 |
| 3 | an `internal` synthetic batch through `--comfy serving`, its pins and phases, and an `action-draw` stage with its pipeline | 8 |
| 4 | the bot with the switch on in the owner's chat, a synthetic seed: frame, portrait, sample, variant, reference frame; rows `ready` with `pictureAttention` `kitchen`, and no `attention_fallback` among the launcher's categories (`docs/gpu.md#bot-card`'s check) | 10 |
| 5 | a picture with a random marker in its prompt; the lane's `logs/` searched for it, counts only | 3 |
| 6 | a reader's picture stopped while waiting and while drawing: stopped, record gone; a reference erased and then swept, a preview swept after its record, a cleanup that fails (ComfyUI paused) left to the sweeper's cap, the sweeper's rows | 8 |
| 7 | 13 minutes idle: the card stops itself; `up` resumes it, a new boot | 18 |
| 8 | the fallback: the launcher's `--stop`, the old `image-serve.sh` under its lock, the switch off, one picture | 5 |
| 9 | destroy, confirmed | 1 |

### Rollback

On the card: `python -m simple_serving.pictures.card --stop`, then the old start as simple-story-chat's
`docs/gpu.md` writes it (`setsid -f nohup flock -n /root/.simple-chat-comfy.lock bash …/image-serve.sh`) on the same
install, with no new download; the old tunnel to 8188; the switch off; the bot restarted. Nothing on the card is
removed by this move, so the old path stays whole until phase 3.

## 16. Review

A GPT-6 Astra session at `high` read this design on 2026-09-28 against the code of both repositories and the pinned
ComfyUI: "sound with changes". It agreed that graphs are sent through, with class checks that look at the files a
graph touches as carefully as at its nodes. Its fifteen findings are in the text above:

1. `/interrupt` is not atomic: the stop uses `/api/jobs/{id}/cancel` (section 4, "Stop").
2. An unconfirmed stop reaches the client as `stop_unconfirmed`, the record is deleted only after a confirmed stop,
   and the job stays counted until it resolves (section 4).
3. Waiting and being drawn are tracked apart for the drain, the job is registered before its submit, and the job
   stopped at the drain's deadline gets `draining` (sections 4, 6).
4. The numbering was right; a test holds the mixed order (section 4).
5. `started` clears a waiting place without the socket, and a read of the queue at the submit starts `drawing_ms`
   where the bot's does (section 4).
6. The client's result and its times are written out, the frame and the copies picked as today, and the record's
   delete does not hold the pictures (sections 4, 13).
7. `action-draw`'s pipeline is kept: `over` carries the last sample and the partial loads (sections 4, 13). Nothing
   of it needs the owner.
8. `onSubmitted` fires at `accepted`, the gateway's own refusals start no fallback, `oom` rides in the error, and a
   specific error is kept (section 13).
9. Upload names are the client's, owned before they are sent, erased by the gateway when an upload fails, and
   erasable during a drain (section 5).
10. The fake ComfyUI is extended for both runs of the dry run (section 14).
11. The health check compares the whole fingerprint, Triton as the server reads its flags and the card included,
    and reads the attention offer again with each new socket (section 6).
12. The card test checks swap and RAM before readers, and the cleanup of references, records and previews,
    including a failed one; the launcher maps `history_readable` (sections 9, 15).
13. The preparation adopts a whole install without running the old bootstrap, passes `GPU_DIR`, and keeps Viggle
    off (section 9).
14. The privacy tests cover refusals, rejections, exceptions, uploads and forged sweeper lines, in answers as in
    logs (section 11).
15. The dependency is tested apart from the text lane's, and the card test has the attention check, the budget, the
    deadline and a confirmed destroy (sections 12, 15).

## Open questions for the owner

1. Readers among themselves: today the later picture goes first (ComfyUI's `front`), which this keeps. First come,
   first served is a change of the readers' numbers to −10⁹ + n; it changes which reader waits when two wait.
2. The engine-variant experiments (`image-pilot`, `image-levers`) keep a ComfyUI of their own on a card of their own,
   outside simple-serving. Or should the picture lane learn to start variants?
3. A harness whose tunnel drops loses its jobs with the connection: the gateway stops them as it stops a reader's.
   Today's harness rides a dropped connection out and finds its job again. Is losing the cell acceptable, or should a
   job be kept for a client that comes back?
4. Phase 3: may `gpu/image-*` leave simple-story-chat once the switch has been on for a week without a fallback?

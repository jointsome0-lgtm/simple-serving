# simple-serving

Text model serving for [simple-story-chat](https://github.com/jointsome0-lgtm/simple-story-chat): a FastAPI gateway in
front of vLLM, run on rented GPUs. The bot calls it over HTTP. Outside clients with a key may call it too.

Status: the gateway of contract v2 is written and tested against a fake engine. It has not run in front of vLLM yet;
that happens on the first rental (contract section 15), whose smoke is written and dry-run against the fake engine.
The API is in [docs/contract-v2.md](docs/contract-v2.md), the shared cases in [contract/](contract/README.md).

Version 2 serves one text model. Pictures stay in simple-story-chat for now.

## Install

Python 3.12 or later and [uv](https://docs.astral.sh/uv/):

```
uv sync
```

The first sync needs the network; after it everything, the tests included, runs offline. `uv.lock` pins FastAPI,
Starlette, uvicorn and httpx, the gateway's part of contract section 12.

## Tests

```
uv run pytest
uv run ruff check .
uv run mypy .
```

No network, GPU or vLLM. The tests start the gateway in uvicorn on loopback ports, in front of a fake vLLM
(`simple_serving/fake_engine.py`), and call it over HTTP:

- `tests/test_cases.py`: every case of `contract/cases-v2.json`, each on a new gateway;
- `tests/test_scenarios.py`: the scenarios of the cases file: a client that leaves in every phase, places and lines,
  counts, wall time, drains;
- `tests/test_privacy.py`: the gateway as its own process, started as in production; every error path carries a marker
  that no answer and no log line may hold;
- `tests/test_gateway.py`: the engine's health and context, what the engine receives, errors inside the gateway;
- `tests/test_sleep.py`: what holds the service awake, the idle interval, the sleep and the stop of the instance, on
  a fake clock and a fake Vast;
- `tests/test_card.py`: the card's launcher, what it keeps of vLLM's output, and the preparation and onstart scripts,
  with stand-ins for vLLM, the gateway, pip, aria2c, curl and flock;
- `tests/test_cli.py`: the command on the owner's machine, against a fake Vast and a fake SSH that forwards to the
  gateway;
- `tests/test_smoke.py`: the first rental's smoke: the privacy probe's verdict, and the whole smoke through the dev
  launcher, whose output holds no key, no prompt and nothing of the model's text;
- `tests/test_units.py`, `tests/test_dev.py`: the pieces one by one, and the dev launcher.

Three kinds of test stay apart (contract section 14). A client's adapter runs the public steps of the cases in its own
repository. The control routes, the sleep and the card are tested here alone. An opt-in test in simple-story-chat runs
the bot's real adapter against this gateway over the fake engine.

The dev group of `pyproject.toml` pins ruff and mypy, and the same file holds their settings, so every machine runs
the same checks.

## Dev launcher

The fake engine and the gateway on loopback, to run a real client against the real gateway:

```
uv run python -m simple_serving.dev --config contract/cases-v2.json --engine-port 8200 --public-port 8201 --control-port 8202
```

It prints, once the gateway is ready:

```
simple-serving dev launcher: ready
  fake engine  http://127.0.0.1:8200
  delays       0 ms before the first event, 0 ms after each event
  public       http://127.0.0.1:8201
  control      http://127.0.0.1:8202
  model        test-model, context 4096 tokens
  keys         bot (reader, agent, internal, scopes); control (control); outside-a (external); outside-b (external)
Ctrl-C stops both.
```

and writes its log to standard error, one JSON object per line. `--config` names a `service` block in the shape of the
cases file, with its test keys in the clear, or a file that holds one under `service`, such as the cases file itself.
With the cases file the bot's key is `test-key-bot` and the control key `test-key-control`. Fields a file leaves out
take the provisional values of contract section 7, with the model `dev-model` and a context of 65 536 tokens, so a file
with keys alone gets the full limits:

```json
{"keys": {
  "dev-key-bot": {"label": "bot", "classes": ["reader", "agent", "internal"], "default": "internal",
                  "scopes": true, "control": false},
  "dev-key-control": {"label": "control", "classes": [], "default": null, "scopes": false, "control": true}}}
```

The fake engine counts a prompt as the characters of all message contents divided by 4, rounded up, and answers every
generation with a fixed synthetic sentence in a few chunks, then `stop` and its usage. With thinking on, a fixed
thought comes first, in the reasoning field. `max_tokens`, counted the same way, cuts the answer and ends it with
`length`, and a schema with a `pattern` that is not a regular expression gets 400 before any stream. It answers in
milliseconds, so requests rarely overlap. `--first-event-delay-ms N` makes it pause N ms before the first event of each
generation, as for a long prompt, and `--event-delay-ms N` after each event; with them, concurrent requests fill the
places and wait. Both default to 0. Like the card's, the gateway falls asleep after 780 seconds without a request of
ours (contract section 8); it has no instance to stop, so it then stays drained until the launcher starts again.

## Running the gateway

```
SIMPLE_SERVING_CONFIG=/path/to/gateway.json uv run python -m simple_serving
```

from the repository. The configuration file stays outside the repository. It holds each key as the SHA-256 of the key,
never the key:

```json
{
  "alias": "the model's served name",
  "engine_url": "http://127.0.0.1:8000",
  "listen": {"public": {"host": "127.0.0.1", "port": 8080}, "control": {"host": "127.0.0.1", "port": 8081}},
  "context_tokens": null,
  "keys": [
    {"sha256": "64 hex digits", "label": "bot", "classes": ["reader", "agent", "internal"], "default": "internal",
     "scopes": true, "control": false},
    {"sha256": "64 hex digits", "label": "control", "classes": [], "default": null, "scopes": false, "control": true}
  ],
  "limits": {
    "reader": {"active": 4, "waiting": 8, "input_tokens": null, "max_tokens": 8192, "wall_s": 300},
    "agent": {"active": 1, "waiting": 2, "input_tokens": null, "max_tokens": 8192, "wall_s": 900},
    "internal": {"active": 2, "waiting": 8, "input_tokens": null, "max_tokens": 8192, "wall_s": 900},
    "external": {"active": 2, "waiting": 4, "input_tokens": 8192, "max_tokens": 1024, "wall_s": 120},
    "external_per_key": {"active": 1, "waiting": 2},
    "shared": {"active": 4}
  },
  "count_limits": {"active": 8, "per_outside_key": 2},
  "engine_priority": {"reader": 0, "agent": 1, "internal": 2, "external": 3},
  "drain_deadline_s": 60,
  "versions": {"vllm": "the pinned version"}
}
```

`context_tokens` is the gateway's own limit, or `null` for the engine's. Optional fields: `body_limit_bytes`
(2 000 000), `max_connections` (64), `idle_timeout_s` (780), `health_interval_s` (5) and `versions`, which the control
key sees in `/v1/state` next to the gateway's own. An outside key has the classes `["external"]` and no scopes. To
hash a key without printing it:

```
read -rs KEY && printf '%s' "$KEY" | sha256sum && unset KEY
```

When the gateway falls asleep it stops the Vast instance it runs on, with the `CONTAINER_ID` and
`CONTAINER_API_KEY` that Vast puts in the container's environment. Without them every stop fails as `unconfigured`,
and the gateway stays drained.

On the card both listeners stay on loopback, reached over the SSH tunnel; a TLS proxy in front of the public listener
comes with outside keys. The gateway refuses to start unless `listen.control.host` and the host of `engine_url` are
loopback IP addresses, in 127.0.0.0/8 or `::1`; a name, even `localhost`, is refused. vLLM listens on loopback with
the served model name equal to `alias`, priority scheduling (`--scheduling-policy priority`), prefix caching,
`--enable-prompt-tokens-details` for cached tokens, a reasoning parser that puts the model's thinking in a field of its
own, a schema's JSON from xgrammar alone, and its request logging off (contract section 10).

## The card

`card/` holds what runs on a rented card: `manifest.env` with the pins and parameters, the two locks, `bootstrap.sh`
for the preparation, and `onstart.sh`, which the rental's own onstart runs at every start. The checkout goes to
`/workspace/simple-serving`, and the card keeps its state in `/workspace/simple-serving-card`: the key hashes, the
weights, the two venvs and the logs.

The preparation, once per rental and never at a resume:

1. Copy the checkout to `/workspace/simple-serving`.
2. `uv run python -m simple_serving.cli keys | ssh <card> bash /workspace/simple-serving/card/bootstrap.sh`. `keys`
   makes the client key and the control key once, and prints only the SHA-256 of each, one per line.

bootstrap.sh installs each lock into a venv of its own, wheels only and every hash checked, and runs `pip check`
there. It fetches the weights and the tokenizer files at their pinned revisions over 16 connections with aria2c,
which it installs with apt-get when the card lacks it, and checks their hashes. It keeps the key hashes in
`keys.json` and starts the service. It refuses while a pin or a lock is missing, or when vLLM's lock is for another
version than the pin, and it never replaces the keys the card holds.

The locks, `card/gateway-requirements.txt` and `card/vllm-requirements.txt`, each name in their header the command
that made them: the gateway's is exported from `uv.lock`, and vLLM's is resolved for the pinned version. A change of
`uv.lock` or of the pin needs a new lock.

Speculative decoding is off until `MTP_SPECULATIVE_TOKENS` in `card/manifest.env` is above 0. Then vLLM runs Gemma 4's
multi-token prediction, `--speculative-config` with method `mtp`: Google's assistant for the 31B, pinned in the
manifest as the weights are, drafts that many tokens a step, and the heretic verifies every drafted token in one pass
and keeps each only as its own sampling would, so the distribution of the answers does not change, only their speed.
The drafter reads the heretic's KV cache and keeps none of its own; its 0.9 GB of weights come out of the cache's
share of the memory. `/v1/state` then names the drafter's revision and the count among the versions, and so does the
smoke's record.

To turn it on for a card that is already prepared, set `MTP_SPECULATIVE_TOKENS=3` in the checkout, copy the checkout
to the card again, and run the preparation again with nothing on stdin, since the card holds the keys. It fetches the
drafter into `drafters/<revision>` of the card's state and checks every hash again, while the pair it started before
serves on. Then end that pair and start the card, which loads the heretic with the drafter:

```
ssh <card> bash /workspace/simple-serving/card/bootstrap.sh < /dev/null
ssh <card> 'cd /workspace/simple-serving &&
  /workspace/simple-serving-card/gateway/bin/python -m simple_serving.card --stop &&
  /workspace/simple-serving-card/gateway/bin/python -m simple_serving.card'
```

`--stop` ends the tunnel's hold too, and a new `up` confirms that the card is ready. `MTP_SPECULATIVE_TOKENS=0` and the
same steps turn it off; the drafter's directory stays.

At every later start of the container the rental's own onstart starts the service. Its last line runs the service's
`onstart.sh` from the persistent disk, once the preparation has put it there:

```
if [[ -f /workspace/simple-serving/card/onstart.sh ]]; then bash /workspace/simple-serving/card/onstart.sh; fi
```

The preparation changes no onstart, since the platform may restore `/root/onstart.sh` from the rental's own at a
start. `onstart.sh` keeps the instance's id and key for SSH sessions and starts the launcher,
`python -m simple_serving.card`. The launcher runs vLLM and the gateway once, with no restarts, and when the pair
fails it stops the instance; its options and exit codes are in `simple_serving/card.py`. The tunnel's remote command
waits while the pair runs:

```
cd /workspace/simple-serving && /workspace/simple-serving-card/gateway/bin/python -m simple_serving.card --hold
```

A load that never becomes ready leaves the file `given-up` in the card's state, and the card has given up. Every
later start, a resume included, then loads nothing: the launcher waits 13 minutes for the owner's retry and otherwise
stops the instance again. `up` and `status` say so. The retry removes the file:

```
ssh <card> 'cd /workspace/simple-serving &&
  /workspace/simple-serving-card/gateway/bin/python -m simple_serving.card --retry'
```

A launcher that still waits looks for that every 2 seconds and runs the pair, so a retry in the last moments may lose
to the 13 minutes; once the launcher stops the instance, the retry is deferred to the next start. `up` is what
confirms that the card is ready.

An attempt to stop the instance that fails, the launcher's or the gateway's, leaves the file `stop-unconfirmed` and
the log row `stop_unconfirmed` until the launcher's next start, and `up`, `sleep` and `status` say that the card's stop
is not confirmed: costs may go on. The attempts go on every 30 seconds, but while Vast refuses the container's key,
401 or 403, none succeeds.

The service stops its card and never deletes it. A disposable trial rental's onstart is simple-story-chat's
`gpu/trial-onstart.sh` with the line above at its end: it adds the owner's SSH key when one is given, writes the
instance's id and key, which the first preparation needs, and arms a guard that deletes the instance three hours after
the first start: rent.mjs's default, `--hours 3`, which the first rental uses. The guard deletes with the container's
key, as the card stops, so a key that Vast refuses defeats both. An onstart for a permanent rental, which writes the
two files, arms no guard and ends with the same line, is decided before permanent use.

## The command

On the owner's machine, `python -m simple_serving.cli` starts the card, holds the tunnel and asks for sleep:

```
uv run python -m simple_serving.cli up | sleep | status | keys | trial --ssh-host HOST
```

- `up` resumes the instance if it is stopped, waits for the gateway to be ready, and holds the tunnel in the
  foreground: `http://127.0.0.1:8080` for the clients, `http://127.0.0.1:8081` for control. Ctrl+C closes it and sends
  nothing. The card falls asleep by itself 13 minutes after the last request of ours, and `up` then ends. One `up`
  runs at a time for a configuration's directory: its lock guards the fixed local ports, not a card.
- `sleep` makes the card fall asleep now, through a short control-only forward of its own, and waits until Vast
  reports the instance stopped.
- `status` shows the instance's state in Vast and, through a forward of its own, the gateway's status, whether it
  serves the manifest's model, and its counts. Nothing of an answer is printed as it came.
- `keys` makes the two gateway keys once, for the preparation above.
- `trial --ssh-host HOST` is for a trial rental. It reads the card's instance id over SSH and writes it, with the
  host, into the configuration, printing neither.

The configuration is `~/.config/simple-serving/config.json`, open to its owner only, or the file that `--config`
names:

```json
{"instance_id": "...", "vast_api_key": "...", "ssh_host": "...", "client_key": "...", "control_key": "..."}
```

`keys` writes the last two. `vast_api_key` is a Vast key allowed GET and PUT on that instance alone, never the account
key, and `ssh_host` a host of `~/.ssh/config` whose host key is known. On a trial rental `trial` writes `instance_id`
and `ssh_host`, and `vast_api_key` stays the owner's key: Vast answers the card's own container key on the card, and
refuses it from the owner's machine with 401 (measured 2026-09-25). The bot's model profile takes the address
`http://127.0.0.1:8080` and the client key; it holds neither the control key nor a Vast key.

## The smoke

`python -m simple_serving.smoke` is the first rental's smoke (contract section 15): the smoke probes and the count
matrix. It is a client of the gateway through `up`'s forward, with the client key and the control key of the
command's configuration, or of the file that `--config` names. Every request is of class `internal`. What the gateway
does not say, the card's own report does: `card.py --inspect`, which the smoke runs over SSH as `up` reaches the card,
with the configuration's `ssh_host` and `up`'s options, and which prints only numbers and flags.

```
uv run python -m simple_serving.smoke [--only NAME[,NAME]] [--before FILE | --after FILE]
```

The probes run in this order, each within a time bound of its own, and the first that fails ends the smoke:

- `lifecycle`, with `--before` or `--after` alone, around the stop and the resume of "The first rental" below: exactly
  one launcher runs, with its pair, and the trial guard's deadline is there. `--before` writes a hash of the boot and
  the deadline into a new file, open to its owner, and `--after` checks that the boot is new and that the deadline is
  the one in that file.
- `state`: `/v1/models` and `/v1/state` serve contract 2, `ready`, with the manifest's alias and context, and the
  control key sees the manifest's pins and the versions of the gateway's lock.
- `completion`: one short completion, and its stream as section 4 has it: the alias in every chunk, one finish, one
  usage chunk, then `[DONE]`.
- `fields`: each optional field of section 4 alone, at a legal value, and the answer with `response_format` follows
  its schema. Beside them, two answers with one seed at temperature 0.8, compared by hash: an observation, which
  passes nothing.
- `reasoning`: thinking on gives reasoning and an answer, thinking off an answer alone.
- `finish`: `length` after exactly 8 tokens, and `stop`.
- `refusal`: a schema whose `pattern` does not compile is 400 `invalid_request`, before any stream. The comment at
  `UNCOMPILABLE` in `simple_serving/smoke.py` says why vLLM 0.30.0 refuses it.
- `abort`: a stream of a count that the class's 8192 tokens cannot end within the probe, closed after its first event,
  frees its place, and the gateway's row of it on the card, of which the probe reads the cancelled flag alone, says
  that it was cancelled. A request that ended by itself shows no cancel and fails as `inconclusive`. A drain then
  finds no work of ours left, counts included, and an open undoes it, read back as `ready`. Of the probe's 120
  seconds, 30 are kept for that undo, which runs even when the probe ran out of its own time.
- `schemas`: the bot's own JSON schemas in strict mode, each with a synthetic prompt of the smoke's, and each answer
  checked against its schema, `minLength`, `maxLength`, `minItems`, `maxItems` and `pattern` included.
  `simple_serving/bot_schemas.json` copies them with the commit and the file of each in simple-story-chat.
- `counts`: the count matrix. For plain text, a system and a user turn, system, user, assistant and user turns, a
  schema, thinking on, thinking off, and a prompt just below the context, found by counting: the count, then a
  generation of a few tokens, whose `usage.prompt_tokens` must equal the count.
- `privacy`: one completion whose prompt holds a fresh random marker, which no file in the card's `logs/` may hold:
  the gateway's rows, the launcher's rows with what it keeps of vLLM's output, the older file beside each, and
  anything else there. The probe counts the marker in each and prints only the counts. One occurrence fails it, and
  so does a request that left no row of its own, since a log that holds nothing proves nothing. This is the marker
  half of step 2 of contract section 15. On the card it stands for the check that request and output logging is off:
  whatever vLLM prints, nothing of a request may reach a log. The cache-scope half is not needed for this trial, which
  has one internal client.

A probe of parts stops at its first part that fails. On a card that works the whole smoke should take minutes; its
bounds add up to just over an hour. It prints one JSON object per probe, read as "The first rental" below says, and
ends with 0 when every probe passed, 1 at the first that failed, and 2 when it could not run, before it sent anything.
Not verified on the card: an engine error in the middle of a stream, which the smoke cannot bring about. The gateway's
side of it is the contract case `engine-breaks-mid-stream`, against the fake engine.

The dry run is `tests/test_smoke.py`. It starts the dev launcher configured as the card's gateway, with the manifest's
alias, context and pins and the cases' test keys, and runs the whole smoke through it with `--fake`. `--card-dir`
names a directory on this machine that stands for the card: the launcher's rows go to its `logs/gateway.jsonl`, and
its `proc/` and the guard's deadline file stand for the card's processes and `/root`. The fake engine cannot answer
three things truthfully: whether an answer follows its schema, whether a count equals the usage, which it computes
the same way, and whether a seed repeats. `--fake` leaves those unchecked and names them in the line's
`not_verifiable`, it refuses the forward's ports 8080 and 8081, and it reads the card's report only from `--card-dir`,
so it never runs against a card. The rest runs as it would on the card. The tests check that nothing the smoke prints
holds a key, a prompt or the fake's text, and fail the privacy probe on a marker planted in any log and on a request
that left no row. Whether vLLM thinks, cuts at `max_tokens` and refuses the schema as the fake does, only the card
shows. By hand, with such a launcher block in `dev.json` and the smoke's two keys in `keys.json`, open to its owner
only:

```
mkdir -p logs/dry-run/logs
uv run python -m simple_serving.dev --config dev.json --engine-port 8200 --public-port 8201 --control-port 8202 --event-delay-ms 40 2>>logs/dry-run/logs/gateway.jsonl
uv run python -m simple_serving.smoke --config keys.json --public-port 8201 --control-port 8202 --fake --card-dir logs/dry-run
```

The pause after each event holds a stream open while the abort probe looks for it.

## The first rental

The card stops itself with the container's key, and the trial's guard deletes with the same key, so a key that Vast
refuses leaves nothing on the card that bounds the costs. Before the launcher runs, the card has not even its own
stop: SSH may not answer, and bootstrap.sh may hang in a download, or end with 1. On the first rental the operator
is that bound: the Claude session that runs the rental, from the moment the instance is created. It reaches the
instance with the owner's account key alone, through simple-story-chat's rent tool, where `ID` is the instance's id:
`npm run gpu:rent -- --show ID` reads it once, and `npm run gpu:rent -- --destroy ID` deletes it and reads it back
until it is gone, or says within five minutes that it could not confirm that. The account key stays in
simple-story-chat's `.env.gpu`, which rent.mjs reads, and is never copied into this repository's configuration.

1. Before the instance is created, the owner chooses the time of day by which the first attempt, the preparation
   included, is over. This README sets none.
2. Before anything long, once the prepared card is ready: `sleep`, which must end with `vast: stopped`. Then `up`,
   a resume with no start over SSH: the card boots anew and runs one pair, which `up` reports ready.
3. Every stop is confirmed from outside, with the owner's own key: `status` prints `vast: stopped`, and
   `npm run gpu:rent -- --show ID` reads the instance with `intended` `stopped` and `actual` `stopped` or `exited`.
   `sleep` waits 10 minutes for that. After the last request of an idle card, `stopped` must come within 25 minutes:
   13 to fall asleep, 2 for the drain, 10 for the stop.
4. Approved in advance, and done at once: at the end of the attempt, whatever its outcome, after the internal work
   below or after a step that failed, and before it when the preparation fails, when SSH or a command stops
   answering, when the attempt's deadline passes, when `stopped` does not come within those 25 minutes, when a
   command says that the card's stop is not confirmed, when `sleep` ends without `vast: stopped`, or when Vast refuses
   a key with 401 or 403, the operator deletes the trial instance with `npm run gpu:rent -- --destroy ID` and its
   read-back, as simple-story-chat's identity experiment ends each rental. A stopped trial is never left to bill its
   disk. When that read-back does not confirm the deletion, the operator tells the owner at once.

Step 1 of contract section 15 runs in two terminals on the owner's machine, from the repository, once the preparation
of "The card" above has started the service:

1. Once SSH to the card works, and before the first `up`: `uv run python -m simple_serving.cli trial --ssh-host HOST`,
   where `HOST` is a host of `~/.ssh/config` whose host key is known. It writes the card's instance id and the host
   into the configuration, whether or not `keys` has run, and keeps what the file already holds, the owner's
   `vast_api_key` included. The smoke reaches the card's report with that host.
2. First terminal: `uv run python -m simple_serving.cli up`, until it says `ready`.
3. Second terminal:

   ```
   mkdir -p logs && uv run python -m simple_serving.smoke --only state,completion --before logs/lifecycle.json
   ```

   One pair and the guard's deadline, which it writes with a hash of the boot into that new file, then the gateway's
   state and pins, and one short completion. `--before` refuses a file that exists.
4. `uv run python -m simple_serving.cli sleep`, which must end with `vast: stopped`; `up` in the first terminal ends
   too. `uv run python -m simple_serving.cli status` then prints `vast: stopped`, and `npm run gpu:rent -- --show ID`
   in simple-story-chat reads the instance stopped.
5. First terminal: `uv run python -m simple_serving.cli up` again. It resumes the instance, and the card boots anew and
   runs one pair, which `up` reports ready.
6. Second terminal:

   ```
   set -o pipefail; uv run python -m simple_serving.smoke --after logs/lifecycle.json | tee logs/smoke.jsonl
   ```

   First one pair, a new boot and the same deadline, then the whole smoke with the real template. With `pipefail` the
   pipeline ends with the smoke's own exit code, which `tee` would otherwise hide.

Each line of the smoke is one probe: `probe`, `ok`, the gateway's HTTP `status` and the contract's `code` where one
came, `failed` with the names of what broke, and numbers. `fields`, `reasoning`, `finish`, `schemas` and `counts` hold
an object per part, each with its own `ok`, and pass when every part passes. A name in `failed` is a rule of section 4's
stream, such as `model` for a chunk without the alias or `done` for a stream without `[DONE]`, or a check of the probe:
`equal` for a count that differs from `usage.prompt_tokens`, `answer` for an answer that breaks its schema, whose field
`answer` lists the keywords it breaks, or `json` for one that is not JSON; `pair`, `boot` or `deadline` in the
lifecycle; `ready`, `status`, `first_event`, `held`, `freed`, `unlogged`, `inconclusive` or `drained` in the abort; and
`code`, `unlogged` or `marker` in the privacy probe. `left_drained` means that the open undoing the abort's drain was
not read back as `ready`, so the gateway may be left drained; `no_answer` that the gateway did not answer,
`no_report` that the card's report did not come or came in another form, `time_bound` that the probe ran past its
bound, and `smoke_error` an error in the smoke itself, whose class `error` names. With `no_report`, `report` says
how: `timeout`, `pipe`, `exit` with the command's code in `exit` (255 is ssh's own), `size`, `json`, `shape` or
`no_host`, and `call` names the request: `plain`, `marker` or `since`. The numbers are the stream's chunks,
the lengths of the answer and of the reasoning in characters, the usage, and the gateway's `wait_ms`,
`first_token_ms` and `total_ms`; in the count matrix `input_tokens` beside `prompt_tokens`, and near the context the
`max_tokens` that fills it and the counts the search took; in the lifecycle the `launchers`, their `children` and the
seconds left to the guard's deadline; in the abort `held`, `freed_ms` and `cancelled`; in the privacy probe the
marker's count in each log under `found`, the request's rows under `logged`, and `echoed`, the times the answer
repeated the marker, which passes nothing. `versions` shows each pin as this checkout has it, or `false` where the
card shows another, and the card's Python. `seed_repeats` passes nothing. No line holds a key, a prompt, the marker,
an answer or reasoning, so the lines may be kept.

Exit 0, with `simple-serving smoke: 11 of 11 probes passed` on standard error, ends section 15's step 1, and its last
probe, `privacy`, is the marker half of its step 2. After it the trial's internal work is the texts of
simple-story-chat's action measurement alone, and then the operator deletes it with
`npm run gpu:rent -- --destroy ID` and its read-back: no eval and no other long run. A probe that fails, in either run
of the smoke, ends the smoke and the attempt there, and the operator deletes the trial as rule 4 says. Nothing is
fixed, retried or tuned on the paid card. The lines say what failed, and another configuration is measured on a later
rental (contract section 15). Exit 2 sent nothing to the card; its line on standard error says why.

Open, and not built: unattended or permanent use needs an independent budget path, one that bounds the costs without
the container's key and without an operator watching.

Open as well: the bot's story audit asks for 16384 tokens, and the `internal` class refuses more than 8192 (contract
section 7). Nothing changes for the first rental, whose `schemas` probe asks each schema for 2048 at most.

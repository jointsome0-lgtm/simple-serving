# simple-serving

Text model serving for [simple-story-chat](https://github.com/jointsome0-lgtm/simple-story-chat): a FastAPI gateway in
front of vLLM, run on rented GPUs. The bot calls it over HTTP. Outside clients with a key may call it too.

Status: the gateway of contract v2 is written and tested against a fake engine. It has not run in front of vLLM yet;
that happens on the first rental (contract section 15). The API is in [docs/contract-v2.md](docs/contract-v2.md), the
shared cases in [contract/](contract/README.md).

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
  with stand-ins for vLLM, the gateway, pip, curl and flock;
- `tests/test_cli.py`: the command on the owner's machine, against a fake Vast and a fake SSH that forwards to the
  gateway;
- `tests/test_bot_schemas.py`: the copy of the bot's JSON schemas, and the check of an answer against one;
- `tests/test_units.py`, `tests/test_dev.py`: the pieces one by one, and the dev launcher with its fake engine's
  default answers.

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
milliseconds, so requests rarely overlap. `--first-event-delay-ms N` makes it pause N ms before the first event of each generation, as
for a long prompt, and `--event-delay-ms N` after each event; with them, concurrent requests fill the places and wait.
Both default to 0. Like the card's, the gateway falls asleep after 780 seconds without a request of ours (contract
section 8); it has no instance to stop, so it then stays drained until the launcher starts again.

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
own, and its request logging off (contract section 10).

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
there. It fetches the weights and the tokenizer files at their pinned revisions and checks their hashes, keeps the
key hashes in `keys.json`, and starts the service. It refuses while a pin or a lock is missing, or when vLLM's lock
is for another version than the pin, and it never replaces the keys the card holds.

The locks, `card/gateway-requirements.txt` and `card/vllm-requirements.txt`, each name in their header the command
that made them: the gateway's is exported from `uv.lock`, and vLLM's is resolved for the pinned version. A change of
`uv.lock` or of the pin needs a new lock.

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
the first start. The guard deletes with the container's key, as the card stops, so a key that Vast refuses defeats
both. An onstart for a permanent rental, which writes the two files, arms no guard and ends with the same line, is
decided before permanent use.

## The command

On the owner's machine, `python -m simple_serving.cli` starts the card, holds the tunnel and asks for sleep:

```
uv run python -m simple_serving.cli up | sleep | status | keys
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

The configuration is `~/.config/simple-serving/config.json`, open to its owner only, or the file that `--config`
names:

```json
{"instance_id": "...", "vast_api_key": "...", "ssh_host": "...", "client_key": "...", "control_key": "..."}
```

`keys` writes the last two. `vast_api_key` is a Vast key allowed GET and PUT on that instance alone, never the account
key, and `ssh_host` a host of `~/.ssh/config` whose host key is known. The bot's model profile takes the address
`http://127.0.0.1:8080` and the client key; it holds neither the control key nor a Vast key.

## The first rental

The card stops itself with the container's key, and the trial's guard deletes with the same key, so a key that Vast
refuses leaves nothing on the card that bounds the costs. Before the launcher runs, the card has not even its own
stop: SSH may not answer, and bootstrap.sh may hang in pip or curl, or end with 1. On the first rental the owner is
that bound, watching from the moment the instance is created:

1. Before the instance is created, the owner chooses the time of day by which the first attempt, the preparation
   included, is over. This README sets none.
2. Before anything long, once the prepared card is ready: `sleep`, which must end with `vast: stopped`. Then `up`,
   a resume with no start over SSH: the card boots anew and runs one pair, which `up` reports ready.
3. Every stop is confirmed from outside, with the owner's own key: `status` prints `vast: stopped`, and Vast's console
   shows the instance stopped. `sleep` waits 10 minutes for that. After the last request of an idle card, `stopped`
   must come within 25 minutes: 13 to fall asleep, 2 for the drain, 10 for the stop.
4. Approved in advance, and done at once: when the preparation fails, when SSH or a command stops answering, when
   the attempt's deadline passes, when `stopped` does not come within those 25 minutes, when a command says that the
   card's stop is not confirmed, when `sleep` ends without `vast: stopped`, or when Vast refuses a key with 401 or
   403, the owner stops the instance in Vast's console, or deletes a trial instance, and checks there that it is
   stopped or gone.

Open, and not built: unattended or permanent use needs an independent budget path, one that bounds the costs without
the container's key and without the owner at the console.

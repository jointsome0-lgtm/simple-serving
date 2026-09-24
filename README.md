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
generation with a fixed synthetic sentence in a few chunks, then `stop` and its usage. It answers in milliseconds, so
requests rarely overlap. `--first-event-delay-ms N` makes it pause N ms before the first event of each generation, as
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
`--enable-prompt-tokens-details` for cached tokens, and its request logging off (contract section 10).

## The card

`card/` holds what runs on a rented card: `manifest.env` with the pins and parameters, `bootstrap.sh` for the
preparation, and `onstart.sh`, which runs at every start of the container. The checkout goes to
`/workspace/simple-serving`, and the card keeps its state in `/workspace/simple-serving-card`: the key hashes, the
weights, the two venvs and the logs.

The preparation, once per rental and never at a resume:

1. On the owner's machine, export the gateway's lock with its hashes:
   `uv export --frozen --no-dev --no-emit-project -o card/gateway-requirements.txt`. vLLM's lock,
   `card/vllm-requirements.txt`, is made with the pin.
2. Copy the checkout to `/workspace/simple-serving`.
3. `ssh <card> bash /workspace/simple-serving/card/bootstrap.sh < key-hashes`, where `key-hashes` holds the SHA-256 of
   the client key and of the control key, one per line.

bootstrap.sh installs each lock into a venv of its own with every hash checked, fetches the weights and the tokenizer
files at their pinned revisions and checks their hashes, keeps the key hashes in `keys.json`, adds `onstart.sh` to the
container's `/root/onstart.sh`, and starts the service. It refuses while a pin is empty, as `VLLM_VERSION` is until
the pin is chosen, and it never replaces the keys the card holds.

At every start of the container `onstart.sh` arms the trial guard, the one of simple-story-chat's
`gpu/trial-onstart.sh` with the same files, and starts the launcher, `python -m simple_serving.card`. The launcher
runs vLLM and the gateway once, with no restarts, and when the pair fails it stops the instance; its options and exit
codes are in `simple_serving/card.py`. The tunnel's remote command waits while the pair runs:

```
cd /workspace/simple-serving && /workspace/simple-serving-card/gateway/bin/python -m simple_serving.card --hold
```

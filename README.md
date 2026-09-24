# simple-serving

Text model serving for [simple-story-chat](https://github.com/jointsome0-lgtm/simple-story-chat): a FastAPI gateway in
front of vLLM, run on rented GPUs. The bot calls it over HTTP. Outside clients with a key may call it too.

Status: the gateway of contract v1 is written and tested against a fake engine. It has not run in front of vLLM yet;
that happens on the first rental (contract section 15). The API is in [docs/contract-v1.md](docs/contract-v1.md), the
shared cases in [contract/](contract/README.md).

Version 1 serves one text model. Pictures stay in simple-story-chat for now.

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
```

No network, GPU or vLLM. The tests start the gateway in uvicorn on loopback ports, in front of a fake vLLM
(`simple_serving/fake_engine.py`), and call it over HTTP:

- `tests/test_cases.py`: every case of `contract/cases-v1.json`, each on a new gateway;
- `tests/test_scenarios.py`: the scenarios of the cases file: a client that leaves in every phase, places and lines,
  counts, wall time, drains;
- `tests/test_privacy.py`: the gateway as its own process, started as in production; every error path carries a marker
  that no answer and no log line may hold;
- `tests/test_gateway.py`: the engine's health and context, what the engine receives, errors inside the gateway;
- `tests/test_units.py`, `tests/test_dev.py`: the pieces one by one, and the dev launcher.

## Dev launcher

The fake engine and the gateway on loopback, to run a real client against the real gateway:

```
uv run python -m simple_serving.dev --config contract/cases-v1.json --engine-port 8200 --public-port 8201 --control-port 8202
```

It prints, once the gateway is ready:

```
simple-serving dev launcher: ready
  fake engine  http://127.0.0.1:8200
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
generation with a fixed synthetic sentence in a few chunks, then `stop` and its usage.

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
(2 000 000), `max_connections` (64), `health_interval_s` (5) and `versions`, which the control key sees in `/v1/state`
next to the gateway's own. An outside key has the classes `["external"]` and no scopes. To hash a key without printing
it:

```
read -rs KEY && printf '%s' "$KEY" | sha256sum && unset KEY
```

The public listener sits behind the TLS proxy; the control listener stays on loopback, reached over the SSH tunnel.
vLLM listens on loopback with the served model name equal to `alias`, priority scheduling
(`--scheduling-policy priority`), prefix caching, `--enable-prompt-tokens-details` for cached tokens, and its request
logging off (contract section 10).

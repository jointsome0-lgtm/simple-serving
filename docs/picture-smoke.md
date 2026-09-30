# Picture smoke

This client checks an already running picture gateway and its ComfyUI. It starts no server, changes no card
configuration and manages no rental. One invocation sends one synthetic Qwen graph, receives one PNG and checks
that its own job no longer appears in the engine's queue or history. It does not retry generation.

## Local fake run

Prepare private temporary files with synthetic keys and the checkout's pinned versions:

```sh
SMOKE_DIR=$(mktemp -d /tmp/simple-serving-picture-smoke.XXXXXX)
export SMOKE_DIR
uv run python - <<'PY'
import json
import os
import secrets
from pathlib import Path
from simple_serving import card
from simple_serving.pictures.smoke import expected_versions

directory = Path(os.environ["SMOKE_DIR"])
client, control = secrets.token_hex(32), secrets.token_hex(32)
files = {
    "client.json": {"client_key": client, "control_key": control},
    "dev.json": {
        "alias": card.read_manifest(folder="card/pictures")["MODEL_ALIAS"],
        "keys": {
            client: {"label": "smoke", "classes": ["internal"], "default": "internal", "scopes": False,
                     "control": False},
            control: {"label": "control", "classes": [], "default": None, "scopes": False, "control": True},
        },
        "versions": expected_versions(),
    },
}
for name, body in files.items():
    descriptor = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(body, file)
print(directory)
PY
```

In one terminal, start the existing dev launcher:

```sh
uv run python -m simple_serving.pictures.dev --config "$SMOKE_DIR/dev.json" \
  --engine-port 8210 --public-port 8214 --control-port 8215
```

In another terminal, set `SMOKE_DIR` to the same printed directory, then run the smoke:

```sh
uv run python -m simple_serving.pictures.smoke --fake --config "$SMOKE_DIR/client.json" \
  --public-port 8214 --control-port 8215 --engine-port 8210
```

The fake returns synthetic pixels. This checks client execution, gateway framing and cleanup; it does not establish
model inference, GPU capacity or picture quality. The report marks GPU behavior `not_verifiable`. Ctrl-C the dev
launcher after the check. The generated keys are local test keys, not an owner configuration.

## A prepared card

Use the client only within an authorised experiment, after preparation and `cli --lane pictures up` have made the
card ready. The usual tunnel supplies the picture gateway's public and control listeners at local 8084 and 8085.
Read-only observation of ComfyUI requires an additional SSH forward, bound to loopback:

```sh
ssh -N -o BatchMode=yes -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8188:127.0.0.1:8188 <card-host>
```

Leave the two forwards running. With the existing owner-only command configuration:

```sh
uv run python -m simple_serving.pictures.smoke --real \
  --config ~/.config/simple-serving/config.json \
  --public-port 8084 --control-port 8085 --engine-port 8188 \
  --wall-s 300 --cleanup-s 20
```

All three ports are explicit and all requests use `127.0.0.1`; redirects and environment proxies are disabled.
The key file supplies `client_key` and `control_key`. No key is accepted on the command line or printed. The smoke
reads its checked-in manifests and workflow, so run it from the checkout matching the prepared gateway.

`--wall-s` bounds preflight and generation. `--cleanup-s` reserves additional time for cleanup after success or
failure. Choose their sum within the external experiment deadline. The client does not stop or delete a rental,
and completing or cancelling it does not establish that billing stopped. Close the extra SSH forward after use.

## What passes

The preflight checks the picture contract, lane, model and ready state, and the same boot ID on public and control
listeners; reported revisions against the checkout's
manifest and `SOURCE`; and engine facts including ComfyUI, PyTorch, Triton and kitchen-attention availability.
Reported revisions are configuration evidence, not a separate verification of installed weight bytes.

The graph comes from `card/pictures/image-workflow-qwen.json`: the pinned model, encoder and VAE filenames, a fixed
synthetic prompt and seed, a 512-by-512 latent, four sampler steps, `PreviewImage` instead of `SaveImage`, and
`ModelAttentionBackend` selecting `comfy kitchen attention`. It uploads no reference and supplies no reader data.

The required event course is `accepted`, `started`, `over`, one `image`, then `[DONE]`. Queue, cache, node and progress
events can occur between these stages. The client validates bounded stream/JSON/PNG data and emits no image file.
PNG validation checks the container, header and chunk checksums; it does not decode or assess the pixels.

The observer must first find the accepted job's exact ID in the supplied ComfyUI's queue or history. After success,
the client polls until that job is absent from both. Empty history on an unrelated server cannot pass. A job too
brief to observe cannot establish this check. No engine history-delete request is sent by the smoke itself.

On failure after acceptance, the client asks the gateway to stop only that job and checks the cleanup result within
its reserve. Unconfirmed cleanup makes the smoke fail. It never drains the gateway or changes another job.

## Scope of the result

The flat JSON report holds fixed codes, booleans and numbers. It excludes keys, job IDs, model paths, graphs, prompts,
PNG bytes and arbitrary engine error text. Received image bytes stay in memory. The report's real-mode GPU value is
`not_measured`; even a successful real run does not measure image quality or hardware performance.

Exit status is 0 on success, 130 on cancellation, 2 for invalid options or configuration, and 1 for other failures.

This is one generation and record-cleanup check. Reader isolation, waiting/drawing cancellation, reference erasure,
preview sweeping, idle/resume, rollback, bot parity and provider activation remain in the
[full card test](pictures.md#the-card-test). Local checks do not complete that test or enable the bot's provider.

Run the local regression tests with:

```sh
uv run python -m pytest -q tests/test_pictures_smoke.py
```

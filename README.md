# simple-serving

Model serving for [simple-story-chat](https://github.com/jointsome0-lgtm/simple-story-chat): FastAPI gateways in
front of vLLM for text and ComfyUI for pictures, on separate rented cards. The bot calls them over HTTP. Each lane
handles its requests, streaming, idle sleep and its card's stop.

The gateway implements contract v2 and has tests against a fake engine. The September 2026 decision record also
contains real vLLM rentals and RTX 5090 measurements, including the choice to keep route A with multi-token
prediction. Those dated results do not verify a current live deployment or every mode of the contract.

Under MTP, the text gateway rejects `min_p > 1e-5` with 400 `min_p_unavailable` unless `temperature` is 0,
before sending anything to vLLM. Counting is unaffected. See the [generation contract](docs/contract-v2.md#4-generation-post-v1chatcompletions).

The text API remains contract v2. The picture lane's phase 1 code is integrated locally, with a fake ComfyUI,
picture card files and `--lane pictures` in the CLI. Its real card test remains pending. The bot's provider switch
stays off by default, so it still draws through its existing ComfyUI path until the card test and activation.
Outside-client and unattended operation still have open validation and budget work.

## Documentation

- [Runbook](docs/runbook.md): development, gateway configuration, card preparation, CLI, smoke and trial procedure.
- [Monitoring](docs/metrics.md): both gateways, the text engine, dashboards and local setup.
- [Text contract v2](docs/contract-v2.md): text API, limits, lifecycle and privacy guarantees.
- [Picture contract and migration](docs/pictures.md): picture API, card procedure, phases and rollback.
- [Shared cases](contract/README.md): the case format used by the gateway and client adapters.
- [Decisions and measurements](docs/decisions.md): dated choices, results and superseded plans.
- [Existing branches](docs/branches.md): where the current groundwork lives, its dependencies and integration limits.
- [Speed experiments](docs/speed.md): questions for later measurements.
- [Prefill interference](docs/prefill-interference.md): a synthetic demonstration and client measurements through
  an existing text gateway, before deciding on prefill/decode disaggregation.

## Install

Use Python 3.12 or later and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync --locked
```

The first sync needs the network. `uv.lock` pins the runtime and development tools. The `dev` group includes
`pictures`, which pins `websockets`; a picture-only runtime can use `uv sync --locked --group pictures --no-dev`.
After installation, the local checks use loopback sockets and synthetic inputs.

## Checks

Run all local checks with one command:

```sh
./scripts/check.sh
```

The script first runs `uv sync --locked --dev`, then ruff, mypy and pytest offline without syncing again.
The sync uses the existing lock file and may need the network to install missing dependencies.
The tests need no GPU, vLLM, ComfyUI, SSH or paid instance. They cover the text cases, both gateways, cancellation,
drain races, privacy, idle sleep, card scripts, the CLI and the text smoke dry run. Card and Vast interactions use
stand-ins. The [runbook's test section](docs/runbook.md#tests) describes the individual suites
and commands.

The [GitHub Actions workflow](.github/workflows/check.yml) runs the same script on Python 3.12 and 3.14 for pushes
and pull requests. It pins the checkout and uv actions, and the uv version.

A client adapter checks the public cases in its own repository. Control routes and the card lifecycle are tested
here. The real adapter's opt-in integration test belongs in simple-story-chat, as
[contract section 14](docs/contract-v2.md#14-shared-cases) specifies.

## Try the text gateway locally

Start the real gateway in front of the fake engine:

```sh
uv run python -m simple_serving.dev --config contract/cases-v2.json --engine-port 8200 --public-port 8201 --control-port 8202
```

Wait for `simple-serving dev launcher: ready`. The public address is `http://127.0.0.1:8201`, and control uses
`http://127.0.0.1:8202`. The cases supply the synthetic keys `test-key-bot` and `test-key-control`, with model
`test-model` and a context of 4096 tokens.

The fake engine returns fixed synthetic text. It can delay the first event or later events to exercise queues and
cancellation. Ctrl-C stops both processes. See the [dev launcher instructions](docs/runbook.md#dev-launcher) for the
configuration shape, delays and idle behavior.

## Try the picture gateway locally

Save the [synthetic keys example](docs/runbook.md#picture-dev-launcher) as `/tmp/simple-serving-pictures-dev.json`,
then start the picture gateway in front of the package's fake ComfyUI:

```sh
uv run python -m simple_serving.pictures.dev --config /tmp/simple-serving-pictures-dev.json --engine-port 8210 --public-port 8214 --control-port 8215
```

It prints `simple-serving pictures dev launcher: ready`. The public address is `http://127.0.0.1:8214`, and control
uses `http://127.0.0.1:8215`. Ctrl-C stops both. This runs synthetic pictures locally; it does not exercise a real
ComfyUI card or activate the bot's provider switch.

## The card

The [runbook](docs/runbook.md#running-the-gateway) holds the configuration examples and exact operating commands.
`card/manifest.env` pins the model, tokenizer, vLLM and card parameters. The two requirements files lock the card's
separate environments. A change to `uv.lock` or the vLLM pin needs the corresponding card lock updated.
`card/pictures/` holds the separate picture manifest, copied ComfyUI scripts and gateway lock; its
[preparation procedure](docs/pictures.md#9-the-card) preserves the pinned engine and existing install.

Gateway configuration stays outside the repository and holds SHA-256 hashes of gateway keys. The owner's CLI
configuration holds the client and control keys plus a restricted Vast key. The bot receives only the service
address and client key.

On the card, both gateway listeners and the engine stay on loopback and are reached over SSH. The rental's own
onstart ends with this line, whose full preparation and resume procedure is in the
[card instructions](docs/runbook.md#the-card):

```sh
if [[ -f /workspace/simple-serving/card/onstart.sh ]]; then bash /workspace/simple-serving/card/onstart.sh; fi
```

For the picture card, use `uv run python -m simple_serving.cli --lane pictures up`, `sleep`, `status` or
`trial --ssh-host HOST`. It has its own instance, tunnel ports and lock. See the
[picture CLI instructions](docs/runbook.md#the-command) and [card test and rollback](docs/pictures.md#the-card-test).

## Monitoring

Both control listeners expose authenticated `/metrics`. The text listener also relays vLLM at `/metrics/engine`;
the picture listener has no engine relay. Prometheus and Grafana run on the owner's machine with pinned images,
through the existing tunnels. Scrapes neither generate work nor keep a card awake.

The [setup and metric catalog](docs/metrics.md#local-setup) cover keys, the optional picture target and persistent
history. Local tests use fake engines; dashboard deployment and real-card monitoring need their own verification.

## The first rental

The first-rental procedure includes an operator deadline, stop readback and confirmed deletion of the disposable
trial. A stopped trial still bills its disk. Read the [full procedure](docs/runbook.md#the-first-rental) before
creating an instance; the service's stop and the trial guard depend on the same container key.

## Open work and verification limits

- The picture lane still needs its real card test. `SIMPLE_CHAT_IMAGE_PROVIDER=comfyui` remains the default; the
  migration plan keeps the old path and rollback until the later phases pass.
- Outside keys still need the TLS proxy and load scenarios described in
  [contract section 15](docs/contract-v2.md#15-the-first-rental). The draft limits need measurements under that load.
- Unattended or permanent use needs an independent budget path that bounds costs without the container's key or an
  operator watching. It is not built.
- The bot's story audit asks for 16384 output tokens, while the `internal` class permits at most 8192. The limits are
  unchanged; the first-rental `schemas` probe asks for at most 2048 per schema.
- An engine error in the middle of a stream has a fake-engine contract case. The smoke cannot force it on the card,
  so the card's side remains unverified.

The detailed 5090 measurements are in the [runbook](docs/runbook.md#the-card). The
[decision record](docs/decisions.md#text-serving) preserves the owner's quality and throughput tradeoff, including
choices that later entries replaced.

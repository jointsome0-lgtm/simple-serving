# Synthetic vision smoke

Updated on 2026-09-30 for the integrated checkout. `VISION=0` remains the manifest default. The usual smoke runs
its eleven probes, including the thinking-budget probe; vision runs only when `--only vision` names it.

This probe checks the path needed for a later pose-captioning experiment. It generates its own coloured PNG blocks
in memory and sends no owner or tester photographs. Recognising these colours does not establish pose-label quality.
A fake-engine pass establishes the local request, stream, refusal and log-check plumbing. A separately authorised
GPU run is still needed to measure model answers, accounting, memory capacity and performance.

## Prerequisites and boundaries

Before sending any picture, the probe checks the control view for `vision: "1"` and compares its vLLM, model and
tokenizer revisions with this checkout's manifest. The integrated pins are vLLM 0.30.0, model revision
`24ac337ecd25c89d9211d40139696dae84591f1f`, and tokenizer revision
`76ac65c75ce10fb7b570e7f98acd385527e7095c`. A mismatch fails with `versions` before an image request is sent.
The probe then obtains the initial gateway-log position so that its final privacy check can identify new requests.

The gateway also requires the client's explicit `vision` permission. Configuration rejects that permission on an
`outside` key: vision remains limited to internal owner/tester clients. The separate picture-generation lane rejects
text vision permission altogether. Preparing or testing the synthetic fixture does not enable a real client's key.

The image limit is 2 MiB (2 097 152 bytes) after base64 decoding, measured as compressed image bytes. Base64 takes
about 2.67 MiB at that limit; the default text HTTP body limit of 4 MiB (4 194 304 bytes) leaves room for text and
JSON. An explicit `body_limit_bytes` still governs the entire request. These limits do not bound decoded pixels.

With vision enabled, the card limits requests to one image and no video or audio. `--mm-processor-cache-gb 0`
disables the processor cache. It does not disable every engine cache. In particular, the pinned engine's image
encoder cache is keyed by image content without the gateway's text cache salt. This work therefore makes no claim
of image-cache isolation between tenants. The internal-client restriction remains necessary.

## Checks and evidence

The complete probe makes three explicit count requests, four accepted generation requests and two requests expected
to be refused. The gateway's own counting of generation requests is additional. It stops measurements at the first
failure, then scans logs if it obtained the initial log position. All images are synthetic: two 64-pixel squares and
one four-colour image made of 64-pixel blocks. A per-run marker is embedded in both the PNG bytes and its data URL.

| Check | Evidence required |
| --- | --- |
| Red and blue captions | Valid completed streams with the whole expected colour word, ignoring case, surrounding whitespace and one final period or exclamation mark; finish reason `stop` |
| Counts against usage | Each counted input equals its generation's `usage.prompt_tokens` on the real engine |
| Image-token bound | Red and blue each add more than zero and at most 282 input tokens compared with the same text without an image; both deltas agree |
| Description | A valid completed stream with nonempty content, limited to 192 output tokens |
| Refusals | Two images and a remote image URL each produce 400 `invalid_request` |
| Privacy | At least one new generation log row, and zero occurrences of the marker across every file under the card's `logs/` |

The pinned template is expected to add 258 tokens for these squares; the fake uses 282. The real count remains a
GPU verification item. The captions and text comparison allow at most eight output tokens each. The description
allows observing decode timing and drafter counters; its content is not a pose-quality evaluation.

The probe reserves 30 seconds of its 300-second bound for the privacy scan and one second to finish reporting. It
performs that scan after measurement failures, including transport errors and timeouts. A clean scan does not turn a failed measurement into
a pass. A missing log row, leaked marker, failed scan or scan timeout fails the probe. Error output uses fixed
failure names and excludes exception text that could contain image data.

`first_token_ms` and decode rates are observations of the executed requests. The count endpoint can preprocess an
image before generation, and engine caches may already hold work. The probe does not establish cold-cache timing or
performance isolation. GPU readiness, KV-cache capacity, load time and memory use must be recorded from a real card's
status and launcher metrics. The fake marks `caption`, `equal`, `picture_tokens`, `performance` and `gpu_capacity`
as `not_verifiable` even when its smoke exits successfully.

## Local verification

The local regression suite launches synthetic fake services on loopback and requires no GPU, rental, real image or
owner key:

```sh
uv run python -m pytest -q tests/test_smoke.py
```

It covers the eleven default probes, the separately selected vision probe, both raw-body and image-byte leaks,
malformed or failed streams, wrong caption words, count mismatches, unequal image-token deltas, transport failures,
measurement timeout, privacy-scan failure and timeout, and pin mismatch. The enabled test launcher uses synthetic
configuration with `vision: "1"` and a synthetic client's `vision: true`; it leaves the checked-in manifest at zero.
It also checks that the fake engine still serves the metrics needed by the ordinary probes.

To select vision against an already running synthetic fixture, use its client configuration, card-state directory
and forwarded ports:

```sh
uv run python -m simple_serving.smoke --only vision --fake \
  --config /path/to/synthetic-config.json --card-dir /path/to/synthetic-state \
  --public-port 8301 --control-port 8302
```

The fixture must advertise the manifest's exact pins and vision version. `--fake` skips the model-dependent checks;
it does not bypass the preflight pins or privacy scan.

## Later GPU verification

A paid run requires separate authorisation and preparation of the chosen checkout with `VISION=1`, following
[The card](../README.md#the-card). After the card reports ready, use its authorised client configuration and
forwards with `--only vision`, omitting `--fake`. Keep the card's lifecycle deadline and provider-side teardown from
the rental runbook. No card was rented, restarted or reconfigured for this integration.

Record the smoke's numerical verdict, launcher capacity and load metrics, and drafter counters. Compare them with
text-only measurements from the same pinned setup and card. A failing caption, accounting check or privacy scan
ends this experiment. Real pose references need their own subsequent evaluation and owner authorisation.

The original branch's 2026-09-28 planning estimate was 6 to 10 minutes, or $0.06 to $0.09 at its assumed $0.55/hour;
its step timeouts totalled 26 minutes. Those are historical estimates, not current rental prices or a billing bound.
The earlier text-only observations were 104492 KV tokens on a first start and 151300 on a warm start, with warm
startup taking 29 to 35 seconds. They are comparison context, not a prediction for this checkout. The vision tower's
estimated 1.07 GiB of weights excludes encoder working memory. Its actual capacity impact remains unmeasured here.
The old queue-specific shell helpers and host names belong to that branch's historical runbook and are not required
by the integrated smoke.

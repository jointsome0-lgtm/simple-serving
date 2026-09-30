# Existing branches, 2026-09-30

The inventory began with 19 local branches, compared with `codex/comfyui-integration` at `c23c282`.
After removing seven redundant names and the integrated `next-card` branch, 11 remain. This document records the remaining work, integration
constraints and the retired names. The review inspected Git ancestry, patch identity, source, tests and documents.
The initial inventory ran no tests, engines or rentals. The later MTP integration passed the shared checks on
Python 3.12 and 3.14, recorded in [decisions](decisions.md#mtp-refusal-integration-2026-09-30).
Historical checks below are evidence from the branches, not a fresh
verification of their combined behavior.

## The current code

`c23c282` includes the text gateway, MTP, optional RAM KV offloading, the picture lane from `7fdd58f`, reference
cleanup fixes from `abcbabf`, and the shared check command. The bot's picture provider switch remains off by
default. The [picture card test](pictures.md#the-card-test) is still pending.

`main` at `7a60fc7` remains as the earlier text baseline, with its `origin/main` upstream. The first-rental smoke,
MTP, phase 2 and picture implementations are already in the current history; their old local names were removed.

The remaining branches are below. A candidate for integration still needs its changes applied to the current
interfaces and checked. A branch's tests passing before the picture lane existed would not establish that.

The `min_p` fix from `8672e62` is now integrated in the working code, adapted to the shared lifecycle.
This is a code-integration status, not a fresh GPU result. Monitoring remains an adaptation candidate.

## Branch work and integration status

| Branch | Tip | Contribution and verdict |
|--------|-----|--------------------------|
| `live-2026-09-28` | `940471a` | Includes `8672e62`, then adds dense prefix retention to the text launcher. Review the default change separately. |
| `next-card-metrics` | `9413d16` | Combines monitoring with `8672e62`, pins monitoring images and supports history outside the checkout via `MONITORING_DATA`. Needs adaptation to the picture lane. |
| `thinking-budget` | `466a82d` | Includes `8672e62`. Implements an optional thinking-token budget. Needs integration checks, then a GPU smoke before activation. |
| `vision-captions` | `47d71ae` | Includes `8672e62`. Implements image input for Gemma, key permissions, launcher/bootstrap and a vision smoke. Needs shared configuration and launcher adaptation, then GPU validation. |
| `held-criterion` | `a898060` | Includes throughput and fixes its cache-capacity stopping criterion. Best candidate for integrating that tooling. The revised criterion needs a GPU measurement. |
| `sweep` | `b6d4919` | Separate sweep against an already running engine, with a deadline and overlap checks. Independent of the A/B harness; keep it identifiable as an experiment. |
| `split-kv-wip` | `bd0503e` | Split-KV plugin files on an older throughput base. The launcher does not register the change. Unfinished. |
| `pictures-two-cards` | `5aa9302` | Design and recorded owner answers only. No distributor or two-card test is implemented. |
| `rehearsal-e2b` | `754a852` | Historical rehearsal. Only the E2B model pin is patch-unique; the other fixes have equivalent patches in the current code. Do not replay the branch wholesale. |

## Dependencies and integration constraints

### The shared text fix

`8672e62` is shared by the live, metrics, thinking-budget and vision work. Its fix is now integrated; preserve it
when adapting those branches rather than replaying it. After checking the
`max_tokens` limit, it refuses generation before count or admission when MTP is enabled, `min_p > 1e-5` and
temperature is not zero. Counting is unaffected.
Its source is `simple_serving/service.py` and `routes.py`, with the input matrix in `tests/test_gateway.py`.
The check and control-only `unavailable_fields` stay in the text `Service`, which inherits the shared `Lifecycle`.

### Monitoring

Use `9413d16` rather than replaying `monitoring` separately. Its `Guard` constructor requires a `Metrics` argument,
but the current [picture apps](../simple_serving/pictures/app.py) use the earlier constructor. The picture control
app also imports the shared control-route table. Copying the metrics routes into that table would expose text-only
handlers to `PictureService`, which has no text engine, admission or count-place fields.

Keep text metrics routes and observers scoped to text, or implement and test a picture metrics contract explicitly.
Preserve the current templated picture paths and PUT/DELETE logging. Reconcile `Card.keys()`, picture bootstrap,
CLI `--lane`, shared work/log fields and the global error-code test. The metrics branch's error-outcome matrix does
not yet include picture errors. The monitoring catalog test can skip without a vLLM checkout.

### Thinking budget and vision

Thinking budget uses `thinking_token_budget` and `--reasoning-config`; the branch contains no vLLM patch.
It is off by default. Noninteger or negative values give `invalid_request`; a valid number while disabled gives
`thinking_budget_unavailable`. When enabled, the upper bound is the class's `max_tokens`, not the request's.
Counting checks type and sign, but not feature availability or the class bound. The historical fake reply used
only 11 thinking tokens, so it did not demonstrate enforcement of the proposed 128/300 budgets.

Vision adds a manifest flag, `VISION=0` by default, and a per-key permission. It accepts one PNG/JPEG/WebP data URL
up to 1 MiB.
`PictureCard.config()` calls the common `Card.keys()`. Do not make a text-only `VISION` flag mandatory for picture
manifests. The common key parser also needs a deliberate picture-lane policy for the new permission.
Budget and vision are functionally independent after the shared text fix. Both need regressions of the text and
picture configuration/lifecycle. Vision's synthetic color/count/privacy smoke does not prove GPU tower capacity,
real captions, token counts or MTP throughput.

### Measurement tools

Use `held-criterion` once, without separately replaying `throughput`. Its corrected timing sweep distinguishes
waiting from preemption and explains the stop in its report. Preserve the current card subclassing and log-field
allowlist when carrying over the launcher, metrics and fake-engine changes.

The A/B launcher stops the serving pair, starts its measurement pair and leaves the card without serving when it
finishes. Decide how serving is restored before using it alongside readers. `sweep` instead measures a running
engine and does not restart it. Its assumed cache capacity and deadline are part of that experiment.

The 2026-09-27 5060 Ti and 5090 results live in `held-criterion:docs/speed.md`. Further concurrency measurements,
RAM offloading with dense retention, and eval remain unfinished. `live-2026-09-28` cites a speed-document item
missing from its own tree; use the fuller measurement branch to assess that setting. The split-KV plugin needs
launcher wiring, log evidence, review and GPU correctness, identity and performance checks.

### Two picture cards

`6559c6e` is the design and `5aa9302` records the owner's answers. Phase A proposes a local distributor and CLI
against two fake ComfyUI engines. Upload fan-out must preserve the current generations and lost-response cleanup
from `abcbabf`. Phase B depends on the single-card test and the design's five review findings.
The answers specify two machines with one GPU each, a second card for batches of at least 40 minutes, a one-hour
default rental, and rental management remaining in simple-story-chat. These are design decisions, not implemented
two-card behavior.

### Historical rehearsal

The equivalent rehearsal patches are in the current history, but later decisions still apply. In particular,
the rehearsal's JSON whitespace ban was subsequently removed. Its E2B model pin is for that rehearsal alone.

## Suggested order

1. The `8672e62` fix is integrated and passes the shared checks for both lanes.
2. Prepare the existing ComfyUI single-card test. Its real result gates picture-provider activation.
3. Adapt `9413d16` for the current interfaces, then check metrics permissions, privacy, cancellation and idle sleep.
4. Carry over `held-criterion` with its dry run and an explicit serving-restoration procedure. Assess the retention
   setting separately from carrying over the tooling.
5. Integrate budget and vision as separate optional changes, with real GPU checks before enabling them.
6. Keep the two-card design and split-KV work separate until their first implemented checks exist.

## Finding the work

The main checkout is `~/work/simple-serving`. The remaining clean worktrees are:

| Location | Branch |
|----------|--------|
| `~/work/simple-serving-two-cards` | `pictures-two-cards` |
| `~/work/simple-serving-vision` | `vision-captions` |
| `/var/tmp/simple-serving-live` | `live-2026-09-28` |
| `/var/tmp/simple-serving-metrics` | `next-card-metrics` |
| `/var/tmp/simple-serving-budget` | `thinking-budget` |
| `/var/tmp/simple-serving-held` | `held-criterion` |

## Removed duplicates

The following local names were removed on 2026-09-30. Each tip is still reachable through the branch listed here;
their history was also saved in a verified Git bundle before cleanup. `next-card` was retired after its integration.
Remote refs were retained.

| Removed name | Tip | History retained in |
|--------------|-----|---------------------|
| `first-rental-smoke` | `d2e8184` | `codex/comfyui-integration` |
| `mtp` | `5c9cd5e` | `codex/comfyui-integration` |
| `phase2` | `f3031ea` | `codex/comfyui-integration` |
| `pictures` | `7fdd58f` | `codex/comfyui-integration` |
| `mtp-min-p` | `8672e62` | `codex/comfyui-integration`, after the `next-card` integration |
| `next-card` | `8672e62` | `codex/comfyui-integration` |
| `monitoring` | `b0b0703` | `next-card-metrics` |
| `throughput` | `a03a62b` | `held-criterion` |

The clean `~/work/simple-serving-pictures` checkout was removed with its reproducible Python environment and
caches. The clean `/var/tmp/simple-serving-next` checkout was retired after the MTP integration.
Their code is in the current branch. One stale registration for a missing scratch checkout was pruned.
The saved cleanup stash was preserved.

## Reading a remaining branch

The refs also work from the main checkout without switching it:

```sh
git show next-card-metrics:docs/metrics.md
git show thinking-budget:docs/thinking-budget.md
git show vision-captions:docs/vision-smoke.md
git show held-criterion:docs/speed.md
git show pictures-two-cards:docs/pictures-two-cards.md
git diff codex/comfyui-integration...next-card-metrics -- simple_serving
```

This is a dated inventory. Recheck the branch tips and worktree state before integrating or retiring anything.

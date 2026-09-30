# Existing branches, 2026-09-30

The inventory began with 19 local branches, compared with `codex/comfyui-integration` at `c23c282`.
After retiring nine redundant or integrated names, 10 remain. This document records the remaining work,
integration constraints and the retired names. The review inspected Git ancestry, patch identity, source, tests and documents.
The initial inventory ran no tests, engines or rentals. The later MTP integration passed the shared checks on
Python 3.12 and 3.14, recorded in [decisions](decisions.md#mtp-refusal-integration-2026-09-30).
Monitoring now passes the shared checks on both Python versions; see
[its integration record](decisions.md#monitoring-integration-2026-09-30). Historical checks below are evidence
from the branches, not a fresh verification of their combined behavior.

## The current code

`c23c282` includes the text gateway, MTP, optional RAM KV offloading, the picture lane from `7fdd58f`, reference
cleanup fixes from `abcbabf`, and the shared check command. The bot's picture provider switch remains off by
default. The [picture card test](pictures.md#the-card-test) is still pending.

`main` at `7a60fc7` remains as the earlier text baseline, with its `origin/main` upstream. The first-rental smoke,
MTP, phase 2 and picture implementations are already in the current history; their old local names were removed.

The remaining branches are below. A candidate for integration still needs its changes applied to the current
interfaces and checked. A branch's tests passing before the picture lane existed would not establish that.

The `min_p` fix from `8672e62` is now integrated in the working code, adapted to the shared lifecycle.
The monitoring from `9413d16` is also integrated for both lanes. These are code-integration statuses, not fresh
GPU results.
The dense prefix retention change from `940471a` is now integrated in the text launcher, with its historical
comparison recorded in [speed](speed.md#dense-prefix-retention).
The thinking budget from `466a82d` is also integrated, with the text manifest switch still 0. Its
[validation procedure](thinking-budget.md) separates local checks from pending GPU enforcement and quality work.
Vision from `47d71ae` is now integrated too, with `VISION=0` and an explicit [synthetic smoke](vision-smoke.md).

## Branch work and integration status

| Branch | Tip | Contribution and verdict |
|--------|-----|--------------------------|
| `live-2026-09-28` | `940471a` | Dense prefix retention is integrated in the text launcher. The branch is retained as the source of that change; its history adds no remaining feature. A new GPU comparison is pending. |
| `thinking-budget` | `466a82d` | Optional thinking-token budgets are integrated and disabled by default. The source branch remains retained, with no unintegrated feature. GPU enforcement and quality evaluation are pending. |
| `vision-captions` | `47d71ae` | Includes `8672e62`. Implements image input for Gemma, key permissions, launcher/bootstrap and a vision smoke. Integrated into the current shared configuration, launcher and smoke, with `VISION=0`. Source retained; GPU validation is pending. |
| `held-criterion` | `a898060` | Includes throughput and fixes its cache-capacity stopping criterion. Workload and metric-selection ideas are reused in the interference client; engine-choice diagnostics from `63baef9` are ported separately. Its A/B launcher remains separate, and the revised criterion needs a GPU measurement. |
| `sweep` | `b6d4919` | Standalone sweep is ported into the current checkout, with explicit current cache capacity and local fake checks. It measures an already running engine; GPU results remain pending. The source branch retains its historical experiment queue. |
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

The monitoring from `9413d16` is integrated. Both lanes use the shared listener observer and expose `/metrics`
on their control listener. Text retains its stage, token and admission metrics and the `/metrics/engine` relay;
pictures expose aggregate jobs and unresolved stops. Route labels preserve picture templates, without client names.
The outcome catalog covers both lanes, including cancellation and failures after HTTP 200.

`Card.keys()` and both preparations accept an optional third metrics key, preserving older two-key installs.
CLI `--lane` remains shared; exporting a metrics key cannot overwrite its configuration or temporary file.
Prometheus has separate text and picture jobs; the picture target is opt-in. Shared Grafana panels select a lane,
and text rows remain text-only. The [catalog and setup](metrics.md) describe the interface and validation limits.

### Thinking budget and vision

The integrated thinking budget uses `thinking_token_budget` and `--reasoning-config`, without a vLLM patch.
It is off by default. Noninteger or negative values give `invalid_request`; a valid number while disabled gives
`thinking_budget_unavailable`. When enabled, the upper bound is the class's `max_tokens`, not the request's.
Counting checks type and sign, but not feature availability or the class bound. The historical fake reply used
only 11 thinking tokens, so it did not demonstrate enforcement of the proposed 128/300 budgets. The current smoke
labels fake enforcement `not_verifiable` and rejects a real check whose baselines do not exercise both caps.

Vision adds a manifest flag, `VISION=0` by default, and a per-key permission. It accepts one PNG/JPEG/WebP data URL
up to 2 MiB of compressed image bytes. The text HTTP body defaults to 4 MiB; explicit body limits still apply.
`Card.keys()` stays independent of the text `VISION` flag; `Card.config()` adds the text client permission.
Picture manifests do not need that switch, and picture configuration rejects a true text vision permission.
Outside keys cannot receive vision while upstream encoder embeddings are cached without the scope salt.
Budget and vision are functionally independent after the shared text fix. Budget's local checks cover the current
text and picture configuration/lifecycle, and vision preserves that behavior. Its synthetic color/count/privacy
smoke with a fake does not prove GPU tower capacity,
real captions, token counts or MTP throughput.

### Measurement tools

The [engine-choice diagnostic](runbook.md#engine-choices) from `63baef9` is ported into the text launcher.
It preserves fixed-code and numeric reporting without the A/B harness's in-memory history or lifecycle changes.
Requested arguments and recognised selection messages have different meanings; missing rows do not establish a
backend or readiness.

The [standalone sweep](sweep.md) from `b6d4919` is now ported into the current checkout. Its capacity is an explicit
input rather than a historical default; the source branch and historical experiment queue remain retained.

The [interference client](prefill-interference.md) now measures a long prompt arriving during another request's
generation through the existing gateway. It reuses selected workload and metric-selection ideas from
`held-criterion`, without integrating its launcher. Its default command uses a local fake; GPU interference and
the value of disaggregation remain unmeasured.

Use `held-criterion` once, without separately replaying `throughput`. Its corrected timing sweep distinguishes
waiting from preemption and explains the stop in its report. Preserve the current card subclassing and log-field
allowlist when carrying over the launcher, metrics and fake-engine changes.

The A/B launcher stops the serving pair, starts its measurement pair and leaves the card without serving when it
finishes. Decide how serving is restored before using it alongside readers. `sweep` instead measures a running
engine and does not restart it. The integrated tool requires current cache capacity and an explicit deadline.

The 2026-09-27 5060 Ti and 5090 results live in `held-criterion:docs/speed.md`. Further concurrency measurements,
RAM offloading with dense retention, and eval remain unfinished. The retained `live-2026-09-28` tree cites a
speed-document item missing from its own tree; the current [summary](speed.md#dense-prefix-retention) extracts its
comparison from the fuller measurement branch. The split-KV plugin needs
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
2. The [picture smoke](picture-smoke.md) prepares one generation and record-cleanup check. Complete the remaining
   ComfyUI single-card test before picture-provider activation.
3. Monitoring is integrated. Install the optional metrics key and enable the picture scrape target when setting
   up monitoring for the cards.
4. Engine-choice diagnostics from `held-criterion` are integrated. Before carrying over its remaining A/B tooling,
   implement and check serving restoration after success and failure. Measure retention separately.
5. Budget and vision are integrated but disabled. Check each on a GPU before activation; their enforcement,
   captions, memory use and combined MTP behavior remain separate measurements.
6. Keep the two-card design and split-KV work separate until their first implemented checks exist.

## Finding the work

The main checkout is `~/work/simple-serving`. The remaining clean worktrees are:

| Location | Branch |
|----------|--------|
| `~/work/simple-serving-two-cards` | `pictures-two-cards` |
| `~/work/simple-serving-vision` | `vision-captions` |
| `/var/tmp/simple-serving-live` | `live-2026-09-28` |
| `/var/tmp/simple-serving-budget` | `thinking-budget` |
| `/var/tmp/simple-serving-held` | `held-criterion` |

## Removed duplicates

The following local names were removed on 2026-09-30. Each tip is still reachable through the branch listed here;
their history was also saved in verified Git bundles before cleanup. `next-card` and `next-card-metrics` were
retired after their integrations.
Remote refs were retained.

| Removed name | Tip | History retained in |
|--------------|-----|---------------------|
| `first-rental-smoke` | `d2e8184` | `codex/comfyui-integration` |
| `mtp` | `5c9cd5e` | `codex/comfyui-integration` |
| `phase2` | `f3031ea` | `codex/comfyui-integration` |
| `pictures` | `7fdd58f` | `codex/comfyui-integration` |
| `mtp-min-p` | `8672e62` | `codex/comfyui-integration`, after the `next-card` integration |
| `next-card` | `8672e62` | `codex/comfyui-integration` |
| `monitoring` | `b0b0703` | `codex/comfyui-integration` |
| `next-card-metrics` | `9413d16` | `codex/comfyui-integration` |
| `throughput` | `a03a62b` | `held-criterion` |

The clean `~/work/simple-serving-pictures` checkout was removed with its reproducible Python environment and
caches. The clean `/var/tmp/simple-serving-next` checkout was retired after the MTP integration.
The clean `/var/tmp/simple-serving-metrics` checkout was retired after the monitoring integration. Its seven
untracked smoke logs were copied and hash-checked in
`/home/jo/Documents/Codex/2026-09-30/qj/work/monitoring-integration/source-metrics-logs`; their contents were not printed.
All three checkouts' code is in the current branch. One stale registration for a missing scratch checkout was pruned.
The saved cleanup stash was preserved.

## Reading a remaining branch

The refs also work from the main checkout without switching it:

```sh
git show codex/comfyui-integration:docs/metrics.md
git show thinking-budget:docs/thinking-budget.md
git show vision-captions:docs/vision-smoke.md
git show held-criterion:docs/speed.md
git show pictures-two-cards:docs/pictures-two-cards.md
git diff codex/comfyui-integration...thinking-budget -- simple_serving
```

This is a dated inventory. Recheck the branch tips and worktree state before integrating or retiring anything.

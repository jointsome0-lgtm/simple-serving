# Decisions and measurements

This is a historical record of the owner's choices and the measurements available when each choice was made.
Entries remain in date order, including choices that later entries supersede. The container-key plan and the ban on
JSON whitespace on 2026-09-25 were both superseded later that day. Read the later entry before using an older choice.

The [contract](contract-v2.md) defines the API and limits. The [runbook](runbook.md) holds the operating instructions
and the detailed 5090 measurements previously in the README. These records do not verify a current live deployment
or establish that outside-client or permanent use has been validated.

## Text serving

- 2026-09-24, the owner: our `internal` work keeps the card awake until the rental's deadline, and outside keys
  are issued by hand only.
- 2026-09-24, the owner: the service stops its own card after 13 minutes without work of ours, and stops rather
  than deletes it. The bot keeps the story flow only; everything about the card and the model service lives in
  simple-serving.
- 2026-09-25, the owner: the trial serves route A, a ready 4-bit NVFP4 conversion of the heretic that vLLM loads in
  tree. Route B, our own conversion, follows only if eval finds A's quality short.
- 2026-09-25, the owner: route A's weights get no mirror or copy, so if llmfan46's NVFP4 repository disappears,
  route B, our own conversion of the heretic's bfloat16 weights, follows as well.
- 2026-09-25, the owner: the first rental's bound is the operator, the Claude session that runs it, not the owner at
  Vast's console. It reads and deletes the instance with the owner's account key through simple-story-chat's
  `npm run gpu:rent`, and tells the owner at once when a deletion is not confirmed.
- 2026-09-25, the owner: on the first rental `vast_api_key` is the card's own container key, which `cli trial` reads
  over SSH and writes into the configuration without printing it. The rental finds out whether that key shows and
  resumes the instance from outside the card; if Vast refuses it, the operator deletes the trial as the
  [runbook's first-rental procedure](runbook.md#the-first-rental) says, since `up` reads the instance with that same key.
  A permanent rental keeps the owner's
  restricted key.
- 2026-09-25, measured on the first rentals: Vast answers the card's own key with 200 on the card and with 401 from
  the owner's machine, so the entry above cannot hold. The owner made a restricted key, which read, confirmed the
  stop of and resumed the next trial from outside. `vast_api_key` is that key on every rental, and `cli trial` writes
  only the instance's id and the SSH host.
- 2026-09-25, the owner, after the rehearsal's `frame` answer ran out its 900 tokens with 1668 characters: vLLM
  builds a schema's JSON with xgrammar alone and no whitespace between its tokens. Whitespace as the cause is not
  measured yet; on an answer cut short the smoke now counts its whitespace and its characters outside ASCII.
- 2026-09-25, the owner, after the action measurement's texts on a 5090: whitespace between a schema's tokens is
  allowed again, and xgrammar stays the one backend. With the ban, 5 of the 18 stories got a sheet with nobody on it,
  in 7 tokens that ended on `stop`, and so did the marker story on one card of two. The four of those five that are
  not sealed, asked again from the same excerpts with whitespace allowed, gave 4, 6, 4 and 4 people, twice each, and
  no answer ran to its limit. The runaway the ban was for, E2B's `frame`, is met as the bot meets it, by the 900-token
  limit and one retry; the smoke still fails an answer cut short.
- 2026-09-26, the owner, after route A's second eval against the llama.cpp Q6_K, with the texts kept and three
  judges: route A stays, with multi-token prediction on, 3 drafted tokens. Both held the same facts in their memory.
  Route A's scenes lost count of a stock more often and refuted a false premise less firmly, and its memory writes
  hyphens for dashes and fewer «ё». The owner takes that small loss for the throughput: on one 5090, two requests at
  once decoded at 123 to 171 tokens a second each, against the Q6_K's 136 on its one slot. Route B does not follow
  on quality; the entry on llmfan46's repository still holds.
- 2026-09-26, the owner: the bot sends the gateway two calls at once by default, so that two readers are served
  together, the reason route A was taken. Section 13's lanes without placement, left for later, are in.

- 2026-09-27, the owner: reject a generation's unsupported `min_p` at the gateway while MTP is on, with 400
  `min_p_unavailable` before any engine call, and check this refusal in the smoke's `fields` probe. On that day's
  5090 with 3 drafted tokens, the smoke stopped at `fields` after 2 of 10 probes. vLLM refused `min_p=0.05`, which
  the gateway then reported as `invalid_request`. With MTP disabled, the smoke passed 10 of 10. MTP remains on
  by default. This records the original run, not validation of the later gateway fix.

## Picture serving

Version 2 serves one text model. Pictures stay in simple-story-chat for now: the owner decided on 2026-09-26 that
they move here after round two of simple-story-chat's action measurement, with ComfyUI inside and the engine
unchanged, so that the bot has one service for both lanes. vLLM could not take ComfyUI's place on that date:
vLLM-Omni had Qwen-Image 2.1 only in open pull requests (#7759 and #8099), took at most 4 reference images where a
frame needs up to 7, loaded only the 33 GB diffusers weights, and could neither start from an image nor redraw under
a mask.

The owner asked on 2026-09-28 to begin the move. Phase 1 was written on the server's `pictures` branch and the
client's `pictures-via-serving` branch, with the provider switch off by default. The
[picture contract](pictures.md#15-phases) records the phases, source commit and card test.

## Local integration, 2026-09-30

The existing phase 1 server implementation from `pictures`, commit `7fdd58f`, is combined in this working checkout
with the documentation cleanup and the shared check command. Its gateway is in
[`simple_serving/pictures/`](../simple_serving/pictures/), and its card files are in
[`card/pictures/`](../card/pictures/). The text API remains contract v2; the picture contract is separate.

The matching simple-chat client is integrated on its current checkout, retaining the newer bot changes. Cleanup
now protects a newer upload from an older request's failure or expiry, and stays armed until the HTTP answer
finishes. The client also keeps its submission timeout until the gateway accepts the job.

Local verification: ruff and mypy pass; all 283 server tests pass on Python 3.12 and 3.14. The client type check passes,
and its tests pass with 300 successful tests and 3 tokenizer-dependent skips. A synthetic bot-to-gateway run delivers
the same five pictures and normalized graphs as direct ComfyUI, with reference cleanup, reader isolation and
cancellation confirmed.

Real ComfyUI card validation remains pending. The bot's provider switch stays off by default; these checks use
local fake services and synthetic data. The next step is the [card test in section 15](pictures.md#the-card-test).

## MTP refusal integration, 2026-09-30

The `min_p` fix from `8672e62` is adapted to the current text `Service` and shared lifecycle in this working code.
It preserves contract v2 and the picture lane. The shared check command passes on Python 3.12 and 3.14, including
ruff, mypy and all 285 tests. The added matrix checks MTP on and off, greedy sampling, both sides of the threshold,
the class-limit order, count behavior and no engine calls for refused requests. The privacy test covers the new
error code, and the synthetic smoke expects it under MTP. Independent review found no blocking issue.
These are local checks against fake engines. The real card checks remain pending.

## Monitoring integration, 2026-09-30

The monitoring from `9413d16` is adapted to the current text and picture gateways. Both control listeners serve
shared request and lifecycle metrics. Text retains its token, timing and admission metrics and the vLLM relay;
pictures report aggregate jobs, class limits and unresolved stops. Request labels use fixed route templates.
The metrics key reads metrics without control or inference rights, and older two-key installations still work.
The CLI export refuses paths that would overwrite the configuration or either writer's temporary file.

Prometheus separates the two gateway jobs and the text engine. Picture scraping is opt-in. Shared Grafana panels
select a lane; text queue and engine panels remain scoped to text. History may stay outside the checkout.
The setup and metric catalog are in [metrics](metrics.md).

The shared check command passed on Python 3.12 and 3.14: ruff, mypy and 328 tests, with one optional vLLM source
catalog check skipped because no source checkout was supplied. Picture scenarios cover permissions, private route
parts, cancellation, unresolved stops, class capacity, drain and idle expiry despite scrapes. The pinned Prometheus
image's `promtool check config` validated the configuration and all 11 rules without networking, using synthetic
secret files. A final dashboard correction passed the catalog and selector checks again. Independent review found
no remaining blocker. Grafana rendering, live scrapes and real GPU behavior were not checked in this integration.

## Prefill interference groundwork, 2026-09-30

The owner asked whether to begin Splitwise-style prefill/decode disaggregation and accepted a measurement first.
The new [interference client](prefill-interference.md) sends synthetic requests through an existing text gateway,
comparing generation alone with a cold long prompt and its exact warmed repeat. It records chunk pauses, overlap,
cache usage and selected numeric diagnostics. It selectively reuses workload and metric-selection ideas from
`held-criterion` at `a898060`; that branch's launcher and remaining throughput work stay separate.

The default command runs against a local fake. A real run requires explicit configuration and a new report file,
checks ready state, quiet gauges and admission capacity, and has request and whole-run deadlines. It performs no
card lifecycle operations or retries. Reports omit keys, scopes, aliases, prompts, output and raw metric labels.

The shared check command passed on Python 3.12 and 3.14: ruff, mypy and 334 tests, with the same optional source
catalog check skipped. Six added tests cover a synthetic cold pause and warm reuse, a full freeze until the second
request finishes, fragmented SSE and metadata, failure cleanup, CLI boundaries, numeric diagnostics, monitor
shutdown and the actual gateway contract. Review corrected the full-freeze criterion and a monitor shutdown race.
The standalone default command also saved a successful owner-only synthetic report.

No GPU result or latency SLO decision exists for this scenario. KV transport and separate prefill/decode workers
are not implemented. Their value still depends on a controlled reader-pause measurement and a one-card batching
and cache comparison, followed by compatible-engine and cost checks if the pause warrants disaggregation.

## Dense prefix retention integration, 2026-09-30

The owner accepted `live-2026-09-28` as the next integration. Its change from `940471a` now reaches the current
text launcher as `--prefix-cache-retention-interval None`. The pinned vLLM 0.30.0 supports this value. It retains
sliding-window prefix checkpoints densely within the existing GPU block pool, preserving eviction and scope salts.
The [speed record](speed.md#dense-prefix-retention) summarizes the original 2026-09-27 RTX 5090 comparison from
`held-criterion` at `a898060`, separately from the current local verification.

The shared check command passed on Python 3.12 and 3.14: ruff, mypy and all 334 tests, with the same optional source
catalog check skipped. The existing launcher test now checks the argument received by its engine stand-in.
Independent review found no remaining blocker. No card was started or restarted and no new GPU measurement was
made. The setting takes effect when the text engine next starts from this checkout; its current performance gain
still needs a controlled comparison. The source branch remains retained, with no remaining unintegrated feature.

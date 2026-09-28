# The vision smoke, a job for the next rental

Written on 2026-09-28 for the next rental's queue, in the form of simple-story-chat's queues
(`docs/action-experiment.md#tester-rental` on its branch `tester-stand`). Nothing is rented, and none of it runs on
today's cards, whose text card runs `next-card` (8672e62). It needs this branch, `vision-captions`, which is not
merged: the text card runs it for this job alone, once the queue's own work on `next-card` is done.

## The job

- **The vision smoke**, last on the text card: after the queue's last job, once `calm 8080` has found no request of
  the bot's in flight and `up` has stopped, and only with 26 minutes left before the card's end. About 6 to 10
  minutes, $0.06 to $0.09 at $0.55 an hour, with no download; 26 minutes and $0.24 at most, if every step runs to its
  timeout. Under 10 KB back, numbers and names alone. Does the heretic's vision tower fit on the 5090 beside the
  drafter at 65536 tokens, and how much of the readers' cache does it take? Does the card read a synthetic picture
  right, count it as vLLM bills it, and keep every piece of it out of its logs? Nothing on the card waits for it, so
  it is dropped whole if Astra does not back it; within it, the drafter's counters go first, then the second start.

**The gain for the bot.** A pose set's pictures are labelled once each, and a frame picks among the groups that the
labels make, by Gemma's `view` field (simple-story-chat's `docs/telegram-ui.md#pose-set`, on its branch `pose-refs`).
openjev 0.8B labels them on the owner's CPU: 87 of 94 whole figures in the right group on characters it was not fitted
on, and its weakest label is the three-quarter turn, 62 of 84, most of the rest read as fronts, so a frame that asks
for a three-quarter view may get a front (`docs/knowledge/pose-captioner-2026-09-28.md` there). The card's Gemma 4
31B may read the turn better. This job does not measure that. It is the gate for the job that would, on a later card:
the 218 synthetic pictures of that measurement labelled once each by the card's Gemma with the same labels, and scored
as openjev was, some 440 short requests with their mirrors. The tower's cost in cache also decides whether it could
stay on while readers use the card, or goes on for a labelling session alone. If the tower does not fit, or a piece of
a picture reaches a log, `VISION` stays 0 and the bot keeps openjev: the job's minutes are all it costs. A reader's
own pictures would go to the card only after that, and as the owner decides, since pose sets keep them on the owner's
machine today.

## Minutes

The text card was prepared from `next-card`, whose manifest has no `VISION`. This branch's manifest differs from it
even at `VISION=0`, so its stamp does too, and the launcher starts nothing until the preparation runs again (README,
"The card"). That preparation downloads nothing: the locks are `next-card`'s, and the processor's three files are
copies of the tokenizer's, which the first preparation fetched.

| Step | Minutes | From |
| --- | --- | --- |
| The drafter's counters of the queue's text; the checkout copied with `VISION=1` | under 1 | |
| The preparation again, with nothing on stdin: both locks found installed, every pinned file's SHA-256 checked again, some 20 GB of weights and the drafter's 0.9 GB, and the processor's files copied | 1 to 3 | not measured; the first preparation's 9.2 minutes on 2026-09-27 were mostly downloads |
| `--stop`, and the first start with the tower | 3 to 4 | vLLM compiles again, since `language_model_only` is among the factors of its compile cache's hash (`vllm/config/model.py` and `vllm/config/vllm.py` at v0.30.0); a fresh container's first start took 137 s on 2026-09-27 |
| `--stop`, and a second start, with the compile on disk | 1 to 1.5 | 29 to 35 s on 2026-09-27 |
| The forward, the probe and the counters | 1 | the dry run below took 1.2 s against the fake |

So 6 to 10 minutes, $0.06 to $0.09. The runbook's timeouts add up to 26 minutes, $0.24. With the job the queue's text
card comes to $0.51 to $0.67 with its download, against $0.45 to $0.58, and to $0.96 at most, against $0.72.

## What it shows

The probe's exit 0 needs the rows of the caption, the count, privacy and the refusals, and the tower fits once
`status` says ready. The rest are numbers for the queue's record, beside the ones given.

| What | Read from | Passes when, or compared with |
| --- | --- | --- |
| The tower fits | `status` after each start | `gateway: ready`. `vast: stopped` means that the launcher gave up, `kv_cache_too_small` most likely, and the job ends there |
| KV tokens | `kv_cache_tokens` and `concurrency_x100` in the launcher's rows | the second start's beside 151300, a start with the compile on disk and no tower on 2026-09-27, and the first start's beside the 104492 of a first start. Some 19000 fewer are expected for the tower's 1.07 GiB, and the encoder's working memory besides |
| Load time | `load_ms` and `weights_mib` there | beside 137 s and 29 to 35 s, and 20070 MiB of weights with the drafter (2026-09-26) |
| One synthetic caption | the probe's `red` and `blue` | the answer names the square's colour, and the stream is as section 4 of the contract has it |
| The count against usage | `input_tokens` and `prompt_tokens` in `red` and `blue`; `picture_tokens` in `tokens` | equal; and a picture adds more than 0 tokens and at most 282, 258 for these squares with the pinned template |
| TTFT | `first_token_ms` in `red`, `blue` and `text` | the first picture since the start, a second that no cache holds, and the same question without a picture |
| The drafter's acceptance with pictures | vLLM's counters after the probe; `decode_per_s` in `describe` | accepted over drafted tokens, and each position's accepted tokens over the drafts, beside the queue's text before its pair ended and 21718 of 34343 on 2026-09-26. The new pair's counters hold the probe's answers alone: the description's 192 tokens at most, 8 at most for each caption, and the 8 of `text`, the one without a picture |
| Privacy | `privacy` | the marker, in each picture's PNG text chunk and spelled by its data URL's base64, `found` 0 times in every file of the card's `logs/`, with `logged` above 0 |
| Refusals | `two_pictures` and `remote` | 400 `invalid_request` from the gateway, before the engine |

## The client

The bot sends simple-serving no picture, so the client is this repository's smoke, `python -m simple_serving.smoke
--only vision`, on the owner's machine (README, "The smoke"). The probe makes its pictures in memory, plain squares and
one of four colours, 64 pixels a block and a few hundred bytes each, and sends them with the configuration's client
key, which a card with `VISION=1` gives the right `vision`. It goes through a forward of its own on 8301 and 8302, not
`up`'s 8080, which the live bot follows: the bot stays off the card while its pair starts twice and while the probe
measures.

## The runbook

After the text card's last job, `calm 8080` and Ctrl+C in `up`'s terminal, as the queue's end has them. `on`, `py`,
`end_text` and the host `simple-chat-vast` are the queue runbook's; the `git` and `uv` lines run in this checkout at
`vision-captions`. A step that fails ends the job, and the queue's end follows, the guard's deadline and `--destroy`;
nothing is fixed or retried on the paid card.

```sh
(( $(date +%s) + 1560 <= end_text )) && echo room    # else the job is dropped
mkdir -p logs; set -o pipefail
on simple-chat-vast 30 "curl -s -m 10 http://127.0.0.1:8092/metrics | grep -E '^vllm:spec_decode_num_[a-z_]+_total'" | tee logs/vision-drafts-text.txt    # the queue's text
git archive vision-captions | on simple-chat-vast 120 'tar -xf - -C /workspace/simple-serving && sed -i s/^VISION=0$/VISION=1/ /workspace/simple-serving/card/manifest.env && grep -cx VISION=1 /workspace/simple-serving/card/manifest.env'    # 1
on simple-chat-vast 300 bash /workspace/simple-serving/card/bootstrap.sh < /dev/null; echo "bootstrap $?"    # 0, while the old pair serves on
numbers() { on simple-chat-vast 20 "grep -oE '\"(weights_mib|kv_cache_tokens|concurrency_x100|load_ms|code)\": (\"[a-z_]+\"|[0-9]+)' /workspace/simple-serving-card/logs/card.jsonl | tail -n 6" | tee -a logs/vision-launcher.txt; }    # the last start's
again() { on simple-chat-vast 120 "cd /workspace/simple-serving && $py -m simple_serving.card --stop && $py -m simple_serving.card"; echo "start $?"
  timeout "$1" bash -c 'until uv run python -m simple_serving.cli status | grep -qE "^(gateway: ready|vast: stop)"; do sleep 10; done'
  uv run python -m simple_serving.cli status; numbers; }
again 300    # the tower's first start, which compiles: gateway ready
again 150    # with the compile on disk
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L 127.0.0.1:8301:127.0.0.1:8090 -L 127.0.0.1:8302:127.0.0.1:8091 simple-chat-vast    # a terminal of its own
timeout 330 uv run python -m simple_serving.smoke --only vision --public-port 8301 --control-port 8302 | tee logs/smoke-vision.jsonl    # exit 0, 1 of 1
on simple-chat-vast 30 "curl -s -m 10 http://127.0.0.1:8092/metrics | grep -E '^vllm:spec_decode_num_[a-z_]+_total'" | tee logs/vision-drafts-pictures.txt
# Ctrl+C in the forward's terminal, then the queue's end.
```

What comes back is those four files in `logs/`: counters, the launcher's numbers and codes, and the probe's line.
None holds a key, a prompt, an answer, the marker or a piece of a picture.

## The dry run

On 2026-09-28, as the README's dry run, with the dev launcher configured as the card's gateway with the tower: the
manifest's alias, context and pins, `"vision": "1"` among the versions, and `"vision": true` on the smoke's client key.
`--only vision --fake --card-dir` passed 1 of 1 in 1.2 s. `red` and `blue` counted 296 tokens each and `text` 14, so
`picture_tokens` 282, the fake's count of a picture; `describe` took 13 tokens; `two_pictures` and `remote` were 400
`invalid_request`; and the marker was in no file of the card's `logs/`, where 6 rows of the requests were. Neither
the line nor the gateway's rows held a key, a prompt, `data:image`, `base64,`, the PNG's `iVBOR` or a run of 32 hex
digits.

Two leaks were then planted in a copy of the gateway, outside this checkout: one wrote each request's body to its log,
the other each picture's bytes. The probe failed on each with `marker`, 7 in `gateway.jsonl`: every picture it sent,
the refused pair's included. `tests/test_smoke.py` keeps the first of them in small. What only the card shows: whether
the tower fits, what Gemma answers, whether the counts equal the usage, the times and the drafter's counters.

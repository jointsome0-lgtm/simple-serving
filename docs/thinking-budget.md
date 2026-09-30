# Thinking budget

The change from `thinking-budget` at `466a82d` is integrated into the shared text and picture checkout on
2026-09-30. `THINKING_BUDGET=0` remains committed in the text manifest. The text gateway refuses the optional
`thinking_token_budget` field until the switch is enabled. Picture manifests and the picture launcher do not use
this switch. Local fake checks cover the route, launcher arguments and smoke verdict; Gemma 4 with MTP has not yet
demonstrated budget enforcement on a GPU in this integration.

This document adapts the rental plan written on 2026-09-28. Its measurements and cost estimates below are dated
background. The original plan required a fresh Astra review before running the job. No rental, card restart or
feature activation is part of the local integration.

## The job

The proposed budget check runs last on a text card, after its other jobs, with at least 45 minutes left before the
card's deadline. It asks whether vLLM holds Gemma 4's thought to a request's budget while MTP drafts tokens, whether
the answer after the forced end is complete and follows its schema, and how long calls take at 128, 300 and 1024
tokens of thought compared with thinking off and an unbudgeted baseline. It returns fixed labels, counts and
timings, without prompts or generated text.

The original note records simple-story-chat's O3 probe of 2026-09-28. Hosted Gemma 4 31B passed 58 of 76 trap
questions without thinking and 59 with it, and a thinking scene took 38.7 and 38.1 s at the median against 8.0 and
12.5 s. Thinking at length bought nothing for its time; a thought of a few hundred tokens might keep what it does
help with at a fraction of the time. Hosted models cannot say: OpenRouter asked Gemma for 512 tokens of thought, and
it thought for 3,122. Those hosted observations motivate a local experiment. The budget smoke checks enforcement
and answer structure; a later trap-question evaluation would measure quality. The committed switch stays off
until the GPU check passes and activation is separately chosen.

## What vLLM 0.30.0's source says

The text manifest still pins vLLM 0.30.0, model revision `24ac337e`, tokenizer revision `76ac65c7`, and the MTP
assistant at `627c5ec1` with three drafts. `card/vllm-requirements.txt` and the manifest's file hashes pin the
installed files. The following source analysis was recorded at tag v0.30.0 on 2026-09-28. The integrated request
validation and marker configuration were rechecked against the pinned source on 2026-09-30.

- **The field.** `thinking_token_budget` on the chat request (`vllm/entrypoints/openai/chat_completion/protocol.py:258`,
  into the sampling parameters at `:756`), a count from 0, or -1 for none (`vllm/sampling_params.py:51-73`). vLLM
  refuses it unless its reasoning config is enabled (`vllm/v1/engine/input_processor.py:115-123`). The gateway's
  narrower contract accepts only nonnegative integers, omits the field when absent, and limits an enabled budget
  to the request class's `max_tokens`. Counts validate type and sign without requiring the feature to be enabled.
  See the pinned [sampling parameters](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/sampling_params.py)
  and the local [text contract](contract-v2.md).
- **The flag and the markers.** `--reasoning-config` takes `reasoning_start_str` and `reasoning_end_str`
  (`vllm/config/reasoning.py:22-27`). Without them vLLM takes the reasoning parser's markers (`reasoning.py:83-95`), and
  `--reasoning-parser` alone makes the config (`vllm/engine/arg_utils.py:2864-2869`, enabled at
  `vllm/config/vllm.py:1956-1957`). Gemma 4's parser has `<|channel>` and `<channel|>` (`vllm/parser/gemma4.py:38-39`
  and `:305-306`, read through `vllm/parser/engine/parser_engine.py:160-165`), which the pinned tokenizer names
  `soc_token` and `eoc_token` (`tokenizer_config.json:66` and `:8` at 76ac65c, whose hashes match the manifest). So the
  source can initialize the markers from its parser; the launcher pins them explicitly when enabled. The
  [reasoning configuration](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/reasoning.py) describes
  forcing the closing marker. Actual enforcement with this Gemma model, tokenizer and MTP remains a GPU question.
- **The chat template** (the same revision): with thinking on it writes `<|think|>` into the system turn
  (`chat_template.jinja:193-194`), and the model opens its thought itself; with thinking off the prompt ends in an
  empty thought, `<|channel>thought\n<channel|>` (`:361-362`), and there is nothing to hold.
- **With MTP.** The budget is not a logits processor, which speculative decoding would drop
  (`vllm/v1/sample/logits_processor/__init__.py:201-210`). Model Runner V2, the default on CUDA
  (`vllm/config/vllm.py:675-719`) and not refused for method `mtp` (`:2839-2850`), holds it in its sampler
  (`vllm/v1/worker/gpu/sample/sampler.py:57` and `:251-258`), and the rejection sampler runs the target's logits
  through it at every drafted position before it accepts or rejects a draft
  (`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:180-188`). Gemma 4 has no sampler of its own there
  (`vllm/v1/worker/gpu/model_states/interface.py:228-236`). The V1 runner does the same
  (`vllm/v1/sample/rejection_sampler.py:352-358`). The pinned
  [reasoning documentation](https://github.com/vllm-project/vllm/blob/v0.30.0/docs/features/reasoning_outputs.md)
  describes support and examples. It is not a test report for this Gemma 4 and MTP combination.
- **The compile cache.** Its key is the hash of the environment's factors, `VllmConfig.compute_hash()`, the traced
  code and the compiler (`vllm/compilation/backends.py:1030-1062`), and `compute_hash` leaves the reasoning config out
  (`vllm/config/vllm.py:462-577`). This informed the original warm-start estimate; a current card still needs its
  own startup and compilation observations.

## Historical time estimates

The 2026-09-28 plan assumed an already prepared card and a rate of $0.55 an hour. These are planning estimates,
not a current offer or a billing limit.

| Step | Minutes | From |
| --- | --- | --- |
| The checkout copied with `THINKING_BUDGET=1`, on this machine and to the card | under 1 | |
| The preparation again, with nothing on stdin: both locks found installed, every pinned file's hash checked again, nothing fetched | 1 to 3 | the manifest's stamp changes with the switch; the first preparation's 9.2 minutes on 2026-09-27 were mostly downloads |
| `--stop`, and a start with the compile on disk | 1 to 1.5 | the flag is not among the compile cache's factors; such a start took 29 to 35 s on 2026-09-27 |
| The forward, and the smoke's `state` and `budget` | 2 to 5 | 15 generations and 30 counts; up to 3 thoughts without a budget, at most 8192 tokens each, at the 123 to 171 tokens a second measured on 2026-09-26 |

That plan estimated 5 to 10 minutes, $0.05 to $0.09. Its step budgets added up to about 42 minutes, $0.38:
1 for the copy, 10 for the preparation,
15 for the load (`LOAD_DEADLINE_S`, after which the launcher stops the instance), and the smoke's own bounds, 30 s
for `state` and 15 minutes for `budget`.

## What it shows

The enabled probe makes 15 generations across three prompts and five arms, with at most 8192 completion tokens
per call, under one 900-second deadline. A complete run also makes 30 counts. Each capped answer must end
in `stop` and follow its schema. SSE errors, missing `[DONE]`, invalid token subtraction and exceeding the time
bound fail the probe. Text stays in memory for these checks and is omitted from the output.

`thinking_tokens` estimates the thought length by subtracting the separately counted answer from
`usage.completion_tokens`. `within` allows 8 extra tokens for marker and tokenization differences. Neither the
subtraction nor that allowance establishes an exact reasoning-token count. A negative estimate fails as
`token_count`.

A baseline is `bound` only when its estimate exceeds the cap by more than that allowance. On a real engine,
`bound_counts` must be at least 2 for both 128 and 300, and every stream and capped answer must pass. Only then
does the line report `enforcement: observed`. Otherwise an unexercised cap gives `enforcement: inconclusive`,
`ok: false`, `budget_not_bound`, and a nonzero smoke exit. The 1024-token cap can still be unexercised; its own
count says so. `seconds` and puzzle correctness `right` remain observations.

With `--fake`, successful routing can give `ok: true`, but `enforcement: not_verifiable` and `not_verifiable`
explicitly exclude cap enforcement, `within`, `bound`, answer quality and correctness. A short fake thought that
ends naturally below every cap cannot prove that any cap was enforced. Disabled mode reports
`enforcement: disabled` after checking the gateway's refusal.

A successful real check supports a separate activation decision and a later quality evaluation. Until then,
the committed switch remains 0. Keep the report and pinned versions with any future GPU result.

## The runbook

For a separately reviewed card job, use a copy of the current integrated checkout. It must contain both the
gateway change and the text launcher's marker configuration. Change only that copy:

```
sed -i 's/^THINKING_BUDGET=0$/THINKING_BUDGET=1/' card/manifest.env   # in the copy only, never committed
```

Copy that checkout to `/workspace/simple-serving` on the selected text card, then prepare again and restart as
described in the [text-card runbook](runbook.md#the-card):

```
ssh <card> timeout 600 bash /workspace/simple-serving/card/bootstrap.sh < /dev/null
ssh <card> 'cd /workspace/simple-serving &&
  /workspace/simple-serving-card/gateway/bin/python -m simple_serving.card --stop &&
  /workspace/simple-serving-card/gateway/bin/python -m simple_serving.card'
```

In the copy, first terminal: `uv run python -m simple_serving.cli up`, until it says `ready`. Second terminal, also
in the copy, since the smoke reads the switch and the pins from its manifest:

```
mkdir -p logs && uv run python -m simple_serving.smoke --only state,budget > logs/thinking-budget.jsonl; echo $?
```

Bring back `logs/thinking-budget.jsonl` and the exit code, then end the card as the queue does.

## Local checks and their limits

The source branch recorded a dry run on 2026-09-28 with the dev launcher configured from the manifest, synthetic
keys, and the smoke's
`fields` and `budget` with `--fake`. With `THINKING_BUDGET=0`, as committed: the command that the card's launcher builds
for vLLM has no `--reasoning-config`, and both probes pass on the gateway's 400 `thinking_budget_unavailable`, before
any count. In a copy with `THINKING_BUDGET=1`: vLLM gets `--reasoning-config` with `<|channel>` and `<channel|>`,
`fields` answers the budget of 64 with 200, and `budget` passes its 15 calls, the gateway's rows holding the budgets 64,
128, 300 and 1024 as numbers. The fake's thought is 11 tokens, so no budget is `bound` there: whether vLLM holds one,
only the card shows. That historical green line is a route check, not evidence of a held cap.

The integrated tests cover both disabled and enabled fake routes without changing the manifest. Scripted streams
also exercise binding and nonbinding baselines, the 8-token allowance, overflow, invalid subtraction, SSE errors,
missing `[DONE]`, timeout cancellation and output privacy. These tests validate the smoke's decisions; a real
Gemma 4 and MTP run is still pending.

# The thinking budget, a job for the next rental

Written on 2026-09-28 for the next text card's queue. Nothing is rented, and none of it runs on today's cards, whose
text card runs `next-card` (8672e62). It needs the branch `thinking-budget`, or `next-card` once the main session has
merged it, with `THINKING_BUDGET=1` in a copy of the checkout that is not committed. It runs only if a fresh Astra
review backs it, the owner's rule for rentals.

## The job

- **The budget check**, last on the text card, after the queue's own jobs, with at least 45 minutes left before the
  card's end. About 5 to 10 minutes, $0.05 to $0.09 at $0.55 an hour, with no download; 42 minutes and $0.38 at
  most, if every step runs to its timeout. Under 10 KB back, numbers and names alone. Does vLLM hold Gemma 4's
  thought to a request's `thinking_token_budget` while it drafts tokens with MTP, does the answer after the forced end
  come out whole and in its schema, and how many seconds does a call take at 128, 300 and 1024 tokens of thought,
  against no thinking and thinking to its own end? Nothing waits for it, so it is dropped whole if Astra does not
  back it.

**The gain for the bot.** In simple-story-chat's O3 probe of 2026-09-28, hosted Gemma 4 31B passed 58 of 76 trap
questions without thinking and 59 with it, and a thinking scene took 38.7 and 38.1 s at the median against 8.0 and
12.5 s. Thinking at length bought nothing for its time; a thought of a few hundred tokens might keep what it does
help with at a fraction of the time. Hosted models cannot say: OpenRouter asked Gemma for 512 tokens of thought, and
it thought for 3,122. This job does not measure quality. It is the gate for the measurement that would, the trap
questions again with a budget, and until it passes, the gateway refuses every budget with 400
`thinking_budget_unavailable`.

## What vLLM 0.30.0's source says

At tag v0.30.0 (ced6857), read on 2026-09-28:

- **The field.** `thinking_token_budget` on the chat request (`vllm/entrypoints/openai/chat_completion/protocol.py:258`,
  into the sampling parameters at `:756`), a count from 0, or -1 for none (`vllm/sampling_params.py:51-73`). vLLM
  refuses it unless its reasoning config is enabled (`vllm/v1/engine/input_processor.py:115-123`).
- **The flag and the markers.** `--reasoning-config` takes `reasoning_start_str` and `reasoning_end_str`
  (`vllm/config/reasoning.py:22-27`). Without them vLLM takes the reasoning parser's markers (`reasoning.py:83-95`), and
  `--reasoning-parser` alone makes the config (`vllm/engine/arg_utils.py:2864-2869`, enabled at
  `vllm/config/vllm.py:1956-1957`). Gemma 4's parser has `<|channel>` and `<channel|>` (`vllm/parser/gemma4.py:38-39`
  and `:305-306`, read through `vllm/parser/engine/parser_engine.py:160-165`), which the pinned tokenizer names
  `soc_token` and `eoc_token` (`tokenizer_config.json:66` and `:8` at 76ac65c, whose hashes match the manifest). So the
  card's vLLM already takes a budget today, and the flag only pins the markers; the gateway is what refuses the field.
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
  (`vllm/v1/sample/rejection_sampler.py:352-358`). vLLM's documentation names Qwen3, DeepSeek and Nemotron as tested
  with the budget (`docs/features/reasoning_outputs.md:250`), not Gemma 4: only a card shows it.
- **The compile cache.** Its key is the hash of the environment's factors, `VllmConfig.compute_hash()`, the traced
  code and the compiler (`vllm/compilation/backends.py:1030-1062`), and `compute_hash` leaves the reasoning config out
  (`vllm/config/vllm.py:462-577`). The switch costs no compile.

## Minutes

| Step | Minutes | From |
| --- | --- | --- |
| The checkout copied with `THINKING_BUDGET=1`, on this machine and to the card | under 1 | |
| The preparation again, with nothing on stdin: both locks found installed, every pinned file's hash checked again, nothing fetched | 1 to 3 | the manifest's stamp changes with the switch; the first preparation's 9.2 minutes on 2026-09-27 were mostly downloads |
| `--stop`, and a start with the compile on disk | 1 to 1.5 | the flag is not among the compile cache's factors; such a start took 29 to 35 s on 2026-09-27 |
| The forward, and the smoke's `state` and `budget` | 2 to 5 | 15 generations and 30 counts; up to 3 thoughts without a budget, at most 8192 tokens each, at the 123 to 171 tokens a second measured on 2026-09-26 |

So 5 to 10 minutes, $0.05 to $0.09. The timeouts come to 42 minutes, $0.38: 1 for the copy, 10 for the preparation,
15 for the load (`LOAD_DEADLINE_S`, after which the launcher stops the instance), and the smoke's own bounds, 30 s
for `state` and 15 minutes for `budget`.

## What it shows

The `budget` line passes when, for each budget and each of the three prompts, the answer ends in `stop` and follows
its schema, and the thought took at most the budget and 8 tokens (`within`: the completion's tokens less the
answer's, counted apart, so the two markers and the end of the turn fall in the 8). It stands for the budget only
where the thought without one ran past it (`bound`): if 128 and 300 are not bound on at least two prompts, the check
is inconclusive for them, whatever it passes. Then, as observations: each call's `seconds` against `off` and `full`
on its prompt, the `thinking_tokens`, and whether the puzzle came out `right`.

If it passes with 128 and 300 bound, the owner may set `THINKING_BUDGET=1` on `next-card`, and simple-story-chat may
measure its trap questions with a budget, as a job of its own. If `within` or `answer` fails, the budget does not hold
on this path: the switch stays 0, and the line goes here as the evidence.

## The runbook

On this machine, in a copy of the checkout at `thinking-budget` (or `next-card` once merged):

```
sed -i 's/^THINKING_BUDGET=0$/THINKING_BUDGET=1/' card/manifest.env   # in the copy only, never committed
```

Copy that checkout to `/workspace/simple-serving` on the card as the queue copies it, then prepare again and restart
(README, "The card"):

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

## The dry run

On 2026-09-28, the dev launcher configured as the card's gateway from the manifest, with synthetic keys, and the smoke's
`fields` and `budget` with `--fake`. With `THINKING_BUDGET=0`, as committed: the command that the card's launcher builds
for vLLM has no `--reasoning-config`, and both probes pass on the gateway's 400 `thinking_budget_unavailable`, before
any count. In a copy with `THINKING_BUDGET=1`: vLLM gets `--reasoning-config` with `<|channel>` and `<channel|>`,
`fields` answers the budget of 64 with 200, and `budget` passes its 15 calls, the gateway's rows holding the budgets 64,
128, 300 and 1024 as numbers. The fake's thought is 11 tokens, so no budget is `bound` there: whether vLLM holds one,
only the card shows.

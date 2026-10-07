# Worlds measurement

`python -m simple_serving.interference.worlds` sends the load of many simulated worlds through a text gateway and
reports what the engine did with it. A world is 5 sessions, each with a system text of about 2500 tokens and a
request that grows by about 300 tokens a turn from 3000 towards 16000, then starts over. The sessions of a world take
turns, one request at a time, and the worlds run at once. Every answer is at most 200 tokens under a strict JSON
schema, with thinking off. The text is invented lines; the report holds counts and times, and no prompt, key or
answer.

## The dry run

```sh
uv run python -m simple_serving.interference.worlds
```

It starts this repository's gateway in front of the fake engine, three times, and prints the runs. The fake's prefix
cache is a stand-in: whole blocks chained from the cache salt, the least recently used dropped first, 100000 tokens in
the first two runs and 2000000 in the third. The first is too small for twenty sessions of 12000 tokens and hits
nothing; the second puts what it drops into a second store, as into a copy in RAM, and finds it there; the third
holds them all. That is the picture a copy of the KV cache in RAM is meant to change, and the dry run shows only what
the report looks like in each case. Its times and sizes are the stand-in's and predict nothing.

`totals.engine_hit_tokens` are the engine's own counters over the run, read through `/metrics/engine`: prompt tokens
served from the GPU's cache (`local`) and from the copy (`external`), and `engine_preemptions` the requests it put
back for want of room. `python -m simple_serving.interference.probes` has two smaller measurements on the same
stand-in: `reload`, what a prefix costs to bring back, and `schemas`, what a new schema costs. Their one planned use
is [the cheap card](cheap-card.md).

## Against a card

Through the tunnel, with the CLI's configuration (`client_key` and `control_key`), into a new report file:

```sh
uv run python -m simple_serving.interference.worlds run --config ~/.config/simple-serving/cli.json \
    --out worlds-off.json --worlds 4 --turns 8
```

It costs card time and sends 160 requests of 5500 to 18500 prompt tokens. Run it once with `KV_OFFLOADING_GIB=0`
and once with the size `preflight.py report` allows, each after a new start, and compare in `totals`:

| Field | What it says |
| --- | --- |
| `later_turns_cached_share` | The share of a later turn's prompt that came from the cache. Near 1 when prefixes survive between a session's turns. |
| `ttft_ms` | Time to the first token, client side. A turn that reads its prefix again pays for it here. |
| `requests_per_s`, `output_tokens_per_s` | The run's throughput over its wall time. |
| `failed` | Requests that did not end well; their `status` in `requests` is a fixed code. |

`totals.by_turn` has the same for each turn of the sessions: its requests, their prompt and cached tokens, the share,
the median first token, and `engine_hit_tokens`, what the engine's two counters gained until the last world ended
that turn. Turn 0 is cold, turn 1 is every session's first return to its prefix, and the turns after it are its
later returns. It is there for one question the cheap card left open on 2026-10-07
([speed.md](speed.md#measured-on-the-cheap-card-2026-10-07)): with a copy on, about one prefix a session was
counted as the copy's and the rest as the GPU's. If a prefix comes from the copy once and then stays on the GPU,
turn 1 holds the `external` tokens and the later turns the `local` ones, with the same first token; if the later
turns' share or first token falls instead, the copy does not hold for a second return. The worlds do not wait for
one another, so a faster world's next turn falls into the slower ones' turn: the counters by turn are approximate,
by a few requests, and the rows of `requests` are exact. A read of the counters that fails leaves a turn without
`engine_hit_tokens`.

The gateway admits 4 reader generations at once and 4 over all classes, and holds 8 more waiting
(`simple_serving/config.py`, `PROVISIONAL`), so up to 12 worlds run without a refusal and at most 4 requests are in
the engine at a time. `--class`, `--sessions`, `--system-tokens`, `--first-tokens`, `--last-tokens`,
`--growth-tokens` and `--answer-tokens` change the shape. All of a world's sessions share one reader scope.

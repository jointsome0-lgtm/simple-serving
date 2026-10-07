# The cheap card: one rental for the copy of the KV cache and for speed

One Blackwell card of 16 GB under $0.25 an hour, with Gemma 4 E2B behind the manifest's `small` profile, for
at most 120 minutes of rental and 100 of work. It is for mechanisms that hold on the 5090 too: whether the copy of
the KV cache in RAM brings a session's prefix back, what a load from it costs, what `/dev/shm` a container gets,
whether the gateway and the engine take eight requests at once, and what a new schema costs. The 31B and Qwen do not
fit in 16 GB, so no speed and no size of theirs is learnt here.

Nothing below has run on a card. Every command of "The steps" that needs no GPU was rehearsed on 2026-10-07 against
the fake engine or in a copy of the checkout ([what was rehearsed](#what-was-rehearsed)). The rental is the owner's
to start, under simple-story-chat's `docs/gpu.md`, "While the cards are paid for". The order and the size of the
steps are as a review of 2026-10-07 left them.

## The switches

Four manifest values, all off by default; with the defaults nothing of Gemma's preparation, arguments or limits
changes (`tests/test_card.py`).

| Value | On the cheap card | What it does |
| --- | --- | --- |
| `MODEL_PROFILE` | `small` | Each `SMALL_<NAME>` stands for `<NAME>`: `google/gemma-4-E2B-it` at `3e22461f`, its tokenizer and its drafter at `2d874ef7`, each file by its hash; alias `gemma-4-e2b-it`; context 16384; text only. |
| `KV_CACHE_MEMORY_MIB` | `768` | `--kv-cache-memory-bytes`: the GPU's KV cache has this size whatever the card has free. vLLM then ignores `GPU_MEMORY_UTILIZATION` (`vllm/config/cache.py:228-235`). |
| `GENERATIONS_AT_ONCE` | `8` | The gateway admits this many generations at once, over all classes and for each of ours, in place of the provisional 4. At most `MAX_NUM_SEQS`. |
| `KV_OFFLOADING_GIB` | `0`, then `4` | The copy in RAM, as before. |

`/v1/state` names each among the versions while it is set.

## Why a session falls out of the GPU's cache

Gemma 4 E2B at the pinned revision (`config.json`): 35 layers, of which the last 20 share the cache of earlier ones
(`num_kv_shared_layers`), so 15 hold a cache: 12 with a sliding window of 512 tokens and 3 global. One KV head each,
of 256 and 512 dimensions. With the fp8 cache a token costs 2 x 256 = 512 B in a sliding layer and 1024 B in a global
one: **12 x 0.5 + 3 x 1 = 9 KiB a token** while every sliding block is kept, which
`--prefix-cache-retention-interval None` does. This is derived from the configuration; the rental of 2026-09-27
measured 3.3 KiB a token without dense retention (4823 MiB, 1526128 tokens), which is the three global layers and a
window's share, and agrees with it.

The worlds run of step 3 is 4 worlds of 5 sessions, 20 sessions in all. A session's prompt is 2500 tokens of system
text and a request that grows from 3000-8600 by 300 a turn and starts over above 10000: 8850 tokens on average over
the run and 12800 at most (the dry run's count, 1415300 prompt tokens in 160 requests). Every session has a system
text of its own, with its own random tag in each line (`worlds.py`, `Session`); the sessions of a world share their
scope and no text. So nothing a later turn finds in the cache is another session's: it is that session's own prefix.

- The cache holds 768 MiB / 9 KiB = **87000 tokens**.
- Four requests run at once, of at most 12800 tokens: 51000 tokens. They fit, 1.7 times over.
- Between two turns of one session the 19 others each take a turn: 19 x 8850 = **168000 tokens**, 1.9 times what the
  cache holds. The GPU drops the least recently used blocks first, and the session's are older than all of those, so
  none of its prefix is left when its turn comes.

If vLLM kept only the global layers' blocks, 3 KiB a token, the cache would hold 262000 tokens and nothing would
fall out; if it gave all 35 layers a cache, 21 KiB a token, it would hold 37000 and four requests at once would not
fit. The run says which, and step 3 has the rule and one bounded correction.

In the copy every prompt block is written as it is computed, 9 KiB a token: the whole run writes 221000 new tokens,
**1.9 GiB**, so a copy of 4 GiB holds all of it and nothing is dropped from it, whatever its order of dropping. By
[speed.md](speed.md) item 5 a hit needs less, the global chunks and one window: 20 x (8850 x 3 KiB + 3 MiB +
300 x 9 KiB) = 0.6 GiB; step 10 tests that with a copy of 1 GiB.

## What the offer must have

`python3 simple_serving/preflight.py plan --offload-gib 4` prints
`{"offload_gib": 4, "min_memory_mib": 16384, "min_ram_gb": 18, "min_shm_mib": 4608}`: 18 GB of RAM in the offer and
4.5 GiB free in `/dev/shm`. The 12 GiB of that for the engine are provisional (step 6 reads the real number). A
copy of 2 GiB needs 16 GB and 2.5 GiB and still holds what the run writes, with nothing to spare. The disk: 10.2 GB of
weights, 0.03 GB of drafter and two venvs; the lane's 60 GB is enough.

simple-story-chat's `small` lane asks for a Blackwell card of 16 GB under $0.25 an hour, 60 GB of disk and 16 GB of
RAM (`local/rent-plan.ts`, `SMALL_RAM_GB`). It has no option for more RAM, so an offer of 16 or 17 GB can be the
cheapest and be taken. The dry run prints each candidate's `ramGb`:

```sh
SIMPLE_CHAT_RENT_DRY_RUN=1 node --env-file-if-exists=.env.gpu gpu/rent.mjs --lane small --hours 2
```

from simple-story-chat's root. It asks Vast for offers and spends nothing. A candidate under 18 GB is passed over
with `--avoid-host ID`, or kept and measured with 2 GiB. What is missing in the renter is an option for the RAM floor
(`--ram-gb 18`), and, if Vast honours it, one for `/dev/shm` (below).

## `/dev/shm` on Vast

What the public documentation says, read on 2026-10-07, and it is not one answer:

- "Shared memory resources (such as `/dev/shm`, controlled by the `shm-size` parameter) are assigned automatically in
  proportion to your GPU fraction via the underlying cgroup resource configuration" (Docker environment page). The
  same page lists the `docker run` options a renter can set: environment variables, the hostname and ports.
- The technical FAQ shows `--shm-size=32gb` among a template's docker options.
- The API reference calls the `env` of an instance's creation "environment variables and port mappings in Docker
  flag format". Nothing is said of privileges or capabilities; Docker in Docker is off.

So whether a `--shm-size` in `env` is honoured is not established, and one rental of minutes would say. The renter
sends `env: { '-p 22:22': '1' }` (`local/rent-plan.ts`, `createBody`); the change would be an option, off by default,
that adds one more entry of the same form, `'--shm-size=24g': '1'`, and a line of its dry run that shows it.
Measured before: 15 GiB of `/dev/shm` on a 5060 Ti host of 251 GB (2026-09-27), where a copy of 16 GiB was refused.

Step 1 tries from inside only when `/dev/shm` is too small for the copy, and once. The remount needs a capability the
container most likely lacks, and then fails with a permission error and changes nothing. When it works it changes
this container's own tmpfs alone, and what is written there stays charged to this container's memory limit, so
neither the host nor another tenant loses anything. A remount that fails says nothing of a `--shm-size` given at
creation: that is Docker's own setting of the mount, and needs no capability inside.

## The steps

Minutes count from the instance's creation. `HOST` is the card's host in `~/.ssh/config`, `CONFIG` the CLI's
configuration (`~/.config/simple-serving/config.json`, with the keys `keys` wrote), `OUT` a new directory for the
reports on the owner's machine. Measurements run on the owner's machine through the tunnel of `up`, 8080 and 8081.
The gateway falls asleep 13 minutes after the last request and stops the instance: no stretch without a request
comes near that, and the longest planned ones are the wait of step 9, five minutes, and a restart, seven.

The steps keep their numbers and run in this order:

| # | Minutes | Step | Deciding number | Carries over to the 5090 |
| --- | --- | --- | --- | --- |
| 0 | 0-10 | Instance up, SSH, the checkout copied | - | - |
| 1 | 10-11 | `preflight.py report`; a remount of `/dev/shm` only if it is too small | `shm_free_mib`, `memory_limit_mib`, `largest_offload_gib` | In part: the same platform and image, another host. The share follows the GPU's fraction of the machine. |
| 2 | 11-23 | Switches, `bootstrap.sh`, first load | Seconds of the preparation and of the load; `kv_cache_tokens` | No: E2B's files and compile. |
| 9 | 23-26 | The sagents run through the gateway | The run's own | Yes: the gateway and the stream are the same. |
| 6 | 26-28 | The schema of sagents' shape; the container's memory | `compile_ms`; `anon` | Schema yes in kind: xgrammar on the CPU, not the model. Memory in part: the runtime's share, not the weights'. |
| 3 | 28-32 | Worlds, copy off | `later_turns_cached_share` at or under 0.1, `engine_preemptions` 0 | Yes, as the reference: the same eviction, by arithmetic. |
| 4 | 32-33 | Reload, copy off, one round | `back_from` = `read_again`; `ttft_ms.cold`, `.gpu`, `.back` | In part: the ratio of a read to a hit; the milliseconds are E2B's. |
| 7 | 33-40 | Copy on at 4 GiB, a new start | Bootstrap's exit 0; seconds of the load | In part: that the size is in the compile key; not the seconds. |
| 8 | 40-45 | Counters; reload with another scope; worlds, copy on | `present.external`; `back_from` = `copy`; `other_scope_cached_tokens` 0; `later_turns_cached_share` and `ttft_ms` against step 3; `ttft_ms.cold` against step 4's | Yes for whether a prefix comes back, from where, and that scopes stay apart. In part for what a load and a write cost: 9 KiB a token here, 440 KiB for the 31B. |
| 10 | 45-54 | Copy of 1 GiB, worlds again, only if the run writes more than 1 GiB | `later_turns_cached_share` against step 8 | Yes: what the copy drops first decides the size the 5090 needs (13 to 23 GiB, or far more). |
| 5 | 54-56 | 4 and 8 requests at once, short | `failed` 0 and `engine_preemptions` 0 at 8 | Yes for the gateway and the engine admitting 8. No for the curve: E2B's does not choose the 31B's level. |
| 11 | 56-61 | Logs off the card, deletion with its read-back | `--destroy` confirms | - |

About 61 minutes of 100. When the card is slow, drop in this order: step 5, step 10, the reload rounds of steps 4
and 8. The worlds with the copy off and with it on, and the deletion, stay whatever happens. At minute 100 step 11
runs whatever is unfinished.

### 0. The card

```sh
git archive kv-offload-ready | ssh HOST 'mkdir -p /workspace/simple-serving && tar -x -C /workspace/simple-serving'
uv run python -m simple_serving.cli trial --ssh-host HOST
mkdir -p OUT
```

from this checkout, on `kv-offload-ready`. The switches of steps 2, 7 and 10 are made in this checkout's
`card/manifest.env` and the file is copied to the card, since the command on the owner's machine reads the model's
alias from its own manifest: `up` waits for the model its manifest names. `git checkout card/manifest.env` after the
rental puts the defaults back.

### 1. What the container has

```sh
ssh HOST 'python3 /workspace/simple-serving/simple_serving/preflight.py report' | tee OUT/preflight-before.json
```

`shm_free_mib` at or above 4608: nothing more here, and `/dev/shm` is left as it is. Never make a larger one
smaller. Under 4608, one attempt and the report again, and no other way tried:

```sh
ssh HOST 'mount -o remount,size=6G /dev/shm; echo "remount exit $?"; df -m /dev/shm | tail -1'
ssh HOST 'python3 /workspace/simple-serving/simple_serving/preflight.py report' | tee OUT/preflight-after.json
```

`largest_offload_gib` of the last report is the largest copy this container holds. Under 4: use 2 in step 7, and
`--worlds 2` in steps 3 and 8 with `KV_CACHE_MEMORY_MIB=384`. Under 2: steps 7, 8 and 10 are off, and the rental
still gives steps 1 to 6 and 9.

### 2. The small model, copy off

```sh
sed -i -E 's/^(MODEL_PROFILE)=.*/\1=small/; s/^(KV_CACHE_MEMORY_MIB)=.*/\1=768/; s/^(GENERATIONS_AT_ONCE)=.*/\1=8/; s/^(KV_OFFLOADING_GIB)=.*/\1=0/' card/manifest.env
scp card/manifest.env HOST:/workspace/simple-serving/card/manifest.env
time (uv run python -m simple_serving.cli keys | ssh HOST bash /workspace/simple-serving/card/bootstrap.sh)
uv run python -m simple_serving.cli up
ssh HOST 'grep -hE "engine_measure|load_ms" /workspace/simple-serving-card/logs/card.jsonl | tail -6' | tee OUT/engine-off.jsonl
```

`up` stays in its own terminal while the pair runs, and says `ready: gemma-4-e2b-it`. On 2026-09-27 the preparation
took 5.5 minutes and the first load 197 seconds. The last line prints what the launcher kept of vLLM's start, numbers
only: `load_ms` of the row `ready`, and `kv_cache_tokens`, vLLM's own count of what the cache holds, to set beside
the 87000 derived above.

### 9. sagents through the gateway

Three minutes, right after the first `ready`. The team lead runs it from the owner's machine through the tunnel of
`up`. The handshake is by files in `OUT`: when the engine is ready and step 2's lines are read, whoever runs these
steps creates the empty file `OUT/sagents-go`, and waits, looking every five seconds, until `OUT/sagents-done` is
there or five minutes have passed; then step 6 goes on either way. Nothing is sent to the gateway during the wait.

What it needs from the card's side: the model `api:gemma-4-e2b-it`; `SAGENTS_API_URL=http://127.0.0.1:8080/v1`, the
tunnel of `up` (the card's own listener is 8090); the client key of `CONFIG` as its key; `SAGENTS_API_STREAM=1`; no
`SAGENTS_API_CACHE_FIELD`. It sends no class and no scope, so its requests are of the key's default class,
`internal`: 2 at once by the contract and 8 here with `GENERATIONS_AT_ONCE=8`. A world asks one thing at a time, so
two at once are enough for two worlds. The context is 16384 tokens, and a request above it gets 400 `context_limit`.

### 6. The schema, and the container's memory

```sh
uv run python -m simple_serving.interference.probes schemas --run --config CONFIG --out OUT/schemas.json
ssh HOST 'cat /sys/fs/cgroup/memory.current; grep -E "^(anon|shmem|file) " /sys/fs/cgroup/memory.stat;
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; df -m /dev/shm | tail -1' | tee OUT/memory-off.txt
```

26 requests of one prompt of 1500 tokens, under a schema of sagents' own shape: ten fields, two nullable strings, an
enum with null, four arrays of objects, 8 labels in each list. A row whose `status` is not `ok` says the engine
refused the shape. `compile_ms` is the median first token under a schema whose text is new less that under one seen
before. It is not pure compile time: it holds whatever else the engine does once for a new grammar, and the noise of
two medians of 12. `anon` is what the engine holds in RAM beside the copy; preflight.py reserves 12 GiB for it without
a measurement.

### 3. Worlds, copy off

```sh
uv run python -m simple_serving.interference.worlds run --config CONFIG --out OUT/worlds-off.json \
    --worlds 4 --sessions 5 --turns 8 --last-tokens 10000
```

160 requests. The sessions share no text, so the rule needs no allowance for a shared prefix: a later turn's cached
tokens are its own session's, and with the copy off `engine_hit_tokens.local` is their sum. In `totals`:

| `later_turns_cached_share` | `engine_preemptions` | What it means | Then |
| --- | --- | --- | --- |
| at or under 0.1 | 0 | The sessions fall out, as derived. A cache that kept them all would give about 0.9. | Step 4. |
| above 0.5 | 0 | The cache holds more than the 168000 tokens between a session's turns: under 4.5 KiB a token. | Correct to `KV_CACHE_MEMORY_MIB=256`. |
| 0.1 to 0.5 | 0 | It holds about those 168000 tokens, and some prefixes last in part. Not a baseline. | The same correction. |
| any | above 0 | Four requests at once do not fit: more than 9 KiB a token. | Correct to `KV_CACHE_MEMORY_MIB=1792`. |

The correction is made once and here, before the copy is switched on, so that the baseline and step 8 have one
cache size: the `sed` with the new value, the `scp`, the bootstrap, the stop and the start of step 7 with
`KV_OFFLOADING_GIB` still 0, then this step again into `OUT/worlds-off-2.json`. A start that changes the cache size
alone took 63 to 83 seconds on 2026-09-27. If the second run does not meet the first row either, steps 7 and 8 run
all the same and the report says that the baseline is not clean.

### 4. Reload, copy off

```sh
uv run python -m simple_serving.interference.probes reload --run --config CONFIG --out OUT/reload-off.json \
    --tokens 8000 --rounds 1 --fillers 20
```

23 requests, one at a time. 20 other sessions of 8000 tokens are 160000 tokens between a session's second turn and
its third, 1.8 times the cache. `back_from` must be `read_again`, and the `back` row's first token near
`ttft_ms.cold`. `ttft_ms.filler` is the median of 20 cold prefills of 8000 tokens with nothing written to a copy:
step 8 sets its own beside it.

### 7. Copy on

Right after step 4.

```sh
sed -i -E 's/^(KV_OFFLOADING_GIB)=.*/\1=4/' card/manifest.env
scp card/manifest.env HOST:/workspace/simple-serving/card/manifest.env
ssh HOST bash /workspace/simple-serving/card/bootstrap.sh < /dev/null &&
  ssh HOST 'cd /workspace/simple-serving &&
    /workspace/simple-serving-card/gateway/bin/python -m simple_serving.card --stop &&
    /workspace/simple-serving-card/gateway/bin/python -m simple_serving.card'
ssh HOST 'grep -c "\"event\": *\"ready\"" /workspace/simple-serving-card/logs/card.jsonl'
ssh HOST 'grep -hE "engine_measure|load_ms" /workspace/simple-serving-card/logs/card.jsonl | tail -6' | tee OUT/engine-on.jsonl
```

The pair is stopped only when the bootstrap ends with 0. It ends with 9 and changes nothing when the container
cannot hold the size, and the pair it left serves on; then the next smaller size, once.

`up` is not started again: it survives `--stop` and the new start, since the instance keeps running and `up` opens
its tunnel again by itself. A second `up` is refused with "another up runs", and on 2026-10-07 waiting on one cost
5.5 idle minutes. `up` prints `ready` once and not again for the new pair, so the wait is for the card's own row: the
count of `ready` rows, read before the stop, is one more once the new pair serves, and the row's `load_ms` is the
load. Ask every 15 seconds or so. A start with a new size of the copy took 172 to 174 seconds on 2026-10-07, three
times, against 271 for the first load, and 228 on 2026-09-27.

An `ssh` line that ends with 255 and no output did not run, twice on 2026-10-07. Check the card's log before
repeating the line of the stop and the start: a second `--stop` would end the pair that the first one started.

### 8. Counters, reload and worlds, copy on

```sh
uv run python -m simple_serving.interference.probes counters --run --config CONFIG --out OUT/counters-on.json
uv run python -m simple_serving.interference.probes reload --run --config CONFIG --out OUT/reload-on.json \
    --tokens 8000 --rounds 1 --fillers 20 --other-scope
```

`counters` sends no generation and says in `present` which of the engine's counters its metrics hold. The reload is
24 requests. Read both before the long run:

- `present.external` true and the `back` row's `copy_hit_tokens` near its `prompt_tokens`: the counter tells a load
  from the copy, and `back_from` is `copy`. Go on.
- The counter absent or 0, and the `back` row's `cached_tokens` near its `prompt_tokens` with `gpu_hit_tokens` 0
  (`back_from` is then `mixed`): the request's usage counts tokens from the copy. Go on, and read
  `later_turns_cached_share` below in place of `engine_hit_tokens.external`.
- Neither, and every `status` is `ok`: only the first tokens are left to tell a load from a read. Read the `back`
  row's against `.cold` and `.gpu`; if it is as `.cold`, the copy brought nothing back and the worlds below will
  say the same.
- A `status` that is not `ok`: mend that first.

`other_scope_cached_tokens` must be 0: the `other` row asks the session's text from a new scope while its prefix is
in the copy alone, and the gateway's cache salt must keep it from both stores. `ttft_ms.filler` against step 4's is
what writing the copy costs a cold prefill of 8000 tokens. The `back` row's first token between `.gpu` and `.cold` is
what a load from RAM costs for 8000 tokens, 72 MiB here.

```sh
uv run python -m simple_serving.interference.worlds run --config CONFIG --out OUT/worlds-on.json \
    --worlds 4 --sessions 5 --turns 8 --last-tokens 10000
ssh HOST 'cat /sys/fs/cgroup/memory.current; grep -E "^(anon|shmem|file) " /sys/fs/cgroup/memory.stat;
  df -m /dev/shm | tail -1' | tee OUT/memory-on.txt
```

The copy works when `later_turns_cached_share` is far above step 3's; it pays when `ttft_ms` and `requests_per_s`
are better than step 3's.

### 10. A copy smaller than what is written

Only if the run of step 8 wrote more than 1 GiB: `prompt_tokens` less `cached_tokens` in its `totals` above 116000,
at 9 KiB a token (above 50000 if step 3 corrected to 1792, at 21 KiB). Otherwise a copy of 1 GiB is under no
pressure and the step says nothing.

```sh
sed -i -E 's/^(KV_OFFLOADING_GIB)=.*/\1=1/' card/manifest.env
```

then the copy of the file, the bootstrap, the stop, the start and the wait for `ready` of step 7, and the worlds of
step 8 into `OUT/worlds-on-1gib.json`. The run writes 1.9 GiB, and a hit needs 0.6 GiB of it. A
`later_turns_cached_share` near step 8's would say the copy drops what no hit needs first; one near step 3's says it
needs room for all that is written. On 2026-10-07 it was the second: 0.0 at 1 GiB, and 0.95 at 2 GiB as at 4 GiB
([speed.md](speed.md#measured-on-the-cheap-card-2026-10-07)).

### 5. Eight at once

```sh
for n in 4 8; do
  uv run python -m simple_serving.interference.worlds run --config CONFIG --out OUT/level-$n.json \
      --worlds $n --sessions 1 --turns 6 --system-tokens 1500 --first-tokens 1500 --last-tokens 4000
done
```

24 and 48 requests, under whatever copy the last start has. Each world is one session whose prefix stays in the
cache (8 x 4700 tokens fit). What is wanted is `failed` 0 and `engine_preemptions` 0 at 8: the gateway and the engine
admit eight generations under the strict schema. The tokens a second of the two levels are E2B's and choose nothing
for the 31B.

### 11. The end

```sh
echo '{}' | ssh HOST 'cd /workspace/simple-serving &&
  /workspace/simple-serving-card/gateway/bin/python -m simple_serving.card --inspect' > OUT/inspect.json
git checkout card/manifest.env
```

and, in simple-story-chat, `npm run gpu:rent -- --destroy ID` with its read-back. The reports of the steps are
already on the owner's machine. They hold counts and times, and no prompt, key or answer.

## What was rehearsed

On 2026-10-07, with no card: the worlds of steps 3, 5 and 8 and the probes of steps 4, 6 and 8 with the commands' own
shapes, against this repository's gateway in front of the fake engine, whose cache, copy and compile time are
stand-ins ([worlds.md](worlds.md)); the `sed` lines of steps 2 and 7 in a copy of the checkout, with the launcher's
`--dry-run` printing vLLM's arguments, and `--inspect` on an empty state; `preflight.py report`, `check` and `plan`
on this machine; the bootstrap with the small profile against stand-ins for pip, aria2c and the hub
(`tests/test_card.py`).

Not rehearsed, since each needs the card: the remount of step 1; the load of E2B with an explicit cache size and with
the copy; every time and every count of a real engine, the copy's counter among them; the two reads of the cgroup's
memory files, whose paths are cgroup v2's; the `scp`, the `ssh` lines and `up`, which need a host; sagents against a
real engine.

# Speed levers for the text card

The owner asked on 2026-09-26 to carry over to this card what simple-story-chat's picture card had just found: its
int8 layers ran on a slow path of their kernel library until a switch put them on Triton's kernels, 2.3 times faster
once warm (simple-story-chat's `docs/knowledge/gpu-measurements.md`, the pilot of 2026-09-26). The proposed experiments
below have not been measured on the text card. "Already in place" lists its configuration and dated results.
Each proposed experiment is a question for a rental, with the evidence that would answer it. The
[picture card](pictures.md#9-the-card) keeps its existing ComfyUI engine and pins during phase 1.

## Already in place

- vLLM runs without `--enforce-eager` (`simple_serving/card.py`), so torch.compile and CUDA graphs are on, and torch's
  compiler writes Triton kernels for the fused operations. FlashInfer and Triton compile kernels on the card at its
  start (the contract, section 15). There is no switch here like the picture card's.
- The KV cache is fp8 (`KV_CACHE_DTYPE`); a bfloat16 cache does not fit at 65536 tokens.
- Gemma 4's multi-token prediction drafts 3 tokens a step, which took each of two requests at once from 57 to 67
  tokens a second to 123 to 171 ([runbook, "The card"](runbook.md#the-card)).
- The text launcher uses dense prefix-cache checkpoints. The dated comparison below motivated this setting.

## Dense prefix retention

The `live-2026-09-28` change from `940471a` is integrated locally on 2026-09-30. The text launcher now passes
`--prefix-cache-retention-interval None` to the pinned vLLM 0.30.0. It keeps sliding-window checkpoints densely
instead of retaining only selected checkpoints. Full-attention groups ignore this setting. Cached blocks still
compete within the fixed GPU pool and remain evictable; this does not allocate unlimited cache or enable RAM
offloading. Scope salts remain part of the prefix hashes.

The original RTX 5090 comparison on 2026-09-27, recorded in `held-criterion` at `a898060`, found that a story's
second turn read 830 of its 16382 prompt tokens and reached its first token in 372 ms, against 6406 ms in the
reference. The identity check kept the same tokens in 16 of 16 requests. Four interleaved stories got no cache
hits and no first-token improvement. Those results do not guarantee the same benefit with today's combined
gateway or a different workload.

Implementation references in the pinned source are the [CLI parser](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/engine/arg_utils.py),
[cache configuration](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/cache.py),
[block pool](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/block_pool.py)
and [prefix hashing](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/kv_cache_utils.py).
The local launcher checks confirm the command argument, without a new GPU measurement. A controlled repeat
should compare cached prompt tokens, reader TTFT and interference pauses against the previous setting on one card.

## To look at, cheapest first

1. **The FP4 kernels.** The launcher's [engine-choice rows](runbook.md#engine-choices) now record recognised
   selection messages. Identify which kernel runs the NVFP4 weights on an RTX 5090 (sm_120), then measure any
   supported alternative. Seeing Marlin or its warning alone does not establish a slow fallback. In pinned vLLM
   0.30.0, Marlin is a normal W4A16 selection; the [kernel selector](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/__init__.py)
   and [Marlin implementation](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/kernels/linear/nvfp4/marlin.py)
   explain that path. The original fallback hypothesis was revised by the source review in `9b73efb`.
2. **The attention backend** vLLM picks for Gemma 4's sliding-window and global layers, from its log, against the
   others it offers on this GPU (FlashInfer, FlashAttention, its own Triton backend), each tried once.
3. **The CUDA graphs** of a decode step with the drafter: the mode and the capture sizes vLLM 0.30.0 uses by default,
   from its log, against full graphs if they are not the default.
4. **The drafter's count.** 3 was measured against none, and 2 and 4 were not. Its acceptance fell from 78 to 62 and
   49 per cent over the three positions.
5. **A copy of the KV cache in RAM** (`KV_OFFLOADING_GIB`, off). The owner asked for it on 2026-09-27, so that a
   reader who reads on the card while it is measured keeps their story warm. The cache on the GPU held 104492
   tokens, and four long stories at once do not fit in it. This item was corrected on 2026-10-07 from vLLM 0.30.0's
   source and from the two measurements of 2026-09-27 that `held-criterion` records at `a898060`; paths below are in
   the `v0.30.0` tree, and `offloading/` is `vllm/distributed/kv_transfer/kv_connector/v1/offloading/`.
   - *Where it lives.* One file in `/dev/shm`, at a fixed path, mapped shared
     (`vllm/v1/kv_offload/cpu/shared_offload_region.py:111,121-151`). vLLM refuses a size above that mount's free
     space at its start (`:138`, `vllm/distributed/device_communicators/shm_broadcast.py:227-247`) and faults every
     page in (`shared_offload_region.py:211-235`), so the whole size is resident from the first minute and counts
     against the container's memory. Pinning it with `cudaHostRegister` is tried, and a failure is a warning
     (`vllm/v1/kv_offload/cpu/gpu_worker.py:193-220`).
   - *`/dev/shm` decides, and the offer does not show it.* Its size was 15 GiB on a host of 251 GB, where 16 GiB was
     refused with "Insufficient space in /dev/shm", and 30 GiB on a host of 63 GB. It is set when the container is
     created (Docker's `--shm-size` or `--ipc=host`, which vLLM's own message names); no option or variable of vLLM
     0.30.0 moves the file elsewhere, and a remount from inside needs a privilege that no rental has been seen to
     give. So the size is read once the container is up, and the buffer follows it:
     `python3 simple_serving/preflight.py report` prints `/dev/shm`'s free space, the memory limit and the largest
     `KV_OFFLOADING_GIB` that fits.
   - *A size that does not fit is refused before it costs anything.* `card/bootstrap.sh` ends with 9 before it
     installs or fetches, and the launcher gives up as `unfit` and stops the instance before it loads
     (`simple_serving/preflight.py`). The rule: `/dev/shm` holds the size and 512 MiB more, and the memory limit
     holds the size and 12 GiB for the engine. The 12 GiB are provisional, since no rental recorded the engine's
     resident memory. vLLM also refuses any copy while `PYTORCH_CUDA_ALLOC_CONF` holds `expandable_segments:True`
     (`vllm/config/vllm.py:1090-1109`); the check refuses that first.
   - *What is written.* Prompt blocks, as they are computed and not when the GPU drops them
     (`offloading/scheduler.py:1464-1740`), one chunk for a GPU block of a cache group, 1.25 MiB for Gemma. The
     global layers cost 40 KiB a token and the sliding layers 400 KiB. With `--prefix-cache-retention-interval None`,
     which the launcher sets, every sliding block is stored
     (`vllm/v1/core/single_type_kv_cache_manager.py:1108-1117`): **440 KiB a token written**, so a prompt of 16000
     tokens writes 6.7 GiB and one of 40000 writes 16.8 GiB. With retention 0 only the windows at the prompt's end
     and at a shared prefix's end are stored, about 412 MiB each, which is what this item said before.
   - *What a hit needs.* Every global chunk of the prefix and one window of sliding chunks at its end
     (`offloading/scheduler.py:642-718,841-866`): N x 40 KiB + about 412 MiB, 1.0 GiB at 16000 tokens. A lookup
     touches only that (`:720-737`, `vllm/v1/kv_offload/cpu/policies/lru.py:43-78`). This item sized the copy by it
     until 2026-10-07, as T x (N x 40 KiB + 412 MiB + D x 440 KiB) for T sessions that grow D tokens a turn, and the
     cheap card refuted that: where the same formula asked 0.6 GiB for Gemma 4 E2B, a copy of 1 GiB gave no hit at
     all.
   - *Sizing.* **Sessions x prompt tokens x the bytes a token writes**: the copy has to hold what all the sessions
     write before the first of them returns. When every session starts cold, each writes its whole prompt, and in a
     smaller copy a session's blocks are gone before its second turn; that turn is then cold again and writes in
     full, and so is every turn after it. `python3 simple_serving/preflight.py plan --sessions S --prompt-tokens T`
     prints the size and what an offer must have for it; `--kib-per-token` is 440 unless given.

     | Sessions x prompt tokens | Bytes a token | Copy | What it is |
     | --- | --- | --- | --- |
     | 20 x 8850 (E2B, first turns 8300) | 9 KiB | 1.5 GiB by the rule | **Measured, E2B, 2026-10-07:** 1 GiB gave 0 of 140 later turns a cached token; 2 GiB and 4 GiB gave 136 of 140 their whole prefix. |
     | 2 x 16000 (31B) | 440 KiB | 14 GiB | Arithmetic. The most that fits the 15 GiB of `/dev/shm` seen on a 5090 host. |
     | 4 x 16000 (31B) | 440 KiB | 27 GiB | Arithmetic. |
     | 20 x 8850 (31B) | 440 KiB | 75 GiB | Arithmetic. |

     The rule is from one model and one workload, three sizes, one run each; where between 1 and 2 GiB the E2B copy
     begins to work was not located. E2B's 9 KiB and the 31B's 440 KiB a token are both derived from the models'
     layers and neither is measured. Sessions that start one after another, not all at once, may get by with less:
     not tried. A size at or below the GPU's own cache, 6 to 9 GiB, adds nothing (vLLM's
     `docs/features/kv_offloading_usage.md:326`).
   - *Measured so far.* On the 5090 with the 31B and its drafter, 12 GiB started, and four interleaved stories read 0
     of 294715 prompt tokens from it, under retention 0. On the cheap card with Gemma 4 E2B, 8 GiB read 146304 of
     295905 under retention 0 (2026-09-27), and under dense retention a copy of 2 GiB or 4 GiB brought every
     session's prefix back (2026-10-07, [below](#measured-on-the-cheap-card-2026-10-07)). With the 31B the copy
     under dense retention has never run.
   - *How it is measured.* The [worlds measurement](worlds.md), with the copy off and on, on one card: cached
     prompt tokens in later turns and each turn's first token.

## Measured on the cheap card (2026-10-07)

One rental, an RTX 5060 Ti of 16311 MiB with Gemma 4 E2B (`MODEL_PROFILE=small`), by
[cheap-card.md](cheap-card.md); one run of each step, with synthetic text. Every time was taken on the owner's
machine through the SSH tunnel, whose round trip is in each of them: a repeated small request took 310 ms in all.
The numbers say how the engine and the gateway behave; none of them is a speed of the 31B.

| What | Measured |
| --- | --- |
| The container | `/dev/shm` 11264 MiB free, memory limit 32630 MiB, largest copy that fits 10 GiB. No remount was tried. |
| Preparation and loads | Bootstrap 323 s. First load 271 s; a start with another size of the copy 172 to 174 s, three times. |
| The GPU cache | `KV_CACHE_MEMORY_MIB=768` gave `kv_cache_tokens` 199136, 3.95 KiB a token by vLLM's count. A world of 160000 tokens did not stay in it, so that count is not what the cache holds under dense retention. |
| The first request after a start | 6283 ms and, after another start, 8711 ms, where the next took 310 ms. `WARM_UP` is meant for this and is not measured. |
| A new strict schema | 26 of 26 answered; first token 320.1 ms with a schema never seen against 309.7 ms repeated: about 10 ms to compile. |
| The engine's memory, copy off | 3.68 GB anonymous in the container, 10702 MiB on the GPU. With a copy of 4 GiB: 4.01 GB anonymous and 4.31 GB shared. |
| Reload, copy off (8000 tokens, 20 other sessions between) | First token cold 1360 ms, from the GPU cache 541 ms (8096 of 8473 tokens), after the others 1438 ms: read again. |
| Reload, copy 4 GiB | After the others 580 ms, 7968 of 8303 tokens from the copy; from the GPU cache 651 ms. A load from RAM cost what a GPU hit costs. A cold prefill took 1331 ms with the copy and without: writing it cost nothing seen. The same text from another scope got 0 cached tokens. |
| The counters | `vllm:prefix_cache_hits_total`, `vllm:external_prefix_cache_hits_total` and `vllm:num_preemptions_total` are all in the engine's metrics, and a request's `usage.cached_tokens` counts tokens from the copy. |
| Worlds, 4 x 5 sessions x 8 turns, copy off | 160 requests, none failed; later turns' cached share 0.0; 196 s, first token p50 2525 ms, 0.82 requests a second. |
| The same, copy 4 GiB | Share 0.9525, 136 of 140 later turns with their whole prefix; 79 s, p50 638 ms, 2.03 requests a second. Hit tokens: 1018720 counted as the GPU's, 141344 as the copy's. |
| The same, copy 2 GiB | Share 0.9528; 86 s, p50 593 ms. Hit tokens 1015744 and 140640. |
| The same, copy 1 GiB | Share 0.0; 204 s; hit tokens 0 and 0. |
| At once, one session a world | 4: 24 requests, none failed, 307 tokens a second. 8: 48 requests, none failed, 0 preemptions, 389 tokens a second, first token p50 470 ms. |

Not explained: with the copy on, the engine counted about 141000 hit tokens as the copy's, near one whole prefix
for each of the 20 sessions, and a million as the GPU's, though with the copy off the same run had no GPU hit at all.
One reading, not checked, is that a session's first return comes from the copy and the prefix then stays in the GPU
cache for its later turns. The worlds report now sets each session's first return apart from its later ones
([worlds.md](worlds.md)), and the next card settles it.

## Before the 5090 with the 31B

What has to be decided before a long run on a 5090 with the 31B, what is known for each, and which step measures
it. The steps are [cheap-card.md](cheap-card.md)'s, run there with `MODEL_PROFILE=gemma` and without
`KV_CACHE_MEMORY_MIB`; nothing here has run with the 31B, and every size in this section is arithmetic from 440 KiB
a token, which is itself derived and not measured.

| To decide | What is known | What measures it |
| --- | --- | --- |
| The copy's size against `/dev/shm` | The copy holds sessions x prompt tokens x 440 KiB. `/dev/shm` was 15 GiB on the one 5090 host seen, where 16 GiB was refused. `plan --sessions 2 --prompt-tokens 16000` and `--sessions 4 --prompt-tokens 8000` both print 14 GiB, 14848 MiB of `/dev/shm` and 28 GB of RAM: the most that fits. Four stories of 16000 tokens need 27 GiB, 28160 MiB and 42 GB, and do not fit without a larger `/dev/shm`. So without one, the copy keeps two long stories warm, not four. | Step 1, `preflight.py report`, in the first minute: `shm_free_mib`, `memory_limit_mib`, `largest_offload_gib`. `plan` with the run's own sessions and tokens before the rental. |
| `--shm-size` at creation | Not established: Vast's pages disagree ([cheap-card.md](cheap-card.md#devshm-on-vast)), and the cheap card got 11264 MiB without asking and tried no remount. simple-story-chat's renter would need one more `env` entry in `createBody` (`local/rent-plan.ts`), beside `'-p 22:22': '1'`: `'--shm-size=32g': '1'`, behind an option that is off by default. That change is simple-story-chat's and is not made here. | Step 1 on a container created with the entry: `shm_free_mib` near 32768 says Vast honours it. A few minutes of any card answer it, the cheapest included. |
| Retention 0 with a copy | Under retention 0 a prompt of 16000 tokens writes 1.0 to 1.4 GiB in place of 6.7 (its global layers and one or two windows), so 14 GiB would hold some ten stories. But the launcher always passes `--prefix-cache-retention-interval None` and has no switch for it; dense retention is what made a story's second turn 372 ms against 6406 on the GPU alone; and the one run of the 31B with a copy, 12 GiB under retention 0, read 0 of 294715 tokens from it, for a reason never found. A switch would be an additive manifest value, not written: it is the owner's choice whether that run is wanted. | Steps 3 and 8, the worlds with the copy off and on, once under each retention, on one card: `later_turns_cached_share`, `ttft_ms` and `by_turn`. |
| `ENGINE_RESERVE_MIB` | 12 GiB, provisional (`simple_serving/preflight.py`). The engine's anonymous memory with E2B was 3.68 GB without a copy and 4.01 GB with one; the 31B's is not known, and the weights' loading may peak above the resting number. Too small a reserve lets a copy through that the container then cannot hold; too large refuses one that fits. | Step 6's read of the cgroup's `memory.current` and `anon`, with the copy off, and step 8's with it on; and `memory.peak` beside them, which the steps do not read yet. |
| 8 at once | The gateway and the engine took 8 generations under the strict schema with E2B: 48 requests, none failed, no preemption. With the 31B a running request holds its global layers and one window, 40 KiB a token and about 412 MiB: 1.0 GiB at 16000 tokens, so 8 of them ask 8.1 GiB of a GPU cache that was 6011 MiB with the drafter, and 4 ask 4.1. By this arithmetic 8 long prompts at once do not fit and the engine puts requests back; 8 of 8000 tokens ask 5.7 GiB of its 5.9 and barely fit. `GENERATIONS_AT_ONCE` stays 0, the provisional 4, until it is measured. | Step 5 with the stories' own sizes (`--system-tokens`, `--first-tokens`, `--last-tokens`) at 4 and 8: `failed`, `engine_preemptions` and `output_tokens_per_s`. The [sweep](sweep.md) has the finer levels. |
| The warm-up | `WARM_UP=1` is the default and has never met an engine. The 31B's first request after a start is not known either. | The launcher's `warm_up` row (`code`, `total_ms`) and the first row of step 6's schema probe, whose first token should be near `ttft_ms.repeated`. With `WARM_UP=0` on another start, the same number is what the warm-up saves. |
| The hit split | With a copy, one prefix a session counted as the copy's and the rest as the GPU's, unexplained ([above](#measured-on-the-cheap-card-2026-10-07)). It decides nothing by itself, but a copy that restores a prefix only once would. | Step 8's worlds, `totals.by_turn`. |

The order that spends least: `plan` before the rental; step 1 in the first minute, since `/dev/shm` decides whether
a copy is tried at all; then the load with `KV_OFFLOADING_GIB=0` and steps 6, 3 and 5; then one start with the
copy, at 172 to 271 s a start on the cheap card and unknown for the 31B, and step 8.

## Concurrency measurement tool

The [standalone sweep](sweep.md) from `b6d4919` is now ported into the current checkout. It plans 24 levels and can
benchmark an already running engine with synthetic random inputs. The operator supplies the current KV capacity;
there is no default from the older 104492-token measurement. Local fake checks establish tool behavior, while GPU
throughput and the time estimates remain unverified. It does not hold the gateway awake or restart serving.

The earlier KV/offload estimates above depend on the cache retention and startup configuration used then. They do
not establish today's capacity with dense retention, vision or different MTP settings. Use the current launcher's
recorded capacity and measure each configuration separately.

## How to measure

As the picture card's pilot did: on one card, a reference run first, then one change at a time, each with the log line
that shows it took effect. Speed is each request's tokens a second at the bot's two calls at once, the prefill and the
decoding apart. Identity is the same requests at temperature 0 before and after: a kernel may change tokens, and a
changed text goes to simple-story-chat's eval before anything is kept. What is kept is the owner's decision.

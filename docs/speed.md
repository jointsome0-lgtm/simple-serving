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
   reader who reads on the card while it is measured keeps their story warm. The cache on the GPU holds 104492
   tokens, and four long stories at once do not fit in it: a story whose prefix was dropped reads all of it again,
   30000 to 44000 tokens, 10 to 15 s of the card at the 3000 tokens a second simple-story-chat's throughput study
   assumed. vLLM's native backend keeps the dropped blocks in a pinned buffer and loads them back over PCIe. By vLLM
   0.30.0's source it stores what a later hit can use: every block of the global layers, 40 KiB a token, and of the
   sliding layers only the windows at the prompt's end and at the shared prefix's end, about 412 MiB each. A story of
   40000 tokens would then take about 2.5 GiB, and 12 GiB would keep four. That is an estimate, not a measurement.
   Before turning it on, read the container's memory (`/sys/fs/cgroup/memory.max`, or `free -g` where that is `max`)
   and the engine's resident memory in the reference run, and leave the engine that and a few GiB more. A buffer that
   does not fit fails the engine at its start, and the launcher then stops the instance. It is measured on four long
   stories interleaved, with and without it: the prompt tokens read again (`input_tokens` against `cached_tokens` in
   the gateway's log) and each turn's first token.

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

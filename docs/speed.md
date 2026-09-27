# Speed levers for the card

The owner asked on 2026-09-26 to carry over to this card what simple-story-chat's picture card had just found: its
int8 layers ran on a slow path of their kernel library until a switch put them on Triton's kernels, 2.3 times faster
once warm (simple-story-chat's `docs/knowledge/gpu-measurements.md`, the pilot of 2026-09-26). On 2026-09-27 the owner
asked for more text per card-hour with the quality almost unchanged, worked out away from the card first. This page is
that plan: what runs today, the changes in the order to try them, and the runbook. A rental follows only with the
owner's yes on its price and its deadline.

**Sources.** "Measured" is the fourth text card's, an RTX 5090 on 2026-09-25/26 (simple-story-chat's
`docs/knowledge/gpu-measurements.md#text-card-4-2026-09-26`). The rest was read on 2026-09-27 from vLLM 0.30.0 (tag
v0.30.0, commit ced6857; a bare path is in that tree) and FlashInfer 0.6.18.post1, in four readings, one per lever:
the FP4 kernels, attention, graphs and the scheduler, the drafter under load. GPT-6 Astra then checked them against the
same sources. "Derived" applies vLLM's formulas to measured inputs, "estimate" is a model of ours, and "model" is the
drafter reading's model of a step's time, calibrated on the fourth card's two requests at once. None of them is
measured.

## What runs today

- **The FP4 layers run on Marlin, by design.** The checkpoint is weight-only (`quant_algo` W4A16_NVFP4), and for
  W4A16 vLLM picks Marlin on every GPU but SM100 and SM103 (`vllm/model_executor/kernels/linear/__init__.py:1114-1125`).
  Marlin prints that the GPU "does not have native support for FP4" on any GPU (`.../linear/nvfp4/marlin.py:34-39`).
  So this is not the picture card's slow path, and no switch turns it into a fast one; this page read that warning as
  a fallback until 2026-09-27. Native FP4 would need a W4A4 checkpoint, and no official source has one of this
  fine-tune.
- **Every attention layer runs on Triton.** The heads are 256 wide on the 50 sliding layers and 512 on the 10 full
  ones, and FlashAttention 4 does not exist for sm_120, so vLLM forces TRITON_ATTN
  (`vllm/model_executor/models/config.py:250-255`). FlashInfer is the only other backend valid here with an fp8 cache.
- **The graphs are the best default already.** Model Runner V2 runs FULL_AND_PIECEWISE (`vllm/config/vllm.py:318,440`):
  a decoding step with the drafter replays a full graph for 1, 2, 4, 6 or 8 requests, a mixed step of up to 64 tokens
  a piecewise one, and a step with a longer prompt chunk none (`vllm/config/vllm.py:2216-2224`, derived). No graph
  setting is in the queue.
- **The cache is fp8**: 6011 MiB and 104492 tokens with the drafter, 126003 without. A bfloat16 cache does not hold
  one request of 65536 tokens (measured).
- **Three drafted tokens** took each of two requests at once from 57-67 tokens a second to 123-171, with 78, 62 and 49
  per cent accepted at the three positions (measured, scenes included; the compactions' JSON is the easiest to draft).
- **Nothing was measured above two requests at once in the engine.** The gateway admits two `internal` generations
  (`simple_serving/config.py:38`) and simple-story-chat sends every eval call as `internal`, so the fourth card's three
  scenarios at once were two in the engine.
- **At the bot's lengths the cache limits the batch, not `MAX_NUM_SEQS`** (derived from
  `vllm/v1/kv_cache_interface.py:816-842` and the pool of 4804 blocks the measured 104492 tokens imply):

  | Prompt tokens | 44000 | 30000 | 20000 | 10000 | 3000 or fewer |
  | --- | --- | --- | --- | --- | --- |
  | Requests at once | 2 | 3 | 4 | 6 | 8 |

- **card.jsonl shows none of this.** The launcher keeps four numbers of vLLM's output and a few categories of failure
  (`simple_serving/card.py:78-104`, `:420-434`); the kernel, backend, runner and graph lines never leave the card.

## The metric

Output tokens a second summed over every request in flight, over a workload's elapsed seconds, with the prompt tokens
read counted apart; and the same per billed card-hour, the card's start and its wait for the eval's judge included. The
workload has the bot's and the eval's shapes: story turns, long prompts that share a prefix; memory compactions, JSON
under the bot's schema at temperature 0.2; scenes of a few hundred tokens at 0.8. Prompts that share nothing, and one
story read turn by turn with gaps, are workloads of their own, since a shared prefix hides the limits of admission. One
reader's latency is secondary.

## Quality

- **Identical**: the same tokens.
- **Rounding**: kernel rounding only, so at temperature 0 most requests give the same tokens. The check is the identity
  set: 16 of the workload's requests at temperature 0, one at a time, on both sides. For each, the harness prints
  whether the tokens are the same and where they first differ, never a token, and the reference runs the set twice for
  its own floor. If fewer than half of the 16 are the same, the change counts as visible.
- **Visible**: anything more. It goes to simple-story-chat's eval (`docs/eval.md`, at least 3 runs per side) before it
  is kept. Route A's pass of `hospital`, `assault` and `dance` took 5 to 6 minutes on the fourth card, the judge
  included, so both sides on one card take about 35 to 40 minutes (estimate; Astra estimated 60 to 120).

What is kept is the owner's decision.

## The queue

In order. "When" sends an item to the card; an item without it always runs. Minutes are the 5090's, estimates. Each
item is a flag of the pinned vLLM 0.30.0 or a gateway or bot setting and brings nothing new onto the card, unless it
names a pin.

1. **The harness and the log lines that are missing.** No card minutes.
   - *Setting.* card.py's filter gains fixed codes, logged as `engine_choice` rows of a code and numbers only: the FP4
     kernel (`Using %s for NVFP4 GEMM`, `.../kernels/linear/__init__.py:1136,1180`) and a backend override (`:245-248`),
     the forced and the chosen attention backend (`.../models/config.py:250-255`; `vllm/platforms/cuda.py:478`),
     FlashInfer's decoding path (`vllm/v1/attention/backends/flashinfer.py:840,910`), the runner (`Using V2 Model
     Runner`, `vllm/v1/worker/gpu_worker.py:440-441`), the graphs' capture time and memory
     (`vllm/v1/worker/gpu/model_runner.py:1067`), the RAM copy (`vllm/v1/kv_offload/factory.py:57`), and from
     `non-default args` (`vllm/entrypoints/serve/utils/api_utils.py:284-286`) the keys a pass changed, as numbers or
     fixed words.
     The harness reads the engine's `/metrics` before and after every pass: prompt, generation and cached tokens,
     `vllm:prompt_tokens_by_source`, prefix-cache queries and hits, preemptions, requests running and waiting by
     reason, cache usage, and the drafter's counts by position (`vllm/v1/metrics/loggers.py:499-712`,
     `vllm/v1/spec_decode/metrics.py:229-255`), which are cumulative where the periodic log line resets. Then the A/B
     launcher, the workloads, the identity set and the microbenchmarks of the runbook.
   - *Took effect.* On a dry run against the fake engine (`simple_serving/fake_engine.py`), vLLM's format strings filled
     with synthetic values give one row per code and no value; on the cheap card the real lines give the same rows.
   - *Commands* (`simple_serving/throughput/`). `python -m simple_serving.throughput dry-run` runs all of it here in
     about 20 s, against the fake engine behind a stand-in `vllm`, with a reader's requests beside the reference, and
     ends `dry run: ok: ...`. On the card, from `/workspace/simple-serving` with the gateway venv's Python: `run
     --change NAME[=VALUE]` with `--pack`, `--workloads`, `--at`, `--identity reference|change` or `--sweep` writes one
     `result.json` of counts, times and fixed words and prints its report, `report RESULT` prints it again, and
     `microbench gemm|attention` runs the runbook's steps 2 and 3 (`--dry-run` prints the plan without a GPU).
   - *Quality.* Identical: the engine does not change.
   - *Gain.* None by itself; every item below is judged by it.

2. **A copy of the KV cache in RAM** (`KV_OFFLOADING_GIB`, simple-serving f3031ea), the owner's addition of
   2026-09-27, as a lever for more readers later.
   - *Setting.* `--kv-offloading-size`, vLLM's native backend. 12 GiB is the estimate for four stories of 40000 tokens
     on a machine of 32 GB; the reference's engine memory and the container's `memory.max` (or `free -g`) size it, since
     a buffer that does not fit fails the engine at its start, and `card` then stops the instance (the A/B launcher of
     item 1 does not).
   - *Took effect.* `Creating offloading spec with name: CPUOffloadingSpec` (`vllm/v1/kv_offload/factory.py:57`),
     `kv_offloading_gib` in `/v1/state`, and `vllm:prompt_tokens_by_source` with `source="external_kv_transfer"` above
     0 in the pass (`vllm/v1/metrics/stats.py:383-398`).
   - *Quality.* Rounding: a hit loads the fp8 bytes the GPU held, a miss computes them again as today.
   - *Gain.* A story whose prefix was dropped reads 30000 to 44000 tokens again, 10 to 15 s of the card at 3000 tokens
     a second (estimate); how often that happens at a larger batch is not known. Four long stories interleaved, with
     reading gaps, with and without: the prompt tokens read again (the gateway's `input_tokens` against
     `cached_tokens`) and each turn's first token. The passes after it run without the copy.
   - *Minutes.* 7-10.

3. **Requests at once, with the drafter's count.** One step, since a larger batch may move the best count.
   - *Setting.* The harness at 1, 2, 3, 4, 6 and 8 requests at once, `MAX_NUM_SEQS` staying 8, with
     `MTP_SPECULATIVE_TOKENS` 0 (no drafter), 2, 3 and 4, and 5 when 4 beats 3. For the bot and the eval afterwards:
     the gateway's `limits.internal.active` and `limits.shared.active` (`simple_serving/config.py:36-43`), the bot's
     `SIMPLE_CHAT_SERVING_LANES` (1 to 4, default 2), and more eval scenarios at once.
   - *Took effect.* `vllm:num_requests_running` reaches the batch in the pass; the engine's config line
     (`vllm/v1/engine/core.py:124`) has `SpeculativeConfig(method='mtp', …, num_spec_tokens=K)`
     (`vllm/config/speculative.py:1952-1966`) and the drafter's counts have K positions; without the drafter the cache
     holds about 126003 tokens.
   - *Quality.* Rounding: a batch changes the kernels' shapes, and verification keeps the target's tokens at
     temperature 0 and its distribution above it, whatever the count
     (`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:571-594,630-663`). The identity set at 1 request
     against the chosen batch, and at 3 drafts against the chosen count.
   - *Gain.* From 2 to 4 requests where the cache holds 4: +50 to +70 per cent (estimate: a step of about 20 ms at two
     requests, 11 ms of it reading the weights), or +96 to +97 per cent at 4000 to 16000 tokens (model). None at 44000
     tokens, where the cache holds 2. Four drafts against three: +8 to +10 per cent at 1 to 4 requests and -5 to +8 at
     8 (model), less once the full layers' third query block is priced: they take queries in blocks of 2
     (`vllm/v1/attention/ops/triton_unified_attention.py:937-940`), so 4 drafts verify 5 tokens in 3 blocks where 3
     take 2, and 5 drafts need no fourth block.
   - *Minutes.* 41-55: the acceptance pass, the timing sweep and the operating point of the runbook.

4. **`GPU_MEMORY_UTILIZATION` 0.94, then 0.95.** When: at the chosen batch, requests wait with reason `capacity`
   (`vllm:num_requests_waiting_by_reason`) while the cache is full.
   - *Took effect.* `kv_cache_mib` and `kv_cache_tokens` in card.jsonl, which the launcher keeps already.
   - *Quality.* Rounding: more blocks change which prefixes hit.
   - *Gain.* +520 or +782 blocks, +11 or +16 per cent of the cache (estimate), and one more request at 20000 tokens
     (derived). The headroom above 0.92 is not measured, and pictures on the same card later would need it.
   - *Minutes.* 10-15.

5. **`MAX_NUM_BATCHED_TOKENS` 512.** When: as 4.
   - *Took effect.* `max_num_scheduled_tokens is set to 512 based on the speculative decoding settings`
     (`vllm/config/vllm.py:2134-2141`), and `concurrency_x100` in card.jsonl from 159 to about 178 (derived).
   - *Quality.* Rounding: the prompt chunks change.
   - *Gain.* A new request reserves less of the sliding layers, which hold two batches in flight
     (`vllm/v1/kv_cache_interface.py:816-842`, `vllm/config/vllm.py:588-610`): 1.784 requests of 65536 tokens instead of
     1.594 (derived), and with 4 at 0.95 one more request at 30000 and at 44000 tokens (derived). A prompt takes twice
     the steps, a few per cent at most (estimate). The compile hash changes, so the start compiles again, for a time
     not known.
   - *Minutes.* 10-15.

6. **Humming for the FP4 layers.** When: the cheap card and the 5090's microbenchmark find it correct and no slower
   than Marlin at up to 32 rows.
   - *Setting.* `--kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a16":"humming"}}'`
     (`vllm/config/kernel.py:328-332`), for card.py to build behind a manifest switch, off by default. Never a global
     `--linear-backend`: a value with no W4A16 kernel moves these layers to CuTe-DSL without a word
     (`.../kernels/linear/__init__.py:358-398`).
   - *Took effect.* `Applied linear backend override for 'nvfp4_w4a16': 'humming'`, then
     `Using HummingNvFp4LinearKernel for NVFP4 GEMM` (`.../kernels/linear/__init__.py:245-248,1180`).
   - *Quality.* Rounding: the same FP4 weights and scales, bf16 activations, fp32 accumulation.
   - *Gain.* Not known. At two requests a step spends 4 to 7 ms above the floor of reading its weights, and only the
     GEMMs' part of that is Humming's to win (estimate); prompt chunks may gain more. Pins: humming-kernels 0.1.12 and
     its compiler, nvidia-cuda-nvrtc 13.0.88, both in `card/vllm-requirements.txt` with their hashes.
   - *Minutes.* About 20.

7. **FlashInfer's CuTe-DSL for the FP4 layers.** When: as 6, and this model's largest input to a linear layer, read by
   a forward hook over synthetic prompts, stays below about 30000.
   - *Setting.* The same key with `flashinfer_cutedsl`.
   - *Took effect.* `Applied linear backend override for 'nvfp4_w4a16': 'flashinfer_cutedsl'`, then
     `Using FlashInferCuteDslNvFp4W4A16LinearKernel for NVFP4 GEMM`.
   - *Quality.* Rounding below that bound: the kernel turns bf16 activations into fp16 before its MMA (FlashInfer's
     `flashinfer/gemm/kernels/cute_dsl/dense_gemm_bf16_fp4_sm12x.py:150-159`), and fp16 saturates above it. Above it,
     visible or broken.
   - *Gain.* Not known; it alone has a GEMV for one row and split-K tuned per shape. Its compilation and autotuning
     count against `LOAD_DEADLINE_S`. Pins: flashinfer-python 0.6.18.post1 and nvidia-cutlass-dsl 4.7.1, in the lock.
   - *Minutes.* About 25.

8. **Dense prefix retention.** When: in the reference, a story's next turn reads about twice its new tokens, as the
   reading predicts for retention 0.
   - *Setting.* `--prefix-cache-retention-interval None` (`vllm/config/cache.py:148-154`; `None` parses as dense,
     `vllm/engine/arg_utils.py:167-173`).
   - *Took effect.* `prefix_cache_retention_interval` in `non-default args`, and more `vllm:prompt_tokens_cached` over
     the same session.
   - *Quality.* Rounding: hits instead of computing again.
   - *Gain.* +5 to +10 per cent for one story read turn by turn (estimate: about 1000 tokens a turn, and 14000 a
     compaction cycle, not read again). It may lose with 3 or more long stories at once: each freed sliding block then
     keeps its hash and joins the end of the free queue, and a prompt of 20000 tokens cycles about 6900 blocks through
     a pool of 4804 (estimate). Kept only if the story read with gaps and the stories at once both gain.
   - *Minutes.* About 15.

9. **How drafts are verified.** When: in the acceptance pass, scenes accept clearly less than compactions.
   - *Setting.* `"draft_sample_method": "probabilistic"` in `--speculative-config`
     (`vllm/config/speculative.py:588-594`), then `"rejection_sample_method": "block"` (`:517-523`, block verification
     after Sun et al.) with the drafting that won. Both apply to every request, compactions included. The second is in
     the pinned vLLM but in none of the four readings; it was found while checking this item.
   - *Took effect.* `non-default args` has them, and the acceptance by position changes.
   - *Quality.* Rounding. At temperature 0 the target keeps its own argmax whatever was drafted
     (`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:571-594`), and above it the test uses the logits the
     draft was sampled from (`vllm/v1/worker/gpu/sample/gumbel.py:159-175`), so the target's distribution is kept. Astra
     wanted the eval for probabilistic drafting; this page does not, and the owner may. Block verification keeps the
     distribution by its paper's proof (arXiv 2403.10444, cited at `rejection_sampler_utils.py:597`), not checked here
     against the kernel, so its eval is the owner's call too.
   - *Gain.* Not known: acceptance becomes the sum of min(p, q) instead of p at the drafter's top token, which helps
     where the target is flat. Kept if the scenes' tokens a step rise by 5 per cent or more and the compactions' do not
     fall.
   - *Minutes.* About 10, and 5 for block verification.

10. **The drafted count by batch size.** When: item 3 finds a smaller count better at 6 to 8 requests.
    - *Setting.* `"num_speculative_tokens_per_batch_size"` in `--speculative-config` (`vllm/config/speculative.py:479`).
      On V2 the drafter still drafts the largest count, and only verification shrinks.
    - *Took effect.* `non-default args` has it, and a set of decoding graphs is captured for each count.
    - *Quality.* Rounding.
    - *Gain.* 0 to +2 per cent (model), or up to +10 per cent at 6 to 8 requests if those steps are bound by compute
      (estimate).
    - *Minutes.* About 7.

11. **`--watermark 0.02`.** When: `vllm:num_preemptions` grows at the chosen batch.
    - *Took effect.* `non-default args` has it; preemptions against the reference.
    - *Quality.* Rounding. *Gain.* None without preemptions (`vllm/config/scheduler.py:178-183`).
    - *Minutes.* 5-10.

12. **An INT4 cache**, `KV_CACHE_DTYPE=int4_per_token_head`. When: requests still wait for the cache after 4 and 5.
    - *Took effect.* `kv_cache_dtype=int4_per_token_head` in the config line, and `kv_cache_tokens` near twice fp8's.
    - *Quality.* Visible: keys and values in 4 bits with a scale per token and head, and a query no longer in fp8
      (`vllm/v1/attention/backends/triton_attn.py:276-287,605-612`;
      `vllm/model_executor/layers/attention/attention.py:465-471`). The eval.
    - *Gain.* About 1.94 to 1.97 times the tokens (estimate: 264 bytes a token and head against 512 at head 256, 520
      against 1024 at 512). Its speed is not known.
    - *Minutes.* 15, and the eval's 35-40.

13. **FlashInfer for attention**: the sliding layers only
    (`--attention-config '{"backend_per_kind":{"sliding_window":"FLASHINFER"}}'`), all layers
    (`--attention-backend FLASHINFER`), then a workspace of 128 MiB
    (`VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728`). When: the cheap card shows FlashInfer clearly faster than
    Triton at head 512, or XQA at 4 and 5 query tokens. Not in the first rental.
    - *Setting as well.* `VLLM_HAS_FLASHINFER_CUBIN=1`, or every start asks NVIDIA's artifactory for a file that sm_120
      does not use; and `CUDA_HOME` at a pinned toolkit, since FlashInfer's JIT otherwise compiles with whatever nvcc
      the image has (FlashInfer's `flashinfer/jit/cpp_ext.py:45-63`). The lock pins nvidia-cuda-nvcc 13.4.92, but
      whether its headers suffice for that JIT is not verified.
    - *Took effect.* `Using AttentionBackendEnum.FLASHINFER backend.` (`vllm/platforms/cuda.py:478`), `FlashInfer
      resolved query dtypes: … decode_backend=xqa …` (`vllm/v1/attention/backends/flashinfer.py:910`), for head 512
      `reverting this KV-cache group to native FlashInfer decode` (`:840`), and for all layers `setting
      cudagraph_mode=PIECEWISE` (`vllm/config/compilation.py:1443,1468`). The workspace: `kv_cache_mib` up by about
      266 MiB against its FlashInfer run.
    - *Quality.* Visible: bf16 queries and probabilities where Triton has fp8. The eval. The workspace alone is
      rounding against its FlashInfer run.
    - *Gain.* The sliding layers: -3 to +4 per cent decoding, ±2 per cent reading, about 394 MiB less cache. All
      layers: -5 to -15 per cent at short contexts, perhaps a gain for one or two requests past 30000 tokens. The
      workspace: about 266 MiB more cache (estimates).
    - *Minutes.* 10-15 each, and the eval.

Beside the card: the eval could write its next run while the judge reads the last one, since the judge runs on the
owner's machine while the card waits (simple-story-chat). Identical, and it takes the judge's share off the eval's card
time; no source measures that share.

## Not in the queue

- No pinnable source: a split-KV path for queries of several tokens (the model's +10 to +70 per cent for one request
  at 4000 to 40000 tokens), a W4A4 conversion of this fine-tune, an untied FP4 `lm_head`.
- Refused by the pins or the card: FP8 activations through Humming (humming-kernels 0.1.12's ModelOpt schema takes
  `nvfp4` and `mxfp8` only, so this checkpoint's `w4a16_nvfp4` raises: its `humming/schema/modelopt.py:18-28`),
  `VLLM_BATCH_INVARIANT` (no W4A16 kernel), Marlin's environment switches, the other attention backends (FLASH_ATTN,
  B12X, FLEX_ATTENTION, TURBOQUANT, HPC: head sizes, cache types or scale), cascade attention (refused by both
  backends, and off under async speculative decoding), adaptive verification (dspark only),
  `prefill_schedule_interval` (data parallelism only), a bfloat16 cache (measured).
- No gain for its cost: full graphs or more capture sizes, `-O3` and the performance modes (the same graphs, or only
  what the card sets already), `MAX_NUM_SEQS` 16 (never the limit, and a compilation), `MAX_NUM_BATCHED_TOKENS` 2048
  or 4096 (about 86200 tokens, then less than one request of 65536), async scheduling off, the reservation of the
  whole prompt off (over-admission), xxhash for the prefix cache (not in the lock, and collisions), `stream_interval`,
  `max_num_queued_tokens` (it answers 503), positive retention intervals, INT8 and FP8 caches per token and head (no
  smaller than fp8, and visible), six drafts.
- A prompt change, not a lever of the card: a story turn that extends the previous prompt would let the default
  retention reuse it, but the narrator rule sits last in the prompt on purpose (simple-story-chat `local/prompt.ts`).
- Checked and dropped: the drafter reading's out-of-bounds write in Triton's 3D kernel assumed a head of 256 for the
  model. vLLM sizes that buffer by the largest layer's, 512 (`vllm/config/model_arch.py:126-134`;
  `vllm/v1/attention/backends/triton_attn.py:121,155`).

## The runbook

### Before any card

1. The card runs simple-serving at phase2 f3031ea or later, for `KV_OFFLOADING_GIB`.
2. Item 1 is written and dry-run on this machine. The A/B launcher stops the launcher's pair (`card --stop`) and starts
   vLLM and the gateway with the card's arguments plus one change and the rental's gateway limits, never editing
   `manifest.env`, so that the card stays prepared and a start that fails does not stop the instance. The workloads are
   built only from simple-story-chat's `examples/` and the public eval pack, their tokens counted. The reader of
   `/metrics` and of the gateway's log flags a pass that overlapped a `reader` request, repeated if time allows. All of
   it runs against the fake engine and canned `/metrics` text, the microbenchmarks as far as their arguments. Nothing
   leaves the harness but counts, times and fixed words.
3. The cheap card's model is pinned (below).
4. The owner has said yes to the price and the deadline below.

### The cheap card

The owner asked on 2026-09-27 whether the weak Blackwell cards would do for the kernels. For what carries over to the
5090, yes, at about a quarter of its price, and every new script meets a real GPU there first.

- **Card.** simple-story-chat's `npm run gpu:rent -- --lane small`: the cheapest Blackwell card of 16 GB or more under
  $0.25 an hour. An RTX 5080 if the dry run offers one, since its 84 SMs are the nearest to the 5090's 170; otherwise
  an RTX 5060 Ti of 16 GB and 36 SMs, which cost $0.146 to $0.189 an hour on 2026-09-25 and ran vLLM 0.30.0's lock.
  The RTX 5070 has 12 GB. The SM counts are NVIDIA's public figures.
- **Model.** None for the microbenchmarks, which take synthetic tensors at the 31B's shapes. For the engine starts,
  Gemma 4 E2B (`google/gemma-4-E2B-it` 3e22461f, pinned on simple-serving's `rehearsal-e2b`, 9042 MiB in bfloat16) with
  `google/gemma-4-E2B-it-assistant` 2d874ef7, pinned by each file's sha256 at download as the manifest pins. E2B is not
  NVFP4, so only the microbenchmark judges the FP4 kernels. Gemma 4 12B (707f0a3b) is closer to the 31B's sliding
  layers, but fits in 16 GB only with fp8 weights made at load, which is not verified on sm_120.
- **Steps**, 45 to 60 minutes (estimate; Astra estimated 40 to 60):
  1. Preparation, the vLLM lock's venv as `bootstrap.sh` builds it, and E2B: 8-10 min. The rehearsal of 2026-09-25
     prepared in 5 minutes and loaded in 203 to 266 s.
  2. FP4 GEMMs, 15-20 min: Marlin, Humming and CuTe-DSL through vLLM's kernel classes on random FP4 weights and scales
     at the shapes below, at 1 to 8 rows, at the verification rows of 1 to 8 requests with 2 to 5 drafts (3 to 48),
     and at prompt chunks up to 1024, under CUDA graphs. Each prints its time, its compile or JIT time and its largest
     error against a bfloat16 reference, and CuTe-DSL is also fed activations above 30000. vLLM's
     `benchmarks/kernels/benchmark_marlin.py` has an FP4 mode to start from.

     | Layer | N | K | Layers |
     | --- | --- | --- | --- |
     | qkv, sliding | 16384 | 5376 | 50 |
     | qkv, full | 20480 | 5376 | 10 |
     | o, sliding | 5376 | 8192 | 50 |
     | o, full | 5376 | 16384 | 10 |
     | gate_up | 43008 | 5376 | 60 |
     | down | 5376 | 21504 | 60 |

  3. Attention, about 10 min: Triton's unified attention at the full layers' shape (32 query heads, 4 KV heads, head
     512, fp8 cache) at 1, 4, 5 and 6 query tokens and 4000, 16000 and 40000 tokens of context, and at the sliding
     layers' (16 KV heads, head 256) at 4 and 5; FlashInfer's XQA at the sliding shape and its native decoding at head
     512, with the workspace each allocates.
  4. Engine starts with E2B and its assistant, 3 to 4 minutes each, 15-20 min: three drafts, where every code and
     count of item 1 is checked against real output; synthetic rejection for the timing sweep; the RAM copy; block
     verification; `int4_per_token_head` with the drafter; FlashInfer on the sliding layers if step 3 was strong.
- **It settles** whether each kernel builds under the lock and the 595 driver, NVRTC beside cuda-bindings, the times to
  compile and tune, agreement with the reference, how CuTe-DSL saturates, which flags start beside the drafter, and the
  harness against a real vLLM; and it drops a clear loser at the decoding sizes, where reading the weights dominates
  and a ranking should carry over.
- **It cannot settle** a speed where a kernel's tiling follows the SM count and the L2: Humming's tuner reads both
  (`humming/tune/base.py:105-112,201-203` in the humming-kernels 0.1.12 wheel), and CuTe-DSL counts SMs in its tactics
  (FlashInfer's `flashinfer/gemm/gemm_bf16_fp4_cute_dsl.py:83-125`). So the 5090 runs step 2 again, 5-10 min, before
  an A/B of the whole engine. And the 31B does not fit in 16 GB (19149 MiB of weights): no throughput, acceptance,
  start within `LOAD_DEADLINE_S` or quality comes from this card.
- **Go on.** Humming and CuTe-DSL go to the 5090 if they build, agree with the reference and are no slower than Marlin
  at up to 32 rows; FlashInfer only after a clear win in step 3; the INT4 cache if E2B starts with it and the drafter.

### The 5090

Rented as the fourth card was: `--lane text`, a 595 driver on the dry run's first offer, the three-hour guard. The
session ends at 150 minutes whatever is left, with `--destroy` read back as gone, and what is left waits for a second
rental and the owner's yes. When the card has stood idle for more than 10 minutes with no work ready, the operator
tells the owner and offers to delete it (simple-story-chat's `docs/gpu.md`, the owner's rules).

| Step | Minutes | When |
| --- | --- | --- |
| 1. Rental to the first ready: the image's start, the preparation, the load | 10-18 | always |
| 2. The FP4 microbenchmark again, on 170 SMs | 5-10 | the cheap card kept Humming or CuTe-DSL |
| 3. Reference: three drafts at 1 and 2 at once, the workloads, the identity set twice, four stories | 10-15 | always |
| 4. The RAM copy (item 2) | 7-10 | always |
| 5. Acceptance at 6 drafts: compactions at 0.2 and scenes at 0.8, 3 at once each | 8-10 | always |
| 6. The timing sweep of item 3 | 25-35 | always; 6-7 more for 5 drafts when 4 beat 3 |
| 7. The operating point: the chosen batch and count against the reference, through the gateway | 8-10 | always |
| 8. Items 4 and 5 | 20-30 | requests wait for the cache |
| 9. Items 6 and 7 | 45 | step 2 kept them |
| 10. Item 8 | 15 | a turn reads about twice its new tokens |
| 11. Item 9 | 10-15 | scenes accept clearly less |
| 12. Items 10 and 11 | 12-17 | a crossover; preemptions |
| 13. Item 12 and its eval | 50-55 | the cache still binds; a second rental unless time is left |
| 14. Stop, delete, read back | 1-5 | always |

Step 1 took 7.5 and 14.1 minutes from the preparation to ready on the fourth and third cards, and the third was
`running` 3 minutes after its rental. Step 5 needs one start: greedy drafts do not depend on the count, so 6 drafts give
the acceptance of every smaller count. Step 6 runs `vllm bench serve` from the card's vLLM venv against the engine's
port: `--dataset-name random` with a shared prefix of the context less 256 tokens, 256 new and 512 out,
`--ignore-eos`, temperature 0, at 1 to 8 requests at once and 4000, 16000 and 40000 tokens wherever the cache holds
them, and prompts that share nothing at 16000; `rejection_sample_method` `synthetic` at step 5's rates
(`vllm/config/speculative.py:517-536`) makes it timing only, never serving. The random dataset needs no package outside
the lock. Step 7 picks the batch and the count with the most tokens a second at the bot's and the eval's lengths, no
preemptions and first tokens within the reference's spread, and runs the workloads there.

### Minutes and price

The 5090 at $0.50 to $0.65 an hour. Traffic is priced by the offer, $2.6 to $52 a TB, and was $2.6 on the fourth card;
a disk of 60 GB adds about $0.017 an hour (simple-story-chat's `docs/knowledge/gpu-measurements.md`, "Prices, disks and
downloads"). The downloads are estimates: E2B and the wheels as simple-story-chat's rent plan counts the small lane, and
for the 5090 the fourth card's 54 GB less llama.cpp's 25.7.

| | Card minutes | The card | Downloads |
| --- | --- | --- | --- |
| The cheap card | 45-60 | $0.11-0.25 | about 16.5 GB: $0.04, at most $0.86 |
| The 5090, the steps that always run | 70-105 | $0.58-1.14 | about 28 GB: $0.07, at most $1.46 |
| The 5090 with the likely conditions (steps 8, 10, 11) | 115-165, cut at 150 | $0.96-1.63 | the same |
| Every condition but step 13, over two rentals | 185-245 and a second start of 10-18 | $1.63-2.85 | twice |
| The eval of a visible change | 35-40 each | $0.29-0.43 | none |

The cheap card and the 5090 up to its 150 minutes come to about $0.8 to $2.1 at the fourth card's traffic price.

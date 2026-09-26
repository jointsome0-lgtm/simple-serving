# Speed levers for the card

The owner asked on 2026-09-26 to carry over to this card what simple-story-chat's picture card had just found: its
int8 layers ran on a slow path of their kernel library until a switch put them on Triton's kernels, 2.3 times faster
once warm (simple-story-chat's `docs/knowledge/gpu-measurements.md`, the pilot of 2026-09-26). Nothing below is
measured on this card yet. Each item is a question for a rental, with the evidence that would answer it.

## Already in place

- vLLM runs without `--enforce-eager` (`simple_serving/card.py`), so torch.compile and CUDA graphs are on, and torch's
  compiler writes Triton kernels for the fused operations. FlashInfer and Triton compile kernels on the card at its
  start (the contract, section 15). There is no switch here like the picture card's.
- The KV cache is fp8 (`KV_CACHE_DTYPE`); a bfloat16 cache does not fit at 65536 tokens.
- Gemma 4's multi-token prediction drafts 3 tokens a step, which took each of two requests at once from 57 to 67
  tokens a second to 123 to 171 (the README, "The card").

## To look at, cheapest first

1. **The FP4 kernels.** Which kernel runs the NVFP4 weights on an RTX 5090 (sm_120). A warning at the engine's start
   that FP4 runs weight-only through the Marlin kernel would mean vLLM's fallback for GPUs without native FP4: this
   card's counterpart of the picture card's slow path, and the one item here that might be as large. It costs
   nothing, since the engine's log says it at the next start.
2. **The attention backend** vLLM picks for Gemma 4's sliding-window and global layers, from its log, against the
   others it offers on this GPU (FlashInfer, FlashAttention, its own Triton backend), each tried once.
3. **The CUDA graphs** of a decode step with the drafter: the mode and the capture sizes vLLM 0.30.0 uses by default,
   from its log, against full graphs if they are not the default.
4. **The drafter's count.** 3 was measured against none, and 2 and 4 were not. Its acceptance fell from 78 to 62 and
   49 per cent over the three positions.

## How to measure

As the picture card's pilot did: on one card, a reference run first, then one change at a time, each with the log line
that shows it took effect. Speed is each request's tokens a second at the bot's two calls at once, the prefill and the
decoding apart. Identity is the same requests at temperature 0 before and after: a kernel may change tokens, and a
changed text goes to simple-story-chat's eval before anything is kept. What is kept is the owner's decision.

# The Qwen3.8-27B profile

`MODEL_PROFILE=qwen` in `card/manifest.env` makes the text card serve
`sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4` with the multi-token prediction layer that is inside its
weights, in place of Gemma 4. The owner asked on 2026-10-07 to be ready to use it. The value is `gemma` by default,
and then nothing of Gemma's preparation, arguments or pins changes.

**The model has never been loaded here.** Everything below is from the repository's metadata and small text files,
read on 2026-10-07 at revision `36276309d30211d9babf72f22d1605dc2dc6357c`, and from vLLM's source at `v0.30.0`. No
weight was fetched and nothing ran on a GPU. Each fact carries its verdict; the [last section](#the-first-rental) is
what a rental has to check before the profile is trusted.

## What the profile does

- **Pins.** `QWEN_MODEL_REPO`, `QWEN_MODEL_REVISION`, `QWEN_MODEL_FILES` (the 19.7 GB of quantized weights, the
  0.85 GB of the drafter layer in bfloat16, `config.json`, `generation_config.json`, each with its SHA-256),
  `QWEN_MODEL_SIZES` (their bytes) and the tokenizer's five files. The two large hashes are the hub's LFS hashes; the
  others are of the files as fetched. A `QWEN_<NAME>` stands for `<NAME>` in `bootstrap.sh` and in
  `simple_serving/card.py` (`profiled`).
- **The preparation.** `bootstrap.sh` fetches the files as it fetches Gemma's, each kept only with its hash, into
  `models/<revision>` and `tokenizers/<revision>`; Gemma's `tokenizer/` and its drafter stay untouched, and no
  drafter repository is fetched. Before it installs or fetches, `simple_serving/preflight.py` refuses with exit 9 a
  card whose compute capability is under 12.0 (`QWEN_MIN_COMPUTE_CAP=120`), whose memory is under 30000 MiB, or
  whose disk lacks the bytes still to fetch and 1 GiB more. `VISION=1` beside the profile ends with 3.
- **The engine's arguments.** As Gemma's, but for: the model's directory and alias
  (`qwen3.8-27b-abliterated-nvfp4`), `--max-model-len 32768`, `--reasoning-parser qwen3`,
  `--speculative-config {"method": "mtp", "num_speculative_tokens": 3}` with no model named,
  `--language-model-only` always, `<think>` and `</think>` as the budget's markers while `THINKING_BUDGET=1`, and
  `--prefix-match-unit` when `QWEN_PREFIX_MATCH_UNIT` is above 0. There is no `--trust-remote-code`.
- **The gateway.** `/v1/state` names `model_profile: qwen` among the versions, and the drafted count, so the gateway
  refuses the same `min_p` as with Gemma's drafter.

## The contract, mapped

| The gateway's contract | With this model |
| --- | --- |
| The model name | `qwen3.8-27b-abliterated-nvfp4`; another name is refused as today. |
| Thinking off, the default | The gateway always sends `chat_template_kwargs.enable_thinking` (`simple_serving/engine.py`), false unless asked. The template then ends the prompt with an empty `<think>` block (`chat_template.jinja:163-168`), and the answer starts at once. |
| Thinking on | `enable_thinking: true`: the prompt ends with `<think>`, and vLLM's `qwen3` parser puts the thought in the reasoning field (`vllm/parser/qwen3.py:41-42,231`). The template's `reasoning_effort` (`xhigh` unless given, `:46-56`) is not a field the gateway takes, so thinking always runs at `xhigh`. |
| A thinking budget | Off as for Gemma (`THINKING_BUDGET=0`). With 1 the launcher gives vLLM `<think>` and `</think>`; not run. |
| A strict JSON schema | xgrammar, as for Gemma. With thinking off the grammar holds from the first token; with it on, from the token after `</think>` (`vllm/v1/structured_output/__init__.py:250-292`). |
| Pictures | Refused, as for Gemma with `VISION=0`. The checkpoint has a vision tower; the profile keeps it unloaded, and its use was not benchmarked by the model card either. |
| Context | 32768 tokens, by the manifest. The model's own limit is 262144. |

## Six facts

1. **vLLM 0.30.0 loads it natively: established from the source.** `Qwen3_5ForConditionalGeneration`, the
   `architectures` of the repository's `config.json`, is in vLLM's registry
   (`vllm/model_executor/models/registry.py:597`), its config class is vLLM's own
   (`vllm/transformers_utils/config.py:150-151`), and the drafter is registered too (`registry.py:699`). The
   repository holds no Python file and no `auto_map`. The model card's serve line has `--trust-remote-code` and was
   written for "vanilla vLLM v0.22.0"; the profile runs without it.
2. **MTP is a speculative method, and it runs under a strict schema: established from the source; its acceptance
   here is not.** `qwen3_5_mtp`, the card's name for it, is listed (`vllm/config/speculative.py:51`) and is
   "deprecated and replaced with mtp" (`:1121-1125`). With no `model` the drafter loads from the target's own
   directory (`:1127-1140`, `:904-923`): one extra decoder layer, run once a drafted token, sharing the embeddings
   and the output head (`vllm/model_executor/models/qwen3_5_mtp.py:171`,
   `vllm/v1/spec_decode/llm_base_proposer.py:1476-1482,1556-1561`). Under a schema the drafts are checked against
   the grammar and cut at the first that breaks it (`vllm/v1/core/sched/scheduler.py:2421,2449`,
   `vllm/v1/structured_output/__init__.py:294-312`), and the token mask is filled one position after another, not
   in parallel, while drafting is on (`:321-349`). The same holds for Gemma's drafter. A compiled schema is cached,
   512 MB by default (`vllm/v1/structured_output/backend_xgrammar.py:66-71`, `vllm/envs.py:1621`). **Not
   established:** that vLLM 0.30.0 keeps the 15 `mtp.*` modules out of quantization as the repository's `ignore`
   list asks (`vllm/model_executor/layers/quantization/compressed_tensors/`). The model card says that a quantized
   drafter gives 0 per cent acceptance, so an acceptance above 0 is the first number to read.
3. **NVFP4 with 4-bit activations runs on other cards too, by a fallback: established from the source.** The scheme
   asks only for compute capability 7.5 (`.../schemes/compressed_tensors_w4a4_nvfp4.py:34-36`). vLLM tries
   FlashInfer's kernels, which need sm_100 or above, one of them sm_120 (`vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:116-123,184-197,384-393`),
   then Cutlass, then weight-only Marlin with the warning "GPU does not have native support for FP4", then an
   emulation (`vllm/model_executor/kernels/linear/__init__.py:551-567,1069-1186`). So a 4090 would start, slower and
   with activations not quantized as the model was calibrated. The bootstrap refuses that card instead. **Not
   established:** which kernel a 5090 takes, and whether fused layers whose scales differ lose accuracy: vLLM fuses
   q/k/v, gate/up and the linear-attention projections and takes the largest scale with a warning
   (`compressed_tensors_w4a4_nvfp4.py:100-111`), which the launcher logs as `fused_scales_differ`.
4. **The prefix cache works for it, on wide blocks: established from the source; its capacity is an estimate.**
   The model is a hybrid: of 64 layers, 16 are full attention and 48 are linear attention with a recurrent state.
   - With prefix caching on, vLLM keeps the recurrent state per block, mode `align`
     (`vllm/model_executor/models/config.py:621-649`). One layer's state is 48 x 128 x 128 float32 and a small
     convolution state: 3,268,608 bytes with three drafts (`vllm/model_executor/layers/mamba/mamba_utils.py:274-295`).
   - An attention layer in fp8 is 2 x 4 x 256 = 2048 bytes a token, 32 KiB a token over the 16 layers, against
     Gemma's 440 KiB over both its kinds.
   - vLLM widens the attention block until one state fits a page (`vllm/platforms/interface.py:778-962`):
     **1600 tokens a block**, 50 MiB for each of four groups by this arithmetic, one of attention and three of
     states. The grouping was not seen in a log.
   - A hit needs the attention blocks of the prefix and one state at its end
     (`vllm/v1/core/single_type_kv_cache_manager.py:1487-1564`), so by default it lands only on a multiple of 1600
     tokens: a request of 3000 tokens reuses 1600 at most, and a turn that adds 300 tokens reuses up to the last
     edge below its end. `--prefix-match-unit` (`vllm/config/cache.py:91-102`) lets a hit land on a finer edge, where
     vLLM keeps a state for the prompt's own end (`single_type_kv_cache_manager.py:2039-2100`);
     `QWEN_PREFIX_MATCH_UNIT` passes it and is 0 until a rental tries it.
   - The copy in RAM supports these groups (`offloading/scheduler.py:136-165,222-240,281-283`). With dense
     retention it writes 50 MiB of attention and 150 MiB of states for 1600 tokens, 128 KiB a token; a hit needs
     N x 32 KiB and 150 MiB.
   - **Capacity, an estimate.** A 5090 has 32607 MiB, and 0.92 of it is 29998 MiB. The weights are 19607 MiB on
     disk with the drafter, about 0.5 GiB below Gemma's 20070 MiB with its drafter. If the engine's overhead is as
     Gemma's, the cache is 7 to 10 GiB, 146 to 199 blocks. A running request holds at least two states in each of
     three groups, 300 MiB, beside its attention blocks. vLLM's own count, which was 104492 tokens for Gemma, depends
     on block counts for drafting that were not established; it is read at the start from `kv_cache_tokens`.
5. **The context is 262144 tokens: established.** `max_position_embeddings` and the tokenizer's `model_max_length`.
   The card serves 32768. Memory at load was not measured.
6. **The template's thinking controls: established.** `enable_thinking`, `reasoning_effort` (`xhigh`, `medium`,
   `low`, each a line of system text) and `preserve_thinking` (`chat_template.jinja:46-56,77-86,116,163-168`). The
   template is byte for byte Qwen's own.

The model card's own numbers, on four RTX PRO 2000 Blackwell cards and vLLM 0.22.0, are 66.6 tokens a second for one
request with 44.6 per cent of drafts accepted, 191 for four and 368 for eight, and 3250 prompt tokens a second. They
are not this card's.

## The first rental

In this order, each a number in `card.jsonl` or the gateway's log:

1. `bootstrap.sh` ends with 0 and the engine is ready within `LOAD_DEADLINE_S`: minutes of the fetch and the load.
2. `kv_cache_tokens`, the `fused_scales_differ` rows and the kernel rows of the engine's choices.
3. Drafted and accepted tokens after ten strict-schema requests: above 0, or the drafter was quantized and
   `QWEN_MTP_SPECULATIVE_TOKENS=0` is the setting.
4. The [worlds measurement](worlds.md): cached tokens in later turns, which shows where hits land, then once more
   with `QWEN_PREFIX_MATCH_UNIT=32`.

Not wired for the profile: `simple_serving/sweep.py` reads the manifest by itself and still plans for Gemma.

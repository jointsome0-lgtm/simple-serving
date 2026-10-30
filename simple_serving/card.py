"""The card's launcher: vLLM and the gateway as one pair, started at every start of the container (contract section 8).

    python -m simple_serving.card [--hold | --stop | --retry | --dry-run | --inspect]

card/onstart.sh runs it from the checkout with the gateway's venv, and card/bootstrap.sh prepares the card once per
rental. Without an option it checks the card, starts the launcher in the background and returns 0, also when one
already runs. Otherwise it returns 3 when the card is not prepared (bootstrap.sh has not run for this manifest and
these locks, or the keys or the instance's credential are missing), 6 when the card has given up, and the file
`given-up` says why, and 7 when a port is taken. `--retry` removes `given-up` and starts; when a launcher runs already,
it says that the retry is left to it or deferred to the next start. `--stop` ends the pair and leaves the instance
running. `--hold` prints one line, then waits while the pair runs and returns 0 once it has ended, 6 once the card has
given up, or 8 once its stop is not confirmed; on such a card, or one that is not prepared, it returns 8, 6 or 3 at once
and prints nothing. It is the remote command of the tunnel, and it never owns the pair. `--dry-run` prints the two
commands and checks, and starts nothing. `--inspect` is the smoke's report of the card (`smoke.py`): it reads one JSON
request on standard input and prints one line of numbers and flags, or returns 2 and prints nothing.

The pair runs once, with no restarts. It ends on SIGTERM, when a process exits, or when the gateway is not ready by
the manifest's load deadline. Unless a signal ended it, the launcher then stops the instance as the gateway does when
it falls asleep (`vast.py`). When the pair never became ready, it first leaves `given-up`, and the card has given up:
a later start, a resume included, loads nothing. Its launcher waits the idle interval for the owner's `--retry`, and
without one stops the instance again, so that no card stays up with nothing to stop it. It looks for the retry every
POLL_S and then runs the pair; a retry in the interval's last moment may lose to its end, and the next start loads.

An attempt to stop the instance that fails, the launcher's or the gateway's, leaves `stop-unconfirmed` until the
launcher's next start: costs may go on. The attempts go on too, but they bound nothing while Vast refuses the key.

The logs are in `logs/` of the state directory, with one older file beside each. The launcher makes a log 0600 when it
opens it; one that it moves aside keeps its mode. `card.jsonl` holds the launcher's rows and, of vLLM's output, only a
few numbers, fixed diagnostic codes of engine choices, and fixed categories of failure and warning. `gateway.jsonl`
holds the gateway's rows that pass `gateway_row`.

The picture lane's launcher is this one with ComfyUI for its engine, in a state of its own (`pictures/card.py`): a
lane is a `Card` whose class names its module, its state and its files.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import resource
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Self

from . import log, preflight, vast
from .config import DRAFTED_TOKENS, IDLE_TIMEOUT_S, MIN_P_WHILE_DRAFTING, PROVISIONAL, THINKING_BUDGET, VISION

CODE = Path(__file__).resolve().parent.parent  # the checkout
STATE = Path(os.environ.get("SIMPLE_SERVING_CARD_DIR", "/workspace/simple-serving-card"))
ROOT = Path(os.environ.get("SIMPLE_SERVING_CARD_ROOT", "/root"))  # where onstart.sh keeps the instance's id and key
PROC = Path(os.environ.get("SIMPLE_SERVING_CARD_PROC", "/proc"))  # the processes, moved for tests as the two above
RUNNING = b"simple_serving.card\0--run"  # in the command line of a launcher
DEADLINE = ".simple-chat-trial-deadline"  # in ROOT: the trial guard's time, which trial-onstart.sh writes once
LOGS = ("card.jsonl", "card.jsonl.1", "gateway.jsonl", "gateway.jsonl.1")  # in the state's logs/
MARKER = re.compile(r"[0-9a-f]{32}")  # the smoke's privacy marker
REQUEST_BYTES = 4096  # the most of a request to --inspect
STAMPED = ("manifest.env", "gateway-requirements.txt", "vllm-requirements.txt")  # the files in card/ that were prepared
NOT_PREPARED, GAVE_UP, PORT_TAKEN, STOP_UNCONFIRMED = 3, 6, 7, 8
GIVEN_UP, UNCONFIRMED = "given-up", "stop-unconfirmed"  # the card's markers
HOLDING = "simple-serving card: holding"  # the first line of --hold
LINE_BYTES = 8192  # a longer line of output reaches the filter cut to this length
CARD_LOG_BYTES, GATEWAY_LOG_BYTES = 1 << 20, 8 << 20
GATEWAY_GRACE_S, ENGINE_GRACE_S = 10, 30  # from SIGTERM to SIGKILL
POLL_S, HOLD_START_S, STOP_WAIT_S = 2, 60, 60
ENGINE_ENV = {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
# The markers of Gemma 4's thought, <|channel>thought\n...<channel|>: the pinned tokenizer's soc_token and eoc_token
# (tokenizer_config.json), and vLLM 0.30.0's CHANNEL_START and CHANNEL_END (vllm/parser/gemma4.py:38-39). vLLM counts
# a budget's tokens from the first, and writes the second once they are spent.
THOUGHT_MARKERS = {"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}
# The manifest's second model, MODEL_PROFILE=qwen (docs/qwen.md): Qwen3.8-27B in NVFP4, vLLM's Qwen3_5 architecture.
# Its thought is <think>...</think> (the pinned chat_template.jinja:165-168; vllm/parser/qwen3.py:41-42).
QWEN = "qwen"
QWEN_THOUGHT_MARKERS = {"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}
# MODEL_PROFILE=small (docs/cheap-card.md): Google's Gemma 4 E2B with its assistant, for a card of 16 GB on which the
# service's mechanisms are tried before the 5090. Its arguments are Gemma's.
SMALL = "small"
PROFILES = {QWEN: "QWEN_", SMALL: "SMALL_"}  # a profile's prefix in the manifest

# vLLM's wording of the numbers kept, as in the source of vLLM 0.30.0, and provisional until the first rental checks
# each against its output. A field's last word names its unit, and SCALE turns vLLM's GiB, seconds and factors into it.
MEASUREMENTS = tuple(re.compile(pattern) for pattern in (
    r"^Model loading took (?P<weights_mib>\d+(?:\.\d+)?) GiB(?: memory)? and (?P<weights_load_ms>\d+(?:\.\d+)?) s",
    r"^Available KV cache memory: (?P<kv_cache_mib>-?\d+(?:\.\d+)?) GiB",  # below 0 when the weights leave no room
    r"^GPU KV cache size: (?P<kv_cache_tokens>(?:\d{1,3}(?:,\d{3})+|\d+)) tokens",
    (r"(?:^|, )Maximum concurrency for (?:\d{1,3}(?:,\d{3})+|\d+) tokens per request: "
     r"(?P<concurrency_x100>\d+(?:\.\d+)?)x"),
))
SCALE = {"mib": 1024, "ms": 1000, "tokens": 1, "x100": 100}
# The categories of failure, each with words of a lowercased line, provisional as well. A line gets the first that fits.
FAILURES = (
    ("out_of_memory", ("out of memory", "outofmemoryerror")),
    ("kv_cache_too_small", ("larger than the available kv cache memory", "no available memory for the cache blocks")),
    # Another process holds a part of the card's memory at the start. vLLM 0.30.0 also prints "Free memory on device"
    # on every start that goes well, so the words are those of its refusal.
    ("memory_taken", ("less than desired gpu memory utilization",)),
    ("memory_profiling_failed", ("error in memory profiling",)),  # another process freed memory while vLLM profiled
    ("cuda_error", ("cuda error", "cudaerror", "cublas_status", "nccl error")),
    ("bind_failed", ("address already in use",)),
    ("argument_error", ("unrecognized arguments", "error: argument")),
    ("engine_dead", ("enginedeaderror", "engine core initialization failed", "frontend process failed")),
    # vLLM prints one for a request it refuses in its stream as well, such as one with a schema it cannot compile.
    ("traceback", ("traceback (most recent call last)",)),
)
# And of warnings, by the same rule. NVFP4 weights that vLLM fuses, q/k/v or gate/up, should share one global scale;
# where they do not, it takes the largest, which "will likely reduce accuracy" (modelopt.py in vLLM 0.30.0).
WARNINGS = (("fused_scales_differ", ("scale differs across parallel layers",)),)
GATEWAY_KEYS = log.FIELDS | {"time", "logger", "level"}

# Diagnostic messages from vLLM 0.30.0: fixed codes and bounded numbers, never its arbitrary strings. An arg_ code
# records a requested argument, not an observed kernel choice. Other codes record the stage the message names,
# not successful inference or a completed transfer. Unknown words become other; an absent row proves nothing.
NVFP4_KERNELS = {  # vLLM's NVFP4 kernel classes, from .../kernels/linear/__init__.py
    "MarlinNvFp4LinearKernel": "marlin", "HummingNvFp4LinearKernel": "humming",
    "FlashInferCuteDslNvFp4W4A16LinearKernel": "flashinfer_cutedsl_w4a16",
    "FlashInferCuteDslNvFp4LinearKernel": "flashinfer_cutedsl",
    "FlashInferCutlassNvFp4LinearKernel": "flashinfer_cutlass",
    "FlashInferTrtllmNvFp4LinearKernel": "flashinfer_trtllm", "FlashInferCudnnNvFp4LinearKernel": "flashinfer_cudnn",
    "FlashInferB12xNvFp4LinearKernel": "flashinfer_b12x", "B12xNvFp4LinearKernel": "b12x",
    "CutlassNvFp4LinearKernel": "cutlass", "FbgemmNvFp4LinearKernel": "fbgemm", "TorchNvFp4LinearKernel": "torch",
    "EmulationNvFp4LinearKernel": "emulation",
}
LINEAR_QUANTS = frozenset({  # the keys of KernelConfig.linear_backend_per_quant
    "nvfp4_w4a16", "nvfp4_w4a4", "fp8_block_w8a8", "fp8_w8a8", "int8_w8a8", "mixed_precision", "mxfp8", "mxfp4",
    "mxfp6", "w8a16_fp8",
})
LINEAR_BACKENDS = frozenset({  # vllm/config/kernel.py's LinearBackend
    "auto", "cutlass", "flashinfer_cutlass", "flashinfer_cutedsl", "flashinfer_trtllm", "flashinfer_cudnn",
    "flashinfer_b12x", "b12x", "marlin", "humming", "triton", "deep_gemm", "torch", "aiter", "machete", "fbgemm",
    "conch", "exllama", "emulation", "xpu", "xpu_woq",
})
ATTENTION_BACKENDS = frozenset({  # vllm/v1/attention/backends/registry.py's AttentionBackendEnum
    "FLASH_ATTN", "FLASH_ATTN_DIFFKV", "TRITON_ATTN", "TRITON_ATTN_DIFFKV", "TRITON_FLASH_ATTN", "ROCM_ATTN",
    "ROCM_AITER_MLA", "ROCM_AITER_TRITON_MLA", "ROCM_AITER_FA", "ROCM_AITER_MLA_SPARSE", "XPU_MLA_SPARSE", "TORCH_SDPA",
    "FLASHINFER", "TRITON_FLASHINFER", "FLASHINFER_MLA", "TOKENSPEED_MLA", "FLASHINFER_MLA_SPARSE",
    "FLASHINFER_MLA_SPARSE_SM120", "FLASHINFER_MLA_SPARSE_SM90", "TRITON_MLA", "CUTLASS_MLA", "FLASHMLA",
    "FLASHMLA_SPARSE", "FLASHMLA_SPARSE_DSV4", "FLASHINFER_MLA_SPARSE_DSV4", "ROCM_FLASHMLA_SPARSE_DSV4",
    "FLASHMLA_SPARSE_DSV41", "FLASHINFER_MLA_SPARSE_DSV41", "B12X", "FLASH_ATTN_MLA", "FLASH_ATTN_MLA_SPARSE",
    "MINIMAX_M3_SPARSE", "CUTLASS_MSA", "TRITON_MSA", "NO_ATTENTION", "FLEX_ATTENTION", "HPC_ATTN",
    "ROCM_AITER_UNIFIED_ATTN", "CPU_ATTN", "CPU_MLA", "AMX_MLA", "TURBOQUANT", "CUSTOM",
})
CACHE_KINDS = frozenset({  # vllm/v1/kv_cache_interface.py's KVCacheSpecKind, the keys of backend_per_kind
    "full_attention", "mla_attention", "sliding_window", "sliding_window_mla", "mamba", "chunked_local_attention",
    "sink_full_attention", "encoder_only_attention", "cross_attention", "unknown", "sparse",
})
CACHE_DTYPES = frozenset({  # vllm/config/cache.py's CacheDType
    "auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla", "nvfp4_ds_mla",
    "turboquant_k8v4", "turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_3bit_nc", "int4_per_token_head",
    "int8_per_token_head", "fp8_per_token_head", "nvfp4", "nvfp4_4over6",
})
DECODE_PATHS = {"xqa": "xqa", "trtllm-gen": "trtllm_gen", "flashinfer-native": "native"}  # FlashInfer's decoding
OFFLOADING_SPECS = {"CPUOffloadingSpec": "cpu", "TieringOffloadingSpec": "tiering"}  # vllm/v1/kv_offload/factory.py
# The keys of non-default args retained as requested numbers or fixed words, after structural parsing.
ARG_NUMBERS = {
    "max_model_len": ("arg_max_model_len", 1), "max_num_seqs": ("arg_max_num_seqs", 1),
    "max_num_batched_tokens": ("arg_max_num_batched_tokens", 1),
    "gpu_memory_utilization": ("arg_gpu_memory_utilization_x1000", 1000), "watermark": ("arg_watermark_x1000", 1000),
    "kv_offloading_size": ("arg_kv_offloading_mib", 1024),  # GiB in vLLM
    "prefix_cache_retention_interval": ("arg_prefix_cache_retention_interval", 1),  # None is dense retention
    "num_speculative_tokens": ("arg_num_speculative_tokens", 1),  # in --speculative-config's JSON
}
ARG_WORDS = {
    "kv_cache_dtype": CACHE_DTYPES, "attention_backend": ATTENTION_BACKENDS,
    "draft_sample_method": frozenset({"greedy", "probabilistic"}),
    "rejection_sample_method": frozenset({"standard", "synthetic", "block"}),
}
ARG_FLAGS = ("num_speculative_tokens_per_batch_size", "synthetic_acceptance_rates")  # their presence only
NUMBER = r"-?\d+(?:\.\d+)?"

DIAGNOSTIC_MAX = (1 << 53) - 1  # a finite JSON-safe integer guard, not a limit imposed on vLLM
ENGINE_PREFIX = re.compile(
    r"^(?:\([^\r\n)]{1,80}\) )?(?:DEBUG|INFO|WARNING|ERROR) \d{2}-\d{2} \d{2}:\d{2}:\d{2} "
    r"\[[^\]\r\n]{1,160}\] ")


@dataclass(frozen=True)
class Command:
    argv: list[str]
    env: dict[str, str] = field(repr=False)
    cwd: Path | None = None


@dataclass(frozen=True)
class Card:
    code: Path  # the checkout, with the lane's manifest.env
    state: Path
    manifest: Mapping[str, str]
    credential: Mapping[str, str] = field(repr=False)  # CONTAINER_ID and CONTAINER_API_KEY, where known
    root: Path = ROOT

    # The lane's: its launcher's module, whose `--run` a running launcher's command line holds, where its state is,
    # and its files in the checkout, of which the preparation stamps these.
    module: ClassVar[str] = "simple_serving.card"
    running: ClassVar[bytes] = RUNNING
    home: ClassVar[Path] = STATE
    folder: ClassVar[str] = "card"
    stamped: ClassVar[tuple[str, ...]] = STAMPED

    @classmethod
    def load(cls, code: Path = CODE, state: Path | None = None, environ: Mapping[str, str] = os.environ,
             root: Path = ROOT) -> Self:
        return cls(code, cls.home if state is None else state, read_manifest(code, cls.folder),
                   credential(environ, root), root)

    @property
    def ports(self) -> dict[str, int]:
        return {name: int(self.manifest[f"{name.upper()}_PORT"]) for name in ("public", "control", "engine")}

    def engine(self) -> Command:
        m = profiled(self.manifest)
        qwen = m.get("MODEL_PROFILE") == QWEN
        other = m.get("MODEL_PROFILE") in PROFILES
        tokenizer = self.state / "tokenizers" / m["TOKENIZER_REVISION"] if other else self.state / "tokenizer"
        return Command([
            str(self.state / "vllm/bin/vllm"), "serve", str(self.state / "models" / m["MODEL_REVISION"]),
            "--served-model-name", m["MODEL_ALIAS"], "--host", "127.0.0.1", "--port", m["ENGINE_PORT"],
            "--tokenizer", str(tokenizer), "--max-model-len", m["CONTEXT_TOKENS"],
            "--max-num-seqs", m["MAX_NUM_SEQS"], "--max-num-batched-tokens", m["MAX_NUM_BATCHED_TOKENS"],
            "--gpu-memory-utilization", m["GPU_MEMORY_UTILIZATION"], "--kv-cache-dtype", m["KV_CACHE_DTYPE"],
            "--scheduling-policy", "priority", "--enable-prefix-caching", "--enable-prompt-tokens-details",
            # Dense sliding-window checkpoints for reuse between story turns; dated measurements are in docs/speed.md.
            "--prefix-cache-retention-interval", "None",
            # The model's thinking in the reasoning field, apart from the answer.
            "--reasoning-parser", "qwen3" if qwen else "gemma4",
            # A schema's JSON from xgrammar alone, which has no fallback: a schema compiles with the pinned version or
            # is refused. Whitespace between its tokens is allowed, as the bot's llama.cpp allows it: forbidden, it left
            # 5 of the action measurement's 18 sheets with nobody on them (contract, "Decided").
            "--structured-outputs-config", '{"backend": "xgrammar"}',
            *self.vision(),
            # Sampling defaults from vLLM: the model's generation_config.json is pinned for its stop ids alone.
            "--generation-config", "vllm",
            "--disable-uvicorn-access-log", "--uvicorn-log-level", "warning",
            *self.speculation(), *self.offloading(), *self.reasoning(), *self.matching(), *self.cache(),
        ], {name: value for name, value in os.environ.items() if name != "CONTAINER_API_KEY"} | ENGINE_ENV
            | self.runner())

    def runner(self) -> dict[str, str]:
        """The V2 model runner, fixed by name while vLLM's refusal of a min_p is lifted (bootstrap.sh): vLLM picks it by
        itself where nothing rules it out (VllmConfig.use_v2_model_runner), and the V1 runner would drop the filter
        without a word. Otherwise nothing, so that the engine's environment stays as it was."""
        return {"VLLM_USE_V2_MODEL_RUNNER": "1"} if min_p_while_drafting(self.manifest) else {}

    def speculation(self) -> list[str]:
        """vLLM's arguments for Gemma 4's multi-token prediction, and none while MTP_SPECULATIVE_TOKENS is 0: the
        drafter that bootstrap.sh fetched drafts that many tokens a step, and the heretic verifies each."""
        m = profiled(self.manifest)
        tokens = int(m["MTP_SPECULATIVE_TOKENS"])
        if not tokens:
            return []
        if m.get("MODEL_PROFILE") == QWEN:
            # Qwen's drafter is a layer inside its own weights (mtp.* in model-mtp-bf16.safetensors), which vLLM
            # 0.30.0 loads from the target's directory when no model is named (vllm/config/speculative.py:1127-1140).
            # The model card's method name, qwen3_5_mtp, is deprecated there for this one (:1121-1125).
            return ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": tokens})]
        drafter = self.state / "drafters" / m["DRAFTER_REVISION"]
        return ["--speculative-config",
                json.dumps({"method": "mtp", "model": str(drafter), "num_speculative_tokens": tokens})]

    def offloading(self) -> list[str]:
        """vLLM's argument for a copy of the KV cache in the card's RAM, a buffer of KV_OFFLOADING_GIB for its native
        backend, and none while that is 0."""
        gib = int(self.manifest["KV_OFFLOADING_GIB"])
        return ["--kv-offloading-size", str(gib)] if gib else []

    def reasoning(self) -> list[str]:
        """vLLM's argument for a thinking budget while THINKING_BUDGET is 1, and none while it is 0. vLLM 0.30.0 takes
        the same markers from --reasoning-parser gemma4 without it (vllm/config/reasoning.py:83-95), so the argument
        pins them rather than turns the budget on: with or without it, the gateway is what refuses a budget while the
        switch is 0. It is not among the factors of vLLM's compile cache (VllmConfig.compute_hash)."""
        if not int(self.manifest["THINKING_BUDGET"]):
            return []
        qwen = self.manifest.get("MODEL_PROFILE") == QWEN
        return ["--reasoning-config", json.dumps(QWEN_THOUGHT_MARKERS if qwen else THOUGHT_MARKERS)]

    def matching(self) -> list[str]:
        """vLLM's argument for where a prefix hit may land, for a profile that sets PREFIX_MATCH_UNIT above 0, and
        none otherwise. Qwen's cache block is as wide as one recurrent state, about 1600 tokens, and by default a hit
        lands only on a block's edge; a finer unit that divides the block lets it land on the last prompt's own edge
        (vllm/config/cache.py:91-102). Gemma's blocks are 16 and 32 tokens, so it has no such value."""
        unit = int(profiled(self.manifest).get("PREFIX_MATCH_UNIT", "0"))
        return ["--prefix-match-unit", str(unit)] if unit else []

    def unfit(self) -> list[str]:
        """Why this container cannot hold the manifest's copy of the KV cache in RAM, by preflight.py's codes, or
        nothing. While KV_OFFLOADING_GIB is 0, nothing is read."""
        return preflight.offload_unfit(int(self.manifest.get("KV_OFFLOADING_GIB", "0")))

    def cache(self) -> list[str]:
        """vLLM's argument for a KV cache of KV_CACHE_MEMORY_MIB on the GPU, and none while that is 0, when vLLM
        sizes the cache from GPU_MEMORY_UTILIZATION. A measurement makes the cache small on purpose with it, so that
        a small model's sessions fall out of the GPU's cache as a large model's do (docs/cheap-card.md). vLLM then
        ignores GPU_MEMORY_UTILIZATION (vllm/config/cache.py:228-235)."""
        mib = int(profiled(self.manifest).get("KV_CACHE_MEMORY_MIB", "0"))
        return ["--kv-cache-memory-bytes", str(mib << 20)] if mib else []

    def limits(self) -> dict[str, Any]:
        """The gateway's limits: the provisional ones, or with GENERATIONS_AT_ONCE above 0 that many generations at
        once over all classes and for each of our three, so that a measurement reaches the engine with more than
        the contract's four. Waiting places and the outside class stay as they are."""
        limits: dict[str, Any] = PROVISIONAL["limits"]
        at_once = int(profiled(self.manifest).get("GENERATIONS_AT_ONCE", "0"))
        if not at_once:
            return limits
        return limits | {"shared": {"active": at_once}} | {
            cls: limits[cls] | {"active": at_once} for cls in ("reader", "agent", "internal")}

    def vision(self) -> list[str]:
        """vLLM's arguments for pictures. While VISION is 0, text only: the vision tower stays off the GPU. With 1 it
        loads, a prompt holds one image at most and no video or audio, and the processor cache is disabled. The
        separate encoder cache still reuses embeddings across scopes (contract section 10)."""
        if not int(self.manifest["VISION"]) or self.manifest.get("MODEL_PROFILE") in PROFILES:
            return ["--language-model-only"]  # another profile is text only: bootstrap.sh refuses it with VISION=1
        return ["--limit-mm-per-prompt", json.dumps({"image": 1, "video": 0, "audio": 0}),
                "--mm-processor-cache-gb", "0"]

    def gateway(self) -> Command:
        return Command([str(self.state / "gateway/bin/python"), "-m", "simple_serving"],
                       {**os.environ, **self.credential, "SIMPLE_SERVING_CONFIG": str(self.state / "gateway.json")},
                       self.code)

    def config(self) -> dict[str, Any]:
        """The gateway's configuration: the provisional limits, the manifest's model and ports, and the hashes of the
        keys. The text client key may send pictures while vLLM takes them."""
        m, keys = profiled(self.manifest), self.keys()
        if int(m["VISION"]):
            keys[0]["vision"] = True
        return PROVISIONAL | {
            "limits": self.limits(),
            "alias": m["MODEL_ALIAS"],
            "context_tokens": int(m["CONTEXT_TOKENS"]),
            "engine_url": f"http://127.0.0.1:{m['ENGINE_PORT']}",
            "listen": self.listen(),
            "keys": keys,
            "versions": pinned_versions(m),
            "warm_up": m.get("WARM_UP", "0") == "1",
        }

    def listen(self) -> dict[str, dict[str, Any]]:
        return {name: {"host": "127.0.0.1", "port": self.ports[name]} for name in ("public", "control")}

    def keys(self) -> list[dict[str, Any]]:
        """The gateway's client, control and optional metrics keys, by the hashes that the preparation keeps."""
        keys = json.loads((self.state / "keys.json").read_text())
        return [
            {"sha256": keys["client"], "label": "client", "classes": ["reader", "agent", "internal"],
             "default": "internal", "scopes": True, "control": False},
            {"sha256": keys["control"], "label": "control", "classes": [], "default": None, "scopes": False,
             "control": True},
            *([{"sha256": keys["metrics"], "label": "metrics", "classes": [], "default": None, "scopes": False,
                "control": False, "metrics": True}] if "metrics" in keys else []),
        ]

    def pair(self) -> Pair:
        return Pair(self.state)

    def prepared(self) -> bool:
        """Whether bootstrap.sh prepared the card for this manifest and these locks, and the gateway will be able to
        stop the instance."""
        try:
            stamp = hashlib.sha256(b"".join((self.code / self.folder / name).read_bytes() for name in self.stamped))
            return ((self.state / "prepared").read_text().strip() == stamp.hexdigest()
                    and (self.state / "keys.json").is_file()
                    and vast.from_environment(self.credential) is not vast.unconfigured)
        except OSError:
            return False


def min_p_while_drafting(manifest: Mapping[str, str]) -> bool:
    """Whether vLLM drafts and its refusal of a min_p is lifted: without a drafter there is nothing to lift. The count
    is the profile's, as in bootstrap.sh."""
    return bool(int(profiled(manifest)["MTP_SPECULATIVE_TOKENS"]) and int(manifest["MIN_P_WHILE_DRAFTING"]))


def pinned_versions(manifest: Mapping[str, str]) -> dict[str, str]:
    """The engine's pins that the gateway shows the control key in /v1/state, beside its own (contract section 12).
    With multi-token prediction on they name the drafter and its tokens too, and with the KV cache's copy in RAM its
    size, so that a record says which it measured. Drafted tokens tell the gateway which min_p vLLM refuses
    (config.DRAFTED_TOKENS), unless the entry of MIN_P_WHILE_DRAFTING says that it filters by it. The thinking budget
    and vision entries allow a request's budget and pictures to reach vLLM (config.THINKING_BUDGET and
    config.VISION)."""
    manifest = profiled(manifest)
    qwen = manifest.get("MODEL_PROFILE") == QWEN
    pins = {"vllm": manifest["VLLM_VERSION"], "model_revision": manifest["MODEL_REVISION"],
            "tokenizer_revision": manifest["TOKENIZER_REVISION"]}
    if manifest.get("MODEL_PROFILE") in PROFILES:
        pins |= {"model_profile": manifest["MODEL_PROFILE"]}
    if int(manifest.get("KV_CACHE_MEMORY_MIB", "0")):
        pins |= {"kv_cache_memory_mib": manifest["KV_CACHE_MEMORY_MIB"]}
    if int(manifest.get("GENERATIONS_AT_ONCE", "0")):
        pins |= {"generations_at_once": manifest["GENERATIONS_AT_ONCE"]}
    if int(manifest["MTP_SPECULATIVE_TOKENS"]):
        pins |= {} if qwen else {"drafter_revision": manifest["DRAFTER_REVISION"]}  # Qwen's drafter is in its weights
        pins |= {DRAFTED_TOKENS: manifest["MTP_SPECULATIVE_TOKENS"]}
    if min_p_while_drafting(manifest):
        pins |= {MIN_P_WHILE_DRAFTING: "on"}
    if int(manifest["KV_OFFLOADING_GIB"]):
        pins |= {"kv_offloading_gib": manifest["KV_OFFLOADING_GIB"]}
    if int(manifest["THINKING_BUDGET"]):
        pins |= {THINKING_BUDGET: "on"}
    if int(manifest["VISION"]):
        pins |= {VISION: manifest["VISION"]}
    return pins


def read_manifest(code: Path = CODE, folder: str = "card", name: str = "manifest.env") -> dict[str, str]:
    lines = (line.strip() for line in (code / folder / name).read_text().splitlines())
    return dict(profiled({name: value for name, _, value in (line.partition("=") for line in lines)
                          if name and not name.startswith("#")}))


def profiled(manifest: Mapping[str, str]) -> Mapping[str, str]:
    """The manifest as its model profile reads it. With MODEL_PROFILE=qwen each QWEN_<NAME> stands for <NAME>, and
    with small each SMALL_<NAME>, as in bootstrap.sh; with gemma, or with no such value, the manifest is itself."""
    prefix = PROFILES.get(manifest.get("MODEL_PROFILE", ""))
    if prefix is None:
        return manifest
    return {**manifest, **{name.removeprefix(prefix): value for name, value in manifest.items()
                           if name.startswith(prefix)}}


def credential(environ: Mapping[str, str], root: Path) -> dict[str, str]:
    """The instance's id and key: from the environment Vast gives the container, or, in an SSH session, which lacks
    it, from the files onstart.sh writes."""
    files = {"CONTAINER_ID": ".simple-chat-instance-id", "CONTAINER_API_KEY": ".simple-chat-instance-api-key"}
    found = {}
    for name, file in files.items():
        with contextlib.suppress(OSError):
            found[name] = environ.get(name) or (root / file).read_text()
    return found


def launcher(state: Path, running: bytes = RUNNING) -> int | None:
    """The pid of the running launcher, if one runs."""
    try:
        pid = int((state / "launcher.pid").read_text())
        found = running in Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return None
    return pid if found else None


def taken(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True
        return False


def start(card: Card, *, dry_run: bool = False) -> int:
    """Start the launcher in the background, unless something stands in the way. A card that has given up gets its
    launcher all the same, since that is what stops the instance, and 6 says so. A dry run prints the two commands and
    starts nothing."""
    if dry_run:
        print(shlex.join(card.engine().argv), shlex.join(card.gateway().argv), sep="\n")
    given_up = (card.state / GIVEN_UP).exists()
    if launcher(card.state, card.running) is not None:
        return GAVE_UP if given_up else 0
    if not given_up and not card.prepared():
        return NOT_PREPARED
    if not given_up and any(taken(port) for port in card.ports.values()):
        return PORT_TAKEN
    if not dry_run:
        subprocess.Popen([sys.executable, "-m", card.module, "--run"], cwd=card.code,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    return GAVE_UP if given_up else 0


def hold(card: Card, pause: Callable[[float], object] = time.sleep) -> int:
    """Wait while the launcher runs. One that onstart has not started yet is waited for HOLD_START_S. The first line
    out tells the command that its forwards are up, since ssh runs its remote command only once it has bound them. A
    card with nothing to hold gets no line: one that a marker ends, or one that is not prepared and runs no launcher."""
    if code := marked(card.state):
        return code
    if launcher(card.state, card.running) is None and not card.prepared():
        return NOT_PREPARED
    print(HOLDING, flush=True)
    waited, seen = 0, False
    while not (code := marked(card.state)):
        if launcher(card.state, card.running) is not None:
            seen = True
        elif seen or waited >= HOLD_START_S:
            return 0
        pause(POLL_S)
        waited += POLL_S
    return code


def marked(state: Path) -> int:
    """What the card's markers say, the stop first: STOP_UNCONFIRMED, GAVE_UP, or 0."""
    return STOP_UNCONFIRMED if (state / UNCONFIRMED).exists() else GAVE_UP if (state / GIVEN_UP).exists() else 0


def stop_pair(state: Path, pause: Callable[[float], object] = time.sleep, running: bytes = RUNNING) -> int:
    """End the pair through its launcher, and wait for it. 1 when the launcher still runs after STOP_WAIT_S."""
    pid = launcher(state, running)
    if pid is not None:
        os.kill(pid, signal.SIGTERM)
    for _ in range(STOP_WAIT_S // POLL_S):
        if launcher(state, running) is None:
            return 0
        pause(POLL_S)
    return 1


def run(card: Card) -> int:
    """The launcher itself, in the background: one pair under the lock, then what its end calls for."""
    lock = os.open(card.state / "launcher.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock)
        return 0  # another launcher runs the pair
    os.umask(0o077)
    (card.state / UNCONFIRMED).unlink(missing_ok=True)  # a new start, whose stop has not failed yet
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # a core dump of either process could hold prompts
    (card.state / "logs").mkdir(exist_ok=True)
    log.setup(JsonLines(card.state / "logs/card.jsonl", CARD_LOG_BYTES))
    pid = card.state / "launcher.pid"
    pid.write_text(f"{os.getpid()}\n")
    try:
        config = json.dumps(card.config())
        with os.fdopen(private(card.state / "gateway.json", os.O_WRONLY | os.O_TRUNC), "w") as file:
            file.write(config)
        asyncio.run(serve(card))
    finally:
        pid.unlink(missing_ok=True)
    return 0


async def serve(card: Card) -> None:
    ended = asyncio.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(signum, ended.set)
    await launch(card, ended, vast.from_environment(card.credential))


async def launch(card: Card, ended: asyncio.Event, stop: vast.Stop) -> str:
    """Run the pair once, then stop the instance unless a signal ended it. Returns how the pair ended, or `given_up`
    when the card had given up and no retry came."""
    pair = card.pair()
    outcome = await retried(card.state / GIVEN_UP, ended)
    if outcome is None and (reasons := card.unfit()):
        # A copy of the KV cache that this container cannot hold would fail vLLM at its start, after the minutes of
        # its load: nothing is started, and the card gives up as after a failed load.
        for reason in reasons:
            log.row("unfit", code=reason)
        outcome = "unfit"
    if outcome is None:
        try:
            outcome = await pair.run(card.engine(), card.gateway(), float(card.manifest["LOAD_DEADLINE_S"]), ended)
        except OSError:  # a process that could not start
            outcome = "spawn_failed"
        if outcome != "signal" and pair.asleep:
            outcome = "asleep"  # the gateway had begun to stop the instance
    log.row("pair_end", reason=outcome)
    if outcome != "signal":
        if outcome not in ("asleep", "given_up") and not pair.ready.is_set():
            (card.state / GIVEN_UP).write_text(f"{outcome}\n")
        await first(vast.stop_until_accepted(stop, partial(unconfirmed, card.state)), ended.wait())
    return outcome


async def retried(marker: Path, ended: asyncio.Event) -> str | None:
    """None when the pair may run: at once on a card that has not given up, or once the owner's --retry has removed
    `marker` within the idle interval. Otherwise `given_up` at the interval's end, or `signal`."""
    if not marker.exists():  # noqa: ASYNC240 - one stat of a local file, in the launcher's own process
        return None
    log.row("wait_for_retry")
    return ("signal", "given_up", None)[await first(ended.wait(), asyncio.sleep(IDLE_TIMEOUT_S), removed(marker))]


async def removed(path: Path) -> None:
    while path.exists():  # noqa: ASYNC240 - as above, every POLL_S
        await asyncio.sleep(POLL_S)


def unconfirmed(state: Path, failure: Mapping[str, Any]) -> None:
    """Note a failed attempt to stop the instance, the launcher's or the gateway's: until the launcher's next start,
    the card's stop is not confirmed and costs may go on."""
    if not (state / UNCONFIRMED).exists():
        (state / UNCONFIRMED).write_text(f"{failure.get('code', 'exception')}\n")
        kept = ("code", "status", "exception")  # vast.py's fields of a failure
        log.row("stop_unconfirmed", **{name: failure[name] for name in kept if name in failure})


class Pair:
    """vLLM and the gateway, each in a process group of its own, their output read to the end."""

    def __init__(self, state: Path) -> None:
        self.state = state
        self.gateway_log = JsonLines(state / "logs/gateway.jsonl", GATEWAY_LOG_BYTES)
        self.ready = asyncio.Event()
        self.asleep = False
        self.noted: set[str] = set()  # the categories logged so far, each once

    async def run(self, engine: Command, gateway: Command, deadline_s: float, ended: asyncio.Event) -> str:
        """Run until a signal, the exit of either process or the load deadline, whichever comes first, and return
        which; then end both, the gateway first, and read the rest of their output."""
        started = asyncio.get_running_loop().time()
        processes: dict[str, asyncio.subprocess.Process] = {}
        readers = []
        try:
            for name, command, handle in (("engine", engine, self.engine_line),
                                          ("gateway", gateway, self.gateway_line)):
                processes[name] = await spawn(command)
                readers.append(asyncio.create_task(read(processes[name], handle)))
            which = await first(ended.wait(), processes["engine"].wait(), processes["gateway"].wait(),
                                self.load(deadline_s, started))
            return ("signal", "engine_exit", "gateway_exit", "load_deadline")[which]
        finally:
            for name, grace_s in (("gateway", GATEWAY_GRACE_S), ("engine", ENGINE_GRACE_S)):
                if name in processes:
                    await end(processes[name], grace_s)
                    log.row("exit", process=name, exit_code=processes[name].returncode)
            for reader in readers:  # the rest of the output, such as the lines before an exit
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(reader, GATEWAY_GRACE_S)

    async def load(self, deadline_s: float, started: float) -> None:
        """Return at the deadline if the gateway is not ready by then. Once it is ready, never return."""
        try:
            async with asyncio.timeout(deadline_s):
                await self.ready.wait()
        except TimeoutError:
            return
        log.row("ready", load_ms=round((asyncio.get_running_loop().time() - started) * 1000))
        await asyncio.Future()

    def engine_line(self, line: bytes) -> None:
        """Retain bounded measurements and diagnostic codes, failure/warning categories, or nothing."""
        text = engine_message(line[:LINE_BYTES].decode(errors="replace"))
        if text.startswith("non-default args: "):
            for code, numbers in non_default_args(text):
                log.row("engine_choice", code=code, **numbers)
            return
        fields = {name: value for pattern in MEASUREMENTS if (found := pattern.search(text))
                  for name, value in found.groupdict().items()}  # vLLM prints some of them on one line
        if fields:
            numbers = {name: number for name, value in fields.items()
                       if (number := diagnostic_number(value.replace(",", ""), SCALE[name.split("_")[-1]],
                                                       signed=name == "kv_cache_mib")) is not None}
            if numbers:
                log.row("engine_measure", **numbers)
            return
        if chosen := engine_choices(text):
            for code, numbers in chosen:
                log.row("engine_choice", code=code, **numbers)
            return
        lowered = text.lower()
        for event, categories in (("engine_failure", FAILURES), ("engine_warning", WARNINGS)):
            category = next((name for name, words in categories if any(word in lowered for word in words)), None)
            if category is not None and category not in self.noted:
                self.noted.add(category)
                log.row(event, code=category)

    def gateway_line(self, line: bytes) -> None:
        row = gateway_row(line)
        if row is None:
            log.row("gateway_output")
            return
        with contextlib.suppress(OSError):  # a full disk must not stop the reading, or the gateway would block
            self.gateway_log.write(line.decode().rstrip("\n") + "\n")
        if row.get("event") == "engine" and row.get("service_status") == "ready":
            self.ready.set()
        if row.get("event") == "stop_failed":  # the gateway's own stop, once it has fallen asleep
            unconfirmed(self.state, row)
        self.asleep = self.asleep or row.get("event") == "sleep"


def engine_message(text: str) -> str:
    """Remove only the usual vLLM log prefix, keeping message matching anchored outside arbitrary quoted values."""
    return ENGINE_PREFIX.sub("", text.strip(), count=1)


def diagnostic_number(value: str | float, factor: int = 1, *, signed: bool = False) -> int | None:
    """Convert a small finite measurement to a bounded integer; malformed or extreme engine output is ignored."""
    if isinstance(value, bool) or (isinstance(value, str) and len(value) > 128):
        return None
    try:
        number = float(value) * factor
        if math.isfinite(number) and abs(number) <= DIAGNOSTIC_MAX and (signed or number >= 0):
            return round(number)
    except (ValueError, OverflowError):
        return None
    return None


def engine_choices(text: str) -> list[tuple[str, dict[str, int]]]:
    """Recognized diagnostic messages, with arbitrary words replaced by other and no raw log content retained."""
    if len(text) > LINE_BYTES:
        return []
    text = engine_message(text)
    if text.startswith("non-default args: "):
        return non_default_args(text)
    if found := re.fullmatch(r"Using (\S+) for NVFP4 GEMM\.?", text):
        return [(f"nvfp4_gemm_{NVFP4_KERNELS.get(found[1], 'other')}", {})]
    if found := re.fullmatch(r"Applied linear backend override for '([^']+)': '([^']+)'\.?", text):
        return [(f"linear_override_{known(found[1], LINEAR_QUANTS)}_{known(found[2], LINEAR_BACKENDS)}", {})]
    if found := re.fullmatch(r"Using AttentionBackendEnum\.(\w+) backend\.", text):
        return [(f"attention_selected_{known(found[1], ATTENTION_BACKENDS)}", {})]
    if found := re.fullmatch(r"Using (\w+) attention backend out of potential backends: .+\.", text):
        return [(f"attention_chosen_{known(found[1], ATTENTION_BACKENDS)}", {})]
    if found := re.fullmatch(r"FlashInfer XQA decode does not support head_dim=(\d+); "
                            r"reverting this KV-cache group to native FlashInfer decode\.", text):
        number = diagnostic_number(found[1])
        return [("flashinfer_xqa_reverted", {"head_dim": number})] if number is not None else []
    if found := re.fullmatch(r"FlashInfer resolved query dtypes: prefill=\S+, decode=\S+, "
                            r"decode_backend=([\w-]+), kv_cache_dtype=\S+, arch=\S+", text):
        return [(f"flashinfer_decode_{DECODE_PATHS.get(found[1], 'other')}", {})]
    if text == "Using V2 Model Runner":
        return [("model_runner_v2", {})]
    if re.fullmatch(r"Model Runner V2 does not yet support .+; using the V1 model runner instead\.", text):
        return [("model_runner_v1", {})]
    if found := re.fullmatch(rf"Graph capturing finished in (\d+) secs, took ({NUMBER}) GiB", text):
        capture_ms, graph_mib = diagnostic_number(found[1], 1000), diagnostic_number(found[2], 1024)
        return ([("graphs_captured", {"capture_ms": capture_ms, "graph_mib": graph_mib})]
                if capture_ms is not None and graph_mib is not None else [])
    if found := re.fullmatch(r"Creating offloading spec with name: (\S+)", text):
        return [(f"kv_offloading_{OFFLOADING_SPECS.get(found[1], 'other')}", {})]
    return []


def argument_fields(node: ast.AST | None, config: str = "") -> dict[str, ast.AST]:
    """Read direct fields of a literal dictionary or a named config's keyword-only repr; never evaluate a call."""
    pairs: list[tuple[str, ast.AST]]
    if isinstance(node, ast.Dict):
        pairs = [(key.value, value) for key, value in zip(node.keys, node.values, strict=True)
                 if isinstance(key, ast.Constant) and isinstance(key.value, str)]
        if len(pairs) != len(node.keys):
            return {}
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == config
          and not node.args and all(keyword.arg is not None for keyword in node.keywords)):
        pairs = [(keyword.arg, keyword.value) for keyword in node.keywords if keyword.arg is not None]
    else:
        return {}
    fields = dict(pairs)
    return fields if len(fields) == len(pairs) else {}  # duplicate keys are ambiguous


def argument_literal(node: ast.AST | None) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        value = argument_literal(node.operand)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return -value if isinstance(node.op, ast.USub) else value
    return node  # an unsupported representation cannot look like a scalar value


def non_default_args(text: str) -> list[tuple[str, dict[str, int]]]:
    """Requested settings from a bounded Python-expression dict, not selections. Quoted strings are never searched.

    Configs may be literal dicts or named calls with keyword fields. Unsupported reprs, including <Enum: value>,
    make the whole line unknown. This deliberately loses diagnostics rather than interpreting arbitrary repr text.
    """
    if len(text) > LINE_BYTES or not text.startswith("non-default args: "):
        return []
    try:
        root = ast.parse(text.removeprefix("non-default args: "), mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return []
    if sum(1 for _ in ast.walk(root)) > 512:
        return []
    fields = argument_fields(root)
    speculative = argument_fields(fields.get("speculative_config"), "SpeculativeConfig")
    chosen: list[tuple[str, dict[str, int]]] = []
    for group, is_speculative in ((fields, False), (speculative, True)):
        for key, (code, factor) in ARG_NUMBERS.items():
            if (key == "num_speculative_tokens") != is_speculative:
                continue
            value = argument_literal(group.get(key))
            if key == "prefix_cache_retention_interval" and key in group and value is None:
                chosen.append((f"{code}_none", {}))
            elif (isinstance(value, int | float) and not isinstance(value, bool)
                  and (number := diagnostic_number(value, factor)) is not None):
                chosen.append((code, {"value": number}))
        for key, words in ARG_WORDS.items():
            if (key in {"draft_sample_method", "rejection_sample_method"}) != is_speculative:
                continue
            value = argument_literal(group.get(key))
            if isinstance(value, str):
                chosen.append((f"arg_{key}_{known(value, words)}", {}))
        if is_speculative:
            chosen += [(f"arg_{key}", {}) for key in ARG_FLAGS if isinstance(group.get(key), ast.List | ast.Tuple)]
    kernel = argument_fields(fields.get("kernel_config"), "KernelConfig")
    for quant, node in argument_fields(kernel.get("linear_backend_per_quant")).items():
        backend = argument_literal(node)
        if isinstance(backend, str):
            chosen.append((f"arg_linear_{known(quant, LINEAR_QUANTS)}_{known(backend, LINEAR_BACKENDS)}", {}))
    attention = argument_fields(fields.get("attention_config"), "AttentionConfig")
    for kind, node in argument_fields(attention.get("backend_per_kind")).items():
        backend = argument_literal(node)
        if isinstance(backend, str):
            chosen.append((f"arg_attention_{known(kind, CACHE_KINDS)}_{known(backend, ATTENTION_BACKENDS)}", {}))
    return chosen


def known(word: str, words: frozenset[str]) -> str:
    return next((candidate.lower() for candidate in words if word.isascii() and word.lower() == candidate.lower()),
                "other")


def gateway_row(line: bytes) -> dict[str, Any] | None:
    """A row the gateway's log wrote, or None. The gateway's other output, such as a traceback, may quote a request."""
    try:
        row = json.loads(line.decode())
    except (ValueError, RecursionError):
        return None
    if not isinstance(row, dict) or not row.keys() <= GATEWAY_KEYS:
        return None
    return row if all(value is None or isinstance(value, str | int | float) for value in row.values()) else None


async def spawn(command: Command) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *command.argv, cwd=command.cwd, env=command.env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True, limit=LINE_BYTES)


async def read(process: asyncio.subprocess.Process, handle: Callable[[bytes], None]) -> None:
    """Read a process's output to its end, a line at a time. Of a line over LINE_BYTES, `handle` gets the first
    LINE_BYTES, and the rest is dropped."""
    assert process.stdout is not None
    starts = True  # whether the next piece read starts a line
    while True:
        try:
            piece = await process.stdout.readuntil(b"\n")
        except asyncio.IncompleteReadError as error:  # the last line, without its newline, or nothing
            piece = error.partial
        except asyncio.LimitOverrunError as error:  # a part of a long line, as much as has arrived
            piece = await process.stdout.read(error.consumed)
        if not piece:
            return
        if starts:
            handle(piece[:LINE_BYTES])
        starts = piece.endswith(b"\n")


async def end(process: asyncio.subprocess.Process, grace_s: float) -> None:
    """End a process and its group, which may outlive it: SIGTERM, then SIGKILL after `grace_s`."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(grace_s):
            await process.wait()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def first(*awaitables: Awaitable[Any]) -> int:
    """Wait for the first of several to end, cancel the others, and return the index of the first."""
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        return next(index for index, task in enumerate(tasks) if task in done)
    finally:
        for task in tasks:
            task.cancel()


class JsonLines:
    """A log file of the card, 0600 once written to. Before a line would take it past `max_bytes`, it becomes
    `<name>.1`, whose mode stays as it was."""

    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes

    def write(self, text: str) -> None:
        data = text.encode()
        with contextlib.suppress(FileNotFoundError):
            if self.path.stat().st_size + len(data) > self.max_bytes:
                self.path.replace(self.path.with_name(self.path.name + ".1"))
        descriptor = private(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(descriptor, data)
        finally:
            os.close(descriptor)

    def flush(self) -> None:
        pass


def private(path: Path, flags: int) -> int:
    """Open a file that its owner alone may read. The mode is set on an older file too, which umask leaves as it was."""
    descriptor = os.open(path, flags | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    os.fchmod(descriptor, 0o600)
    return descriptor


def inspect(request: Mapping[str, Any], state: Path = STATE, root: Path = ROOT, proc: Path = PROC,
            running: bytes = RUNNING) -> dict[str, Any]:
    """The smoke's report of the card: how many launchers run and how many processes they started, the trial guard's
    deadline, and of each log its rows and, for a `marker`, the times it occurs. With `since`, also the cancelled flag
    of each chat request that gateway.jsonl logged after its first `since` rows. No line of a log leaves the card."""
    marker = request.get("marker")
    started = launchers(proc, running)
    logs = {name: {"rows": 0, **({"found": 0} if marker else {})} for name in (*LOGS, "other")}
    for path in sorted((state / "logs").rglob("*")):
        if path.is_file() and not path.is_symlink():
            rows, found = scan(path, marker)
            entry = logs[path.name if path.parent == state / "logs" and path.name in LOGS else "other"]
            entry["rows"] += rows
            if marker:
                entry["found"] += found
    report: dict[str, Any] = {"launchers": len(started), "children": sum(started.values()),
                              "deadline": trial_deadline(root), "now": int(time.time()), "logs": logs}
    if "since" in request:
        report["cancelled"] = cancelled_after(state / "logs/gateway.jsonl", request["since"])
    return report


def launchers(proc: Path = PROC, running: bytes = RUNNING) -> dict[int, int]:
    """The launchers that run, each with the number of its processes that run: the pair's two."""
    found: dict[int, int] = {}
    parents: dict[int, int] = {}
    entries = []
    with contextlib.suppress(OSError):
        entries = list(proc.iterdir())
    for entry in entries:
        if entry.name.isdigit():
            with contextlib.suppress(OSError, ValueError, IndexError):  # a process that ended while it was read
                if running in (entry / "cmdline").read_bytes():
                    found[int(entry.name)] = 0
                # The name in parentheses may hold spaces and parentheses; the parent is the second field after it.
                parents[int(entry.name)] = int((entry / "stat").read_bytes().rsplit(b")", 1)[1].split()[1])
    for parent in parents.values():
        if parent in found:
            found[parent] += 1
    return found


def trial_deadline(root: Path = ROOT) -> int | None:
    """The trial guard's deadline in seconds since the epoch, as trial-onstart.sh wrote it, or None."""
    try:
        text = (root / DEADLINE).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None
    return int(text) if re.fullmatch(r"[1-9][0-9]{0,11}", text) else None


def scan(path: Path, marker: str | None) -> tuple[int, int]:
    """The lines of a file, and the times `marker` occurs in it, one piece at a time."""
    needle, tail, rows, found = (marker or "").encode(), b"", 0, 0
    with contextlib.suppress(OSError), path.open("rb") as file:
        while piece := file.read(1 << 20):
            rows += piece.count(b"\n")
            if needle:
                found += (tail + piece).count(needle)  # the tail is too short to hold the marker whole
                tail = (tail + piece)[1 - len(needle):]
    return rows, found


def cancelled_after(path: Path, since: int) -> list[bool] | None:
    """The cancelled flag of each chat request that the gateway logged after the file's first `since` rows, and
    nothing else of the row; None when the file has fewer rows, since it was moved aside."""
    try:
        lines = path.read_bytes().split(b"\n")[:-1]  # without a last line that is still being written
    except FileNotFoundError:
        lines = []
    if len(lines) < since:
        return None
    rows = (gateway_row(line) for line in lines[since:])
    return [row.get("cancelled") is True for row in rows
            if row is not None and row.get("event") == "request" and row.get("route") == "/v1/chat/completions"]


def inspect_command(state: Path = STATE, running: bytes = RUNNING) -> int:
    """--inspect: one request on standard input, `{"marker": ..., "since": ...}` with either or neither, and one line
    of report on standard output."""
    try:
        request = json.loads(sys.stdin.buffer.readline(REQUEST_BYTES))
    except (ValueError, RecursionError):
        return 2
    if not isinstance(request, dict) or not request.keys() <= {"marker", "since"}:
        return 2
    marker, since = request.get("marker"), request.get("since", 0)
    if (marker is not None and not (isinstance(marker, str) and MARKER.fullmatch(marker))) or type(since) is not int \
            or since < 0:
        return 2
    print(json.dumps(inspect(request, state, running=running)), flush=True)
    return 0


def main(argv: list[str] | None = None, lane: type[Card] = Card) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {lane.module}",
                                     description=(sys.modules[lane.__module__].__doc__ or "").split("\n\n")[0])
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--hold", action="store_true", help="wait while the pair runs: the remote command of a tunnel")
    options.add_argument("--stop", action="store_true", help="end the pair, and leave the instance running")
    options.add_argument("--retry", action="store_true", help="remove given-up and start")
    options.add_argument("--dry-run", action="store_true", help="print the two commands and check; start nothing")
    options.add_argument("--inspect", action="store_true",
                         help="the smoke's report: one JSON request on standard input, one line of numbers and flags")
    options.add_argument("--run", action="store_true", help=argparse.SUPPRESS)  # the launcher that start() spawns
    args = parser.parse_args(argv)
    if args.stop:  # before the manifest is read, which ending the pair does not need
        return stop_pair(lane.home, running=lane.running)
    if args.inspect:  # as well
        return inspect_command(lane.home, lane.running)
    card = lane.load()
    if args.hold:
        return hold(card)
    if args.run:
        return run(card)
    if args.retry:
        (card.state / GIVEN_UP).unlink(missing_ok=True)
        if launcher(card.state, card.running) is not None:
            print(f"simple-serving card: a launcher runs. If it still waits for a retry, it runs the pair within "
                  f"{POLL_S} seconds; if it already stops the instance, the retry is deferred to the next start. up "
                  "tells which.", flush=True)
            return 0
    return start(card, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

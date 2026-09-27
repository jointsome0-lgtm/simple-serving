"""The microbenchmarks of the cheap card (docs/speed.md, the runbook's "The cheap card", steps 2 and 3): the FP4 GEMMs
through vLLM 0.30.0's own W4A16 kernel classes, and attention through the kernels vLLM calls, on synthetic tensors at
the 31B's shapes. A script for the card's vLLM venv, which has torch, vLLM and FlashInfer, run by its path:

    $STATE/vllm/bin/python simple_serving/throughput/microbench.py gemm --out RESULT.json
    $STATE/vllm/bin/python simple_serving/throughput/microbench.py attention --out RESULT.json

It needs the GPU to itself, so `python -m simple_serving.throughput microbench` ends the launcher's pair first.
`--dry-run` prints the plan, every shape, row count, case and kernel, as JSON, and checks the shapes against the 31B's
dimensions; it imports neither torch nor vLLM, and it is all of this script that runs off the card.

Step 2 builds each of the six linears as vLLM's ModelOpt W4A16 layer holds it before its kernel prepares it (packed
weights of N by K/2 bytes, low nibble first, fp8 scales of 16 values, a global scale), lets Marlin, Humming or
CuTe-DSL prepare it, and times a call under a CUDA graph at each row count: 1 to 8 requests, the verification rows of
1 to 8 requests with 2 to 5 drafts, and prompt chunks up to 1024. The graph cycles through copies of the weights, at
least twice the L2 cache, so that a call reads its weights from memory, as 60 layers in a row do. Each cell keeps its
time, its first call (where a kernel compiles), the largest error against the same weights dequantized to bfloat16,
and the count of outputs that are not finite; CuTe-DSL is tuned first as vLLM's warmup tunes it. Activations of
30000 and more, which CuTe-DSL turns into fp16, run through all three.

Step 3 times Triton's unified attention as TritonAttentionImpl.forward calls it (an fp8 query, an fp8 cache), at the
full layers' shape and the sliding layers', FlashInfer's XQA at the sliding shape as vLLM's FlashInfer backend calls
it, and FlashInfer's native decoding at head 512, each with the workspace it is given and the memory its first call
takes on top.

A result holds times, sizes, errors, counts and fixed words: never a tensor, and an error only by its class's name.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# The 31B's dimensions (config.json of the pinned checkpoint, docs/speed.md's table).
HIDDEN, INTERMEDIATE, HEADS = 5376, 21504, 32
LAYERS = {  # the two kinds of attention layer, with their count and their cache's block
    "sliding": {"kv_heads": 16, "head": 256, "block": 16, "window": 1024, "layers": 50},
    "full": {"kv_heads": 4, "head": 512, "block": 32, "window": 0, "layers": 10},
}


def _qkv(kind: str) -> tuple[int, ...]:
    layer = LAYERS[kind]
    return HEADS * layer["head"], layer["kv_heads"] * layer["head"], layer["kv_heads"] * layer["head"]


# Step 2: each NVFP4 linear as K, the parts vLLM fuses into N, and the layers that have it.
LINEARS: dict[str, tuple[int, tuple[int, ...], int]] = {
    "qkv_sliding": (HIDDEN, _qkv("sliding"), LAYERS["sliding"]["layers"]),
    "qkv_full": (HIDDEN, _qkv("full"), LAYERS["full"]["layers"]),  # K's weights fill V's slot too (k_eq_v)
    "o_sliding": (HEADS * LAYERS["sliding"]["head"], (HIDDEN,), LAYERS["sliding"]["layers"]),
    "o_full": (HEADS * LAYERS["full"]["head"], (HIDDEN,), LAYERS["full"]["layers"]),
    "gate_up": (HIDDEN, (INTERMEDIATE, INTERMEDIATE), 60),
    "down": (INTERMEDIATE, (HIDDEN,), 60),
}
GEMM_KERNELS = {  # vLLM 0.30.0's W4A16 kernels: module and class (vllm/model_executor/kernels/linear/nvfp4/)
    "marlin": ("vllm.model_executor.kernels.linear.nvfp4.marlin", "MarlinNvFp4LinearKernel"),
    "humming": ("vllm.model_executor.kernels.linear.nvfp4.humming", "HummingNvFp4LinearKernel"),
    "cutedsl": ("vllm.model_executor.kernels.linear.nvfp4.flashinfer", "FlashInferCuteDslNvFp4W4A16LinearKernel"),
}
DECODE_ROWS = tuple(range(1, 9))  # 1 to 8 requests of one token
VERIFY_ROWS = tuple(sorted({requests * (drafts + 1) for requests in range(1, 9) for drafts in range(2, 6)}))
CHUNK_ROWS = (64, 128, 256, 512, 1024)  # prompt chunks up to MAX_NUM_BATCHED_TOKENS
GROUP = 16  # values a scale covers
GLOBAL_SCALE = 1 / 128  # outputs of about unit size at these K
LARGE = (30000.0, 60000.0, 120000.0)  # fp16's largest value is 65504
LARGE_ROWS = (1, 8, 64)
LARGE_EVERY = 64  # one column in this many gets the large value, the others stay normal
TUNE_TOKENS = 1024  # vLLM's warmup tunes FlashInfer at MAX_NUM_BATCHED_TOKENS
L2_TIMES = 2  # the weight copies a graph cycles through hold at least this many times the L2 cache
MAX_COPIES = 8

# Step 3: the cases, each a kernel, a layer kind and the query tokens of a request (1, or a verification of 3 to 5
# drafts), at each context and each count of requests.
ATTENTION = (
    ("triton", "full", (1, 4, 5, 6)),
    ("triton", "sliding", (4, 5)),
    ("xqa", "sliding", (4, 5)),
    ("native", "full", (1,)),
)
CONTEXTS = (4000, 16000, 40000)
BATCHES = (1, 4, 8)  # one request, half and all of MAX_NUM_SEQS
MAX_MODEL_LEN = 65536
SEQ_THRESHOLD_3D = 8  # 128 // the model's 16 KV heads, a graph size (vllm/v1/attention/backends/triton_attn.py:139-158)
SEGMENTS = 16  # NUM_PAR_SOFTMAX_SEGMENTS
WORKSPACE_BYTES = 394 << 20  # VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE's default (vllm/envs.py:220)
PREFILL_BYTES_PER_ELEMENT = 16  # vllm/v1/attention/backends/flashinfer.py:95

WARMUP, ITERATIONS, REPEATS = 3, 20, 5
SEED = 7
GIB = 1 << 30


def gemm_rows() -> list[int]:
    return sorted({*DECODE_ROWS, *VERIFY_ROWS, *CHUNK_ROWS})


def attention_cases() -> list[dict[str, Any]]:
    return [{"kernel": kernel, "layer": layer, "q": q, "context": context, "batch": batch}
            for kernel, layer, queries in ATTENTION for q in queries for context in CONTEXTS for batch in BATCHES]


def kv_bytes(layer: str, context: int, q: int, batch: int) -> int:
    """The case's fp8 cache: every block of each request's context and queries, K and V."""
    spec = LAYERS[layer]
    return batch * math.ceil((context + q) / spec["block"]) * spec["block"] * spec["kv_heads"] * spec["head"] * 2


def gemm_plan(kernels: list[str]) -> dict[str, Any]:
    shapes = [{"name": name, "n": sum(parts), "k": k, "parts": list(parts), "layers": layers,
               "weight_bytes": sum(parts) * k // 2 + sum(parts) * k // GROUP}
              for name, (k, parts, layers) in LINEARS.items()]
    rows = gemm_rows()
    return {"kind": "gemm", "kernels": kernels, "shapes": shapes, "rows": rows,
            "large": {"values": list(LARGE), "rows": list(LARGE_ROWS), "every": LARGE_EVERY},
            "cells": len(kernels) * len(shapes) * (len(rows) + len(LARGE) * len(LARGE_ROWS)),
            "copies": {"l2_times": L2_TIMES, "most": MAX_COPIES}, "tune_tokens": TUNE_TOKENS,
            "timing": {"warmup": WARMUP, "iterations": ITERATIONS, "repeats": REPEATS, "graphs": True}}


def attention_plan(kernels: list[str]) -> dict[str, Any]:
    cases = [case | {"kv_mib": round(kv_bytes(case["layer"], case["context"], case["q"], case["batch"]) / (1 << 20))}
             for case in attention_cases() if case["kernel"] in kernels]
    return {"kind": "attention", "kernels": kernels, "heads": HEADS, "layers": LAYERS, "cases": cases,
            "kv_cache": "fp8_e4m3", "seq_threshold_3d": SEQ_THRESHOLD_3D,
            "timing": {"warmup": WARMUP, "iterations": ITERATIONS, "repeats": REPEATS, "graphs": True}}


class PlanError(ValueError):
    """A plan that does not fit the 31B or the card. The message names the shape or case."""


def check(plan: dict[str, Any]) -> None:
    """That every shape is one the kernels take and every case fits the engine's limits. Raises PlanError."""
    if plan["kind"] == "gemm":
        for shape in plan["shapes"]:
            if shape["k"] % GROUP or shape["n"] % GROUP or shape["n"] != sum(shape["parts"]):
                raise PlanError(f"{shape['name']} is not a shape of 16-value groups")
        if max(plan["rows"]) > TUNE_TOKENS or min(plan["rows"]) < 1:
            raise PlanError("a row count outside 1 to MAX_NUM_BATCHED_TOKENS")
        return
    for case in plan["cases"]:
        if case["context"] + case["q"] > MAX_MODEL_LEN or case["batch"] > 8:
            raise PlanError(f"{case['kernel']} {case['layer']} beyond the engine's limits")
        if case["kv_mib"] * (1 << 20) > 4 * GIB:
            raise PlanError(f"{case['kernel']} {case['layer']} needs more cache than a 16 GB card spares")


# What follows needs the card: torch, vLLM and FlashInfer, imported only here.

def module(name: str) -> Any:
    """A module of the card's vLLM venv, imported only when a measurement runs."""
    return importlib.import_module(name)


def device_facts(torch: Any) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "capability": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
            "memory_mib": props.total_memory >> 20, "l2_mib": round(getattr(props, "L2_cache_size", 0) / (1 << 20), 1)}


def versions() -> dict[str, str]:
    found = {}
    for name in ("torch", "vllm", "flashinfer-python", "humming-kernels", "nvidia-cutlass-dsl", "triton"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "absent"
    return found


def timed(torch: Any, call: Callable[[], object]) -> tuple[float, bool]:
    """The median time of a call in microseconds, replayed from a CUDA graph, or eagerly if it cannot be captured."""
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    run: Callable[[], object] = call
    graphed = True
    try:
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            call()
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            call()
        run = graph.replay
    except Exception:  # noqa: BLE001 - the time is kept without a graph, and says so
        graphed = False
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(REPEATS):
        start.record()
        for _ in range(ITERATIONS):
            run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / ITERATIONS)
    return statistics.median(samples), graphed


def first_call(torch: Any, call: Callable[[], Any]) -> tuple[Any, float, int]:
    """A call's result, its time in milliseconds with whatever it compiles, and the memory it took on top in MiB."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    began = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, round((time.perf_counter() - began) * 1000, 1), (torch.cuda.max_memory_allocated() - before) >> 20


def significant(value: float) -> float:
    return float(f"{value:.4g}")


def errors(torch: Any, got: Any, expected: Any) -> dict[str, Any]:
    got, expected = got.float(), expected.float()
    finite = torch.isfinite(got)
    largest = (got - expected).abs()[finite].max() if bool(finite.any()) else None
    return {"max_error": None if largest is None else significant(float(largest)),
            "ref_max": significant(float(expected.abs().max())), "nonfinite": int((~finite).sum())}


def random_weights(torch: Any, n: int, k: int) -> tuple[Any, Any]:
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scale = (torch.rand(n, k // GROUP, device="cuda") * 0.75 + 0.25).to(torch.float8_e4m3fn)
    return packed, scale


def dequantized(torch: Any, packed: Any, scale: Any, chunk: int = 2048) -> Any:
    """The weights in bfloat16: each nibble (low first) as E2M1 times its group's scale times the global scale."""
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                         dtype=torch.float32, device=packed.device)
    rows, half = packed.shape
    out = torch.empty(rows, half * 2, dtype=torch.bfloat16, device=packed.device)
    for start in range(0, rows, chunk):
        part = packed[start:start + chunk].long()
        values = torch.stack((table[part & 0x0F], table[part >> 4]), dim=-1).reshape(part.shape[0], -1, GROUP)
        blocks = scale[start:start + chunk].float() * GLOBAL_SCALE
        out[start:start + chunk] = (values * blocks[..., None]).reshape(part.shape[0], -1).to(torch.bfloat16)
    return out


def fp4_layer(torch: Any, packed: Any, scale: Any, k: int, parts: tuple[int, ...]) -> Any:
    """A linear as ModelOpt's W4A16 scheme leaves it for its kernel (vllm/model_executor/layers/quantization/
    modelopt.py, KNvfp4Static and the linear method's create_weights)."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(torch.tensor(GLOBAL_SCALE, dtype=torch.float32, device="cuda"),
                                                   requires_grad=False)
    layer.input_size_per_partition, layer.output_size_per_partition = k, sum(parts)
    layer.output_partition_sizes, layer.logical_widths = list(parts), list(parts)
    layer.params_dtype, layer.has_bias = torch.bfloat16, False
    return layer


def activations(torch: Any, rows: int, k: int, large: float | None = None) -> Any:
    x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
    if large is not None:
        signs = torch.where(torch.arange(rows, device="cuda") % 2 == 0, 1.0, -1.0)[:, None]
        x[:, ::LARGE_EVERY] = (signs * large).to(torch.bfloat16)
    return x


def gemm_shape(torch: Any, name: str, kernel_class: Any, shape: str, l2_bytes: int) -> list[dict[str, Any]]:
    config = module("vllm.model_executor.kernels.linear.nvfp4.base").NvFp4LinearLayerConfig
    k, parts, _ = LINEARS[shape]
    n = sum(parts)
    copies = max(1, min(MAX_COPIES, math.ceil(L2_TIMES * l2_bytes / (n * k // 2 + n * k // GROUP))))
    packed, scale = random_weights(torch, n, k)
    reference = dequantized(torch, packed, scale)
    began = time.perf_counter()
    pairs = []
    for _ in range(copies):
        kernel, layer = kernel_class(config()), fp4_layer(torch, packed.clone(), scale.clone(), k, parts)
        kernel.process_weights_after_loading(layer)
        pairs.append((kernel, layer))
    torch.cuda.synchronize()
    head: dict[str, Any] = {"kernel": name, "shape": shape, "copies": copies,
                            "prepare_ms": round((time.perf_counter() - began) * 1000 / copies, 1)}
    kernel, layer = pairs[0]
    if name == "cutedsl":
        flashinfer = module("vllm.utils.flashinfer")
        buckets = tuple(flashinfer.flashinfer_get_hybrid_num_tokens_buckets(TUNE_TOKENS))
        x = activations(torch, TUNE_TOKENS, k)
        with flashinfer.autotune(tuning_buckets=buckets):
            _, head["tune_ms"], _ = first_call(torch, lambda: kernel.apply_weights(layer, x))
        head["tune_buckets"] = len(buckets)
    cells = []
    for rows in gemm_rows():
        x = activations(torch, rows, k)
        got, first_ms, _ = first_call(torch, lambda: kernel.apply_weights(layer, x))  # noqa: B023 - called at once
        agreement = errors(torch, got, torch.nn.functional.linear(x, reference))
        us, graphed = timed(torch, lambda: [pair[0].apply_weights(pair[1], x) for pair in pairs])  # noqa: B023
        cells.append(head | {"rows": rows, "us": round(us / copies, 2), "first_ms": first_ms, "graph": graphed}
                     | agreement)
        print(f"gemm {name} {shape} rows={rows} us={us / copies:.2f} first_ms={first_ms} "
              f"max_error={agreement['max_error']} nonfinite={agreement['nonfinite']}", flush=True)
    for value in LARGE:
        for rows in LARGE_ROWS:
            x = activations(torch, rows, k, value)
            got = kernel.apply_weights(layer, x)
            cells.append(head | {"rows": rows, "large": value}
                         | errors(torch, got, torch.nn.functional.linear(x, reference)))
    return cells


def gemm(out: Path, kernels: list[str]) -> int:
    torch = module("torch")
    torch.manual_seed(SEED)
    facts = device_facts(torch)
    result: dict[str, Any] = {"kind": "gemm", "device": facts, "versions": versions(), "cells": []}
    l2_bytes = int(facts["l2_mib"] * (1 << 20)) or 64 << 20
    for name in kernels:
        path, class_name = GEMM_KERNELS[name]
        try:
            kernel_class = getattr(module(path), class_name)
            supported, _ = kernel_class.is_supported()
        except Exception as error:  # noqa: BLE001 - kept by its class's name
            result["cells"].append({"kernel": name, "error": type(error).__name__})
            continue
        if not supported:
            result["cells"].append({"kernel": name, "error": "unsupported"})
            continue
        for shape in LINEARS:
            try:
                result["cells"] += gemm_shape(torch, name, kernel_class, shape, l2_bytes)
            except Exception as error:  # noqa: BLE001
                result["cells"].append({"kernel": name, "shape": shape, "error": type(error).__name__})
            torch.cuda.empty_cache()
            save(out, result)
    save(out, result)
    return 0


def kv_cache(torch: Any, layer: str, context: int, q: int, batch: int) -> tuple[Any, Any]:
    """A cache of random fp8 values, never NaN, as vLLM lays it out ([blocks, KV heads, block, K then V]), and each
    request's blocks in a shuffled table; block 0 stays unused, as vLLM's null block."""
    spec = LAYERS[layer]
    per_request = math.ceil((context + q) / spec["block"])
    blocks = batch * per_request + 1
    raw = torch.randint(0, 256, (blocks, spec["kv_heads"], spec["block"], 2 * spec["head"]), dtype=torch.uint8,
                        device="cuda")
    raw.bitwise_and_(0xBF)  # no exponent of all ones: no NaN, values within 1.875
    table = (torch.randperm(blocks - 1, device="cuda")[:batch * per_request] + 1).to(torch.int32)
    return raw.view(torch.float8_e4m3fn), table.view(batch, per_request)


def attention_call(torch: Any, kernel: str, layer: str, q: int, context: int, batch: int) -> tuple[Callable[[], Any],
                                                                                                    int]:
    """The case's call as vLLM makes it, and the bytes of the workspace it is given."""
    spec = LAYERS[layer]
    kv_heads, head, block, window = spec["kv_heads"], spec["head"], spec["block"], spec["window"]
    cache, table = kv_cache(torch, layer, context, q, batch)
    tokens, length = batch * q, context + q
    query = torch.randn(tokens, HEADS, head, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(tokens, HEADS, head, dtype=torch.bfloat16, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    scale = head ** -0.5
    if kernel == "triton":  # vllm/v1/attention/backends/triton_attn.py, TritonAttentionImpl.forward
        unified_attention = module("vllm.v1.attention.ops.triton_unified_attention").unified_attention
        key, value = cache.transpose(1, 2).split(head, dim=-1)
        one = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        padded = 1 << (head - 1).bit_length()
        buffers = (torch.empty(SEQ_THRESHOLD_3D, HEADS, SEGMENTS, padded, dtype=torch.float32, device="cuda"),
                   torch.empty(SEQ_THRESHOLD_3D, HEADS, SEGMENTS, dtype=torch.float32, device="cuda"),
                   torch.empty(SEQ_THRESHOLD_3D, HEADS, SEGMENTS, dtype=torch.float32, device="cuda"))
        fp8_query = query.to(torch.float8_e4m3fn)  # the layer's static query quantization, scale 1.0
        starts = torch.arange(0, tokens + 1, q, dtype=torch.int32, device="cuda")

        def triton() -> Any:
            return unified_attention(
                q=fp8_query, k=key, v=value, out=out, cu_seqlens_q=starts, max_seqlen_q=q, seqused_k=lengths,
                max_seqlen_k=length, softmax_scale=scale, causal=True,
                window_size=(window - 1, 0) if window else (-1, -1), block_table=table, softcap=0, q_descale=one,
                k_descale=one.expand(batch, kv_heads), v_descale=one.expand(batch, kv_heads),
                seq_threshold_3D=SEQ_THRESHOLD_3D, num_par_softmax_segments=SEGMENTS, softmax_segm_output=buffers[0],
                softmax_segm_max=buffers[1], softmax_segm_expsum=buffers[2])

        return triton, 0
    key, value = cache.split(head, dim=-1)  # vLLM's FlashInfer backend: HND views of the same cache
    if kernel == "xqa":  # vllm/v1/attention/backends/flashinfer.py, the XQA decode
        xqa = module("vllm.utils.flashinfer").flashinfer_xqa_batch_decode_with_kv_cache
        draft_mask = module("vllm.v1.attention.backends.flashinfer")._make_xqa_draft_block_mask
        workspace = torch.zeros(int(os.environ.get("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", WORKSPACE_BYTES)),
                                dtype=torch.uint8, device="cuda")
        mask = None if q == 1 else draft_mask(q, True, query.device).unsqueeze(0).expand(
            batch, -1, -1).contiguous()
        return (lambda: xqa(query=query, kv_cache=(key, value), workspace_buffer=workspace, block_tables=table,
                            seq_lens=lengths, max_seq_len=length, bmm1_scale=scale, bmm2_scale=1.0,
                            window_left=window - 1 if window else -1, out=out, sinks=None, kv_layout="HND",
                            q_len_per_req=q, mask=mask, q_cu_seq_lens=None)), workspace.numel()
    decoding = module("flashinfer").BatchDecodeWithPagedKVCacheWrapper  # the native decoding, one query token
    size = max(int(os.environ.get("VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", WORKSPACE_BYTES)),
               TUNE_TOKENS * HEADS * head * PREFILL_BYTES_PER_ELEMENT)
    workspace = torch.zeros(size, dtype=torch.uint8, device="cuda")
    per_request = table.shape[1]
    starts = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * per_request
    indices = table.flatten().contiguous()
    last = torch.full((batch,), (length - 1) % block + 1, dtype=torch.int32, device="cuda")
    wrapper = decoding(workspace, "HND", use_cuda_graph=True, use_tensor_cores=True, paged_kv_indptr_buffer=starts,
                       paged_kv_indices_buffer=indices, paged_kv_last_page_len_buffer=last)
    wrapper.plan(starts, indices, last, HEADS, kv_heads, head, block, pos_encoding_mode="NONE",
                 q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn, sm_scale=scale)
    return (lambda: wrapper.run(query, (key, value), k_scale=1.0, v_scale=1.0, out=out)), size


def attention(out: Path, kernels: list[str]) -> int:
    torch = module("torch")
    torch.manual_seed(SEED)
    result: dict[str, Any] = {"kind": "attention", "device": device_facts(torch), "versions": versions(), "cells": []}
    for case in attention_cases():
        if case["kernel"] not in kernels:
            continue
        try:
            call, workspace = attention_call(torch, **case)
            _, first_ms, peak_mib = first_call(torch, call)
            us, graphed = timed(torch, call)
            cell = case | {"us": round(us, 2), "first_ms": first_ms, "graph": graphed,
                           "workspace_mib": workspace >> 20, "first_call_mib": peak_mib}
            print(f"attention {case['kernel']} {case['layer']} q={case['q']} context={case['context']} "
                  f"batch={case['batch']} us={us:.2f} first_ms={first_ms}", flush=True)
        except Exception as error:  # noqa: BLE001 - kept by its class's name
            cell = case | {"error": type(error).__name__}
        result["cells"].append(cell)
        torch.cuda.empty_cache()
        save(out, result)
    return 0


def save(path: Path, result: dict[str, Any]) -> None:
    """The result so far, 0600, replaced whole."""
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(result, file, indent=1)
    os.replace(temporary, path)


KINDS: dict[str, tuple[tuple[str, ...], Callable[[list[str]], dict[str, Any]], Callable[[Path, list[str]], int]]] = {
    "gemm": (tuple(GEMM_KERNELS), gemm_plan, gemm),
    "attention": (tuple(dict.fromkeys(kernel for kernel, _, _ in ATTENTION)), attention_plan, attention),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="microbench.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=tuple(KINDS))
    parser.add_argument("--kernels", help="a comma-separated subset of the kind's kernels")
    parser.add_argument("--out", type=Path, help="the result's JSON file (not with --dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan as JSON; needs no GPU")
    args = parser.parse_args(argv)
    known, plan_of, measure = KINDS[args.kind]
    kernels = list(known) if not args.kernels else args.kernels.split(",")
    if unknown := [kernel for kernel in kernels if kernel not in known]:
        parser.error(f"unknown kernel {unknown[0]!r}; known: {', '.join(known)}")
    plan = plan_of(kernels)
    try:
        check(plan)
    except PlanError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(plan))
        return 0
    if args.out is None:
        parser.error("--out is needed to measure")
    return measure(args.out, kernels)


if __name__ == "__main__":
    sys.exit(main())

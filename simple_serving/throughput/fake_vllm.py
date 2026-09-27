"""A stand-in for the card's `vllm` in the throughput harness's dry run (`python -m simple_serving.throughput dry-run`),
never on a card.

`vllm serve MODEL ...` prints, from its own arguments, the lines of vLLM 0.30.0 that card.py's filter turns into
engine_choice rows: the `non-default args` dict as vLLM prints it, the NVFP4 kernel, the runner, the graphs and the RAM
copy. Then it serves the fake engine on --port until it is ended, each generation after a short pause so that the
requests of a pass overlap. `vllm bench serve ...` takes only the arguments the pinned bench takes (sweep.check), keeps
--max-concurrency generations at once on the engine, and writes a result file with the bench's keys, its generated
texts among them, which the sweep must leave out.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from ..config import Listener
from ..fake_engine import FakeEngine, Script
from ..server import Servers, bind
from . import sweep

HOST = "127.0.0.1"
PAUSE_S = 0.2  # before a generation's first event, as for a prompt being read
DEFAULT_GPU_MEMORY = 0.92
KERNEL_CLASSES = {"marlin": "MarlinNvFp4LinearKernel", "humming": "HummingNvFp4LinearKernel",
                  "flashinfer_cutedsl": "FlashInferCuteDslNvFp4W4A16LinearKernel"}


def option(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv[:-1] else None


def lines(argv: list[str]) -> list[str]:
    """What vLLM would print of these arguments, in its own words."""
    model, port = argv[0], int(option(argv, "--port") or 8000)
    shown: dict[str, Any] = {"model_tag": model, "model": model, "host": HOST, "port": port,
                             "served_model_name": [option(argv, "--served-model-name")]}
    for flag, key in (("--max-model-len", "max_model_len"), ("--max-num-seqs", "max_num_seqs"),
                      ("--max-num-batched-tokens", "max_num_batched_tokens")):
        if (value := option(argv, flag)) is not None:
            shown[key] = int(value)
    if (value := option(argv, "--gpu-memory-utilization")) is not None and float(value) != DEFAULT_GPU_MEMORY:
        shown["gpu_memory_utilization"] = float(value)
    if (value := option(argv, "--kv-cache-dtype")) not in (None, "auto"):
        shown["kv_cache_dtype"] = value
    if (value := option(argv, "--kv-offloading-size")) is not None:
        shown["kv_offloading_size"] = float(value)
    for flag, key in (("--speculative-config", "speculative_config"), ("--kernel-config", "kernel_config"),
                      ("--attention-config", "attention_config")):
        if (value := option(argv, flag)) is not None:
            shown[key] = json.loads(value)
    if (value := option(argv, "--attention-backend")) is not None:
        shown["attention_backend"] = value
    backend = (shown.get("kernel_config") or {}).get("linear_backend_per_quant", {}).get("nvfp4_w4a16", "marlin")
    kernel = KERNEL_CLASSES.get(backend, "MarlinNvFp4LinearKernel")
    printed = [f"(APIServer pid={os.getpid()}) INFO 09-27 12:00:00 [utils.py:286] non-default args: {shown!r}",
               f"(EngineCore pid={os.getpid()}) INFO 09-27 12:00:05 [__init__.py:1136] Using {kernel} for NVFP4 GEMM",
               f"(EngineCore pid={os.getpid()}) INFO 09-27 12:00:05 [gpu_worker.py:441] Using V2 Model Runner"]
    if "kv_offloading_size" in shown:
        printed.append(f"(EngineCore pid={os.getpid()}) INFO 09-27 12:00:06 [factory.py:57] Creating offloading spec "
                       "with name: CPUOffloadingSpec")
    printed.append(f"(EngineCore pid={os.getpid()}) INFO 09-27 12:00:09 [model_runner.py:1067] Graph capturing "
                   "finished in 3 secs, took 0.50 GiB")
    return printed


async def serve(argv: list[str]) -> int:
    for line in lines(argv):
        print(line, flush=True)
    fake = FakeEngine(option(argv, "--served-model-name") or "dev-model", int(option(argv, "--max-model-len") or 65536))
    fake.script = Script(first_delay_s=PAUSE_S)
    engine = Servers([(fake, bind(Listener(HOST, int(option(argv, "--port") or 8000))))])
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    await engine.start()
    try:
        await stop.wait()
    finally:
        await engine.stop()
    return 0


async def bench(argv: list[str]) -> int:
    """The pinned bench's run, as far as the harness reads it: `--num-prompts` requests, `--max-concurrency` at once."""
    sweep.check(["vllm", "bench", "serve", *argv])
    url = f"http://{option(argv, '--host')}:{option(argv, '--port')}/v1/chat/completions"
    prompts, at = int(option(argv, "--num-prompts") or 0), int(option(argv, "--max-concurrency") or 1)
    places, began = asyncio.Semaphore(at), time.monotonic()
    body = {"model": option(argv, "--model"), "messages": [{"role": "user", "content": "A synthetic prompt."}],
            "max_tokens": 16, "stream": True}

    async def one(client: httpx.AsyncClient) -> float:
        async with places:
            started = time.monotonic()
            async with client.stream("POST", url, json=body) as response:
                async for _ in response.aiter_bytes():
                    pass
            return time.monotonic() - started

    async with httpx.AsyncClient(timeout=30) as client:
        times = await asyncio.gather(*(one(client) for _ in range(prompts)))
    duration = time.monotonic() - began
    output = 13 * prompts
    result = {
        "date": "20260927-120000", "backend": "openai", "model_id": option(argv, "--model"), "num_prompts": prompts,
        "duration": duration, "completed": prompts, "failed": 0, "total_input_tokens": prompts * 256,
        "total_output_tokens": output, "request_throughput": prompts / duration, "output_throughput": output / duration,
        "total_token_throughput": (output + prompts * 256) / duration, "max_concurrent_requests": at,
        "mean_ttft_ms": PAUSE_S * 1000, "median_ttft_ms": PAUSE_S * 1000, "p99_ttft_ms": PAUSE_S * 1000,
        "mean_tpot_ms": 1.0, "median_tpot_ms": 1.0, "p99_tpot_ms": 1.0,
        "mean_e2el_ms": 1000 * sum(times) / max(1, len(times)),
        "generated_texts": ["The keeper lights the lamp and watches the grey sea."] * prompts, "errors": [""] * prompts,
    }
    path = Path(option(argv, "--result-dir") or ".") / (option(argv, "--result-filename") or "result.json")
    path.write_text(json.dumps(result), encoding="utf-8")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["serve"]:
        return asyncio.run(serve(argv[1:]))
    if argv[:2] == ["bench", "serve"]:
        return asyncio.run(bench(argv[2:]))
    print("fake vllm: serve or bench serve", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""What the container has, against what the manifest asks of it, before anything is installed, fetched or loaded.

    python3 simple_serving/preflight.py check [--offload-gib N] [--min-compute-cap C --min-gpu-mib M]
                                              [--files name:bytes,... --into DIR]
    python3 simple_serving/preflight.py report
    python3 simple_serving/preflight.py plan --offload-gib N
    python3 simple_serving/preflight.py plan --sessions S --prompt-tokens T [--kib-per-token K]

card/bootstrap.sh runs `check` with the card's own python3, before a venv exists, so this file imports the standard
library alone and runs by its path; the launcher imports it for the same rule before it starts the pair (`card.py`).
`check` prints one line of numbers and returns 0, or 9 with the reasons in that line. `report` prints the same numbers
with nothing asked, and the largest KV_OFFLOADING_GIB that fits: the first command of a rental that means to use the
copy. `plan` prints what an offer must have for a size, for the renter's filter, which is not in this repository;
with --sessions and --prompt-tokens it first works the size out: what all the sessions write before one returns.

The copy of the KV cache in RAM (docs/speed.md, item 5). vLLM 0.30.0 keeps it in one file in /dev/shm, whose path is
fixed (vllm/v1/kv_offload/cpu/shared_offload_region.py:111), refuses a size above that mount's free space (:138 and
vllm/distributed/device_communicators/shm_broadcast.py:227-247), and faults every page in at the start (:211-235), so
the whole size is resident from the first minute and counts against the container's memory as shared memory. The
engine then needs its own memory beside it. Two numbers bound the size, and neither is in an offer: the free space of
/dev/shm and the container's memory limit. A third refusal is vLLM's own: PYTORCH_CUDA_ALLOC_CONF with
expandable_segments:True and any KV connector (vllm/config/vllm.py:1090-1109).

SIMPLE_SERVING_CARD_SHM, SIMPLE_SERVING_CARD_CGROUP and SIMPLE_SERVING_CARD_PROC move /dev/shm, /sys/fs/cgroup and
/proc, for tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

UNFIT = 9  # bootstrap.sh's exit code for a container that does not fit the manifest
MIB, GIB = 1 << 20, 1 << 30
# Left free in /dev/shm beside the copy: vLLM and PyTorch keep small segments of their own there.
SHM_MARGIN_MIB = 512
# The memory the engine and the gateway need beside the copy. Provisional: no rental has recorded the engine's
# resident memory. The weights are read through the page cache, which the kernel gives back; what stays is the CUDA
# context, the tokenizer, xgrammar's compiled schemas (VLLM_XGRAMMAR_CACHE_MB, 512 by default) and Python.
ENGINE_RESERVE_MIB = 12 * 1024
NO_LIMIT = 1 << 60  # a cgroup limit at or above this is no limit
# What a token of a prompt writes to the copy with the 31B under dense retention: 40 KiB in the global layers and 400
# in the sliding ones (docs/speed.md, item 5). Derived from the model's layers, not measured.
DENSE_KIB_PER_TOKEN = 440


def shm_free_mib(path: Path) -> int | None:
    try:
        return shutil.disk_usage(path).free // MIB
    except OSError:
        return None


def memory_limit_mib(cgroup: Path, proc: Path) -> int | None:
    """The container's memory limit: cgroup v2's, or v1's, and never above the machine's RAM."""
    total = None
    try:
        for line in (proc / "meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        total = None
    limit = None
    for name in ("memory.max", "memory/memory.limit_in_bytes"):
        try:
            text = (cgroup / name).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < NO_LIMIT:
            limit = int(text)
        break
    known = [value for value in (total, limit) if value is not None]
    return min(known) // MIB if known else None


def offload_reasons(gib: int, shm_mib: int | None, memory_mib: int | None, alloc_conf: str = "") -> list[str]:
    """Why a copy of `gib` GiB does not fit, by fixed codes; none when it fits. A number that cannot be read refuses
    the copy as well: a guess here costs the load and the stop of a rented card."""
    if not gib:
        return []
    reasons = []
    if shm_mib is None or shm_mib < gib * 1024 + SHM_MARGIN_MIB:
        reasons.append("shm_too_small")
    if memory_mib is None or memory_mib < gib * 1024 + ENGINE_RESERVE_MIB:
        reasons.append("memory_too_small")
    if "expandable_segments:true" in alloc_conf.replace(" ", "").lower():
        reasons.append("expandable_segments")
    return reasons


def largest_offload_gib(shm_mib: int | None, memory_mib: int | None) -> int:
    """The largest whole KV_OFFLOADING_GIB that passes `offload_reasons` here, or 0."""
    if shm_mib is None or memory_mib is None:
        return 0
    return max(0, min(shm_mib - SHM_MARGIN_MIB, memory_mib - ENGINE_RESERVE_MIB) // 1024)


def offer_needs(gib: int) -> dict[str, int]:
    """What an offer must have for a copy of `gib` GiB. Vast counts RAM in decimal megabytes, so `min_ram_gb` is
    rounded up from GiB; the offer does not show /dev/shm, which `report` reads once the container is up."""
    memory_mib = gib * 1024 + ENGINE_RESERVE_MIB
    return {"offload_gib": gib, "min_memory_mib": memory_mib, "min_ram_gb": math.ceil(memory_mib * MIB / 1e9),
            "min_shm_mib": gib * 1024 + SHM_MARGIN_MIB}


def copy_gib(sessions: int, prompt_tokens: int, kib_per_token: int) -> int:
    """The copy that holds every session's whole prompt as it is written, in whole GiB: sessions x tokens x the bytes
    a token writes. A smaller one drops a session's blocks before its second turn, and then gives nothing: 1 GiB did
    so where this rule asks 2 (Gemma 4 E2B, 2026-10-07; docs/speed.md, item 5)."""
    return math.ceil(sessions * prompt_tokens * kib_per_token * 1024 / GIB)


def gpu() -> tuple[int, int] | None:
    """The first GPU's compute capability times ten (120 is sm_120, Blackwell) and its memory in MiB."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30, check=False)
        capability, memory = done.stdout.splitlines()[0].split(",")
        return round(float(capability) * 10), int(memory)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_reasons(found: tuple[int, int] | None, min_compute_cap: int, min_gpu_mib: int) -> list[str]:
    if found is None:
        return ["gpu_unknown"]
    return [code for code, short in (("gpu_compute_cap_too_low", found[0] < min_compute_cap),
                                     ("gpu_memory_too_small", found[1] < min_gpu_mib)) if short]


def missing_bytes(files: str, into: Path) -> int:
    """The bytes still to fetch of `files`, each name:bytes, into `into`: a file in place, whole or in part, counts
    for what it holds."""
    need = 0
    for entry in files.split(","):
        name, _, size = entry.partition(":")
        held = 0
        for path in (into / name, into / f".{name}.part"):
            try:
                held = max(held, path.stat().st_size)
            except OSError:
                continue
        need += max(0, int(size) - held)
    return need


def disk_free(path: Path) -> int | None:
    """The free bytes of the disk that will hold `path`, which may not exist yet."""
    for candidate in (path, *path.parents):
        try:
            return shutil.disk_usage(candidate).free
        except OSError:
            continue
    return None


def measured(environ: dict[str, str] | os._Environ[str]) -> dict[str, int | None]:
    return {"shm_free_mib": shm_free_mib(Path(environ.get("SIMPLE_SERVING_CARD_SHM", "/dev/shm"))),
            "memory_limit_mib": memory_limit_mib(Path(environ.get("SIMPLE_SERVING_CARD_CGROUP", "/sys/fs/cgroup")),
                                                 Path(environ.get("SIMPLE_SERVING_CARD_PROC", "/proc")))}


def offload_unfit(gib: int, environ: dict[str, str] | os._Environ[str] = os.environ) -> list[str]:
    """The launcher's question: why a copy of `gib` GiB does not fit this container, or nothing."""
    if not gib:
        return []
    numbers = measured(environ)
    return offload_reasons(gib, numbers["shm_free_mib"], numbers["memory_limit_mib"],
                           environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="preflight.py", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("command", choices=("check", "report", "plan"))
    parser.add_argument("--offload-gib", type=int, default=0)
    parser.add_argument("--min-compute-cap", type=int)
    parser.add_argument("--min-gpu-mib", type=int, default=0)
    parser.add_argument("--files")
    parser.add_argument("--into", type=Path)
    parser.add_argument("--sessions", type=int, help="plan: sessions that take turns")
    parser.add_argument("--prompt-tokens", type=int, help="plan: tokens of a session's prompt at its longest")
    parser.add_argument("--kib-per-token", type=int, default=DENSE_KIB_PER_TOKEN,
                        help="plan: KiB a token writes to the copy; 440 is the 31B's under dense retention")
    args = parser.parse_args(argv)
    if args.offload_gib < 0:
        parser.error("--offload-gib is a count of GiB")
    if args.command == "plan":
        if (args.sessions is None) != (args.prompt_tokens is None):
            parser.error("--sessions goes with --prompt-tokens")
        if args.sessions is None:
            print(json.dumps(offer_needs(args.offload_gib)))
            return 0
        if args.offload_gib or min(args.sessions, args.prompt_tokens, args.kib_per_token) < 1:
            parser.error("--sessions, --prompt-tokens and --kib-per-token are counts, and stand in place of "
                         "--offload-gib")
        gib = copy_gib(args.sessions, args.prompt_tokens, args.kib_per_token)
        print(json.dumps({"sessions": args.sessions, "prompt_tokens": args.prompt_tokens,
                          "kib_per_token": args.kib_per_token} | offer_needs(gib)))
        return 0
    numbers, alloc_conf = measured(os.environ), os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    line: dict[str, object] = dict(numbers)
    reasons = offload_reasons(args.offload_gib, numbers["shm_free_mib"], numbers["memory_limit_mib"], alloc_conf)
    if args.command == "report":
        found = gpu()
        line |= {"largest_offload_gib": largest_offload_gib(numbers["shm_free_mib"], numbers["memory_limit_mib"]),
                 "expandable_segments": "expandable_segments" in offload_reasons(1, None, None, alloc_conf),
                 "gpu_compute_cap": found[0] if found else None, "gpu_mib": found[1] if found else None}
        print(json.dumps(line))
        return 0
    if args.offload_gib:
        line |= {name: value for name, value in offer_needs(args.offload_gib).items() if name != "min_ram_gb"}
    if args.min_compute_cap is not None:
        found = gpu()
        line |= {"gpu_compute_cap": found[0] if found else None, "gpu_mib": found[1] if found else None}
        reasons += gpu_reasons(found, args.min_compute_cap, args.min_gpu_mib)
    if args.files and args.into is not None:
        need, free = missing_bytes(args.files, args.into), disk_free(args.into)
        line |= {"need_disk_mib": math.ceil(need / MIB), "disk_free_mib": None if free is None else free // MIB}
        if free is None or free < need + GIB:  # a GiB over, for the venvs and the logs
            reasons.append("disk_too_small")
    print(json.dumps(line | {"ok": not reasons, "reasons": reasons}))
    return UNFIT if reasons else 0


if __name__ == "__main__":
    sys.exit(main())

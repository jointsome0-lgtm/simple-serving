"""Client-only benchmark through an existing text gateway. With no arguments, run the synthetic demonstration."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..cli import Refusal, read_config
from . import metrics
from .runner import Options, Target, run


class RunError(Exception):
    """A fixed code; never quote config, request or response text."""


def _base(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        raise RunError("loopback_url_required") from None
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise RunError("loopback_url_required")
    try:
        if parsed.port is None or parsed.port == 0:
            raise RunError("port_required")
    except ValueError:
        raise RunError("port_required") from None
    return value.rstrip("/")


def _annotate(report: dict[str, Any], monitor: metrics.Monitor) -> None:
    """Flag evidence limits rather than infer a GPU cause from client delivery pauses."""
    for trial in report.get("trials", []):
        before, after = trial.get("diagnostic_before"), trial.get("diagnostic_after")
        flags: list[str] = []
        if not before or not after:
            flags.append("metrics_missing")
        else:
            if before["missing"] or after["missing"]:
                flags.append("metrics_missing")
            gateway_before = {k: v for k, v in before["gateway"].items() if k.startswith("requests_total:")}
            gateway_after = {k: v for k, v in after["gateway"].items() if k.startswith("requests_total:")}
            gateway, reset = metrics.changes(gateway_before, gateway_after)
            engine_before = {k: v for k, v in before["engine"].items() if k in metrics.ENGINE_COUNTERS}
            engine_after = {k: v for k, v in after["engine"].items() if k in metrics.ENGINE_COUNTERS}
            engine, engine_reset = metrics.changes(engine_before, engine_after)
            trial["counter_delta"] = {"gateway": gateway, "engine": engine}
            if "num_preemptions_total" not in engine:
                flags.append("engine_diagnostics_incomplete")
            if reset or engine_reset or before["reset_suspected"] or after["reset_suspected"]:
                flags.append("reset_suspected")
            if engine.get("num_preemptions_total", 0) > 0:
                flags.append("preempted")
            expected = 1 + len(trial["aggressors"])
            if trial["case"] == "warm":
                expected += report["options"]["concurrency"] - 1
            if not gateway:
                flags.append("metrics_missing")
            elif sum(gateway.values()) > expected:
                flags.append("contaminated")
            elif sum(gateway.values()) < expected:
                flags.append("metrics_window_incomplete")
            window = [sample for sample in monitor.samples if before["at_ms"] <= sample["at_ms"] <= after["at_ms"]]
            if any(value > 0 for sample in window for key, value in sample["gateway"].items()
                   if key.startswith("generations_waiting:")):
                flags.append("admission_wait_observed")
        if monitor.boot_changed:
            flags.append("reset_suspected")
        if monitor.sampling_capped:
            flags.append("sampling_capped")
        trial["diagnostic_flags"] = sorted(set(flags))
        if flags and trial["status"] == "ok":
            trial["status"], trial["code"] = "inconclusive", "diagnostics_confounded"
    if any(trial["status"] == "inconclusive" for trial in report.get("trials", [])) and report["status"] == "ok":
        report["status"], report["code"] = "inconclusive", "diagnostics_confounded"


async def _experiment(public: str, control: str, key: str, control_key: str, options: Options,
                      max_seconds: float, synthetic: bool) -> dict[str, Any]:
    monitor: metrics.Monitor | None = None
    report: dict[str, Any] = {"schema_version": 1, "status": "failed", "code": "not_started"}
    try:
        async with httpx.AsyncClient(timeout=None, follow_redirects=False, trust_env=False) as client:
            try:
                async with asyncio.timeout(max_seconds):
                    raw = await metrics.read(client, control, "/v1/state", control_key)
                    try:
                        state = json.loads(raw)
                    except (ValueError, RecursionError):
                        raise RunError("state_invalid") from None
                    if not isinstance(state, dict) or state.get("status") != "ready":
                        raise RunError("target_not_ready")
                    alias, context, boot = state.get("model"), state.get("context_tokens"), state.get("boot_id")
                    if (not isinstance(alias, str) or not alias or len(alias) > 256 or type(context) is not int
                            or context < 1 or not isinstance(boot, str) or not boot or len(boot) > 256):
                        raise RunError("state_invalid")
                    if options.long_tokens + options.aggressor_tokens > context or options.victim_tokens >= context:
                        raise RunError("context_limit")
                    monitor = metrics.Monitor(client, control, control_key, boot)
                    initial = await monitor.snapshot()
                    if initial["missing"]:
                        raise RunError("diagnostics_unavailable")
                    metrics.capacity(initial, options.concurrency, options.aggressor_class)
                    monitor.start()
                    run_offset_ms = round((time.monotonic() - monitor.started) * 1000, 3)
                    report = await run(client, Target(public, key, alias), options, diagnostics=monitor.snapshot)
                    report["run_offset_ms"] = run_offset_ms
                    await monitor.check_boot()
                    _annotate(report, monitor)
            finally:
                if monitor is not None:
                    await monitor.close()
    except (RunError, metrics.DiagnosticError) as error:
        report["status"], report["code"] = "failed", str(error)
    except TimeoutError:
        report["status"], report["code"] = "failed", "budget_expired"
    except httpx.HTTPError:
        report["status"], report["code"] = "failed", "transport_error"
    report["synthetic"] = synthetic
    report["max_seconds"] = max_seconds
    if monitor is not None:
        report["diagnostic_samples"] = monitor.samples
    return report


async def _dry(options: Options, max_seconds: float) -> dict[str, Any]:
    from .fake import synthetic_gateway  # noqa: PLC0415 - only the explicit synthetic command starts a fake

    async with synthetic_gateway() as gateway:
        return await _experiment(gateway.url, gateway.url, gateway.key, gateway.key, options, max_seconds, True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("dry-run", "run"), default="dry-run")
    parser.add_argument("--config", type=Path, help="run: owner-only CLI configuration with client/control keys")
    parser.add_argument("--out", type=Path, help="new JSON report file; required for run, never overwritten")
    parser.add_argument("--public", default="http://127.0.0.1:8080")
    parser.add_argument("--control", default="http://127.0.0.1:8081")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--long-tokens", type=int)
    parser.add_argument("--victim-tokens", type=int)
    parser.add_argument("--aggressor-tokens", type=int)
    parser.add_argument("--concurrency", type=int, choices=(2, 4), default=2)
    parser.add_argument("--trigger-chunks", type=int)
    parser.add_argument("--aggressor-class", choices=("reader", "agent", "internal"), default="internal")
    parser.add_argument("--max-seconds", type=float, default=600, help="whole experiment budget, at most 1800 seconds")
    args = parser.parse_args(argv)
    if args.command == "run" and (args.config is None or args.out is None):
        parser.error("run requires --config and --out")
    if args.command == "dry-run" and args.config is not None:
        parser.error("--config goes with run")
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 1800:
        parser.error("--max-seconds must be finite, positive and at most 1800")
    defaults = asdict(Options())
    if args.command == "dry-run":
        defaults.update(rounds=1, long_tokens=2048, victim_tokens=32, trigger_chunks=4,
                        start_timeout_s=2, trial_timeout_s=5)
    for name in ("rounds", "long_tokens", "victim_tokens", "aggressor_tokens", "trigger_chunks"):
        if getattr(args, name) is not None:
            defaults[name] = getattr(args, name)
    defaults.update(concurrency=args.concurrency, aggressor_class=args.aggressor_class)
    if (not 1 <= defaults["rounds"] <= 10 or not 64 <= defaults["long_tokens"] <= 50000
            or not 1 <= defaults["victim_tokens"] <= 8192 or not 1 <= defaults["aggressor_tokens"] <= 8192
            or not 1 <= defaults["trigger_chunks"] <= 8192):
        parser.error("dimensions outside bounds: rounds 1..10, long tokens 64..50000, other counts 1..8192")
    options = Options(**defaults)
    output = None
    report: dict[str, Any]
    try:
        # Reserve a new owner-only report before any request could start.
        if args.out is not None:
            output = os.fdopen(os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")
        if args.command == "dry-run":
            report = asyncio.run(_dry(options, args.max_seconds))
        else:
            public, control = _base(args.public), _base(args.control)
            config = read_config(args.config)
            if not all(isinstance(config.get(name), str) and config[name] for name in ("client_key", "control_key")):
                raise RunError("config_keys_missing")
            if any(len(config[name]) > 512 or not all(33 <= ord(c) <= 126 for c in config[name])
                   for name in ("client_key", "control_key")):
                raise RunError("config_keys_invalid")
            report = asyncio.run(_experiment(public, control, config["client_key"], config["control_key"],
                                             options, args.max_seconds, False))
    except Refusal:
        report = {"status": "failed", "code": "config_invalid"}
    except RunError as error:
        report = {"status": "failed", "code": str(error)}
    except OSError:
        report = {"status": "failed", "code": "output_or_config_unavailable"}
    except KeyboardInterrupt:
        report = {"status": "failed", "code": "interrupted"}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    try:
        if output is not None:
            with output:
                output.write(text)
        else:
            print(text, end="")
    except OSError:
        with contextlib.suppress(OSError):
            os.write(2, b'{"status":"failed","code":"report_write_failed"}\n')
        return 2
    return 0 if report["status"] == "ok" else 2

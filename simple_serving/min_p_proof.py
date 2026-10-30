"""The proof on a card that a min_p filters drafted tokens, for MIN_P_WHILE_DRAFTING=1 (docs/runbook.md).

    python -m simple_serving.min_p_proof --until EPOCH [--plan] [--only STEP ...] [--card-dir DIR]

It asks vLLM directly on the card's loopback, since the gateway takes no logprobs, with synthetic prompts, while no
other request runs. Each step prints one JSON line of numbers and fixed words, never a prompt or an answer:

- runner: the launcher's last row of vLLM's runner says V2, and the manifest's switch is on.
- filter: each token's logprob against the position's best one. vLLM keeps a token when its probability after the
  temperature is at least min_p of the best, so exp((logprob - best) / temperature) >= min_p, on raw logprobs
  (vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:258). At min_p 0.08 no token may fall below that, while the
  control at min_p 0 must show such tokens: without them the step says nothing, and it fails as inconclusive.
- repeat: a repetition_penalty of 1.5 must lower the share of tokens that the answer already holds.
- schema: answers with a schema at min_p 0.08 and temperature 0.8, beside plain requests that come and go, so that
  batches with and without a schema alternate as the bot's do. Each must end in `stop` and follow it. A count of 0
  failures is an observation of vLLM's issue 54437, not a proof that it cannot happen: 0.30.0 has no fix for it.
- speed: tokens a second at min_p 0, 0.05 and 0.08 with two requests at once, and the drafts accepted. An observation
  that passes nothing.

A step starts only if its estimate fits before --until and vLLM is idle. Exit codes: 0 every step asked passed; 1 a
step failed; 2 vLLM cannot be read, runs other requests, or --until leaves too little.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from .card import STATE, min_p_while_drafting, read_manifest
from .sweep import parse_metrics

MIN_P, TEMPERATURE, PENALTY = 0.08, 0.8, 1.5
MARGIN = 1e-3  # of the threshold: logprobs arrive as rounded floats
ENOUGH_TOKENS, ENOUGH_BELOW = 2000, 5  # the least of a filter step that can say anything
SPEED_S, CALL_S = 150, 120
ESTIMATES = {"runner": 5, "filter": 180, "repeat": 60, "schema": 120, "speed": 3 * SPEED_S + CALL_S}
PROMPTS = ("Write a long story about a lighthouse keeper and a stray dog on a winter night.",
           "Опиши подробно один день смотрителя маяка и бродячей собаки, которая к нему прибилась.",
           "Write a long story about two sisters repairing an old boat before a storm.",
           "Опиши подробно, как две сестры чинят старую лодку перед штормом.")
REPEATING = "Write the words red, green and blue over and over, in any order, without stopping."
SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}, "people": {
    "type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                               "required": ["name", "age"], "additionalProperties": False}}},
          "required": ["title", "people"], "additionalProperties": False}
SCHEMA_PROMPT = "Invent a short title and three people for a story about a lighthouse, as JSON."


def below(entries: Sequence[dict[str, Any]], min_p: float, temperature: float) -> int:
    """How many tokens of an answer's logprobs a filter of min_p should have removed."""
    return sum(math.exp((entry["logprob"] - entry["top_logprobs"][0]["logprob"]) / temperature) < min_p * (1 - MARGIN)
               for entry in entries)


def repeated(tokens: Sequence[str]) -> int:
    """How many tokens the answer already held."""
    return len(tokens) - len(set(tokens))


def filtered(control: dict[str, int], armed: dict[str, int]) -> str:
    """The filter step's verdict: `inconclusive` unless the control shows what a filter would remove, and tokens were
    drafted in the arm that counts."""
    if min(control["tokens"], armed["tokens"]) < ENOUGH_TOKENS or control["below"] < ENOUGH_BELOW:
        return "inconclusive"
    if not armed["accepted"]:
        return "inconclusive"
    return "passed" if armed["below"] == 0 else "failed"


def follows(content: str) -> bool:
    """Whether an answer follows SCHEMA, at no depth beyond it."""
    try:
        answer = json.loads(content)
    except ValueError:
        return False
    return (isinstance(answer, dict) and answer.keys() == {"title", "people"} and isinstance(answer["title"], str)
            and isinstance(answer["people"], list)
            and all(isinstance(person, dict) and person.keys() == {"name", "age"} and isinstance(person["name"], str)
                    and type(person["age"]) is int for person in answer["people"]))


class Engine:
    def __init__(self, client: httpx.Client, port: str, alias: str) -> None:
        self.client, self.url, self.alias = client, f"http://127.0.0.1:{port}", alias
        self.calls = 0

    def metrics(self) -> dict[str, float]:
        answer = self.client.get(f"{self.url}/metrics", timeout=CALL_S)
        answer.raise_for_status()
        return parse_metrics(answer.text)

    def ask(self, prompt: str, max_tokens: int, **fields: Any) -> dict[str, Any]:
        """The first choice of an answer, or an empty one for any status but 200."""
        self.calls += 1
        answer = self.client.post(f"{self.url}/v1/chat/completions", timeout=CALL_S, json={
            "model": self.alias, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "seed": self.calls, **fields})
        return answer.json()["choices"][0] if answer.status_code == 200 else {}

    def tokens(self, prompt: str, max_tokens: int, **fields: Any) -> list[dict[str, Any]]:
        choice = self.ask(prompt, max_tokens, logprobs=True, top_logprobs=1, **fields)
        return list((choice.get("logprobs") or {}).get("content") or [])


def together[T](calls: Sequence[Callable[[], T]]) -> list[T]:
    """The calls two at a time, as the bot sends them."""
    with ThreadPoolExecutor(2) as pool:
        return [future.result() for future in [pool.submit(call) for call in calls]]


def counted[T](engine: Engine, work: Callable[[], T]) -> tuple[T, dict[str, int]]:
    """What the work returns, and what vLLM drafted and accepted meanwhile."""
    before = engine.metrics()
    result = work()
    after = engine.metrics()
    return result, {name: round(after.get(name, 0) - before.get(name, 0))
                    for name in ("drafts", "draft_tokens", "accepted")}


def step_runner(engine: Engine, manifest: dict[str, str], card_dir: Path) -> dict[str, Any]:
    codes = []
    for name in ("card.jsonl.1", "card.jsonl"):
        path = card_dir / "logs" / name
        for line in path.read_text().splitlines() if path.exists() else ():
            row = json.loads(line)
            if row.get("event") == "engine_choice" and row.get("code") in ("model_runner_v1", "model_runner_v2"):
                codes.append(row["code"])
    runner = codes[-1] if codes else "unknown"
    switch = min_p_while_drafting(manifest)
    return {"ok": runner == "model_runner_v2" and switch, "runner": runner, "switch": switch}


def step_filter(engine: Engine, manifest: dict[str, str], card_dir: Path) -> dict[str, Any]:
    arms: dict[str, dict[str, int]] = {}
    for name, min_p in (("control", 0.0), ("armed", MIN_P)):
        calls = [partial(engine.tokens, prompt, 400, temperature=TEMPERATURE, min_p=min_p)
                 for prompt in PROMPTS for _ in range(4)]
        answers, drafted = counted(engine, partial(together, calls))
        arms[name] = {"tokens": sum(map(len, answers)),
                      "below": sum(below(answer, MIN_P, TEMPERATURE) for answer in answers), **drafted}
    verdict = filtered(arms["control"], arms["armed"])
    return {"ok": verdict == "passed", "verdict": verdict, "min_p": MIN_P, "temperature": TEMPERATURE, **arms}


def step_repeat(engine: Engine, manifest: dict[str, str], card_dir: Path) -> dict[str, Any]:
    shares = {}
    for name, penalty in (("plain", 1.0), ("penalised", PENALTY)):
        answers = [[entry["token"] for entry in engine.tokens(
            REPEATING, 200, temperature=TEMPERATURE, repetition_penalty=penalty)] for _ in range(4)]
        shares[name] = round(sum(map(repeated, answers)) / max(1, sum(map(len, answers))), 3)
    return {"ok": shares["penalised"] < shares["plain"], "penalty": PENALTY, **shares}


def step_schema(engine: Engine, manifest: dict[str, str], card_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + ESTIMATES["schema"] - CALL_S / 4
    format_ = {"type": "json_schema", "json_schema": {"name": "people", "strict": True, "schema": SCHEMA}}
    counts = {"answers": 0, "cut": 0, "broken": 0, "plain": 0}

    def plain() -> None:
        while time.monotonic() < deadline:
            engine.ask(PROMPTS[0], 48, temperature=TEMPERATURE, min_p=MIN_P)
            counts["plain"] += 1
            time.sleep(0.2)  # a gap, so that the schema's answer has batches of its own

    def structured() -> None:
        while time.monotonic() < deadline:
            choice = engine.ask(SCHEMA_PROMPT, 300, temperature=TEMPERATURE, min_p=MIN_P, response_format=format_)
            counts["answers"] += 1
            if choice.get("finish_reason") != "stop":
                counts["cut"] += 1
            elif not follows(choice["message"].get("content") or ""):
                counts["broken"] += 1

    together([plain, structured])
    return {"ok": bool(counts["answers"]) and not counts["cut"] and not counts["broken"], **counts}


def speeds(engine: Engine, prompt: str, min_p: float, deadline: float) -> list[float]:
    """The tokens a second of each answer that began before the deadline."""
    found = []
    while time.monotonic() < deadline:
        began = time.monotonic()
        count = len(engine.tokens(prompt, 400, temperature=TEMPERATURE, min_p=min_p))
        found.append(count / (time.monotonic() - began))
    return found


def step_speed(engine: Engine, manifest: dict[str, str], card_dir: Path) -> dict[str, Any]:
    arms = {}
    for min_p in (0.0, 0.05, MIN_P):
        calls = [partial(speeds, engine, prompt, min_p, time.monotonic() + SPEED_S) for prompt in PROMPTS[:2]]
        found, drafted = counted(engine, partial(together, calls))
        answers = [speed for one in found for speed in one]
        median = round(statistics.median(answers), 1) if answers else 0
        arms[str(min_p)] = {"calls": len(answers), "median_tok_s": median, **drafted}
    return {"ok": True, "observation": True, **arms}


STEPS = {"runner": step_runner, "filter": step_filter, "repeat": step_repeat, "schema": step_schema,
         "speed": step_speed}


def run(engine: Engine, manifest: dict[str, str], card_dir: Path, names: Sequence[str], until: float) -> int:
    failed = False
    for name in names:
        if time.time() + ESTIMATES[name] > until:
            print(json.dumps({"step": name, "ok": False, "code": "no_time"}), flush=True)
            return 2
        now = engine.metrics()
        if now.get("running") or now.get("waiting"):
            print(json.dumps({"step": name, "ok": False, "code": "engine_busy"}), flush=True)
            return 2
        result = STEPS[name](engine, manifest, card_dir)
        failed = failed or not result["ok"]
        print(json.dumps({"step": name, **result}), flush=True)
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--until", type=float, required=True, help="the epoch after which no step may end")
    parser.add_argument("--plan", action="store_true", help="print the steps with their estimates and run nothing")
    parser.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--card-dir", type=Path, default=STATE, help="the card's state, with the launcher's logs")
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps({"steps": {name: ESTIMATES[name] for name in args.only}, "seconds": sum(
            ESTIMATES[name] for name in args.only)}))
        return 0
    manifest = read_manifest()
    # trust_env=False: all of this stays on loopback, never through a proxy of the environment.
    with httpx.Client(trust_env=False) as client:
        try:
            return run(Engine(client, manifest["ENGINE_PORT"], manifest["MODEL_ALIAS"]), manifest, args.card_dir,
                       args.only, args.until)
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            print(json.dumps({"ok": False, "code": "engine_unreadable"}), flush=True)
            return 2


if __name__ == "__main__":
    sys.exit(main())

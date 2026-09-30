"""Synthetic prompts only, built in memory; no real reader data or generated text is retained."""

from __future__ import annotations

import secrets
from typing import Any

SAMPLING = {"temperature": 0.2, "top_p": 0.95, "top_k": 64, "min_p": 0, "repetition_penalty": 1}
UNIT = ("The survey team crossed the quiet valley, recorded the bridge and river, and returned to the station.\n")
MAX_UNITS = 32768


def scope() -> str:
    return "reader." + secrets.token_urlsafe(18)


def body(alias: str, messages: list[dict[str, str]], max_tokens: int, seed: int) -> dict[str, Any]:
    return {"model": alias, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "seed": seed, **SAMPLING}


def victim(alias: str, max_tokens: int, round_index: int) -> dict[str, Any]:
    return body(alias, [{"role": "system", "content": "You write detailed fictional travel journals."},
                        {"role": "user", "content": "Describe a long journey across an imaginary valley. "
                         "Include many distinct scenes and continue until the response limit."}],
                max_tokens, round_index)


def aggressor(alias: str, max_tokens: int, units: int) -> dict[str, Any]:
    return body(alias, [{"role": "system", "content": "Read this fictional journal and follow the final instruction."},
                        {"role": "user", "content": UNIT * units + "\nReply with the single word: read."}],
                max_tokens, 0)

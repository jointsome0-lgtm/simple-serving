"""The stream the gateway sends (contract section 4), rebuilt from the engine's stream.

Each engine event is checked field by field and new chunks are built, so nothing else the engine adds reaches the
client. An engine event is a chunk of the served model: its `object` is "chat.completion.chunk", its `model` the alias,
and its `choices` a list of at most one choice, which has `index` 0 and a `delta` object. The delta holds no field but
`role`, `content` and the engine's reasoning field: v1 is text only, so a tool call breaks the rules. A field that vLLM
leaves out or sends as null means none: `usage`, `finish_reason`, the cached count and each field of the delta. The
other fields of an event or a choice, such as `id`, `logprobs` and `stop_reason`, are not read.

Every chunk sent has the alias as its model and one choice with index 0, or no choice at all. A delta holds nothing,
`role`, `reasoning_content` or `content`. An engine event becomes one chunk, or one for each of those fields when its
delta holds several; the engine's reasoning field, whatever its name, becomes `reasoning_content`. An event that breaks
the rules raises EngineError: before the first chunk the gateway answers 503, after it the stream ends with an error
event.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any

from .engine import EngineError

CHUNK_OBJECT = "chat.completion.chunk"
FINISH_REASONS = ("stop", "length")
REASONING_FIELDS = ("reasoning_content", "reasoning")  # vLLM has used both names
DELTA_FIELDS = frozenset({"role", "content", *REASONING_FIELDS})


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None  # None: the engine did not say


class Translator:
    def __init__(self, alias: str) -> None:
        self._alias = alias
        self._head = {"id": "chatcmpl-" + secrets.token_hex(12), "object": CHUNK_OBJECT, "created": int(time.time()),
                      "model": alias}
        self.finish_reason: str | None = None
        self.usage: Usage | None = None
        self.generated = False  # whether a token of any kind, reasoning included, has arrived

    def chunks(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """The chunks to send for one engine event."""
        # An error event is not read further: its text may quote the prompt.
        if "error" in event or event.get("object") != CHUNK_OBJECT or event.get("model") != self._alias:
            raise EngineError("engine_unavailable")
        if event.get("usage") is not None:
            self.usage = _usage(event["usage"])
        choices = event.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise EngineError("engine_unavailable")
        return self._choice(choices[0]) if choices else []

    def end(self) -> Usage:
        """The usage of a stream the engine has ended, which must have had its finish and its usage."""
        if self.finish_reason is None or self.usage is None:
            raise EngineError("engine_unavailable")
        return self.usage

    def usage_chunk(self, usage: Usage, measurements: dict[str, int]) -> dict[str, Any]:
        fields: dict[str, Any] = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
        if usage.cached_tokens is not None:
            fields["prompt_tokens_details"] = {"cached_tokens": usage.cached_tokens}
        fields["simple_serving"] = measurements
        return {**self._head, "choices": [], "usage": fields}

    def _choice(self, choice: Any) -> list[dict[str, Any]]:
        index = choice.get("index") if isinstance(choice, dict) else None
        if type(index) is not int or index != 0:
            raise EngineError("engine_unavailable")
        parts = _parts(choice.get("delta"))
        reason = choice.get("finish_reason")
        if self.finish_reason is not None:
            if parts or reason is not None:  # a second finish, or text after the finish
                raise EngineError("engine_unavailable")
            return []
        if reason is not None and reason not in FINISH_REASONS:  # such as vLLM's "abort": the request did not finish
            raise EngineError("engine_unavailable")
        if any("role" not in part for part in parts):
            self.generated = True
        chunks = [self._chunk(part) for part in parts or [{}]]
        if reason is not None:
            chunks[-1]["choices"][0]["finish_reason"] = reason
            self.finish_reason = reason
        return chunks

    def _chunk(self, delta: dict[str, str]) -> dict[str, Any]:
        return {**self._head, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def _parts(delta: Any) -> list[dict[str, str]]:
    """A delta as single-field deltas, in the order role, reasoning, content. A null field is left out, and so is
    empty text. Any field but those breaks the rules, a tool call or a function call included."""
    if not isinstance(delta, dict):
        raise EngineError("engine_unavailable")
    fields = {name: value for name, value in delta.items() if value is not None}
    if not fields.keys() <= DELTA_FIELDS or fields.get("role", "assistant") != "assistant":
        raise EngineError("engine_unavailable")
    if not all(isinstance(fields[name], str) for name in fields.keys() - {"role"}):
        raise EngineError("engine_unavailable")
    parts = [{"role": "assistant"}] if "role" in fields else []
    reasoning = next((fields[name] for name in REASONING_FIELDS if name in fields), "")
    for name, text in (("reasoning_content", reasoning), ("content", fields.get("content", ""))):
        if text:
            parts.append({name: text})
    return parts


def _usage(value: Any) -> Usage:
    """The counts of a usage object. A cached count left out or null is unknown: vLLM sends it only with
    `--enable-prompt-tokens-details`."""
    if not isinstance(value, dict):
        raise EngineError("engine_unavailable")
    details = value.get("prompt_tokens_details")
    if details is None:
        details = {}
    if not isinstance(details, dict):
        raise EngineError("engine_unavailable")
    prompt: Any = value.get("prompt_tokens")
    completion: Any = value.get("completion_tokens")
    cached: Any = details.get("cached_tokens")
    if not (_is_count(prompt) and _is_count(completion) and (cached is None or _is_count(cached))):
        raise EngineError("engine_unavailable")
    return Usage(prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached)


def _is_count(value: Any) -> bool:
    return type(value) is int and value >= 0

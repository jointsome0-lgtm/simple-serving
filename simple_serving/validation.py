"""Strict checks of request bodies: generation and counting (contract section 4) and the control routes (section 8).

Nothing is coerced: a boolean is not an integer, and "1" is not 1. A field outside the contract, at any depth, is
`unsupported_field`; a missing value, a wrong type or a value out of range is `invalid_request`. The schema inside
`response_format.json_schema.schema` is not looked into and passes on to the engine as it is. While the engine takes
pictures (config.VISION), a user message's content may also be a list of parts: text, and one picture in the whole
request (PICTURE).
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .errors import ServiceError

ROLES = ("system", "user", "assistant")
MAX_DEPTH = 64  # far deeper than any schema a client needs, far shallower than what breaks the JSON encoder
SCHEMA = object()  # the one place whose content is not checked
LEAF = None
# Which fields exist at each depth of a generation body. A list holds the shape of its elements.
CHAT_SHAPE: dict[str, Any] = {
    "model": LEAF,
    "messages": [{"role": LEAF, "content": LEAF}],
    "max_tokens": LEAF,
    "stream": LEAF,
    "stream_options": {"include_usage": LEAF},
    "temperature": LEAF,
    "top_p": LEAF,
    "top_k": LEAF,
    "min_p": LEAF,
    "repetition_penalty": LEAF,
    "seed": LEAF,
    "response_format": {"type": LEAF, "json_schema": {"name": LEAF, "strict": LEAF, "schema": SCHEMA}},
    "chat_template_kwargs": {"enable_thinking": LEAF},
    "thinking_token_budget": LEAF,
}
# The same while the engine takes pictures, where a content may be a list of parts.
VISION_SHAPE: dict[str, Any] = CHAT_SHAPE | {
    "messages": [{"role": LEAF, "content": [{"type": LEAF, "text": LEAF, "image_url": {"url": LEAF}}]}]}
# A picture is a data URL, which vLLM 0.30.0 decodes itself. It would fetch an http URL, and read a file URL under
# --allowed-local-media-path, so the gateway passes on neither. vLLM tells the type by the bytes and ignores the URL's
# (vllm/multimodal/media/connector.py and image.py), so the gateway checks that the two agree.
PICTURE = re.compile(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/]*={0,2})")
SIGNATURES = {"png": b"\x89PNG\r\n\x1a\n", "jpeg": b"\xff\xd8\xff"}  # WebP's is split, as RIFF....WEBP
# The compressed image file after base64 decoding. This request-size bound does not bound raster memory;
# vLLM applies its own pixel limit and decoder checks.
PICTURE_BYTES = 2 << 20


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict[str, Any]]  # each content a string, or while the engine takes pictures a list of parts
    max_tokens: int
    sampling: dict[str, int | float]  # the optional sampling fields the client sent
    response_format: dict[str, Any] | None
    enable_thinking: bool
    thinking_token_budget: int | None  # the most tokens of the thought; the class limit is the service's to check
    pictures: int = 0  # 0 or 1


def parse_json(raw: bytes) -> Any:
    """JSON in UTF-8, without NaN, infinities or repeated keys, nested at most MAX_DEPTH deep."""
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_refuse, parse_float=_finite,
                           object_pairs_hook=_unique_keys)
    except (ValueError, RecursionError):
        raise ServiceError("invalid_request") from None
    _require(_depth(value) <= MAX_DEPTH)
    return value


def check_chat(body: Any, alias: str, *, vision: bool = False) -> ChatRequest:
    """A body of generation or counting. `vision` says that the engine takes pictures; whether the key may send one
    is checked after this (routes.py)."""
    if not isinstance(body, dict):
        raise ServiceError("invalid_request")
    if _has_unknown_field(body, VISION_SHAPE if vision else CHAT_SHAPE):
        raise ServiceError("unsupported_field")
    _require(body.get("model") == alias)
    messages: Any = body.get("messages")
    _require(isinstance(messages, list) and len(messages) > 0)
    for message in messages:
        _require(isinstance(message, dict) and message.get("role") in ROLES)
        _require(isinstance(message.get("content"), str) or (vision and message["role"] == "user" and _parts(message)))
    pictures = sum(part["type"] == "image_url" for message in messages if isinstance(message["content"], list)
                   for part in message["content"])
    _require(pictures <= 1)
    max_tokens: Any = body.get("max_tokens")
    _require(_is_integer(max_tokens) and max_tokens >= 1)
    _require(body.get("stream") is True)
    stream_options = body.get("stream_options")
    _require(isinstance(stream_options, dict) and stream_options.get("include_usage") is True)
    return ChatRequest(
        messages=[{"role": message["role"], "content": _content(message["content"])} for message in messages],
        max_tokens=max_tokens,
        sampling=_sampling(body),
        response_format=_response_format(body),
        enable_thinking=_enable_thinking(body),
        thinking_token_budget=_thinking_token_budget(body),
        pictures=pictures,
    )


def check_boot(body: Any) -> str:
    """The `boot_id` of a drain or a sleep."""
    return _control(body, ("boot_id",))[0]


def check_open(body: Any) -> tuple[str, int]:
    """The `boot_id` and `drain_generation` of an open."""
    boot_id, generation = _control(body, ("boot_id", "drain_generation"))
    return boot_id, generation


def _parts(message: dict[str, Any]) -> bool:
    """Whether a message's content is a list of parts, each text or a picture."""
    parts = message.get("content")
    return isinstance(parts, list) and len(parts) > 0 and all(_part(part) for part in parts)


def _part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    if part.keys() == {"type", "text"}:
        return part["type"] == "text" and isinstance(part["text"], str)
    return part.keys() == {"type", "image_url"} and part["type"] == "image_url" and _picture(part["image_url"])


def _picture(image_url: Any) -> bool:
    """Whether an `image_url` holds a picture: a data URL of PNG, JPEG or WebP in base64, as strict as vLLM decodes
    it, whose signature matches the type it names and whose decoded base64 is at most PICTURE_BYTES.
    The engine checks complete image decoding; compressed size does not bound the pixel count."""
    url = image_url.get("url") if isinstance(image_url, dict) else None
    found = PICTURE.fullmatch(url) if isinstance(url, str) else None
    if found is None:
        return False
    try:
        data = base64.b64decode(found[2], validate=True)
    except binascii.Error:
        return False
    kind = "webp" if data[:4] == b"RIFF" and data[8:12] == b"WEBP" else next(
        (name for name, signature in SIGNATURES.items() if data.startswith(signature)), None)
    return len(data) <= PICTURE_BYTES and kind == found[1]


def _content(content: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """A content as the engine gets it: the string, or each part built anew from its own fields."""
    if isinstance(content, str):
        return content
    return [{"type": "text", "text": part["text"]} if part["type"] == "text"
            else {"type": "image_url", "image_url": {"url": part["image_url"]["url"]}} for part in content]


def _sampling(body: dict[str, Any]) -> dict[str, int | float]:
    checks = {
        "temperature": lambda v: _is_number(v) and 0 <= v <= 2,
        "top_p": lambda v: _is_number(v) and 0 < v <= 1,
        "top_k": lambda v: _is_integer(v) and v >= 1,
        "min_p": lambda v: _is_number(v) and 0 <= v <= 1,
        "repetition_penalty": lambda v: _is_number(v) and 0 < v <= 2,
        "seed": _is_integer,
    }
    sampling = {}
    for name, check in checks.items():
        if name in body:
            _require(check(body[name]))
            sampling[name] = body[name]
    return sampling


def _response_format(body: dict[str, Any]) -> dict[str, Any] | None:
    if "response_format" not in body:
        return None
    value = body["response_format"]
    _require(isinstance(value, dict) and value.get("type") == "json_schema")
    schema = value.get("json_schema")
    _require(isinstance(schema, dict) and isinstance(schema.get("name"), str)
             and isinstance(schema.get("strict"), bool) and isinstance(schema.get("schema"), dict))
    return value


def _enable_thinking(body: dict[str, Any]) -> bool:
    if "chat_template_kwargs" not in body:
        return False
    value = body["chat_template_kwargs"]
    _require(isinstance(value, dict))
    thinking = value.get("enable_thinking", False)
    _require(isinstance(thinking, bool))
    return thinking


def _thinking_token_budget(body: dict[str, Any]) -> int | None:
    if "thinking_token_budget" not in body:
        return None
    budget = body["thinking_token_budget"]
    _require(_is_integer(budget) and budget >= 0)
    return budget


def _control(body: Any, fields: tuple[str, ...]) -> list[Any]:
    if not isinstance(body, dict):
        raise ServiceError("invalid_request")
    if not set(body) <= set(fields):
        raise ServiceError("unsupported_field")
    _require(isinstance(body.get("boot_id"), str))
    if "drain_generation" in fields:
        generation: Any = body.get("drain_generation")
        _require(_is_integer(generation) and generation >= 0)
    return [body[name] for name in fields]


def _depth(value: Any) -> int:
    """How deep objects and arrays nest, counted without recursion."""
    deepest, pending = 0, [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, (dict, list)):
            deepest = max(deepest, depth)
            pending.extend((child, depth + 1) for child in (item.values() if isinstance(item, dict) else item))
    return deepest


def _has_unknown_field(value: Any, shape: Any) -> bool:
    """Whether `value` has a field that `shape` does not know. The shape, not the input, bounds the depth."""
    if shape is LEAF or shape is SCHEMA:
        return False
    if isinstance(shape, dict):
        return isinstance(value, dict) and any(
            name not in shape or _has_unknown_field(item, shape[name]) for name, item in value.items())
    return isinstance(value, list) and any(_has_unknown_field(item, shape[0]) for item in value)


def _is_integer(value: Any) -> bool:
    return type(value) is int  # a bool is an int in Python, not in the contract


def _is_number(value: Any) -> bool:
    return type(value) in (int, float)


def _require(condition: bool) -> None:
    if not condition:
        raise ServiceError("invalid_request")


def _refuse(constant: str) -> Any:
    raise ValueError("not JSON")


def _finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):  # 1e999 would become an infinity
        raise ValueError("not a finite number")
    return value


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("a repeated key")
    return result

"""The JSON schemas of simple-story-chat's structured answers, as the smoke sends them, and a check of an answer
against one (contract section 15).

bot_schemas.json holds a copy of each schema the bot sends in `response_format`, with the commit and the file it was
copied from, the bot's `max_tokens` and purpose, and a short synthetic prompt of the smoke's own. `problems` checks an
answer with the keywords those schemas use, as JSON Schema defines them. A schema with any other keyword is refused
when the copy is read, so a new copy cannot pass unchecked.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).with_name("bot_schemas.json")
KEYWORDS = frozenset({"type", "properties", "required", "additionalProperties", "items", "enum", "const", "minItems",
                      "maxItems", "minLength", "maxLength", "pattern", "minimum", "maximum"})
TYPES = ("object", "array", "string", "integer", "number", "boolean", "null")


@dataclass(frozen=True)
class BotSchema:
    name: str
    path: str  # in simple-story-chat, at the copy's commit
    bot_max_tokens: int
    memory: bool  # the purpose memory, which the bot samples at temperature 0.2
    system: str
    user: str
    schema: dict[str, Any]


def load(path: Path = FIXTURE) -> list[BotSchema]:
    """The schemas of the copy. One with a keyword that `problems` does not know is refused."""
    copy = json.loads(path.read_text(encoding="utf-8"))
    schemas = []
    for entry in copy["schemas"]:
        for node in _nodes(entry["schema"]):
            if unknown := set(node) - KEYWORDS:
                raise ValueError(f"{entry['name']}: keywords the check does not know: {', '.join(sorted(unknown))}")
            if node.get("type", "object") not in TYPES or node.get("additionalProperties", False) not in (True, False):
                raise ValueError(f"{entry['name']}: a type or additionalProperties the check does not know")
            try:
                _pattern(node.get("pattern", ""))
            except re.error:
                raise ValueError(f"{entry['name']}: a pattern that is not a regular expression") from None
        schemas.append(BotSchema(entry["name"], entry["path"], entry["bot_max_tokens"], entry["purpose"] == "memory",
                                 entry["system"], entry["user"], entry["schema"]))
    return schemas


def problems(instance: Any, schema: dict[str, Any]) -> set[str]:
    """The keywords of `schema` that `instance` breaks: empty when it is valid."""
    found: set[str] = set()
    _check(instance, schema, found)
    return found


def _check(value: Any, schema: dict[str, Any], found: set[str]) -> None:
    if "type" in schema and not _is(value, schema["type"]):
        found.add("type")
        return
    if "enum" in schema and not any(_equal(value, option) for option in schema["enum"]):
        found.add("enum")
    if "const" in schema and not _equal(value, schema["const"]):
        found.add("const")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(name not in value for name in schema.get("required", ())):
            found.add("required")
        if schema.get("additionalProperties") is False and value.keys() - properties.keys():
            found.add("additionalProperties")
        for name, item in value.items():
            if name in properties:
                _check(item, properties[name], found)
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            found.add("minItems")
        if len(value) > schema.get("maxItems", len(value)):
            found.add("maxItems")
        for item in value if "items" in schema else ():
            _check(item, schema["items"], found)
    elif isinstance(value, str):
        # JSON Schema counts characters, as Python does: code points, not UTF-16 units.
        if len(value) < schema.get("minLength", 0):
            found.add("minLength")
        if len(value) > schema.get("maxLength", len(value)):
            found.add("maxLength")
        if "pattern" in schema and not _pattern(schema["pattern"]).search(value):
            found.add("pattern")
    elif _is(value, "number"):
        if value < schema.get("minimum", value):
            found.add("minimum")
        if value > schema.get("maximum", value):
            found.add("maximum")


def _is(value: Any, kind: str) -> bool:
    """JSON's types, where true is not a number and 1.0 is an integer."""
    if kind == "integer":
        return type(value) is int or (type(value) is float and value.is_integer())
    if kind == "number":
        return type(value) in (int, float)
    kinds: dict[str, type | None] = {"object": dict, "array": list, "string": str, "boolean": bool, "null": None}
    expected = kinds[kind]
    return value is None if expected is None else type(value) is expected


def _equal(a: Any, b: Any) -> bool:
    if _is(a, "number") and _is(b, "number"):
        return bool(a == b)
    return type(a) is type(b) and bool(a == b)


@functools.cache
def _pattern(source: str) -> re.Pattern[str]:
    """A pattern as JSON Schema reads it, after ECMA-262: it may match anywhere, `$` ends the string (Python's would
    also match before a final newline), and a class such as `\\d` is ASCII."""
    parts, escaped, in_class = [], False, False
    for char in source:
        if escaped or char == "\\":
            escaped = not escaped
        elif in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
        elif char == "$":
            char = r"\Z"
        parts.append(char)
    return re.compile("".join(parts), re.ASCII)


def _nodes(schema: Any) -> Iterator[dict[str, Any]]:
    """Every schema inside a schema, itself included, as far as the keywords above lead."""
    if not isinstance(schema, dict):
        raise TypeError("a schema must be an object")
    yield schema
    for item in schema.get("properties", {}).values():
        yield from _nodes(item)
    if "items" in schema:
        yield from _nodes(schema["items"])

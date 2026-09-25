"""The copy of simple-story-chat's JSON schemas (simple_serving/bot_schemas.json), and the check of an answer against
one, which the smoke's `schemas` probe uses."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from simple_serving import bot_schemas

# A valid answer to each schema of the copy.
VALID: dict[str, Any] = {
    "sheet": {"characters": [{"name": "Pavel", "look": "old, grey beard", "outfit": "oilskin coat"}]},
    "frame": {"moment": "the key changes hands", "shot": "medium", "setting": "the foot of the tower",
              "objects": "a door", "props": "a key", "light": "dusk",
              "people": [{"who": "Pavel", "look": "old", "clothes": "a coat", "state": "calm", "action": "gives"}]},
    "memory_plain": {"facts": [{"kind": "event", "at": "2 August, 20:00", "text": "Pavel gives Vera the key.",
                                "source": ["n3"]}]},
    "memory_sgr": {"evidence": [{"id": "e1", "scene": "n3", "part": "text", "quote": "Pavel gives Vera the key"}],
                   "conflicts": [], "facts": [{"kind": "event", "at": "20:00", "status": "actual",
                                               "text": "Pavel gives Vera the key.", "evidence": ["e1"]}]},
    "walk_judge": {"verdict": "inconsistent",
                   "contradictions": [{"now": "green", "before": "red", "where": "the lamp", "kind": "item"}]},
    "walk_cross": {"checks": [{"finding": 1, "confirmed": True, "note": "no reason"},
                              {"finding": 2, "confirmed": False, "note": "he is there"}]},
    "seed_audit": {"issues": [{"kind": "contradiction", "quote": "alone", "note": "and with his brother"}]},
    "story_audit": {"issues": [{"kind": "contradiction", "scene": 2, "quote": "green", "note": "it was red"}]},
    "model_probe": {"schema_check": "ok"},
    "memory_recall": {"answers": [{"key": "keeper", "value": "Pavel"}, {"key": "time", "value": "20:00"}]},
    "scene_judge": {"answers": [{"key": "given", "value": "yes"}, {"key": "locked", "value": "no"}]},
    "judge_extract": {"items": [
        {"key": "give_key", "status": "completed", "actor": "Pavel", "object": "key", "number": "", "quote": "gives"},
        {"key": "open_door", "status": "attempted", "actor": "Vera", "object": "door", "number": "", "quote": "tries"},
    ]},
}
SCHEMAS = {entry.name: entry.schema for entry in bot_schemas.load()}


def test_the_copy_of_the_bots_schemas_names_its_source() -> None:
    copied = json.loads(bot_schemas.FIXTURE.read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", copied["commit"])
    entries = bot_schemas.load()
    assert [entry.name for entry in entries] == list(VALID)
    for entry in entries:
        assert re.fullmatch(r"local/[a-z-]+\.ts", entry.path)
        assert type(entry.bot_max_tokens) is int and entry.bot_max_tokens > 0
        assert entry.system and entry.user
    # Between them the copies use every keyword the contract names for the smoke, in strict mode.
    used = {key for entry in entries for node in bot_schemas._nodes(entry.schema) for key in node}
    assert {"minLength", "maxLength", "minItems", "maxItems", "pattern"} <= used


@pytest.mark.parametrize("name", list(VALID))
def test_each_schema_takes_a_valid_answer(name: str) -> None:
    assert bot_schemas.problems(VALID[name], SCHEMAS[name]) == set()


def changed(name: str, path: str, value: Any) -> Any:
    """The valid answer of a schema with one value changed, found by its path of keys and indexes; `...` removes it."""
    answer = copy.deepcopy(VALID[name])
    *parents, last = [int(part) if part.isdigit() else part for part in path.split("/")]
    target = answer
    for part in parents:
        target = target[part]
    if value is ...:
        del target[last]
    else:
        target[last] = value
    return answer


@pytest.mark.parametrize(("name", "path", "value", "keyword"), [
    ("memory_plain", "facts", [], "minItems"),
    ("memory_plain", "facts", VALID["memory_plain"]["facts"] * 201, "maxItems"),
    ("memory_plain", "facts/0/at", "", "minLength"),
    ("memory_plain", "facts/0/at", "x" * 201, "maxLength"),
    ("memory_plain", "facts/0/source/0", "n5", "enum"),
    ("memory_plain", "facts/0/kind", 1, "type"),
    ("memory_sgr", "evidence/0/id", "e0", "pattern"),
    ("memory_sgr", "evidence/0/id", "e12345", "pattern"),
    ("memory_sgr", "evidence/0/id", "e12\n", "pattern"),  # JSON Schema's $ ends the string; Python's would not
    ("memory_sgr", "facts/0/evidence/0", "E1", "pattern"),
    ("sheet", "characters/0/age", "old", "additionalProperties"),
    ("sheet", "characters/0/look", ..., "required"),
    ("sheet", "characters", VALID["sheet"]["characters"] * 7, "maxItems"),
    ("model_probe", "schema_check", "no", "const"),
    ("walk_cross", "checks/0/finding", 3, "maximum"),
    ("walk_cross", "checks/0/finding", 0, "minimum"),
    ("walk_cross", "checks/0/finding", "1", "type"),
    ("walk_cross", "checks/0/finding", True, "type"),  # true is not a number
    ("walk_cross", "checks/0/confirmed", 1, "type"),
    ("walk_cross", "checks", VALID["walk_cross"]["checks"][:1], "minItems"),
    ("story_audit", "issues/0/scene", 1.5, "type"),
    ("judge_extract", "items/0/number", "1" * 21, "maxLength"),
    ("scene_judge", "answers/0/value", "maybe", "enum"),
])
def test_an_answer_that_breaks_a_keyword_is_caught(name: str, path: str, value: Any, keyword: str) -> None:
    assert bot_schemas.problems(changed(name, path, value), SCHEMAS[name]) == {keyword}


def test_json_numbers_and_code_points() -> None:
    assert bot_schemas.problems(changed("walk_cross", "checks/0/finding", 1.0), SCHEMAS["walk_cross"]) == set()
    assert bot_schemas.problems(changed("memory_plain", "facts/0/at", "ё" * 200), SCHEMAS["memory_plain"]) == set()
    assert bot_schemas.problems(changed("memory_plain", "facts/0/at", "😀" * 200), SCHEMAS["memory_plain"]) == set()


@pytest.mark.parametrize("node", [{"type": "string", "format": "date"}, {"anyOf": []}, {"type": ["string", "null"]},
                                  {"type": "object", "additionalProperties": {"type": "string"}},
                                  {"type": "string", "pattern": "(unclosed"}])
def test_a_copy_with_a_schema_the_check_does_not_know_is_refused(tmp_path: Path, node: dict[str, Any]) -> None:
    copied = json.loads(bot_schemas.FIXTURE.read_text(encoding="utf-8"))
    copied["schemas"][0]["schema"]["properties"]["extra"] = node
    path = tmp_path / "copy.json"
    path.write_text(json.dumps(copied))
    with pytest.raises((ValueError, TypeError)):
        bot_schemas.load(path)

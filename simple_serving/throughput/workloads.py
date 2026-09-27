"""The harness's workloads, built in memory from simple-story-chat's public eval pack (docs/speed.md, "The metric").

The pack (`--pack`, a copy of ~/simple-story-chat-eval on the card) holds five synthetic scenarios, each a seed and a
story of 16 scenes, a reader's input and the narrator's text each. A *book* is one scenario's seed and scenes followed
by the other four's scenes, about 80 scenes, so that a prompt can reach the bot's lengths. Nothing of it is written to
disk or printed: a workload is a list of lanes, each a list of request bodies in the bot's form (simple-story-chat's
`local/serving.ts`), and the harness reports only their counts.

- turns: one lane per request at once, each a book of its own read from a target context on, turn after turn, so that
  each turn's prompt extends the one before as a story's does; 0.8.
- compactions: JSON memory under the bot's schema (`memory_plain` of bot_schemas.json) of four scenes, at 0.2.
- scenes: short prompts whose answer is a scene of a few hundred tokens, at 0.8.
- unshared: prompts at the target context in scopes of their own, so that nothing is shared.
- reading: one book read turn by turn with a gap between turns, as one reader reads.
- interleaved: four long books of 30000 to 42000 tokens that take turns, one request at a time with a gap between
  turns, for the copy of the cache in RAM (item 2): each story's prefix meets the others' in the cache.

Every request is of class `internal`. A lane names a `reader.<opaque>` scope of its own, fresh for each pass, so that
lanes share no cache with each other or with an earlier pass.
"""

from __future__ import annotations

import copy
import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .. import bot_schemas

SCENARIOS = ("assault", "battle", "chess", "dance", "hospital")  # the public pack's replay scenarios
# The harness's own instructions, synthetic as the pack: the bot's prompts are not copied.
SYSTEM = ("You narrate an interactive story. Continue it from the reader's last message with the next scene, in the "
          "story's language.")
RULE = "Write the next scene in the story's language, a few paragraphs, and stop where the reader can act."
FACTS = "List the facts of the new scenes that the story needs later, each with the scenes it comes from."
SAMPLING = {"top_p": 0.95, "top_k": 64, "min_p": 0, "repetition_penalty": 1}  # the bot's (local/serving.ts)
STORY_T, MEMORY_T = 0.8, 0.2
MAX_TOKENS = {"turns": 1024, "scenes": 400, "compactions": 4096, "unshared": 256, "reading": 1024,
              "interleaved": 1024}
NAMES = tuple(MAX_TOKENS)
COMPACTED = 4  # scenes a compaction takes
INTERLEAVED = 4  # the stories of item 2
STEP_TOKENS = 4000  # between the contexts of the interleaved stories
DEFAULT_CONTEXT = {"turns": 16000, "unshared": 16000, "reading": 8000, "interleaved": 30000}


@dataclass(frozen=True)
class Scene:
    input: str
    text: str


@dataclass(frozen=True)
class Story:
    seed: str
    scenes: tuple[Scene, ...]


@dataclass(frozen=True)
class Request:
    workload: str
    lane: int
    turn: int
    scope: str
    body: dict[str, Any]
    gap_s: float = 0.0  # the pause before it, after the lane's previous request ended


def load_pack(path: Path) -> list[Story]:
    """The five scenarios of the pack, each its seed and its story's scenes from the first on."""
    stories = []
    for name in SCENARIOS:
        state = json.loads((path / name / "frozen.json").read_text(encoding="utf-8"))["state"]
        seed = next(iter(state["seeds"].values()))
        nodes = next(iter(state["stories"].values()))["nodes"]
        order, parent = [], None
        children = {node.get("parent"): node for node in nodes.values()}
        while (node := children.get(parent)) is not None:
            order.append(Scene(node["input"], node["text"]))
            parent = node["id"]
        stories.append(Story(f"{seed['title']}\n{seed['startTime']}\n\n{seed['text']}", tuple(order)))
    return stories


def books(stories: list[Story]) -> list[Story]:
    """Each story's seed with its own scenes and then the others', in the pack's order."""
    return [Story(story.seed, tuple(scene for other in stories[index:] + stories[:index] for scene in other.scenes))
            for index, story in enumerate(stories)]


def turn_messages(book: Story, at: int) -> list[dict[str, str]]:
    """The prompt of the turn that answers scene `at`'s input: the seed, the scenes before it, its input and the
    rule last, as the bot's prompt ends with its narrator rule."""
    messages = [{"role": "system", "content": f"{SYSTEM}\n\n{book.seed}"}]
    for scene in book.scenes[:at]:
        messages += [{"role": "user", "content": scene.input}, {"role": "assistant", "content": scene.text}]
    messages.append({"role": "user", "content": f"{book.scenes[at].input}\n\n{RULE}"})
    return messages


def characters(messages: list[dict[str, str]]) -> int:
    return sum(len(message["content"]) for message in messages)


def scene_at(book: Story, tokens: int, chars_per_token: float, room: int = 1) -> int:
    """The last scene whose turn stays within `tokens`, leaving `room` scenes after it. The first scene at least."""
    at = 0
    while at + room < len(book.scenes) and characters(turn_messages(book, at + 1)) / chars_per_token <= tokens:
        at += 1
    return at


def body(alias: str, messages: list[dict[str, str]], max_tokens: int, temperature: float,
         response_format: dict[str, Any] | None = None) -> dict[str, Any]:
    """A generation body as the bot sends it (contract section 4)."""
    request: dict[str, Any] = {"model": alias, "messages": messages, "max_tokens": max_tokens, "stream": True,
                               "stream_options": {"include_usage": True}, "temperature": temperature, **SAMPLING,
                               "chat_template_kwargs": {"enable_thinking": False}}
    if response_format is not None:
        request["response_format"] = response_format
    return request


def compaction(alias: str, story: Story, start: int) -> dict[str, Any]:
    """A compaction of COMPACTED scenes from `start` into memory, under the bot's schema with those scenes' ids."""
    memory = next(schema for schema in bot_schemas.load() if schema.name == "memory_plain")
    start %= max(1, len(story.scenes) - COMPACTED + 1)
    taken = story.scenes[start:start + COMPACTED]
    ids = [f"n{start + offset + 1}" for offset in range(len(taken))]
    schema = copy.deepcopy(memory.schema)
    source = schema["properties"]["facts"]["items"]["properties"]["source"]
    source["items"]["enum"], source["maxItems"] = ids, len(ids)
    scenes = [{"id": id_, "input": scene.input, "text": scene.text} for id_, scene in zip(ids, taken, strict=True)]
    user = json.dumps({"seed": story.seed, "previousMemory": [], "newScenes": scenes}, ensure_ascii=False)
    user += f"\n\n{FACTS}"
    fmt = {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}}
    return body(alias, [{"role": "system", "content": memory.system}, {"role": "user", "content": user}],
                MAX_TOKENS["compactions"], MEMORY_T, fmt)


def scope() -> str:
    return f"reader.{secrets.token_urlsafe(18)}"


@dataclass(frozen=True)
class Shape:
    """What a pass asks for beside its workload: requests at once, the target context, turns and the gap."""

    at: int = 1
    context: int | None = None  # None: the workload's default
    turns: int = 4
    gap_s: float = 10.0
    chars_per_token: float = 4.0  # measured on the card with the gateway's count, before the first pass

    def target(self, workload: str) -> int:
        return self.context if self.context is not None else DEFAULT_CONTEXT.get(workload, 0)


def build(workload: str, stories: list[Story], alias: str, shape: Shape,
          new_scope: Callable[[], str] = scope) -> list[list[Request]]:
    """The lanes of one pass. A lane's requests run one after another and `at_once` lanes run at once; a request's
    `lane` names its reader, whose scope it has."""
    chained = books(stories)
    target, cpt = shape.target(workload), shape.chars_per_token

    def story(reader: int, book: Story, tokens: int, gap_s: float) -> list[Request]:
        first, own = scene_at(book, tokens, cpt, room=shape.turns), new_scope()
        return [Request(workload, reader, turn, own,
                        body(alias, turn_messages(book, first + turn), MAX_TOKENS[workload], STORY_T),
                        gap_s if turn else 0.0)
                for turn in range(min(shape.turns, len(book.scenes) - first))]

    def single(reader: int, request: dict[str, Any]) -> list[Request]:
        return [Request(workload, reader, 0, new_scope(), request)]

    if workload == "turns":
        return [story(reader, chained[reader % len(chained)], target, 0.0) for reader in range(shape.at)]
    if workload == "reading":
        return [story(0, chained[0], target, shape.gap_s)]
    if workload == "interleaved":
        # One reader at a time, the four stories in turn, so that each story's prefix meets the others' in the cache.
        readers = [story(reader, chained[reader], target + reader * STEP_TOKENS, 0.0) for reader in range(INTERLEAVED)]
        turns = [request for turn in range(shape.turns) for lane in readers for request in lane[turn:turn + 1]]
        return [[replace(request, gap_s=shape.gap_s if index else 0.0) for index, request in enumerate(turns)]]
    count = 2 * shape.at
    if workload == "compactions":
        return [single(reader, compaction(alias, stories[reader % len(stories)], reader // len(stories) * COMPACTED))
                for reader in range(count)]
    if workload == "scenes":
        return [single(reader, body(alias, turn_messages(chained[reader % len(chained)], 1 + reader % 3),
                                    MAX_TOKENS[workload], STORY_T))
                for reader in range(count)]
    if workload == "unshared":
        return [single(reader, body(alias, turn_messages(book, scene_at(book, target, cpt)), MAX_TOKENS[workload],
                                    STORY_T))
                for reader in range(count) for book in [chained[reader % len(chained)]]]
    raise ValueError(f"unknown workload {workload!r}")


def at_once(workload: str, shape: Shape) -> int:
    """How many lanes a pass runs at once: one for the story read with gaps and for the four stories, which take
    turns, else `shape.at`."""
    return 1 if workload in ("reading", "interleaved") else shape.at

"""What a graph may be and touch on the card, by class (docs/pictures.md, sections 2, 4 and 5), and the pictures on
their way.

A graph is ComfyUI's API format, passed through unchanged apart from ComfyUI's per-job preview nonce. ComfyUI checks
every node and input against its registry itself; the gateway checks what a graph may touch: the nodes of a reader's
or an agent's graph, and the files every class's graph reads. A refusal is `graph_refused`, before anything reaches
the card, and says nothing of the graph.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ..errors import ServiceError

NODE_FIELDS = {"class_type", "inputs", "_meta"}
NODE_ID = re.compile(r"[A-Za-z0-9_:.-]{1,64}")
MAX_COPIES = 16
# A reader's reference as the bot writes it into its graph, under the name the bot chose before sending it.
UPLOAD_NAME = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.(png|jpg|webp)")
UPLOADED = re.compile(r"bot-references/(?P<name>[^/\[\]]+) \[temp\]")
# An experiment's reference in the input directory, as `uploadReference` names it; any plain name of that directory.
INPUT_NAME = re.compile(r"ref-[0-9a-f]{16}\.png")
PLAIN_INPUT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
TEMP = " [temp]"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
KEPT_CHUNKS = {b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND"}
UPLOAD_TYPES = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}


class NotPng(ValueError):
    """Not a whole PNG: `not_a_png` or `truncated_png`, as simple-story-chat's `stripPngMetadata` says."""


def check(value: Any, copies: Any, *, cls: str, reader_nodes: frozenset[str], max_nodes: int,
          owns: Callable[[str], bool]) -> tuple[dict[str, Any], list[str], int]:
    """The graph, its copies and how many of the caller's uploads it reads, or `graph_refused`. `owns` says whether an
    upload name is a live upload of the caller's scope."""
    if not isinstance(value, dict) or not 0 < len(value) <= max_nodes:
        raise ServiceError("graph_refused")
    references = 0
    for node_id, node in value.items():
        if not NODE_ID.fullmatch(node_id) or not isinstance(node, dict) or not set(node) <= NODE_FIELDS:
            raise ServiceError("graph_refused")
        kind, inputs = node.get("class_type"), node.get("inputs")
        if not isinstance(kind, str) or not isinstance(inputs, dict):
            raise ServiceError("graph_refused")
        if cls != "internal" and kind not in reader_nodes:
            raise ServiceError("graph_refused")
        if kind == "LoadImage":
            image = inputs.get("image")
            upload = UPLOADED.fullmatch(image) if isinstance(image, str) else None
            if upload is not None and owns(upload["name"]):
                references += 1
            elif cls != "internal" or not isinstance(image, str) or not PLAIN_INPUT.fullmatch(image):
                raise ServiceError("graph_refused")
        # A temporary file is some reader's reference or picture: only the caller's own uploads may be named.
        for text in _strings(inputs):
            if text.endswith(TEMP):
                upload = UPLOADED.fullmatch(text)
                if upload is None or not owns(upload["name"]):
                    raise ServiceError("graph_refused")
    if copies is None:
        copies = []
    if not isinstance(copies, list) or len(copies) > MAX_COPIES or len(set(map(str, copies))) != len(copies) \
            or not all(isinstance(node, str) and node in value for node in copies):
        raise ServiceError("graph_refused")
    return value, copies, references


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for one in value.values() for text in _strings(one)]
    if isinstance(value, list):
        return [text for one in value for text in _strings(one)]
    return []


def with_nonce(graph: dict[str, Any], nonce: str) -> dict[str, Any]:
    """The graph with one fresh key on every preview node, which is what simple-story-chat's `freshPreviews` does: a
    preview's cached output would name an earlier job's file, which the sweeper has deleted."""
    return {node_id: {**node, "inputs": {**node["inputs"], "nonce": nonce}} if node.get("class_type") == "PreviewImage"
            else node for node_id, node in graph.items()}


def strip_png(data: bytes) -> bytes:
    """The picture without its text chunks, which hold the whole prompt and graph: only IHDR, PLTE, tRNS, IDAT and
    IEND are kept, as simple-story-chat's `stripPngMetadata` keeps them."""
    if len(data) < 8 or data[:8] != PNG_SIGNATURE:
        raise NotPng("not_a_png")
    kept, at = [data[:8]], 8
    while at + 8 <= len(data):
        length, kind = int.from_bytes(data[at:at + 4], "big"), data[at + 4:at + 8]
        end = at + 12 + length
        if end > len(data):
            raise NotPng("truncated_png")
        if kind in KEPT_CHUNKS:
            kept.append(data[at:end])
        at = end
        if kind == b"IEND":
            break
    if len(kept) < 3:
        raise NotPng("not_a_png")
    return b"".join(kept)


def image_type(data: bytes) -> str | None:
    """The upload's type by its magic bytes, as simple-story-chat's `imageFormat` tells them."""
    if data[:8] == PNG_SIGNATURE:
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def js_order(keys: list[str]) -> list[str]:
    """Keys in the order a JavaScript object gives them, which is how the bot picked a record's pictures: integer
    keys first, ascending, then the rest as they came."""
    def integer(key: str) -> bool:
        return key.isascii() and key.isdigit() and (key == "0" or not key.startswith("0")) and int(key) < 2**32 - 1

    return sorted((key for key in keys if integer(key)), key=int) + [key for key in keys if not integer(key)]


Shown = tuple[str, dict[str, Any]]  # a picture of the record: its output node, and ComfyUI's name of its file


def pictures_of(record: dict[str, Any], copies: list[str]) -> tuple[Shown | None, list[Shown]]:
    """The frame, the first picture of the outputs that are not copies, and the first picture of each copy in the
    order asked, a copy without one left out: what simple-story-chat's `follow` fetches."""
    outputs = record.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else {}

    def images(node: str) -> list[dict[str, Any]]:
        output = outputs.get(node)
        found = output.get("images") if isinstance(output, dict) else None
        return [one for one in found if isinstance(one, dict)] if isinstance(found, list) else []

    frames = [(node, image) for node in js_order(list(outputs)) if node not in copies for image in images(node)]
    return (frames[0] if frames else None), [(node, images(node)[0]) for node in copies if images(node)]

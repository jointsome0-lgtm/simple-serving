"""The picture gateway's configuration (docs/pictures.md, sections 4, 6 and 7).

The file has the text gateway's form (`simple_serving/config.py`), named by the same SIMPLE_SERVING_CONFIG, with
`"lane": "pictures"` and the picture lane's own fields: the limits of each class in jobs, wall time and uploads, the
reader allowlist, and the engine facts that the health check expects. Tests and the dev launcher build it from a
block whose test keys are in the clear (`from_block`).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..config import (
    CLASSES,
    IDLE_TIMEOUT_S,
    SHA256_HEX,
    ConfigError,
    Key,
    Listener,
    _boolean,
    _engine_url,
    _fields,
    _integer,
    _key,
    _listener,
    _object,
    _seconds,
    _string,
)
from . import LANE

FIELDS = {"lane", "alias", "engine_url", "listen", "keys", "limits", "reader_nodes", "expect", "drain_deadline_s",
          "idle_timeout_s", "health_interval_s", "max_connections", "body_limit_bytes", "upload_limit_bytes",
          "max_nodes", "versions"}
NODE_CLASS = re.compile(r"[A-Za-z0-9_]{1,64}")
# The reader allowlist of phase 1: the nodes of the bot's graphs as they are drawn today, with the saving node turned
# into a preview (simple-story-chat's `previewOnly`), its references and the kitchen's attention. No node that saves.
# The manifest's READER_NODES says it on the card.
READER_NODES = ("UNETLoader", "CLIPLoader", "VAELoader", "CheckpointLoaderSimple", "CLIPTextEncode",
                "ConditioningZeroOut", "TextEncodeQwenImage21", "QwenImage21Cache", "EmptyLatentImage", "KSampler",
                "VAEDecode", "PreviewImage", "ModelAttentionBackend", "LoadImage", "ImageScale")
# The engine the dev launcher and the Python fake stand for: the bot card's (card/pictures/image-manifest.env).
CARD_ENGINE = {"comfyui": "0.37.0", "pytorch": "2.11.0+cu130", "triton": True}
# The provisional values of section 7, until a card has measured its own.
PROVISIONAL: dict[str, Any] = {
    "limits": {
        "reader": {"jobs": 32, "wall_s": 1860, "uploads": 32},
        "agent": {"jobs": 4, "wall_s": 1860, "uploads": 16},
        "internal": {"jobs": 32, "wall_s": 3600, "uploads": 32},
        "external": {"jobs": 0, "wall_s": 60, "uploads": 0},
    },
    "drain_deadline_s": 60,
}


@dataclass(frozen=True)
class PictureLimits:
    jobs: int  # accepted jobs and stops not yet confirmed, at once
    wall_s: float  # from the acceptance to the last event
    uploads: int  # a scope's live uploads


@dataclass(frozen=True)
class Expect:
    """The engine facts a check must find before the service is ready (section 6)."""

    comfyui: str
    pytorch: str
    triton: bool


@dataclass(frozen=True)
class PictureConfig:
    alias: str
    engine_url: str
    keys: tuple[Key, ...]
    limits: Mapping[str, PictureLimits]
    reader_nodes: frozenset[str]
    expect: Expect
    drain_deadline_s: float
    idle_timeout_s: float
    health_interval_s: float
    public: Listener
    control: Listener
    max_connections: int
    body_limit_bytes: int
    upload_limit_bytes: int
    max_nodes: int
    versions: Mapping[str, str]


def load(path: str) -> PictureConfig:
    """Read the picture gateway's configuration file, whose keys are SHA-256 hashes."""
    try:
        with open(path, "rb") as file:
            data = json.load(file)
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot read the configuration file ({type(error).__name__})") from None
    data = _object(data, "configuration")
    entries = data.get("keys")
    if not isinstance(entries, list):
        raise ConfigError("keys must be a list")
    keys = []
    for index, entry in enumerate(entries):
        where = f"keys[{index}]"
        entry = _object(entry, where)
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
            raise ConfigError(f"{where}.sha256 must be 64 lowercase hex digits")
        keys.append(_key(bytes.fromhex(digest), {k: v for k, v in entry.items() if k != "sha256"}, where))
    return _config(data, tuple(keys))


def from_block(block: Mapping[str, Any], *, engine_url: str, public: Listener | None = None,
               control: Listener | None = None) -> PictureConfig:
    """The configuration from a block whose keys are test keys in the clear, with the provisional limits, the phase 1
    allowlist and the card's engine facts where the block leaves them out."""
    data = {"alias": "dev-pictures", "reader_nodes": list(READER_NODES), "expect": dict(CARD_ENGINE), **PROVISIONAL,
            **_object(block, "block")}
    data["limits"] = {**PROVISIONAL["limits"], **_object(data.get("limits"), "limits")}
    entries = _object(data.pop("keys", None), "keys")
    keys = tuple(_key(hashlib.sha256(raw.encode()).digest(), entry, f"keys[{index}]")
                 for index, (raw, entry) in enumerate(entries.items()))
    listen = {"public": public or Listener("127.0.0.1", 0), "control": control or Listener("127.0.0.1", 0)}
    data.update(lane=LANE, engine_url=engine_url,
                listen={name: {"host": listener.host, "port": listener.port} for name, listener in listen.items()})
    return _config(data, keys)


def _config(data: dict[str, Any], keys: tuple[Key, ...]) -> PictureConfig:
    _fields(data, "the configuration", FIELDS)
    if data.get("lane") != LANE:
        raise ConfigError(f'lane must be "{LANE}": this is the picture gateway')
    if not keys:
        raise ConfigError("keys must hold at least one key")
    if len({key.digest for key in keys}) != len(keys) or len({key.label for key in keys}) != len(keys):
        raise ConfigError("keys must differ from each other, and so must their labels")
    if any(key.vision for key in keys):
        raise ConfigError("the picture lane has no text vision permission")
    if any(key.outside for key in keys):
        raise ConfigError("the picture lane has no outside keys")
    limits = _fields(data.get("limits"), "limits", set(CLASSES))
    listen = _fields(data.get("listen"), "listen", {"public", "control"})
    nodes = data.get("reader_nodes")
    if not isinstance(nodes, list) or not nodes or not all(isinstance(n, str) and NODE_CLASS.fullmatch(n)
                                                           for n in nodes):
        raise ConfigError("reader_nodes must list node classes")
    expect = _fields(data.get("expect"), "expect", {"comfyui", "pytorch", "triton"})
    versions = _object(data.get("versions", {}), "versions")
    if not all(isinstance(value, str) for value in versions.values()):
        raise ConfigError("versions must map names to strings")
    return PictureConfig(
        alias=_string(data.get("alias"), "alias"),
        engine_url=_engine_url(data.get("engine_url")).rstrip("/"),
        keys=keys,
        limits=MappingProxyType({name: _limits(limits.get(name), f"limits.{name}") for name in CLASSES}),
        reader_nodes=frozenset(nodes),
        expect=Expect(comfyui=_string(expect.get("comfyui"), "expect.comfyui"),
                      pytorch=_string(expect.get("pytorch"), "expect.pytorch"),
                      triton=_boolean(expect.get("triton"), "expect.triton")),
        drain_deadline_s=_seconds(data.get("drain_deadline_s"), "drain_deadline_s"),
        idle_timeout_s=_seconds(data.get("idle_timeout_s", IDLE_TIMEOUT_S), "idle_timeout_s"),
        health_interval_s=_seconds(data.get("health_interval_s", 5), "health_interval_s"),
        public=_listener(listen.get("public"), "listen.public"),
        control=_listener(listen.get("control"), "listen.control", loopback=True),
        max_connections=_integer(data.get("max_connections", 64), "max_connections", 1),
        body_limit_bytes=_integer(data.get("body_limit_bytes", 1 << 20), "body_limit_bytes", 1),
        upload_limit_bytes=_integer(data.get("upload_limit_bytes", 10 << 20), "upload_limit_bytes", 1),
        max_nodes=_integer(data.get("max_nodes", 64), "max_nodes", 1),
        versions=MappingProxyType(dict(versions)),
    )


def _limits(value: Any, path: str) -> PictureLimits:
    value = _fields(value, path, {"jobs", "wall_s", "uploads"})
    return PictureLimits(jobs=_integer(value.get("jobs"), f"{path}.jobs", 0),
                         wall_s=_seconds(value.get("wall_s"), f"{path}.wall_s"),
                         uploads=_integer(value.get("uploads"), f"{path}.uploads", 0))

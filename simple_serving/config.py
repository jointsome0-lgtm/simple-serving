"""The gateway's configuration: model, engine, listeners, keys and limits.

The gateway reads a JSON file named by SIMPLE_SERVING_CONFIG (`load`). The file holds each key as its SHA-256 hash,
never the key itself. Tests and the dev launcher build the same configuration from the `service` block of the shared
cases (`from_service_block`), whose test keys are hashed as they are read.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

ENV_VAR = "SIMPLE_SERVING_CONFIG"
# The classes in the order of contract section 2. A freed shared place goes to the first class in this order.
CLASSES = ("reader", "agent", "internal", "external")
SHARED = ("agent", "internal", "external")  # the classes that share places; readers have their own

FIELDS = {"alias", "engine_url", "listen", "context_tokens", "body_limit_bytes", "max_connections", "keys", "limits",
          "count_limits", "engine_priority", "drain_deadline_s", "idle_timeout_s", "health_interval_s", "versions"}
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
LOOPBACK_V4, LOOPBACK_V6 = ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_address("::1")


class ConfigError(Exception):
    """A configuration the gateway cannot start with. The message names the field, never its value."""


@dataclass(frozen=True)
class Key:
    digest: bytes  # SHA-256 of the key
    label: str  # what logs show instead of the key
    classes: frozenset[str]
    default: str | None
    scopes: bool  # may name cache scopes
    control: bool  # may call the control routes

    @property
    def outside(self) -> bool:
        return "external" in self.classes


@dataclass(frozen=True)
class ClassLimits:
    active: int
    waiting: int
    input_tokens: int | None  # None means the context
    max_tokens: int
    wall_s: float


@dataclass(frozen=True)
class Listener:
    host: str
    port: int  # 0 picks a free port


@dataclass(frozen=True)
class Config:
    alias: str
    engine_url: str
    context_tokens: int | None  # the gateway's own limit, when it is smaller than the engine's
    body_limit_bytes: int
    keys: tuple[Key, ...]
    limits: Mapping[str, ClassLimits]
    per_key_active: int  # class external, for each outside key
    per_key_waiting: int
    shared_places: int  # the places that agent, internal and external share
    count_active: int
    count_per_outside_key: int
    engine_priority: Mapping[str, int]
    drain_deadline_s: float
    idle_timeout_s: float
    public: Listener
    control: Listener
    max_connections: int
    health_interval_s: float
    versions: Mapping[str, str]


def load(path: str) -> Config:
    """Read the gateway's configuration file, whose keys are SHA-256 hashes."""
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
        path = f"keys[{index}]"
        entry = _object(entry, path)
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
            raise ConfigError(f"{path}.sha256 must be 64 lowercase hex digits")
        keys.append(_key(bytes.fromhex(digest), {k: v for k, v in entry.items() if k != "sha256"}, path))
    return _config(data, tuple(keys))


def from_service_block(block: Mapping[str, Any], *, engine_url: str, public: Listener | None = None,
                       control: Listener | None = None, **extra: Any) -> Config:
    """Build the configuration from a `service` block of the shared cases, whose keys are test keys in the clear."""
    data = dict(_object(block, "service"))
    entries = _object(data.pop("keys", None), "keys")
    keys = tuple(_key(hashlib.sha256(raw.encode()).digest(), entry, f"keys[{index}]")
                 for index, (raw, entry) in enumerate(entries.items()))
    listen = {"public": public or Listener("127.0.0.1", 0), "control": control or Listener("127.0.0.1", 0)}
    data.update(engine_url=engine_url,
                listen={name: {"host": listener.host, "port": listener.port} for name, listener in listen.items()},
                **extra)
    return _config(data, keys)


def _config(data: dict[str, Any], keys: tuple[Key, ...]) -> Config:
    _fields(data, "the configuration", FIELDS)
    if not keys:
        raise ConfigError("keys must hold at least one key")
    if len({key.digest for key in keys}) != len(keys) or len({key.label for key in keys}) != len(keys):
        raise ConfigError("keys must differ from each other, and so must their labels")
    limits = _fields(data.get("limits"), "limits", {*CLASSES, "external_per_key", "shared"})
    per_key = _fields(limits.get("external_per_key"), "limits.external_per_key", {"active", "waiting"})
    shared = _fields(limits.get("shared"), "limits.shared", {"active"})
    counts = _fields(data.get("count_limits"), "count_limits", {"active", "per_outside_key"})
    priority = _fields(data.get("engine_priority"), "engine_priority", set(CLASSES))
    listen = _fields(data.get("listen"), "listen", {"public", "control"})
    engine_url = _engine_url(data.get("engine_url"))
    versions = _object(data.get("versions", {}), "versions")
    if not all(isinstance(value, str) for value in versions.values()):
        raise ConfigError("versions must map names to strings")
    return Config(
        alias=_string(data.get("alias"), "alias"),
        engine_url=engine_url.rstrip("/"),
        context_tokens=_optional_integer(data.get("context_tokens"), "context_tokens", 1),
        body_limit_bytes=_integer(data.get("body_limit_bytes", 2_000_000), "body_limit_bytes", 1),
        keys=keys,
        limits=MappingProxyType({name: _class_limits(limits.get(name), f"limits.{name}") for name in CLASSES}),
        per_key_active=_integer(per_key.get("active"), "limits.external_per_key.active", 1),
        per_key_waiting=_integer(per_key.get("waiting"), "limits.external_per_key.waiting", 0),
        shared_places=_integer(shared.get("active"), "limits.shared.active", 1),
        count_active=_integer(counts.get("active"), "count_limits.active", 1),
        count_per_outside_key=_integer(counts.get("per_outside_key"), "count_limits.per_outside_key", 1),
        engine_priority=MappingProxyType({name: _integer(priority.get(name), f"engine_priority.{name}", None)
                                          for name in CLASSES}),
        drain_deadline_s=_seconds(data.get("drain_deadline_s"), "drain_deadline_s"),
        idle_timeout_s=_seconds(data.get("idle_timeout_s", 780), "idle_timeout_s"),
        public=_listener(listen.get("public"), "listen.public"),
        control=_listener(listen.get("control"), "listen.control", loopback=True),
        max_connections=_integer(data.get("max_connections", 64), "max_connections", 1),
        health_interval_s=_seconds(data.get("health_interval_s", 5), "health_interval_s"),
        versions=MappingProxyType(dict(versions)),
    )


def _key(digest: bytes, entry: Any, path: str) -> Key:
    entry = _fields(entry, path, {"label", "classes", "default", "scopes", "control"})
    classes = entry.get("classes")
    if not isinstance(classes, list) or len(set(classes)) != len(classes) or not set(classes) <= set(CLASSES):
        raise ConfigError(f"{path}.classes must list distinct classes")
    default = entry.get("default")
    if default is not None and default not in classes:
        raise ConfigError(f"{path}.default must be one of the key's classes, or null")
    key = Key(digest=digest, label=_string(entry.get("label"), f"{path}.label"), classes=frozenset(classes),
              default=default, scopes=_boolean(entry.get("scopes"), f"{path}.scopes"),
              control=_boolean(entry.get("control"), f"{path}.control"))
    # Contract section 2: outside keys use `external` only, and each is its own cache scope.
    if key.outside and (key.classes != {"external"} or key.scopes):
        raise ConfigError(f"{path}: an outside key may use the class external only and may not name scopes")
    return key


def _class_limits(value: Any, path: str) -> ClassLimits:
    value = _fields(value, path, {"active", "waiting", "input_tokens", "max_tokens", "wall_s"})
    return ClassLimits(
        active=_integer(value.get("active"), f"{path}.active", 1),
        waiting=_integer(value.get("waiting"), f"{path}.waiting", 0),
        input_tokens=_optional_integer(value.get("input_tokens"), f"{path}.input_tokens", 1),
        max_tokens=_integer(value.get("max_tokens"), f"{path}.max_tokens", 1),
        wall_s=_seconds(value.get("wall_s"), f"{path}.wall_s"),
    )


def _engine_url(value: Any) -> str:
    """The engine's base URL, on loopback like the engine itself (contract section 1)."""
    url = _string(value, "engine_url")
    try:
        parts = urlsplit(url)
        loopback = parts.scheme in ("http", "https") and _is_loopback(parts.hostname)
    except ValueError:  # such as an IPv6 address without its closing bracket
        loopback = False
    if not loopback:
        raise ConfigError("engine_url must be an http URL whose host is a loopback IP address, 127.0.0.0/8 or ::1")
    return url


def _listener(value: Any, path: str, *, loopback: bool = False) -> Listener:
    value = _fields(value, path, {"host", "port"})
    host = _string(value.get("host"), f"{path}.host")
    if loopback and not _is_loopback(host):
        raise ConfigError(f"{path}.host must be a loopback IP address, 127.0.0.0/8 or ::1")
    port = _integer(value.get("port"), f"{path}.port", 0)
    if port > 65535:
        raise ConfigError(f"{path}.port must be at most 65535")
    return Listener(host=host, port=port)


def _is_loopback(host: str | None) -> bool:
    """Whether a host is a loopback IP address. A name is not, localhost included: what a name stands for is up to
    the machine's resolver."""
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    return address in LOOPBACK_V4 or address == LOOPBACK_V6


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be an object")
    return value


def _fields(value: Any, path: str, known: set[str]) -> dict[str, Any]:
    """An object whose fields are all known: a misspelt limit must not pass unnoticed."""
    value = _object(value, path)
    unknown = sorted(set(value) - known)
    if unknown:
        raise ConfigError(f"{path} has unknown fields: {', '.join(unknown)}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be true or false")
    return value


def _integer(value: Any, path: str, minimum: int | None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ConfigError(f"{path} must be an integer" + (f" of at least {minimum}" if minimum is not None else ""))
    return value


def _optional_integer(value: Any, path: str, minimum: int) -> int | None:
    return None if value is None else _integer(value, path, minimum)


def _seconds(value: Any, path: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{path} must be a positive number of seconds")
    return float(value)

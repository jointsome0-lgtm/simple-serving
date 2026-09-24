"""Who may do what: keys, classes and cache scopes (contract section 2)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
from dataclasses import dataclass

from .config import Config, Key
from .errors import ServiceError

CLASS_HEADER = "x-simple-serving-class"
SCOPE_HEADER = "x-simple-serving-scope"
READER_SCOPE = re.compile(r"reader\.[A-Za-z0-9_-]{8,64}")
NAMED_SCOPES = ("agent", "internal")


@dataclass(frozen=True)
class Caller:
    """A request's key, class and cache scope. The scope's opaque part never leaves the gateway."""

    key: Key
    cls: str
    scope: str | None  # `reader.<opaque>`, `agent` or `internal`; None for an outside key, which is its own scope

    @property
    def scope_kind(self) -> str:
        """What a log may show of the scope: `reader`, `agent`, `internal` or `external`."""
        return "external" if self.scope is None else self.scope.partition(".")[0]

    @property
    def salt_input(self) -> str:
        """What the cache salt is made of. Our scopes and outside keys have separate namespaces, so an outside key
        never shares a salt with one of our scopes, whatever its label; labels are unique."""
        return f"key:{self.key.label}" if self.scope is None else f"scope:{self.scope}"

    @property
    def outside_key(self) -> Key | None:
        """The key whose own caps apply to this request, if it is an outside key."""
        return self.key if self.key.outside else None


def find_key(config: Config, authorization: bytes | None) -> Key | None:
    """The key of an `Authorization: Bearer <key>` header, found by its hash and compared in constant time."""
    scheme, _, presented = (authorization or b"").partition(b" ")
    if scheme.lower() != b"bearer" or not presented:
        return None
    digest = hashlib.sha256(presented).digest()
    found = None
    for key in config.keys:  # no early exit: the time does not depend on which key matched
        if hmac.compare_digest(digest, key.digest):
            found = key
    return found


def resolve(key: Key, class_header: str | None, scope_header: str | None) -> Caller:
    """The class and the scope of a request, checked in the contract's order: first the class, then the scope."""
    cls = key.default if class_header is None else class_header
    if cls is None or cls not in key.classes:
        raise ServiceError("class_not_allowed")
    if scope_header is not None and not key.scopes:
        raise ServiceError("scope_not_allowed")
    if key.outside:
        return Caller(key=key, cls=cls, scope=None)
    if scope_header is None:
        scope = cls
    elif READER_SCOPE.fullmatch(scope_header) or scope_header in NAMED_SCOPES:
        scope = scope_header
    else:
        raise ServiceError("invalid_request")
    # Readers never share a cache by accident: a reader's request names its reader.
    if cls == "reader" and not scope.startswith("reader."):
        raise ServiceError("invalid_request")
    return Caller(key=key, cls=cls, scope=scope)


def cache_salt(secret: bytes, caller: Caller) -> str:
    """The engine's `cache_salt` for a request: an HMAC under a secret made at every start, 43 characters of base64."""
    digest = hmac.new(secret, caller.salt_input.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

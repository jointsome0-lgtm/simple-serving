"""Places for generations and counts (contract sections 5 and 7).

A generation is Waiting from the moment it may wait for a place until the gateway hands it to the engine, and Active
from then until its engine request has ended locally. Each class has caps for both, and each outside key has its own.
Readers have places of their own, their Active cap; `agent`, `internal` and `external` share a few places, each class
within its cap. When a shared place frees, the first waiting request of the first class in the order of section 2 takes
it, among the classes that have room; within a class the order is arrival. A request with room takes a free place at
once: a request still waiting while a place is free is held by its own caps.

Counts have places of their own, shared by all classes. A count past them waits, in the same order; an outside key has
a cap on its counts, waiting ones included.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from collections.abc import Iterator

from .config import CLASSES, SHARED, Config, Key
from .errors import ServiceError


class Ticket:
    """One request's place: waiting, then active, then left."""

    def __init__(self, cls: str, owner: Key | None) -> None:
        self.cls = cls
        self.owner = owner  # the outside key whose own caps apply, if any
        self.state = "waiting"
        self.granted: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    @property
    def waiting(self) -> bool:
        return self.state == "waiting"

    @property
    def active(self) -> bool:
        return self.state == "active"


class Line:
    """Waiting tickets, one queue per class, taken in the order of section 2 and then by arrival."""

    def __init__(self) -> None:
        self._queues: dict[str, deque[Ticket]] = {cls: deque() for cls in CLASSES}

    def add(self, ticket: Ticket) -> None:
        self._queues[ticket.cls].append(ticket)

    def remove(self, ticket: Ticket) -> None:
        self._queues[ticket.cls].remove(ticket)

    def counts(self) -> dict[str, int]:
        return {cls: len(queue) for cls, queue in self._queues.items()}

    def __len__(self) -> int:
        return sum(map(len, self._queues.values()))

    def in_order(self) -> Iterator[Ticket]:
        for cls in CLASSES:
            yield from self._queues[cls]


def _grant(ticket: Ticket) -> None:
    ticket.state = "active"
    if not ticket.granted.done():  # a stopped request's future is cancelled; its cleanup frees the place
        ticket.granted.set_result(None)


class Admission:
    """The Active and Waiting caps of generations (section 7)."""

    def __init__(self, config: Config) -> None:
        self._limits = config.limits
        self._per_key_active = config.per_key_active
        self._per_key_waiting = config.per_key_waiting
        self._shared_places = config.shared_places
        self._line = Line()
        self._active: Counter[str] = Counter()
        self._owner_active: Counter[Key] = Counter()
        self._owner_waiting: Counter[Key] = Counter()

    def enter(self, cls: str, owner: Key | None) -> Ticket:
        """A new ticket, active at once or waiting. Past the Waiting caps it is refused with 429 queue_full."""
        ticket = Ticket(cls, owner)
        if self._has_room(ticket):
            self._activate(ticket)
        elif self._line.counts()[cls] >= self._limits[cls].waiting or (
                owner is not None and self._owner_waiting[owner] >= self._per_key_waiting):
            raise ServiceError("queue_full")
        else:
            self._line.add(ticket)
            if owner is not None:
                self._owner_waiting[owner] += 1
        return ticket

    def leave(self, ticket: Ticket) -> None:
        """Take a ticket out. A waiting ticket leaves the line; an active one frees its place for the next."""
        if ticket.state == "left":
            return
        if ticket.waiting:
            self._line.remove(ticket)
            if ticket.owner is not None:
                self._owner_waiting[ticket.owner] -= 1
        else:
            self._active[ticket.cls] -= 1
            if ticket.owner is not None:
                self._owner_active[ticket.owner] -= 1
        ticket.state = "left"
        self._dispatch()

    def active_counts(self) -> dict[str, int]:
        return {cls: self._active[cls] for cls in CLASSES}

    def waiting_counts(self) -> dict[str, int]:
        return self._line.counts()

    def _dispatch(self) -> None:
        """Give free places to waiting tickets, until no waiting ticket has room."""
        while (ticket := next((t for t in self._line.in_order() if self._has_room(t)), None)) is not None:
            self._line.remove(ticket)
            if ticket.owner is not None:
                self._owner_waiting[ticket.owner] -= 1
            self._activate(ticket)

    def _has_room(self, ticket: Ticket) -> bool:
        if self._active[ticket.cls] >= self._limits[ticket.cls].active:
            return False
        if ticket.owner is not None and self._owner_active[ticket.owner] >= self._per_key_active:
            return False
        return ticket.cls not in SHARED or sum(self._active[cls] for cls in SHARED) < self._shared_places

    def _activate(self, ticket: Ticket) -> None:
        self._active[ticket.cls] += 1
        if ticket.owner is not None:
            self._owner_active[ticket.owner] += 1
        _grant(ticket)


class CountPlaces:
    """At most `count_limits.active` counts at once, and `per_outside_key` for one outside key (section 5).

    A count past the places waits for one, in the order of section 2 and then by arrival. An outside key's count past
    its own cap, waiting counts included, is refused with 429 queue_full. The count inside a generation takes a place
    here as well.
    """

    def __init__(self, config: Config) -> None:
        self._places = config.count_active
        self._per_key = config.count_per_outside_key
        self._line = Line()
        self._running = 0
        self._by_owner: Counter[Key] = Counter()

    def enter(self, cls: str, owner: Key | None) -> Ticket:
        if owner is not None and self._by_owner[owner] >= self._per_key:
            raise ServiceError("queue_full")
        ticket = Ticket(cls, owner)
        if owner is not None:
            self._by_owner[owner] += 1
        if self._running < self._places:
            self._running += 1
            _grant(ticket)
        else:
            self._line.add(ticket)
        return ticket

    def leave(self, ticket: Ticket) -> None:
        if ticket.state == "left":
            return
        if ticket.waiting:
            self._line.remove(ticket)
        else:
            self._running -= 1
        if ticket.owner is not None:
            self._by_owner[ticket.owner] -= 1
        ticket.state = "left"
        while self._running < self._places and (first := next(self._line.in_order(), None)) is not None:
            self._line.remove(first)
            self._running += 1
            _grant(first)

    @property
    def running(self) -> int:
        return self._running

    @property
    def waiting(self) -> int:
        return len(self._line)

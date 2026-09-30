"""The gateway's own metrics, in Prometheus's text format 0.0.4, written by hand. docs/metrics.md is the catalog.

The card's wheels are pinned by hash, so the gateway takes no metrics library: it needs a few counters, gauges and
histograms. Every label comes from a fixed set: a listener, a route of contract section 3, a class, an outcome, a
code of section 9, an HTTP status or a service status. No label holds a key, a scope, a request's id or anything of a
body (section 10).

Each request is counted once, when its log row is written (`Guard` in app.py): its HTTP status, and for inference or
a picture mutation its outcome. The outcome comes from the row: the code the gateway answered or began to answer,
whether the request was accepted, and whether it was cancelled and by which stop. A stream that fails after its 200,
with an error event, is not a success. The stages of an accepted request are observed as each ends: the wait for a count
place, the count, the wait for a generation place, the whole wait until the engine has the request, the first token
and the end of the engine's stream. A stage that was not reached is not observed. The gauges are read from the service
at each scrape, and a scrape changes nothing: it is not work of ours, so it holds no card awake (section 8).
"""

from __future__ import annotations

import asyncio
import bisect
import math
import time
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, ClassVar

from .config import CLASSES
from .errors import STATUS
from .inference import STOP_CODES

if TYPE_CHECKING:
    from .log import RequestRecord
    from .service import Lifecycle

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
PREFIX = "simple_serving_"
NONE = "none"  # the class of a request refused before its class was known, and the code of one without a code
GENERATION, COUNT = "/v1/chat/completions", "/v1/chat/completions/input_tokens"
PICTURE = "/v1/pictures"
PICTURE_METHODS = {PICTURE: ("POST",), "/v1/pictures/{job}/stop": ("POST",),
                   "/v1/pictures/uploads/{name}": ("PUT", "DELETE"), "/v1/pictures/inputs/{name}": ("PUT",)}
ROUTES = (GENERATION, COUNT, *PICTURE_METHODS, "/v1/pictures/engine", "/v1/models", "/v1/state", "/v1/control/drain",
          "/v1/control/sleep", "/v1/control/open", "/metrics", "/metrics/engine")
OTHER = "other"  # a path outside section 3
LISTENERS = ("public", "control")
SERVICE_STATUSES = ("starting", "ready", "draining", "drained", "failed")
HTTP_STATUSES = (*(str(status) for status in sorted({200, 201, 202, 204, *STATUS.values()})), NONE)
# The outcome of a generation or a count, and the codes each may come with (docs/metrics.md, "Outcomes").
OUTCOMES: Mapping[str, tuple[str, ...]] = {
    "success": (NONE,),
    "rejected": ("invalid_request", "unsupported_field", "limit_exceeded", "context_limit", "min_p_unavailable",
                 "unauthorized", "class_not_allowed", "scope_not_allowed", "body_too_large", "queue_full", "timeout",
                 "graph_refused", "graph_rejected", "forbidden", "not_found", "stale_boot", "stale_generation",
                 "sleep_pending"),
    "unavailable": ("starting", "engine_unavailable", "socket_unavailable", NONE),
    "draining": ("draining", "drained"),
    "timeout": ("timeout",),
    "engine_error": ("engine_unavailable", "socket_unavailable", "picture_failed", "stop_unconfirmed"),
    "internal_error": ("internal_error",),
    "client_disconnect": (NONE,),
    "client_cancelled": (NONE,),
}
# Seconds, from the few milliseconds of a free place to the wall times of section 7: 300 s for readers, 900 s for
# agent and internal.
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0)
# The stages of an accepted request, each with its help and whether it has the route label: a count and a
# generation's count stage share the first two.
STAGES: Mapping[str, tuple[str, bool]] = {
    "count_place_wait": ("From entering the count stage to a count place: the gateway's count queue.", True),
    "count": ("The engine's count of the input, once it has a count place.", True),
    "generation_place_wait": ("From entering generation admission to a place: the gateway's own queue.", False),
    "wait": ("From acceptance until the engine has the generation: wait_ms, the count stage included.", False),
    "time_to_first_token": (("From acceptance to the first token received from the engine, reasoning included: "
                             "first_token_ms, observed only when a token arrived. The gateway's time, not the "
                             "reader's."), False),
    "total": (("From acceptance to the end of the engine's stream, for streams that ended well: total_ms. The "
               "gateway's time, not the reader's."), False),
}
TOKENS: Mapping[str, str] = {
    "input_tokens": "Input tokens, as the gateway counted them, of generations whose engine stream ended well.",
    "output_tokens": "Tokens generated, reasoning included, in generations whose engine stream ended well.",
    "cached_tokens": "Input tokens the engine read from its prefix cache, where it said, in those generations.",
    "count_mismatches": "Generations whose engine prompt_tokens differed from the gateway's count (section 5).",
}
LABELS: Mapping[str, tuple[str, ...]] = {
    "listener": LISTENERS, "route": (*ROUTES, OTHER), "class": (*CLASSES, NONE), "outcome": tuple(OUTCOMES),
    "code": (*sorted(STATUS), NONE), "status": (*HTTP_STATUSES, *SERVICE_STATUSES),
}


def outcome_of(code: str, accepted: bool) -> str:
    """The outcome of a request that ended with an error code, whether its answer went out or was given up."""
    if code == "timeout":
        return "timeout" if accepted else "rejected"  # before acceptance, a body that did not arrive in 30 s
    if code in ("engine_unavailable", "socket_unavailable"):
        return "engine_error" if accepted else "unavailable"
    if code in ("picture_failed", "stop_unconfirmed"):
        return "engine_error"
    if code in ("draining", "drained"):
        return "draining"
    if code == "starting":
        return "unavailable"
    if code == "internal_error":
        return "internal_error"
    return "rejected"  # the request, its key or a cap refused it


def request_outcome(record: RequestRecord, *, escaped: bool = False) -> tuple[str, str]:
    """The outcome and code of a generation or a count, from its log row. `escaped`: an exception reached the
    listener's last resort."""
    code = record.code
    if code is None and record.cancelled and record.stop is not None:
        code = STOP_CODES.get(record.stop)  # a stop whose code the gateway could not send
    if code is None and escaped:
        code = "internal_error"
    if code is not None:
        return outcome_of(code, record.accepted), code
    if record.cancelled and record.stop == "client":
        return "client_cancelled", NONE
    if record.cancelled:
        return ("unavailable" if record.stop == "shutdown" else "client_disconnect"), NONE
    return "success", NONE


class Histogram:
    def __init__(self) -> None:
        self.buckets = [0] * (len(BUCKETS) + 1)  # each bucket's own count, the last one above every bound
        self.sum = 0.0
        self.count = 0

    def observe(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        self.buckets[bisect.bisect_left(BUCKETS, seconds)] += 1  # the first bound at or above the value
        self.sum += seconds
        self.count += 1


class Metrics:
    """Counters and lifecycle gauges common to both lanes; each real lane adds its own measurements."""

    request_methods: ClassVar[Mapping[str, tuple[str, ...]]] = {}
    stage_definitions: ClassVar[Mapping[str, tuple[str, bool]]] = {}
    token_definitions: ClassVar[Mapping[str, str]] = {}

    def __init__(self) -> None:
        self.start_time = time.time()
        pairs = [(outcome, code) for outcome, codes in OUTCOMES.items() for code in codes]
        # Every series exists from the start at 0, so that Prometheus's increase() sees the first event of each.
        self.requests = {(route, cls, outcome, code): 0 for route in self.request_methods for cls in (*CLASSES, NONE)
                         for outcome, code in pairs}
        self.responses: dict[tuple[str, str, str], int] = {}
        self.stages = {(stage, route, cls): Histogram() for stage, (_, routed) in self.stage_definitions.items()
                       for route in ((GENERATION, COUNT) if routed else (GENERATION,)) for cls in CLASSES}
        self.tokens = {(name, cls): 0 for name in self.token_definitions for cls in CLASSES}

    def listener(self, name: str, paths: Iterable[str]) -> None:
        """Start the status series of the listener's fixed route templates at 0."""
        for route in (*paths, OTHER):
            route = route if route in ROUTES else OTHER
            for status in HTTP_STATUSES:
                self.responses.setdefault((name, route, status), 0)

    def observe(self, stage: str, cls: str, seconds: float, route: str | None = GENERATION) -> None:
        """One text stage has ended. Picture jobs do not report text timing or token metrics."""
        routed = self.stage_definitions[stage][1]
        self.stages[stage, (route or GENERATION) if routed else GENERATION, cls].observe(seconds)

    def request_ended(self, record: RequestRecord, *, escaped: bool = False) -> None:
        """Count a request once, as its log row is written. Paths stay within the fixed template catalog."""
        route = record.route if record.route in ROUTES else OTHER
        status = NONE if record.status is None else str(record.status)
        key = (record.listener, route, status)
        self.responses[key] = self.responses.get(key, 0) + 1
        if record.method not in self.request_methods.get(route, ()):
            return
        outcome, code = request_outcome(record, escaped=escaped)
        series = (route, record.cls or NONE, outcome, code)
        self.requests[series] = self.requests.get(series, 0) + 1
        if route == GENERATION and record.cls is not None and record.output_tokens is not None:
            self.tokens["input_tokens", record.cls] += record.input_tokens or 0
            self.tokens["output_tokens", record.cls] += record.output_tokens
            self.tokens["cached_tokens", record.cls] += record.cached_tokens or 0
            self.tokens["count_mismatches", record.cls] += record.count_matches is False

    def render(self, service: Lifecycle) -> bytes:
        """The lane's exposition; gauges are snapshots and never keep the card awake."""
        out = Exposition()
        out.family("requests_total", "counter", "Inference and picture mutations by outcome, once at their end.",
                   (({"route": route, "class": cls, "outcome": outcome, "code": code}, value)
                    for (route, cls, outcome, code), value in sorted(self.requests.items())))
        out.family("http_responses_total", "counter", "Requests by listener, route and the HTTP status sent; 'none' "
                   "when none was sent. A stream that fails after its 200 counts as 200 here.",
                   (({"listener": listener, "route": route, "status": status}, value)
                    for (listener, route, status), value in sorted(self.responses.items())))
        for stage, (text, routed) in self.stage_definitions.items():
            out.histogram(f"{stage}_seconds", text,
                          (({"route": route, "class": cls} if routed else {"class": cls}, histogram)
                           for (name, route, cls), histogram in self.stages.items() if name == stage))
        for name, text in self.token_definitions.items():
            out.family(f"{name}_total", "counter", text,
                       (({"class": cls}, self.tokens[name, cls]) for cls in CLASSES))
        self._gauges(out, service)
        return out.text().encode()

    def _gauges(self, out: Exposition, service: Lifecycle) -> None:
        self._common_gauges(out, service, verified=service.idle_since > 0)

    def _common_gauges(self, out: Exposition, service: Lifecycle, *, verified: bool) -> None:
        in_flight = dict.fromkeys(self.request_methods, 0)
        for work in service.work:
            route = work.exchange.record.route
            if route in in_flight:
                in_flight[route] += 1
        # A verified engine may later fail: that does not reset the idle interval.
        idle = 0.0 if service.ours_inflight or not verified else asyncio.get_running_loop().time() - service.idle_since
        out.gauge("start_time_seconds", "When the gateway process started, in Unix time: a new value is a new boot.",
                  [({}, self.start_time)])
        out.gauge("status", "1 for the service's status of /v1/state, 0 for the others.",
                  (({"status": name}, int(name == service.status)) for name in SERVICE_STATUSES))
        out.gauge("sleep_requested", "1 once a sleep has begun: the instance is being stopped.",
                  [({}, int(service.sleep_requested))])
        out.gauge("drain_generation", "The drain generation of /v1/state.", [({}, service.drain_generation)])
        out.gauge("accepted_in_flight", "Accepted work that has not answered yet, by route.",
                  (({"route": route}, value) for route, value in in_flight.items()))
        out.gauge("ours_in_flight", "Requests of reader, agent and internal between their authorization and their "
                  "end: what holds the card awake.", [({}, service.ours_inflight)])
        out.gauge("idle_seconds", "Seconds without work of ours since the last ended, 0 while one runs or before the "
                  "first ready. The service falls asleep at idle_timeout_seconds.", [({}, max(0.0, idle))])
        out.gauge("idle_timeout_seconds", "The idle interval of section 8.", [({}, service.config.idle_timeout_s)])
        out.gauge("wall_time_limit_seconds", "The wall time of each class.",
                  (({"class": cls}, service.config.limits[cls].wall_s) for cls in CLASSES))


class TextMetrics(Metrics):
    """Text admission, token and timing measurements, with the shared request and lifecycle measurements."""

    request_methods: ClassVar[Mapping[str, tuple[str, ...]]] = {GENERATION: ("POST",), COUNT: ("POST",)}
    stage_definitions = STAGES
    token_definitions = TOKENS

    def _gauges(self, out: Exposition, service: Lifecycle) -> None:
        from .service import Service  # noqa: PLC0415 - Service imports this module for its metrics adapter

        assert isinstance(service, Service)
        self._common_gauges(out, service, verified=service.context_tokens is not None)
        config, counts = service.config, service.count_places
        per_class = (
            ("generations_active", "Generations the engine has from the gateway, by class (section 7's Active).",
             service.admission.active_counts()),
            ("generations_waiting", "Generations waiting for a generation place in the gateway, by class.",
             service.admission.waiting_counts()),
            ("counts_active", "Counts running in a count place, a generation's count stage included, by class.",
             counts.active_counts()),
            ("counts_waiting", "Counts waiting for a count place, a generation's count stage included, by class.",
             counts.waiting_counts()),
        )
        for name, text, values in per_class:
            out.gauge(name, text, (({"class": cls}, values[cls]) for cls in CLASSES))
        out.gauge("generation_active_limit", "Section 7's Active cap of each class.",
                  (({"class": cls}, config.limits[cls].active) for cls in CLASSES))
        out.gauge("generation_waiting_limit", "Section 7's Waiting cap of each class.",
                  (({"class": cls}, config.limits[cls].waiting) for cls in CLASSES))
        out.gauge("generation_shared_limit", "The places that agent, internal and external share.",
                  [({}, config.shared_places)])
        out.gauge("count_active_limit", "The count places of section 5.", [({}, config.count_active)])


class PictureMetrics(Metrics):
    """Only gateway-owned picture jobs are counted; ComfyUI's unrelated jobs and private identifiers stay out."""

    request_methods = PICTURE_METHODS

    def _gauges(self, out: Exposition, service: Lifecycle) -> None:
        from .pictures.jobs import WAITING  # noqa: PLC0415 - text installs do not include the picture dependencies
        from .pictures.service import PictureService  # noqa: PLC0415 - the same optional lane dependency

        assert isinstance(service, PictureService)
        self._common_gauges(out, service, verified=service.fingerprint is not None)
        active, waiting, unresolved = (dict.fromkeys(CLASSES, 0) for _ in range(3))
        for job in service.jobs.values():
            (waiting if job.phase in WAITING else active)[job.caller.cls] += 1
        for stop in service.unresolved.values():
            unresolved[stop.cls] += 1
        for name, text, values in (
            ("picture_jobs_active", "Gateway picture jobs drawing or delivering their completed output.", active),
            ("picture_jobs_waiting", "Gateway picture jobs preparing, submitting or pending in ComfyUI.", waiting),
            ("picture_stops_unresolved", "Unconfirmed picture stops that still consume class capacity.", unresolved),
        ):
            out.gauge(name, text, (({"class": cls}, values[cls]) for cls in CLASSES))
        out.gauge("picture_job_limit", "Accepted jobs and unresolved stops allowed at once for each class.",
                  (({"class": cls}, service.config.limits[cls].jobs) for cls in CLASSES))


class Exposition:
    """Lines of the text format 0.0.4: HELP and TYPE, then the samples of a family."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def family(self, name: str, kind: str, text: str, samples: Iterable[tuple[Mapping[str, str], float]]) -> None:
        self._head(name, kind, text)
        self._lines.extend(f"{PREFIX}{name}{_labels(labels)} {_number(value)}" for labels, value in samples)

    def gauge(self, name: str, text: str, samples: Iterable[tuple[Mapping[str, str], float]]) -> None:
        self.family(name, "gauge", text, samples)

    def histogram(self, name: str, text: str, series: Iterable[tuple[Mapping[str, str], Histogram]]) -> None:
        self._head(name, "histogram", text)
        for labels, histogram in series:
            total = 0
            for bound, count in zip((*BUCKETS, math.inf), histogram.buckets, strict=True):
                total += count
                self._lines.append(f"{PREFIX}{name}_bucket{_labels({**labels, 'le': _number(bound)})} {total}")
            self._lines.append(f"{PREFIX}{name}_sum{_labels(labels)} {_number(histogram.sum)}")
            self._lines.append(f"{PREFIX}{name}_count{_labels(labels)} {histogram.count}")

    def text(self) -> str:
        return "\n".join(self._lines) + "\n"

    def _head(self, name: str, kind: str, text: str) -> None:
        self._lines += [f"# HELP {PREFIX}{name} {_escape(text, quote=False)}", f"# TYPE {PREFIX}{name} {kind}"]


def _labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{name}="{_escape(value, quote=True)}"' for name, value in labels.items()) + "}"


def _escape(text: str, *, quote: bool) -> str:
    text = text.replace("\\", "\\\\").replace("\n", "\\n")
    return text.replace('"', '\\"') if quote else text


def _number(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return str(int(value)) if float(value).is_integer() else repr(float(value))

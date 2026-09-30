"""The metric catalog, docs/metrics.md, against the gateway that serves it and the monitoring that reads it.

The catalog is what a gateway in another language would serve, so it must hold the gateway's families as they are:
each name, its type and its labels, and nothing the gateway does not serve. The dashboards, the rules and
prometheus.yml may use no name outside it, and select no label that its name does not have. vLLM's names in it are
checked against vLLM's own source when SIMPLE_SERVING_VLLM_SOURCE names a git checkout of the version that
card/manifest.env pins; otherwise that test is skipped. Nothing here needs the network or a parser beyond the
standard library.
"""

from __future__ import annotations

import collections
import json
import os
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from .support import CONTROL, ROOT, call, running

CATALOG = ROOT / "docs" / "metrics.md"
MONITORING = ROOT / "monitoring"
# A row of the catalog's tables: the name, the type, then the labels, each in backticks.
ROW = re.compile(r"^\| `([^`]+)` \| (counter|gauge|histogram) \|([^|]*)\|", re.MULTILINE)
GATEWAY = "simple_serving_"
VLLM_SOURCE = "SIMPLE_SERVING_VLLM_SOURCE"
# PromQL's words that are not metric names, and the parts of an expression that hold none.
KEYWORDS = frozenset({"and", "or", "unless", "by", "without", "on", "ignoring", "group_left", "group_right", "bool",
                      "offset", "inf", "nan"})
STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|`[^`]*`')
LABEL_LIST = re.compile(r"\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)")
VARIABLE = re.compile(r"\$\{?\w+\}?")  # Grafana's, such as $__rate_interval and $model_name
IDENTIFIER = re.compile(r"(?<![\w.:$])[A-Za-z_:][\w:]*")
SELECTOR = re.compile(r"(?<![\w:])([A-Za-z_:][\w:]*)\s*\{([^}]*)\}")
MATCHER = re.compile(r"(\w+)\s*(?:=~|!~|!=|=)")
# The gateway's and vLLM's names wherever they appear in monitoring/, in a description as much as in a query.
PREFIXED = re.compile(r"(?<![\w:])(?:simple_serving_\w+|vllm:\w+)")
TARGET_LABELS = frozenset({"job", "instance", "lane"})


@dataclass(frozen=True)
class Family:
    kind: str
    labels: frozenset[str]


def catalog() -> dict[str, Family]:
    """Every name that docs/metrics.md gives, the gateway's, vLLM's, Prometheus's own and the exporters'."""
    families: dict[str, Family] = {}
    for name, kind, labels in ROW.findall(CATALOG.read_text(encoding="utf-8")):
        assert name not in families, f"{name} is in docs/metrics.md twice"
        families[name] = Family(kind, frozenset(re.findall(r"`([^`]+)`", labels)))
    return families


def served(text: str) -> dict[str, Family]:
    """The families of an exposition in the text format 0.0.4, with the labels of their samples; a histogram's `le`
    is its buckets', not a label of its own."""
    kinds: dict[str, str] = {}
    labels: dict[str, set[str]] = collections.defaultdict(set)
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split(" ")
            kinds[name] = kind
        elif line and not line.startswith("#"):
            name, _, rest = line.rpartition(" ")[0].partition("{")
            for suffix in ("_bucket", "_sum", "_count"):
                if name.endswith(suffix) and kinds.get(name.removesuffix(suffix)) == "histogram":
                    name = name.removesuffix(suffix)
            labels[name].update(re.findall(r'(\w+)="', rest))
    return {name: Family(kind, frozenset(labels[name] - {"le"})) for name, kind in kinds.items()}


COMMON = frozenset({"requests_total", "http_responses_total", "start_time_seconds", "status", "sleep_requested",
                    "drain_generation", "accepted_in_flight", "ours_in_flight", "idle_seconds",
                    "idle_timeout_seconds", "wall_time_limit_seconds"})


def gateway_catalog(lane: str) -> dict[str, Family]:
    """Only the families that this lane promises, without requiring text-only placeholders on pictures."""
    assert lane in {"text", "pictures"}
    return {name: family for name, family in catalog().items() if name.startswith(GATEWAY)
            and (name.removeprefix(GATEWAY) in COMMON
                 or name.startswith(GATEWAY + "picture_") == (lane == "pictures"))}


@pytest.mark.anyio
async def test_the_catalog_gives_each_family_the_gateway_serves_with_its_type_and_labels_and_no_other() -> None:
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        answer = await call(client, "GET", stack.control + "/metrics", key=CONTROL)
    assert answer.status == 200
    documented = gateway_catalog("text")
    assert served(answer.body.decode()) == documented


def family_of(name: str, families: dict[str, Family]) -> str | None:
    """The catalog's family of a name used in a query: itself, or a histogram's for its `_bucket`, `_sum` or
    `_count`."""
    if name in families:
        return name
    for suffix in ("_bucket", "_sum", "_count"):
        base = name.removesuffix(suffix)
        if base != name and base in families and families[base].kind == "histogram":
            return base
    return None


def metric_names(expr: str) -> set[str]:
    """The metric names of a PromQL expression: every identifier outside strings, label matchers, ranges and label
    lists that is neither a keyword nor called as a function or an aggregation."""
    text = LABEL_LIST.sub(" ", re.sub(r"\[[^\]]*\]", " ", re.sub(r"\{[^}]*\}", " ", STRING.sub('""', expr))))
    text = VARIABLE.sub(" ", text)
    return {match.group() for match in IDENTIFIER.finditer(text)
            if match.group().lower() not in KEYWORDS and not re.match(r"\s*\(", text[match.end():])}


def selected_labels(expr: str) -> Iterator[tuple[str, set[str]]]:
    """Each name of an expression with the labels its matchers select."""
    for name, matchers in SELECTOR.findall(STRING.sub('""', expr)):
        yield name, set(MATCHER.findall(matchers))


def balanced(expr: str) -> bool:
    opened: list[str] = []
    for char in STRING.sub('""', expr):
        if char in "([{":
            opened.append(char)
        elif char in ")]}" and (not opened or opened.pop() != "([{"[")]}".index(char)]):
            return False
    return not opened


def dashboard_queries(node: Any) -> Iterator[str]:
    """The PromQL of a dashboard: each target's `expr`, and each query variable's `label_values(...)` as the
    selector it reads."""
    if isinstance(node, dict):
        if isinstance(node.get("expr"), str):
            yield node["expr"]
        if node.get("type") == "query":
            query = node.get("query")
            text = query.get("query", "") if isinstance(query, dict) else query or ""
            match = re.fullmatch(r"\s*label_values\((.*),\s*(\w+)\s*\)\s*", text)
            assert match, f"a query variable that is not label_values(selector, label): {text}"
            yield match.group(1)
        for value in node.values():
            yield from dashboard_queries(value)
    elif isinstance(node, list):
        for value in node:
            yield from dashboard_queries(value)


def rule_expressions(text: str) -> Iterator[str]:
    """The `expr` of each rule in a rules file: on its line, or in the block below it."""
    lines = text.splitlines()
    for number, line in enumerate(lines):
        match = re.match(r"^(\s*)expr:\s*(.*)$", line)
        if not match:
            continue
        indent, value = len(match.group(1)), match.group(2).strip()
        if value in ("|", "|-", ">", ">-"):
            block = []
            for following in lines[number + 1:]:
                if following.strip() and len(following) - len(following.lstrip()) <= indent:
                    break
                block.append(following.strip())
            value = " ".join(block)
        yield value


def expressions() -> dict[str, list[str]]:
    """Every PromQL expression of the monitoring, by its file."""
    found = {path.name: list(dashboard_queries(json.loads(path.read_text(encoding="utf-8"))))
             for path in sorted((MONITORING / "grafana" / "dashboards").glob("*.json"))}
    found["rules.yml"] = list(rule_expressions((MONITORING / "prometheus" / "rules.yml").read_text(encoding="utf-8")))
    return found


def test_the_monitoring_uses_only_the_catalogs_names_and_selects_only_their_labels() -> None:
    families = catalog()
    used: set[str] = set()
    unknown: set[tuple[str, str]] = set()
    wrong: set[tuple[str, str, str]] = set()
    for source, found in expressions().items():
        for expr in found:
            assert balanced(expr), (source, expr)
            names = metric_names(expr)
            assert names, (source, expr)
            used |= names
            unknown |= {(source, name) for name in names if family_of(name, families) is None}
            for name, labels in selected_labels(expr):
                family = family_of(name, families)
                if family is not None:
                    allowed = families[family].labels | TARGET_LABELS | ({"le"} if name.endswith("_bucket") else set())
                    wrong |= {(source, name, label) for label in labels - allowed}
    for path in sorted(MONITORING.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".yml", ".yaml"):
            unknown |= {(path.name, name) for name in PREFIXED.findall(path.read_text(encoding="utf-8"))
                        if family_of(name, families) is None}
    assert not unknown
    assert not wrong
    assert {"simple_serving_requests_total", "simple_serving_generations_waiting"} <= used


@pytest.mark.skipif(not os.environ.get(VLLM_SOURCE), reason=f"{VLLM_SOURCE} names no checkout of vLLM")
def test_the_catalogs_vllm_names_are_in_vllms_source_at_the_pinned_version_with_their_types() -> None:
    source = Path(os.environ[VLLM_SOURCE])
    pinned = re.search(r"^VLLM_VERSION=(\S+)$", (ROOT / "card" / "manifest.env").read_text(encoding="utf-8"),
                       re.MULTILINE)
    assert pinned
    tag = subprocess.run(["git", "-C", str(source), "describe", "--tags", "--exact-match"], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert tag == f"v{pinned.group(1)}"
    defined: dict[str, str] = {}
    for path in ("vllm/v1/metrics/loggers.py", "vllm/v1/spec_decode/metrics.py"):
        text = (source / path).read_text(encoding="utf-8")
        definitions = re.findall(r'_(gauge|counter|histogram)_cls\(\s*name="(vllm:\w+)"', text)
        defined |= {name: kind for kind, name in definitions}
        defined |= dict.fromkeys(re.findall(r'\(\s*"(vllm:spec_decode_\w+)",', text), "counter")  # counter_specs
    vllm = {name: family for name, family in catalog().items() if name.startswith("vllm:")}
    assert vllm
    for name, family in vllm.items():
        assert defined.get(name.removesuffix("_total") if family.kind == "counter" else name) == family.kind, name


def scrape_jobs() -> dict[str, str]:
    """The explicit job blocks in our small YAML configuration, ignoring commented optional exporters."""
    text = (MONITORING / "prometheus" / "prometheus.yml").read_text()
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return dict(re.findall(r"^  - job_name: (\w+)\n(.*?)(?=^  - job_name:|\Z)", text,
                           re.MULTILINE | re.DOTALL))


def test_scrapes_use_control_tunnels_and_lane_credentials_without_a_picture_engine_relay() -> None:
    jobs = scrape_jobs()
    expected = {"gateway_text": ("text", "/metrics", "metrics_key", "127.0.0.1:8081"),
                "gateway_pictures": ("pictures", "/metrics", "picture_metrics_key", None),
                "engine_text": ("text", "/metrics/engine", "metrics_key", "127.0.0.1:8081")}
    assert jobs.keys() == expected.keys(), "Unexpected scraper or missing gateway/engine job"
    compose = (MONITORING / "compose.yaml").read_text()
    for job, (lane, path, secret, target) in expected.items():
        block = jobs[job]
        assert f"metrics_path: {path}\n" in block, job
        assert f"lane: {lane}\n" in block, job
        assert f"credentials_file: /run/secrets/{secret}\n" in block, job
        assert f"      - {secret}\n" in compose and f"  {secret}:\n" in compose, job
        targets = re.search(r"targets: (\[[^\n]*?\])", block)
        assert targets
        assert json.loads(targets.group(1)) == ([] if target is None else [target]), job
    assert 'PICTURE_METRICS_KEY_FILE:-${METRICS_KEY_FILE}' in compose
    assert '--web.listen-address=127.0.0.1:9090' in compose
    assert 'GF_SERVER_HTTP_ADDR: "127.0.0.1"' in compose
    assert re.findall(r"image: (\S+)", compose) and all(
        re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", image)
        for image in re.findall(r"image: (\S+)", compose))


def test_dashboard_selectors_keep_vllm_on_text_and_shared_state_on_one_lane() -> None:
    seen_picture = False
    for source in ("overview.json", "vllm.json"):
        dashboard = json.loads((MONITORING / "grafana" / "dashboards" / source).read_text())
        for panel in [*dashboard["panels"], dashboard.get("templating", {})]:
            text_places = source == "overview.json" and panel.get("id") == 20
            for expr in dashboard_queries(panel):
                for name, matchers in SELECTOR.findall(expr):
                    if name.startswith("vllm:"):
                        assert 'job="engine_text"' in matchers, (source, expr)
                    if name.startswith(GATEWAY + "picture_"):
                        seen_picture = True
                        assert 'job="gateway_pictures"' in matchers, expr
                    if text_places and name.startswith(GATEWAY):
                        # This panel compares text work with text shared capacity, regardless of the selected lane.
                        assert 'job="gateway_text"' in matchers and "$lane" not in matchers, expr
                    elif name.removeprefix(GATEWAY) in COMMON and name.startswith(GATEWAY):
                        assert 'lane="$lane"' in matchers and 'job=~"gateway_text|gateway_pictures"' in matchers, expr
                # A bare metric would bypass the selector checks and could silently aggregate both lanes.
                for name in metric_names(expr):
                    if name.startswith((GATEWAY, "vllm:")):
                        assert re.search(re.escape(name) + r"\s*\{", expr), (source, name, expr)
    assert seen_picture, "The overview must expose picture jobs, not only text metrics"
    rules = (MONITORING / "prometheus" / "rules.yml").read_text()
    missing = rules.split("alert: EngineMetricsMissing", 1)[1].split("for:", 1)[0]
    assert 'job="gateway_text"' in missing and 'job="engine_text"' in missing and 'on(lane)' in missing
    assert 'job="gateway_pictures"' not in missing, "Picture readiness must not alert about missing vLLM"

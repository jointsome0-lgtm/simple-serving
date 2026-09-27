"""The metric catalog, docs/metrics.md, against the gateway that serves it.

The catalog is what a gateway in another language would serve, so it must hold the gateway's families as they are:
each name, its type and its labels, and nothing the gateway does not serve.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass

import httpx
import pytest

from .support import CONTROL, ROOT, call, running

CATALOG = ROOT / "docs" / "metrics.md"
# A row of the catalog's tables: the name, the type, then the labels, each in backticks.
ROW = re.compile(r"^\| `([^`]+)` \| (counter|gauge|histogram) \|([^|]*)\|", re.MULTILINE)
GATEWAY = "simple_serving_"


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


@pytest.mark.anyio
async def test_the_catalog_gives_each_family_the_gateway_serves_with_its_type_and_labels_and_no_other() -> None:
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        answer = await call(client, "GET", stack.control + "/metrics", key=CONTROL)
    assert answer.status == 200
    documented = {name: family for name, family in catalog().items() if name.startswith(GATEWAY)}
    assert served(answer.body.decode()) == documented

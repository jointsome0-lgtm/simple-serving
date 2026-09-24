import io
import logging
from collections.abc import Iterator

import pytest

from simple_serving import log


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def logged() -> Iterator[io.StringIO]:
    """Everything logged during the test, formatted as the gateway writes its log."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(log.JsonFormatter())
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        root.removeHandler(handler)
        root.setLevel(level)

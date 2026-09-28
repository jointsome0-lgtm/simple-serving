"""The picture card's launcher (docs/pictures.md, section 9): the text lane's (`simple_serving/card.py`) with ComfyUI
for its engine.

    python -m simple_serving.pictures.card [--hold | --stop | --retry | --dry-run | --inspect]

card/pictures/onstart.sh runs it from the checkout with the lane's gateway venv, and card/pictures/bootstrap.sh prepares
the card once per rental. The one pair with no restarts, the load deadline, the stop of the instance, the markers, the
options and the exit codes are the text launcher's, in a state of its own, /workspace/simple-serving-pictures-card,
with logs of its own.

The engine is the copied image-serve.sh (card/pictures/SOURCE) with the manifest's settings, Viggle's node off whatever
the environment says, and without the card's own key. It runs under `flock -n` on /root/.simple-chat-comfy.lock, the
lock that the old start takes, held for as long as ComfyUI runs, so the two cannot run at once. Of its output, ComfyUI's
lines leave fixed categories of failure and warning, each once, and nothing else; the sweeper's rows pass as rows of
their own, with only their counts as integers and their codes from the sweeper's closed set.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, ClassVar

from .. import card, log
from ..card import Card, Command, Pair, read_manifest
from . import LANE
from .config import PROVISIONAL

STATE = Path(os.environ.get("SIMPLE_SERVING_PICTURES_CARD_DIR", "/workspace/simple-serving-pictures-card"))
# simple-story-chat's scripts and the files they read beside themselves, as SOURCE names them.
COPIES = ("image-bootstrap.sh", "image-serve.sh", "image-sweeper.py", "image-manifest.env", "image-workflow.json",
          "image-workflow-qwen.json", "image-workflow-qwen-edit.json")
LOCK = ".simple-chat-comfy.lock"  # in the card's /root: the old start's (simple-story-chat's docs/gpu.md#bot-card)
# image-serve.sh's settings, each from the manifest's field.
SETTINGS = {"SIMPLE_CHAT_GPU_DIR": "GPU_DIR", "SIMPLE_CHAT_IMAGE_QWEN": "IMAGE_QWEN",
            "SIMPLE_CHAT_IMAGE_TORCH": "IMAGE_TORCH", "SIMPLE_CHAT_IMAGE_TRITON": "IMAGE_TRITON",
            "SIMPLE_CHAT_IMAGE_GPU": "IMAGE_GPU", "SIMPLE_CHAT_IMAGE_PORT": "ENGINE_PORT"}
# The categories of a lowercased line of ComfyUI's, or of image-serve.sh's own, each with its words. A line gets the
# first that fits.
FAILURES = (
    ("out_of_memory", ("out of memory", "outofmemoryerror")),
    ("cuda_error", ("cuda error", "cudaerror", "cublas_status")),
    ("bind_failed", ("address already in use",)),
    ("exception", ("!!! exception during processing !!!",)),  # a node that raised (execution.py)
    ("traceback", ("traceback (most recent call last)",)),
    # image-serve.sh's refusals to start: an install that is not whole, or a temp directory that is not in RAM.
    ("install_incomplete", ("image-bootstrap.sh",)),
    ("temp_not_tmpfs", ("is not a writable tmpfs", "on a tmpfs. refusing")),
)
WARNINGS = (
    ("partial_load", ("loaded partially", "unloaded partially")),  # comfy/model_management.py
    ("attention_fallback", ("is unavailable; using pytorch attention",)),  # comfy_extras/nodes_model_advanced.py
)
# The sweeper's rows (image-sweeper.py `emit`), each with the fields it may keep and their names here.
SWEEPER = {
    "sweeper_started": {"graceSeconds": "grace_s", "fileCapSeconds": "file_cap_s",
                        "historyCapSeconds": "history_cap_s"},
    "swept": {"files": "files", "failedFiles": "failed_files", "records": "records"},
    "records_undeleted": {"code": "code", "count": "count"},
    "history_unreadable": {"code": "code"},
    "history_readable": {},
    "sweep_failed": {"code": "code"},
    "sweeper_stopped": {"reason": "reason"},
}
SWEEPER_CODE = re.compile(  # image-sweeper.py `failure_code`, and its one reason to stop
    r"http_(?:[1-5][0-9][0-9]|other)|connection_refused|timeout|invalid_json|invalid_shape|os_error|other|server_gone")


class PictureCard(Card):
    module: ClassVar[str] = "simple_serving.pictures.card"
    running: ClassVar[bytes] = b"simple_serving.pictures.card\0--run"
    home: ClassVar[Path] = STATE
    folder: ClassVar[str] = "card/pictures"
    stamped: ClassVar[tuple[str, ...]] = ("manifest.env", "gateway-requirements.txt", "SOURCE", *COPIES)

    def engine(self) -> Command:
        environ = {name: value for name, value in os.environ.items()
                   if name != "CONTAINER_API_KEY" and not name.startswith("SIMPLE_CHAT_")}
        environ |= {name: self.manifest[setting] for name, setting in SETTINGS.items()}
        return Command(["flock", "-n", str(self.root / LOCK), "bash", str(self.code / self.folder / "image-serve.sh")],
                       environ | {"SIMPLE_CHAT_IMAGE_VIGGLE": "false"})

    def gateway(self) -> Command:
        return Command([str(self.state / "gateway/bin/python"), "-m", "simple_serving.pictures"],
                       {**os.environ, **self.credential, "SIMPLE_SERVING_CONFIG": str(self.state / "gateway.json")},
                       self.code)

    def config(self) -> dict[str, Any]:
        """The picture gateway's configuration: the provisional limits, the manifest's model, ports and allowlist, the
        engine facts of the copied image-manifest.env, and the hashes of the two keys."""
        m, pins = self.manifest, read_manifest(self.code, self.folder, "image-manifest.env")
        source = (self.code / self.folder / "SOURCE").read_text()
        return PROVISIONAL | {
            "lane": LANE,
            "alias": m["MODEL_ALIAS"],
            "engine_url": f"http://127.0.0.1:{m['ENGINE_PORT']}",
            "listen": self.listen(),
            "keys": self.keys(),
            "reader_nodes": m["READER_NODES"].split(","),
            "expect": {"comfyui": pins["COMFYUI_VERSION"], "pytorch": pins["TORCH_CU130_VERSION"],
                       "triton": m["IMAGE_TRITON"] == "1"},
            "versions": {"comfyui_revision": pins["COMFYUI_REVISION"], "qwen_revision": pins["IMAGE_QWEN_REVISION"],
                         "scripts_commit": re.findall(r"^commit (\S+)$", source, re.MULTILINE)[0]},
        }

    def pair(self) -> Pair:
        return PicturePair(self.state)


class PicturePair(Pair):
    """ComfyUI and the picture gateway."""

    def engine_line(self, line: bytes) -> None:
        swept = sweeper_row(line)
        if swept is not None:
            log.row(swept[0], **swept[1])
            return
        lowered = line.decode(errors="replace").lower()
        for event, categories in (("engine_failure", FAILURES), ("engine_warning", WARNINGS)):
            category = next((name for name, words in categories if any(word in lowered for word in words)), None)
            if category is not None and category not in self.noted:
                self.noted.add(category)
                log.row(event, code=category)


def sweeper_row(line: bytes) -> tuple[str, dict[str, Any]] | None:
    """A row of the sweeper's, as its event and the fields that pass: a count or seconds as a whole number, a code of
    its closed set. None for any other line."""
    try:
        row = json.loads(line)
    except (ValueError, RecursionError):
        return None
    if not isinstance(row, dict) or not isinstance(row.get("event"), str) or row["event"] not in SWEEPER:
        return None
    fields: dict[str, Any] = {}
    for name, field in SWEEPER[row["event"]].items():
        value = row.get(name)
        if field in ("code", "reason"):
            if isinstance(value, str) and SWEEPER_CODE.fullmatch(value):
                fields[field] = value
        elif isinstance(value, int | float) and not isinstance(value, bool) and 0 <= value < 10**9:  # nor NaN
            fields[field] = round(value)
    return row["event"], fields


def main(argv: list[str] | None = None) -> int:
    return card.main(argv, PictureCard)


if __name__ == "__main__":
    sys.exit(main())

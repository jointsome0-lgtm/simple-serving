"""The picture lane's two listeners (docs/pictures.md, section 3), built as the text lane's are
(`simple_serving/app.py`): the public listener serves pictures, their stops, uploads, the engine's facts and state;
the control listener, on loopback, serves the contract's state, drain, sleep and open."""

from __future__ import annotations

from typing import Any

from .. import routes as shared
from ..app import CONTROL_ROUTES, Guard, _application
from . import routes
from .service import PictureService

PUBLIC_ROUTES: list[tuple[str, str, Any]] = [
    ("/v1/pictures", "POST", routes.picture),
    ("/v1/pictures/engine", "GET", routes.engine),
    ("/v1/pictures/uploads/{name}", "PUT", routes.put_upload),
    ("/v1/pictures/uploads/{name}", "DELETE", routes.erase_upload),
    ("/v1/pictures/inputs/{name}", "PUT", routes.put_input),
    ("/v1/pictures/{job}/stop", "POST", routes.stop),
    ("/v1/state", "GET", shared.state),
]


def public_app(service: PictureService) -> Guard:
    return Guard(_application(service, PUBLIC_ROUTES), "public", PUBLIC_ROUTES,
                 max_connections=service.config.max_connections)


def control_app(service: PictureService) -> Guard:
    return Guard(_application(service, CONTROL_ROUTES), "control", CONTROL_ROUTES)

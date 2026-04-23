from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseController
from .ik_id_knee import IkIdKneeOnnxController

REGISTRY: Dict[str, Type[BaseController]] = {
    IkIdKneeOnnxController.name: IkIdKneeOnnxController,
}


def build_controller(name: str, **kwargs: Any) -> BaseController:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown controller: {name}. Available: {sorted(REGISTRY)}")
    return cls(**kwargs)

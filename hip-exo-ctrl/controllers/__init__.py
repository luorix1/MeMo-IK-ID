from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseController
from .ik_id_hip import IkIdHipOnnxController

REGISTRY: Dict[str, Type[BaseController]] = {
    IkIdHipOnnxController.name: IkIdHipOnnxController,
}


def build_controller(name: str, **kwargs: Any) -> BaseController:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown controller: {name}. Available: {sorted(REGISTRY)}")
    return cls(**kwargs)

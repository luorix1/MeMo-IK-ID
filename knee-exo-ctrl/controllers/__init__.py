from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseController
from .ik_id_knee import IkIdKneeOnnxController
from .ik_id_knee_trt import IkIdKneeTrtController

REGISTRY: Dict[str, Type[BaseController]] = {
    IkIdKneeOnnxController.name: IkIdKneeOnnxController,
    IkIdKneeTrtController.name: IkIdKneeTrtController,
}


def build_controller(name: str, **kwargs: Any) -> BaseController:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown controller: {name}. Available: {sorted(REGISTRY)}")
    return cls(**kwargs)

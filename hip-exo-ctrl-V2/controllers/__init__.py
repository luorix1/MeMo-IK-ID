from .cascade import CascadeHip
from .state2torque import State2Torque

REGISTRY = {
    State2Torque.name: State2Torque,
    CascadeHip.name: CascadeHip,
}


def build_controller(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown controller: '{name}'. Available: {list(REGISTRY.keys())}")
    if "config" in kwargs:
        return cls(kwargs["config"])
    return cls(**kwargs)

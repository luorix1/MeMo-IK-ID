from .biotorque import biotorque
from .cascade import CascadeHip
from .dofc import DOFC

REGISTRY = {
    DOFC.name: DOFC,
    biotorque.name: biotorque,
    CascadeHip.name: CascadeHip,
}


def build_controller(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown controller: '{name}'. Available: {list(REGISTRY.keys())}")
    return cls(**kwargs)

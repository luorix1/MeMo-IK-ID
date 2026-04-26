from .biotorque_hip import HipBiotorque

REGISTRY = {
    HipBiotorque.name: HipBiotorque,
}


def build_controller(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown controller: '{name}'. Available: {list(REGISTRY.keys())}"
        )
    return cls(**kwargs)

from .state2torque_k5 import State2TorqueK5

REGISTRY = {
    State2TorqueK5.name: State2TorqueK5,
}


def build_controller(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown controller: '{name}'. Available: {list(REGISTRY.keys())}")
    if "config" in kwargs:
        return cls(kwargs["config"])
    return cls(**kwargs)

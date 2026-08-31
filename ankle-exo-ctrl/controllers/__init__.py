from .test import Test
from .impedance_enc import ImpedanceEnc
from .cascade import CascadeUni

REGISTRY = {
    Test.name: Test,
    ImpedanceEnc.name: ImpedanceEnc,
    CascadeUni.name: CascadeUni,
}


def build_controller(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown controller: '{name}'. Available: {list(REGISTRY.keys())}")
    if "config" in kwargs:
        return cls(kwargs["config"])
    return cls(**kwargs)

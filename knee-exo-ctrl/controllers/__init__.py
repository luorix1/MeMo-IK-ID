from .dofc import DOFC
from .test import TEST
from .impedance_rl import impedance_rl
from .impedance_rl_uni import impedance_rl_uni
from .biotorque import biotorque


REGISTRY = {
    DOFC.name: DOFC,
    TEST.name: TEST,
    impedance_rl.name: impedance_rl,
    impedance_rl_uni.name: impedance_rl_uni,
    biotorque.name: biotorque,
}

def build_controller(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown controller: {name}")
    return cls(**kwargs)

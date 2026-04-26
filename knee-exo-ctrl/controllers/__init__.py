from .dofc import DOFC
from .test import Test
from .impedance_rl import ImpedanceRL
from .impedance_rl_uni import ImpedanceRLUni
from .biotorque import Biotorque


REGISTRY = {
    DOFC.name: DOFC,
    Test.name: Test,
    ImpedanceRL.name: ImpedanceRL,
    ImpedanceRLUni.name: ImpedanceRLUni,
    Biotorque.name: Biotorque,
}

def build_controller(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown controller: '{name}'. Available: {list(REGISTRY.keys())}")
    return cls(**kwargs)

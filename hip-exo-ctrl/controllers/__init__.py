from .biotorque import biotorque
from .dofc import DOFC
from .gyro1ch_trt import Gyro1ChTRT
from .simgyro3 import simgyro3

REGISTRY = {
    Gyro1ChTRT.name: Gyro1ChTRT,
    DOFC.name: DOFC,
    biotorque.name: biotorque,
    simgyro3.name: simgyro3,
}


def build_controller(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown controller: '{name}'. Available: {list(REGISTRY.keys())}")
    return cls(**kwargs)

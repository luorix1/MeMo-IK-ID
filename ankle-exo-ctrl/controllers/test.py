"""Zero-torque bring-up controller. Logs encoder pos / dθ/dt for Teleplot."""

from .base import BaseController, Sensors, CtrlResult


class Test(BaseController):
    name = "TEST"

    def __init__(self, config: dict):
        self.cfg = config

    def step(self, s: Sensors) -> CtrlResult:
        return CtrlResult(
            model_out_R=0.0,
            model_out_L=0.0,
            applied_R=0.0,
            applied_L=0.0,
            extra={
                "ankle_pos_L": s.pos_L,
                "ankle_pos_R": s.pos_R,
                "ankle_vel_L": s.vel_L,
                "ankle_vel_R": s.vel_R,
            },
        )

"""Simple bilateral ankle impedance using encoder angle and d(encoder)/dt.

    τ = K * (θ_ref - θ) - B * ω

where θ is the sign-corrected encoder angle (rad) and ω is its time derivative
(rad/s). No IMU / Muse required.
"""

from .base import BaseController, Sensors, CtrlResult


class ImpedanceEnc(BaseController):
    name = "impedance_enc"

    def __init__(self, config: dict):
        self.K = float(config.get("K", 5.0))
        self.B = float(config.get("B", 0.5))
        self.theta_ref_L = float(config.get("theta_ref_L", config.get("theta_ref", 0.0)))
        self.theta_ref_R = float(config.get("theta_ref_R", config.get("theta_ref", 0.0)))
        self.scale = float(config.get("scale", 1.0))

    def step(self, s: Sensors) -> CtrlResult:
        tau_L = self.K * (self.theta_ref_L - s.pos_L) - self.B * s.vel_L
        tau_R = self.K * (self.theta_ref_R - s.pos_R) - self.B * s.vel_R
        tau_L *= self.scale
        tau_R *= self.scale

        return CtrlResult(
            model_out_R=tau_R,
            model_out_L=tau_L,
            applied_R=tau_R,
            applied_L=tau_L,
            extra={
                "ankle_pos_L": s.pos_L,
                "ankle_pos_R": s.pos_R,
                "ankle_vel_L": s.vel_L,
                "ankle_vel_R": s.vel_R,
                "tau_raw_L": tau_L,
                "tau_raw_R": tau_R,
            },
        )

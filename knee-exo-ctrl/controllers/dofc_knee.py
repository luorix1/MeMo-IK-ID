import numpy as np
from collections import deque

from .base import BaseController, Sensors, CtrlResult


class impedance_rl(BaseController):
    name = "dofc_knee"

    def __init__(self, config: dict):

        self.fs = int(config["fs"])
        self.dt = 1.0 / self.fs

        self.scale_stance = float(3.0)
        self.scale_swing  = float(0.8)

        self.torque_limit = float(5.0)
        self.cmd_rate_max = float(20)

        self.knee_filter_tau = float(0.1)
        self.blend_tau = float(0.05)
        self.delay_steps = int(4)

        self.swing_enter_vel  = float(config.get("swing_enter_vel", -0.35))
        self.stance_enter_vel = float(config.get("stance_enter_vel", +0.25))

        self.swing_enter_angle  = float(config.get("swing_enter_angle", -0.20))
        self.stance_enter_angle = float(config.get("stance_enter_angle", -0.05))

        self.swing_force_angle  = float(config.get("swing_force_angle", -0.45))
        self.stance_force_angle = float(config.get("stance_force_angle", +0.05))

        self.vel_deadband = float(0.04)
        self.min_state_time = float(0.2)

        self.thigh_gyR = 0.0
        self.shank_gyR = 0.0
        self.thigh_gyL = 0.0
        self.shank_gyL = 0.0

        self.knee_r_filt = 0.0
        self.knee_l_filt = 0.0
        self.knee_r_u_filt = 0.0
        self.knee_l_u_filt = 0.0

        self.knee_r_u_gyr_filt = 0.0
        self.knee_l_u_gyr_filt = 0.0

        self.knee_r_u_gyr_buf = deque([0.0] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)
        self.knee_l_u_gyr_buf = deque([0.0] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)

        self.state_r = "stance"
        self.state_l = "stance"

        self.time_in_state_r = 0.0
        self.time_in_state_l = 0.0

        self.blend_r = 1.0
        self.blend_l = 1.0

        self.prev_cmd_r = 0.0
        self.prev_cmd_l = 0.0


    def _alpha(self, tau: float) -> float:
        if tau <= 0.0:
            return 1.0
        a = self.dt / tau
        return min(max(a, 0.0), 1.0)

    def _lpf(self, x_prev: float, x_raw: float, tau: float) -> float:
        a = self._alpha(tau)
        return float(x_prev + a * (x_raw - x_prev))

    def _rate_limit(self, current: float, prev: float, rate_max: float) -> float:
        max_step = rate_max * self.dt
        return float(prev + np.clip(current - prev, -max_step, +max_step))

    def _delay_push_and_get(self, buf: deque, x: float) -> float:
        buf.append(float(x))
        return float(buf[0])

    def _apply_deadband(self, x: float, db: float) -> float:
        if abs(x) < db:
            return 0.0
        return float(x)

    def _update_blend(self, blend_prev: float, target: float) -> float:
        a = self._alpha(self.blend_tau)
        return float(blend_prev + a * (target - blend_prev))

    def _effective_scale(self, blend: float) -> float:
        return float(blend * self.scale_stance + (1.0 - blend) * self.scale_swing)

    def _next_state(self, state: str, knee_angle: float, knee_u_delayed: float, time_in_state: float) -> str:
        if time_in_state < self.min_state_time:
            return state

        if state == "stance":
            cond_main = (knee_u_delayed < self.swing_enter_vel) and (knee_angle < self.swing_enter_angle)
            cond_force = (knee_angle < self.swing_force_angle)
            if cond_main or cond_force:
                return "swing"
            return "stance"

        elif state == "swing":
            cond_main = (knee_u_delayed > self.stance_enter_vel) and (knee_angle > self.stance_enter_angle)
            cond_force = (knee_angle > self.stance_force_angle)

            if cond_main or cond_force:
                return "stance"
            return "swing"

        return "stance"


    def step(self, s: Sensors) -> CtrlResult:
        thigh_gyR_raw = float(s.imu_R1[5])
        shank_gyR_raw = float(s.imu_R2[5])

        thigh_gyL_raw = -float(s.imu_L1[5])
        shank_gyL_raw = -float(s.imu_L2[5])

        knee_r_raw   = np.deg2rad(float(s.pos_R))
        knee_r_u_raw = np.deg2rad(float(s.vel_R))

        knee_l_raw   = -np.deg2rad(float(s.pos_L))
        knee_l_u_raw = -np.deg2rad(float(s.vel_L))

        self.thigh_gyR = self._lpf(self.thigh_gyR, thigh_gyR_raw, self.knee_filter_tau)
        self.shank_gyR = self._lpf(self.shank_gyR, shank_gyR_raw, self.knee_filter_tau)
        self.thigh_gyL = self._lpf(self.thigh_gyL, thigh_gyL_raw, self.knee_filter_tau)
        self.shank_gyL = self._lpf(self.shank_gyL, shank_gyL_raw, self.knee_filter_tau)

        self.knee_r_filt   = self._lpf(self.knee_r_filt, knee_r_raw, self.knee_filter_tau)
        self.knee_r_u_filt = self._lpf(self.knee_r_u_filt, knee_r_u_raw, self.knee_filter_tau)
        self.knee_l_filt   = self._lpf(self.knee_l_filt, knee_l_raw, self.knee_filter_tau)
        self.knee_l_u_filt = self._lpf(self.knee_l_u_filt, knee_l_u_raw, self.knee_filter_tau)

        self.knee_r_u_gyr_filt = self._lpf(
            self.knee_r_u_gyr_filt,
            (-self.thigh_gyR + self.shank_gyR),
            self.knee_filter_tau,
        )
        self.knee_l_u_gyr_filt = self._lpf(
            self.knee_l_u_gyr_filt,
            (-self.thigh_gyL + self.shank_gyL),
            self.knee_filter_tau,
        )

        knee_r_u_gyr_use = self._apply_deadband(self.knee_r_u_gyr_filt, self.vel_deadband)
        knee_l_u_gyr_use = self._apply_deadband(self.knee_l_u_gyr_filt, self.vel_deadband)


        knee_r_u_gyr_delayed = self._delay_push_and_get(self.knee_r_u_gyr_buf, knee_r_u_gyr_use)
        knee_l_u_gyr_delayed = self._delay_push_and_get(self.knee_l_u_gyr_buf, knee_l_u_gyr_use)

        self.time_in_state_r += self.dt
        self.time_in_state_l += self.dt

        next_state_r = self._next_state(
            self.state_r,
            self.knee_r_filt,
            knee_r_u_gyr_delayed,
            self.time_in_state_r,
        )
        next_state_l = self._next_state(
            self.state_l,
            self.knee_l_filt,
            knee_l_u_gyr_delayed,
            self.time_in_state_l,
        )

        if next_state_r != self.state_r:
            self.state_r = next_state_r
            self.time_in_state_r = 0.0

        if next_state_l != self.state_l:
            self.state_l = next_state_l
            self.time_in_state_l = 0.0


        target_blend_r = 1.0 if self.state_r == "stance" else 0.0
        target_blend_l = 1.0 if self.state_l == "stance" else 0.0

        self.blend_r = self._update_blend(self.blend_r, target_blend_r)
        self.blend_l = self._update_blend(self.blend_l, target_blend_l)

        scale_eff_r = self._effective_scale(self.blend_r)
        scale_eff_l = self._effective_scale(self.blend_l)

        tau_r = scale_eff_r*knee_r_u_gyr_delayed
        tau_l = scale_eff_l*knee_l_u_gyr_delayed

        tau_r = self._rate_limit(tau_r, self.prev_cmd_r, self.cmd_rate_max)
        tau_l = self._rate_limit(tau_l, self.prev_cmd_l, self.cmd_rate_max)

        self.prev_cmd_r = tau_r
        self.prev_cmd_l = tau_l

        return CtrlResult(
            model_out_R=float(tau_r),
            model_out_L=float(tau_l),
            applied_R=float(tau_r),
            applied_L=float(tau_l),
            extra={
                "state_r": self.state_r,
                "state_l": self.state_l,
                "blend_r": float(self.blend_r),
                "blend_l": float(self.blend_l),
                "scale_eff_r": float(scale_eff_r),
                "scale_eff_l": float(scale_eff_l),
                "knee_angle_r": float(self.knee_r_filt),
                "knee_angle_l": float(self.knee_l_filt),
                "knee_u_gyr_r": float(knee_r_u_gyr_delayed),
                "knee_u_gyr_l": float(knee_l_u_gyr_delayed),
                "time_in_state_r": float(self.time_in_state_r),
                "time_in_state_l": float(self.time_in_state_l),
            },
        )

    def start(self):
        pass

    def close(self):
        pass
"""
CascadeUni — Unilateral knee controller driven by the os_kinetics TCN moment estimator.

Model I/O (matches TRTWorkerUni):
  input  : (1, 2, T)  — [knee_angle_norm, knee_vel_imu_norm]
  output : (1, 1)     → scalar Nm

Channel convention (sign-corrected so both sides look like right):
  ch0 — knee encoder angle  (rad):      +ve = extension  / −ve = flexion
  ch1 — knee angular velocity (rad/s):  shank_gz − thigh_gz, sign-corrected per side

Normalization is taken from training checkpoint:
  ch0 : pos_mean[9], pos_std[9]   (knee_angle_r from the dataset; left side is sign-flipped
                                    by the unilateral pipeline to match this convention)
  ch1 : vel_mean[9], vel_std[9]
"""

import multiprocessing as mp
from queue import Empty, Full

import numpy as np

from .base import BaseController, CtrlResult, RollingWindow, Sensors
from .trt_worker_uni import TRTWorkerUni


class CascadeUni(BaseController):
    name = "cascade_uni"

    def __init__(self, config: dict):
        self.side = str(config["side"]).lower()
        if self.side not in ("right", "left"):
            raise ValueError(f"Invalid side: '{self.side}'. Must be 'right' or 'left'.")

        self.engine_path = config["trt_engine_path"]
        self.T   = int(config["frame_length"])
        self.fs  = int(config["fs"])
        self.dt  = 1.0 / self.fs

        # normalization stats (from training checkpoint)
        self.knee_angle_mean = float(config["knee_angle_mean"])
        self.knee_angle_std  = float(config["knee_angle_std"])
        self.knee_vel_mean   = float(config["knee_vel_mean"])
        self.knee_vel_std    = float(config["knee_vel_std"])

        # Model outputs Nm/kg (moments stored as N*m/kg in training dataset).
        # Multiply by subject mass to recover Nm, then by torque_scale for tuning.
        self.mass         = float(config["mass"])
        self.torque_scale = float(config.get("torque_scale", 1.0))
        self.torque_limit = float(config.get("torque_limit", 20.0))

        self.input_size  = int(config.get("input_size",  2))
        self.output_size = int(config.get("output_size", 1))
        if self.input_size != 2:
            raise ValueError(f"cascade_uni expects input_size=2, got {self.input_size}")
        if self.output_size != 1:
            raise ValueError(f"cascade_uni expects output_size=1, got {self.output_size}")

        self.in_shape  = (1, self.input_size, self.T)  # (1, 2, T)
        self.out_shape = (self.output_size,)            # (1,)

        # rolling input window
        self.x = RollingWindow((self.input_size, self.T))
        self.last_out = np.zeros(self.out_shape, dtype=np.float32)

        # TRT inference worker
        self.in_q  = mp.Queue(maxsize=1)
        self.out_q = mp.Queue(maxsize=1)
        self.worker = TRTWorkerUni(
            self.in_q, self.out_q,
            self.engine_path,
            self.in_shape,
            self.out_shape,
        )
        self.worker.daemon = True

        # filtered states
        self.knee_angle_filt   = 0.0   # encoder angle (rad)
        self.knee_vel_imu_filt = 0.0   # IMU-based knee angular velocity (rad/s)
        self.torque_filt       = 0.0   # output torque (Nm)

        # 1st-order LPF time constants (seconds)
        self.knee_filter_tau   = 0.05
        self.imu_filter_tau    = 0.15
        self.torque_filter_tau = 0.05

        # rate limiting
        self.cmd_rate_max = 200.0   # Nm/s
        self.prev_cmd     = 0.0

        # motion gating
        self.motion_score     = 0.0
        self.assist_gate      = 0.0
        self.motion_state     = "idle"
        self.start_timer      = 0.0
        self.motion_on_count  = 0
        self.motion_off_count = 0

        self.motion_score_tau  = 0.08
        self.motion_window_s   = 0.20
        self.start_thresh      = 0.4
        self.stop_thresh       = 0.25
        self.start_confirm_s   = 0.15
        self.stop_confirm_s    = 0.15
        self.start_delay_s     = 0.10
        self.ramp_up_s         = 0.40
        self.ramp_down_s       = 0.12

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    def start(self):
        self.worker.start()

    def close(self):
        try:
            self.in_q.put_nowait(None)
        except Exception:
            pass
        try:
            self.worker.join(timeout=1.5)
        except Exception:
            pass
        for q in (self.out_q, self.in_q):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _alpha(self, tau: float) -> float:
        return self.dt / tau if tau > 0.0 else 1.0

    def _lpf(self, x_prev: float, x_raw: float, tau: float) -> float:
        a = self._alpha(tau)
        return float(x_prev + a * (x_raw - x_prev))

    def _normalize(self, x: float, mean: float, std: float) -> float:
        s = float(std) if float(std) != 0.0 else 1.0
        return float((x - mean) / s)

    def _rate_limit(self, current: float, prev: float, rate_max: float) -> float:
        max_step = rate_max * self.dt
        return float(prev + np.clip(current - prev, -max_step, +max_step))

    def _get_latest_inference(self):
        latest = None
        try:
            while True:
                latest = self.out_q.get_nowait()
        except Empty:
            pass
        return latest

    def _try_put_latest(self, x: np.ndarray) -> None:
        try:
            self.in_q.put_nowait(x)
        except Full:
            pass

    def _update_motion_gate(self, seq: np.ndarray) -> None:
        """Gate on knee velocity channel (ch1) energy."""
        n = max(1, int(self.motion_window_s * self.fs))
        score_raw = float(np.mean(np.abs(seq[1:2, -n:])))
        self.motion_score = self._lpf(self.motion_score, score_raw, self.motion_score_tau)

        start_req = max(1, int(self.start_confirm_s * self.fs))
        stop_req  = max(1, int(self.stop_confirm_s  * self.fs))

        if self.motion_score > self.start_thresh:
            self.motion_on_count  += 1
            self.motion_off_count  = 0
        elif self.motion_score < self.stop_thresh:
            self.motion_off_count += 1
            self.motion_on_count   = 0
        else:
            self.motion_on_count  = 0
            self.motion_off_count = 0

        if self.motion_state == "idle":
            if self.motion_on_count >= start_req:
                self.motion_state = "starting"
                self.start_timer  = 0.0
        elif self.motion_state == "starting":
            if self.motion_off_count >= stop_req:
                self.motion_state = "idle"
                self.start_timer  = 0.0
            else:
                self.start_timer += self.dt
                if self.start_timer >= self.start_delay_s:
                    self.motion_state = "active"
        elif self.motion_state == "active":
            if self.motion_off_count >= stop_req:
                self.motion_state = "idle"
                self.start_timer  = 0.0

        gate_target = 1.0 if self.motion_state == "active" else 0.0
        gate_tau    = self.ramp_up_s if gate_target > self.assist_gate else self.ramp_down_s
        self.assist_gate = float(np.clip(
            self._lpf(self.assist_gate, gate_target, gate_tau), 0.0, 1.0
        ))

    # ------------------------------------------------------------------ #
    #  Main control step
    # ------------------------------------------------------------------ #
    def step(self, s: Sensors) -> CtrlResult:
        # ---- raw sensor extraction (sign-corrected: both sides → right convention) ----
        if self.side == "right":
            encoder_raw  = np.deg2rad(float(s.pos_R))
            thigh_gz_raw = float(s.imu_R1[5])
            shank_gz_raw = float(s.imu_R2[5])
        else:
            encoder_raw  = -np.deg2rad(float(s.pos_L))
            thigh_gz_raw = -float(s.imu_L1[5])
            shank_gz_raw = -float(s.imu_L2[5])

        # knee angular velocity from IMUs: shank_gz − thigh_gz
        knee_vel_imu_raw = shank_gz_raw - thigh_gz_raw

        # ---- LPF for filtered telemetry / state display ----
        self.knee_angle_filt   = self._lpf(self.knee_angle_filt,   encoder_raw,      self.knee_filter_tau)
        self.knee_vel_imu_filt = self._lpf(self.knee_vel_imu_filt, knee_vel_imu_raw, self.imu_filter_tau)

        # ---- build normalized model input ----
        x_last = np.array([
            self._normalize(encoder_raw,      self.knee_angle_mean, self.knee_angle_std),
            self._normalize(knee_vel_imu_raw, self.knee_vel_mean,   self.knee_vel_std),
        ], dtype=np.float32)

        seq = self.x.push_last(x_last)                                   # (2, T)
        x   = seq.reshape(self.in_shape).astype(np.float32, copy=False)  # (1, 2, T)

        # ---- motion gating ----
        self._update_motion_gate(seq)

        # ---- TRT inference (non-blocking, keep latest) ----
        latest = self._get_latest_inference()
        if latest is not None:
            self.last_out = np.asarray(latest, dtype=np.float32).reshape(self.out_shape)
        self._try_put_latest(x)

        # ---- post-process model output ----
        # model output is Nm/kg → multiply by mass → Nm, then apply torque_scale
        moment_raw = float(self.last_out[0]) * self.mass * self.torque_scale
        moment_gated = moment_raw * self.assist_gate

        # torque LPF → rate limit → hard clamp
        self.torque_filt = self._lpf(self.torque_filt, moment_gated, self.torque_filter_tau)
        tau = self._rate_limit(self.torque_filt, self.prev_cmd, self.cmd_rate_max)
        tau = float(np.clip(tau, -self.torque_limit, self.torque_limit))
        self.prev_cmd = tau

        # ---- pack into bilateral CtrlResult (zero the inactive side) ----
        state_int = {"idle": 0, "starting": 1, "active": 2}.get(self.motion_state, 0)
        tau_r = tau if self.side == "right" else 0.0
        tau_l = tau if self.side == "left"  else 0.0

        return CtrlResult(
            model_out_R=tau_r,
            model_out_L=tau_l,
            applied_R=tau_r,
            applied_L=tau_l,
            extra={
                "side":            self.side,
                "knee_angle":      float(self.knee_angle_filt),
                "knee_vel_imu":    float(self.knee_vel_imu_filt),
                "moment_raw":      float(moment_raw),
                "moment_gated":    float(moment_gated),
                "assist_gate":     float(self.assist_gate),
                "motion_score":    float(self.motion_score),
                "state":           state_int,
                # bilateral aliases for logging / teleplot
                "knee_angle_r":    float(self.knee_angle_filt) if self.side == "right" else 0.0,
                "knee_angle_l":    float(self.knee_angle_filt) if self.side == "left"  else 0.0,
                "knee_vel_imu_r":  float(self.knee_vel_imu_filt) if self.side == "right" else 0.0,
                "knee_vel_imu_l":  float(self.knee_vel_imu_filt) if self.side == "left"  else 0.0,
                "assist_gate_r":   float(self.assist_gate) if self.side == "right" else 0.0,
                "assist_gate_l":   float(self.assist_gate) if self.side == "left"  else 0.0,
                "state_r":         state_int if self.side == "right" else 0,
                "state_l":         state_int if self.side == "left"  else 0,
            },
        )

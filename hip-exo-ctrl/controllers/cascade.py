"""
CascadeHip — Bilateral hip controller driven by the os_kinetics TCN moment estimator.

Model I/O:
  input  : (2, 2, T)  — batch of 2 (right=0, left=1), each [hip_angle, hip_vel_imu]
  output : (2, 1)     — Nm/kg per side; left output is sign-flipped before applying

Hip angular velocity convention (applied before sign-correction per side):
  hip_vel = −thigh_Gyr_Y − pelvis_Gyr_Y
  (thigh Y-axis negated, pelvis Y-axis positive)

Both sides are sign-corrected to right convention before being fed to the model:
  right: encoder_raw  = −pos_R  (FIXME: negated — verify encoder sign on hardware)
         hip_vel_imu  = −imu_R_Gyr_Y − pelvis_Gyr_Y
  left:  encoder_raw  = −pos_L  (negated)
         hip_vel_imu  = −(−imu_L_Gyr_Y − pelvis_Gyr_Y)   (sign-flip to right convention)
                      =  imu_L_Gyr_Y + pelvis_Gyr_Y

Applied torques:
  right: −model_out[0] × mass × torque_scale  (FIXME: sign flip — verify on hardware)
  left:  +model_out[1] × mass × torque_scale  (FIXME: sign flip — verify on hardware)

IMU layout — 6-vector [Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z]:
  index 4 (Gyr_Y) is the sagittal-plane angular velocity for hip flex/extension.
  Gyro data arrives in deg/s from ICM20948 — converted to rad/s here.
"""

import multiprocessing as mp
from queue import Empty, Full

import numpy as np

from .base import BaseController, CtrlResult, RollingWindow, Sensors
from .trt_worker_uni import TRTWorkerUni


class _CausalLowPass:
    """Streaming causal low-pass via cascaded 1st-order sections."""

    def __init__(self, fs_hz: float, cutoff_hz: float, order: int = 4):
        self.order = max(1, int(order))
        if cutoff_hz <= 0.0:
            self.alpha = 1.0
        else:
            dt = 1.0 / float(fs_hz)
            tau = 1.0 / (2.0 * np.pi * float(cutoff_hz))
            self.alpha = dt / (tau + dt)
        self.state = [0.0] * self.order
        self.initialized = False

    def update(self, x: float) -> float:
        x = float(x)
        if not self.initialized:
            self.state = [x] * self.order
            self.initialized = True
            return x
        y = x
        for i in range(self.order):
            self.state[i] = self.state[i] + self.alpha * (y - self.state[i])
            y = self.state[i]
        return float(y)


class _MotionGate:
    """Per-side motion gate based on hip velocity channel energy."""

    def __init__(self, fs: int, dt: float):
        self.fs = fs
        self.dt = dt

        self.motion_score_tau = 0.08
        self.motion_window_s = 0.20
        self.start_thresh = 0.4
        self.stop_thresh = 0.25
        self.start_confirm_s = 0.15
        self.stop_confirm_s = 0.15
        self.start_delay_s = 0.10
        self.ramp_up_s = 0.40
        self.ramp_down_s = 0.12

        self.motion_score = 0.0
        self.assist_gate = 0.0
        self.motion_state = "idle"
        self.start_timer = 0.0
        self.motion_on_count = 0
        self.motion_off_count = 0

    def _lpf(self, x_prev: float, x_raw: float, tau: float) -> float:
        a = self.dt / tau if tau > 0.0 else 1.0
        return float(x_prev + a * (x_raw - x_prev))

    def update(self, vel_window: np.ndarray) -> float:
        """vel_window: 1-D array of recent (sign-corrected) hip velocities."""
        n = max(1, int(self.motion_window_s * self.fs))
        score_raw = float(np.mean(np.abs(vel_window[-n:])))
        self.motion_score = self._lpf(self.motion_score, score_raw, self.motion_score_tau)

        start_req = max(1, int(self.start_confirm_s * self.fs))
        stop_req = max(1, int(self.stop_confirm_s * self.fs))

        if self.motion_score > self.start_thresh:
            self.motion_on_count += 1
            self.motion_off_count = 0
        elif self.motion_score < self.stop_thresh:
            self.motion_off_count += 1
            self.motion_on_count = 0
        else:
            self.motion_on_count = 0
            self.motion_off_count = 0

        if self.motion_state == "idle":
            if self.motion_on_count >= start_req:
                self.motion_state = "starting"
                self.start_timer = 0.0
        elif self.motion_state == "starting":
            if self.motion_off_count >= stop_req:
                self.motion_state = "idle"
                self.start_timer = 0.0
            else:
                self.start_timer += self.dt
                if self.start_timer >= self.start_delay_s:
                    self.motion_state = "active"
        elif self.motion_state == "active":
            if self.motion_off_count >= stop_req:
                self.motion_state = "idle"
                self.start_timer = 0.0

        gate_target = 1.0 if self.motion_state == "active" else 0.0
        gate_tau = self.ramp_up_s if gate_target > self.assist_gate else self.ramp_down_s
        self.assist_gate = float(np.clip(
            self._lpf(self.assist_gate, gate_target, gate_tau), 0.0, 1.0
        ))
        return self.assist_gate

    def state_int(self) -> int:
        return {"idle": 0, "starting": 1, "active": 2}.get(self.motion_state, 0)


class CascadeHip(BaseController):
    name = "cascade_hip"

    def __init__(self, config: dict):
        self.engine_path  = str(config["trt_engine_path"])
        self.T            = int(config["frame_length"])
        self.fs           = int(config["fs"])
        self.dt           = 1.0 / self.fs

        self.hip_angle_mean = float(config.get("hip_angle_mean", 0.0))
        self.hip_angle_std  = float(config.get("hip_angle_std",  1.0))
        self.hip_vel_mean   = float(config.get("hip_vel_mean",   0.0))
        self.hip_vel_std    = float(config.get("hip_vel_std",    1.0))

        self.mass         = float(config["mass"])
        # accept either "scale" (flat-config style, like cascade_0425) or "torque_scale"
        self.torque_scale = float(config.get("scale", config.get("torque_scale", 1.0)))
        self.torque_limit = float(config.get("torque_limit", 15.0))

        self.input_size  = int(config.get("input_size",  2))
        self.output_size = int(config.get("output_size", 1))
        if self.input_size != 2:
            raise ValueError(f"cascade_hip expects input_size=2, got {self.input_size}")
        if self.output_size != 1:
            raise ValueError(f"cascade_hip expects output_size=1, got {self.output_size}")

        # batch=2: index 0 = right, index 1 = left
        self.in_shape  = (2, self.input_size, self.T)
        self.out_shape = (self.output_size,)  # per-sample shape → worker returns (2, 1)

        self.x_r      = RollingWindow((self.input_size, self.T))
        self.x_l      = RollingWindow((self.input_size, self.T))
        self.last_out = np.zeros((2, self.output_size), dtype=np.float32)

        self.in_q  = mp.Queue(maxsize=1)
        self.out_q = mp.Queue(maxsize=1)
        self.worker = TRTWorkerUni(
            self.in_q, self.out_q,
            self.engine_path,
            self.in_shape,
            self.out_shape,
        )
        self.worker.daemon = True

        # per-channel inference LPF (applied before model input)
        self.infer_lpf_hz    = float(config.get("infer_lpf_hz",    4.0))
        self.infer_lpf_order = int(config.get("infer_lpf_order",   4))
        self.infer_angle_lpf_r = _CausalLowPass(self.fs, self.infer_lpf_hz, self.infer_lpf_order)
        self.infer_angle_lpf_l = _CausalLowPass(self.fs, self.infer_lpf_hz, self.infer_lpf_order)
        self.infer_vel_lpf_r = _CausalLowPass(self.fs, self.infer_lpf_hz, self.infer_lpf_order)
        self.infer_vel_lpf_l = _CausalLowPass(self.fs, self.infer_lpf_hz, self.infer_lpf_order)
        self.infer_out_lpf_r = _CausalLowPass(self.fs, self.infer_lpf_hz, self.infer_lpf_order)
        self.infer_out_lpf_l = _CausalLowPass(self.fs, self.infer_lpf_hz, self.infer_lpf_order)

        # filtered states for telemetry
        self.hip_angle_filt_r = 0.0
        self.hip_vel_filt_r = 0.0
        self.hip_angle_filt_l = 0.0
        self.hip_vel_filt_l = 0.0

        self.hip_filter_tau = 0.05
        self.imu_filter_tau = 0.15
        self.torque_filter_tau = 0.05

        # rate limiting
        self.cmd_rate_max = 200.0
        self.prev_cmd_r = 0.0
        self.prev_cmd_l = 0.0
        self.torque_filt_r = 0.0
        self.torque_filt_l = 0.0

        # per-side motion gates
        self.gate_r = _MotionGate(self.fs, self.dt)
        self.gate_l = _MotionGate(self.fs, self.dt)

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
    def _lpf(self, x_prev: float, x_raw: float, tau: float) -> float:
        a = self.dt / tau if tau > 0.0 else 1.0
        return float(x_prev + a * (x_raw - x_prev))

    def _normalize(self, x: float, mean: float, std: float) -> float:
        s = float(std) if float(std) != 0.0 else 1.0
        return float((x - mean) / s)

    def _rate_limit(self, current: float, prev: float) -> float:
        max_step = self.cmd_rate_max * self.dt
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

    # ------------------------------------------------------------------ #
    #  Main control step
    # ------------------------------------------------------------------ #
    def step(self, s: Sensors) -> CtrlResult:
        pelvis_gyr_y = np.deg2rad(float(s.imu_P[4]))   # positive pelvis Y-axis

        # ---- right side (positive convention) ----
        enc_r_raw = float(s.pos_R)
        # hip_vel = −thigh_Gyr_Y − pelvis_Gyr_Y
        vel_r_raw = -np.deg2rad(float(s.imu_R[4])) - pelvis_gyr_y

        # ---- left side (sign-corrected to match right convention) ----
        enc_l_raw = -float(s.pos_L)
        # raw left: −thigh_Gyr_Y − pelvis_Gyr_Y → negate for right convention
        vel_l_raw = np.deg2rad(float(s.imu_L[4])) + pelvis_gyr_y

        # ---- telemetry LPFs (display only) ----
        self.hip_angle_filt_r = self._lpf(self.hip_angle_filt_r, enc_r_raw, self.hip_filter_tau)
        self.hip_vel_filt_r = self._lpf(self.hip_vel_filt_r, vel_r_raw, self.imu_filter_tau)
        self.hip_angle_filt_l = self._lpf(self.hip_angle_filt_l, enc_l_raw, self.hip_filter_tau)
        self.hip_vel_filt_l = self._lpf(self.hip_vel_filt_l, vel_l_raw, self.imu_filter_tau)

        # ---- inference-path LPF then build rolling windows ----
        angle_r = self.infer_angle_lpf_r.update(enc_r_raw)
        vel_r = self.infer_vel_lpf_r.update(vel_r_raw)
        angle_l = self.infer_angle_lpf_l.update(enc_l_raw)
        vel_l = self.infer_vel_lpf_l.update(vel_l_raw)

        x_last_r = np.array([angle_r, vel_r], dtype=np.float32)
        x_last_l = np.array([angle_l, vel_l], dtype=np.float32)

        seq_r = self.x_r.push_last(x_last_r)   # (2, T)
        seq_l = self.x_l.push_last(x_last_l)   # (2, T)

        # ---- per-side motion gates ----
        assist_gate_r = self.gate_r.update(seq_r[1])   # velocity channel
        assist_gate_l = self.gate_l.update(seq_l[1])

        # ---- batch input: (2, 2, T) — row 0 = right, row 1 = left ----
        x_batch = np.stack([seq_r, seq_l], axis=0).astype(np.float32)   # (2, 2, T)

        # ---- TRT inference (non-blocking, keep latest) ----
        latest = self._get_latest_inference()
        if latest is not None:
            # latest shape: (2, 1) from TRTWorkerUni
            self.last_out = np.asarray(latest, dtype=np.float32).reshape(2, self.output_size)
        self._try_put_latest(x_batch)

        # ---- post-process outputs ----
        # model outputs Nm/kg (right convention for both sides)
        model_out_r_raw = float(self.last_out[0, 0])
        model_out_l_raw = float(self.last_out[1, 0])

        model_out_r = self.infer_out_lpf_r.update(model_out_r_raw)
        model_out_l = self.infer_out_lpf_l.update(model_out_l_raw)

        # FIXME: temporary sign flip — remove once output sign is verified on hardware
        moment_r = -(model_out_r * self.mass * self.torque_scale)
        # left output is in right convention — negate to get actual left-side torque
        # FIXME: temporary sign flip — remove once output sign is verified on hardware
        moment_l = model_out_l * self.mass * self.torque_scale

        # ---- torque LPF → rate limit → clamp ----
        self.torque_filt_r = self._lpf(self.torque_filt_r, moment_r, self.torque_filter_tau)
        self.torque_filt_l = self._lpf(self.torque_filt_l, moment_l, self.torque_filter_tau)

        tau_r = self._rate_limit(self.torque_filt_r, self.prev_cmd_r)
        tau_l = self._rate_limit(self.torque_filt_l, self.prev_cmd_l)

        tau_r = float(np.clip(tau_r, -self.torque_limit, self.torque_limit))
        tau_l = float(np.clip(tau_l, -self.torque_limit, self.torque_limit))

        self.prev_cmd_r = tau_r
        self.prev_cmd_l = tau_l

        return CtrlResult(
            model_out_R=float(model_out_r),
            model_out_L=float(model_out_l),
            applied_R=tau_r,
            applied_L=tau_l,
            extra={
                "hip_angle_r":         float(self.hip_angle_filt_r),
                "hip_angle_l":         float(self.hip_angle_filt_l),
                "hip_vel_r":           float(self.hip_vel_filt_r),
                "hip_vel_l":           float(self.hip_vel_filt_l),
                "model_in_angle_raw":  float(enc_r_raw),
                "model_in_vel_raw":    float(vel_r_raw),
                "model_out_nmpkg":     float(model_out_r),
                "moment_raw":          float(moment_r),
                "assist_gate_r":       assist_gate_r,
                "assist_gate_l":       assist_gate_l,
                "motion_score_r":      float(self.gate_r.motion_score),
                "motion_score_l":      float(self.gate_l.motion_score),
                "state_r":             self.gate_r.state_int(),
                "state_l":             self.gate_l.state_int(),
            },
        )

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

Optional YAML `delay` (seconds): FIFO delay on scaled joint torque before LPF / rate limit
(same as knee biotorque / hip cascade_hip). Delay length in samples ≈ round(delay * fs).
"""

import multiprocessing as mp
from collections import deque
from queue import Empty, Full

import numpy as np

from .base import BaseController, CtrlResult, RollingWindow, Sensors
from .trt_worker_uni import TRTWorkerUni


class _CausalLowPass:
    """Streaming causal low-pass via cascaded 1st-order sections."""

    def __init__(self, fs_hz: float, cutoff_hz: float, order: int = 4):
        self.fs_hz = float(fs_hz)
        self.cutoff_hz = float(cutoff_hz)
        self.order = max(1, int(order))
        if self.cutoff_hz <= 0.0:
            self.alpha = 1.0
        else:
            dt = 1.0 / self.fs_hz
            tau = 1.0 / (2.0 * np.pi * self.cutoff_hz)
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

        # Training run 0423_ik_id_knee_huber_noise used normalize=False.
        # Inputs to the deployed model should therefore be raw (sign-corrected) values.
        # Keep optional stats only for debug telemetry if provided in YAML.
        self.knee_angle_mean = float(config.get("knee_angle_mean", 0.0))
        self.knee_angle_std  = float(config.get("knee_angle_std", 1.0))
        self.knee_vel_mean   = float(config.get("knee_vel_mean", 0.0))
        self.knee_vel_std    = float(config.get("knee_vel_std", 1.0))

        # Per-signal inference-path LPF settings.
        # Each signal reads its own key; falls back to infer_lpf_hz / infer_lpf_order if absent.
        _lpf_hz_default    = float(config.get("infer_lpf_hz",    4.0))
        _lpf_order_default = int(config.get("infer_lpf_order",   4))

        _angle_lpf_hz    = float(config.get("angle_lpf_hz",    _lpf_hz_default))
        _angle_lpf_order = int(config.get("angle_lpf_order",   _lpf_order_default))
        _vel_lpf_hz      = float(config.get("vel_lpf_hz",      _lpf_hz_default))
        _vel_lpf_order   = int(config.get("vel_lpf_order",     _lpf_order_default))
        _out_lpf_hz      = float(config.get("out_lpf_hz",      _lpf_hz_default))
        _out_lpf_order   = int(config.get("out_lpf_order",     _lpf_order_default))

        # Model outputs Nm/kg (moments stored as N*m/kg in training dataset).
        # Multiply by subject mass to recover Nm, then by torque_scale for tuning.
        self.mass         = float(config["mass"])
        self.torque_scale = float(config.get("scale", 1.0))
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

        # Inference-path causal low-pass filters (each signal independently configured)
        self.infer_angle_lpf = _CausalLowPass(self.fs, _angle_lpf_hz, _angle_lpf_order)
        self.infer_vel_lpf   = _CausalLowPass(self.fs, _vel_lpf_hz,   _vel_lpf_order)
        self.infer_out_lpf   = _CausalLowPass(self.fs, _out_lpf_hz,   _out_lpf_order)

        # filtered states
        self.knee_angle_filt   = 0.0   # encoder angle (rad)
        self.knee_u_enc_filt   = 0.0   # encoder-derived knee velocity (rad/s)
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

        # Optional output delay (seconds): FIFO on scaled torque before LPF / rate limit.
        self.delay = float(config.get("delay", 0.0))
        self.delay_steps = max(0, int(round(self.delay * self.fs)))
        dlen = max(1, self.delay_steps + 1)
        self.torque_buf = deque([0.0] * dlen, maxlen=dlen)

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

    def _delay_push_and_get(self, buf: deque, value: float) -> float:
        buf.append(float(value))
        return float(buf[0])

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
            encoder_raw     = np.deg2rad(float(s.pos_R))
            enc_vel_raw     = np.deg2rad(float(s.vel_R))
            thigh_gz_raw    = float(s.imu_R1[5])
            shank_gz_raw    = float(s.imu_R2[5])
        else:
            encoder_raw     = -np.deg2rad(float(s.pos_L))
            enc_vel_raw     = -np.deg2rad(float(s.vel_L))
            thigh_gz_raw    = -float(s.imu_L1[5])
            shank_gz_raw    = -float(s.imu_L2[5])

        # knee angular velocity from IMUs: shank_gz − thigh_gz
        knee_vel_imu_raw = shank_gz_raw - thigh_gz_raw

        # ---- LPF for filtered telemetry / state display ----
        self.knee_angle_filt   = self._lpf(self.knee_angle_filt,   encoder_raw,      self.knee_filter_tau)
        self.knee_u_enc_filt   = self._lpf(self.knee_u_enc_filt,   enc_vel_raw,      self.knee_filter_tau)
        self.knee_vel_imu_filt = self._lpf(self.knee_vel_imu_filt, knee_vel_imu_raw, self.imu_filter_tau)

        # ---- build model input (raw, sign-corrected; matches training normalize=False) ----
        encoder_for_model = self.infer_angle_lpf.update(encoder_raw)
        knee_vel_for_model = self.infer_vel_lpf.update(knee_vel_imu_raw)
        knee_angle_norm = self._normalize(encoder_for_model, self.knee_angle_mean, self.knee_angle_std)
        knee_vel_norm = self._normalize(knee_vel_for_model, self.knee_vel_mean, self.knee_vel_std)
        x_last = np.array([
            encoder_for_model,
            knee_vel_for_model,
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
        model_out_nmpkg_raw = float(self.last_out[0])
        model_out_nmpkg = self.infer_out_lpf.update(model_out_nmpkg_raw)
        moment_raw = model_out_nmpkg * self.mass * self.torque_scale

        # Keep assist_gate as a diagnostic signal (biotorque parity), but do not
        # apply it to torque.
        moment_cmd = moment_raw

        moment_delayed = self._delay_push_and_get(self.torque_buf, moment_cmd)

        # torque LPF → rate limit
        self.torque_filt = self._lpf(self.torque_filt, moment_delayed, self.torque_filter_tau)
        tau = self._rate_limit(self.torque_filt, self.prev_cmd, self.cmd_rate_max)
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
                "model_in_knee_angle_raw": float(encoder_raw),
                "model_in_knee_vel_raw": float(knee_vel_imu_raw),
                "model_in_knee_angle_lpf": float(encoder_for_model),
                "model_in_knee_vel_lpf": float(knee_vel_for_model),
                # "norm" fields are debug-only references; model input uses raw fields.
                "model_in_knee_angle_norm": float(knee_angle_norm),
                "model_in_knee_vel_norm": float(knee_vel_norm),
                "model_out_nmpkg_raw": float(model_out_nmpkg_raw),
                "model_out_nmpkg": float(model_out_nmpkg),
                "moment_raw":      float(moment_raw),
                "moment_delayed":  float(moment_delayed),
                "moment_cmd":      float(moment_cmd),
                "assist_gate":     float(self.assist_gate),
                "motion_score":    float(self.motion_score),
                "state":           state_int,
                # keys read by main_knee.py teleplot + data_log
                "knee_angle_r":    float(self.knee_angle_filt) if self.side == "right" else 0.0,
                "knee_angle_l":    float(self.knee_angle_filt) if self.side == "left"  else 0.0,
                "knee_angle_r_u":  float(self.knee_u_enc_filt) if self.side == "right" else 0.0,
                "knee_angle_l_u":  float(self.knee_u_enc_filt) if self.side == "left"  else 0.0,
                "knee_r_u_gyr":    float(self.knee_vel_imu_filt) if self.side == "right" else 0.0,
                "knee_l_u_gyr":    float(self.knee_vel_imu_filt) if self.side == "left"  else 0.0,
                "assist_gate_r":   float(self.assist_gate) if self.side == "right" else 0.0,
                "assist_gate_l":   float(self.assist_gate) if self.side == "left"  else 0.0,
                "state_r":         state_int if self.side == "right" else 0,
                "state_l":         state_int if self.side == "left"  else 0,
            },
        )

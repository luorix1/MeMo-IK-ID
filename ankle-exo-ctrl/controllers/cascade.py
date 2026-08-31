"""
CascadeUni — Unilateral ankle controller driven by the os_kinetics TCN moment estimator.

Model I/O (matches TRTWorkerUni / sagittal_ankle unilateral training):
  input  : (1, 2, T)  — [ankle_angle, ankle_vel_enc]
  output : (1, 1)     → scalar Nm/kg

Channel convention (sign-corrected so both sides look like right):
  ch0 — ankle encoder angle (rad)
  ch1 — ankle angular velocity (rad/s) = time derivative of encoder
        (``vel_source: encoder_diff`` in main_ankle.py)

Units: Sensors.pos_* / vel_* are already radians / rad/s (TMotor path).
Unlike knee Teensy path, do **not** apply deg2rad here.

Optional YAML `delay` (seconds): FIFO delay on scaled joint torque before LPF / rate limit.
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
        self.T = int(config["frame_length"])
        self.fs = int(config["fs"])
        self.dt = 1.0 / self.fs

        # Training typically uses normalize=False; keep optional stats for debug.
        self.ankle_angle_mean = float(config.get("ankle_angle_mean", 0.0))
        self.ankle_angle_std = float(config.get("ankle_angle_std", 1.0))
        self.ankle_vel_mean = float(config.get("ankle_vel_mean", 0.0))
        self.ankle_vel_std = float(config.get("ankle_vel_std", 1.0))

        _lpf_hz_default = float(config.get("infer_lpf_hz", 4.0))
        _lpf_order_default = int(config.get("infer_lpf_order", 4))

        _angle_lpf_hz = float(config.get("angle_lpf_hz", _lpf_hz_default))
        _angle_lpf_order = int(config.get("angle_lpf_order", _lpf_order_default))
        _vel_lpf_hz = float(config.get("vel_lpf_hz", _lpf_hz_default))
        _vel_lpf_order = int(config.get("vel_lpf_order", _lpf_order_default))
        _out_lpf_hz = float(config.get("out_lpf_hz", _lpf_hz_default))
        _out_lpf_order = int(config.get("out_lpf_order", _lpf_order_default))
        self.input_lpf_enabled = bool(config.get("input_lpf_enabled", True))

        self.mass = float(config["mass"])
        self.torque_scale = float(config.get("scale", 1.0))
        self.torque_limit = float(config.get("torque_limit", 20.0))

        self.input_size = int(config.get("input_size", 2))
        self.output_size = int(config.get("output_size", 1))
        if self.input_size != 2:
            raise ValueError(f"cascade_uni expects input_size=2, got {self.input_size}")
        if self.output_size != 1:
            raise ValueError(f"cascade_uni expects output_size=1, got {self.output_size}")

        self.in_shape = (1, self.input_size, self.T)
        self.out_shape = (self.output_size,)

        self.x = RollingWindow((self.input_size, self.T))
        self.last_out = np.zeros(self.out_shape, dtype=np.float32)

        self.in_q = mp.Queue(maxsize=1)
        self.out_q = mp.Queue(maxsize=1)
        self.worker = TRTWorkerUni(
            self.in_q,
            self.out_q,
            self.engine_path,
            self.in_shape,
            self.out_shape,
        )
        self.worker.daemon = True

        self.infer_angle_lpf = _CausalLowPass(self.fs, _angle_lpf_hz, _angle_lpf_order)
        self.infer_vel_lpf = _CausalLowPass(self.fs, _vel_lpf_hz, _vel_lpf_order)
        self.infer_out_lpf = _CausalLowPass(self.fs, _out_lpf_hz, _out_lpf_order)

        self.ankle_angle_filt = 0.0
        self.ankle_vel_enc_filt = 0.0
        self.torque_filt = 0.0

        self.ankle_filter_tau = 0.05
        self.torque_filter_tau = 0.05

        self.cmd_rate_max = 200.0
        self.prev_cmd = 0.0

        self.motion_score = 0.0
        self.assist_gate = 0.0
        self.motion_state = "idle"
        self.start_timer = 0.0
        self.motion_on_count = 0
        self.motion_off_count = 0

        self.motion_score_tau = 0.08
        self.motion_window_s = 0.20
        self.start_thresh = 0.4
        self.stop_thresh = 0.25
        self.start_confirm_s = 0.15
        self.stop_confirm_s = 0.15
        self.start_delay_s = 0.10
        self.ramp_up_s = 0.40
        self.ramp_down_s = 0.12

        self.delay = float(config.get("delay", 0.0))
        self.delay_steps = max(0, int(round(self.delay * self.fs)))
        dlen = max(1, self.delay_steps + 1)
        self.torque_buf = deque([0.0] * dlen, maxlen=dlen)

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
        n = max(1, int(self.motion_window_s * self.fs))
        score_raw = float(np.mean(np.abs(seq[1:2, -n:])))
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
        self.assist_gate = float(
            np.clip(self._lpf(self.assist_gate, gate_target, gate_tau), 0.0, 1.0)
        )

    def step(self, s: Sensors) -> CtrlResult:
        # Sign-correct to right convention. Sensors are already rad / rad/s.
        if self.side == "right":
            encoder_raw = float(s.pos_R)
            enc_vel_raw = float(s.vel_R)
        else:
            encoder_raw = -float(s.pos_L)
            enc_vel_raw = -float(s.vel_L)

        self.ankle_angle_filt = self._lpf(
            self.ankle_angle_filt, encoder_raw, self.ankle_filter_tau
        )
        self.ankle_vel_enc_filt = self._lpf(
            self.ankle_vel_enc_filt, enc_vel_raw, self.ankle_filter_tau
        )

        if self.input_lpf_enabled:
            encoder_for_model = self.infer_angle_lpf.update(encoder_raw)
            ankle_vel_for_model = self.infer_vel_lpf.update(enc_vel_raw)
        else:
            encoder_for_model = float(encoder_raw)
            ankle_vel_for_model = float(enc_vel_raw)

        ankle_angle_norm = self._normalize(
            encoder_for_model, self.ankle_angle_mean, self.ankle_angle_std
        )
        ankle_vel_norm = self._normalize(
            ankle_vel_for_model, self.ankle_vel_mean, self.ankle_vel_std
        )

        # Training normalize=False → feed raw (optionally LPF'd) channels.
        x_last = np.array([encoder_for_model, ankle_vel_for_model], dtype=np.float32)
        seq = self.x.push_last(x_last)
        x = seq.reshape(self.in_shape).astype(np.float32, copy=False)

        self._update_motion_gate(seq)

        latest = self._get_latest_inference()
        if latest is not None:
            self.last_out = np.asarray(latest, dtype=np.float32).reshape(self.out_shape)
        self._try_put_latest(x)

        model_out_nmpkg_raw = float(self.last_out[0])
        model_out_nmpkg = self.infer_out_lpf.update(model_out_nmpkg_raw)
        moment_raw = model_out_nmpkg * self.mass * self.torque_scale
        moment_cmd = moment_raw
        moment_delayed = self._delay_push_and_get(self.torque_buf, moment_cmd)

        self.torque_filt = self._lpf(self.torque_filt, moment_delayed, self.torque_filter_tau)
        tau = self._rate_limit(self.torque_filt, self.prev_cmd, self.cmd_rate_max)
        self.prev_cmd = tau

        state_int = {"idle": 0, "starting": 1, "active": 2}.get(self.motion_state, 0)
        tau_r = tau if self.side == "right" else 0.0
        tau_l = tau if self.side == "left" else 0.0

        return CtrlResult(
            model_out_R=tau_r,
            model_out_L=tau_l,
            applied_R=tau_r,
            applied_L=tau_l,
            extra={
                "side": self.side,
                "ankle_angle": float(self.ankle_angle_filt),
                "ankle_vel_enc": float(self.ankle_vel_enc_filt),
                "model_in_ankle_angle_raw": float(encoder_raw),
                "model_in_ankle_vel_raw": float(enc_vel_raw),
                "model_in_ankle_angle_raw_r": float(encoder_raw) if self.side == "right" else 0.0,
                "model_in_ankle_angle_raw_l": float(encoder_raw) if self.side == "left" else 0.0,
                "model_in_ankle_vel_raw_r": float(enc_vel_raw) if self.side == "right" else 0.0,
                "model_in_ankle_vel_raw_l": float(enc_vel_raw) if self.side == "left" else 0.0,
                "model_in_ankle_angle_lpf": float(encoder_for_model),
                "model_in_ankle_vel_lpf": float(ankle_vel_for_model),
                "model_in_ankle_angle_norm": float(ankle_angle_norm),
                "model_in_ankle_vel_norm": float(ankle_vel_norm),
                "model_out_nmpkg_raw": float(model_out_nmpkg_raw),
                "model_out_nmpkg": float(model_out_nmpkg),
                "moment_raw": float(moment_raw),
                "moment_delayed": float(moment_delayed),
                "moment_cmd": float(moment_cmd),
                "assist_gate": float(self.assist_gate),
                "motion_score": float(self.motion_score),
                "state": state_int,
                "ankle_angle_r": float(self.ankle_angle_filt) if self.side == "right" else 0.0,
                "ankle_angle_l": float(self.ankle_angle_filt) if self.side == "left" else 0.0,
                "ankle_vel_r": float(self.ankle_vel_enc_filt) if self.side == "right" else 0.0,
                "ankle_vel_l": float(self.ankle_vel_enc_filt) if self.side == "left" else 0.0,
                "assist_gate_r": float(self.assist_gate) if self.side == "right" else 0.0,
                "assist_gate_l": float(self.assist_gate) if self.side == "left" else 0.0,
                "state_r": state_int if self.side == "right" else 0,
                "state_l": state_int if self.side == "left" else 0,
            },
        )

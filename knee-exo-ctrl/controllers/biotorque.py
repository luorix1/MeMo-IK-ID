from collections import deque
import multiprocessing as mp
from queue import Empty, Full

import numpy as np

from .base import BaseController, CtrlResult, RollingWindow, Sensors
from .trt_worker import TRTWorker


class Biotorque(BaseController):
    name = "biotorque"

    def __init__(self, config: dict):
        self.engine_path = config["trt_engine_path"]
        self.T = int(config["frame_length"])
        self.fs = int(config["fs"])
        self.dt = 1.0 / self.fs

        self.mass = float(config["mass"])
        self.biotorque_gain = float(config["biotorque_gain"])
        self.delay = float(config["delay"])

        self.thigh_gy_mean = float(config["thigh_gy_mean"])
        self.thigh_gy_std = float(config["thigh_gy_std"])
        self.shank_gy_mean = float(config["shank_gy_mean"])
        self.shank_gy_std = float(config["shank_gy_std"])
        self.knee_mean = float(config["knee_mean"])
        self.knee_std = float(config["knee_std"])

        self.input_size = int(config["input_size"])
        if self.input_size != 3:
            raise ValueError(f"biotorque expects input_size=3, got {self.input_size}")

        self.output_size = int(config.get("output_size", 1))
        if self.output_size != 1:
            raise ValueError(f"biotorque expects output_size=1, got {self.output_size}")

        self.in_shape = (1, self.input_size, self.T)
        self.out_shape = (self.output_size,)

        self.x_r = RollingWindow((self.input_size, self.T))
        self.x_l = RollingWindow((self.input_size, self.T))

        self.last_out_r = np.zeros(self.out_shape, dtype=np.float32)
        self.last_out_l = np.zeros(self.out_shape, dtype=np.float32)

        self.in_q = mp.Queue(maxsize=1)
        self.out_q = mp.Queue(maxsize=1)
        self.worker = TRTWorker(
            self.in_q,
            self.out_q,
            self.engine_path,
            self.in_shape,
            self.out_shape,
        )
        self.worker.daemon = True

        self.knee_filter_tau = 0.05
        self.imu_filter_tau = 0.15
        self.torque_filter_tau = 0.15
        self.cmd_rate_max = 10000

        self.knee_r_filt = 0.0
        self.knee_l_filt = 0.0
        self.knee_r_u_filt = 0.0
        self.knee_l_u_filt = 0.0
        self.thigh_gyR = 0.0
        self.shank_gyR = 0.0
        self.thigh_gyL = 0.0
        self.shank_gyL = 0.0
        self.torque_r_filt = 0.0
        self.torque_l_filt = 0.0
        self.prev_cmd_r = 0.0
        self.prev_cmd_l = 0.0

        self.motion_score_r = 0.0
        self.motion_score_l = 0.0
        self.assist_gate_r = 0.0
        self.assist_gate_l = 0.0
        self.motion_state_r = "idle"
        self.motion_state_l = "idle"
        self.start_timer_r = 0.0
        self.start_timer_l = 0.0
        self.motion_on_count_r = 0
        self.motion_on_count_l = 0
        self.motion_off_count_r = 0
        self.motion_off_count_l = 0

        self.motion_score_tau = 0.08
        self.motion_window_s = 0.20
        self.start_thresh = 0.4
        self.stop_thresh = 0.25
        self.start_confirm_s = 0.15
        self.stop_confirm_s = 0.15
        self.start_delay_s = 0.10
        self.ramp_up_s = 0.40
        self.ramp_down_s = 0.12

        self.delay_steps = max(0, int(round(self.delay * self.fs)))
        self.torque_buf_r = deque([0.0] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)
        self.torque_buf_l = deque([0.0] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)

    def _update_motion_gate(self, seq, score_prev, gate_prev, state, start_timer, on_count, off_count):
        n = max(1, int(self.motion_window_s * self.fs))
        recent = seq[:, -n:]
        score_raw = float(np.mean(np.abs(recent)))
        score = self._lpf(score_prev, score_raw, self.motion_score_tau)

        start_count_req = max(1, int(self.start_confirm_s * self.fs))
        stop_count_req = max(1, int(self.stop_confirm_s * self.fs))

        if score > self.start_thresh:
            on_count += 1
            off_count = 0
        elif score < self.stop_thresh:
            off_count += 1
            on_count = 0
        else:
            on_count = 0
            off_count = 0

        if state == "idle":
            if on_count >= start_count_req:
                state = "starting"
                start_timer = 0.0
        elif state == "starting":
            if off_count >= stop_count_req:
                state = "idle"
                start_timer = 0.0
            else:
                start_timer += self.dt
                if start_timer >= self.start_delay_s:
                    state = "active"
        elif state == "active":
            if off_count >= stop_count_req:
                state = "idle"
                start_timer = 0.0

        gate_target = 1.0 if state == "active" else 0.0
        gate_tau = self.ramp_up_s if gate_target > gate_prev else self.ramp_down_s
        gate = float(np.clip(self._lpf(gate_prev, gate_target, gate_tau), 0.0, 1.0))
        return gate, score, state, start_timer, on_count, off_count

    def _alpha(self, tau: float) -> float:
        if tau <= 0.0:
            return 1.0
        return self.dt / tau

    def _lpf(self, x_prev: float, x_raw: float, tau: float) -> float:
        a = self._alpha(tau)
        return float(x_prev + a * (x_raw - x_prev))

    def _normalize(self, x: float, x_mean: float, x_std: float) -> float:
        return float((x - x_mean) / x_std)

    def _get_latest_inference(self):
        latest = None
        try:
            while True:
                latest = self.out_q.get_nowait()
        except Empty:
            pass
        return latest

    def _try_put_latest(self, payload) -> None:
        try:
            self.in_q.put_nowait(payload)
        except Full:
            pass

    def _rate_limit(self, current: float, prev: float, rate_max: float) -> float:
        max_step = rate_max * self.dt
        return float(prev + np.clip(current - prev, -max_step, max_step))

    def _delay_push_and_get(self, buf: deque, value: float) -> float:
        buf.append(float(value))
        return float(buf[0])

    def _build_model_input(self, thigh_gy: float, shank_gy: float, knee_angle: float) -> np.ndarray:
        x_last = np.array(
            [
                self._normalize(thigh_gy, self.thigh_gy_mean, self.thigh_gy_std),
                self._normalize(shank_gy, self.shank_gy_mean, self.shank_gy_std),
                self._normalize(knee_angle, self.knee_mean, self.knee_std),
            ],
            dtype=np.float32,
        )
        return x_last

    def _extract_scalar(self, y: np.ndarray) -> float:
        return float(np.asarray(y, dtype=np.float32).reshape(-1)[0])

    def step(self, s: Sensors) -> CtrlResult:
        thigh_gyR_raw = float(s.imu_R1[5])
        shank_gyR_raw = float(s.imu_R2[5])
        thigh_gyL_raw = -float(s.imu_L1[5])
        shank_gyL_raw = -float(s.imu_L2[5])

        knee_r_raw = np.deg2rad(float(s.pos_R))
        knee_r_u_raw = np.deg2rad(float(s.vel_R))
        knee_l_raw = -np.deg2rad(float(s.pos_L))
        knee_l_u_raw = -np.deg2rad(float(s.vel_L))

        self.thigh_gyR = self._lpf(self.thigh_gyR, thigh_gyR_raw, self.imu_filter_tau)
        self.shank_gyR = self._lpf(self.shank_gyR, shank_gyR_raw, self.imu_filter_tau)
        self.thigh_gyL = self._lpf(self.thigh_gyL, thigh_gyL_raw, self.imu_filter_tau)
        self.shank_gyL = self._lpf(self.shank_gyL, shank_gyL_raw, self.imu_filter_tau)

        self.knee_r_filt = self._lpf(self.knee_r_filt, knee_r_raw, self.knee_filter_tau)
        self.knee_r_u_filt = self._lpf(self.knee_r_u_filt, knee_r_u_raw, self.knee_filter_tau)
        self.knee_l_filt = self._lpf(self.knee_l_filt, knee_l_raw, self.knee_filter_tau)
        self.knee_l_u_filt = self._lpf(self.knee_l_u_filt, knee_l_u_raw, self.knee_filter_tau)

        xr_last = self._build_model_input(thigh_gyR_raw, shank_gyR_raw, self.knee_r_filt)
        xl_last = self._build_model_input(thigh_gyL_raw, shank_gyL_raw, self.knee_l_filt)

        seq_r = self.x_r.push_last(xr_last)
        seq_l = self.x_l.push_last(xl_last)

        gate_seq_r = seq_r[:2, :]
        gate_seq_l = seq_l[:2, :]

        x_r = seq_r.reshape(self.in_shape).astype(np.float32, copy=False)
        x_l = seq_l.reshape(self.in_shape).astype(np.float32, copy=False)

        new_result = self._get_latest_inference()
        if new_result is not None:
            self.last_out_r = np.asarray(new_result[0], dtype=np.float32).reshape(self.out_shape)
            self.last_out_l = np.asarray(new_result[1], dtype=np.float32).reshape(self.out_shape)
        self._try_put_latest({"r": x_r, "l": x_l})

        (
            self.assist_gate_r,
            self.motion_score_r,
            self.motion_state_r,
            self.start_timer_r,
            self.motion_on_count_r,
            self.motion_off_count_r,
        ) = self._update_motion_gate(
            gate_seq_r,
            self.motion_score_r,
            self.assist_gate_r,
            self.motion_state_r,
            self.start_timer_r,
            self.motion_on_count_r,
            self.motion_off_count_r,
        )

        (
            self.assist_gate_l,
            self.motion_score_l,
            self.motion_state_l,
            self.start_timer_l,
            self.motion_on_count_l,
            self.motion_off_count_l,
        ) = self._update_motion_gate(
            gate_seq_l,
            self.motion_score_l,
            self.assist_gate_l,
            self.motion_state_l,
            self.start_timer_l,
            self.motion_on_count_l,
            self.motion_off_count_l,
        )

        torque_r_raw = self._extract_scalar(self.last_out_r) * self.mass * self.biotorque_gain
        torque_l_raw = self._extract_scalar(self.last_out_l) * self.mass * self.biotorque_gain

        torque_r_delayed = self._delay_push_and_get(self.torque_buf_r, torque_r_raw)
        torque_l_delayed = self._delay_push_and_get(self.torque_buf_l, torque_l_raw)


        tau_r = self._rate_limit(torque_r_delayed, self.prev_cmd_r, self.cmd_rate_max)
        tau_l = self._rate_limit(torque_l_delayed, self.prev_cmd_l, self.cmd_rate_max)

        self.torque_r_filt = self._lpf(self.torque_r_filt, tau_r, self.torque_filter_tau)
        self.torque_l_filt = self._lpf(self.torque_l_filt, tau_l, self.torque_filter_tau)

        tau_r = self.torque_r_filt
        tau_l = self.torque_l_filt

        self.prev_cmd_r = tau_r
        self.prev_cmd_l = tau_l

        return CtrlResult(
            model_out_R=tau_r,
            model_out_L=tau_l,
            applied_R=tau_r,
            applied_L=tau_l,
            extra={
                "knee_angle_r": float(self.knee_r_filt),
                "knee_angle_l": float(self.knee_l_filt),
                "knee_angle_r_u": float(self.knee_r_u_filt),
                "knee_angle_l_u": float(self.knee_l_u_filt),
                "assist_gate_r": float(self.assist_gate_r),
                "assist_gate_l": float(self.assist_gate_l),
                "biotorque_raw_r": float(torque_r_raw),
                "biotorque_raw_l": float(torque_l_raw),
                "biotorque_delayed_r": float(torque_r_delayed),
                "biotorque_delayed_l": float(torque_l_delayed),
                "biotorque_filtered_r": float(self.torque_r_filt),
                "biotorque_filtered_l": float(self.torque_l_filt),
            },
        )

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

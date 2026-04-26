import numpy as np
import multiprocessing as mp
from queue import Empty, Full

from .base import BaseController, Sensors, CtrlResult, RollingWindow
from .trt_worker_uni import TRTWorkerUni


class ImpedanceRLUni(BaseController):
    """
    Unilateral knee impedance controller with TCN inference via TensorRT.

    Model input  : (1, 2, T)  — [encoder_pos_rad, knee_vel_imu]
    Model output : (2,)       — [K_cmd in [-1,1], gait_cmd in [-1,1]]

    Inputs:
      ch0 — knee motor encoder position (radians, sign-corrected per side)
      ch1 — knee angular velocity from IMUs: shank_gz - thigh_gz
             (sign-corrected per side, matching the bilateral controller convention)
    """
    name = "impedance_rl_uni"

    def __init__(self, config: dict):
        self.side = str(config["side"]).lower()
        if self.side not in ("right", "left"):
            raise ValueError(f"Invalid side: '{self.side}'. Must be 'right' or 'left'.")

        self.engine_path = config["trt_engine_path"]
        self.T  = int(config["frame_length"])   # 100
        self.fs = int(config["fs"])
        self.dt = 1.0 / self.fs

        # normalization stats for the two input channels
        self.encoder_mean    = float(config["encoder_mean"])
        self.encoder_std     = float(config["encoder_std"])
        self.knee_vel_mean   = float(config["knee_vel_mean"])
        self.knee_vel_std    = float(config["knee_vel_std"])

        self.input_size  = int(config["input_size"])   # must be 2
        self.output_size = int(config["output_size"])  # typically 2: [K, gait]

        if self.input_size != 2:
            raise ValueError(f"impedance_rl_uni expects input_size=2, got {self.input_size}")

        self.in_shape  = (1, self.input_size, self.T)  # (1, 2, T)
        self.out_shape = (self.output_size,)

        # -------- rolling window --------
        self.x = RollingWindow((self.input_size, self.T))
        self.last_out = np.zeros(self.out_shape, dtype=np.float32)
        self.last_out[0] = -1.0  # initialise K_cmd to minimum

        # -------- TRT worker --------
        self.in_q  = mp.Queue(maxsize=1)
        self.out_q = mp.Queue(maxsize=1)
        self.worker = TRTWorkerUni(
            self.in_q, self.out_q,
            self.engine_path,
            self.in_shape,
            self.out_shape,
        )
        self.worker.daemon = True

        # -------- filtered states --------
        self.encoder      = 0.0   # filtered encoder pos (rad)
        self.knee_u_enc   = 0.0   # filtered encoder velocity (rad/s)
        self.knee_vel_imu = 0.0   # filtered IMU-based knee velocity (rad/s)

        # -------- impedance params --------
        self.K_max = 60.0
        self.K_min = 0.0
        self.gait_sigma = 0.3
        self.B_coeff = 0.2
        self.knee_max_torque = 22.0
        self.flex_K_mult = 0.3

        self.damp_flex_mult = 0.0
        self.damp_ext_mult  = 4.0

        # knee reference clamps (rad)
        self.q_ext  = -np.deg2rad(0.0)
        self.q_flex = -np.deg2rad(65.0)

        # 1st-order LPF time constants
        self.knee_filter_tau      = 0.05
        self.imu_filter_tau       = 0.15
        self.impedance_filter_tau = 0.05
        self.gait_filter_tau      = 0.1

        # filtered model outputs
        self.K_filt    = 0.0
        self.gait_filt = 0.0
        self.gait_raw_prev = 0.0

        # rate limiting
        self.soft_ctrl_rate_max = 100000.0
        self.cmd_rate_max       = 200.0
        self.prev_cmd           = 0.0

        # -------- motion gating --------
        self.motion_score  = 0.0
        self.assist_gate   = 0.0
        self.motion_state  = "idle"
        self.start_timer   = 0.0
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

    def normalize(self, x: float, mean: float, std: float) -> float:
        std = float(std) if float(std) != 0.0 else 1.0
        return float((x - mean) / std)

    def _map_K(self, u: float) -> float:
        u = float(np.clip(u, -1.0, 1.0))
        return float(self.K_min + (self.K_max - self.K_min) * (u + 1.0) * 0.5)

    def _gait_weights(self, g: float):
        g = float(np.clip(g, -1.0, 1.0))
        c = np.array([-0.85, 0.0, 0.85], dtype=np.float32)
        w = np.exp(-0.5 * ((g - c) / self.gait_sigma) ** 2)
        w = w / np.sum(w)
        return float(w[0]), float(w[1]), float(w[2])

    def _rate_limit(self, current: float, prev: float, rate_max: float) -> float:
        max_step = rate_max * self.dt
        return float(prev + np.clip(current - prev, -max_step, +max_step))

    def _torque_one_leg(self, knee_f: float, knee_u_f: float, K_f: float, gait_f: float) -> float:
        w_flex, w_trans, w_ext = self._gait_weights(gait_f)
        dq_flex = float(knee_f - self.q_flex)
        dq_ext  = float(knee_f - self.q_ext)

        B_flex = self.B_coeff * np.sqrt(max(K_f * self.flex_K_mult, 0.0)) * self.damp_flex_mult
        B_ext  = (0.0 if knee_u_f > 0.0
                  else self.B_coeff * np.sqrt(max(K_f, 0.0)) * self.damp_ext_mult)

        tau_trans = 0.0
        tau_flex  = -K_f * self.flex_K_mult * dq_flex - B_flex * knee_u_f
        tau_ext   = -K_f * dq_ext - B_ext * knee_u_f

        tau = w_trans * tau_trans + w_flex * tau_flex + w_ext * tau_ext
        return float(self.knee_max_torque * np.tanh(tau / self.knee_max_torque))

    def _update_motion_gate(self, seq):
        """Update motion gating using the knee-velocity channel (ch1) of seq."""
        n = int(self.motion_window_s * self.fs)
        recent = seq[1:2, -n:]   # channel 1 = knee_vel_imu (normalised)
        score_raw = float(np.mean(np.abs(recent)))
        self.motion_score = self._lpf(self.motion_score, score_raw, self.motion_score_tau)

        start_count_req = int(self.start_confirm_s * self.fs)
        stop_count_req  = int(self.stop_confirm_s  * self.fs)

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
            if self.motion_on_count >= start_count_req:
                self.motion_state = "starting"
                self.start_timer  = 0.0
        elif self.motion_state == "starting":
            if self.motion_off_count >= stop_count_req:
                self.motion_state = "idle"
                self.start_timer  = 0.0
            else:
                self.start_timer += self.dt
                if self.start_timer >= self.start_delay_s:
                    self.motion_state = "active"
        elif self.motion_state == "active":
            if self.motion_off_count >= stop_count_req:
                self.motion_state = "idle"
                self.start_timer  = 0.0

        gate_target = 1.0 if self.motion_state == "active" else 0.0
        gate_tau    = self.ramp_up_s if gate_target > self.assist_gate else self.ramp_down_s
        self.assist_gate = float(np.clip(
            self._lpf(self.assist_gate, gate_target, gate_tau), 0.0, 1.0
        ))

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
        # ---- raw sensor extraction (sign-corrected per side) ----
        if self.side == "right":
            encoder_raw      = np.deg2rad(float(s.pos_R))
            encoder_vel_raw  = np.deg2rad(float(s.vel_R))
            thigh_gz_raw     = float(s.imu_R1[5])
            shank_gz_raw     = float(s.imu_R2[5])
        else:
            encoder_raw      = -np.deg2rad(float(s.pos_L))
            encoder_vel_raw  = -np.deg2rad(float(s.vel_L))
            thigh_gz_raw     = -float(s.imu_L1[5])
            shank_gz_raw     = -float(s.imu_L2[5])

        # knee angular velocity from IMUs: shank_z - thigh_z
        knee_vel_imu_raw = shank_gz_raw - thigh_gz_raw

        # ---- LPF for control signals ----
        self.encoder      = self._lpf(self.encoder,      encoder_raw,      self.knee_filter_tau)
        self.knee_u_enc   = self._lpf(self.knee_u_enc,   encoder_vel_raw,  self.knee_filter_tau)
        self.knee_vel_imu = self._lpf(self.knee_vel_imu, knee_vel_imu_raw, self.imu_filter_tau)

        # ---- build model input (1, 2, T) ----
        x_last = np.array([
            self.normalize(encoder_raw,      self.encoder_mean,  self.encoder_std),
            self.normalize(knee_vel_imu_raw, self.knee_vel_mean, self.knee_vel_std),
        ], dtype=np.float32)

        seq = self.x.push_last(x_last)                                   # (2, T)
        x   = seq.reshape(self.in_shape).astype(np.float32, copy=False)  # (1, 2, T)

        # ---- motion gating ----
        self._update_motion_gate(seq)

        # ---- TRT inference (non-blocking, keep latest) ----
        latest = self._get_latest_inference()
        if latest is not None:
            self.last_out = np.asarray(latest, dtype=np.float32).copy()
        self._try_put_latest(x)

        y = self.last_out.copy()

        # ---- post-process model outputs ----
        K_raw   = self._map_K(float(y[0]))
        gait_raw = float(np.clip(y[1], -1.0, 1.0)) if self.output_size >= 2 else 0.0

        gait_raw = self._rate_limit(gait_raw, self.gait_raw_prev, self.soft_ctrl_rate_max)
        self.gait_raw_prev = gait_raw

        self.K_filt    = self._lpf(self.K_filt,    K_raw,    self.impedance_filter_tau)
        self.gait_filt = self._lpf(self.gait_filt, gait_raw, self.gait_filter_tau)

        # ---- impedance torque ----
        K_ctrl = self.K_filt * self.assist_gate
        tau = self._torque_one_leg(self.encoder, self.knee_vel_imu, K_ctrl, self.gait_filt)

        # ---- output rate limiting ----
        tau = self._rate_limit(tau, self.prev_cmd, self.cmd_rate_max)
        self.prev_cmd = tau

        # ---- pack into bilateral CtrlResult (zero the inactive side) ----
        if self.motion_state == "idle":
            state_int = 0
        elif self.motion_state == "starting":
            state_int = 1
        else:
            state_int = 2

        if self.side == "right":
            tau_r, tau_l = tau, 0.0
        else:
            tau_r, tau_l = 0.0, tau

        return CtrlResult(
            model_out_R=float(tau_r),
            model_out_L=float(tau_l),
            applied_R=float(tau_r),
            applied_L=float(tau_l),
            extra={
                "side":        self.side,
                "knee_angle":  float(self.encoder),
                "knee_u_enc":  float(self.knee_u_enc),
                "knee_u_gyr":  float(self.knee_vel_imu),
                "K":           float(self.K_filt),
                "Soft_ctrl":   float(self.gait_filt),
                "assist_gate": float(self.assist_gate),
                "state":       state_int,
                # bilateral keys so existing main_knee.py logging/teleplot still works
                "knee_angle_r": float(self.encoder) if self.side == "right" else 0.0,
                "knee_angle_l": float(self.encoder) if self.side == "left"  else 0.0,
                "knee_r_u_gyr": float(self.knee_vel_imu) if self.side == "right" else 0.0,
                "knee_l_u_gyr": float(self.knee_vel_imu) if self.side == "left"  else 0.0,
                "K_r":           float(self.K_filt) if self.side == "right" else 0.0,
                "K_l":           float(self.K_filt) if self.side == "left"  else 0.0,
                "Soft_ctrl_r":   float(self.gait_filt) if self.side == "right" else 0.0,
                "Soft_ctrl_l":   float(self.gait_filt) if self.side == "left"  else 0.0,
                "assist_gate_r": float(self.assist_gate) if self.side == "right" else 0.0,
                "assist_gate_l": float(self.assist_gate) if self.side == "left"  else 0.0,
                "state_l":       state_int if self.side == "left" else 0,
                "state_r":       state_int if self.side == "right" else 0,
            },
        )

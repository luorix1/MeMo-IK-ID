import numpy as np
import multiprocessing as mp
from queue import Empty, Full
#C
from .base import BaseController, Sensors, CtrlResult, RollingWindow
from .trt_worker import TRTWorker

class impedance_rl(BaseController):
    name = "impedance_rl"

    def __init__(self, config: dict):
        self.engine_path = config["trt_engine_path"]
        self.T = int(config["frame_length"])
        self.fs = int(config["fs"])
        self.dt = 1.0 / self.fs

        self.thigh_gy_mean = float(config["thigh_gy_mean"])
        self.thigh_gy_std = float(config["thigh_gy_std"])
        self.shank_gy_mean = float(config["shank_gy_mean"])
        self.shank_gy_std = float(config["shank_gy_std"])
        self.knee_mean = float(config["knee_mean"])
        self.knee_std = float(config["knee_std"])

        self.input_size = int(config["input_size"])
        self.output_size = int(config["output_size"])

        self.in_shape = (1, self.input_size, self.T)
        self.out_shape = (self.output_size,)

        self.x_r = RollingWindow((self.input_size, self.T))
        self.x_l = RollingWindow((self.input_size, self.T))
        self.vt_rot_r = RollingWindow((1, self.T))
        self.vt_rot_l = RollingWindow((1, self.T))

        self.last_out_r = np.array([-1.0, 0.0], dtype=np.float32)
        self.last_out_l = np.array([-1.0, 0.0], dtype=np.float32)

        # -------- TRT worker --------
        self.in_q = mp.Queue(maxsize=1)
        self.out_q = mp.Queue(maxsize=1)
        self.worker = TRTWorker(self.in_q, self.out_q, self.engine_path, self.in_shape, self.out_shape)
        self.worker.daemon = True

        # -------- state (filtered signals) --------
        self.knee_r_filt = 0.0
        self.knee_l_filt = 0.0
        self.knee_r_u_filt = 0.0
        self.knee_l_u_filt = 0.0

        self.K_r_raw = 0.0
        self.K_l_raw = 0.0
        self.K_r_filt = 0.0
        self.K_l_filt = 0.0

        self.gait_r_filt = 0.0
        self.gait_l_filt = 0.0
        self.gait_r_raw_prev = 0.0
        self.gait_l_raw_prev = 0.0

        self.thigh_gyR = 0.0
        self.shank_gyR = 0.0
        self.thigh_gyL = 0.0
        self.shank_gyL = 0.0

        # -------- impedance params --------
        self.K_max = 60.0
        self.K_min = 0.0
        self.gait_sigma =  0.3
        self.B_coeff = 0.2
        self.knee_max_torque = 22.0
        self.flex_K_mult = 0.3

        self.knee_filter_tau = 0.05
        self.imu_filter_tau = 0.15
        # self.imu_filter_tau = 0.05
        self.impedance_filter_tau = 0.05
        self.gait_filter_tau = 0.1

        self.damp_flex_mult = 0.0
        self.damp_ext_mult = 4.0

        # qref clamp
        self.q_ext = -np.deg2rad(0.0)
        self.q_flex = -np.deg2rad(65.0)

        # torque rate limiting
        self.rate_max = 100000 #(x-y)/0.01
        self.soft_ctrl_r_prev = 0.0
        self.soft_ctrl_l_prev = 0.0
        self.K_r_prev = 0.0
        self.K_l_prev = 0.0
        self.cmd_rate_max = 200
        self.prev_cmd_r = 0.0
        self.prev_cmd_l = 0.0

        # -------- motion gating --------
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

        # -------- tuning params --------
        self.motion_score_tau = 0.08
        self.motion_window_s = 0.20

        self.start_thresh = 0.4
        self.stop_thresh = 0.25

        self.start_confirm_s = 0.15
        self.stop_confirm_s = 0.15
        self.start_delay_s = 0.10
        self.ramp_up_s = 0.40
        self.ramp_down_s = 0.12


    """
    -------------------------------------
    Soft Max Impedance Controller
    -------------------------------------
    """

    def _map_K(self, u: float) -> float:
        u = float(np.clip(u, -1.0, 1.0))
        return self.K_min + (self.K_max - self.K_min) * (u + 1.0) / 2.0


    def _gait_weights(self, g: float):
        g = float(np.clip(g, -1.0, 1.0))
        c = np.array([-0.85, 0.0, 0.85], dtype=np.float32)
        w = np.exp(-0.5 * ((g - c) / self.gait_sigma) ** 2)
        w = w / np.sum(w)
        return float(w[0]), float(w[1]), float(w[2])

    def _torque_one_leg(self, knee_f: float, knee_u_f: float, K_f: float, gait_f: float):
        w_flex, w_trans, w_ext = self._gait_weights(gait_f)
        dq_flex = float(knee_f - self.q_flex)
        dq_ext  = float(knee_f - self.q_ext)

        B_flex = self.B_coeff * np.sqrt(max(K_f*self.flex_K_mult, 0.0)) * self.damp_flex_mult

        if knee_u_f > 0.0:
            B_ext = 0.0
        else:
            B_ext  = self.B_coeff * np.sqrt(max(K_f, 0.0)) * self.damp_ext_mult

        tau_trans = 0.0 #!!
        tau_flex  = -K_f*self.flex_K_mult * dq_flex - B_flex * knee_u_f
        tau_ext   = -K_f * dq_ext  - B_ext  * knee_u_f
        tau = w_trans * tau_trans + w_flex * tau_flex + w_ext * tau_ext
        out  = self.knee_max_torque * np.tanh(tau / self.knee_max_torque)
        return out

    # ---------------- helpers ----------------
    def _update_motion_gate(self, seq, score_prev, gate_prev, state, start_timer, on_count, off_count):
        n = int(self.motion_window_s * self.fs)
        recent = seq[:, -n:]
        score_raw = float(np.mean(np.abs(recent)))
        score = self._lpf(score_prev, score_raw, self.motion_score_tau)

        start_count_req = int(self.start_confirm_s * self.fs)
        stop_count_req = int(self.stop_confirm_s * self.fs)

        # hysteresis-based counting
        if score > self.start_thresh:
            on_count += 1
            off_count = 0
        elif score < self.stop_thresh:
            off_count += 1
            on_count = 0
        else:
            on_count = 0
            off_count = 0

        # state machine
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

        # gate target
        gate_target = 1.0 if state == "active" else 0.0
        gate_tau = self.ramp_up_s if gate_target > gate_prev else self.ramp_down_s
        gate = self._lpf(gate_prev, gate_target, gate_tau)
        gate = float(np.clip(gate, 0.0, 1.0))
        return gate, score, state, start_timer, on_count, off_count

    def _alpha(self, tau: float) -> float:
        if tau <= 0.0:
            return 1.0
        return self.dt / tau

    def _lpf(self, x_prev: float, x_raw: float, tau: float) -> float:
        a = self._alpha(tau)
        return float(x_prev + a * (x_raw - x_prev))

    def normalize(self, x, x_mean, x_std):
        return (x - x_mean) / x_std

    
    def _get_latest_inference(self):
        latest_data = None
        try:
            while True:
                latest_data = self.out_q.get_nowait()
        except Empty:
            pass
        return latest_data

    def _try_put_latest(self, payload) -> None:
        try:
            self.in_q.put_nowait(payload)
        except Full:
            pass

    def _try_get_output(self, timeout_s: float = 0.02):
        y_r, y_l = self.last_out_r, self.last_out_l
        try:
            y_r, y_l = self.out_q.get(timeout=timeout_s)
            self.last_out_r, self.last_out_l = y_r, y_l
        except Empty:
            pass
        return y_r, y_l


    def _rate_limit(self, current: float, prev: float, rate_max: float) -> float:
        max_step = rate_max * self.dt
        return float(prev + np.clip(current - prev, -max_step, +max_step))
    def _smoothstep(self, x, x0, x1):
        z = np.clip((x - x0) / (x1 - x0 + 1e-8), 0.0, 1.0)
        return float(z * z * (3 - 2 * z))

    # ---------------- main ----------------
    def step(self, s: Sensors) -> CtrlResult:
        # ---- raw sensors ----
        thigh_gyR_raw = float(s.imu_R1[5])
        shank_gyR_raw = float(s.imu_R2[5])
        
        thigh_gyL_raw = -float(s.imu_L1[5])
        shank_gyL_raw = -float(s.imu_L2[5])

        thigh_vert_rot_r_raw = np.abs(float(s.imu_R1[3]))
        thigh_vert_rot_l_raw = np.abs(float(s.imu_L1[3]))

        knee_r_raw = np.deg2rad(float(s.pos_R))
        knee_r_u_raw = np.deg2rad(float(s.vel_R))
        knee_l_raw = -np.deg2rad(float(s.pos_L))
        knee_l_u_raw = -np.deg2rad(float(s.vel_L))

        # ---- LPF signals used for control ----
        self.thigh_gyR = self._lpf(self.thigh_gyR, thigh_gyR_raw, self.imu_filter_tau)
        self.shank_gyR = self._lpf(self.shank_gyR, shank_gyR_raw, self.imu_filter_tau)
        self.thigh_gyL = self._lpf(self.thigh_gyL, thigh_gyL_raw, self.imu_filter_tau)
        self.shank_gyL = self._lpf(self.shank_gyL, shank_gyL_raw, self.imu_filter_tau)

        self.knee_r_filt = self._lpf(self.knee_r_filt, knee_r_raw, self.knee_filter_tau)
        self.knee_r_u_filt = self._lpf(self.knee_r_u_filt, knee_r_u_raw, self.knee_filter_tau)
        self.knee_l_filt = self._lpf(self.knee_l_filt, knee_l_raw, self.knee_filter_tau)
        self.knee_l_u_filt = self._lpf(self.knee_l_u_filt, knee_l_u_raw, self.knee_filter_tau)

        # ---- build model input sequ_ence ----
        if self.input_size == 2:
            xr_last = np.array([self.normalize(thigh_gyR_raw, self.thigh_gy_mean, self.thigh_gy_std),
            self.normalize(shank_gyR_raw, self.shank_gy_mean, self.shank_gy_std)], dtype=np.float32)
            xl_last = np.array([self.normalize(thigh_gyL_raw, self.thigh_gy_mean, self.thigh_gy_std),
            self.normalize(shank_gyL_raw, self.shank_gy_mean, self.shank_gy_std)], dtype=np.float32)
        elif self.input_size == 3:
            xr_last = np.array([self.normalize(thigh_gyR_raw, self.thigh_gy_mean, self.thigh_gy_std),
            self.normalize(shank_gyR_raw, self.shank_gy_mean, self.shank_gy_std),
            self.normalize(self.knee_r_filt, self.knee_mean, self.knee_std)], dtype=np.float32)
            xl_last = np.array([self.normalize(thigh_gyL_raw, self.thigh_gy_mean, self.thigh_gy_std),
            self.normalize(shank_gyL_raw, self.shank_gy_mean, self.shank_gy_std),
            self.normalize(self.knee_l_filt, self.knee_mean, self.knee_std)], dtype=np.float32)
        else:
            raise ValueError("Unsupported input size")



        seq_r = self.x_r.push_last(xr_last)
        seq_l = self.x_l.push_last(xl_last)

        # Motion gating always uses the two gyro channels only,
        # regardless of whether the model input also includes knee angle.
        gate_seq_r = seq_r[:2, :]
        gate_seq_l = seq_l[:2, :]
  

        x_r = seq_r.reshape(1, self.input_size, self.T).astype(np.float32, copy=False)
        x_l = seq_l.reshape(1, self.input_size, self.T).astype(np.float32, copy=False)

        # ---- TRT inference (latest only) ----
        new_result = self._get_latest_inference()
        if new_result is not None:
            self.last_out_r = new_result[0].copy()
            self.last_out_l = new_result[1].copy()
        self._try_put_latest({"r": x_r, "l": x_l})

        y_r = self.last_out_r.copy()
        y_l = self.last_out_l.copy()

        K_r_raw = self._map_K(y_r[0])
        K_l_raw = self._map_K(y_l[0])

        gait_r_cmd = float(np.clip(y_r[1], -1.0, 1.0))
        gait_l_cmd = float(np.clip(y_l[1], -1.0, 1.0))

        gait_r_cmd = self.gait_r_raw_prev + np.clip(gait_r_cmd-self.gait_r_raw_prev, -self.rate_max*self.dt, self.rate_max*self.dt)
        gait_l_cmd = self.gait_l_raw_prev + np.clip(gait_l_cmd-self.gait_l_raw_prev, -self.rate_max*self.dt, self.rate_max*self.dt)

        self.K_r_raw = K_r_raw
        self.K_l_raw = K_l_raw

        self.K_r_filt = self._lpf(self.K_r_filt, K_r_raw, self.impedance_filter_tau)
        self.K_l_filt = self._lpf(self.K_l_filt, K_l_raw, self.impedance_filter_tau)
        self.gait_r_filt = self._lpf(self.gait_r_filt, gait_r_cmd, self.gait_filter_tau)
        self.gait_l_filt = self._lpf(self.gait_l_filt, gait_l_cmd, self.gait_filter_tau)

        self.gait_r_raw_prev = gait_r_cmd
        self.gait_l_raw_prev = gait_l_cmd


        knee_r_u_gyr = -self.thigh_gyR +self.shank_gyR
        knee_l_u_gyr = -self.thigh_gyL+ self.shank_gyL


        """
        --------------------
        Motion gating logic
        --------------------
        """

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

        if self.motion_state_l == "idle":
            state_l = 0
        elif self.motion_state_l == "starting":
            state_l = 1
        else:
            state_l = 2

        if self.motion_state_r == "idle":
            state_r = 0
        elif self.motion_state_r == "starting":
            state_r = 1
        else:
            state_r = 2

        K_r_ctrl = self.K_r_filt * self.assist_gate_r
        K_l_ctrl = self.K_l_filt * self.assist_gate_l

        tau_r = self._torque_one_leg(self.knee_r_filt, knee_r_u_gyr, K_r_ctrl, self.gait_r_filt)
        tau_l = self._torque_one_leg(self.knee_l_filt, knee_l_u_gyr, K_l_ctrl, self.gait_l_filt)

        # if self.motion_state_r == "idle" and self.assist_gate_r < 0.03:
        #     tau_r = 0.0
        # if self.motion_state_l == "idle" and self.assist_gate_l < 0.03:
        #     tau_l = 0.0

        

        seq_vt_rot_r = self.vt_rot_r.push_last(thigh_vert_rot_r_raw)
        seq_vt_rot_l = self.vt_rot_l.push_last(thigh_vert_rot_l_raw)
        if np.mean(seq_vt_rot_l)<=1.5:
            turn_coeff_l = 1
        elif np.mean(seq_vt_rot_l)>1.5 and np.mean(seq_vt_rot_l)<=3:
            turn_coeff_l = -(1/1.5)*(np.mean(seq_vt_rot_l)-3)
        else:
            turn_coeff_l = 0.0
        
        if np.mean(seq_vt_rot_r)<=1.5:
            turn_coeff_r = 1
        elif np.mean(seq_vt_rot_r)>1.5 and np.mean(seq_vt_rot_r)<=3:
            turn_coeff_r = -(1/1.5)*(np.mean(seq_vt_rot_r)-3)
        else:
            turn_coeff_r = 0.0

        tau_r *=turn_coeff_r
        tau_l *=turn_coeff_l



        # dec_st_angle = 10
        # dec_ed_angle = 0
        # if np.abs(self.knee_r_filt) < np.deg2rad(dec_st_angle) and np.abs(self.knee_r_filt) >= np.deg2rad(dec_ed_angle):
        #     decay_multiplier = self._smoothstep(np.abs(self.knee_r_filt), np.deg2rad(dec_ed_angle), np.deg2rad(dec_st_angle))
        #     tau_r *= decay_multiplier 
        # if np.abs(self.knee_l_filt) < np.deg2rad(dec_st_angle) and np.abs(self.knee_l_filt) >= np.deg2rad(dec_ed_angle):
        #     decay_multiplier = self._smoothstep(np.abs(self.knee_l_filt), np.deg2rad(dec_ed_angle), np.deg2rad(dec_st_angle))
        #     tau_l *= decay_multiplier

        tau_r = self._rate_limit(tau_r, self.prev_cmd_r, self.cmd_rate_max)
        tau_l = self._rate_limit(tau_l, self.prev_cmd_l, self.cmd_rate_max)

        self.prev_cmd_r = tau_r
        self.prev_cmd_l = tau_l


        return CtrlResult(
            model_out_R=float(tau_r), model_out_L=float(tau_l),
            applied_R=float(tau_r), applied_L=float(tau_l),
            extra={
                "knee_angle_r": float(self.knee_r_filt),
                "knee_angle_l": float(self.knee_l_filt),
                "knee_angle_r_u": float(self.knee_r_u_filt),
                "knee_angle_l_u": float(self.knee_l_u_filt),
                "knee_r_u_gyr": float(knee_r_u_gyr),
                "knee_l_u_gyr": float(knee_l_u_gyr),
                "K_r": float(self.K_r_filt),
                "K_l": float(self.K_l_filt),
                "Soft_ctrl_r": float(self.gait_r_filt),
                "Soft_ctrl_l": float(self.gait_l_filt),
                "assist_gate_r": float(self.assist_gate_r),
                "assist_gate_l": float(self.assist_gate_l),
                "state_l": int(state_l),
                "state_r": int(state_r),
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

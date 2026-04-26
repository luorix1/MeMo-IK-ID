# controllers/biotorque_hip.py
"""Hip biotorque controller.

Pipeline (mirrors the original allnewK5 controller):
  1. Mirror left-side IMU/motor data to create a right-referenced left input
  2. Normalize inputs with per-channel mean/std from model directory
  3. Append to rolling input windows
  4. Send windows to HipTRTWorker (async); use last result if not yet available
  5. net_torque  = model_output  * body_mass_kg
  6. bio_torque  = net_torque    - current_applied_torque  (feedback)
  7. scaled_torque = bio_torque  * scale_factor
  8. delayed_torque: look back delay_steps frames in the scaled buffer
  9. filtered_torque: realtime Butterworth (sosfilt, stateful)
 10. Clamp to ±torque_limit_Nm
 11. Return CtrlResult with all intermediate signals in `extra`
"""

import multiprocessing as mp
import os
from collections import deque
from queue import Empty, Full

import numpy as np
from scipy import signal as sp_signal

from .base import BaseController, CtrlResult, RollingWindow, Sensors
from .trt_worker_hip import HipTRTWorker


class HipBiotorque(BaseController):
    name = "hip_biotorque"

    def __init__(self, config: dict):
        self.fs = int(config["fs"])
        self.dt = 1.0 / self.fs
        self.T = int(config["frame_length"])

        self.mass = float(config["mass"])
        self.scale_factor = float(config["scale_factor"])
        self.torque_limit = float(config["torque_limit"])

        # Delay: desired_delay_ms → delay_steps using the same formula as original.
        # delay_factor = int(desired_delay_ms / 10 - 4) corresponds to looking back
        # delay_factor+1 frames in the scaled-torque window.
        desired_delay_ms = float(config["desired_delay_ms"])
        self.delay_steps = max(0, int(desired_delay_ms / 10 - 4))

        # Lowpass filter (Butterworth, realtime SOS)
        lpf_cutoff = float(config.get("lpf_cutoff_Hz", 10.0))
        lpf_order = int(config.get("lpf_order", 2))
        self._sos = sp_signal.butter(lpf_order, lpf_cutoff, btype="low",
                                     fs=self.fs, output="sos")
        self._zi_R = None   # filter state, initialized on first sample
        self._zi_L = None

        # Load input normalization stats (npy files next to the engine)
        engine_path = config["trt_engine_path"]
        model_dir = os.path.dirname(engine_path)
        self._input_mean = np.load(os.path.join(model_dir, "input_mean.npy")).astype(np.float32)
        self._input_std = np.load(os.path.join(model_dir, "input_std.npy")).astype(np.float32)
        label_mean_path = os.path.join(model_dir, "label_mean.npy")
        label_std_path = os.path.join(model_dir, "label_std.npy")
        self._num_features = self._input_mean.shape[0]

        # Rolling input windows: shape (1, C, T) per side
        self._win_R = RollingWindow((1, self._num_features, self.T))
        self._win_L = RollingWindow((1, self._num_features, self.T))

        # Rolling torque buffers (delay + filter)
        buf_len = self.delay_steps + 2   # +2 to safely index delay_steps+1 back
        self._scaled_buf_R = deque([0.0] * buf_len, maxlen=buf_len)
        self._scaled_buf_L = deque([0.0] * buf_len, maxlen=buf_len)

        self._last_out_R = np.zeros((1,), dtype=np.float32)
        self._last_out_L = np.zeros((1,), dtype=np.float32)
        self._applied_R = 0.0
        self._applied_L = 0.0

        # Inference worker
        self._in_q: mp.Queue = mp.Queue(maxsize=1)
        self._out_q: mp.Queue = mp.Queue(maxsize=1)
        self._worker = HipTRTWorker(
            in_q=self._in_q,
            out_q=self._out_q,
            engine_path=engine_path,
            label_mean_path=label_mean_path,
            label_std_path=label_std_path,
            single_in_shape=(1, self._num_features, self.T),
            single_out_shape=(1,),
        )
        self._worker.daemon = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        self._worker.start()

    def close(self):
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        try:
            self._worker.join(timeout=2.0)
        except Exception:
            pass
        for q in (self._out_q, self._in_q):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._input_mean) / self._input_std

    def _get_latest_inference(self):
        latest = None
        try:
            while True:
                latest = self._out_q.get_nowait()
        except Empty:
            pass
        return latest

    def _try_put_input(self, payload):
        try:
            self._in_q.put_nowait(payload)
        except Full:
            pass

    def _lpf_step(self, value: float, zi, init_val: float):
        """Single-sample realtime Butterworth filter step.

        Returns (filtered_value, new_zi).  On first call pass zi=None.
        """
        x = np.array([value], dtype=np.float64)
        if zi is None:
            zi = sp_signal.sosfilt_zi(self._sos) * init_val
        y, zi_new = sp_signal.sosfilt(self._sos, x, zi=zi)
        return float(y[0]), zi_new

    def _delayed_value(self, buf: deque) -> float:
        """Return the value from delay_steps steps ago."""
        # buf[-1] = just pushed (current), buf[-(delay_steps+1)] = delay_steps ago
        idx = -(self.delay_steps + 1)
        try:
            return float(buf[idx])
        except IndexError:
            return 0.0

    # ------------------------------------------------------------------
    # Control step
    # ------------------------------------------------------------------
    def step(self, s: Sensors) -> CtrlResult:
        # ── 1. Mirror left IMU / motor for right-referenced left inference ──
        p_reflected = s.imu_P.copy()
        p_reflected[1] *= -1    # acc_y
        p_reflected[3] *= -1    # gyr_x
        p_reflected[5] *= -1    # gyr_z

        l_reflected = s.imu_L.copy()
        l_reflected[1] *= -1
        l_reflected[3] *= -1
        l_reflected[5] *= -1

        # ── 2. Build per-side feature vectors and normalize ──
        # Original uses only the 6-ch IMU (pelvis not concatenated in IMU-only mode).
        # If you want to add pos/vel features, extend right_feat / left_feat here.
        right_feat = self._normalize(s.imu_R.astype(np.float32))
        left_feat = self._normalize(l_reflected.astype(np.float32))

        # ── 3. Roll windows ──
        x_R = self._win_R.push_last(right_feat).copy()   # (1, C, T)
        x_L = self._win_L.push_last(left_feat).copy()    # (1, C, T)

        # ── 4. Async inference ──
        self._try_put_input({"r": x_R, "l": x_L})
        result = self._get_latest_inference()
        if result is not None:
            self._last_out_R, self._last_out_L = result

        model_out_R = float(self._last_out_R[0])   # Nm/kg
        model_out_L = float(self._last_out_L[0])   # Nm/kg

        # ── 5. Net torque ──
        net_R = model_out_R * self.mass
        net_L = model_out_L * self.mass

        # ── 6. Bio torque (feedback: subtract currently applied torque) ──
        bio_R = net_R - self._applied_R
        bio_L = net_L - self._applied_L

        # ── 7. Scale ──
        scaled_R = bio_R * self.scale_factor
        scaled_L = bio_L * self.scale_factor
        self._scaled_buf_R.append(scaled_R)
        self._scaled_buf_L.append(scaled_L)

        # ── 8. Delay ──
        delayed_R = self._delayed_value(self._scaled_buf_R)
        delayed_L = self._delayed_value(self._scaled_buf_L)

        # ── 9. Realtime Butterworth filter ──
        filtered_R, self._zi_R = self._lpf_step(delayed_R, self._zi_R, delayed_R)
        filtered_L, self._zi_L = self._lpf_step(delayed_L, self._zi_L, delayed_L)

        # Track what we're about to apply (for next cycle's feedback)
        self._applied_R = filtered_R
        self._applied_L = filtered_L

        # ── 10. Clamp ──
        cmd_R = float(np.clip(filtered_R, -self.torque_limit, self.torque_limit))
        cmd_L = float(np.clip(filtered_L, -self.torque_limit, self.torque_limit))

        return CtrlResult(
            model_out_R=model_out_R,
            model_out_L=model_out_L,
            applied_R=cmd_R,
            applied_L=cmd_L,
            extra={
                "net_R": net_R,
                "net_L": net_L,
                "bio_R": bio_R,
                "bio_L": bio_L,
                "scaled_R": scaled_R,
                "scaled_L": scaled_L,
                "delayed_R": delayed_R,
                "delayed_L": delayed_L,
                "filtered_R": filtered_R,
                "filtered_L": filtered_L,
            },
        )

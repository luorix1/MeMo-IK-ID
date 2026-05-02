import multiprocessing as mp
from queue import Empty

import numpy as np

from .base import BaseController, CtrlResult, RollingWindow, Sensors
from .trt_worker import TRTWorker


class simgyro3(BaseController):
    name = "simgyro3"

    def __init__(self, engine_path: str, frame_len: int, mean_std_path: str, fs: int = 100):
        self.engine_path = engine_path
        self.T = int(frame_len)
        stats = np.load(mean_std_path)
        self.in_mean = stats["mean"]
        self.in_std = stats["std"]
        self.fs = fs
        self.scale_factor = 15
        self.in_shape = (1, 3, self.T)
        self.out_shape = (1,)

        self.x_r = RollingWindow((3, self.T))
        self.x_l = RollingWindow((3, self.T))

        self.y_last = np.zeros(2, dtype=np.float32)
        self.alpha = min(1.0, max(0.0, (1.0 / self.fs) / 0.2))

        self.in_q_r, self.out_q_r = mp.Queue(maxsize=1), mp.Queue(maxsize=1)
        self.in_q_l, self.out_q_l = mp.Queue(maxsize=1), mp.Queue(maxsize=1)
        self.worker_r = TRTWorker(self.in_q_r, self.out_q_r, self.engine_path, self.in_shape, self.out_shape)
        self.worker_l = TRTWorker(self.in_q_l, self.out_q_l, self.engine_path, self.in_shape, self.out_shape)
        self.worker_r.daemon = True
        self.worker_l.daemon = True

        self.last_out_r = 0.0
        self.last_out_l = 0.0

    def start(self):
        self.worker_r.start()
        self.worker_l.start()

    def _norm(self, v):
        norm_data = np.zeros_like(v, dtype=np.float32)
        for i in range(0, 3):
            norm_data[i] = (v[i] - self.in_mean[i]) / self.in_std[i]
        return norm_data

    def _drain(self, q):
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass

    def step(self, s: Sensors) -> CtrlResult:
        gyR = np.deg2rad(s.imu_R[3:6])
        gyL = np.deg2rad(s.imu_L[3:6])

        gyR_rot = np.array([-gyR[2], gyR[0], -gyR[1]], dtype=np.float32)
        gyL_rot = np.array([gyL[2], -gyL[0], -gyL[1]], dtype=np.float32)
        xr_last = np.array([self._norm(gyR_rot)], dtype=np.float32)
        xl_last = np.array([self._norm(gyL_rot)], dtype=np.float32)

        seq_r = self.x_r.push_last(xr_last)
        seq_l = self.x_l.push_last(xl_last)

        self._drain(self.out_q_r)
        try:
            self.in_q_r.put_nowait(seq_r.reshape(1, 3, self.T))
        except mp.queues.Full:
            pass

        y_r = self.last_out_r
        try:
            y_r = float(self.out_q_r.get(timeout=0.02)[0])
            self.last_out_r = y_r
        except Empty:
            pass

        self._drain(self.out_q_l)
        try:
            self.in_q_l.put_nowait(seq_l.reshape(1, 3, self.T))
        except mp.queues.Full:
            pass

        y_l = self.last_out_l
        try:
            y_l = float(self.out_q_l.get(timeout=0.02)[0])
            self.last_out_l = y_l
        except Empty:
            pass

        y_raw = np.array([y_r, y_l], dtype=np.float32)
        self.y_last = self.y_last + self.alpha * (y_raw - self.y_last)
        y_applied = self.y_last * self.scale_factor

        return CtrlResult(
            model_out_R=y_r,
            model_out_L=y_l,
            applied_R=float(y_applied[0]),
            applied_L=float(y_applied[1]),
            extra={},
        )

    def close(self):
        for in_q, out_q, w in [
            (self.in_q_r, self.out_q_r, self.worker_r),
            (self.in_q_l, self.out_q_l, self.worker_l),
        ]:
            try:
                in_q.put_nowait(None)
            except Exception:
                pass
            try:
                w.join(timeout=1.5)
            except Exception:
                pass
            for q in (out_q, in_q):
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass

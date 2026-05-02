from collections import deque
from queue import Empty

import multiprocessing as mp
import numpy as np

from .base import BaseController, CtrlResult, RollingWindow, Sensors
from .trt_worker import TRTWorker


class biotorque(BaseController):
    name = "biotorque"

    def __init__(
        self,
        engine_path: str,
        n_of_channel: int,
        frame_len: int,
        input_mean_path: str,
        input_std_path: str,
        out_mean_path: str,
        out_std_path: str,
        body_mass: float,
        scale_factor: float = 0.2,
        delay: float = 0.1,
        fs: int = 100,
    ):
        self.engine_path = engine_path
        self.T = int(frame_len)
        self.n_of_channel = n_of_channel
        self.in_mean = np.load(input_mean_path)
        self.in_std = np.load(input_std_path)
        self.out_mean = np.load(out_mean_path)
        self.out_std = np.load(out_std_path)
        self.fs = fs
        self.body_mass = body_mass
        self.in_shape = (1, self.n_of_channel, self.T)
        self.out_shape = (1,)
        self.scale_factor = scale_factor

        self.x_r = RollingWindow((self.n_of_channel, self.T))
        self.x_l = RollingWindow((self.n_of_channel, self.T))

        self.y_last = np.zeros(2, dtype=np.float32)
        self.alpha = min(1.0, max(0.0, (1.0 / self.fs) / 0.1))

        self.in_q_r, self.out_q_r = mp.Queue(maxsize=1), mp.Queue(maxsize=1)
        self.in_q_l, self.out_q_l = mp.Queue(maxsize=1), mp.Queue(maxsize=1)
        self.worker_r = TRTWorker(self.in_q_r, self.out_q_r, self.engine_path, self.in_shape, self.out_shape)
        self.worker_l = TRTWorker(self.in_q_l, self.out_q_l, self.engine_path, self.in_shape, self.out_shape)
        self.worker_r.daemon = True
        self.worker_l.daemon = True

        self.last_out_r = 0.0
        self.last_out_l = 0.0

        self.delay = float(delay)
        self.delay_samples = max(0, int(round(self.delay * self.fs)))
        self.yr_buffer = deque([0.0] * self.delay_samples, maxlen=self.delay_samples or 1)
        self.yl_buffer = deque([0.0] * self.delay_samples, maxlen=self.delay_samples or 1)

    def start(self):
        self.worker_r.start()
        self.worker_l.start()
        self.yr_buffer.clear()
        self.yl_buffer.clear()
        if self.delay_samples > 0:
            for _ in range(self.delay_samples):
                self.yr_buffer.append(0.0)
                self.yl_buffer.append(0.0)
        else:
            self.yr_buffer.append(0.0)
            self.yl_buffer.append(0.0)

    def input_norm(self, input_data):
        return (input_data - self.in_mean) / self.in_std

    def output_norm(self, output_data):
        output_data = output_data * self.out_std + self.out_mean
        return output_data[0]

    def _drain(self, q):
        try:
            while True:
                q.get_nowait()
        except Empty:
            pass

    def step(self, s: Sensors) -> CtrlResult:
        imu_r = np.asarray(s.imu_R, dtype=np.float32).copy()
        imu_l = np.asarray(s.imu_L, dtype=np.float32).copy()
        imu_l[1] = -imu_l[1]
        imu_l[3] = -imu_l[3]
        imu_l[5] = -imu_l[5]

        xr_last = self.input_norm(imu_r).astype(np.float32)
        xl_last = self.input_norm(imu_l).astype(np.float32)

        seq_r = self.x_r.push_last(xr_last)
        seq_l = self.x_l.push_last(xl_last)

        self._drain(self.out_q_r)
        try:
            self.in_q_r.put_nowait(seq_r.reshape(1, self.n_of_channel, self.T))
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
            self.in_q_l.put_nowait(seq_l.reshape(1, self.n_of_channel, self.T))
        except mp.queues.Full:
            pass

        y_l = self.last_out_l
        try:
            y_l = float(self.out_q_l.get(timeout=0.02)[0])
            self.last_out_l = y_l
        except Empty:
            pass

        y_r = self.output_norm(y_r)
        y_l = self.output_norm(y_l)

        y_raw = np.array([y_r, y_l], dtype=np.float32)
        self.y_last = self.y_last + self.alpha * (y_raw - self.y_last)
        y_applied = self.y_last * self.body_mass * self.scale_factor

        if self.delay_samples > 0:
            yr_delayed = self.yr_buffer.popleft()
            yl_delayed = self.yl_buffer.popleft()
            self.yr_buffer.append(float(self.y_last[0] * self.body_mass * self.scale_factor))
            self.yl_buffer.append(float(self.y_last[1] * self.body_mass * self.scale_factor))
        else:
            yr_delayed = float(self.y_last[0] * self.body_mass * self.scale_factor)
            yl_delayed = float(self.y_last[1] * self.body_mass * self.scale_factor)

        return CtrlResult(
            model_out_R=y_r,
            model_out_L=y_l,
            applied_R=float(-yr_delayed),
            applied_L=float(-yl_delayed),
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

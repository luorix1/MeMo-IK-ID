import numpy as np, multiprocessing as mp
from .base import BaseController, Sensors, CtrlResult, RollingWindow
from .trt_worker import TRTWorker
from queue import Empty

class Gyro1ChTRT(BaseController):
    name = "gyro1ch_trt"

    def __init__(self, engine_path:str, frame_len:int, input_mean:float, input_std:float, fs:int=100):
        self.engine_path = engine_path
        self.T = int(frame_len)
        self.in_mean = float(input_mean)
        self.in_std  = float(input_std)
        self.fs = fs
        self.scale_factor = 10
        self.in_shape  = (1, 1, self.T)
        self.out_shape = (1,)

        self.x_r = RollingWindow((1, self.T))
        self.x_l = RollingWindow((1, self.T))

        self.y_last = np.zeros(2, dtype=np.float32)
        self.alpha = min(1.0, max(0.0, (1.0/self.fs)/0.15))

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
        return (v - self.in_mean) / (self.in_std if self.in_std != 0 else 1.0)

    def _drain(self, q):
        try:
            while True: q.get_nowait()
        except Empty:
            pass

    def step(self, s: Sensors) -> CtrlResult:
        gyR = -np.deg2rad(float(s.imu_R[4]))
        gyL = -np.deg2rad(float(s.imu_L[4]))

        xr_last = np.array([self._norm(gyR)], dtype=np.float32)
        xl_last = np.array([self._norm(gyL)], dtype=np.float32)

        seq_r = self.x_r.push_last(xr_last)  # (1,T)
        seq_l = self.x_l.push_last(xl_last)  # (1,T)

        # --- Right ---
        self._drain(self.out_q_r)  # 최신 1개만 유지
        try: self.in_q_r.put_nowait(seq_r.reshape(1,1,self.T))
        except mp.queues.Full: pass

        y_r = self.last_out_r
        try:
            y_r = float(self.out_q_r.get(timeout=0.02)[0])  # 블록으로 정확히 매칭
            self.last_out_r = y_r
        except Empty:
            pass

        # --- Left ---
        self._drain(self.out_q_l)
        try: self.in_q_l.put_nowait(seq_l.reshape(1,1,self.T))
        except mp.queues.Full: pass

        y_l = self.last_out_l
        try:
            y_l = float(self.out_q_l.get(timeout=0.02)[0])
            self.last_out_l = y_l
        except Empty:
            pass

        # LPF
        y_raw = np.array([y_r, y_l], dtype=np.float32)
        self.y_last = self.y_last + self.alpha * (y_raw - self.y_last)
        y_applied = self.y_last*self.scale_factor

        return CtrlResult(
            model_out_R=y_r, model_out_L=y_l,
            applied_R=float(y_applied[0]), applied_L=float(y_applied[1]),
            extra={}
        )

    def close(self):
        for in_q, out_q, w in [(self.in_q_r, self.out_q_r, self.worker_r),
                               (self.in_q_l, self.out_q_l, self.worker_l)]:
            try: in_q.put_nowait(None)
            except: pass
            try: w.join(timeout=1.5)
            except: pass
            for q in (out_q, in_q):
                try:
                    q.close(); q.join_thread()
                except: pass

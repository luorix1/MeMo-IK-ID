import multiprocessing as mp

import numpy as np
import tensorrt as trt
import torch


class TRTWorker(mp.Process):
    def __init__(self, in_q, out_q, engine_path, in_shape, out_shape):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.engine_path = engine_path
        self.in_shape = in_shape
        self.out_shape = out_shape

    def run(self):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            print("[TRTWorker] Engine load failed")
            return
        context = engine.create_execution_context()

        d_in = torch.empty(self.in_shape, dtype=torch.float32, device="cuda")
        d_out = torch.empty(self.out_shape, dtype=torch.float32, device="cuda")
        bindings = [int(d_in.data_ptr()), int(d_out.data_ptr())]

        d_in.zero_()
        for _ in range(10):
            context.execute_v2(bindings=bindings)

        last = np.zeros(self.out_shape, dtype=np.float32)
        while True:
            data = self.in_q.get()
            if data is None:
                break
            d_in.copy_(torch.from_numpy(data))
            context.execute_v2(bindings=bindings)
            y = d_out.detach().cpu().numpy()
            last = y
            try:
                while True:
                    self.out_q.get_nowait()
            except mp.queues.Empty:
                pass
            try:
                self.out_q.put_nowait(last)
            except mp.queues.Full:
                pass

        del context, engine, runtime
        print("[TRTWorker] exit")

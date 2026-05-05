import multiprocessing as mp
from queue import Empty, Full

import numpy as np
import tensorrt as trt
import torch


class TRTWorkerUni(mp.Process):
    """
    Unilateral TRT worker. Controller sends:
        x: np.ndarray of shape (1, C, T)   [batch=1, channels, time]

    Worker runs ONE TensorRT inference and returns:
        y: np.ndarray of shape (O,)

    The TRT engine is expected to have a static input shape of (1, C, T)
    and output shape of (1, O).
    """

    def __init__(self, in_q, out_q, engine_path, in_shape, out_shape):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.engine_path = engine_path

        self.in_shape = tuple(in_shape)  # (B, C, T)
        self.out_shape = tuple(out_shape)  # (O,)
        self._batch = self.in_shape[0]

        if len(self.in_shape) != 3:
            raise ValueError(f"in_shape must be (B, C, T), got {self.in_shape}")

        self._trt_out_shape = (self._batch, *self.out_shape)

    def _find_io_names(self, engine):
        if not hasattr(engine, "num_io_tensors"):
            return None, None

        in_name, out_name = None, None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                in_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                out_name = name

        return in_name, out_name

    def _get_latest_input(self):
        data = self.in_q.get()
        if data is None:
            return None

        while True:
            try:
                newer = self.in_q.get_nowait()
                if newer is None:
                    return None
                data = newer
            except Empty:
                break
        return data

    def _put_latest_output(self, item):
        try:
            while True:
                self.out_q.get_nowait()
        except Empty:
            pass

        try:
            self.out_q.put_nowait(item)
        except Full:
            pass

    def run(self):
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            print("[TRTWorkerUni] Engine load failed!")
            return

        context = engine.create_execution_context()
        stream = torch.cuda.Stream()

        d_in = torch.empty(self.in_shape, dtype=torch.float32, device="cuda")
        d_out = torch.empty(self._trt_out_shape, dtype=torch.float32, device="cuda")

        use_tensor_api = hasattr(engine, "num_io_tensors") and hasattr(engine, "get_tensor_name")
        bindings = None

        if use_tensor_api:
            in_name, out_name = self._find_io_names(engine)
            if in_name is None or out_name is None:
                raise RuntimeError("[TRTWorkerUni] Failed to find TRT input/output tensor names")

            ok = context.set_input_shape(in_name, self.in_shape)
            if ok is False:
                raise RuntimeError(f"[TRTWorkerUni] TRT refused input shape {self.in_shape}")

            context.set_tensor_address(in_name, int(d_in.data_ptr()))
            context.set_tensor_address(out_name, int(d_out.data_ptr()))
        else:
            bindings = [int(d_in.data_ptr()), int(d_out.data_ptr())]

        d_in.zero_()
        for _ in range(5):
            if use_tensor_api:
                context.execute_async_v3(stream_handle=stream.cuda_stream)
            else:
                context.execute_v2(bindings=bindings)
        stream.synchronize()

        print("[TRTWorkerUni] Ready.")

        while True:
            data = self._get_latest_input()
            if data is None:
                break

            x = np.asarray(data, dtype=np.float32)

            if x.shape != self.in_shape:
                print(f"[TRTWorkerUni] Bad input shape: {x.shape}, expected {self.in_shape}")
                continue

            x = np.ascontiguousarray(x)
            d_in.copy_(torch.from_numpy(x), non_blocking=True)

            if use_tensor_api:
                context.execute_async_v3(stream_handle=stream.cuda_stream)
            else:
                context.execute_v2(bindings=bindings)

            stream.synchronize()

            y_batch = d_out.detach().cpu().numpy().copy()

            self._put_latest_output(y_batch)

        print("[TRTWorkerUni] Exiting...")

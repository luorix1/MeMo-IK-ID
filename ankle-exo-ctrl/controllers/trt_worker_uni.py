import multiprocessing as mp
from queue import Empty, Full

import numpy as np


class TRTWorkerUni(mp.Process):
    """
    Unilateral TRT worker. Controller sends:
        x: np.ndarray of shape (1, C, T)   [batch=1, channels, time]

    Worker runs ONE TensorRT inference and returns:
        y: np.ndarray of shape (O,)

    The TRT engine is expected to have a static input shape of (1, C, T)
    and output shape of (1, O).

    Uses TensorRT + libcudart (ctypes). Does **not** use torch.cuda, so a
    mismatched pip PyTorch CUDA wheel on Jetson will not break inference.
    ``tensorrt`` / cudart are loaded inside ``run()``.
    """

    def __init__(self, in_q, out_q, engine_path, in_shape, out_shape):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.engine_path = engine_path

        self.in_shape = tuple(in_shape)  # (1, C, T)
        self.out_shape = tuple(out_shape)  # (O,)

        if len(self.in_shape) != 3 or self.in_shape[0] != 1:
            raise ValueError(f"in_shape must be (1, C, T), got {self.in_shape}")

        self._trt_out_shape = (1, *self.out_shape)

    def _find_io_names(self, engine, trt):
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
        import os
        import sys

        # Project root must be importable in the child process.
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if root not in sys.path:
            sys.path.insert(0, root)

        import tensorrt as trt

        from utils.cudart_runtime import CudaRuntime

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            print("[TRTWorkerUni] Engine load failed!")
            return

        mem = CudaRuntime()
        context = engine.create_execution_context()

        h_in = np.zeros(self.in_shape, dtype=np.float32)
        h_out = np.empty(self._trt_out_shape, dtype=np.float32)
        d_in = mem.malloc(h_in.nbytes)
        d_out = mem.malloc(h_out.nbytes)

        use_tensor_api = hasattr(engine, "num_io_tensors") and hasattr(
            engine, "get_tensor_name"
        )
        bindings = None

        if use_tensor_api:
            in_name, out_name = self._find_io_names(engine, trt)
            if in_name is None or out_name is None:
                raise RuntimeError(
                    "[TRTWorkerUni] Failed to find TRT input/output tensor names"
                )

            ok = context.set_input_shape(in_name, self.in_shape)
            if ok is False:
                raise RuntimeError(
                    f"[TRTWorkerUni] TRT refused input shape {self.in_shape}"
                )

            context.set_tensor_address(in_name, d_in)
            context.set_tensor_address(out_name, d_out)
        else:
            bindings = [d_in, d_out]

        mem.memset(d_in, 0, h_in.nbytes)
        for _ in range(5):
            if use_tensor_api:
                context.execute_async_v3(stream_handle=mem.stream_handle())
            else:
                context.execute_v2(bindings=bindings)
        mem.sync()

        print(f"[TRTWorkerUni] Ready. (CUDA={mem.backend}, no torch.cuda)")

        while True:
            data = self._get_latest_input()
            if data is None:
                break

            x = np.asarray(data, dtype=np.float32)

            if x.shape != self.in_shape:
                print(
                    f"[TRTWorkerUni] Bad input shape: {x.shape}, expected {self.in_shape}"
                )
                continue

            x = np.ascontiguousarray(x)
            mem.h2d(x, d_in)

            if use_tensor_api:
                context.execute_async_v3(stream_handle=mem.stream_handle())
            else:
                context.execute_v2(bindings=bindings)

            mem.d2h(d_out, h_out)
            mem.sync()

            y = h_out[0].reshape(self.out_shape).copy()
            self._put_latest_output(y)

        try:
            mem.free(d_in)
            mem.free(d_out)
        except Exception:
            pass

        print("[TRTWorkerUni] Exiting...")

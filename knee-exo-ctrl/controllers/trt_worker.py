from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full
from typing import Tuple

import numpy as np


class TRTWorker(mp.Process):
    """
    TensorRT inference worker for dual-leg batched execution.

    Controller sends:
        {"r": x_r, "l": x_l}
    where
        x_r.shape == x_l.shape == single_in_shape == (1, C, T)

    Worker stacks into one batch:
        x_batch.shape == (2, C, T)

    Returns:
        (y_r, y_l)
    where each y_* has shape single_out_shape.
    """

    def __init__(
        self,
        in_q: mp.Queue,
        out_q: mp.Queue,
        engine_path: str,
        single_in_shape: Tuple[int, int, int],
        single_out_shape: Tuple[int, ...],
    ):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.engine_path = engine_path
        self.single_in_shape = tuple(single_in_shape)      # (1, C, T)
        self.single_out_shape = tuple(single_out_shape)    # e.g. (1,) or (1,T)

        if len(self.single_in_shape) != 3 or self.single_in_shape[0] != 1:
            raise ValueError(f"single_in_shape must be (1, C, T), got {self.single_in_shape}")

        self.batch_in_shape = (2, self.single_in_shape[1], self.single_in_shape[2])
        self.batch_out_shape = (2, *self.single_out_shape)

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
        # Keep only the latest pending input to avoid queue latency buildup.
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

    def _run_single(self, x_single: np.ndarray, context, stream, d_in, d_out, use_tensor_api: bool, bindings):
        """Execute one single-leg inference and return output array."""
        import torch

        x_single = np.ascontiguousarray(x_single, dtype=np.float32)
        d_in.copy_(torch.from_numpy(x_single), non_blocking=True)
        if use_tensor_api:
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        else:
            context.execute_v2(bindings=bindings)
        stream.synchronize()
        return d_out.detach().cpu().numpy().copy()

    def run(self):
        import torch
        import tensorrt as trt

        try:
            torch.cuda.set_device(0)
        except Exception:
            pass

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            print("[TRTWorker] Engine load failed")
            return

        context = engine.create_execution_context()
        stream = torch.cuda.Stream()
        use_tensor_api = hasattr(engine, "num_io_tensors") and hasattr(engine, "get_tensor_name")
        bindings = None
        use_batch2 = True
        d_in = None
        d_out = None

        if use_tensor_api:
            in_name, out_name = self._find_io_names(engine, trt)
            if in_name is None or out_name is None:
                raise RuntimeError("Failed to find TRT I/O tensor names")

            ok = context.set_input_shape(in_name, self.batch_in_shape)
            if ok is False:
                ok1 = context.set_input_shape(in_name, self.single_in_shape)
                if ok1 is False:
                    raise RuntimeError(
                        f"TRT refused both shapes: batch={self.batch_in_shape}, single={self.single_in_shape}"
                    )
                use_batch2 = False
                print(f"[TRTWorker] Engine is single-batch ({self.single_in_shape}); using sequential per-leg inference.")
            else:
                use_batch2 = True

            in_shape = self.batch_in_shape if use_batch2 else self.single_in_shape
            out_shape = self.batch_out_shape if use_batch2 else self.single_out_shape
            d_in = torch.empty(in_shape, dtype=torch.float32, device="cuda")
            d_out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
            context.set_tensor_address(in_name, int(d_in.data_ptr()))
            context.set_tensor_address(out_name, int(d_out.data_ptr()))
        else:
            use_batch2 = False
            print("[TRTWorker] Legacy TRT API; using sequential per-leg inference.")
            d_in = torch.empty(self.single_in_shape, dtype=torch.float32, device="cuda")
            d_out = torch.empty(self.single_out_shape, dtype=torch.float32, device="cuda")
            bindings = [int(d_in.data_ptr()), int(d_out.data_ptr())]

        # Warmup
        d_in.zero_()
        for _ in range(5):
            if use_tensor_api:
                context.execute_async_v3(stream_handle=stream.cuda_stream)
            else:
                context.execute_v2(bindings=bindings)
        stream.synchronize()

        print("[TRTWorker] Ready")

        while True:
            data = self._get_latest_input()
            if data is None:
                break
            if not isinstance(data, dict) or "r" not in data or "l" not in data:
                continue

            x_r = np.asarray(data["r"], dtype=np.float32)
            x_l = np.asarray(data["l"], dtype=np.float32)
            if x_r.shape != self.single_in_shape or x_l.shape != self.single_in_shape:
                continue

            if use_batch2:
                x_batch = np.concatenate([x_r, x_l], axis=0)
                x_batch = np.ascontiguousarray(x_batch, dtype=np.float32)
                d_in.copy_(torch.from_numpy(x_batch), non_blocking=True)

                if use_tensor_api:
                    context.execute_async_v3(stream_handle=stream.cuda_stream)
                else:
                    context.execute_v2(bindings=bindings)
                stream.synchronize()

                y_batch = d_out.detach().cpu().numpy().copy()
                if y_batch.shape != self.batch_out_shape:
                    continue
                y_r = y_batch[0].reshape(self.single_out_shape).copy()
                y_l = y_batch[1].reshape(self.single_out_shape).copy()
            else:
                y_r_arr = self._run_single(x_r, context, stream, d_in, d_out, use_tensor_api, bindings)
                y_l_arr = self._run_single(x_l, context, stream, d_in, d_out, use_tensor_api, bindings)
                y_r = y_r_arr.reshape(self.single_out_shape).copy()
                y_l = y_l_arr.reshape(self.single_out_shape).copy()
            self._put_latest_output((y_r, y_l))

        print("[TRTWorker] Exiting")

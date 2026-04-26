# controllers/trt_worker_hip.py
"""TensorRT inference worker for the hip exoskeleton.

The original hip model was exported with batch_size=1 and requires sequential
R/L inference.  This worker runs in a separate process and handles:
  - Engine deserialization + warm-up
  - Sequential right-then-left inference
  - Output denormalization  (label_mean / label_std .npy files)

Protocol
--------
Main process sends:  {"r": x_r, "l": x_l}
    x_r, x_l : np.ndarray, shape (1, C, T), already input-normalized

Worker returns:      (y_r, y_l)
    y_r, y_l : np.ndarray, shape (1,), in Nm/kg (denormalized)

Sending ``None`` as the input signals the worker to shut down.
"""

import multiprocessing as mp
from queue import Empty, Full

import numpy as np
import torch
import tensorrt as trt


class HipTRTWorker(mp.Process):
    def __init__(
        self,
        in_q: mp.Queue,
        out_q: mp.Queue,
        engine_path: str,
        label_mean_path: str,
        label_std_path: str,
        single_in_shape: tuple,   # (1, C, T)
        single_out_shape: tuple,  # (O,)  — typically (1,)
    ):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.engine_path = engine_path
        self.label_mean_path = label_mean_path
        self.label_std_path = label_std_path
        self.single_in_shape = tuple(single_in_shape)
        self.single_out_shape = tuple(single_out_shape)

        if len(self.single_in_shape) != 3 or self.single_in_shape[0] != 1:
            raise ValueError(
                f"single_in_shape must be (1, C, T), got {self.single_in_shape}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_io_names(self, engine):
        if not hasattr(engine, "num_io_tensors"):
            return None, None
        in_name = out_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                in_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                out_name = name
        return in_name, out_name

    def _run_once(self, x: np.ndarray, context, d_in, d_out,
                  use_tensor_api, bindings, stream) -> np.ndarray:
        """Run a single forward pass and return the CPU output array."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        d_in.copy_(torch.from_numpy(x), non_blocking=True)
        if use_tensor_api:
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        else:
            context.execute_v2(bindings=bindings)
        stream.synchronize()
        return d_out.detach().cpu().numpy().copy()

    def _get_latest_input(self):
        """Block until an item arrives; drain stale entries."""
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

    # ------------------------------------------------------------------
    # Process entry point
    # ------------------------------------------------------------------
    def run(self):
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass

        label_mean = np.load(self.label_mean_path).astype(np.float32)
        label_std = np.load(self.label_std_path).astype(np.float32)

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            print("[HipTRTWorker] Engine load failed!")
            return

        context = engine.create_execution_context()
        stream = torch.cuda.Stream()

        d_in = torch.empty(self.single_in_shape, dtype=torch.float32, device="cuda")
        d_out = torch.empty(self.single_out_shape, dtype=torch.float32, device="cuda")

        use_tensor_api = hasattr(engine, "num_io_tensors") and hasattr(engine, "get_tensor_name")
        bindings = None

        if use_tensor_api:
            in_name, out_name = self._find_io_names(engine)
            if in_name is None or out_name is None:
                raise RuntimeError("Failed to find TRT input/output tensor names")
            context.set_input_shape(in_name, self.single_in_shape)
            context.set_tensor_address(in_name, int(d_in.data_ptr()))
            context.set_tensor_address(out_name, int(d_out.data_ptr()))
        else:
            bindings = [int(d_in.data_ptr()), int(d_out.data_ptr())]

        # warm-up
        d_in.zero_()
        for _ in range(10):
            self._run_once(
                d_in.cpu().numpy(), context, d_in, d_out,
                use_tensor_api, bindings, stream
            )
        print("[HipTRTWorker] Ready.")

        while True:
            data = self._get_latest_input()
            if data is None:
                break

            if not isinstance(data, dict) or "r" not in data or "l" not in data:
                continue

            x_r = np.asarray(data["r"], dtype=np.float32)
            x_l = np.asarray(data["l"], dtype=np.float32)

            if x_r.shape != self.single_in_shape:
                print(f"[HipTRTWorker] Bad x_r shape {x_r.shape}, expected {self.single_in_shape}")
                continue
            if x_l.shape != self.single_in_shape:
                print(f"[HipTRTWorker] Bad x_l shape {x_l.shape}, expected {self.single_in_shape}")
                continue

            # Sequential inference (engine was compiled for batch_size=1)
            y_r_raw = self._run_once(x_r, context, d_in, d_out, use_tensor_api, bindings, stream)
            y_l_raw = self._run_once(x_l, context, d_in, d_out, use_tensor_api, bindings, stream)

            y_r = (y_r_raw.reshape(self.single_out_shape) * label_std + label_mean).astype(np.float32)
            y_l = (y_l_raw.reshape(self.single_out_shape) * label_std + label_mean).astype(np.float32)

            self._put_latest_output((y_r, y_l))

        print("[HipTRTWorker] Exiting.")

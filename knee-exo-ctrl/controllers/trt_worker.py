import multiprocessing as mp
from queue import Empty, Full
import numpy as np
import torch
import tensorrt as trt


class TRTWorker(mp.Process):
    """
    Controller sends:
        {"r": x_r, "l": x_l}
    where
        x_r.shape == x_l.shape == single_in_shape == (1, C, T)

    Worker stacks them into:
        x_batch.shape == (2, C, T)

    Runs ONE TensorRT inference and returns:
        (y_r, y_l)
    where
        y_r.shape == y_l.shape == single_out_shape == (O,)
    """

    def __init__(self, in_q, out_q, engine_path, single_in_shape, single_out_shape):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.engine_path = engine_path

        self.single_in_shape = tuple(single_in_shape)      # e.g. (1, C, T)
        self.single_out_shape = tuple(single_out_shape)    # e.g. (O,)

        if len(self.single_in_shape) != 3 or self.single_in_shape[0] != 1:
            raise ValueError(f"single_in_shape must be (1, C, T), got {self.single_in_shape}")

        self.batch_in_shape = (2, self.single_in_shape[1], self.single_in_shape[2])
        self.batch_out_shape = (2, *self.single_out_shape)

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

        # Drain queue so we keep only the latest input
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
            print("[TRTWorker] Engine load failed!")
            return

        context = engine.create_execution_context()
        stream = torch.cuda.Stream()

        d_in = torch.empty(self.batch_in_shape, dtype=torch.float32, device="cuda")
        d_out = torch.empty(self.batch_out_shape, dtype=torch.float32, device="cuda")

        use_tensor_api = hasattr(engine, "num_io_tensors") and hasattr(engine, "get_tensor_name")
        bindings = None

        if use_tensor_api:
            in_name, out_name = self._find_io_names(engine)
            if in_name is None or out_name is None:
                raise RuntimeError("Failed to find TRT input/output tensor names")

            ok = context.set_input_shape(in_name, self.batch_in_shape)
            if ok is False:
                raise RuntimeError(f"TRT refused input shape {self.batch_in_shape}")

            context.set_tensor_address(in_name, int(d_in.data_ptr()))
            context.set_tensor_address(out_name, int(d_out.data_ptr()))
        else:
            bindings = [int(d_in.data_ptr()), int(d_out.data_ptr())]

        # warmup
        d_in.zero_()
        for _ in range(5):
            if use_tensor_api:
                context.execute_async_v3(stream_handle=stream.cuda_stream)
            else:
                context.execute_v2(bindings=bindings)
        stream.synchronize()

        print("[TRTWorker] Ready.")

        while True:
            data = self._get_latest_input()
            if data is None:
                break

            if not isinstance(data, dict):
                continue
            if "r" not in data or "l" not in data:
                continue

            x_r = np.asarray(data["r"], dtype=np.float32)
            x_l = np.asarray(data["l"], dtype=np.float32)

            if x_r.shape != self.single_in_shape:
                print(f"[TRTWorker] Bad x_r shape: {x_r.shape}, expected {self.single_in_shape}")
                continue
            if x_l.shape != self.single_in_shape:
                print(f"[TRTWorker] Bad x_l shape: {x_l.shape}, expected {self.single_in_shape}")
                continue

            # (1, C, T) + (1, C, T) -> (2, C, T)
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
                print(f"[TRTWorker] Bad output shape: {y_batch.shape}, expected {self.batch_out_shape}")
                continue

            y_r = y_batch[0].reshape(self.single_out_shape).copy()
            y_l = y_batch[1].reshape(self.single_out_shape).copy()

            self._put_latest_output((y_r, y_l))

        print("[TRTWorker] Exiting...")
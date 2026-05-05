"""TensorRT subprocess worker: bilateral IMU-window model (scalar output, R then L)."""

import multiprocessing as mp

import numpy as np
import tensorrt as trt
import torch


def _trt_inference(input_data: np.ndarray, context, d_input: torch.Tensor, d_output: torch.Tensor) -> float:
    src = torch.from_numpy(input_data).to(dtype=torch.float32)
    d_input.copy_(src, non_blocking=True)
    bindings = [int(d_input.data_ptr()), int(d_output.data_ptr())]
    context.execute_v2(bindings=bindings)
    return float(d_output.item())


class TRTWorkerState2Torque(mp.Process):
    """One shared engine; each job runs right window then left window."""

    def __init__(
        self,
        input_q: mp.Queue,
        output_q: mp.Queue,
        trt_engine_path: str,
        input_mean_path: str,
        input_std_path: str,
        label_mean_path: str,
        label_std_path: str,
        num_input_features: int,
        frame_length: int,
    ):
        super().__init__()
        self.input_q = input_q
        self.output_q = output_q
        self.trt_engine_path = trt_engine_path
        self.input_mean_path = input_mean_path
        self.input_std_path = input_std_path
        self.label_mean_path = label_mean_path
        self.label_std_path = label_std_path
        self.num_input_features = int(num_input_features)
        self.frame_length = int(frame_length)
        self.daemon = True

    def run(self):
        if not torch.cuda.is_available():
            print("[TRTWorkerState2Torque] CUDA not available; worker exiting.")
            return

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.trt_engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        if engine is None:
            print("[TRTWorkerState2Torque] Failed to deserialize TensorRT engine.")
            return
        context = engine.create_execution_context()

        input_mean = np.load(self.input_mean_path).astype(np.float32)
        input_std = np.load(self.input_std_path).astype(np.float32)
        label_mean = np.load(self.label_mean_path).astype(np.float32)
        label_std = np.load(self.label_std_path).astype(np.float32)

        d_input = torch.empty(
            (1, self.num_input_features, self.frame_length), dtype=torch.float32, device="cuda"
        )
        d_output = torch.empty((1,), dtype=torch.float32, device="cuda")

        dummy = np.zeros((1, self.num_input_features, self.frame_length), dtype=np.float32)
        for _ in range(10):
            _ = _trt_inference(dummy, context, d_input, d_output)
        print("TensorRT engine warmed up.\nTrigger the trial to start...")

        while True:
            try:
                data_in = self.input_q.get()
                if data_in is None:
                    print("[TRTWorkerState2Torque] Stop signal received. Exiting.")
                    break

                model_input_r_arr, model_input_l_arr = data_in

                out_r_norm = _trt_inference(model_input_r_arr, context, d_input, d_output)
                model_output_r = np.array([out_r_norm], dtype=np.float32) * label_std + label_mean

                out_l_norm = _trt_inference(model_input_l_arr, context, d_input, d_output)
                model_output_l = np.array([out_l_norm], dtype=np.float32) * label_std + label_mean

                self.output_q.put((model_output_r, model_output_l))
            except Exception as e:
                print(f"[TRTWorkerState2Torque] Error during inference: {e}")
                break

        del context
        del engine
        del runtime
        print("[TRTWorkerState2Torque] Exited.")

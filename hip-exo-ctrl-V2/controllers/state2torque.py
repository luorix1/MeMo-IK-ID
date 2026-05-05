"""
Bilateral State2Torque controller: IMU-window TensorRT + biotorque pipeline.

Ported from the standalone State2Torque reference controller: 6-DOF IMU per side
(right thigh raw, left thigh with reflected Y acc/gyr), scalar output per side,
mass scaling, feedback cancellation, gain delay, Butterworth filtering.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue

import numpy as np

from utils.utils import LowpassFilter, fast_roll

from .base import BaseController, CtrlResult, Sensors
from .trt_worker_state2torque import TRTWorkerState2Torque


class State2Torque(BaseController):
    name = "state2torque"

    def __init__(self, config: dict):
        self.trt_engine_path = str(config["trt_engine_path"])
        base_model_path = os.path.dirname(self.trt_engine_path)
        self.input_mean_path = str(
            config.get("input_mean_path") or os.path.join(base_model_path, "input_mean.npy")
        )
        self.input_std_path = str(
            config.get("input_std_path") or os.path.join(base_model_path, "input_std.npy")
        )
        self.label_mean_path = str(
            config.get("label_mean_path") or os.path.join(base_model_path, "label_mean.npy")
        )
        self.label_std_path = str(
            config.get("label_std_path") or os.path.join(base_model_path, "label_std.npy")
        )

        self.frame_length = int(config.get("frame_length", 95))
        self.fs = int(config["fs"])

        self.body_mass_kg = float(config["body_mass_kg"])
        self.scale_factor = float(config.get("scale_factor_percent", 20)) / 100.0

        desired_delay_ms = float(config.get("desired_delay_ms", 110))
        self.delay_factor = int(desired_delay_ms / 10.0 - 4)

        mean = np.load(self.input_mean_path).astype(np.float32)
        std = np.load(self.input_std_path).astype(np.float32)
        self.current_input_mean = mean
        self.current_input_std = std
        self.num_input_features = int(mean.shape[0])

        self._lpf_R = LowpassFilter(order=2, cutoff=10.0, fs=float(self.fs))
        self._lpf_L = LowpassFilter(order=2, cutoff=10.0, fs=float(self.fs))
        self._zi_R = None
        self._zi_L = None

        self.model_input_arr = np.zeros(
            (2, self.num_input_features, self.frame_length), dtype=np.float32
        )
        self.scaled_torque_arr = np.zeros((2, self.frame_length), dtype=np.float32)
        self.delayed_torque_arr = np.zeros((2, self.frame_length), dtype=np.float32)
        self.filtered_torque_arr = np.zeros((2, self.frame_length), dtype=np.float32)
        self.applied_torque_arr = np.zeros((2, self.frame_length), dtype=np.float32)

        self.last_model_output_r = np.array([0.0], dtype=np.float32)
        self.last_model_output_l = np.array([0.0], dtype=np.float32)

        self.input_q: mp.Queue | None = None
        self.output_q: mp.Queue | None = None
        self.worker: TRTWorkerState2Torque | None = None

    def start(self):
        self.input_q = mp.Queue()
        self.output_q = mp.Queue()
        self.worker = TRTWorkerState2Torque(
            self.input_q,
            self.output_q,
            self.trt_engine_path,
            self.input_mean_path,
            self.input_std_path,
            self.label_mean_path,
            self.label_std_path,
            self.num_input_features,
            self.frame_length,
        )
        self.worker.start()

    def close(self):
        try:
            if self.input_q is not None:
                self.input_q.put(None)
        except Exception:
            pass
        try:
            if self.worker is not None:
                self.worker.join(timeout=3.0)
        except Exception:
            pass

    def step(self, s: Sensors) -> CtrlResult:
        local_l_data = np.asarray(s.imu_L, dtype=np.float32).reshape(6).copy()
        local_r_data = np.asarray(s.imu_R, dtype=np.float32).reshape(6).copy()

        l_data_reflected = local_l_data.copy()
        l_data_reflected[1] *= -1
        l_data_reflected[3] *= -1
        l_data_reflected[5] *= -1

        right_data = np.hstack((local_r_data))
        left_data = np.hstack((l_data_reflected))

        right_latest_input = (right_data - self.current_input_mean) / self.current_input_std
        left_latest_input = (left_data - self.current_input_mean) / self.current_input_std

        self.model_input_arr = fast_roll(self.model_input_arr)
        self.model_input_arr[0, :, -1] = right_latest_input
        self.model_input_arr[1, :, -1] = left_latest_input

        if self.input_q is not None:
            self.input_q.put(
                (
                    self.model_input_arr[0:1, :, :].copy(),
                    self.model_input_arr[1:2, :, :].copy(),
                )
            )

        try:
            if self.output_q is not None:
                model_output_r_val, model_output_l_val = self.output_q.get_nowait()
                self.last_model_output_r = model_output_r_val
                self.last_model_output_l = model_output_l_val
        except queue.Empty:
            model_output_r_val = self.last_model_output_r
            model_output_l_val = self.last_model_output_l

        model_output_combined = np.hstack((model_output_r_val, model_output_l_val))

        current_applied_torque = self.applied_torque_arr[:, -1].copy()
        net_torque_combined = model_output_combined * self.body_mass_kg
        bio_torque_combined = net_torque_combined - current_applied_torque

        self.scaled_torque_arr = fast_roll(self.scaled_torque_arr)
        self.scaled_torque_arr[:, -1] = bio_torque_combined * self.scale_factor

        self.delayed_torque_arr = fast_roll(self.delayed_torque_arr)
        self.delayed_torque_arr[:, -1] = self.scaled_torque_arr[:, -self.delay_factor - 1]

        self.filtered_torque_arr = fast_roll(self.filtered_torque_arr)
        r_f, self._zi_R = self._lpf_R.realtimeButterworth(
            self.delayed_torque_arr[0, -1], zi=self._zi_R
        )
        l_f, self._zi_L = self._lpf_L.realtimeButterworth(
            self.delayed_torque_arr[1, -1], zi=self._zi_L
        )
        self.filtered_torque_arr[0, -1] = r_f
        self.filtered_torque_arr[1, -1] = l_f

        self.applied_torque_arr = fast_roll(self.applied_torque_arr)
        self.applied_torque_arr[:, -1] = self.filtered_torque_arr[:, -1]

        return CtrlResult(
            model_out_R=float(model_output_combined[0]),
            model_out_L=float(model_output_combined[1]),
            applied_R=float(self.filtered_torque_arr[0, -1]),
            applied_L=float(self.filtered_torque_arr[1, -1]),
            extra={
                "net_torque_R": float(net_torque_combined[0]),
                "net_torque_L": float(net_torque_combined[1]),
                "bio_torque_R": float(bio_torque_combined[0]),
                "bio_torque_L": float(bio_torque_combined[1]),
                "scaled_torque_R": float(self.scaled_torque_arr[0, -1]),
                "scaled_torque_L": float(self.scaled_torque_arr[1, -1]),
                "delayed_torque_R": float(self.delayed_torque_arr[0, -1]),
                "delayed_torque_L": float(self.delayed_torque_arr[1, -1]),
                "filtered_torque_R": float(self.filtered_torque_arr[0, -1]),
                "filtered_torque_L": float(self.filtered_torque_arr[1, -1]),
                "applied_torque_R": float(self.applied_torque_arr[0, -1]),
                "applied_torque_L": float(self.applied_torque_arr[1, -1]),
            },
        )

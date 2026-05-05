"""
Hip exoskeleton V2 entrypoint — central `main_hip` + `controllers/` (State2Torque K5).

Behavior matches `V2_Hip_Exo-main/State2Torque 2 Controllers/allnewK5_6min_Cleaned_PreAlloc last 2min.py`:
IMU-window TRT, biotorque pipeline, ring-buffer CSV logging, GPIO sync pulses, mocap/typing triggers.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from typing import Optional

import can
import Jetson.GPIO as GPIO
import numpy as np
import pandas as pd
import torch
import yaml

from controllers import build_controller
from controllers.base import Sensors
from utils.Header_Mocap_trigger import Mocap_trigger
from utils.k5_helper import TeleplotBatch


def _prepend_paths(paths: list) -> None:
    for p in reversed(paths or []):
        if p and p not in sys.path:
            sys.path.insert(0, p)


class TMotorV3HipHardwareV2:
    """PCB2 dict IMUs + TMotorV3 ActuatorGroup; positions/velocities in degrees (script convention)."""

    def __init__(self, cfg: dict):
        _prepend_paths(cfg.get("sensor_python_paths") or [])
        imu_mod = cfg.get("imu_module", "Header_ICM20948_I2C_pcb2")
        icm = __import__(imu_mod, fromlist=["ICM20948_I2C_IMUs"])
        ICM20948_I2C_IMUs = icm.ICM20948_I2C_IMUs

        from epicpower_tmotorV3.actuator_group import ActuatorGroup
        from epicpower_tmotorV3.tmotor_v3 import TMotorV3

        self.can_id_L = int(cfg["can_id_L"])
        self.can_id_R = int(cfg["can_id_R"])
        self.motor_model = str(cfg.get("motor_model", "AK80-9"))
        self.control_freq_hz = float(cfg["fs"])

        input(
            f"[TMotorV3HipHardwareV2] Press Enter to initialize motors "
            f"(CAN ids {self.can_id_L}, {self.can_id_R})..."
        )
        init_list = [
            TMotorV3(mtr_id, self.motor_model) for mtr_id in [self.can_id_L, self.can_id_R]
        ]
        self.mtr_comms = ActuatorGroup(init_list)
        self.imus = ICM20948_I2C_IMUs()

        try:
            self.bus = can.Bus(
                interface="socketcan", channel=str(cfg.get("can_channel", "can0"))
            )
        except Exception as e:
            print(f"Error initializing CAN bus: {e}")
            raise
        self.notifier = can.Notifier(self.bus, [])

    def read_imu_triplet(self):
        imu_dict = self.imus.read_IMUs()
        imu_P = np.asarray(imu_dict["IMU_PELVIS"], dtype=np.float32)
        imu_L = np.asarray(imu_dict["IMU_THIGH_LEFT"], dtype=np.float32)
        imu_R = np.asarray(imu_dict["IMU_THIGH_RIGHT"], dtype=np.float32)
        return imu_P, imu_L, imu_R

    def motor_pos_vel_torque(self, can_id: int):
        pos = self.mtr_comms.get_position(can_id, degrees=True)
        vel = self.mtr_comms.get_velocity(can_id, degrees=True)
        torque = self.mtr_comms.get_torque(can_id)
        return float(pos), float(vel), float(torque)

    def set_torque(self, can_id: int, torque_nm: float):
        self.mtr_comms.set_torque(can_id, float(torque_nm))

    def shutdown(self):
        try:
            self.notifier.stop()
            self.bus.shutdown()
            print("CAN resources cleaned up successfully")
        except Exception as e:
            print(f"Error during CAN cleanup: {e}")


def init_data_buffers(n_samples: int) -> dict:
    f1 = lambda: np.full((n_samples,), np.nan, dtype=np.float32)
    f6 = lambda: np.full((n_samples, 6), np.nan, dtype=np.float32)
    return {
        "timestamp": f1(),
        "mtr_cmd_L": f1(),
        "mtr_cmd_R": f1(),
        "mtr_pos_L": f1(),
        "mtr_pos_R": f1(),
        "mtr_vel_L": f1(),
        "mtr_vel_R": f1(),
        "imu_P": f6(),
        "imu_L": f6(),
        "imu_R": f6(),
        "model_output_L": f1(),
        "model_output_R": f1(),
        "net_torque_L": f1(),
        "net_torque_R": f1(),
        "bio_torque_L": f1(),
        "bio_torque_R": f1(),
        "scaled_torque_L": f1(),
        "scaled_torque_R": f1(),
        "delayed_torque_L": f1(),
        "delayed_torque_R": f1(),
        "filtered_torque_L": f1(),
        "filtered_torque_R": f1(),
        "applied_torque_L": f1(),
        "applied_torque_R": f1(),
        "actual_torque_L": f1(),
        "actual_torque_R": f1(),
        "gpio_output": f1(),
    }


def save_data(data_to_save: dict, trial_name: str, logged_samples: int, max_samples: int):
    if logged_samples > 0:
        valid_len = logged_samples
    else:
        valid_len = int(np.count_nonzero(~np.isnan(data_to_save["timestamp"])))

    buffer_size = data_to_save["timestamp"].shape[0]
    effective_len = min(valid_len, buffer_size)

    print(f"Total data length collected: {valid_len} (saving recent {effective_len})")

    if effective_len == 0:
        print("ERROR: No data collected! valid_len is 0")
        return

    if valid_len <= buffer_size:
        ordered_data = {k: v[:effective_len].copy() for k, v in data_to_save.items()}
    else:
        start = valid_len % buffer_size
        ordered_data = {}
        for k, v in data_to_save.items():
            ordered_data[k] = np.concatenate((v[start:], v[:start]), axis=0)

    start_idx = 0
    end_idx = effective_len
    print(f"Slicing data from recent window, index range: {start_idx} to {end_idx}")

    ts = ordered_data["timestamp"][start_idx:end_idx]
    t0 = ts[0] if len(ts) > 0 else 0.0
    timestamp_sliced = [t - t0 for t in ts]

    slice_keys_scalar = (
        "mtr",
        "model_output",
        "net_torque",
        "bio_torque",
        "scaled_torque",
        "delayed_torque",
        "filtered_torque",
        "applied_torque",
        "actual_torque",
    )
    sliced_data = {"time": timestamp_sliced}
    for k, v in ordered_data.items():
        if k == "timestamp":
            continue
        if k.startswith("imu"):
            sliced_data[k] = v[start_idx:end_idx, :]
        elif any(k.startswith(p) for p in slice_keys_scalar) or k == "gpio_output":
            sliced_data[k] = v[start_idx:end_idx]
        else:
            sliced_data[k] = v[start_idx:end_idx]

    motor_data_keys = ["time", "mtr_pos_L", "mtr_pos_R", "mtr_vel_L", "mtr_vel_R"]
    if sliced_data.get("gpio_output") is not None:
        motor_data_keys.append("gpio_output")
    df_mtr = pd.DataFrame({k: sliced_data[k] for k in motor_data_keys})
    mtr_csv = f"{trial_name}_input_motor.csv"
    df_mtr.to_csv(mtr_csv, index=False)
    print(f"Motor Data saved to {mtr_csv} | shape {df_mtr.shape}")

    imu_data = {
        "time": sliced_data["time"],
        "Pelvis_Acc_X": sliced_data["imu_P"][:, 0],
        "Pelvis_Acc_Y": sliced_data["imu_P"][:, 1],
        "Pelvis_Acc_Z": sliced_data["imu_P"][:, 2],
        "Pelvis_Gyr_X": sliced_data["imu_P"][:, 3],
        "Pelvis_Gyr_Y": sliced_data["imu_P"][:, 4],
        "Pelvis_Gyr_Z": sliced_data["imu_P"][:, 5],
        "Thigh_L_Acc_X": sliced_data["imu_L"][:, 0],
        "Thigh_L_Acc_Y": sliced_data["imu_L"][:, 1],
        "Thigh_L_Acc_Z": sliced_data["imu_L"][:, 2],
        "Thigh_L_Gyr_X": sliced_data["imu_L"][:, 3],
        "Thigh_L_Gyr_Y": sliced_data["imu_L"][:, 4],
        "Thigh_L_Gyr_Z": sliced_data["imu_L"][:, 5],
        "Thigh_R_Acc_X": sliced_data["imu_R"][:, 0],
        "Thigh_R_Acc_Y": sliced_data["imu_R"][:, 1],
        "Thigh_R_Acc_Z": sliced_data["imu_R"][:, 2],
        "Thigh_R_Gyr_X": sliced_data["imu_R"][:, 3],
        "Thigh_R_Gyr_Y": sliced_data["imu_R"][:, 4],
        "Thigh_R_Gyr_Z": sliced_data["imu_R"][:, 5],
    }
    if sliced_data.get("gpio_output") is not None:
        imu_data["gpio_output"] = sliced_data["gpio_output"]
    df_imu = pd.DataFrame(imu_data)
    imu_csv = f"{trial_name}_input_imu.csv"
    df_imu.to_csv(imu_csv, index=False)
    print(f"IMU Data saved to {imu_csv} | shape {df_imu.shape}")

    torque_keys = [
        "time",
        "model_output_L",
        "model_output_R",
        "net_torque_L",
        "net_torque_R",
        "bio_torque_L",
        "bio_torque_R",
        "scaled_torque_L",
        "scaled_torque_R",
        "delayed_torque_L",
        "delayed_torque_R",
        "filtered_torque_L",
        "filtered_torque_R",
        "applied_torque_L",
        "applied_torque_R",
        "mtr_cmd_L",
        "mtr_cmd_R",
        "actual_torque_L",
        "actual_torque_R",
        "gpio_output",
    ]
    df_torque = pd.DataFrame({k: sliced_data[k] for k in torque_keys})
    torque_csv = f"{trial_name}_output_torque.csv"
    df_torque.to_csv(torque_csv, index=False)
    print(f"Torque data saved to {torque_csv} | shape {df_torque.shape}")


class GPIOControl:
    def __init__(self, pin: int):
        self.pin = int(pin)
        self._state = 0
        try:
            GPIO.setwarnings(False)
            GPIO.cleanup()
        except Exception:
            pass
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

    def pulse_start(self):
        GPIO.output(self.pin, GPIO.HIGH)
        self._state = 1

    def pulse_end(self):
        GPIO.output(self.pin, GPIO.LOW)
        self._state = 0

    def state(self) -> int:
        return self._state

    def close(self):
        try:
            GPIO.output(self.pin, GPIO.LOW)
            GPIO.cleanup()
        except Exception:
            pass


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


COLOR_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"
HIGHLIGHT_KEYS = {
    "trial_name",
    "exo_on",
    "trt_engine_path",
    "body_mass_kg",
    "scale_factor_percent",
    "desired_delay_ms",
    "trigger_type",
    "target_duration_sec",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hip exoskeleton V2 (State2Torque K5) control loop.")
    p.add_argument("config", help="Path to YAML config file.")
    return p.parse_args()


_PATH_KEYS = {
    "trt_engine_path",
    "input_mean_path",
    "input_std_path",
    "label_mean_path",
    "label_std_path",
}


def _resolve_relative_paths(cfg: dict, cfg_dir: str) -> None:
    for k in _PATH_KEYS:
        v = cfg.get(k)
        if isinstance(v, str) and not os.path.isabs(v):
            cfg[k] = os.path.normpath(os.path.join(cfg_dir, v))


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg).__name__}")
    cfg["config_path"] = os.path.abspath(config_path)
    _resolve_relative_paths(cfg, os.path.dirname(cfg["config_path"]))
    return cfg


def print_config(cfg: dict) -> None:
    print("=== CONFIG ===")
    for k, v in cfg.items():
        if k in HIGHLIGHT_KEYS:
            print(f"{COLOR_GREEN}{k}: {v}{COLOR_RESET}")
        else:
            print(f"{k}: {v}")
    input("\n==== Check Config ====\nHit Enter to continue...")


_RUNNER: Optional["K5Runner"] = None


class K5Runner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hw: Optional[TMotorV3HipHardwareV2] = None
        self.controller = None
        self.gpio: Optional[GPIOControl] = None
        self.tp: Optional[TeleplotBatch] = None
        self.mocap_trigger = None
        self.data_to_save: dict = {}
        self.logged_samples = 0
        self.max_samples = 0

    def setup(self):
        self.gpio = GPIOControl(int(self.cfg["GPIO_OUTPUT_PIN"]))
        self.tp = TeleplotBatch(self.cfg["teleplot_ip"], int(self.cfg["teleplot_port"]))

        if self.cfg.get("trigger_type") == "mocap":
            self.mocap_trigger = Mocap_trigger(
                server_ip=str(self.cfg["mocap_server_ip"]),
                port_number=int(self.cfg["mocap_server_port"]),
            )
            self.mocap_trigger.start_client()

        self.controller = build_controller(
            self.cfg["controller_name"],
            config=self.cfg,
        )
        self.hw = TMotorV3HipHardwareV2(self.cfg)

        self.max_samples = int(float(self.cfg["collect_last_sec"]) * float(self.cfg["fs"]))
        self.data_to_save = init_data_buffers(self.max_samples)
        self.logged_samples = 0

        self.controller.start()

    def shutdown(self):
        if self.hw:
            try:
                self.hw.set_torque(self.hw.can_id_L, 0.0)
                self.hw.set_torque(self.hw.can_id_R, 0.0)
            except Exception as e:
                print(f"[Shutdown] zero torque failed: {e}")
            try:
                self.hw.shutdown()
            except Exception as e:
                print(f"[Shutdown] hardware.shutdown failed: {e}")

        try:
            if self.controller:
                self.controller.close()
        except Exception as e:
            print(f"[Shutdown] controller.close failed: {e}")

        try:
            if self.gpio:
                self.gpio.pulse_end()
                self.gpio.close()
        except Exception as e:
            print(f"[Shutdown] GPIO error: {e}")

        try:
            if self.tp:
                self.tp.close()
        except Exception:
            pass

        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def save_logs(self):
        save_data(
            self.data_to_save,
            str(self.cfg["trial_name"]),
            self.logged_samples,
            self.max_samples,
        )

    def run(self):
        if self.hw is None or self.controller is None:
            raise RuntimeError("setup() must be called before run().")

        exo_on = bool(self.cfg["exo_on"])
        trigger_type = str(self.cfg["trigger_type"])
        telemetry_target_hz = float(self.cfg["telemetry_target_hz"])
        telemetry_every_n = max(1, int(round(self.hw.control_freq_hz / telemetry_target_hz)))

        target_duration_sec = float(self.cfg["target_duration_sec"])
        target_time_range = float(self.cfg["target_time_range"])
        torque_limit = float(self.cfg["torque_limit"])

        logging_started = False
        first_pulse_sent = False
        first_pulse_end_time: Optional[float] = None
        second_pulse_sent = False
        second_pulse_end_time: Optional[float] = None
        start_time: Optional[float] = None
        start_index = 1

        if trigger_type == "typing":
            input("Wait for the tensorRT to warm up...\n")

        while True:
            if trigger_type == "mocap" and not logging_started:
                if self.mocap_trigger is not None:
                    self.mocap_trigger.wait_for_trigger()
                print("Mocap trigger received - starting data logging")
                start_time = time.time()
                logging_started = True
            elif trigger_type == "typing" and not logging_started:
                start_time = time.time()
                logging_started = True

            if not logging_started or start_time is None:
                continue

            idx = self.logged_samples % self.max_samples
            time_1 = time.time()

            current_pos_L, current_vel_L, current_torque_L = self.hw.motor_pos_vel_torque(
                self.hw.can_id_L
            )
            current_pos_R, current_vel_R, current_torque_R = self.hw.motor_pos_vel_torque(
                self.hw.can_id_R
            )

            imu_P, imu_L, imu_R = self.hw.read_imu_triplet()

            self.data_to_save["mtr_pos_L"][idx] = current_pos_L
            self.data_to_save["mtr_pos_R"][idx] = -current_pos_R
            self.data_to_save["mtr_vel_L"][idx] = current_vel_L
            self.data_to_save["mtr_vel_R"][idx] = -current_vel_R
            self.data_to_save["imu_P"][idx, :] = imu_P
            self.data_to_save["imu_L"][idx, :] = imu_L
            self.data_to_save["imu_R"][idx, :] = imu_R

            s = Sensors(
                imu_P=imu_P,
                imu_L=imu_L,
                imu_R=imu_R,
                pos_L=float(current_pos_L),
                pos_R=float(current_pos_R),
                vel_L=float(current_vel_L),
                vel_R=float(current_vel_R),
            )
            time_3 = time.time()
            r = self.controller.step(s)
            time_4 = time.time()

            motor_cmd_val_L = clamp(r.applied_L, -torque_limit, torque_limit)
            motor_cmd_val_R = clamp(r.applied_R, -torque_limit, torque_limit)
            if not exo_on:
                motor_cmd_val_L, motor_cmd_val_R = 0.0, 0.0

            self.hw.set_torque(self.hw.can_id_L, motor_cmd_val_L)
            self.hw.set_torque(self.hw.can_id_R, -motor_cmd_val_R)

            actual_motor_torque_L = self.hw.mtr_comms.get_torque(self.hw.can_id_L)
            actual_motor_torque_R = -self.hw.mtr_comms.get_torque(self.hw.can_id_R)

            current_time = time.time() - start_time
            gpio_first_at = float(self.cfg.get("gpio_first_pulse_at_sec", 3.0))
            gpio_second_offset = float(
                self.cfg.get("gpio_second_pulse_offset_sec", target_time_range)
            )

            if current_time >= gpio_first_at and not first_pulse_sent:
                try:
                    self.gpio.pulse_start()
                except Exception as e:
                    print(f"[GPIO] first pulse_start error: {e}")
                first_pulse_sent = True
                first_pulse_end_time = current_time + float(self.cfg["PULSE_WIDTH_SEC"])
                print("First GPIO pulse started")

            if (
                first_pulse_sent
                and first_pulse_end_time is not None
                and current_time >= first_pulse_end_time
            ):
                try:
                    self.gpio.pulse_end()
                except Exception as e:
                    print(f"[GPIO] first pulse_end error: {e}")
                first_pulse_end_time = None
                print("First GPIO pulse ended")

            if current_time >= (gpio_first_at + gpio_second_offset) and not second_pulse_sent:
                try:
                    self.gpio.pulse_start()
                except Exception as e:
                    print(f"[GPIO] second pulse_start error: {e}")
                second_pulse_sent = True
                second_pulse_end_time = current_time + float(self.cfg["PULSE_WIDTH_SEC"])
                print("Second GPIO pulse started")

            if (
                second_pulse_sent
                and second_pulse_end_time is not None
                and current_time >= second_pulse_end_time
            ):
                try:
                    self.gpio.pulse_end()
                except Exception as e:
                    print(f"[GPIO] second pulse_end error: {e}")
                second_pulse_end_time = None
                print("Second GPIO pulse ended")

            self.data_to_save["mtr_cmd_R"][idx] = motor_cmd_val_R
            self.data_to_save["mtr_cmd_L"][idx] = motor_cmd_val_L
            self.data_to_save["model_output_R"][idx] = r.model_out_R
            self.data_to_save["model_output_L"][idx] = r.model_out_L
            self.data_to_save["net_torque_R"][idx] = r.extra["net_torque_R"]
            self.data_to_save["net_torque_L"][idx] = r.extra["net_torque_L"]
            self.data_to_save["bio_torque_R"][idx] = r.extra["bio_torque_R"]
            self.data_to_save["bio_torque_L"][idx] = r.extra["bio_torque_L"]
            self.data_to_save["scaled_torque_R"][idx] = r.extra["scaled_torque_R"]
            self.data_to_save["scaled_torque_L"][idx] = r.extra["scaled_torque_L"]
            self.data_to_save["delayed_torque_R"][idx] = r.extra["delayed_torque_R"]
            self.data_to_save["delayed_torque_L"][idx] = r.extra["delayed_torque_L"]
            self.data_to_save["filtered_torque_R"][idx] = r.extra["filtered_torque_R"]
            self.data_to_save["filtered_torque_L"][idx] = r.extra["filtered_torque_L"]
            self.data_to_save["applied_torque_R"][idx] = r.extra["applied_torque_R"]
            self.data_to_save["applied_torque_L"][idx] = r.extra["applied_torque_L"]
            self.data_to_save["actual_torque_R"][idx] = actual_motor_torque_R
            self.data_to_save["actual_torque_L"][idx] = actual_motor_torque_L
            self.data_to_save["gpio_output"][idx] = float(self.gpio.state())

            time_0 = time.time()
            loop_time = time_0 - time_1

            telemetry_data = {
                "time": time.time() - start_time,
                "loop_time": loop_time,
                "inference_time": time_4 - time_3,
                "mtr_cmd_R": motor_cmd_val_R,
                "mtr_cmd_L": motor_cmd_val_L,
                "actual_torque_R": actual_motor_torque_R,
                "actual_torque_L": actual_motor_torque_L,
                "gpio_output": self.gpio.state(),
                "output_R": r.model_out_R,
                "output_L": r.model_out_L,
                "scaled_torque_R": r.extra["scaled_torque_R"],
                "scaled_torque_L": r.extra["scaled_torque_L"],
                "delayed_torque_R": r.extra["delayed_torque_R"],
                "delayed_torque_L": r.extra["delayed_torque_L"],
                "filtered_torque_R": r.extra["filtered_torque_R"],
                "filtered_torque_L": r.extra["filtered_torque_L"],
            }
            if (start_index % telemetry_every_n) == 0 and self.tp is not None:
                self.tp.send(telemetry_data)

            if (time.time() - start_time) > (start_index / self.hw.control_freq_hz):
                print(
                    f"Loop time exceeded: "
                    f"{((time.time() - start_time) - (start_index / self.hw.control_freq_hz)):.6f}"
                )
            else:
                while (time.time() - start_time) < (start_index / self.hw.control_freq_hz):
                    pass

            self.data_to_save["timestamp"][idx] = time.time() - start_time
            start_index += 1
            self.logged_samples += 1

            if (time.time() - start_time) >= target_duration_sec:
                print("Target duration reached. Stopping trial.")
                break


def _handle_signal(sig, frame):
    global _RUNNER
    print(f"\nSignal {sig} received; shutting down...")
    try:
        if _RUNNER:
            _RUNNER.shutdown()
            _RUNNER.save_logs()
    except Exception as e:
        print(f"[Signal] error: {e}")
    try:
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    print("Exiting program")
    os._exit(0)


def main():
    global _RUNNER
    args = parse_args()
    cfg = load_config(args.config)
    print_config(cfg)

    runner = K5Runner(cfg)
    _RUNNER = runner
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    gc.disable()
    try:
        runner.setup()
        runner.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt.")
    except Exception as e:
        print(f"[MAIN ERROR] {e}")
        traceback.print_exc()
    finally:
        runner.shutdown()
        runner.save_logs()
        gc.enable()


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    mp.set_start_method("spawn", force=True)
    main()

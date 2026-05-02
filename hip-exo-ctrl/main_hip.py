"""
Dual hip exoskeleton control entrypoint (YAML config + modular controllers).

Directory layout matches `os_kinetics/knee-exo-ctrl/` (cfg/, controllers/, utils/,
tcn_model/, data_analysis/, root main). Hip motor/IMU backends live in this file
alongside the runner, analogous to knee Jetson/Teensy wiring in `main_knee.py`.
"""

from __future__ import annotations

import argparse
import enum
import gc
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from typing import Optional, Protocol, Tuple

import can
import Jetson.GPIO as GPIO
import numpy as np
import torch
import yaml

from controllers import build_controller
from controllers.base import Sensors
from utils.Header_Mocap_trigger import Mocap_trigger
from utils.teleplot import Teleplot
from utils.utils import RateKeeper


# --- Hip plant I/O (mirrors knee hardware setup living in main_knee.py) ---


class HipHardware(Protocol):
    can_id_L: int
    can_id_R: int
    control_freq_hz: float
    frame_length: int

    def read_imu_triplet(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...

    def motor_pos_vel(self, can_id: int) -> Tuple[float, float]:
        ...

    def set_torque(self, can_id: int, torque_nm: float) -> None:
        ...

    def shutdown(self) -> None:
        ...


def _prepend_paths(paths: list) -> None:
    for p in reversed(paths or []):
        if p and p not in sys.path:
            sys.path.insert(0, p)


class EpicPowerHipHardware:
    """Melody `main.py` stack: `epicpower.actuation.Motors` + ICM20948 dict layout."""

    def __init__(self, cfg: dict):
        _prepend_paths(cfg.get("sensor_python_paths") or [])
        from epicpower import actuation

        imu_mod = cfg.get("imu_module_epicpower", "Header_ICM20948_I2C_pcb2")
        icm = __import__(imu_mod, fromlist=["ICM20948_I2C_IMUs"])
        ICM20948_I2C_IMUs = icm.ICM20948_I2C_IMUs

        self.can_id_L = int(cfg["can_id_L"])
        self.can_id_R = int(cfg["can_id_R"])
        self.motor_model = str(cfg.get("motor_model", "AK80-9"))
        self.control_freq_hz = float(cfg["fs"])
        self.frame_length = int(cfg.get("frame_length", 95))

        input(f"[EpicPowerHipHardware] Press Enter to initialize motors (CAN ids {self.can_id_L}, {self.can_id_R})...")
        init_dict = {mtr_id: self.motor_model for mtr_id in [self.can_id_L, self.can_id_R]}
        self.mtr_comms = actuation.Motors(init_dict)
        self.imus = ICM20948_I2C_IMUs()

        self.bus = can.Bus(interface="socketcan", channel=str(cfg.get("can_channel", "can0")))
        self.notifier = can.Notifier(self.bus, [])

    def read_imu_triplet(self):
        imu = self.imus.read_IMUs()
        imu_P = np.asarray(imu["IMU_PELVIS"], dtype=np.float32)
        imu_L = np.asarray(imu["IMU_THIGH_LEFT"], dtype=np.float32)
        imu_R = np.asarray(imu["IMU_THIGH_RIGHT"], dtype=np.float32)
        return imu_P, imu_L, imu_R

    def motor_pos_vel(self, can_id: int):
        pos = self.mtr_comms.get_position(can_id, degrees=False)
        vel = self.mtr_comms.get_velocity(can_id, degrees=False)
        return float(pos), float(vel)

    def set_torque(self, can_id: int, torque_nm: float):
        self.mtr_comms.set_torque(can_id, float(torque_nm))

    def shutdown(self):
        try:
            self.notifier.stop()
            self.bus.shutdown()
        except Exception:
            pass


class TMotorV3HipHardware:
    """Melody `main_v1exo.py` stack: TMotorV3 + ActuatorGroup + flat 18-vector IMUs."""

    def __init__(self, cfg: dict):
        _prepend_paths(cfg.get("sensor_python_paths") or [])
        pkg = str(cfg.get("sensor_package", "sensor_motor"))
        if pkg == "hip_exo.sensor_motor":
            from hip_exo.sensor_motor.Header_ICM20948_I2C import ICM20948_I2C_IMUs
            from hip_exo.sensor_motor.epicpower_tmotorV3.actuator_group import ActuatorGroup
            from hip_exo.sensor_motor.epicpower_tmotorV3.tmotor_v3 import TMotorV3
        else:
            from sensor_motor.Header_ICM20948_I2C import ICM20948_I2C_IMUs
            from sensor_motor.epicpower_tmotorV3.actuator_group import ActuatorGroup
            from sensor_motor.epicpower_tmotorV3.tmotor_v3 import TMotorV3

        self.can_id_L = int(cfg["can_id_L"])
        self.can_id_R = int(cfg["can_id_R"])
        self.motor_model = str(cfg.get("motor_model", "AK80-9"))
        self.control_freq_hz = float(cfg["fs"])
        self.frame_length = int(cfg.get("frame_length", 95))

        input(f"[TMotorV3HipHardware] Press Enter to initialize motors (CAN ids {self.can_id_L}, {self.can_id_R})...")
        init_list = [
            TMotorV3(self.can_id_L, self.motor_model),
            TMotorV3(self.can_id_R, self.motor_model),
        ]
        self.mtr_comms = ActuatorGroup(init_list)
        self.imus = ICM20948_I2C_IMUs()

        self.bus = can.Bus(interface="socketcan", channel=str(cfg.get("can_channel", "can0")))
        self.notifier = can.Notifier(self.bus, [])

    def read_imu_triplet(self):
        imu = self.imus.read_IMUs()
        flat = np.asarray(imu, dtype=np.float32).reshape(-1)
        if flat.size < 18:
            raise RuntimeError(f"Expected 18 IMU scalars, got length {flat.size}")
        imu_P = flat[0:6].copy()
        imu_L = flat[6:12].copy()
        imu_R = flat[12:18].copy()
        return imu_P, imu_L, imu_R

    def motor_pos_vel(self, can_id: int):
        pos = self.mtr_comms.get_position(can_id, degrees=False)
        vel = self.mtr_comms.get_velocity(can_id, degrees=False)
        return float(pos), float(vel)

    def set_torque(self, can_id: int, torque_nm: float):
        self.mtr_comms.set_torque(can_id, float(torque_nm))

    def shutdown(self):
        try:
            self.notifier.stop()
            self.bus.shutdown()
        except Exception:
            pass


def build_hip_hardware(cfg: dict) -> HipHardware:
    backend = str(cfg.get("motor_backend", "tmotor_v3")).lower()
    if backend == "tmotor_v3":
        return TMotorV3HipHardware(cfg)
    if backend == "epicpower":
        return EpicPowerHipHardware(cfg)
    raise ValueError(f"Unknown motor_backend: {backend}. Use 'tmotor_v3' or 'epicpower'.")

COLOR_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"

HIGHLIGHT_KEYS = {
    "controller_name",
    "exo_on",
    "scale",
    "trial_name",
    "exp_time_sec",
    "motor_backend",
    "fs",
    "teleplot_ip",
    "teleplot_port",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run dual hip exoskeleton control loop.")
    p.add_argument("config", help="Path to YAML config file.")
    return p.parse_args()


_PATH_KEYS = {
    "engine_path", "mean_std_path",
    "input_mean_path", "input_std_path", "out_mean_path", "out_std_path",
}


def _resolve_relative_paths(cfg: dict, cfg_dir: str) -> None:
    """Resolve relative paths in controller sub-dict against the config file's directory."""
    controller_cfg = cfg.get("controller") or {}
    for k, v in controller_cfg.items():
        if k in _PATH_KEYS and isinstance(v, str) and not os.path.isabs(v):
            controller_cfg[k] = os.path.normpath(os.path.join(cfg_dir, v))


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


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


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


class Side(enum.Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


def build_data_log(cfg: dict) -> dict:
    log_size = int(cfg["exp_time_sec"] * cfg["fs"])
    return {
        "time": np.zeros(log_size),
        "hip_pos_L": np.zeros(log_size),
        "hip_pos_R": np.zeros(log_size),
        "hip_vel_L": np.zeros(log_size),
        "hip_vel_R": np.zeros(log_size),
        "cmd_L": np.zeros(log_size),
        "cmd_R": np.zeros(log_size),
        "model_out_L": np.zeros(log_size),
        "model_out_R": np.zeros(log_size),
        "applied_L": np.zeros(log_size),
        "applied_R": np.zeros(log_size),
        "imu_P": np.zeros((log_size, 6)),
        "imu_L": np.zeros((log_size, 6)),
        "imu_R": np.zeros((log_size, 6)),
        "GPIO": np.zeros(log_size),
    }


class DualHipRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_log = build_data_log(cfg)
        self.hw: Optional[HipHardware] = None
        self.controller = None
        self.gpio = GPIOControl(int(cfg["GPIO_OUTPUT_PIN"]))
        self.tp: Optional[Teleplot] = None
        self.mocap_trigger = None
        self.current_idx = 0

    def setup(self):
        if self.cfg.get("trigger_type") == "mocap":
            try:
                self.mocap_trigger = Mocap_trigger(server_ip="172.24.44.177", port_number=10)
                self.mocap_trigger.start_client()
            except Exception as e:
                print(f"[Mocap] start_client error: {e}")

        self.tp = Teleplot(self.cfg["teleplot_ip"], self.cfg["teleplot_port"])

        self.hw = build_hip_hardware(self.cfg)

        ctrl_kw = dict(self.cfg.get("controller") or {})
        ctrl_kw.setdefault("fs", int(self.cfg["fs"]))
        self.controller = build_controller(self.cfg["controller_name"], **ctrl_kw)
        self.controller.start()

        print("\n--- Dual Hip Exo Control Loop Started ---")
        print(f"Controller: {self.cfg['controller_name']} | backend: {self.cfg.get('motor_backend')}")
        print(f"Teleplot: {self.cfg['teleplot_ip']}:{self.cfg['teleplot_port']}")
        print("Press Ctrl+C to stop.")

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
            self.gpio.pulse_end()
            self.gpio.close()
        except Exception as e:
            print(f"[Shutdown] GPIO error: {e}")

        try:
            if self.tp is not None:
                self.tp.close()
        except Exception:
            pass

        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        print("Shutdown complete.")

    def save_data(self):
        try:
            print("Preparing data for saving...")
            for key in self.data_log.keys():
                self.data_log[key] = self.data_log[key][: self.current_idx]
            np.savez(self.cfg["trial_name"], **self.data_log)
            print(f"=== Data saved: {self.cfg['trial_name']}.npz ===")
        except Exception as e:
            print(f"[save_data] error: {e}")

    def run(self):
        if self.hw is None:
            raise RuntimeError("setup() must be called before run().")

        if self.cfg["trigger_type"] == "typing":
            input("Press Enter to start...\n")
        elif self.cfg["trigger_type"] == "mocap":
            if self.mocap_trigger is not None:
                self.mocap_trigger.wait_for_trigger()
            else:
                print("[WARN] mocap_trigger is None; starting immediately.")
        else:
            raise ValueError(f"Unknown trigger_type: {self.cfg['trigger_type']}")

        side = Side(self.cfg["side"])

        rk = RateKeeper(self.cfg["fs"])
        rk.start()
        t0 = time.perf_counter()

        trial_start_sec = 0.0
        trial_dur_sec = float(self.cfg["exp_time_sec"])
        pulse_after_start = float(self.cfg["GPIO_START_DELAY_SEC"])

        first_pulse_sent = False
        first_pulse_end: Optional[float] = None
        second_pulse_sent = False
        second_pulse_end: Optional[float] = None

        invert_right = bool(self.cfg.get("invert_right_torque_cmd", True))
        LOG_DIVIDER = 1

        while True:
            _, _, k = rk.wait()
            step_start = time.perf_counter()
            actual_time = step_start - t0

            use_left = side in (Side.LEFT, Side.BOTH)
            use_right = side in (Side.RIGHT, Side.BOTH)

            imu_P, imu_L, imu_R = self.hw.read_imu_triplet()

            pos_L = vel_L = 0.0
            pos_R = vel_R = 0.0
            if use_left:
                pos_L, vel_L = self.hw.motor_pos_vel(self.hw.can_id_L)
            if use_right:
                pos_R, vel_R = self.hw.motor_pos_vel(self.hw.can_id_R)

            s = Sensors(
                imu_P=imu_P,
                imu_L=imu_L,
                imu_R=imu_R,
                pos_L=pos_L,
                pos_R=pos_R,
                vel_L=vel_L,
                vel_R=vel_R,
            )

            r = self.controller.step(s)

            cmd_L = float(r.applied_L) if use_left else 0.0
            cmd_R = float(r.applied_R) if use_right else 0.0

            if not self.cfg["exo_on"]:
                cmd_L = 0.0
                cmd_R = 0.0

            cmd_L *= float(self.cfg["scale"])
            cmd_R *= float(self.cfg["scale"])

            lim = float(self.cfg["torque_limit"])
            cmd_L = clamp(cmd_L, -lim, lim)
            cmd_R = clamp(cmd_R, -lim, lim)

            if use_left:
                self.hw.set_torque(self.hw.can_id_L, cmd_L)
            else:
                self.hw.set_torque(self.hw.can_id_L, 0.0)

            if use_right:
                self.hw.set_torque(self.hw.can_id_R, (-cmd_R) if invert_right else cmd_R)
            else:
                self.hw.set_torque(self.hw.can_id_R, 0.0)

            now = actual_time

            if (not first_pulse_sent) and (now >= trial_start_sec + pulse_after_start):
                try:
                    self.gpio.pulse_start()
                except Exception as e:
                    print(f"[GPIO] first pulse_start error: {e}")
                first_pulse_sent = True
                first_pulse_end = now + float(self.cfg["PULSE_WIDTH_SEC"])

            if first_pulse_sent and first_pulse_end is not None and now >= first_pulse_end:
                try:
                    self.gpio.pulse_end()
                except Exception as e:
                    print(f"[GPIO] first pulse_end error: {e}")
                first_pulse_end = None

            if (not second_pulse_sent) and (now >= trial_start_sec + trial_dur_sec):
                try:
                    self.gpio.pulse_start()
                except Exception as e:
                    print(f"[GPIO] second pulse_start error: {e}")
                second_pulse_sent = True
                second_pulse_end = now + float(self.cfg["PULSE_WIDTH_SEC"])

            if second_pulse_sent and second_pulse_end is not None and now >= second_pulse_end:
                try:
                    self.gpio.pulse_end()
                except Exception as e:
                    print(f"[GPIO] second pulse_end error: {e}")
                second_pulse_end = None

            step_end = time.perf_counter()
            gpio_state = float(self.gpio.state())

            if k % LOG_DIVIDER == 0:
                try:
                    self.tp.sendValue("cmd_L", cmd_L)
                    self.tp.sendValue("cmd_R", cmd_R)
                    self.tp.sendValue("model_out_L", r.model_out_L)
                    self.tp.sendValue("model_out_R", r.model_out_R)
                    self.tp.sendValue("applied_L", r.applied_L)
                    self.tp.sendValue("applied_R", r.applied_R)
                    self.tp.sendValue("hip_L", pos_L)
                    self.tp.sendValue("hip_R", pos_R)
                    self.tp.sendValue("gyro_r", float(imu_R[4]))
                    self.tp.sendValue("gyro_l", float(imu_L[4]))
                    self.tp.sendValue("GPIO", gpio_state)
                    self.tp.sendValue("roop_time", step_end - step_start)
                except Exception:
                    pass

            if self.current_idx < len(self.data_log["time"]):
                self.data_log["time"][self.current_idx] = actual_time
                self.data_log["hip_pos_L"][self.current_idx] = pos_L
                self.data_log["hip_pos_R"][self.current_idx] = pos_R
                self.data_log["hip_vel_L"][self.current_idx] = vel_L
                self.data_log["hip_vel_R"][self.current_idx] = vel_R
                self.data_log["cmd_L"][self.current_idx] = cmd_L
                self.data_log["cmd_R"][self.current_idx] = cmd_R
                self.data_log["model_out_L"][self.current_idx] = r.model_out_L
                self.data_log["model_out_R"][self.current_idx] = r.model_out_R
                self.data_log["applied_L"][self.current_idx] = r.applied_L
                self.data_log["applied_R"][self.current_idx] = r.applied_R
                self.data_log["imu_P"][self.current_idx] = imu_P
                self.data_log["imu_L"][self.current_idx] = imu_L
                self.data_log["imu_R"][self.current_idx] = imu_R
                self.data_log["GPIO"][self.current_idx] = gpio_state

            self.current_idx += 1


_RUNNER: Optional[DualHipRunner] = None


def _handle_signal(sig, frame):
    global _RUNNER
    print(f"\nSignal {sig} received; shutting down...")
    try:
        if _RUNNER:
            _RUNNER.shutdown()
    except Exception as e:
        print(f"[Signal] shutdown error: {e}")
    try:
        if _RUNNER:
            _RUNNER.save_data()
    except Exception as e:
        print(f"[Signal] save_data error: {e}")
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

    runner = DualHipRunner(cfg)
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
        gc.enable()


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    mp.set_start_method("spawn", force=True)
    main()

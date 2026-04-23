#!/usr/bin/env python3
"""
Dual-hip exoskeleton control (os_kinetics), mirroring ``knee-exo-ctrl``.

Sensors:
  * Pelvis IMU + left thigh + right thigh (6-vector each: accel xyz, gyro xyz).
  * Pelvis is read from the device configured by ``pelvis_imu_teensy`` (``left`` | ``right``)
    and ``pelvis_imu_which`` (1 or 2), defaulting to **left** Teensy **IMU2** (right keeps IMU1 = thigh only).
  * Left / right thigh: ``IMU1`` on each leg Teensy (same layout as knee ``thigh`` = IMU1).

Inference uses motor **encoder** hip angle and **joint_velocity_source**:
  * ``imu_gyro_delta``: rate = LPF(−gyro_Y_thigh) − LPF(−gyro_Y_pelvis) per leg (rad/s after unit conversion).
  * ``encoder``: motor velocity (deg/s → rad/s), same path as knee.

Hardware deps load from ``jetson_teensy_python_path`` in the YAML.
"""
from __future__ import annotations

import argparse
import atexit
import enum
import gc
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from controllers import build_controller
from controllers.base import Sensors
from utils.rate_keeper import RateKeeper
from utils.teleplot import Teleplot

COLOR_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"

HIGHLIGHT_KEYS = {
    "exo_on",
    "scale",
    "controller_name",
    "exp_time_sec",
    "run_dir",
    "joint_velocity_source",
    "moment_mass_kg",
    "pelvis_imu_teensy",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual hip exoskeleton control (os_kinetics hip-exo-ctrl).")
    p.add_argument("config", help="Path to YAML config file.")
    return p.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg).__name__}")
    cfg["config_path"] = os.path.abspath(config_path)
    return cfg


def print_config(cfg: dict) -> None:
    print("=== CONFIG ===")
    for k, v in cfg.items():
        if k in HIGHLIGHT_KEYS:
            print(f"{COLOR_GREEN}{k}: {v}{COLOR_RESET}")
        else:
            print(f"{k}: {v}")
    input("\n==== Check Config ====\nHit Enter to continue...")


def build_data_log(cfg: dict) -> dict:
    log_size = int(cfg["exp_time_sec"] * cfg["fs"])
    return {
        "time": np.zeros(log_size),
        "hip_angle_r": np.zeros(log_size),
        "hip_angle_l": np.zeros(log_size),
        "hip_angle_r_u_gyr": np.zeros(log_size),
        "hip_angle_l_u_gyr": np.zeros(log_size),
        "gyro_neg_y_pelvis": np.zeros(log_size),
        "gyro_neg_y_thigh_r": np.zeros(log_size),
        "gyro_neg_y_thigh_l": np.zeros(log_size),
        "cmd_L": np.zeros(log_size),
        "cmd_R": np.zeros(log_size),
        "GPIO": np.zeros(log_size),
    }


class GPIOControl:
    def __init__(self, pin: int):
        import Jetson.GPIO as GPIO  # type: ignore

        self._GPIO = GPIO
        self.pin = int(pin)
        self._state = 0
        try:
            GPIO.setwarnings(False)
            GPIO.cleanup()
        except Exception:
            pass
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

    def pulse_start(self) -> None:
        self._GPIO.output(self.pin, self._GPIO.HIGH)
        self._state = 1

    def pulse_end(self) -> None:
        self._GPIO.output(self.pin, self._GPIO.LOW)
        self._state = 0

    def state(self) -> int:
        return self._state

    def close(self) -> None:
        try:
            self._GPIO.output(self.pin, self._GPIO.LOW)
            self._GPIO.cleanup()
        except Exception:
            pass


class Side(enum.Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def imu6_zeros() -> np.ndarray:
    return np.zeros((6,), dtype=np.float32)


def pack_imu6_from_device(dev, which: int) -> np.ndarray:
    if which == 1:
        a = dev.IMU1_accel_data[-1] if dev.IMU1_accel_data else (0.0, 0.0, 0.0)
        g = dev.IMU1_gyro_data[-1] if dev.IMU1_gyro_data else (0.0, 0.0, 0.0)
    else:
        a = dev.IMU2_accel_data[-1] if dev.IMU2_accel_data else (0.0, 0.0, 0.0)
        g = dev.IMU2_gyro_data[-1] if dev.IMU2_gyro_data else (0.0, 0.0, 0.0)
    return np.asarray([a[0], a[1], a[2], g[0], g[1], g[2]], dtype=np.float32)


def latest_pos_vel(dev) -> Tuple[float, float]:
    pos = float(dev.Motor_pos_data[-1]) if dev.Motor_pos_data else 0.0
    vel = float(dev.Motor_vel_data[-1]) if dev.Motor_vel_data else 0.0
    return pos, vel


class DualHipRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_log = build_data_log(cfg)
        self.left_dev = None
        self.right_dev = None
        self.left_exo = None
        self.right_exo = None
        self.controller = None
        self.current_idx = 0
        self.gpio = None
        self.mocap_trigger = None
        self.tp: Optional[Teleplot] = None

        use_gpio = bool(cfg.get("use_gpio", True))
        if use_gpio:
            try:
                self.gpio = GPIOControl(int(cfg["GPIO_OUTPUT_PIN"]))
            except Exception as e:
                print(f"[GPIO] disabled ({e}); continuing without sync pulses.")
                self.gpio = None

    def setup(self) -> None:
        jetson_path = self.cfg.get("jetson_teensy_python_path")
        if jetson_path:
            jp = str(jetson_path)
            if jp not in sys.path:
                sys.path.insert(0, jp)

        from Jetson_Teensy import Device, JetsonCanInterface, configure_can_interface  # type: ignore

        if self.cfg.get("trigger_type") == "mocap":
            mocap_mod = self.cfg.get("mocap_trigger_module")
            if mocap_mod:
                try:
                    mod = __import__(str(mocap_mod), fromlist=["Mocap_trigger"])
                    Mocap_trigger = getattr(mod, "Mocap_trigger")
                    self.mocap_trigger = Mocap_trigger(
                        server_ip=str(self.cfg.get("mocap_server_ip", "127.0.0.1")),
                        port_number=int(self.cfg.get("mocap_port", 10)),
                    )
                    self.mocap_trigger.start_client()
                except Exception as e:
                    print(f"[Mocap] start_client error: {e}")

        configure_can_interface(channel=self.cfg["can_channel"])
        self.tp = Teleplot(self.cfg["teleplot_ip"], int(self.cfg["teleplot_port"]))

        self.left_dev = Device(teleplot_ip=self.cfg["teleplot_ip"], teleplot_port=int(self.cfg["teleplot_port"]))
        self.right_dev = Device(teleplot_ip=self.cfg["teleplot_ip"], teleplot_port=int(self.cfg["teleplot_port"]))

        self.left_exo = JetsonCanInterface(
            device_storage=self.left_dev, channel=self.cfg["can_channel"], teensy_id=self.cfg["teensy_id_left"]
        )
        self.right_exo = JetsonCanInterface(
            device_storage=self.right_dev, channel=self.cfg["can_channel"], teensy_id=self.cfg["teensy_id_right"]
        )

        print(f"Initializing Left hip (ID {hex(int(self.cfg['teensy_id_left']))})...")
        self.left_exo.connect()
        print(f"Initializing Right hip (ID {hex(int(self.cfg['teensy_id_right']))})...")
        self.right_exo.connect()

        self.controller = build_controller(self.cfg["controller_name"], config=self.cfg)
        self.controller.start()

        try:
            t0 = time.perf_counter()
            self.left_exo.set_reference_time(t0)
            self.right_exo.set_reference_time(t0)
        except Exception:
            pass

        print("\n--- Dual Hip Exo Control Loop Started (os_kinetics) ---")
        print(f"Teleplot: {self.cfg['teleplot_ip']}:{self.cfg['teleplot_port']}")
        print("Press Ctrl+C to stop.")

    def shutdown(self) -> None:
        try:
            if self.left_exo:
                self.left_exo.setTorque(0.0)
            if self.right_exo:
                self.right_exo.setTorque(0.0)
        except Exception as e:
            print(f"[Shutdown] setTorque(0) failed: {e}")

        time.sleep(0.1)

        for exo, label in ((self.left_exo, "left"), (self.right_exo, "right")):
            try:
                if exo:
                    exo.close()
            except Exception as e:
                print(f"[Shutdown] {label}_exo.close failed: {e}")

        try:
            if self.controller:
                self.controller.close()
        except Exception as e:
            print(f"[Shutdown] controller.close failed: {e}")

        try:
            if self.gpio is not None:
                self.gpio.pulse_end()
                self.gpio.close()
        except Exception as e:
            print(f"[Exit] GPIO cleanup error: {e}")

        try:
            if self.tp is not None:
                self.tp.close()
        except Exception:
            pass

        try:
            gc.enable()
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        print("System Shutdown Complete.")

    def _pelvis_dev_which(self) -> Tuple[object, int]:
        side = str(self.cfg.get("pelvis_imu_teensy", "left")).lower()
        which = int(self.cfg.get("pelvis_imu_which", 2))
        dev = self.left_dev if side == "left" else self.right_dev
        return dev, which

    def run(self) -> None:
        tt = self.cfg["trigger_type"]
        if tt == "typing":
            _ = input("Press Enter to start...\n")
        elif tt == "mocap":
            if self.mocap_trigger is not None:
                self.mocap_trigger.wait_for_trigger()
            else:
                print("[WARN] mocap_trigger is None. Starting immediately.")
        else:
            raise NotImplementedError(f"Unknown trigger_type: {tt}")

        side = Side(self.cfg["side"])
        rk = RateKeeper(float(self.cfg["fs"]))
        rk.start()
        t0 = time.perf_counter()
        prev_loop_time: Optional[float] = None

        trial_start_sec = 0.0
        trial_dur_sec = float(self.cfg["exp_time_sec"])
        pulse_after_start = float(self.cfg["GPIO_START_DELAY_SEC"])

        first_pulse_sent = False
        first_pulse_end: Optional[float] = None
        second_pulse_sent = False
        second_pulse_end: Optional[float] = None

        pelv_dev, pelv_which = self._pelvis_dev_which()
        gy_idx = int(self.cfg.get("teleplot_gyro_y_index", 4))

        while True:
            _, _, _ = rk.wait()
            loop_now = time.perf_counter()
            step_start = loop_now
            if prev_loop_time is None:
                loop_dt = 0.0
            else:
                loop_dt = loop_now - prev_loop_time
            prev_loop_time = loop_now
            _ = loop_dt

            use_left = side in (Side.LEFT, Side.BOTH)
            use_right = side in (Side.RIGHT, Side.BOTH)

            if use_left:
                self.left_exo.getParameters()
            if use_right:
                self.right_exo.getParameters()

            pos_L = vel_L = 0.0
            pos_R = vel_R = 0.0
            imu_P = imu_L = imu_R = imu6_zeros()

            if use_left:
                pos_L, vel_L = latest_pos_vel(self.left_dev)
                imu_L = pack_imu6_from_device(self.left_dev, which=1)
            if use_right:
                pos_R, vel_R = latest_pos_vel(self.right_dev)
                imu_R = pack_imu6_from_device(self.right_dev, which=1)

            imu_P = pack_imu6_from_device(pelv_dev, which=pelv_which)

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

            cmd_L = cmd_L * float(self.cfg["scale"])
            cmd_R = cmd_R * float(self.cfg["scale"])

            tlim = float(self.cfg["torque_limit"])
            cmd_L = clamp(cmd_L, -tlim, tlim)
            cmd_R = clamp(cmd_R, -tlim, tlim)

            if use_left:
                self.left_exo.setTorque(-cmd_L)
            if use_right:
                self.right_exo.setTorque(cmd_R)

            now = step_start - t0

            if self.gpio is not None:
                if (not first_pulse_sent) and (now >= trial_start_sec + pulse_after_start):
                    try:
                        self.gpio.pulse_start()
                    except Exception as e:
                        print(f"[GPIO] first pulse_start error: {e}")
                    first_pulse_sent = True
                    first_pulse_end = now + float(self.cfg["PULSE_WIDTH_SEC"])

                if first_pulse_sent and (first_pulse_end is not None) and (now >= first_pulse_end):
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

                if second_pulse_sent and (second_pulse_end is not None) and (now >= second_pulse_end):
                    try:
                        self.gpio.pulse_end()
                    except Exception as e:
                        print(f"[GPIO] second pulse_end error: {e}")
                    second_pulse_end = None

            step_end = time.perf_counter()
            actual_time = step_start - t0

            if self.tp is not None:
                try:
                    gpio_state = float(self.gpio.state()) if self.gpio is not None else 0.0
                    self.tp.sendValue("cmd_R", cmd_R)
                    self.tp.sendValue("cmd_L", cmd_L)
                    self.tp.sendValue("hip_angle_r_u_gyr", r.extra.get("hip_r_u_gyr", 0.0))
                    self.tp.sendValue("hip_angle_l_u_gyr", r.extra.get("hip_l_u_gyr", 0.0))
                    self.tp.sendValue("hip_angle_r_u", r.extra.get("hip_angle_r_u", 0.0))
                    self.tp.sendValue("hip_angle_l_u", r.extra.get("hip_angle_l_u", 0.0))
                    self.tp.sendValue("hip_angle_r", r.extra.get("hip_angle_r", 0.0))
                    self.tp.sendValue("hip_angle_l", r.extra.get("hip_angle_l", 0.0))
                    self.tp.sendValue("GPIO", gpio_state)
                    self.tp.sendValue("gyro_neg_y_pelvis", float(-imu_P[gy_idx]))
                    self.tp.sendValue("gyro_neg_y_thigh_r", float(-imu_R[gy_idx]))
                    self.tp.sendValue("gyro_neg_y_thigh_l", float(-imu_L[gy_idx]))
                    self.tp.sendValue("moment_nm_kg_r", r.extra.get("moment_nm_kg_r", 0.0))
                    self.tp.sendValue("moment_nm_kg_l", r.extra.get("moment_nm_kg_l", 0.0))
                    self.tp.sendValue("hip_enc_vel_r", r.extra.get("hip_encoder_vel_r", 0.0))
                    self.tp.sendValue("hip_enc_vel_l", r.extra.get("hip_encoder_vel_l", 0.0))
                    self.tp.sendValue("roop_time", step_end - step_start)
                except Exception:
                    pass

            if self.current_idx < len(self.data_log["time"]):
                self.data_log["time"][self.current_idx] = actual_time
                self.data_log["hip_angle_r"][self.current_idx] = pos_R
                self.data_log["hip_angle_l"][self.current_idx] = -pos_L
                self.data_log["hip_angle_r_u_gyr"][self.current_idx] = r.extra.get("hip_r_u_gyr", 0.0)
                self.data_log["hip_angle_l_u_gyr"][self.current_idx] = r.extra.get("hip_l_u_gyr", 0.0)
                self.data_log["gyro_neg_y_pelvis"][self.current_idx] = float(-imu_P[gy_idx])
                self.data_log["gyro_neg_y_thigh_r"][self.current_idx] = float(-imu_R[gy_idx])
                self.data_log["gyro_neg_y_thigh_l"][self.current_idx] = float(-imu_L[gy_idx])
                self.data_log["cmd_L"][self.current_idx] = cmd_L
                self.data_log["cmd_R"][self.current_idx] = cmd_R
                self.data_log["GPIO"][self.current_idx] = float(self.gpio.state()) if self.gpio is not None else 0.0

            self.current_idx += 1

            if self.current_idx >= len(self.data_log["time"]):
                break


_RUNNER: Optional[DualHipRunner] = None


def _handle_signal(sig, frame) -> None:
    global _RUNNER
    print(f"\nSignal {sig} received. Shutting down...")
    try:
        if _RUNNER:
            _RUNNER.shutdown()
    finally:
        sys.exit(0)


def main() -> None:
    global _RUNNER
    args = parse_args()
    cfg = load_config(args.config)
    print_config(cfg)

    runner = DualHipRunner(cfg)
    _RUNNER = runner

    atexit.register(runner.shutdown)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    gc.disable()
    try:
        runner.setup()
        runner.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received.")
    except Exception as e:
        print(f"[MAIN ERROR] {e}")
        traceback.print_exc()
    finally:
        runner.shutdown()
        gc.enable()
        gc.collect()
        print("Preparing data for saving...")
        for key in runner.data_log.keys():
            runner.data_log[key] = runner.data_log[key][: runner.current_idx]
        np.savez(runner.cfg["trial_name"], **runner.data_log)
        print(f"=== Saving data for trial: {runner.cfg['trial_name']} ===")


if __name__ == "__main__":
    main()

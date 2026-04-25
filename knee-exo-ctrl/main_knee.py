#!/usr/bin/env python3
"""
Dual-knee exoskeleton control loop (os_kinetics).

Uses ``controllers/ik_id_knee*.py`` with encoder angle (Teensy **degrees** in-process,
converted to **rad** for the model) and configurable joint velocity (encoder or IMU);
trial ``.npz`` logs knee angles in **rad** and rates in **rad/s** to match training.
ONNX / TRT providers are selected in YAML.

Torque pipeline: model outputs **N·m/kg** → ``moment_mass_kg`` → controller ``applied_*`` /
``extra["torque_cmd_*"]`` (**N·m, before YAML** ``scale`` **and** ``torque_limit``) → here
``cmd_*`` = ``clamp(applied * scale, ±torque_limit)`` → ``setTorque`` (left leg negated for wiring).

Hardware deps (Jetson + Teensy CAN stack) are loaded from ``jetson_teensy_python_path`` in the config.

**Parity with** ``test_knee/main_knee.py`` **(same hardware loop, different model I/O):**
CAN read/write order, ``Side`` gating, ``exo_on`` / ``scale`` / ``torque_limit``, GPIO pulse schedule,
``RateKeeper.wait()`` tick ``k`` for Teleplot decimation, Teleplot channel names (plus IK/ID-only fields),
and ``npz`` layout (angles as rad / gyros as rad/s here vs deg in legacy test logs). Mocap and Jetson
Python path are YAML-driven here; optional ``use_gpio`` and per-``side`` CAN connect skip test defaults.
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
from run_bundle import load_train_config, resolve_run_dir
from utils.post_mortem_log import json_safe_for_log, write_post_mortem_json
from utils.rate_keeper import RateKeeper
from utils.teleplot import Teleplot

COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"

HIGHLIGHT_KEYS = {
    "exo_on",
    "scale",
    "controller_name",
    "exp_time_sec",
    "run_dir",
    "side",
    "joint_velocity_source",
    "moment_mass_kg",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual knee exoskeleton control (os_kinetics knee-exo-ctrl).")
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
    """
    Per-sample arrays saved to ``.npz`` (same SI units as the IK/ID model):

    - ``knee_angle_r``, ``knee_angle_l``: motor encoder knee angle **radians**
      (``deg2rad(pos_R)``, ``deg2rad(-pos_L)`` — same convention as ``IkIdKnee*Controller`` raw ``q``).
    - ``knee_angle_*_u_gyr``: IMU-derived knee rate **rad/s** (controller ``extra``).
    - ``gyro_thigh_r``, ``gyro_shank_r``: IMU z gyro **rad/s** (from ``imu_gyro_z_units`` in YAML).
    """
    log_size = int(cfg["exp_time_sec"] * cfg["fs"])
    return {
        "time": np.zeros(log_size),
        "knee_angle_r": np.zeros(log_size),
        "knee_angle_l": np.zeros(log_size),
        "knee_angle_r_u_gyr": np.zeros(log_size),
        "knee_angle_l_u_gyr": np.zeros(log_size),
        "gyro_thigh_r": np.zeros(log_size),
        "gyro_shank_r": np.zeros(log_size),
        "cmd_L": np.zeros(log_size),
        "cmd_R": np.zeros(log_size),
        "K_r": np.zeros(log_size),
        "Soft_ctrl_r": np.zeros(log_size),
        "K_l": np.zeros(log_size),
        "Soft_ctrl_l": np.zeros(log_size),
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


def imu_gyro_z_to_rad_s_scale(cfg: dict) -> float:
    """Scale raw IMU gyro z (per ``imu_gyro_z_units``) to rad/s for logging / teleplot."""
    units = str(cfg.get("imu_gyro_z_units", "deg_per_s")).lower().replace(" ", "")
    if units in ("deg/s", "deg_per_s", "dps"):
        return float(np.deg2rad(1.0))
    if units in ("rad/s", "rad_per_s", "rps"):
        return 1.0
    print(f"[WARN] Unknown imu_gyro_z_units={cfg.get('imu_gyro_z_units')!r}; assuming deg/s → rad/s.")
    return float(np.deg2rad(1.0))


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


def _describe_inference_backend(cfg: dict) -> str:
    controller_name = str(cfg.get("controller_name", ""))
    if controller_name == "ik_id_knee_onnx":
        providers = [str(p) for p in cfg.get("onnx_providers", ["CPUExecutionProvider"])]
        if any(p.lower() == "tensorrtexecutionprovider" for p in providers):
            return f"ONNX Runtime TensorRT path ({providers})"
        return f"ONNX Runtime non-TRT path ({providers})"
    return controller_name or "unknown"


def latest_pos_vel(dev) -> Tuple[float, float]:
    pos = float(dev.Motor_pos_data[-1]) if dev.Motor_pos_data else 0.0
    vel = float(dev.Motor_vel_data[-1]) if dev.Motor_vel_data else 0.0
    return pos, vel


class DualKneeRunner:
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

        # Post-mortem JSON (see utils/post_mortem_log.py); filled by main loop / main().
        self._pm_written = False
        self._pm_exit_reason = "unknown"
        self._pm_exception_message: Optional[str] = None
        self._pm_exception_traceback: Optional[str] = None
        self._pm_run_start_perf: Optional[float] = None
        self._pm_wall_iso_start: Optional[str] = None
        self._loop_timing = {"n": 0, "sum_dt": 0.0, "min_dt": float("inf"), "max_dt": 0.0}
        self._last_tick: dict = {}

        self._gyro_z_to_rad_s = imu_gyro_z_to_rad_s_scale(cfg)

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

        side = Side(self.cfg["side"])
        use_left = side in (Side.LEFT, Side.BOTH)
        use_right = side in (Side.RIGHT, Side.BOTH)

        if use_left:
            self.left_dev = Device(teleplot_ip=self.cfg["teleplot_ip"], teleplot_port=int(self.cfg["teleplot_port"]))
            self.left_exo = JetsonCanInterface(
                device_storage=self.left_dev,
                channel=self.cfg["can_channel"],
                teensy_id=self.cfg["teensy_id_left"],
            )
            print(f"Initializing Left Exoskeleton (ID {hex(int(self.cfg['teensy_id_left']))})...")
            self.left_exo.connect()
        else:
            self.left_dev = None
            self.left_exo = None
            print("[Hardware] side does not include left; skipping left CAN device.")

        if use_right:
            self.right_dev = Device(teleplot_ip=self.cfg["teleplot_ip"], teleplot_port=int(self.cfg["teleplot_port"]))
            self.right_exo = JetsonCanInterface(
                device_storage=self.right_dev,
                channel=self.cfg["can_channel"],
                teensy_id=self.cfg["teensy_id_right"],
            )
            print(f"Initializing Right Exoskeleton (ID {hex(int(self.cfg['teensy_id_right']))})...")
            self.right_exo.connect()
        else:
            self.right_dev = None
            self.right_exo = None
            print("[Hardware] side does not include right; skipping right CAN device.")

        # Causal Butterworth on model inputs/outputs (defaults from training ``config.json``).
        try:
            _rd = resolve_run_dir(str(self.cfg["run_dir"]))
            _tc = load_train_config(_rd)
        except Exception:
            _tc = {}
        self.cfg.setdefault("model_io_lowpass_cutoff_hz", float(_tc.get("lowpass_cutoff_hz", 4.0)))
        self.cfg.setdefault("model_io_lowpass_order", int(_tc.get("lowpass_order", 4)))
        self.cfg.setdefault("model_io_lowpass_enable", True)
        print(
            f"[Model I/O LPF] causal Butterworth {self.cfg['model_io_lowpass_order']}th-order "
            f"@ {self.cfg['model_io_lowpass_cutoff_hz']} Hz on q, qd (pre-norm) and moments (post-norm); "
            f"enable={self.cfg['model_io_lowpass_enable']}"
        )

        self.controller = build_controller(self.cfg["controller_name"], config=self.cfg)
        self.controller.start()
        backend_desc = _describe_inference_backend(self.cfg)
        print(f"[Inference] backend: {backend_desc}")
        if (
            str(self.cfg.get("controller_name", "")).lower() == "ik_id_knee_onnx"
            and "tensorrtexecutionprovider"
            not in {str(p).lower() for p in self.cfg.get("onnx_providers", [])}
        ):
            print("[Inference][WARN] TensorRT provider is not enabled; this run is not using TRT.")

        try:
            t0 = time.perf_counter()
            if self.left_exo is not None:
                self.left_exo.set_reference_time(t0)
            if self.right_exo is not None:
                self.right_exo.set_reference_time(t0)
        except Exception:
            pass

        print("\n--- Dual Knee Exo Control Loop Started (os_kinetics) ---")
        print(f"Teleplot: {self.cfg['teleplot_ip']}:{self.cfg['teleplot_port']}")
        print(
            "[Data log] knee_angle_r/l = encoder rad; knee_angle_*_u_gyr = rad/s; "
            "gyro_thigh/shank_r = rad/s (per imu_gyro_z_units)."
        )
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
        self._pm_run_start_perf = t0
        self._pm_wall_iso_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        prev_loop_time: Optional[float] = None

        trial_start_sec = 0.0
        trial_dur_sec = float(self.cfg["exp_time_sec"])
        pulse_after_start = float(self.cfg["GPIO_START_DELAY_SEC"])

        first_pulse_sent = False
        first_pulse_end: Optional[float] = None
        second_pulse_sent = False
        second_pulse_end: Optional[float] = None

        gz_idx = int(self.cfg.get("teleplot_gyro_z_index", 5))
        # Match ``test_knee/main_knee.py``: optional decimation of Teleplot sends only (not CAN / npz).
        log_divider = max(1, int(self.cfg.get("teleplot_log_divider", 1)))

        while True:
            _, _, k = rk.wait()
            loop_now = time.perf_counter()
            step_start = loop_now
            if prev_loop_time is None:
                loop_dt = 0.0
            else:
                loop_dt = loop_now - prev_loop_time
                lt = self._loop_timing
                lt["n"] = int(lt["n"]) + 1
                lt["sum_dt"] = float(lt["sum_dt"]) + float(loop_dt)
                lt["min_dt"] = min(float(lt["min_dt"]), float(loop_dt))
                lt["max_dt"] = max(float(lt["max_dt"]), float(loop_dt))
            prev_loop_time = loop_now

            use_left = side in (Side.LEFT, Side.BOTH)
            use_right = side in (Side.RIGHT, Side.BOTH)

            if use_left:
                self.left_exo.getParameters()
            if use_right:
                self.right_exo.getParameters()

            pos_L = vel_L = 0.0
            pos_R = vel_R = 0.0
            imu_L1 = imu_L2 = imu6_zeros()
            imu_R1 = imu_R2 = imu6_zeros()

            if use_left:
                pos_L, vel_L = latest_pos_vel(self.left_dev)
                imu_L1 = pack_imu6_from_device(self.left_dev, which=1)
                imu_L2 = pack_imu6_from_device(self.left_dev, which=2)
            if use_right:
                pos_R, vel_R = latest_pos_vel(self.right_dev)
                imu_R1 = pack_imu6_from_device(self.right_dev, which=1)
                imu_R2 = pack_imu6_from_device(self.right_dev, which=2)

            s = Sensors(
                imu_L1=imu_L1,
                imu_L2=imu_L2,
                imu_R1=imu_R1,
                imu_R2=imu_R2,
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

            # ``r.extra["torque_cmd_*"]`` is N·m after mass only (``sign * m * moment_mass_kg``), *before*
            # ``scale`` / ``torque_limit``. These fields match ``cmd_*`` in the trial ``.npz`` / Teleplot ``cmd_*``.
            try:
                r.extra["exo_cmd_torque_l_n_m"] = float(cmd_L)
                r.extra["exo_cmd_torque_r_n_m"] = float(cmd_R)
            except Exception:
                pass

            if use_left:
                self.left_exo.setTorque(-cmd_L)
            if use_right:
                self.right_exo.setTorque(cmd_R)

            # Match ``test_knee/main_knee.py``: ``now`` / GPIO after ``setTorque``; wall time for log/post-mortem.
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

            actual_time = step_start - t0
            self._last_tick = {
                "t_s": float(actual_time),
                "cmd_L": float(cmd_L),
                "cmd_R": float(cmd_R),
                "applied_L": float(r.applied_L),
                "applied_R": float(r.applied_R),
                "model_out_L": float(r.model_out_L),
                "model_out_R": float(r.model_out_R),
                "extra": json_safe_for_log(dict(r.extra)),
            }

            step_end = time.perf_counter()

            # Teleplot channel set aligned with ``test_knee/main_knee.py`` (plus IK/ID extras).
            if self.tp is not None and (k % log_divider) == 0:
                try:
                    gpio_state = float(self.gpio.state()) if self.gpio is not None else 0.0
                    self.tp.sendValue("cmd_R", cmd_R)
                    self.tp.sendValue("cmd_L", cmd_L)
                    self.tp.sendValue("knee_angle_r_u_gyr", r.extra.get("knee_r_u_gyr", 0.0))
                    self.tp.sendValue("knee_angle_l_u_gyr", r.extra.get("knee_l_u_gyr", 0.0))
                    self.tp.sendValue("knee_angle_r_u", r.extra.get("knee_angle_r_u", 0.0))
                    self.tp.sendValue("knee_angle_l_u", r.extra.get("knee_angle_l_u", 0.0))
                    self.tp.sendValue("knee_angle_r", r.extra.get("knee_angle_r", 0.0))
                    self.tp.sendValue("knee_angle_l", r.extra.get("knee_angle_l", 0.0))
                    self.tp.sendValue("Soft_ctrl_r", r.extra.get("Soft_ctrl_r", 0.0))
                    self.tp.sendValue("Soft_ctrl_l", r.extra.get("Soft_ctrl_l", 0.0))
                    self.tp.sendValue("assist_gate_r", r.extra.get("assist_gate_r", 0.0))
                    self.tp.sendValue("assist_gate_l", r.extra.get("assist_gate_l", 0.0))
                    self.tp.sendValue("state_l", r.extra.get("state_l", 0.0))
                    self.tp.sendValue("GPIO", gpio_state)
                    gz = self._gyro_z_to_rad_s
                    self.tp.sendValue("gyro_thigh_r", float(imu_R1[gz_idx]) * gz)
                    self.tp.sendValue("gyro_thigh_l", float(-imu_L1[gz_idx]) * gz)
                    self.tp.sendValue("gyro_shank_r", float(imu_R2[gz_idx]) * gz)
                    self.tp.sendValue("gyro_shank_l", float(-imu_L2[gz_idx]) * gz)
                    self.tp.sendValue("moment_nm_kg_r", r.extra.get("moment_nm_kg_r", 0.0))
                    self.tp.sendValue("moment_nm_kg_l", r.extra.get("moment_nm_kg_l", 0.0))
                    self.tp.sendValue("knee_enc_vel_r", r.extra.get("knee_encoder_vel_r", 0.0))
                    self.tp.sendValue("knee_enc_vel_l", r.extra.get("knee_encoder_vel_l", 0.0))
                    self.tp.sendValue("K_l", r.extra.get("K_l", 0.0))
                    self.tp.sendValue("roop_time", step_end - step_start)
                except Exception:
                    pass

            if self.current_idx < len(self.data_log["time"]):
                self.data_log["time"][self.current_idx] = actual_time
                # Encoder: Teensy ``pos_*`` is degrees → log radians (model / controller convention).
                self.data_log["knee_angle_r"][self.current_idx] = float(np.deg2rad(pos_R))
                self.data_log["knee_angle_l"][self.current_idx] = float(np.deg2rad(-pos_L))
                self.data_log["knee_angle_r_u_gyr"][self.current_idx] = r.extra.get("knee_r_u_gyr", 0.0)
                self.data_log["knee_angle_l_u_gyr"][self.current_idx] = r.extra.get("knee_l_u_gyr", 0.0)
                gz = self._gyro_z_to_rad_s
                self.data_log["gyro_thigh_r"][self.current_idx] = float(imu_R1[gz_idx]) * gz
                self.data_log["gyro_shank_r"][self.current_idx] = float(imu_R2[gz_idx]) * gz
                self.data_log["cmd_L"][self.current_idx] = cmd_L
                self.data_log["cmd_R"][self.current_idx] = cmd_R
                self.data_log["K_r"][self.current_idx] = r.extra.get("K_r", 0.0)
                self.data_log["Soft_ctrl_r"][self.current_idx] = r.extra.get("Soft_ctrl_r", 0.0)
                self.data_log["K_l"][self.current_idx] = r.extra.get("K_l", 0.0)
                self.data_log["Soft_ctrl_l"][self.current_idx] = r.extra.get("Soft_ctrl_l", 0.0)
                self.data_log["GPIO"][self.current_idx] = float(self.gpio.state()) if self.gpio is not None else 0.0

            self.current_idx += 1

            if self.current_idx >= len(self.data_log["time"]):
                self._pm_exit_reason = "completed"
                break


_RUNNER: Optional[DualKneeRunner] = None


def _atexit_post_mortem() -> None:
    global _RUNNER
    try:
        if _RUNNER is not None and not getattr(_RUNNER, "_pm_written", False):
            write_post_mortem_json(
                _RUNNER,
                npz_path=None,
                note="atexit_fallback; main finally may not have run — npz may be missing or partial",
            )
    except Exception:
        pass


def _handle_signal(sig, frame) -> None:
    global _RUNNER
    print(f"\nSignal {sig} received. Shutting down...")
    try:
        if _RUNNER:
            _RUNNER._pm_exit_reason = f"signal:{int(sig)}"
            _RUNNER.shutdown()
    finally:
        sys.exit(0)


def main() -> None:
    global _RUNNER
    args = parse_args()
    cfg = load_config(args.config)
    print_config(cfg)

    runner = DualKneeRunner(cfg)
    _RUNNER = runner

    atexit.register(_atexit_post_mortem)
    atexit.register(runner.shutdown)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    gc.disable()
    try:
        runner.setup()
        runner.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received.")
        runner._pm_exit_reason = "keyboard_interrupt"
    except Exception as e:
        runner._pm_exit_reason = "exception"
        runner._pm_exception_message = str(e)
        runner._pm_exception_traceback = traceback.format_exc()
        print(f"[MAIN ERROR] {e}")
        traceback.print_exc()
    finally:
        runner.shutdown()
        gc.enable()
        gc.collect()
        print("Preparing data for saving...")
        for key in runner.data_log.keys():
            runner.data_log[key] = runner.data_log[key][: runner.current_idx]
        trial_key = runner.cfg["trial_name"]
        npz_path = Path(f"{trial_key}.npz").resolve()
        np.savez(str(npz_path), **runner.data_log)
        print(f"=== Saving data for trial: {trial_key} ===")
        write_post_mortem_json(runner, npz_path=str(npz_path))


if __name__ == "__main__":
    main()

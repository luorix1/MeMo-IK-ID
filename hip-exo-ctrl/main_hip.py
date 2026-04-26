"""main_hip.py — Hip Exoskeleton Control Loop

Usage
-----
    python main_hip.py cfg/hip_biotorque_default.yaml

Hardware
--------
  Motors  : AK80-9 (TMotorV3) via epicpower_tmotorV3, CAN bus
  IMUs    : ICM20948 (I2C) — pelvis, left thigh, right thigh
  GPIO    : Jetson board pin 7 for sync pulses

Sign convention (motor)
-----------------------
  LEFT  motor : flexion = positive torque / positive position
  RIGHT motor : flexion = NEGATIVE (motor reads opposite sense), so we negate
                position, velocity, and torque command before/after the motor.

Config keys (YAML)
------------------
  See cfg/hip_biotorque_default.yaml for the full reference.
"""

import atexit
import gc
import os
import signal
import sys
import time
import traceback
from typing import Optional

import can
import multiprocessing as mp
import numpy as np
import pandas as pd
import yaml

from epicpower_tmotorV3.actuator_group import ActuatorGroup
from epicpower_tmotorV3.tmotor_v3 import TMotorV3
from Header_ICM20948_I2C_pcb2 import ICM20948_I2C_IMUs

from controllers import build_controller
from controllers.base import Sensors
from utils.gpio_control import GPIOControl
from utils.Header_Mocap_trigger import Mocap_trigger
from utils.teleplot import Teleplot
from utils.utils import RateKeeper

COLOR_GREEN  = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET  = "\033[0m"

HIGHLIGHT_KEYS = {
    "exo_on", "scale_factor", "controller_name",
    "exp_time_sec", "mass", "desired_delay_ms",
}


# ======================================================================
# Config helpers
# ======================================================================
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Hip exoskeleton control loop.")
    parser.add_argument("config", help="Path to YAML config file.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg).__name__}")
    cfg["config_path"] = os.path.abspath(path)
    return cfg


def print_config(cfg: dict) -> None:
    print("=== CONFIG ===")
    for k, v in cfg.items():
        if k in HIGHLIGHT_KEYS:
            print(f"{COLOR_GREEN}{k}: {v}{COLOR_RESET}")
        else:
            print(f"{k}: {v}")
    input("\n==== Check Config ====\nHit Enter to continue...")


# ======================================================================
# Data log helpers
# ======================================================================
def build_data_log(cfg: dict) -> dict:
    n = int(cfg["exp_time_sec"] * cfg["fs"]) + int(5 * cfg["fs"])
    f1 = lambda: np.full(n, np.nan, dtype=np.float32)
    f6 = lambda: np.full((n, 6), np.nan, dtype=np.float32)
    return {
        "timestamp":        f1(),
        "mtr_pos_L":        f1(), "mtr_pos_R":        f1(),
        "mtr_vel_L":        f1(), "mtr_vel_R":        f1(),
        "mtr_cmd_L":        f1(), "mtr_cmd_R":        f1(),
        "actual_torque_L":  f1(), "actual_torque_R":  f1(),
        "imu_P":            f6(), "imu_L":            f6(), "imu_R": f6(),
        "model_output_L":   f1(), "model_output_R":   f1(),
        "net_torque_L":     f1(), "net_torque_R":     f1(),
        "bio_torque_L":     f1(), "bio_torque_R":     f1(),
        "scaled_torque_L":  f1(), "scaled_torque_R":  f1(),
        "delayed_torque_L": f1(), "delayed_torque_R": f1(),
        "filtered_torque_L":f1(), "filtered_torque_R":f1(),
        "gpio_output":      f1(),
    }


def save_data(log: dict, cfg: dict, valid_len: int) -> None:
    trial = cfg["trial_name"]
    n = valid_len
    if n == 0:
        print("[save_data] No data collected — skipping.")
        return

    t = log["timestamp"][:n]
    t_rel = t - t[0]

    # Motor
    df_mtr = pd.DataFrame({
        "time":      t_rel,
        "mtr_pos_L": log["mtr_pos_L"][:n],
        "mtr_pos_R": log["mtr_pos_R"][:n],
        "mtr_vel_L": log["mtr_vel_L"][:n],
        "mtr_vel_R": log["mtr_vel_R"][:n],
        "gpio_output": log["gpio_output"][:n],
    })
    df_mtr.to_csv(f"{trial}_input_motor.csv", index=False)
    print(f"Saved {trial}_input_motor.csv  {df_mtr.shape}")

    # IMU
    imu_data = {"time": t_rel}
    for key, prefix in [("imu_P", "Pelvis"), ("imu_L", "Thigh_L"), ("imu_R", "Thigh_R")]:
        arr = log[key][:n, :]
        for ci, ch in enumerate(["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]):
            imu_data[f"{prefix}_{ch}"] = arr[:, ci]
    imu_data["gpio_output"] = log["gpio_output"][:n]
    df_imu = pd.DataFrame(imu_data)
    df_imu.to_csv(f"{trial}_input_imu.csv", index=False)
    print(f"Saved {trial}_input_imu.csv  {df_imu.shape}")

    # Torque pipeline
    torque_keys = [
        "model_output_L", "model_output_R",
        "net_torque_L",   "net_torque_R",
        "bio_torque_L",   "bio_torque_R",
        "scaled_torque_L","scaled_torque_R",
        "delayed_torque_L","delayed_torque_R",
        "filtered_torque_L","filtered_torque_R",
        "mtr_cmd_L", "mtr_cmd_R",
        "actual_torque_L", "actual_torque_R",
        "gpio_output",
    ]
    df_torque = pd.DataFrame({"time": t_rel, **{k: log[k][:n] for k in torque_keys}})
    df_torque.to_csv(f"{trial}_output_torque.csv", index=False)
    print(f"Saved {trial}_output_torque.csv  {df_torque.shape}")


# ======================================================================
# Runner
# ======================================================================
class DualHipRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_log = build_data_log(cfg)
        self.current_idx = 0

        self.mtr_comms: Optional[ActuatorGroup] = None
        self.imus: Optional[ICM20948_I2C_IMUs] = None
        self.bus: Optional[can.Bus] = None
        self.notifier: Optional[can.Notifier] = None
        self.gpio: Optional[GPIOControl] = None
        self.tp: Optional[Teleplot] = None
        self.controller = None
        self.mocap_trigger = None

    # ------------------------------------------------------------------
    def setup(self):
        # Mocap trigger client (connect early so it's ready)
        if self.cfg["trigger_type"] == "mocap":
            try:
                self.mocap_trigger = Mocap_trigger(
                    server_ip=self.cfg["mocap_server_ip"],
                    port_number=self.cfg["mocap_server_port"],
                )
                self.mocap_trigger.start_client()
            except Exception as e:
                print(f"[Mocap] start_client error: {e}")

        # GPIO
        self.gpio = GPIOControl(self.cfg["gpio_output_pin"])

        # Teleplot
        self.tp = Teleplot(self.cfg["teleplot_ip"], self.cfg["teleplot_port"])

        # Motors
        _ = input("Press Enter to initialize motors: ")
        cid_L = self.cfg["can_id_L"]
        cid_R = self.cfg["can_id_R"]
        mtr_type = self.cfg["motor_type"]
        init_list = [TMotorV3(cid_L, mtr_type), TMotorV3(cid_R, mtr_type)]
        self.mtr_comms = ActuatorGroup(init_list)

        # CAN bus (for notifier / background listener)
        self.bus = can.Bus(interface="socketcan", channel=self.cfg["can_channel"])
        self.notifier = can.Notifier(self.bus, [])

        # IMUs
        self.imus = ICM20948_I2C_IMUs()

        # Controller
        self.controller = build_controller(
            self.cfg["controller_name"], config=self.cfg
        )
        self.controller.start()

        print("\n--- Hip Exo Control Loop Ready ---")
        print(f"Teleplot: {self.cfg['teleplot_ip']}:{self.cfg['teleplot_port']}")

    # ------------------------------------------------------------------
    def shutdown(self):
        cid_L = self.cfg.get("can_id_L", 1)
        cid_R = self.cfg.get("can_id_R", 2)
        try:
            if self.mtr_comms:
                self.mtr_comms.set_torque(cid_L, 0.0)
                self.mtr_comms.set_torque(cid_R, 0.0)
        except Exception as e:
            print(f"[Shutdown] zero-torque failed: {e}")
        time.sleep(0.05)

        try:
            if self.notifier: self.notifier.stop()
            if self.bus:      self.bus.shutdown()
            print("CAN bus closed.")
        except Exception as e:
            print(f"[Shutdown] CAN cleanup: {e}")

        try:
            if self.controller: self.controller.close()
        except Exception as e:
            print(f"[Shutdown] controller.close: {e}")

        try:
            if self.gpio:
                self.gpio.pulse_end()
                self.gpio.close()
        except Exception as e:
            print(f"[Shutdown] GPIO cleanup: {e}")

        try:
            if self.tp: self.tp.close()
        except Exception:
            pass

        gc.enable()
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

        print("Shutdown complete.")

    # ------------------------------------------------------------------
    def run(self):
        cfg = self.cfg
        cid_L = cfg["can_id_L"]
        cid_R = cfg["can_id_R"]
        fs = cfg["fs"]
        exp_time = cfg["exp_time_sec"]
        exo_on = cfg["exo_on"]
        torque_limit = cfg["torque_limit"]
        pulse_delay_sec = cfg.get("gpio_start_delay_sec", 3.0)
        pulse_width_sec = cfg.get("pulse_width_sec", 0.2)
        target_time_range = cfg.get("target_time_range_sec", exp_time)

        # Trigger
        if cfg["trigger_type"] == "typing":
            input("Wait for TensorRT to warm up...\nPress Enter to start trial.\n")
        elif cfg["trigger_type"] == "mocap":
            print("Waiting for mocap trigger...")
            if self.mocap_trigger:
                self.mocap_trigger.wait_for_trigger()
            else:
                print("[WARN] No mocap trigger — starting immediately.")
        else:
            raise NotImplementedError(f"Unknown trigger_type: {cfg['trigger_type']}")

        rk = RateKeeper(fs)
        rk.start()
        t0 = time.perf_counter()

        first_pulse_sent = False
        first_pulse_end = None
        second_pulse_sent = False
        second_pulse_end = None

        max_samples = len(self.data_log["timestamp"])

        while True:
            overrun, t_sched, k = rk.wait()
            now = time.perf_counter() - t0

            if self.current_idx >= max_samples:
                print("Buffer full — stopping.")
                break

            idx = self.current_idx

            # ── Sensor reads ──────────────────────────────────────────
            pos_L, vel_L, torque_L = (
                self.mtr_comms.get_position(cid_L, degrees=True),
                self.mtr_comms.get_velocity(cid_L, degrees=True),
                self.mtr_comms.get_torque(cid_L),
            )
            pos_R_raw, vel_R_raw, torque_R_raw = (
                self.mtr_comms.get_position(cid_R, degrees=True),
                self.mtr_comms.get_velocity(cid_R, degrees=True),
                self.mtr_comms.get_torque(cid_R),
            )
            # Negate right-side readings to match left-referenced convention
            pos_R  = -pos_R_raw
            vel_R  = -vel_R_raw
            torque_R = -torque_R_raw

            imu_dict = self.imus.read_IMUs()
            imu_P = imu_dict["IMU_PELVIS"]
            imu_L = imu_dict["IMU_THIGH_LEFT"]
            imu_R = imu_dict["IMU_THIGH_RIGHT"]

            # ── Controller step ───────────────────────────────────────
            s = Sensors(
                imu_P=imu_P, imu_L=imu_L, imu_R=imu_R,
                pos_L=pos_L, pos_R=pos_R,
                vel_L=vel_L, vel_R=vel_R,
            )
            r = self.controller.step(s)

            cmd_L = r.applied_L if exo_on else 0.0
            cmd_R = r.applied_R if exo_on else 0.0

            # Apply torque (right motor sign flipped back at hardware level)
            self.mtr_comms.set_torque(cid_L,  cmd_L)
            self.mtr_comms.set_torque(cid_R, -cmd_R)

            actual_L =  self.mtr_comms.get_torque(cid_L)
            actual_R = -self.mtr_comms.get_torque(cid_R)

            # ── GPIO pulses ────────────────────────────────────────────
            if (not first_pulse_sent) and (now >= pulse_delay_sec):
                self.gpio.pulse_start()
                first_pulse_sent = True
                first_pulse_end = now + pulse_width_sec

            if first_pulse_sent and first_pulse_end and now >= first_pulse_end:
                self.gpio.pulse_end()
                first_pulse_end = None

            if (not second_pulse_sent) and (now >= pulse_delay_sec + target_time_range):
                self.gpio.pulse_start()
                second_pulse_sent = True
                second_pulse_end = now + pulse_width_sec

            if second_pulse_sent and second_pulse_end and now >= second_pulse_end:
                self.gpio.pulse_end()
                second_pulse_end = None

            # ── Data log ───────────────────────────────────────────────
            ex = r.extra
            self.data_log["timestamp"][idx]        = now
            self.data_log["mtr_pos_L"][idx]        = pos_L
            self.data_log["mtr_pos_R"][idx]        = pos_R
            self.data_log["mtr_vel_L"][idx]        = vel_L
            self.data_log["mtr_vel_R"][idx]        = vel_R
            self.data_log["mtr_cmd_L"][idx]        = cmd_L
            self.data_log["mtr_cmd_R"][idx]        = cmd_R
            self.data_log["actual_torque_L"][idx]  = actual_L
            self.data_log["actual_torque_R"][idx]  = actual_R
            self.data_log["imu_P"][idx, :]         = imu_P
            self.data_log["imu_L"][idx, :]         = imu_L
            self.data_log["imu_R"][idx, :]         = imu_R
            self.data_log["model_output_L"][idx]   = r.model_out_L
            self.data_log["model_output_R"][idx]   = r.model_out_R
            self.data_log["net_torque_L"][idx]     = ex.get("net_L", 0.0)
            self.data_log["net_torque_R"][idx]     = ex.get("net_R", 0.0)
            self.data_log["bio_torque_L"][idx]     = ex.get("bio_L", 0.0)
            self.data_log["bio_torque_R"][idx]     = ex.get("bio_R", 0.0)
            self.data_log["scaled_torque_L"][idx]  = ex.get("scaled_L", 0.0)
            self.data_log["scaled_torque_R"][idx]  = ex.get("scaled_R", 0.0)
            self.data_log["delayed_torque_L"][idx] = ex.get("delayed_L", 0.0)
            self.data_log["delayed_torque_R"][idx] = ex.get("delayed_R", 0.0)
            self.data_log["filtered_torque_L"][idx]= ex.get("filtered_L", 0.0)
            self.data_log["filtered_torque_R"][idx]= ex.get("filtered_R", 0.0)
            self.data_log["gpio_output"][idx]      = float(self.gpio.state())

            # ── Teleplot ───────────────────────────────────────────────
            self.tp.sendBatch({
                "time":            now,
                "loop_overrun_ms": overrun * 1000,
                "cmd_L":           cmd_L,
                "cmd_R":           cmd_R,
                "actual_L":        actual_L,
                "actual_R":        actual_R,
                "model_out_L":     r.model_out_L,
                "model_out_R":     r.model_out_R,
                "scaled_L":        ex.get("scaled_L", 0.0),
                "scaled_R":        ex.get("scaled_R", 0.0),
                "delayed_L":       ex.get("delayed_L", 0.0),
                "delayed_R":       ex.get("delayed_R", 0.0),
                "filtered_L":      ex.get("filtered_L", 0.0),
                "filtered_R":      ex.get("filtered_R", 0.0),
                "gpio":            self.gpio.state(),
            })

            self.current_idx += 1


# ======================================================================
# Signal handling + entry point
# ======================================================================
_RUNNER: Optional[DualHipRunner] = None


def _handle_signal(sig, frame):
    global _RUNNER
    print(f"\nSignal {sig} received. Shutting down...")
    if _RUNNER:
        try:
            _RUNNER.shutdown()
        except Exception:
            pass
    sys.exit(0)


def main():
    global _RUNNER
    args = parse_args()
    cfg = load_config(args.config)
    print_config(cfg)

    gc.disable()
    mp.set_start_method("spawn", force=True)

    runner = DualHipRunner(cfg)
    _RUNNER = runner

    atexit.register(runner.shutdown)
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

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
        gc.enable()
        gc.collect()
        valid = runner.current_idx
        print(f"Saving {valid} samples...")
        save_data(runner.data_log, cfg, valid)


if __name__ == "__main__":
    main()

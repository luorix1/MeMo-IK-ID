"""Ankle exo control loop — encoder position + time derivative (no Muse yet).

Hardware: Robstride RS-0x via vendored ``robstride_dynamics`` (same stack as
``Ankle_Exo/``). MIT-mode torque: kp=kd=0, feedforward torque only.
Controller I/O: ``Sensors`` / ``CtrlResult`` (see ``controllers/base.py``).

Run:
  python main_ankle.py cfg/bringup.yaml
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
from typing import Optional, Tuple

import numpy as np
import yaml

from controllers import build_controller
from controllers.base import CtrlResult, Sensors
from utils.teleplot import Teleplot
from utils.utils import RateKeeper

COLOR_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"

HIGHLIGHT_KEYS = {
    "exo_on",
    "scale",
    "controller_name",
    "exp_time_sec",
    "K",
    "B",
    "mass",
    "torque_limit",
    "vel_source",
    "side",
    "trt_engine_path",
}

try:
    import Jetson.GPIO as GPIO  # type: ignore

    _HAS_GPIO = True
except Exception:
    GPIO = None
    _HAS_GPIO = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dual ankle exoskeleton control loop.")
    parser.add_argument("config", help="Path to YAML config file.")
    return parser.parse_args()


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
    input("\n====Check Config====\nHit Enter to continue...")


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def imu6_zeros() -> np.ndarray:
    return np.zeros((6,), dtype=np.float32)


def build_data_log(cfg: dict) -> dict:
    log_size = int(cfg["exp_time_sec"] * cfg["fs"])
    return {
        "time": np.zeros(log_size),
        "ankle_pos_L": np.zeros(log_size),
        "ankle_pos_R": np.zeros(log_size),
        "ankle_vel_L": np.zeros(log_size),
        "ankle_vel_R": np.zeros(log_size),
        "cmd_L": np.zeros(log_size),
        "cmd_R": np.zeros(log_size),
        "model_out_L": np.zeros(log_size),
        "model_out_R": np.zeros(log_size),
        "model_in_ankle_angle_raw_r": np.zeros(log_size),
        "model_in_ankle_angle_raw_l": np.zeros(log_size),
        "model_in_ankle_vel_raw_r": np.zeros(log_size),
        "model_in_ankle_vel_raw_l": np.zeros(log_size),
        "model_in_ankle_angle_lpf": np.zeros(log_size),
        "model_in_ankle_vel_lpf": np.zeros(log_size),
        "model_out_nmpkg": np.zeros(log_size),
        "model_out_nmpkg_raw": np.zeros(log_size),
        "moment_raw": np.zeros(log_size),
        "assist_gate": np.zeros(log_size),
        "motion_score": np.zeros(log_size),
        "state": np.zeros(log_size),
        "GPIO": np.zeros(log_size),
        "loop_dt": np.zeros(log_size),
    }


class Side(enum.Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class GPIOControl:
    def __init__(self, pin: int):
        self.pin = int(pin)
        self._state = 0
        self._enabled = False
        if not _HAS_GPIO:
            print("[GPIO] Jetson.GPIO not available — sync pulses disabled.")
            return
        try:
            GPIO.setwarnings(False)
            GPIO.cleanup()
        except Exception:
            pass
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        self._enabled = True

    def pulse_start(self):
        if not self._enabled:
            return
        GPIO.output(self.pin, GPIO.HIGH)
        self._state = 1

    def pulse_end(self):
        if not self._enabled:
            return
        GPIO.output(self.pin, GPIO.LOW)
        self._state = 0

    def state(self) -> int:
        return self._state

    def close(self):
        if not self._enabled:
            return
        try:
            GPIO.output(self.pin, GPIO.LOW)
            GPIO.cleanup()
        except Exception:
            pass


class EncoderDiff:
    """Finite-difference velocity from successive encoder samples."""

    def __init__(self):
        self._prev_pos: Optional[float] = None
        self._prev_t: Optional[float] = None

    def reset(self):
        self._prev_pos = None
        self._prev_t = None

    def update(self, pos: float, t: float) -> float:
        if self._prev_pos is None or self._prev_t is None:
            self._prev_pos = pos
            self._prev_t = t
            return 0.0
        dt = t - self._prev_t
        vel = (pos - self._prev_pos) / dt if dt > 1e-6 else 0.0
        self._prev_pos = pos
        self._prev_t = t
        return float(vel)


class AngleZeroing:
    """Average the first ``n_frames`` ankle angles, then subtract that mean (rad)."""

    def __init__(self, n_frames: int = 100):
        self.n_frames = max(1, int(n_frames))
        self._sum = 0.0
        self._count = 0
        self.offset = 0.0
        self.done = False

    def update(self, angle_rad: float) -> float:
        """Feed one sample; return zeroed angle (0 during collection until locked)."""
        if not self.done:
            self._sum += float(angle_rad)
            self._count += 1
            if self._count >= self.n_frames:
                self.offset = self._sum / float(self._count)
                self.done = True
                print(
                    f"[Zero] ankle angle offset locked after {self._count} frames: "
                    f"{self.offset:+.4f} rad ({np.degrees(self.offset):+.2f} deg)"
                )
            # Hold model input near 0 while collecting the zero.
            return 0.0
        return float(angle_rad) - self.offset



def _pkg_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _setup_can_interface(channel: str, bitrate: int = 1_000_000) -> None:
    """Bring up SocketCAN the same way as ``Ankle_Exo/Ankle_Controller.py``."""
    cmds = [
        f"sudo ip link set {channel} down",
        (
            f"sudo ip link set {channel} type can bitrate {int(bitrate)} "
            f"restart-ms 100 berr-reporting off"
        ),
        f"sudo ip link set {channel} up",
    ]
    for c in cmds:
        rc = os.system(c)
        if rc != 0:
            raise RuntimeError(f"CAN setup failed (rc={rc}): {c}")


class RobstrideAnkleHardware:
    """Robstride RS-0x motors (MIT torque mode) — matches ``Ankle_Exo``.

    Convention from Ankle_Exo comments: encoder flexion (−), extension (+), rad.
    Torque commands use MIT operation frames with kp=kd=0.
    """

    KEY_L = "motor_L"
    KEY_R = "motor_R"

    def __init__(self, cfg: dict):
        pkg_dir = _pkg_root()
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)

        from robstride_dynamics import Motor, RobstrideBus

        self.channel = str(cfg.get("can_channel", "can0"))
        self.bitrate = int(cfg.get("can_bitrate", 1_000_000))
        self.motor_model = str(cfg.get("motor_model", "rs-02"))
        self.can_id_L = int(cfg.get("can_id_L", 2))
        self.can_id_R = int(cfg.get("can_id_R", 1))  # Ankle_Exo default motor id=1

        side = str(cfg.get("side", "right")).lower()
        motors = {}
        if side in ("left", "both"):
            motors[self.KEY_L] = Motor(id=self.can_id_L, model=self.motor_model)
        if side in ("right", "both"):
            motors[self.KEY_R] = Motor(id=self.can_id_R, model=self.motor_model)
        if not motors:
            raise ValueError(f"No motors selected for side={side!r}")

        motor_desc = ", ".join(f"{k}(id={m.id})" for k, m in motors.items())
        input(
            f"[RobstrideAnkleHardware] Press Enter to initialize {self.motor_model} "
            f"on {self.channel}: {motor_desc}..."
        )

        _setup_can_interface(self.channel, self.bitrate)
        self.bus = RobstrideBus(self.channel, motors, bitrate=self.bitrate)
        self.bus.connect()
        self._motor_keys = list(motors.keys())
        for name in self._motor_keys:
            self.bus.enable(name)
            time.sleep(0.3)  # Ankle_Controller uses 0.3s after enable

        # MIT status frames are replies to OPERATION writes. Do not leave a
        # dangling unread status from init — the control loop will prime after
        # the user hits Enter (avoids stale/lost RX across the start pause).
        for name in self._motor_keys:
            self.set_torque(name, 0.0)
            try:
                self.bus.read_operation_frame(name)
            except Exception as e:
                print(f"[RobstrideAnkleHardware] init status drain ({name}): {e}")
        time.sleep(0.05)

    def has_motor(self, key: str) -> bool:
        return key in self._motor_keys

    def motor_pos_vel(self, key: str) -> Tuple[float, float]:
        pos, vel, _tor, _temp = self.bus.read_operation_frame(key)
        return float(pos), float(vel)

    def set_torque(self, key: str, torque_nm: float) -> None:
        if key not in self._motor_keys:
            return
        self.bus.write_operation_frame(
            motor=key,
            position=0.0,
            kp=0.0,
            kd=0.0,
            velocity=0.0,
            torque=float(torque_nm),
        )

    def prime_mit(self) -> None:
        """Write zero torque so the next ``read_operation_frame`` has a status RX."""
        for name in self._motor_keys:
            self.set_torque(name, 0.0)
        time.sleep(0.01)

    def shutdown(self) -> None:
        try:
            for name in self._motor_keys:
                self.set_torque(name, 0.0)
        except Exception as e:
            print(f"[Hardware] zero torque failed: {e}")
        time.sleep(0.1)
        try:
            if self.bus is not None and self.bus.is_connected:
                self.bus.disconnect(disable_torque=True)
        except Exception as e:
            print(f"[Hardware] disconnect failed: {e}")


class DualAnkleRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_log = build_data_log(cfg)
        self.hw: Optional[RobstrideAnkleHardware] = None
        self.controller = None
        self.tp: Optional[Teleplot] = None
        self.gpio = GPIOControl(self.cfg["GPIO_OUTPUT_PIN"])
        self.mocap_trigger = None
        self.current_idx = 0
        self._diff_L = EncoderDiff()
        self._diff_R = EncoderDiff()

    def setup(self):
        if self.cfg.get("trigger_type") == "mocap":
            try:
                from utils.Header_Mocap_trigger import Mocap_trigger

                self.mocap_trigger = Mocap_trigger(
                    server_ip=str(self.cfg.get("mocap_server_ip", "172.24.44.177")),
                    port_number=int(self.cfg.get("mocap_server_port", 10)),
                )
                self.mocap_trigger.start_client()
            except Exception as e:
                print(f"[Mocap] start_client error: {e}")

        self.tp = Teleplot(self.cfg["teleplot_ip"], self.cfg["teleplot_port"])

        # Start TRT / controller process BEFORE opening SocketCAN.
        # Forking after can0 is open duplicates the CAN FD and breaks MIT RX.
        self.controller = build_controller(self.cfg["controller_name"], config=self.cfg)
        if hasattr(self.controller, "start"):
            self.controller.start()

        self.hw = RobstrideAnkleHardware(self.cfg)
        self._diff_L.reset()
        self._diff_R.reset()

        print("\n--- Ankle Exo Control Loop Started (Robstride) ---")
        print(f"Controller: {self.cfg['controller_name']}")
        print(f"Motor model: {self.cfg.get('motor_model', 'rs-02')}")
        print(f"vel_source: {self.cfg.get('vel_source', 'encoder_diff')}")
        print(f"Teleplot: {self.cfg['teleplot_ip']}:{self.cfg['teleplot_port']}")
        print("Press Ctrl+C to stop.")

    def shutdown(self):
        try:
            if self.hw is not None:
                self.hw.shutdown()
        except Exception as e:
            print(f"[Shutdown] hardware: {e}")

        try:
            if self.controller is not None:
                self.controller.close()
        except Exception as e:
            print(f"[Shutdown] controller: {e}")

        try:
            if self.gpio is not None:
                self.gpio.pulse_end()
                self.gpio.close()
        except Exception as e:
            print(f"[Shutdown] GPIO: {e}")

        try:
            if self.tp is not None:
                self.tp.close()
        except Exception:
            pass

        try:
            gc.collect()
        except Exception:
            pass
        print("System Shutdown Complete.")

    def _read_side(self, motor_key: str, diff: EncoderDiff, t_now: float) -> Tuple[float, float]:
        pos, motor_vel = self.hw.motor_pos_vel(motor_key)
        src = str(self.cfg.get("vel_source", "encoder_diff")).lower()
        if src == "motor":
            vel = motor_vel
        else:
            vel = diff.update(pos, t_now)
        return pos, vel

    def run(self):
        if self.hw is None or self.controller is None:
            raise RuntimeError("setup() must be called before run().")

        if self.cfg["trigger_type"] == "typing":
            _ = input("Press Enter to start...\n")
        elif self.cfg["trigger_type"] == "mocap":
            if self.mocap_trigger is not None:
                self.mocap_trigger.wait_for_trigger()
            else:
                print("[WARN] mocap_trigger is None. Starting immediately.")
        else:
            raise NotImplementedError(f"Unknown trigger_type: {self.cfg['trigger_type']}")

        # Fresh MIT write → status RX after any start pause (Ankle_Controller i==0).
        self.hw.prime_mit()

        side = Side(self.cfg["side"])
        # Defaults match Ankle_Exo raw convention (flexion− / extension+); flip only if needed.
        invert_enc_R = bool(self.cfg.get("invert_right_encoder", False))
        invert_tau_R = bool(self.cfg.get("invert_right_torque_cmd", False))
        invert_enc_L = bool(self.cfg.get("invert_left_encoder", False))
        invert_tau_L = bool(self.cfg.get("invert_left_torque_cmd", False))
        torque_limit = float(self.cfg["torque_limit"])
        exo_on = bool(self.cfg["exo_on"])

        # Quick zero: mean of first N frames (~1 s @ 100 Hz) subtracted from ankle angle.
        zero_frames = int(self.cfg.get("zero_frames", 100))
        zero_enabled = bool(self.cfg.get("zero_ankle_angle", True))
        zero_L = AngleZeroing(zero_frames) if zero_enabled else None
        zero_R = AngleZeroing(zero_frames) if zero_enabled else None
        if zero_enabled:
            print(
                f"[Zero] collecting {zero_frames} frames "
                f"(~{zero_frames / float(self.cfg['fs']):.2f} s) — hold still, torque forced to 0."
            )

        rk = RateKeeper(self.cfg["fs"])
        rk.start()
        t0 = time.perf_counter()
        prev_loop_time = None

        trial_dur_sec = float(self.cfg["exp_time_sec"])
        pulse_after_start = float(self.cfg["GPIO_START_DELAY_SEC"])
        first_pulse_sent = False
        first_pulse_end = None
        second_pulse_sent = False
        second_pulse_end = None

        while True:
            _, _, k = rk.wait()
            loop_now = time.perf_counter()
            step_start = loop_now
            loop_dt = 0.0 if prev_loop_time is None else (loop_now - prev_loop_time)
            prev_loop_time = loop_now
            actual_time = step_start - t0

            use_left = side in (Side.LEFT, Side.BOTH) and self.hw.has_motor(self.hw.KEY_L)
            use_right = side in (Side.RIGHT, Side.BOTH) and self.hw.has_motor(self.hw.KEY_R)

            # Decide zeroing *before* update so the lock frame still skips the model.
            zeroing = (
                (use_left and zero_L is not None and not zero_L.done)
                or (use_right and zero_R is not None and not zero_R.done)
            )

            # Robstride MIT: status frames are replies to OPERATION writes.
            # Pattern matches Ankle_Exo: read (from previous write) → compute → write.
            pos_L = vel_L = 0.0
            pos_R = vel_R = 0.0
            if use_left:
                pos_L, vel_L = self._read_side(self.hw.KEY_L, self._diff_L, step_start)
                if invert_enc_L:
                    pos_L *= -1.0
                    vel_L *= -1.0
                if zero_L is not None:
                    pos_L = zero_L.update(pos_L)
            if use_right:
                pos_R, vel_R = self._read_side(self.hw.KEY_R, self._diff_R, step_start)
                if invert_enc_R:
                    pos_R *= -1.0
                    vel_R *= -1.0
                if zero_R is not None:
                    pos_R = zero_R.update(pos_R)

            if zeroing:
                # Do not run TCN / cascade during pre-zero — no model torque at all.
                r = CtrlResult(
                    model_out_R=0.0,
                    model_out_L=0.0,
                    applied_R=0.0,
                    applied_L=0.0,
                    extra={},
                )
                cmd_L = 0.0
                cmd_R = 0.0
            else:
                s = Sensors(
                    imu_L1=imu6_zeros(),
                    imu_L2=imu6_zeros(),
                    imu_R1=imu6_zeros(),
                    imu_R2=imu6_zeros(),
                    pos_L=pos_L,
                    pos_R=pos_R,
                    vel_L=vel_L,
                    vel_R=vel_R,
                )
                r = self.controller.step(s)

                cmd_L = float(r.applied_L) if use_left else 0.0
                cmd_R = float(r.applied_R) if use_right else 0.0
                if not exo_on:
                    cmd_L = 0.0
                    cmd_R = 0.0
                cmd_L = clamp(cmd_L, -torque_limit, torque_limit)
                cmd_R = clamp(cmd_R, -torque_limit, torque_limit)

            # Write every cycle so the next iteration's read gets a status frame.
            if use_left:
                self.hw.set_torque(self.hw.KEY_L, (-cmd_L) if invert_tau_L else cmd_L)
            if use_right:
                self.hw.set_torque(self.hw.KEY_R, (-cmd_R) if invert_tau_R else cmd_R)

            # GPIO sync pulses
            if (not first_pulse_sent) and (actual_time >= pulse_after_start):
                try:
                    self.gpio.pulse_start()
                except Exception as e:
                    print(f"[GPIO] first pulse_start error: {e}")
                first_pulse_sent = True
                first_pulse_end = actual_time + float(self.cfg["PULSE_WIDTH_SEC"])

            if first_pulse_sent and first_pulse_end is not None and actual_time >= first_pulse_end:
                try:
                    self.gpio.pulse_end()
                except Exception as e:
                    print(f"[GPIO] first pulse_end error: {e}")
                first_pulse_end = None

            if (not second_pulse_sent) and (actual_time >= trial_dur_sec):
                try:
                    self.gpio.pulse_start()
                except Exception as e:
                    print(f"[GPIO] second pulse_start error: {e}")
                second_pulse_sent = True
                second_pulse_end = actual_time + float(self.cfg["PULSE_WIDTH_SEC"])

            if second_pulse_sent and second_pulse_end is not None and actual_time >= second_pulse_end:
                try:
                    self.gpio.pulse_end()
                except Exception as e:
                    print(f"[GPIO] second pulse_end error: {e}")
                second_pulse_end = None
                # End trial cleanly after end pulse
                print(f"\nTrial duration {trial_dur_sec}s reached. Stopping.")
                break

            step_end = time.perf_counter()

            if self.tp is not None:
                try:
                    self.tp.sendValue("cmd_L", cmd_L)
                    self.tp.sendValue("cmd_R", cmd_R)
                    self.tp.sendValue("ankle_pos_L", pos_L)
                    self.tp.sendValue("ankle_pos_R", pos_R)
                    self.tp.sendValue("ankle_vel_L", vel_L)
                    self.tp.sendValue("ankle_vel_R", vel_R)
                    self.tp.sendValue("model_in_ankle_angle_raw_r", r.extra.get("model_in_ankle_angle_raw_r", 0.0))
                    self.tp.sendValue("model_in_ankle_angle_raw_l", r.extra.get("model_in_ankle_angle_raw_l", 0.0))
                    self.tp.sendValue("model_in_ankle_vel_raw_r", r.extra.get("model_in_ankle_vel_raw_r", 0.0))
                    self.tp.sendValue("model_in_ankle_vel_raw_l", r.extra.get("model_in_ankle_vel_raw_l", 0.0))
                    self.tp.sendValue("model_in_ankle_angle_lpf", r.extra.get("model_in_ankle_angle_lpf", 0.0))
                    self.tp.sendValue("model_in_ankle_vel_lpf", r.extra.get("model_in_ankle_vel_lpf", 0.0))
                    self.tp.sendValue("model_out_nmpkg", r.extra.get("model_out_nmpkg", 0.0))
                    self.tp.sendValue("moment_raw", r.extra.get("moment_raw", 0.0))
                    self.tp.sendValue("assist_gate", r.extra.get("assist_gate", 0.0))
                    self.tp.sendValue("motion_score", r.extra.get("motion_score", 0.0))
                    self.tp.sendValue("state", r.extra.get("state", 0.0))
                    self.tp.sendValue("GPIO", self.gpio.state())
                    self.tp.sendValue("loop_time", step_end - step_start)
                    self.tp.sendValue("loop_dt", loop_dt)
                except Exception:
                    pass

            if self.current_idx < len(self.data_log["time"]):
                i = self.current_idx
                self.data_log["time"][i] = actual_time
                self.data_log["ankle_pos_L"][i] = pos_L
                self.data_log["ankle_pos_R"][i] = pos_R
                self.data_log["ankle_vel_L"][i] = vel_L
                self.data_log["ankle_vel_R"][i] = vel_R
                self.data_log["cmd_L"][i] = cmd_L
                self.data_log["cmd_R"][i] = cmd_R
                self.data_log["model_out_L"][i] = float(r.model_out_L)
                self.data_log["model_out_R"][i] = float(r.model_out_R)
                self.data_log["model_in_ankle_angle_raw_r"][i] = r.extra.get("model_in_ankle_angle_raw_r", 0.0)
                self.data_log["model_in_ankle_angle_raw_l"][i] = r.extra.get("model_in_ankle_angle_raw_l", 0.0)
                self.data_log["model_in_ankle_vel_raw_r"][i] = r.extra.get("model_in_ankle_vel_raw_r", 0.0)
                self.data_log["model_in_ankle_vel_raw_l"][i] = r.extra.get("model_in_ankle_vel_raw_l", 0.0)
                self.data_log["model_in_ankle_angle_lpf"][i] = r.extra.get("model_in_ankle_angle_lpf", 0.0)
                self.data_log["model_in_ankle_vel_lpf"][i] = r.extra.get("model_in_ankle_vel_lpf", 0.0)
                self.data_log["model_out_nmpkg"][i] = r.extra.get("model_out_nmpkg", 0.0)
                self.data_log["model_out_nmpkg_raw"][i] = r.extra.get("model_out_nmpkg_raw", 0.0)
                self.data_log["moment_raw"][i] = r.extra.get("moment_raw", 0.0)
                self.data_log["assist_gate"][i] = r.extra.get("assist_gate", 0.0)
                self.data_log["motion_score"][i] = r.extra.get("motion_score", 0.0)
                self.data_log["state"][i] = r.extra.get("state", 0.0)
                self.data_log["GPIO"][i] = self.gpio.state()
                self.data_log["loop_dt"][i] = loop_dt
            self.current_idx += 1


_RUNNER: Optional[DualAnkleRunner] = None


def _handle_signal(sig, frame):
    global _RUNNER
    print(f"\nSignal {sig} received. Shutting down...")
    try:
        if _RUNNER:
            _RUNNER.shutdown()
    finally:
        sys.exit(0)


def main():
    global _RUNNER
    args = parse_args()
    cfg = load_config(args.config)
    print_config(cfg)

    runner = DualAnkleRunner(cfg)
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
        print(f"=== Saved: {runner.cfg['trial_name']}.npz ===")


if __name__ == "__main__":
    main()

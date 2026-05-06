import os, sys, time, signal, atexit, gc, traceback
from typing import Optional, Tuple
import time
import numpy as np
import argparse
import enum
import Jetson.GPIO as GPIO
import yaml

from controllers import build_controller
from controllers.base import Sensors
from utils.utils import RateKeeper
from utils.Header_Mocap_trigger import Mocap_trigger
from utils.teleplot import Teleplot
sys.path.insert(0, "/home/exov3/Documents/Rajiv/Jetson_Teensy_Comms")
from Jetson_Teensy import JetsonCanInterface, Device, configure_can_interface

COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_RESET = '\033[0m'

HIGHLIGHT_KEYS = {"exo_on", "scale", "controller_name",
"exp_time_sec", "thigh_gy_mean", "thigh_gy_std", "shank_gy_mean", "shank_gy_std"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dual knee exoskeleton control loop.")
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


def build_data_log(cfg: dict) -> dict:
    log_size = int(cfg["exp_time_sec"] * cfg["fs"])
    return {
        "time": np.zeros(log_size),
        "knee_angle_r": np.zeros(log_size),
        "knee_angle_l": np.zeros(log_size),
        "knee_angle_r_u": np.zeros(log_size),
        "knee_angle_l_u": np.zeros(log_size),
        "knee_angle_r_u_gyr": np.zeros(log_size),
        "knee_angle_l_u_gyr": np.zeros(log_size),
        "gyro_thigh_r": np.zeros(log_size),
        "gyro_shank_r": np.zeros(log_size),
        "cmd_L": np.zeros(log_size),
        "cmd_R": np.zeros(log_size),
        "model_in_knee_angle_raw": np.zeros(log_size),
        "model_in_knee_vel_raw": np.zeros(log_size),
        "model_in_knee_angle_norm": np.zeros(log_size),
        "model_in_knee_vel_norm": np.zeros(log_size),
        "model_out_nmpkg": np.zeros(log_size),
        "moment_raw": np.zeros(log_size),
        "K_r": np.zeros(log_size),
        "Soft_ctrl_r": np.zeros(log_size),
        "K_l": np.zeros(log_size),
        "Soft_ctrl_l": np.zeros(log_size),
        "GPIO": np.zeros(log_size),
    }

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

# ==============================
# Helpers
# ==============================
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def imu6_zeros() -> np.ndarray:
    return np.zeros((6,), dtype=np.float32)

def pack_imu6_from_device(dev: Device, which: int) -> np.ndarray:
    if which == 1:
        a = dev.IMU1_accel_data[-1] if dev.IMU1_accel_data else (0.0, 0.0, 0.0)
        g = dev.IMU1_gyro_data[-1]  if dev.IMU1_gyro_data  else (0.0, 0.0, 0.0)
    else:
        a = dev.IMU2_accel_data[-1] if dev.IMU2_accel_data else (0.0, 0.0, 0.0)
        g = dev.IMU2_gyro_data[-1]  if dev.IMU2_gyro_data  else (0.0, 0.0, 0.0)
    return np.asarray([a[0], a[1], a[2], g[0], g[1], g[2]], dtype=np.float32)


def latest_pos_vel(dev: Device) -> Tuple[float, float]:
    pos = float(dev.Motor_pos_data[-1]) if dev.Motor_pos_data else 0.0
    vel = float(dev.Motor_vel_data[-1]) if dev.Motor_vel_data else 0.0
    return pos, vel


# ==============================
# Runner
# ==============================
class DualKneeRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_log = build_data_log(cfg)
        self.left_dev: Optional[Device] = None
        self.right_dev: Optional[Device] = None
        self.left_exo: Optional[JetsonCanInterface] = None
        self.right_exo: Optional[JetsonCanInterface] = None
        self.controller = None
        self._last_print = time.perf_counter()
        self.current_idx = 0
        self.gpio = GPIOControl(self.cfg["GPIO_OUTPUT_PIN"])
        self.mocap_trigger = None


    def setup(self):
        if self.cfg['trigger_type'] == 'mocap':
            try:
                self.mocap_trigger = Mocap_trigger(server_ip="172.24.44.177", port_number=10)
                self.mocap_trigger.start_client()
            except Exception as e:
                print(f"[Mocap] start_client error: {e}")
        configure_can_interface(channel=self.cfg["can_channel"])
        self.tp = Teleplot(self.cfg["teleplot_ip"], self.cfg["teleplot_port"])

        self.left_dev = Device(teleplot_ip=self.cfg["teleplot_ip"], teleplot_port=self.cfg["teleplot_port"])
        self.right_dev = Device(teleplot_ip=self.cfg["teleplot_ip"], teleplot_port=self.cfg["teleplot_port"])

        self.left_exo = JetsonCanInterface(device_storage=self.left_dev, channel=self.cfg["can_channel"], teensy_id=self.cfg["teensy_id_left"])
        self.right_exo = JetsonCanInterface(device_storage=self.right_dev, channel=self.cfg["can_channel"], teensy_id=self.cfg["teensy_id_right"])

        print(f"Initializing Left Exoskeleton (ID {hex(self.cfg['teensy_id_left'])})...")
        self.left_exo.connect()
        print(f"Initializing Right Exoskeleton (ID {hex(self.cfg['teensy_id_right'])})...")
        self.right_exo.connect()

        self.controller = build_controller(
            self.cfg["controller_name"],
            config=self.cfg
        )
        self.controller.start()

        try:
            t0 = time.perf_counter()
            self.left_exo.set_reference_time(t0)
            self.right_exo.set_reference_time(t0)
        except Exception:
            pass

        print("\n--- Dual Knee Exo Control Loop Started ---")
        print(f"Teleplot: {self.cfg['teleplot_ip']}:{self.cfg['teleplot_port']}")
        print("Press Ctrl+C to stop.")

    def shutdown(self):
        # torque 0
        try:
            if self.left_exo:  self.left_exo.setTorque(0.0)
            if self.right_exo: self.right_exo.setTorque(0.0)
        except Exception as e:
            print(f"[Shutdown] setTorque(0) failed: {e}")

        time.sleep(0.1)

        # close
        try:
            if self.left_exo: self.left_exo.close()
        except Exception as e:
            print(f"[Shutdown] left_exo.close failed: {e}")

        try:
            if self.right_exo: self.right_exo.close()
        except Exception as e:
            print(f"[Shutdown] right_exo.close failed: {e}")

        try:
            if self.controller: self.controller.close()
        except Exception as e:
            print(f"[Shutdown] controller.close failed: {e}")

        try:
            if self.gpio is not None:
                self.gpio.pulse_end()
                self.gpio.close()
        except Exception as e:
            print(f"[Exit] GPIO cleanup error: {e}")

        try:
            gc.enable()
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        print("System Shutdown Complete.")

    def run(self):
        if self.cfg["trigger_type"] == "typing":
            _ = input("Press Enter to start...\n")
        elif self.cfg["trigger_type"] == "mocap":
            if self.mocap_trigger is not None:
                self.mocap_trigger.wait_for_trigger()
            else:
                print("[WARN] mocap_trigger is None. Starting immediately.")
        else:
            raise NotImplementedError(f"Unknown trigger_type: {self.cfg['trigger_type']}")

        side = Side(self.cfg["side"])  # validate

        rk = RateKeeper(self.cfg["fs"])
        rk.start()
        t0 = time.perf_counter()
        prev_loop_time = None
        LOG_DIVIDER = 1

        trial_start_sec = 0.0
        trial_dur_sec = self.cfg["exp_time_sec"]
        pulse_after_start = self.cfg["GPIO_START_DELAY_SEC"]

        first_pulse_sent = False
        first_pulse_end = None

        second_pulse_sent = False
        second_pulse_end = None
        while True:
            overrun, t_sched, k = rk.wait()
            
            loop_now = time.perf_counter()
            step_start = loop_now

            if prev_loop_time is None:
                loop_dt = 0.0
            else:
                loop_dt = loop_now - prev_loop_time
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
                # print(imu_L2[0])
                # print(imu_R2[0])
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

            cmd_L = clamp(cmd_L, -self.cfg["torque_limit"], self.cfg["torque_limit"])
            cmd_R = clamp(cmd_R, -self.cfg["torque_limit"], self.cfg["torque_limit"])

            if use_left:
                self.left_exo.setTorque(-cmd_L)
            if use_right:
                self.right_exo.setTorque(cmd_R)

            
            now = step_start - t0

            # 1) start pulse
            if (not first_pulse_sent) and (now >= trial_start_sec + pulse_after_start):
                try:
                    self.gpio.pulse_start()
                except Exception as e:
                    print(f"[GPIO] first pulse_start error: {e}")
                first_pulse_sent = True
                first_pulse_end = now + self.cfg["PULSE_WIDTH_SEC"]

            if first_pulse_sent and (first_pulse_end is not None) and (now >= first_pulse_end):
                try:
                    self.gpio.pulse_end()
                except Exception as e:
                    print(f"[GPIO] first pulse_end error: {e}")
                first_pulse_end = None

            # 2) end pulse
            if (not second_pulse_sent) and (now >= trial_start_sec + trial_dur_sec):
                try:
                    self.gpio.pulse_start()
                except Exception as e:
                    print(f"[GPIO] second pulse_start error: {e}")
                second_pulse_sent = True
                second_pulse_end = now + self.cfg["PULSE_WIDTH_SEC"]

            if second_pulse_sent and (second_pulse_end is not None) and (now >= second_pulse_end):
                try:
                    self.gpio.pulse_end()
                except Exception as e:
                    print(f"[GPIO] second pulse_end error: {e}")
                second_pulse_end = None
                    


            step_end = time.perf_counter()
            actual_time = step_start - t0

            if k % LOG_DIVIDER == 0:
                try:
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
                    self.tp.sendValue("GPIO", self.gpio.state())
                    self.tp.sendValue("model_in_knee_angle_norm", r.extra.get("model_in_knee_angle_norm", 0.0))
                    self.tp.sendValue("model_in_knee_vel_norm", r.extra.get("model_in_knee_vel_norm", 0.0))
                    self.tp.sendValue("model_out_nmpkg", r.extra.get("model_out_nmpkg", 0.0))
                    self.tp.sendValue("moment_raw", r.extra.get("moment_raw", 0.0))

                    self.tp.sendValue("gyro_thigh_r", imu_R1[5])
                    self.tp.sendValue("gyro_thigh_l", -imu_L1[5])
                    self.tp.sendValue("gyro_shank_r", imu_R2[5])
                    self.tp.sendValue("gyro_shank_l", -imu_L2[5])


                    # self.tp.sendValue("K_r", r.extra.get("K_r", 0.0))
                    self.tp.sendValue("K_l", r.extra.get("K_l", 0.0))
                    self.tp.sendValue("roop_time", step_end - step_start)
                except Exception:
                    pass

            #Data logging
            if self.current_idx < len(self.data_log["time"]):
                self.data_log["time"][self.current_idx]  = actual_time
                # Keep logged knee states aligned with the controller/model convention (radians).
                self.data_log["knee_angle_r"][self.current_idx] = r.extra.get("knee_angle_r", 0.0)
                self.data_log["knee_angle_l"][self.current_idx] = r.extra.get("knee_angle_l", 0.0)
                self.data_log["knee_angle_r_u"][self.current_idx] = r.extra.get("knee_angle_r_u", 0.0)
                self.data_log["knee_angle_l_u"][self.current_idx] = r.extra.get("knee_angle_l_u", 0.0)
                self.data_log["knee_angle_r_u_gyr"][self.current_idx] = r.extra.get("knee_r_u_gyr", 0.0)
                self.data_log["knee_angle_l_u_gyr"][self.current_idx] = r.extra.get("knee_l_u_gyr", 0.0)
                self.data_log["gyro_thigh_r"][self.current_idx] = imu_R1[5]
                self.data_log["gyro_shank_r"][self.current_idx] = imu_R2[5]
                self.data_log["cmd_L"][self.current_idx] = cmd_L
                self.data_log["cmd_R"][self.current_idx] = cmd_R
                self.data_log["model_in_knee_angle_raw"][self.current_idx] = r.extra.get("model_in_knee_angle_raw", 0.0)
                self.data_log["model_in_knee_vel_raw"][self.current_idx] = r.extra.get("model_in_knee_vel_raw", 0.0)
                self.data_log["model_in_knee_angle_norm"][self.current_idx] = r.extra.get("model_in_knee_angle_norm", 0.0)
                self.data_log["model_in_knee_vel_norm"][self.current_idx] = r.extra.get("model_in_knee_vel_norm", 0.0)
                self.data_log["model_out_nmpkg"][self.current_idx] = r.extra.get("model_out_nmpkg", 0.0)
                self.data_log["moment_raw"][self.current_idx] = r.extra.get("moment_raw", 0.0)
                self.data_log["K_r"][self.current_idx] = r.extra.get("K_r", 0.0)
                self.data_log["Soft_ctrl_r"][self.current_idx] = r.extra.get("Soft_ctrl_r", 0.0)
                self.data_log["K_l"][self.current_idx] = r.extra.get("K_l", 0.0)
                self.data_log["Soft_ctrl_l"][self.current_idx] = r.extra.get("Soft_ctrl_l", 0.0)
                self.data_log["GPIO"][self.current_idx] = self.gpio.state()

            self.current_idx += 1


_RUNNER: Optional[DualKneeRunner] = None

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

    runner = DualKneeRunner(cfg)
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
            runner.data_log[key] = runner.data_log[key][:runner.current_idx]
        np.savez(runner.cfg["trial_name"], **runner.data_log)
        print(f"=== Saving data for trial: {runner.cfg['trial_name']} ===")


if __name__ == "__main__":
    main()

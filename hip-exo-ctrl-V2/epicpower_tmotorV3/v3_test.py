from actuator_group import ActuatorGroup
from tmotor_v3 import (
    TMotorV3,
    _create_mit_message,
)
import time


# T_MIN = -18.0
# T_MAX = 18.0
# P_MIN = -12.56
# P_MAX = 12.56
# V_MIN = -65.0
# V_MAX = 65.0
# KP_MIN = 0
# KP_MAX = 500.0
# KD_MIN = 0
# KD_MAX = 5.0

ID = 2 # Make this whatever you need

cmd_torque = 0.0  # Nm
cmd_position = 0.0  # rad
cmd_velocity = 0.0  # rad/s
cmd_kp = 0.0  # Nms/rad
cmd_kd = 0.0  # Nms/rad/s

acts = ActuatorGroup([TMotorV3(ID, "AK80-9-V3")])
# acts.enable_actuators()

print(_create_mit_message(ID, cmd_position, cmd_velocity, cmd_kp, cmd_kd, cmd_torque))
# acts.set_position(ID, cmd_position, cmd_kp, cmd_kd)
# print("Start spinning")
# acts.set_velocity(ID, cmd_velocity, cmd_kd)
# time.sleep(2)

# print("re-up")
# acts.set_velocity(ID, cmd_velocity, cmd_kd)
# time.sleep(2)

# print("re-up")
# acts.set_velocity(ID, cmd_velocity, cmd_kd)
# time.sleep(2)

# print("re-up")
# acts.set_velocity(ID, cmd_velocity, cmd_kd)
# time.sleep(2)

# print("re-up")
# acts.set_velocity(ID, cmd_velocity, cmd_kd)
# time.sleep(2)

# print("Stop spinning")

# acts._exit_gracefully(1, 2)
# acts.notifier.stop()
# acts.bus.shutdown()

while True:
    acts.set_torque(ID, cmd_torque)
    # acts.set_position(ID, cmd_position, cmd_kp, cmd_kd)
    # acts.set_velocity(ID, cmd_velocity, cmd_kd)
    time.sleep(0.05)
    print(
        f"{acts.get_torque(ID)} Nm, {acts.get_position(ID)} rad,"
        f" {acts.get_velocity(ID)} rad/s, {acts.get_temperature(ID)} C"
    )

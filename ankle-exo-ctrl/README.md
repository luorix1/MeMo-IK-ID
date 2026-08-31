# ankle-exo-ctrl

Real-time ankle exoskeleton control on Jetson + **Robstride RS-0x** (SocketCAN).

Hardware matches [`Ankle_Exo/`](../../Ankle_Exo): MIT torque mode via vendored `robstride_dynamics/`
(not the CubeMars / TMotor stack used by hip/knee).

## Controllers

| `controller_name` | Behavior |
|---|---|
| `TEST` | Always 0 Nm; streams pos/vel to Teleplot |
| `impedance_enc` | `τ = K*(θ_ref − θ) − B·ω` |
| `cascade_uni` | Unilateral TCN (TRT): encoder angle + dθ/dt → Nm/kg → Nm |

## Motor API (from Ankle_Exo)

```python
from robstride_dynamics import RobstrideBus, Motor

motors = {"motor_R": Motor(id=1, model="rs-02")}
bus = RobstrideBus("can0", motors)
bus.connect()
bus.enable("motor_R")

# MIT torque (kp=kd=0)
bus.write_operation_frame("motor_R", position=0, kp=0, kd=0, velocity=0, torque=cmd_nm)
pos, vel, tor, temp = bus.read_operation_frame("motor_R")  # rad, rad/s, Nm, °C
```

Encoder convention in Ankle_Exo: **flexion (−), extension (+)**.

`Ankle_Exo` currently uses a **single** motor (`id=1`). Set `side: right` (or `left`)
in YAML; use `side: both` only when a second CAN id is wired.

## Bring-up

```bash
cd os_kinetics/ankle-exo-ctrl
pip install -r requirements.txt
python main_ankle.py cfg/bringup.yaml
```

1. `TEST` + `exo_on: false` — check `ankle_pos_*` / `ankle_vel_*` on Teleplot  
2. Flip `invert_*` flags only if signs disagree with your controller convention  
3. Then `impedance_enc` or `cascade_uni` with low `torque_limit`

## TCN deploy

```bash
python utils/pt2trt.py --pt ../checkpoints/ankle/best_model.pt \
                       --trt ../checkpoints/ankle/best_model.trt \
                       --cfg cfg/final.yaml
python main_ankle.py cfg/final.yaml
```

Throughput check (Jetson, after `.trt` exists):

```bash
python utils/bench_trt_throughput.py --trt best_model.trt --cfg cfg/final.yaml
# closer to controller path (H2D input + D2H output each iter):
python utils/bench_trt_throughput.py --trt best_model.trt --host-copy
```

## Layout

```
ankle-exo-ctrl/
  main_ankle.py              # RobstrideAnkleHardware + control loop
  robstride_dynamics/        # vendored from Ankle_Exo
  cfg/{bringup.yaml, final.yaml}
  controllers/               # TEST, impedance_enc, cascade_uni, TRT worker
  utils/                     # RateKeeper, Teleplot, pt2trt, ...
```

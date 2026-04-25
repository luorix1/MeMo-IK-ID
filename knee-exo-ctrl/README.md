# knee-exo-ctrl

Real-time bilateral/unilateral knee exoskeleton control running on a Jetson Orin. Controllers communicate with Teensy microcontrollers over CAN bus and run at a fixed rate (default 100 Hz).

---

## Hardware sign conventions

| Signal | Positive direction |
|---|---|
| Left motor torque | Flexion |
| Right motor torque | Extension |
| Left IMU1 (thigh) gyro-z | Left mediolateral axis |
| Left IMU2 (shank) gyro-z | Left mediolateral axis |

All controllers internally negate the left-side encoder and IMU-z readings so that flexion-positive convention holds symmetrically for both legs.

---

## Software architecture

### `Sensors` and `CtrlResult` — the controller I/O contract (`controllers/base.py`)

Every control step takes a `Sensors` struct and returns a `CtrlResult` struct.

```
Sensors
  imu_L1, imu_L2 : np.ndarray (6,)   # left thigh / shank: [ax, ay, az, gx, gy, gz]
  imu_R1, imu_R2 : np.ndarray (6,)   # right thigh / shank
  pos_L, pos_R   : float              # motor encoder angle (degrees, raw)
  vel_L, vel_R   : float              # motor encoder velocity (degrees/s, raw)

CtrlResult
  model_out_R, model_out_L : float    # raw model output before any post-processing
  applied_R, applied_L     : float    # final torque command sent to each motor (Nm)
  extra                    : dict     # arbitrary telemetry (knee angles, stiffness, gate, etc.)
```

`main_knee.py` reads `applied_R` / `applied_L`, applies the global `scale` and `torque_limit` from config, then sends via CAN. The `extra` dict is forwarded to Teleplot and the `.npz` data log.

---

### `BaseController` — the controller interface (`controllers/base.py`)

All controllers subclass `BaseController` and implement three lifecycle methods:

| Method | Purpose |
|---|---|
| `start()` | Called once after construction (e.g. launch TRT worker process) |
| `step(s: Sensors) -> CtrlResult` | Called every control tick |
| `close()` | Called on shutdown (zero torques, join processes, etc.) |

---

### `RollingWindow` — fixed-length time-series buffer (`controllers/base.py`)

ML-based controllers accumulate sensor history for the TCN model using `RollingWindow`. It wraps a NumPy array of shape `(C, T)` (channels × time steps). Calling `push_last(x_last)` shifts the buffer left by one and inserts the newest frame at the end, returning the full `(C, T)` window.

---

### Controller registry (`controllers/__init__.py`)

Controllers are registered by string name in `REGISTRY`. `build_controller(name, config=cfg)` instantiates the right class. The `controller_name` field in the YAML config selects which controller runs.

| `controller_name` | Class | File |
|---|---|---|
| `dofc_knee` | `impedance_rl` | `controllers/dofc_knee.py` |
| `impedance_rl` | `impedance_rl` | `controllers/impedance_rl.py` |
| `impedance_rl_uni` | `impedance_rl_uni` | `controllers/impedance_rl_uni.py` |
| `biotorque` | `biotorque` | `controllers/biotorque.py` |
| `test` | `TEST` | `controllers/test.py` |

---

### Available controllers

#### `dofc_knee` — bilateral gait-phase damping controller
Rule-based bilateral controller. Estimates gait phase (stance / swing) from IMU angular velocity and encoder angle thresholds with a minimum dwell time. Applies velocity-proportional damping torque with separate stance and swing gain scales. Smooth transitions between phases are handled by a first-order blend filter. No ML model.

#### `impedance_rl` — bilateral TCN impedance controller
Bilateral version of the TCN impedance controller below. Runs a shared TRT inference worker for both legs simultaneously.

#### `impedance_rl_uni` — unilateral TCN impedance controller
TCN runs in a separate `mp.Process` (`TRTWorkerUni`) so inference never blocks the control loop. The main loop sends the latest input frame via a `mp.Queue(maxsize=1)` and drains the output queue each tick to get the most recent result.

- **Model input**: `(1, 2, T)` — `[encoder_pos_rad, knee_vel_imu]` (z-scored per channel)
  - `ch0`: motor encoder position (sign-corrected, radians)
  - `ch1`: knee angular velocity from IMUs: `shank_gz − thigh_gz` (sign-corrected, rad/s)
- **Model output**: `[K_cmd ∈ [−1,1], gait_cmd ∈ [−1,1]]`
  - `K_cmd` maps linearly to stiffness `K ∈ [K_min, K_max]`
  - `gait_cmd` interpolates between three impedance reference postures: flexion, transition, extension (Gaussian weights)
- **Impedance law**: `τ = w_flex·τ_flex + w_trans·τ_trans + w_ext·τ_ext` with stiffness-proportional damping, soft-saturated via `tanh`
- **Motion gating**: an `idle → starting → active` state machine monitors normalized knee-velocity energy. Assistance is ramped in/out with configurable time constants. The gate scales `K` to zero when idle.

#### `biotorque` — bilateral TCN biotorque predictor
Predicts knee biological torque from IMU gyros and knee angle using a TRT-accelerated TCN. Scales the prediction by subject mass and a tunable gain.

- **Model input**: `(1, 3, T)` — `[thigh_gy, shank_gy, knee_angle]` (z-scored)
- **Model output**: scalar torque (Nm/kg), multiplied by `mass × biotorque_gain`
- An optional configurable output delay and motion gate (same `idle/starting/active` logic) are applied before commanding the motor.

---

### TRT inference worker pattern

Inference-based controllers offload TensorRT execution to a dedicated `mp.Process` to avoid GPU latency spikes in the real-time loop:

```
Control loop (main process)          TRT worker (subprocess)
────────────────────────────         ──────────────────────────
in_q.put_nowait(x)          ──▶      data = in_q.get()
latest = out_q.get_nowait() ◀──      out_q.put_nowait(y)
```

Both queues have `maxsize=1`. The control loop uses non-blocking puts/gets (dropping stale frames), so it always runs at the target rate regardless of GPU latency. The worker drains its input queue to always process the newest frame.

---

### `DualKneeRunner` — top-level orchestrator (`main_knee.py`)

Lifecycle:

1. **`setup()`** — configures CAN interface, instantiates `JetsonCanInterface` for left and right legs, calls `build_controller()`, optionally connects to the motion capture trigger server, starts Teleplot UDP stream.
2. **`run()`** — waits for start trigger (keyboard or mocap), then enters the fixed-rate loop:
   - `exo.getParameters()` — polls CAN for latest sensor data
   - Packs data into a `Sensors` struct
   - `controller.step(s)` — runs one control tick
   - Applies `scale`, `torque_limit` clamp, checks `exo_on` flag
   - `exo.setTorque(cmd)` — sends torque commands over CAN
   - Sends GPIO sync pulses at trial start and end (for motion capture alignment)
   - Streams telemetry via Teleplot
   - Accumulates data in `data_log`
3. **`shutdown()`** — zeros torques, closes CAN, joins controller processes, saves `trial_name.npz`

---

### `RateKeeper` — high-precision loop timer (`utils/utils.py`)

Maintains a fixed control rate using `time.perf_counter_ns()`. It sleeps for most of the period and busy-spins for the last ~50 µs to minimize jitter. If the loop falls behind by more than `catchup_cycles` periods, it fast-forwards the schedule instead of trying to catch up, preventing burst overruns.

---

## Config file structure

All parameters are passed as a flat YAML dict. Key fields:

| Field | Description |
|---|---|
| `controller_name` | Selects controller from the registry |
| `side` | `"right"`, `"left"`, or `"both"` |
| `fs` | Control loop rate (Hz) |
| `exp_time_sec` | Trial duration (s) — sets data log size |
| `exo_on` | If `false`, zeroes all torque commands (sensor-only mode) |
| `scale` | Global torque multiplier applied after the controller |
| `torque_limit` | Hard clamp on final torque command (Nm) |
| `trt_engine_path` | Path to compiled TensorRT engine (`.trt`) |
| `frame_length` | TCN context window length `T` |
| `input_size` / `output_size` | Model I/O channel counts (validated at construction) |
| `trigger_type` | `"typing"` (Enter key) or `"mocap"` |
| `trial_name` | Output `.npz` filename |
| `GPIO_OUTPUT_PIN` | Jetson GPIO board pin for sync pulse |
| `PULSE_WIDTH_SEC` | TTL pulse width (s) |
| `GPIO_START_DELAY_SEC` | Delay after trial start before first pulse (s) |
| `can_channel` | CAN interface name (e.g. `"can0"`) |
| `teensy_id_left/right` | CAN IDs of the left/right Teensy boards |
| `teleplot_ip/port` | Teleplot UDP destination |

Per-controller normalization stats (`encoder_mean`, `encoder_std`, `knee_vel_mean`, `knee_vel_std`, `thigh_gy_mean`, etc.) must be updated from the training dataset before each deployment.

---

## Running

```bash
# On the Jetson (via SSH or directly):
python /home/exov3/Documents/Knee_CTRL/main_knee.py path/to/config.yaml
```

---

## Jetson hotspot setup

First time:
```bash
sudo nmcli device wifi hotspot ifname wlP1p1s0 ssid "JetsonHotspot" password "12345678"
ssh exov3@10.42.0.1
```

Subsequent connections:
```bash
sudo nmcli connection up Hotspot
```

After experiment (restore normal Wi-Fi):
```bash
sudo nmcli connection modify Hotspot connection.autoconnect no
sudo nmcli connection modify CMU-DEVICE connection.autoconnect yes
```

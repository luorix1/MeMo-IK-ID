# hip-exo-ctrl

Real-time bilateral hip exoskeleton control running on a Jetson Orin. The controller reads pelvis and bilateral thigh IMUs, runs a TensorRT-accelerated TCN to predict biological hip torque, and commands two AK80-9 motors over CAN bus at a fixed rate (default 100 Hz).

Refactored from `V2_Hip_Exo/State2Torque 2 Controllers/allnewK5_6min_Cleaned_PreAlloc.py`.

---

## Hardware sign conventions

| Signal | Positive direction |
|---|---|
| Left motor torque / position | Hip flexion |
| Right motor torque / position | Hip flexion (negated in software — motor reads opposite sense) |
| Pelvis IMU acc / gyro | Raw ICM20948 axes |
| Thigh IMUs acc / gyro | Raw ICM20948 axes |

The right motor's position, velocity, and torque command are all negated before/after the hardware call so that both sides use a flexion-positive convention internally.

When mirroring the left side for right-referenced inference, `acc_y`, `gyr_x`, and `gyr_z` are sign-flipped to reflect the sagittal plane.

---

## Software architecture

### `Sensors` and `CtrlResult` — the controller I/O contract (`controllers/base.py`)

Every control step takes a `Sensors` struct and returns a `CtrlResult` struct.

```
Sensors
  imu_P           : np.ndarray (6,)   # pelvis:      [ax, ay, az, gx, gy, gz]
  imu_L, imu_R    : np.ndarray (6,)   # left/right thigh: same layout
  pos_L, pos_R    : float             # motor encoder angle (degrees, sign-corrected)
  vel_L, vel_R    : float             # motor encoder velocity (degrees/s, sign-corrected)

CtrlResult
  model_out_R, model_out_L : float    # raw TCN output (Nm/kg), before mass scaling
  applied_R, applied_L     : float    # final torque command sent to each motor (Nm)
  extra                    : dict     # full pipeline signals (net, bio, scaled, delayed, filtered)
```

`main_hip.py` reads `applied_R` / `applied_L`, checks the `exo_on` flag, then sends via CAN. The `extra` dict is forwarded to Teleplot and the CSV data log.

---

### `BaseController` — the controller interface (`controllers/base.py`)

All controllers subclass `BaseController` and implement three lifecycle methods:

| Method | Purpose |
|---|---|
| `start()` | Called once after construction (e.g. launch TRT worker process) |
| `step(s: Sensors) -> CtrlResult` | Called every control tick |
| `close()` | Called on shutdown (stop worker process, release queues) |

---

### `RollingWindow` — fixed-length time-series buffer (`controllers/base.py`)

The TCN model requires a fixed-length history window. `RollingWindow` wraps a NumPy array of shape `(..., T)`. Calling `push_last(x_last)` shifts the buffer left by one frame and inserts the newest sample at the end, returning the full window. It is used for both the bilateral input windows `(1, C, T)`.

---

### Controller registry (`controllers/__init__.py`)

Controllers are registered by string name in `REGISTRY`. `build_controller(name, config=cfg)` instantiates the right class. The `controller_name` field in the YAML config selects which controller runs.

| `controller_name` | Class | File |
|---|---|---|
| `hip_biotorque` | `HipBiotorque` | `controllers/biotorque_hip.py` |

---

### Available controllers

#### `hip_biotorque` — bilateral TCN biotorque predictor

Predicts hip biological torque from thigh IMU data using a TRT-accelerated TCN model. The prediction is scaled by subject mass and a tunable gain, then passed through a delay buffer and a realtime Butterworth filter before commanding the motor.

**Model input** (per side): `(1, C, T)` — the `C`-channel IMU feature vector z-scored with `input_mean.npy` / `input_std.npy` from the model directory.

- Right side: raw right thigh IMU data
- Left side: right-referenced mirror of left thigh IMU (`acc_y`, `gyr_x`, `gyr_z` negated)

**Model output**: scalar (Nm/kg), denormalized using `label_mean.npy` / `label_std.npy`.

**Full pipeline per side**:

```
model_out (Nm/kg)
  × mass              → net_torque
  − applied_torque    → bio_torque     (feedback: subtract what was applied last tick)
  × scale_factor      → scaled_torque
  delay_steps frames  → delayed_torque
  Butterworth LPF     → filtered_torque
  clamp ±torque_limit → cmd
```

The `bio_torque` feedback term subtracts the previously applied command from the model's net torque prediction. This prevents the exo from fighting itself when the model already accounts for the device's contribution.

The delay formula matches the original controller: `delay_steps = int(desired_delay_ms / 10 − 4)`, with a minimum of 0. At 100 Hz and `desired_delay_ms = 110`, this gives 7 steps (70 ms of additional pipeline delay).

---

### TRT inference worker pattern

Inference is offloaded to a dedicated `mp.Process` (`HipTRTWorker`) so GPU latency never stalls the real-time loop:

```
Control loop (main process)          HipTRTWorker (subprocess)
────────────────────────────         ──────────────────────────
in_q.put_nowait({"r": x_R,  ──▶      data = in_q.get()
                  "l": x_L})          (drains queue, keeps latest)
latest = out_q.get_nowait() ◀──      out_q.put_nowait((y_r, y_l))
```

Both queues have `maxsize=1`. The control loop uses non-blocking puts/gets, so it runs at the target rate even if inference takes longer than one period. The worker drains its input queue on each iteration to always process the newest frame.

**Difference from the knee worker**: the hip model was compiled with `batch_size=1`, so the worker runs two sequential forward passes (right then left) with the same engine, rather than a single batched pass. Output denormalization (`label_mean / label_std`) is applied inside the worker before returning results to the main process.

The worker performs a 10-step warm-up on startup and prints `[HipTRTWorker] Ready.` before the main loop accepts a start trigger.

---

### `DualHipRunner` — top-level orchestrator (`main_hip.py`)

Lifecycle:

1. **`setup()`** — initializes GPIO, instantiates `TMotorV3` motors via `ActuatorGroup`, opens CAN bus, initializes `ICM20948_I2C_IMUs`, calls `build_controller()`, optionally connects to the mocap trigger server, starts Teleplot UDP stream.
2. **`run()`** — waits for start trigger (Enter key or mocap), then enters the fixed-rate loop driven by `RateKeeper`:
   - Reads motor encoder and IMU data
   - Packs data into a `Sensors` struct
   - `controller.step(s)` — runs one control tick
   - Checks `exo_on` flag; zeros commands if disabled
   - `mtr_comms.set_torque()` — sends commands over CAN (right side negated)
   - Sends GPIO sync pulses at trial start and end (for motion capture alignment)
   - Streams telemetry via Teleplot
   - Accumulates data in pre-allocated NumPy arrays
3. **`shutdown()`** — zeros torques, closes CAN bus and notifier, joins controller worker, cleans up GPIO, saves CSVs

Signal handling: `SIGINT` and `SIGTERM` call `shutdown()` and exit cleanly. `atexit` also registers `shutdown()` as a fallback.

---

### `RateKeeper` — high-precision loop timer (`utils/utils.py`)

Maintains a fixed control rate using `time.perf_counter_ns()`. It sleeps for most of the period and busy-spins for the last ~50 µs to minimize OS scheduling jitter. If the loop falls behind by more than `catchup_cycles` periods, it fast-forwards the schedule instead of trying to catch up, preventing burst overruns after a stall.

---

## Config file structure

All parameters are passed as a flat YAML dict. Key fields:

| Field | Description |
|---|---|
| `controller_name` | Selects controller from the registry |
| `fs` | Control loop rate (Hz) |
| `exp_time_sec` | Trial duration (s) — sets data log buffer size |
| `exo_on` | If `false`, zeroes all torque commands (sensor-only / warm-up mode) |
| `scale_factor` | Global bio-torque multiplier (0 = zero assist, 1 = full) |
| `torque_limit` | Hard clamp on final torque command (Nm) |
| `mass` | Subject body mass (kg), used to scale model output |
| `desired_delay_ms` | Pipeline delay to add (ms); minimum meaningful value ≈ 40 |
| `lpf_cutoff_Hz` | Butterworth low-pass cutoff frequency (Hz) |
| `lpf_order` | Butterworth filter order |
| `trt_engine_path` | Path to compiled TensorRT engine (`.trt`) |
| `frame_length` | TCN context window length `T` (frames) |
| `trigger_type` | `"typing"` (Enter key) or `"mocap"` |
| `mocap_server_ip/port` | TCP address of the mocap trigger server |
| `trial_name` | Prefix for saved CSV files |
| `can_channel` | CAN interface name (e.g. `"can0"`) |
| `can_id_L` / `can_id_R` | CAN node IDs for left and right motors |
| `motor_type` | Motor model string passed to `TMotorV3` (e.g. `"AK80-9"`) |
| `gpio_output_pin` | Jetson GPIO board pin for sync pulse |
| `pulse_width_sec` | TTL pulse width (s) |
| `gpio_start_delay_sec` | Delay after trial start before first pulse (s) |
| `target_time_range_sec` | Duration between start and end GPIO pulses (s) |
| `teleplot_ip` / `teleplot_port` | Teleplot UDP destination |

---

## Running

```bash
# On the Jetson (via SSH or directly):

# Copy the reference config for the current subject
cp cfg/hip_biotorque_default.yaml cfg/SUB01_session1.yaml
# Edit: trial_name, mass, trt_engine_path, exo_on, scale_factor, desired_delay_ms, …

python main_hip.py cfg/SUB01_session1.yaml
```

Press **Enter** (typing trigger) or wait for the mocap `"exo on"` message.  
Press **Ctrl-C** at any time — data is saved automatically before exit.

---

## Outputs

Three CSV files are written to the working directory after each trial:

| File | Contents |
|---|---|
| `{trial_name}_input_motor.csv` | `time`, motor position, velocity, GPIO state |
| `{trial_name}_input_imu.csv` | `time`, pelvis + left/right thigh acc & gyro (6 ch each), GPIO state |
| `{trial_name}_output_torque.csv` | `time`, full torque pipeline (`model_output` → `net` → `bio` → `scaled` → `delayed` → `filtered` → `cmd`), actual measured torques, GPIO state |

All arrays are pre-allocated with NaN and sliced to the actual number of samples collected.

---

## Jetson hotspot setup

First time:
```bash
sudo nmcli device wifi hotspot ifname wlP1p1s0 ssid "JetsonHotspot" password "12345678"
ssh metamobility2@10.42.0.1
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

---

## Dependencies

- `epicpower_tmotorV3` — `ActuatorGroup`, `TMotorV3`
- `Header_ICM20948_I2C_pcb2` — `ICM20948_I2C_IMUs`
- `python-can`
- `Jetson.GPIO`
- `tensorrt`, `torch`
- `numpy`, `scipy`, `pandas`, `pyyaml`

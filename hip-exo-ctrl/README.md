# hip-exo-ctrl

Dual hip exoskeleton control repo with the **same directory layout** as `os_kinetics/knee-exo-ctrl/`:

- `main_hip.py` — entrypoint (hip motor/IMU wiring lives here, like Teensy/CAN setup in `main_knee.py`).
- `cfg/` — YAML configs (`default.yaml` mirrors knee field names: `exp_time_sec`, `teleplot_ip` / `teleplot_port`, GPIO pins, etc.).
- `controllers/` — hip controllers (`simgyro3`, `gyro1ch_trt`, `biotorque`, `DOFC`) plus shared `trt_worker.py`.
- `utils/` — `utils.py` (`RateKeeper`), `teleplot.py`, `Header_Mocap_trigger.py`, TRT helpers, `jetson-orin-gpio-patch/`, etc. (mirrors knee tree).
- `tcn_model/` — same placeholder/model header module path as knee.
- `data_analysis/` — notebook slot matching knee.

## Quick start

```bash
cd ~/Jinwoo/hip-exo-ctrl
python main_hip.py cfg/default.yaml
```

Configure `sensor_python_paths`, CAN IDs, TRT paths, and `controller_name` / `controller:` in the YAML.

## Dependencies

Same classes of deps as knee/melody Jetson stacks: `numpy`, `pyyaml`, `python-can`, TensorRT + CUDA torch for TRT controllers, optional `Jetson.GPIO`, plus `sensor_motor` / `hip_exo.sensor_motor` or `epicpower` IMU+motor drivers.

Teleplot is always constructed in `setup()` from `teleplot_ip` / `teleplot_port` (same pattern as `main_knee.py`). Trial logs are saved with `np.savez(trial_name, ...)` like knee.

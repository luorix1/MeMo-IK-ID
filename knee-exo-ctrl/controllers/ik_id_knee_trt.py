"""
TensorRT inference controller for ``ik_id/trainV2.py`` sagittal_knee models.

Same feature engineering and normalization as ``IkIdKneeOnnxController``, but
inference is executed by a dedicated TRT worker process that batches both legs
into one engine call: (2, C, T).
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path
from queue import Empty
from typing import Any, Dict, Tuple

import numpy as np

_KNEE_ROOT = Path(__file__).resolve().parents[1]
if str(_KNEE_ROOT) not in sys.path:
    sys.path.insert(0, str(_KNEE_ROOT))

from run_bundle import (
    ik_indices_unilateral_paired,
    load_checkpoint_metadata,
    load_train_config,
    normalization_for_dof,
    resolve_deploy_path,
    resolve_run_dir,
)
from utils.causal_butter import make_model_io_filter_bank

from .base import BaseController, CtrlResult, RollingWindow, Sensors
from .trt_worker import TRTWorker


def _tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except Exception:
        return False


class IkIdKneeTrtController(BaseController):
    name = "ik_id_knee_trt"

    def __init__(self, config: Dict[str, Any]):
        if not _tensorrt_available():
            raise ImportError("ik_id_knee_trt requires TensorRT Python package.")

        self.cfg = config
        self.fs = int(config["fs"])
        self.dt = 1.0 / self.fs

        run_dir = resolve_run_dir(str(config["run_dir"]))
        try:
            train_cfg = load_train_config(run_dir)
        except FileNotFoundError:
            train_cfg = {}

        ckpt_path = resolve_deploy_path(config, Path(config.get("checkpoint_path") or (run_dir / "best_model.pt")))
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Need {ckpt_path} for normalization and IK indices."
            )
        ckpt = load_checkpoint_metadata(ckpt_path)

        mc = ckpt["model_config"]
        self._n_in = int(mc["n_input_channels"])
        self._n_out = int(mc["n_output_channels"])
        if self._n_in != 2 or self._n_out != 1:
            raise ValueError(
                f"This controller expects sagittal_knee model (n_in=2, n_out=1), got {self._n_in}/{self._n_out}."
            )

        inp_idx = list(ckpt["input_indices"])
        self._ik_r, self._ik_l = ik_indices_unilateral_paired(inp_idx)
        norm = ckpt["normalization"]
        self._pr_m, self._pr_s, self._vr_m, self._vr_s = normalization_for_dof(norm, self._ik_r)
        self._pl_m, self._pl_s, self._vl_m, self._vl_s = normalization_for_dof(norm, self._ik_l)

        self.frame_length = int(config.get("frame_length", ckpt["window_size"]))
        if self.frame_length != int(ckpt["window_size"]) and not config.get("allow_window_mismatch", False):
            raise ValueError(
                f"frame_length={self.frame_length} != training window_size={ckpt['window_size']}."
            )

        trt_path = resolve_deploy_path(config, Path(config.get("trt_path") or (run_dir / "best_model.trt")))
        if not trt_path.is_file():
            raise FileNotFoundError(f"Missing TRT engine: {trt_path}")

        self.x_r = RollingWindow((2, self.frame_length))
        self.x_l = RollingWindow((2, self.frame_length))

        self.moment_mass_kg = float(config.get("moment_mass_kg", 1.0))
        self.torque_sign_r = float(config.get("torque_sign_r", 1.0))
        self.torque_sign_l = float(config.get("torque_sign_l", 1.0))

        self.knee_filter_tau = float(config.get("knee_filter_tau", 0.05))
        self.imu_gyro_filter_tau = float(config.get("imu_gyro_filter_tau", 0.15))
        src = str(config.get("joint_velocity_source", "imu_gyro_delta")).lower()
        if src not in ("imu_gyro_delta", "encoder"):
            raise ValueError("joint_velocity_source must be 'imu_gyro_delta' or 'encoder'")
        self._vel_source = src

        units = str(config.get("imu_gyro_z_units", "deg_per_s")).lower().replace(" ", "")
        if units in ("deg/s", "deg_per_s", "dps"):
            self._gyro_to_rad_s = float(np.deg2rad(1.0))
        elif units in ("rad/s", "rad_per_s", "rps"):
            self._gyro_to_rad_s = 1.0
        else:
            raise ValueError(f"Unknown imu_gyro_z_units: {config.get('imu_gyro_z_units')!r}")

        self._q_r_f = 0.0
        self._q_l_f = 0.0
        self._qd_r_f = 0.0
        self._qd_l_f = 0.0
        self._thigh_gy_r = self._shank_gy_r = 0.0
        self._thigh_gy_l = self._shank_gy_l = 0.0

        self._train_rollout = int(train_cfg.get("rollout_decimate_step", config.get("rollout_decimate_step", 1)))
        self._expected_fs = 200.0 / float(self._train_rollout) if self._train_rollout > 1 else 200.0
        if abs(self._expected_fs - float(self.fs)) > 1.0 and not config.get("allow_fs_mismatch", False):
            raise ValueError(
                f"Training effective rate ≈ {self._expected_fs:.1f} Hz, but fs={self.fs}."
            )

        _lp_en = bool(config.get("model_io_lowpass_enable", True))
        _lp_fc = float(config.get("model_io_lowpass_cutoff_hz", train_cfg.get("lowpass_cutoff_hz", 4.0)))
        _lp_ord = int(config.get("model_io_lowpass_order", train_cfg.get("lowpass_order", 4)))
        self._io_lp = make_model_io_filter_bank(self.fs, cutoff_hz=_lp_fc, order=_lp_ord, enabled=_lp_en)

        self._in_q = mp.Queue(maxsize=1)
        self._out_q = mp.Queue(maxsize=1)
        self._worker = TRTWorker(
            in_q=self._in_q,
            out_q=self._out_q,
            engine_path=str(trt_path),
            single_in_shape=(1, 2, self.frame_length),
            single_out_shape=(1,),
        )
        self._worker.daemon = True
        self._last_m_r = 0.0
        self._last_m_l = 0.0

    def _alpha(self, tau: float) -> float:
        if tau <= 0.0:
            return 1.0
        return self.dt / tau

    def _lpf(self, prev: float, raw: float, tau: float) -> float:
        a = self._alpha(tau)
        return float(prev + a * (raw - prev))

    def _exo_knee_encoder(self, s: Sensors) -> Tuple[float, float, float, float]:
        q_r = float(np.deg2rad(s.pos_R))
        qd_r_enc = float(np.deg2rad(s.vel_R))
        q_l = float(-np.deg2rad(s.pos_L))
        qd_l_enc = float(-np.deg2rad(s.vel_L))
        return q_r, qd_r_enc, q_l, qd_l_enc

    def _imu_knee_rates_rad_s(self, s: Sensors) -> Tuple[float, float]:
        gzr = float(s.imu_R1[5]) * self._gyro_to_rad_s
        szr = float(s.imu_R2[5]) * self._gyro_to_rad_s
        gzl = -float(s.imu_L1[5]) * self._gyro_to_rad_s
        szl = -float(s.imu_L2[5]) * self._gyro_to_rad_s
        self._thigh_gy_r = self._lpf(self._thigh_gy_r, gzr, self.imu_gyro_filter_tau)
        self._shank_gy_r = self._lpf(self._shank_gy_r, szr, self.imu_gyro_filter_tau)
        self._thigh_gy_l = self._lpf(self._thigh_gy_l, gzl, self.imu_gyro_filter_tau)
        self._shank_gy_l = self._lpf(self._shank_gy_l, szl, self.imu_gyro_filter_tau)
        qd_r = -self._thigh_gy_r + self._shank_gy_r
        qd_l = -self._thigh_gy_l + self._shank_gy_l
        return float(qd_r), float(qd_l)

    def _norm_pair(self, q: float, qd: float, pm: float, ps: float, vm: float, vs: float) -> Tuple[float, float]:
        return float((q - pm) / ps), float((qd - vm) / vs)

    def _drain_out_q(self) -> None:
        try:
            while True:
                self._out_q.get_nowait()
        except Empty:
            pass

    def _send_latest(self, x_r: np.ndarray, x_l: np.ndarray) -> None:
        payload = {"r": x_r, "l": x_l}
        try:
            self._in_q.put_nowait(payload)
        except Exception:
            try:
                self._in_q.get_nowait()
            except Exception:
                pass
            try:
                self._in_q.put_nowait(payload)
            except Exception:
                pass

    def _try_recv(self, timeout_s: float = 0.002) -> Tuple[float, float]:
        try:
            y_r, y_l = self._out_q.get(timeout=timeout_s)
            mr = float(np.asarray(y_r).reshape(-1)[0])
            ml = float(np.asarray(y_l).reshape(-1)[0])
            self._last_m_r, self._last_m_l = mr, ml
            return mr, ml
        except Empty:
            return self._last_m_r, self._last_m_l

    def step(self, s: Sensors) -> CtrlResult:
        q_r, qd_r_enc, q_l, qd_l_enc = self._exo_knee_encoder(s)
        self._q_r_f = self._lpf(self._q_r_f, q_r, self.knee_filter_tau)
        self._q_l_f = self._lpf(self._q_l_f, q_l, self.knee_filter_tau)

        if self._vel_source == "encoder":
            qd_r_raw, qd_l_raw = qd_r_enc, qd_l_enc
            self._qd_r_f = self._lpf(self._qd_r_f, qd_r_raw, self.knee_filter_tau)
            self._qd_l_f = self._lpf(self._qd_l_f, qd_l_raw, self.knee_filter_tau)
        else:
            qd_r_raw, qd_l_raw = self._imu_knee_rates_rad_s(s)
            self._qd_r_f = float(qd_r_raw)
            self._qd_l_f = float(qd_l_raw)

        q_r_m = self._io_lp["q_r"].step(self._q_r_f)
        q_l_m = self._io_lp["q_l"].step(self._q_l_f)
        qd_r_m = self._io_lp["qd_r"].step(self._qd_r_f)
        qd_l_m = self._io_lp["qd_l"].step(self._qd_l_f)

        nr_r, ndr_r = self._norm_pair(q_r_m, qd_r_m, self._pr_m, self._pr_s, self._vr_m, self._vr_s)
        nr_l, ndr_l = self._norm_pair(q_l_m, qd_l_m, self._pl_m, self._pl_s, self._vl_m, self._vl_s)

        seq_r = self.x_r.push_last(np.array([nr_r, ndr_r], dtype=np.float32))
        seq_l = self.x_l.push_last(np.array([nr_l, ndr_l], dtype=np.float32))
        x_r = seq_r.reshape(1, 2, self.frame_length)
        x_l = seq_l.reshape(1, 2, self.frame_length)

        self._drain_out_q()
        self._send_latest(x_r, x_l)
        m_r_raw, m_l_raw = self._try_recv(timeout_s=0.002)
        m_r = self._io_lp["m_r"].step(m_r_raw)
        m_l = self._io_lp["m_l"].step(m_l_raw)

        tau_r = self.torque_sign_r * m_r * self.moment_mass_kg
        tau_l = self.torque_sign_l * m_l * self.moment_mass_kg

        # ``torque_cmd_*`` / ``applied_*``: N·m after ``moment_mass_kg``; ``main_knee`` applies ``scale`` / limit → ``cmd_*``.
        extra = {
            "knee_angle_r": float(q_r_m),
            "knee_angle_l": float(q_l_m),
            "knee_angle_r_u": float(qd_r_m),
            "knee_angle_l_u": float(qd_l_m),
            "knee_encoder_vel_r": float(qd_r_enc),
            "knee_encoder_vel_l": float(qd_l_enc),
            "joint_velocity_source": self._vel_source,
            "moment_nm_kg_r": float(m_r),
            "moment_nm_kg_l": float(m_l),
            "torque_cmd_r": float(tau_r),
            "torque_cmd_l": float(tau_l),
            "ik_index_r": int(self._ik_r),
            "ik_index_l": int(self._ik_l),
        }
        if self._vel_source == "imu_gyro_delta":
            extra["knee_r_u_gyr"] = float(-self._thigh_gy_r + self._shank_gy_r)
            extra["knee_l_u_gyr"] = float(-self._thigh_gy_l + self._shank_gy_l)

        return CtrlResult(
            model_out_R=float(m_r),
            model_out_L=float(m_l),
            applied_R=float(tau_r),
            applied_L=float(tau_l),
            extra=extra,
        )

    def start(self) -> None:
        self._worker.start()

    def close(self) -> None:
        try:
            self._in_q.put_nowait(None)
        except Exception:
            pass
        try:
            self._worker.join(timeout=1.5)
        except Exception:
            pass
        for q in (self._out_q, self._in_q):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

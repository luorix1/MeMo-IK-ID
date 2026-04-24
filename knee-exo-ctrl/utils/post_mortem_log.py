"""
Write a JSON sidecar after a knee control run for post-mortem debugging.

Intended to be called once after ``DualKneeRunner.shutdown()`` and trial
``np.savez`` (see ``main_knee.py``). Safe to call from ``atexit`` as a fallback.
"""
from __future__ import annotations

import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = "knee_exo_post_mortem_v1"


def json_safe_for_log(x: Any) -> Any:
    """Recursively convert values to JSON-serializable forms."""
    import numpy as np

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): json_safe_for_log(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe_for_log(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return str(x)


def write_post_mortem_json(
    runner: Any,
    *,
    npz_path: Optional[str] = None,
    note: Optional[str] = None,
) -> Optional[Path]:
    """
    Write ``{trial_name}_post_mortem.json`` in the process current working directory.

    Idempotent: returns ``None`` if already written for this runner instance.
    """
    if getattr(runner, "_pm_written", False):
        return None

    trial = str(runner.cfg.get("trial_name", "trial"))
    out_path = Path.cwd() / f"{trial}_post_mortem.json"

    lt = getattr(runner, "_loop_timing", None) or {}
    n_loops = int(lt.get("n", 0))
    sum_dt = float(lt.get("sum_dt", 0.0))
    min_dt = lt.get("min_dt")
    max_dt = lt.get("max_dt", 0.0)
    mean_dt = (sum_dt / n_loops) if n_loops > 0 else None

    wall_end = datetime.now(timezone.utc).isoformat()
    t_start = getattr(runner, "_pm_run_start_perf", None)
    duration_perf: Optional[float] = None
    if t_start is not None:
        import time

        duration_perf = float(time.perf_counter() - t_start)

    payload: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "written_at_wall_utc": wall_end,
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "trial_name": trial,
        "npz_path": npz_path,
        "n_samples_saved": int(getattr(runner, "current_idx", 0)),
        "exit_reason": getattr(runner, "_pm_exit_reason", "unknown"),
        "exception_message": getattr(runner, "_pm_exception_message", None),
        "exception_traceback": getattr(runner, "_pm_exception_traceback", None),
        "wall_clock": {
            "run_start_utc": getattr(runner, "_pm_wall_iso_start", None),
            "log_end_utc": wall_end,
            "duration_perf_counter_s": duration_perf,
        },
        "loop_timing_s": {
            "n_intervals": n_loops,
            "mean_dt": mean_dt,
            "min_dt": float(min_dt) if min_dt is not None and min_dt != float("inf") else None,
            "max_dt": float(max_dt) if n_loops > 0 else None,
        },
        "last_tick": json_safe_for_log(getattr(runner, "_last_tick", {}) or {}),
        "config": json_safe_for_log(dict(getattr(runner, "cfg", {}) or {})),
    }
    if note:
        payload["note"] = note

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    runner._pm_written = True
    print(f"=== Post-mortem log: {out_path} ===")
    return out_path

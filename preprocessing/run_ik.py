"""Step 3 — Run OpenSim Inverse Kinematics on all marker TRC files.

For each trial-type folder (awinda, hip-exo, knee-exo) an appropriate
OpenSim model is located in output/opensim/ and IK is solved for every
marker/*.trc file.  Results are written to ik/{trial}_ik.mot.

Marker weights
--------------
  awinda   : all markers at weight 1.0  (no-exo condition)
  hip-exo  : LPSI, RPSI → 0.0  (hidden under exo frame)
             RLFC, LLFC → 0.1  (partially obstructed)
  knee-exo : same as hip-exo

Usage (from project root):
    python code/run_ik.py
    python code/run_ik.py --output-dir data/AB01_Jinwoo
    python code/run_ik.py --output-dir data/AB01_Jinwoo --type hip-exo
    python code/run_ik.py --output-dir data/AB01_Jinwoo --trial 0p8mps_lg

Note (Mac Apple Silicon):
    OpenSim 4.5 bindings are x86_64.  Run via:
        PYTHONPATH="<opensim_sdk_python_path>" \\
          arch -x86_64 <x86_python> code/run_ik.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import opensim as osim

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    FOLDER_TYPE_TO_OUTPUT,
    IK_ACCURACY,
    MARKER_WEIGHTS,
    find_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "AB01_Jinwoo"


# ---------------------------------------------------------------------------
# IK helpers
# ---------------------------------------------------------------------------

def _marker_names(trc_file: Path) -> list[str]:
    data = osim.MarkerData(str(trc_file))
    names = data.getMarkerNames()
    return [names.get(i) for i in range(names.getSize())]


def _time_range(trc_file: Path) -> tuple[float, float]:
    data = osim.MarkerData(str(trc_file))
    return float(data.getStartFrameTime()), float(data.getLastFrameTime())


def _build_tasks(
    trc_file: Path,
    zero_weight: set[str],
    tiny_weight: set[str],
    small_weight: set[str],
) -> osim.IKTaskSet:
    task_set = osim.IKTaskSet()
    for name in _marker_names(trc_file):
        task = osim.IKMarkerTask()
        task.setName(name)
        task.setApply(True)
        if name in zero_weight:
            task.setWeight(0.0)
        elif name in tiny_weight:
            task.setWeight(0.01)
        elif name in small_weight:
            task.setWeight(0.1)
        else:
            task.setWeight(1.0)
        task_set.adoptAndAppend(task)
    return task_set


def _run_ik(
    trc_file: Path,
    model_file: Path,
    output_file: Path,
    zero_weight: set[str],
    tiny_weight: set[str],
    small_weight: set[str],
) -> None:
    start, end = _time_range(trc_file)
    model = osim.Model(str(model_file))

    tool = osim.InverseKinematicsTool()
    tool.setName(output_file.stem)
    tool.setModel(model)
    tool.setMarkerDataFileName(str(trc_file))
    tool.setStartTime(start)
    tool.setEndTime(end)
    tool.setOutputMotionFileName(str(output_file))
    tool.setResultsDir(str(output_file.parent))
    tool.set_report_errors(False)
    tool.set_accuracy(IK_ACCURACY)
    tool.set_IKTaskSet(_build_tasks(trc_file, zero_weight, tiny_weight, small_weight))
    tool.run()


# ---------------------------------------------------------------------------
# Per-subject IK runner
# ---------------------------------------------------------------------------

def run_ik_for_output(
    output_dir: Path,
    type_filter: str | None = None,
    trial_filter: str | None = None,
) -> None:
    """Run IK for every trial in every trial-type folder (optionally filtered)."""
    opensim_dir = output_dir / "opensim"

    for folder_type, output_name in FOLDER_TYPE_TO_OUTPUT.items():
        if type_filter and type_filter.lower() != output_name:
            continue

        type_dir    = output_dir / output_name
        marker_dir  = type_dir / "marker"
        ik_dir      = type_dir / "ik"

        if not marker_dir.is_dir():
            print(f"[{output_name}] marker/ not found — skipping.")
            continue

        model_file = find_model(opensim_dir, folder_type)
        if model_file is None:
            print(f"[{output_name}] No model found in opensim/ for type '{folder_type}' — skipping IK.")
            continue

        zero_w, tiny_w, small_w = MARKER_WEIGHTS[folder_type]
        ik_dir.mkdir(parents=True, exist_ok=True)

        trc_files = sorted(marker_dir.glob("*.trc"))
        if trial_filter:
            trc_files = [f for f in trc_files if trial_filter.lower() in f.stem.lower()]

        if not trc_files:
            print(f"[{output_name}] No TRC files matched — skipping.")
            continue

        for trc_file in trc_files:
            trial       = trc_file.stem
            output_file = ik_dir / f"{trial}_ik.mot"

            print(f"\n[{output_name}] IK  {trc_file.name}  →  {output_file.name}", flush=True)
            print(f"  model : {model_file.name}", flush=True)
            print(f"  zero  : {sorted(zero_w) or '—'}", flush=True)
            print(f"  tiny  : {sorted(tiny_w) or '—'}", flush=True)
            print(f"  small : {sorted(small_w) or '—'}", flush=True)

            _run_ik(trc_file, model_file, output_file, zero_w, tiny_w, small_w)

    print("\nIK complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OpenSim IK on marker TRC files.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output directory. Default: {DEFAULT_OUTPUT}")
    p.add_argument("--type",  dest="type_filter",
                   help="Restrict to one trial-type folder, e.g. hip-exo.")
    p.add_argument("--trial", dest="trial_filter",
                   help="Restrict to trials whose name contains this string.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_ik_for_output(args.output_dir, args.type_filter, args.trial_filter)


if __name__ == "__main__":
    main()

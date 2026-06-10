"""Step 4 — Run OpenSim Inverse Dynamics from IK + GRF data.

For each trial-type folder (awinda, hip-exo, knee-exo), every non-static
ik/{trial}_ik.mot is paired with grf/{trial}_grf.xml and run through the
InverseDynamicsTool.  Results are written to id/{trial}_id.sto.

IK coordinates are low-pass filtered at 6 Hz inside the tool.
Muscle forces are excluded (forces_to_exclude = Muscles).

Usage (from project root):
    python code/run_id.py
    python code/run_id.py --output-dir data/AB01_Jinwoo
    python code/run_id.py --output-dir data/AB01_Jinwoo --type hip-exo
    python code/run_id.py --output-dir data/AB01_Jinwoo --trial 0p8mps_lg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import opensim as osim

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    FOLDER_TYPE_TO_OUTPUT,
    FORCES_TO_EXCLUDE,
    LOWPASS_CUTOFF_HZ,
    find_model,
    is_static_trial,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "AB01_Jinwoo"


# ---------------------------------------------------------------------------
# ID helper
# ---------------------------------------------------------------------------

def _time_range(mot_file: Path) -> tuple[float, float]:
    storage = osim.Storage(str(mot_file))
    return float(storage.getFirstTime()), float(storage.getLastTime())


def _run_id(
    ik_file: Path,
    grf_xml: Path,
    model_file: Path,
    output_file: Path,
) -> None:
    start, end = _time_range(ik_file)

    excluded = osim.ArrayStr()
    for f in FORCES_TO_EXCLUDE:
        excluded.append(f)

    tool = osim.InverseDynamicsTool()
    tool.setName(output_file.stem)
    tool.setModelFileName(str(model_file))
    tool.setCoordinatesFileName(str(ik_file))
    tool.setExternalLoadsFileName(str(grf_xml))
    tool.setStartTime(start)
    tool.setEndTime(end)
    tool.setLowpassCutoffFrequency(LOWPASS_CUTOFF_HZ)
    tool.setExcludedForces(excluded)
    tool.setResultsDir(str(output_file.parent))
    tool.setOutputGenForceFileName(output_file.name)
    tool.run()


# ---------------------------------------------------------------------------
# Per-subject ID runner
# ---------------------------------------------------------------------------

def run_id_for_output(
    output_dir: Path,
    type_filter: str | None = None,
    trial_filter: str | None = None,
) -> None:
    """Run ID for every non-static IK result in every trial-type folder."""
    opensim_dir = output_dir / "opensim"

    for folder_type, output_name in FOLDER_TYPE_TO_OUTPUT.items():
        if type_filter and type_filter.lower() != output_name:
            continue

        type_dir = output_dir / output_name
        ik_dir   = type_dir / "ik"
        grf_dir  = type_dir / "grf"
        id_dir   = type_dir / "id"

        if not ik_dir.is_dir():
            print(f"[{output_name}] ik/ not found — skipping.")
            continue

        model_file = find_model(opensim_dir, folder_type)
        if model_file is None:
            print(f"[{output_name}] No model found in opensim/ for type '{folder_type}' — skipping ID.")
            continue

        id_dir.mkdir(parents=True, exist_ok=True)

        ik_files = sorted(ik_dir.glob("*_ik.mot"))
        ik_files = [f for f in ik_files if not is_static_trial(f.stem)]
        if trial_filter:
            ik_files = [f for f in ik_files if trial_filter.lower() in f.stem.lower()]

        if not ik_files:
            print(f"[{output_name}] No non-static IK files matched — skipping.")
            continue

        for ik_file in ik_files:
            # strip _ik suffix to get bare trial name
            trial = ik_file.stem[:-3]
            grf_xml = grf_dir / f"{trial}_grf.xml"

            if not grf_xml.is_file():
                print(f"[{output_name}] GRF XML missing for '{trial}' — skipping.")
                continue

            output_file = id_dir / f"{trial}_id.sto"
            print(f"\n[{output_name}] ID  {ik_file.name}  →  {output_file.name}", flush=True)
            print(f"  model : {model_file.name}", flush=True)
            print(f"  grf   : {grf_xml.name}", flush=True)
            print(f"  LP    : {LOWPASS_CUTOFF_HZ:g} Hz   exclude: {', '.join(FORCES_TO_EXCLUDE)}", flush=True)

            _run_id(ik_file, grf_xml, model_file, output_file)

    print("\nID complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OpenSim ID on IK + GRF results.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output directory. Default: {DEFAULT_OUTPUT}")
    p.add_argument("--type",  dest="type_filter",
                   help="Restrict to one trial-type folder, e.g. hip-exo.")
    p.add_argument("--trial", dest="trial_filter",
                   help="Restrict to trials whose name contains this string.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id_for_output(args.output_dir, args.type_filter, args.trial_filter)


if __name__ == "__main__":
    main()

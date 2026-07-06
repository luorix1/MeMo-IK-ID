"""One-off: reprocess AB01 Jinwoo knee-exo with re-scaled knee models.

Sources
-------
  OSIM   : /media/metamobility3/Samsung_T52/Results/osim_knee_scaled/
  Trials : /media/metamobility3/Samsung_T52/Results/ab01_jinwoo_knee/
           (.trc / .mot / .csv — knee-exo trials only)

Output
------
  /media/metamobility3/Samsung_T52/Results/AB01_Jinwoo_knee/
  (standalone knee-exo processed tree; does not modify processed/AB01_Jinwoo)

Usage
-----
    python run_ab01_knee_scaled.py
    python run_ab01_knee_scaled.py --dry-run
    python run_ab01_knee_scaled.py --skip-ik --skip-id
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_OSIM_DIR = Path("/media/metamobility3/Samsung_T52/Results/osim_knee_scaled")
DEFAULT_TRIALS_DIR = Path("/media/metamobility3/Samsung_T52/Results/ab01_jinwoo_knee")
DEFAULT_OUTPUT_DIR = Path("/media/metamobility3/Samsung_T52/Results/AB01_Jinwoo_knee")

SUBJECT_ID = "ab01_jinwoo"
TRIAL_EXTENSIONS = {".trc", ".mot", ".csv"}
_SEP = "=" * 60


def _header(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


def is_knee_exo_trial(filename: str) -> bool:
    """Return True for knee-exo session files, excluding no-exo static exports."""
    name = filename.lower()
    return "_knee_" in name and "_no_exo_" not in name


def stage_subject_root(
    staging_root: Path,
    osim_dir: Path,
    trials_dir: Path,
    dry_run: bool,
) -> None:
    """Build a minimal cascade-style subject root for organise.py."""
    knee_exo_dir = staging_root / f"{SUBJECT_ID}_knee_exo"

    osim_files = sorted(osim_dir.glob("*knee_exo*.osim"))
    if not osim_files:
        raise FileNotFoundError(f"No knee-exo .osim files found in {osim_dir}")

    trial_files = sorted(
        f for f in trials_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in TRIAL_EXTENSIONS
        and is_knee_exo_trial(f.name)
    )
    if not trial_files:
        raise FileNotFoundError(f"No knee-exo trial files found in {trials_dir}")

    print(f"Staging → {staging_root}")
    print(f"  osim   ({len(osim_files)}): {', '.join(f.name for f in osim_files)}")
    print(f"  trials ({len(trial_files)}): {', '.join(f.name for f in trial_files)}")

    if dry_run:
        return

    staging_root.mkdir(parents=True, exist_ok=True)
    knee_exo_dir.mkdir(parents=True, exist_ok=True)

    for osim_file in osim_files:
        shutil.copy2(osim_file, staging_root / osim_file.name)

    for trial_file in trial_files:
        shutil.copy2(trial_file, knee_exo_dir / trial_file.name)


def remove_stale_knee_models(opensim_dir: Path, dry_run: bool) -> None:
    """Remove stale knee-exo *_added_inertia.osim so it is regenerated from before_inertia."""
    if not opensim_dir.is_dir():
        return

    for model in sorted(opensim_dir.glob("*knee_exo*_added_inertia.osim")):
        if dry_run:
            print(f"  [dry] remove {model.name}")
        else:
            model.unlink()
            print(f"  removed {model.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--osim-dir", type=Path, default=DEFAULT_OSIM_DIR)
    p.add_argument("--trials-dir", type=Path, default=DEFAULT_TRIALS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--staging-dir", type=Path, default=None,
                   help="Cascade staging folder (default: temp dir, removed after run).")
    p.add_argument("--keep-staging", action="store_true",
                   help="Keep the staging directory after the run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned staging only; do not copy or run OpenSim.")
    p.add_argument("--skip-exo-inertia", action="store_true")
    p.add_argument("--skip-grf-xml", action="store_true")
    p.add_argument("--skip-repair", action="store_true")
    p.add_argument("--skip-ik", action="store_true")
    p.add_argument("--skip-id", action="store_true")
    p.add_argument("--trial", dest="trial_filter",
                   help="Restrict IK/ID to trials whose name contains this string.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    type_filter = "knee-exo"

    owns_staging = args.staging_dir is None
    if owns_staging:
        staging_ctx = tempfile.TemporaryDirectory(prefix="ab01_knee_staging_")
        # organise.py derives subject_lower from subject_root.name — must be ab01_jinwoo.
        staging_root = Path(staging_ctx.name) / SUBJECT_ID
    else:
        staging_ctx = None
        staging_root = args.staging_dir

    try:
        stage_subject_root(staging_root, args.osim_dir, args.trials_dir, args.dry_run)
        if args.dry_run:
            print("\nDry run complete — no files written, pipeline not executed.")
            return

        _header("STEP 1 — Organise source files")
        from organise import organise
        organise(staging_root, args.output_dir, dry_run=False)

        if not args.skip_exo_inertia:
            _header("STEP 2 — Add exoskeleton mass & inertia (knee-exo models)")
            remove_stale_knee_models(args.output_dir / "opensim", dry_run=False)
            from add_exo_inertia import add_exo_inertia_for_output
            add_exo_inertia_for_output(args.output_dir)

        if not args.skip_grf_xml:
            _header("STEP 3 — Create GRF XML files")
            from create_grf_xml import create_grf_xmls
            created = create_grf_xmls(args.output_dir)
            print(f"\nCreated {len(created)} GRF XML file(s).")

        if not args.skip_repair:
            _header("STEP 4 — Repair malformed TRC files")
            from repair_trc import repair_output_dir
            n = repair_output_dir(args.output_dir)
            print(f"\nRepaired {n} TRC file(s).")

        if not args.skip_ik:
            _header("STEP 5 — Inverse Kinematics (knee-exo)")
            from run_ik import run_ik_for_output
            run_ik_for_output(args.output_dir, type_filter, args.trial_filter)

        if not args.skip_id:
            _header("STEP 6 — Inverse Dynamics (knee-exo)")
            from run_id import run_id_for_output
            run_id_for_output(args.output_dir, type_filter, args.trial_filter)

        _header(f"Pipeline complete  →  {args.output_dir}")

    finally:
        if owns_staging and not args.keep_staging:
            if staging_ctx is not None:
                staging_ctx.cleanup()
        elif args.keep_staging and not args.dry_run:
            print(f"\nStaging kept at {staging_root}")


if __name__ == "__main__":
    main()

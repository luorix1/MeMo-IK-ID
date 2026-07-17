"""Process a flat Vicon session folder through the OpenSim pipeline.

Unlike cascade/ subjects, these Results folders keep trials + session-scaled
.osim files in one flat directory (same layout as test_knee / ab02_oscar_redo).

This script stages a temporary cascade subject root, then runs the standard
pipeline into a standalone ``*_processed`` output (does not touch processed/).

Presets
-------
  ab05_maria_knee   → knee-exo only
  ab07_amy_knee     → knee-exo only
  ab02_oscar_redo   → awinda + hip-exo + knee-exo

Usage
-----
    python run_flat_session.py --preset ab05_maria_knee
    python run_flat_session.py --preset ab02_oscar_redo --dry-run
    python run_flat_session.py --source-dir ... --subject-id ab05_maria --output-dir ...
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reformat_results import classify_trial_type

RESULTS_ROOT = Path("/media/metamobility3/Samsung_T52/Results")
TRIAL_EXTENSIONS = {".trc", ".mot", ".csv"}
_SEP = "=" * 60

# preset → (source_dir_name, subject_id, output_dir_name)
PRESETS: dict[str, tuple[str, str, str]] = {
    "ab05_maria_knee": ("ab05_maria_knee", "ab05_maria", "ab05_maria_knee_processed"),
    "ab07_amy_knee":   ("ab07_amy_knee",   "ab07_amy",   "ab07_amy_knee_processed"),
    "ab02_oscar_redo": ("ab02_oscar_redo", "ab02_oscar", "ab02_oscar_redo_processed"),
}

FOLDER_TYPE_TO_STAGING = {
    "awinda":   "awinda",
    "hip_exo":  "hip_exo",
    "knee_exo": "knee_exo",
}


def _header(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


def is_usable_trial(filename: str, trial_type: str) -> bool:
    """Drop no-exo static exports from exo sessions; keep awinda no_exo trials."""
    name = filename.lower()
    if trial_type == "knee_exo":
        return "_knee_" in name and "_no_exo_" not in name
    if trial_type == "hip_exo":
        return "_hip_" in name and "_no_exo_" not in name
    if trial_type == "awinda":
        return "_awinda_" in name or name.endswith("_awinda_static")
    return False


def collect_trials_by_type(source_dir: Path) -> dict[str, list[Path]]:
    by_type: dict[str, list[Path]] = {}
    for f in sorted(source_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in TRIAL_EXTENSIONS:
            continue
        trial_type = classify_trial_type(f.name)
        if trial_type is None or not is_usable_trial(f.name, trial_type):
            continue
        by_type.setdefault(trial_type, []).append(f)
    if not by_type:
        raise FileNotFoundError(f"No recognised trial files in {source_dir}")
    return by_type


def collect_osim_files(source_dir: Path) -> list[Path]:
    """Use session .osim next to the trials (not osim_scaled/)."""
    osim_files = sorted(source_dir.glob("*.osim"))
    if not osim_files:
        raise FileNotFoundError(f"No .osim files found in {source_dir}")
    return osim_files


def stage_subject_root(
    staging_root: Path,
    source_dir: Path,
    subject_id: str,
    dry_run: bool,
) -> dict[str, list[Path]]:
    """Build a cascade-style subject root that organise.py understands."""
    by_type = collect_trials_by_type(source_dir)
    osim_files = collect_osim_files(source_dir)

    print(f"Source  → {source_dir}")
    print(f"Staging → {staging_root}")
    print(f"  osim ({len(osim_files)}): {', '.join(f.name for f in osim_files)}")
    for trial_type, files in sorted(by_type.items()):
        print(f"  {trial_type} ({len(files)}): {', '.join(f.name for f in files)}")

    if dry_run:
        return by_type

    staging_root.mkdir(parents=True, exist_ok=True)
    for osim_file in osim_files:
        shutil.copy2(osim_file, staging_root / osim_file.name)

    for trial_type, files in by_type.items():
        staging_name = FOLDER_TYPE_TO_STAGING[trial_type]
        trial_dir = staging_root / f"{subject_id}_{staging_name}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        for trial_file in files:
            shutil.copy2(trial_file, trial_dir / trial_file.name)

    return by_type


def remove_stale_added_inertia(opensim_dir: Path) -> None:
    """Remove stale *_added_inertia.osim so step 2 regenerates from before_inertia."""
    if not opensim_dir.is_dir():
        return
    for model in sorted(opensim_dir.glob("*_added_inertia.osim")):
        model.unlink()
        print(f"  removed {model.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--preset", choices=sorted(PRESETS),
                   help="Named session preset (sets source / subject / output).")
    p.add_argument("--source-dir", type=Path, default=None)
    p.add_argument("--subject-id", type=str, default=None,
                   help="Subject id used for cascade naming, e.g. ab05_maria.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--staging-dir", type=Path, default=None)
    p.add_argument("--keep-staging", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-exo-inertia", action="store_true")
    p.add_argument("--skip-grf-xml", action="store_true")
    p.add_argument("--skip-repair", action="store_true")
    p.add_argument("--skip-ik", action="store_true")
    p.add_argument("--skip-id", action="store_true")
    p.add_argument("--type", dest="type_filter",
                   help="Restrict IK/ID to one trial-type folder, e.g. knee-exo.")
    p.add_argument("--trial", dest="trial_filter",
                   help="Restrict IK/ID to trials whose name contains this string.")
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, str, Path]:
    if args.preset:
        src_name, subject_id, out_name = PRESETS[args.preset]
        source_dir = args.source_dir or (RESULTS_ROOT / src_name)
        output_dir = args.output_dir or (RESULTS_ROOT / out_name)
        subject_id = args.subject_id or subject_id
    else:
        if not args.source_dir or not args.subject_id or not args.output_dir:
            raise SystemExit(
                "Provide --preset, or all of --source-dir / --subject-id / --output-dir."
            )
        source_dir = args.source_dir
        subject_id = args.subject_id
        output_dir = args.output_dir

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    return source_dir, subject_id, output_dir


def main() -> None:
    args = parse_args()
    source_dir, subject_id, output_dir = resolve_paths(args)

    owns_staging = args.staging_dir is None
    if owns_staging:
        staging_ctx = tempfile.TemporaryDirectory(prefix=f"{subject_id}_flat_staging_")
        staging_root = Path(staging_ctx.name) / subject_id
    else:
        staging_ctx = None
        staging_root = args.staging_dir

    try:
        by_type = stage_subject_root(staging_root, source_dir, subject_id, args.dry_run)
        if args.dry_run:
            print("\nDry run complete — no files written, pipeline not executed.")
            return

        _header("STEP 1 — Organise source files")
        from organise import organise
        organise(staging_root, output_dir, dry_run=False)

        if not args.skip_exo_inertia:
            _header("STEP 2 — Add exoskeleton mass & inertia")
            remove_stale_added_inertia(output_dir / "opensim")
            from add_exo_inertia import add_exo_inertia_for_output
            add_exo_inertia_for_output(output_dir)

        if not args.skip_grf_xml:
            _header("STEP 3 — Create GRF XML files")
            from create_grf_xml import create_grf_xmls
            created = create_grf_xmls(output_dir)
            print(f"\nCreated {len(created)} GRF XML file(s).")

        if not args.skip_repair:
            _header("STEP 4 — Repair malformed TRC files")
            from repair_trc import repair_output_dir
            n = repair_output_dir(output_dir)
            print(f"\nRepaired {n} TRC file(s).")

        # Default type filter: if only knee trials staged, restrict to knee-exo.
        type_filter = args.type_filter
        if type_filter is None and set(by_type) == {"knee_exo"}:
            type_filter = "knee-exo"

        if not args.skip_ik:
            _header("STEP 5 — Inverse Kinematics")
            from run_ik import run_ik_for_output
            run_ik_for_output(output_dir, type_filter, args.trial_filter)

        if not args.skip_id:
            _header("STEP 6 — Inverse Dynamics")
            from run_id import run_id_for_output
            run_id_for_output(output_dir, type_filter, args.trial_filter)

        _header(f"Pipeline complete  →  {output_dir}")

    finally:
        if owns_staging and not args.keep_staging:
            if staging_ctx is not None:
                staging_ctx.cleanup()
        elif args.keep_staging and not args.dry_run:
            print(f"\nStaging kept at {staging_root}")


if __name__ == "__main__":
    main()

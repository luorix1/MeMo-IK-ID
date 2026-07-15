"""One-off: process flat test_knee Vicon export through the OpenSim pipeline.

test_knee is not cascade-shaped.  Layout on disk::

    test_knee/                          flat Vicon session files (optional)
    test_knee/ab08_seokhyun_knee/       same trials + session-scaled .osim
                                        (folder suffix is ``_knee``, not ``_knee_exo``)

This script stages a temporary cascade subject root, then runs the standard
pipeline steps into ``test_knee_processed/``.

Sources
-------
  Trials / OSIM : /media/metamobility3/Samsung_T52/Results/test_knee/
                  (prefers nested ab08_seokhyun_knee/; falls back to flat root)

Output
------
  /media/metamobility3/Samsung_T52/Results/test_knee_processed/
  (knee-exo tree only; mirrors AB01_Jinwoo_knee layout)

Usage
-----
    python run_test_knee.py
    python run_test_knee.py --dry-run
    python run_test_knee.py --skip-ik --skip-id
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_SOURCE_DIR = Path("/media/metamobility3/Samsung_T52/Results/test_knee")
DEFAULT_OUTPUT_DIR = Path("/media/metamobility3/Samsung_T52/Results/test_knee_processed")

SUBJECT_ID = "ab08_seokhyun"
NESTED_SESSION_DIRNAME = "ab08_seokhyun_knee"
TRIAL_EXTENSIONS = {".trc", ".mot", ".csv"}
_SEP = "=" * 60


def _header(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


def resolve_source_dirs(source_dir: Path) -> tuple[Path, Path | None]:
    """Return (trials_dir, nested_dir_or_None).

    Prefer the nested session folder when it exists; otherwise use the flat root.
    """
    nested = source_dir / NESTED_SESSION_DIRNAME
    if nested.is_dir():
        return nested, nested
    return source_dir, None


def is_knee_exo_trial(filename: str) -> bool:
    """Keep knee-exo session files; drop no-exo static / unrelated exports."""
    name = filename.lower()
    return "_knee_" in name and "_no_exo_" not in name


def collect_trial_files(trials_dir: Path) -> list[Path]:
    files = sorted(
        f for f in trials_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in TRIAL_EXTENSIONS
        and is_knee_exo_trial(f.name)
    )
    if not files:
        raise FileNotFoundError(f"No knee-exo trial files (.trc/.mot/.csv) in {trials_dir}")
    return files


def collect_osim_files(source_dir: Path, nested_dir: Path | None) -> list[Path]:
    """Prefer session .osim next to the trials; do not fall back to osim_scaled.

    These models differ from Results/osim_scaled for this recapture.
    """
    search_dirs: list[Path] = []
    if nested_dir is not None:
        search_dirs.append(nested_dir)
    search_dirs.append(source_dir)

    found: dict[str, Path] = {}
    for d in search_dirs:
        for pattern in ("*knee_exo*.osim", "*_no_exo.osim"):
            for osim_file in sorted(d.glob(pattern)):
                found.setdefault(osim_file.name, osim_file)

    osim_files = sorted(found.values(), key=lambda p: p.name)
    if not any("knee_exo" in p.name for p in osim_files):
        raise FileNotFoundError(
            f"No knee-exo .osim found under {source_dir} "
            f"(expected e.g. {SUBJECT_ID}_knee_exo_before_inertia.osim)"
        )
    return osim_files


def stage_subject_root(
    staging_root: Path,
    source_dir: Path,
    dry_run: bool,
) -> None:
    """Build a cascade-style subject root that organise.py understands."""
    trials_dir, nested_dir = resolve_source_dirs(source_dir)
    knee_exo_dir = staging_root / f"{SUBJECT_ID}_knee_exo"

    trial_files = collect_trial_files(trials_dir)
    osim_files = collect_osim_files(source_dir, nested_dir)

    print(f"Source trials → {trials_dir}")
    print(f"Staging       → {staging_root}")
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
    """Remove stale *_added_inertia.osim so step 2 regenerates from before_inertia."""
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
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
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

    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    owns_staging = args.staging_dir is None
    if owns_staging:
        staging_ctx = tempfile.TemporaryDirectory(prefix="test_knee_staging_")
        # organise.py derives subject_lower from subject_root.name — must be ab08_seokhyun.
        staging_root = Path(staging_ctx.name) / SUBJECT_ID
    else:
        staging_ctx = None
        staging_root = args.staging_dir

    try:
        stage_subject_root(staging_root, args.source_dir, args.dry_run)
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

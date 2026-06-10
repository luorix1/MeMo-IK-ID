"""Reformat Samsung_T52/Results into cascade-style source trees for pipeline.py.

The pipeline expects each subject root to contain:
  {subject_id}_awinda/    Vicon no-exo trials
  {subject_id}_hip_exo/   Vicon hip-exo trials
  {subject_id}_knee_exo/  Vicon knee-exo trials
  *.osim                  scaled OpenSim models at the subject root

Samsung_T52/Results stores trials flat per session folder (sometimes split
across *_no_exo and *_exo directories).  This script merges those sources,
sorts files into the expected sub-folders, and copies scaled models from
osim_scaled/.

Usage:
    python reformat_results.py
    python reformat_results.py --subject ab02_oscar
    python reformat_results.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_RESULTS_ROOT = Path("/media/metamobility3/Samsung_T52/Results")
DEFAULT_CASCADE_ROOT = DEFAULT_RESULTS_ROOT / "cascade"

# subject_id → (display name, source session dirs relative to Results root)
SUBJECT_SOURCES: dict[str, tuple[str, list[str]]] = {
    "ab01_jinwoo":     ("AB01_Jinwoo",     ["ab01_jinwoo"]),
    "ab02_oscar":      ("AB02_Oscar",      ["ab02_oscar_no_exo", "ab02_oscar_exo"]),
    "ab03_ilseung":    ("AB03_Ilseung",    ["ab03_ilseung_no_exo", "ab03_ilseung_exo"]),
    "ab04_changseob":  ("AB04_Changseob",  ["ab04_changseob_no_exo", "ab04_changseob_exo"]),
    "ab05_maria":      ("AB05_Maria",      ["ab05_maria"]),
    "ab06_jimin":      ("AB06_Jimin",      ["ab06_jimin"]),
    "ab07_amy":        ("AB07_Amy",        ["ab07_amy"]),
    "ab08_seokhyun":   ("AB08_Seokhyun",   ["ab08_seokhyun"]),
}

COPY_EXTENSIONS = {".trc", ".mot", ".csv"}
SKIP_EXTENSIONS = {".c3d", ".x1d", ".x2d", ".xcp", ".history", ".enf", ".system", ".mp", ".vsk", ".patient"}


def classify_trial_type(filename: str) -> str | None:
    """Return awinda / hip_exo / knee_exo from a Vicon filename, or None."""
    name = filename.lower()
    if "_awinda_" in name or name.endswith("_awinda_static"):
        return "awinda"
    if "_hip_" in name or name.endswith("_hip_static"):
        return "hip_exo"
    if "_knee_" in name or name.endswith("_knee_static"):
        return "knee_exo"
    return None


def reformat_subject(
    subject_id: str,
    results_root: Path,
    cascade_root: Path,
    dry_run: bool,
) -> Path:
    display_name, source_names = SUBJECT_SOURCES[subject_id]
    subject_out = cascade_root / subject_id

    if not dry_run:
        subject_out.mkdir(parents=True, exist_ok=True)

    # Copy scaled models from osim_scaled/
    osim_dir = results_root / "osim_scaled"
    copied_osim = 0
    for osim_file in sorted(osim_dir.glob(f"{subject_id}_*.osim")):
        dst = subject_out / osim_file.name
        if dry_run:
            print(f"  [dry] osim  {osim_file.name}  →  {dst.relative_to(cascade_root)}")
        else:
            shutil.copy2(osim_file, dst)
            print(f"  copied osim  {osim_file.name}")
        copied_osim += 1
    if copied_osim == 0:
        print(f"  [warning] No .osim files found for {subject_id} in osim_scaled/")

    # Copy trial files into typed sub-folders
    counts: dict[str, int] = {"awinda": 0, "hip_exo": 0, "knee_exo": 0}
    for source_name in source_names:
        source_dir = results_root / source_name
        if not source_dir.is_dir():
            print(f"  [warning] Source not found: {source_dir}")
            continue

        for src in sorted(source_dir.iterdir()):
            if not src.is_file():
                continue
            ext = src.suffix.lower()
            if ext in SKIP_EXTENSIONS:
                continue
            if ext not in COPY_EXTENSIONS:
                print(f"  [skip] {src.name}")
                continue

            trial_type = classify_trial_type(src.name)
            if trial_type is None:
                print(f"  [skip] unrecognised trial type: {src.name}")
                continue

            dst_dir = subject_out / f"{subject_id}_{trial_type}"
            dst = dst_dir / src.name
            if dry_run:
                print(f"  [dry] {trial_type:<9}  {src.name}")
            else:
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            counts[trial_type] += 1

    print(
        f"  {subject_id} ({display_name}): "
        f"awinda={counts['awinda']}, hip_exo={counts['hip_exo']}, "
        f"knee_exo={counts['knee_exo']}, osim={copied_osim}"
    )
    return subject_out


def reformat_all(
    results_root: Path,
    cascade_root: Path,
    subjects: list[str] | None,
    dry_run: bool,
) -> dict[str, Path]:
    targets = subjects or list(SUBJECT_SOURCES)
    print(f"Results root : {results_root}")
    print(f"Cascade root : {cascade_root}")
    print(f"Subjects     : {', '.join(targets)}\n")

    out: dict[str, Path] = {}
    for subject_id in targets:
        if subject_id not in SUBJECT_SOURCES:
            raise ValueError(f"Unknown subject: {subject_id}")
        print(f"{'='*60}\n{subject_id}")
        out[subject_id] = reformat_subject(subject_id, results_root, cascade_root, dry_run)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--cascade-root", type=Path, default=DEFAULT_CASCADE_ROOT)
    p.add_argument("--subject", action="append", dest="subjects",
                   help="Process only this subject (repeatable). Default: all.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reformat_all(args.results_root, args.cascade_root, args.subjects, args.dry_run)
    print(f"\nDone → {args.cascade_root}")


if __name__ == "__main__":
    main()

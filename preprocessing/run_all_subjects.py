"""Run pipeline.py for every reformatted cascade subject.

Usage:
    python run_all_subjects.py --subject ab02_oscar --trial static
    python run_all_subjects.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from reformat_results import DEFAULT_CASCADE_ROOT, DEFAULT_RESULTS_ROOT, SUBJECT_SOURCES

PIPELINE = Path(__file__).parent / "pipeline.py"
DEFAULT_PROCESSED_ROOT = DEFAULT_RESULTS_ROOT / "processed"


def run_pipeline(
    subject_id: str,
    cascade_root: Path,
    processed_root: Path,
    extra_args: list[str],
) -> int:
    display_name, _ = SUBJECT_SOURCES[subject_id]
    subject_root = cascade_root / subject_id
    output_dir = processed_root / display_name

    if not subject_root.is_dir():
        print(f"[skip] cascade source missing: {subject_root}")
        return 1

    cmd = [
        sys.executable, "-u", str(PIPELINE),
        "--subject-root", str(subject_root),
        "--output-dir", str(output_dir),
        *extra_args,
    ]
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    return subprocess.call(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cascade-root", type=Path, default=DEFAULT_CASCADE_ROOT)
    p.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    p.add_argument("--subject", action="append", dest="subjects",
                   help="Run only this subject (repeatable). Default: all.")
    p.add_argument("--trial", dest="trial_filter", help="Pass --trial to pipeline.py")
    p.add_argument("--type", dest="type_filter", help="Pass --type to pipeline.py")
    p.add_argument("--skip-organise", action="store_true")
    p.add_argument("--skip-exo-inertia", action="store_true")
    p.add_argument("--skip-grf-xml", action="store_true")
    p.add_argument("--skip-repair", action="store_true")
    p.add_argument("--skip-ik", action="store_true")
    p.add_argument("--skip-id", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    subjects = args.subjects or list(SUBJECT_SOURCES)

    extra: list[str] = []
    if args.skip_organise:
        extra.append("--skip-organise")
    if args.skip_exo_inertia:
        extra.append("--skip-exo-inertia")
    if args.skip_grf_xml:
        extra.append("--skip-grf-xml")
    if args.skip_repair:
        extra.append("--skip-repair")
    if args.skip_ik:
        extra.append("--skip-ik")
    if args.skip_id:
        extra.append("--skip-id")
    if args.type_filter:
        extra.extend(["--type", args.type_filter])
    if args.trial_filter:
        extra.extend(["--trial", args.trial_filter])

    failures: list[str] = []
    for subject_id in subjects:
        rc = run_pipeline(subject_id, args.cascade_root, args.processed_root, extra)
        if rc != 0:
            failures.append(subject_id)

    if failures:
        print(f"\nFailed subjects: {', '.join(failures)}")
        raise SystemExit(1)
    print("\nAll subjects completed successfully.")


if __name__ == "__main__":
    main()

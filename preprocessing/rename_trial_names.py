"""One-off script: rename existing output files from {speed}_{dir} to {DIR}_{speed}.

Before: 0p8mps_lg.trc  /  0p8mps_lg_ik.mot  /  0p8mps_lg_grf.xml
After:  LG_0p8mps.trc  /  LG_0p8mps_ik.mot  /  LG_0p8mps_grf.xml

Skips files that are already in the new format or don't match the pattern
(e.g. 'static', opensim model files).

Usage:
    python code/rename_trial_names.py
    python code/rename_trial_names.py --dry-run
    python code/rename_trial_names.py --output-dir data/AB01_Jinwoo
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "AB01_Jinwoo"

# Matches stems like: 0p8mps_lg  /  0p8mps_lg_ik  /  1p2mps_rd_grf
# Groups: (speed)(direction)(optional _suffix)
_OLD = re.compile(r"^(\d+p\d+mps)_([a-z]{2,4})(_.+)?$")


def new_stem(old: str) -> str | None:
    """Return the renamed stem, or None if the file doesn't need renaming."""
    m = _OLD.match(old)
    if not m:
        return None
    speed, direction, rest = m.group(1), m.group(2), m.group(3) or ""
    return f"{direction.upper()}_{speed}{rest}"


def rename_all(output_dir: Path, dry_run: bool) -> None:
    renamed = 0
    for f in sorted(output_dir.rglob("*")):
        if not f.is_file():
            continue
        replacement = new_stem(f.stem)
        if replacement is None:
            continue
        dst = f.with_name(f"{replacement}{f.suffix}")
        if dry_run:
            print(f"  [dry]  {f.relative_to(output_dir)}  →  {dst.name}")
        else:
            f.rename(dst)
            print(f"  renamed  {f.relative_to(output_dir)}  →  {dst.name}")
        renamed += 1

    print(f"\n{'Would rename' if dry_run else 'Renamed'} {renamed} file(s).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rename trial files from speed_dir to DIR_speed format.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rename_all(args.output_dir, args.dry_run)

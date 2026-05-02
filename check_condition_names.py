#!/usr/bin/env python3
"""
Scan a Processed/Jinwoo_* style HDF5 root: for each subject bundle (S###.h5),
list task condition names (top-level groups under /<condition>/<trial_*>/...).

Does not load IK/ID arrays — only traverses the HDF5 group hierarchy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

try:
    import h5py
except ImportError as e:
    raise SystemExit("h5py is required: pip install h5py") from e

SUBJECT_H5_RE = re.compile(r"^S\d{3}\.h5$", re.IGNORECASE)


def condition_names_for_subject(h5_path: Path) -> List[str]:
    """Return sorted condition (task) group names present in one subject H5."""
    names: Set[str] = set()
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Group):
                names.add(str(key))
    return sorted(names, key=lambda s: (s.lower(), s))


def scan_root(root: Path) -> Dict[str, List[str]]:
    """
    Map subject_id (e.g. S001) -> sorted list of condition names.
    Only includes files named S###.h5.
    """
    h5_files = sorted(root.glob("S*.h5"))
    out: Dict[str, List[str]] = {}
    for p in h5_files:
        if not SUBJECT_H5_RE.match(p.name):
            continue
        sid = p.stem.upper()
        try:
            out[sid] = condition_names_for_subject(p)
        except OSError as e:
            print(f"WARN: could not read {p}: {e}", file=sys.stderr)
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="List task condition names per subject under a flat S###.h5 root."
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/media/metamobility3/Samsung_T51/Processed/Jinwoo_EPIC"),
        help="Directory containing S###.h5 subject bundles",
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write mapping subject_id -> [conditions] as JSON to this path",
    )
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    by_subject = scan_root(root)

    if not by_subject:
        print(f"No S###.h5 files found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Root: {root}")
    print(f"Subjects (S###.h5): {len(by_subject)}")
    for sid, conds in by_subject.items():
        print(f"\n{sid}  ({len(conds)} conditions)")
        for c in conds:
            print(f"  {c}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(by_subject, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote JSON: {args.json.resolve()}")


if __name__ == "__main__":
    main()

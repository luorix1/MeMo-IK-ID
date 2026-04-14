#!/usr/bin/env python3
"""
Scan Processed/Jinwoo_* style H5 roots: list unique condition and trial group names
and flag naming convention issues.

Layout: <root>/S###.h5  with groups /<condition_name>/<trial_NN>/...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import h5py
except ImportError as e:
    raise SystemExit("h5py is required: pip install h5py") from e

# MeMo-style trial groups: trial_01, trial_02, ...
TRIAL_LEAF_RE = re.compile(r"^trial_\d+$")
# Subject bundle
SUBJECT_H5_RE = re.compile(r"^S\d{3}\.h5$", re.IGNORECASE)


def _walk_h5_trials(h5_path: Path) -> List[Tuple[str, str, str]]:
    """Return list of (subject_id, condition, trial_leaf) for one file."""
    sid = h5_path.stem.upper()
    out: List[Tuple[str, str, str]] = []
    with h5py.File(h5_path, "r") as f:
        for cond in f.keys():
            g = f[cond]
            if not isinstance(g, h5py.Group):
                continue
            for trial in g.keys():
                tg = g[trial]
                if not isinstance(tg, h5py.Group):
                    continue
                out.append((sid, str(cond), str(trial)))
    return out


def audit_root(root: Path) -> Dict[str, Any]:
    h5_files = sorted(root.glob("S*.h5"))
    issues: List[Dict[str, Any]] = []

    unique_conditions: Set[str] = set()
    unique_trial_leaves: Set[str] = set()
    triples: Set[Tuple[str, str, str]] = set()
    condition_counts: Dict[str, int] = defaultdict(int)
    trial_leaf_counts: Dict[str, int] = defaultdict(int)

    bad_subject_files: List[str] = []
    for p in h5_files:
        if not SUBJECT_H5_RE.match(p.name):
            bad_subject_files.append(str(p))

    for p in h5_files:
        if not SUBJECT_H5_RE.match(p.name):
            continue
        try:
            rows = _walk_h5_trials(p)
        except OSError as e:
            issues.append({"type": "h5_read_error", "path": str(p), "error": str(e)})
            continue

        for sid, cond, trial in rows:
            unique_conditions.add(cond)
            unique_trial_leaves.add(trial)
            triples.add((sid, cond, trial))
            condition_counts[cond] += 1
            trial_leaf_counts[trial] += 1

            if not cond or not cond.strip():
                issues.append({"type": "empty_condition", "file": str(p), "subject": sid})
            if "/" in cond or "\\" in cond:
                issues.append({"type": "condition_has_slash", "file": str(p), "condition": cond})

            if not TRIAL_LEAF_RE.match(trial):
                issues.append(
                    {
                        "type": "trial_leaf_nonstandard",
                        "file": str(p),
                        "subject": sid,
                        "condition": cond,
                        "trial": trial,
                        "expected_pattern": "trial_<digits> (e.g. trial_01)",
                    }
                )

    return {
        "root": str(root.resolve()),
        "n_subject_h5_files": len([p for p in h5_files if SUBJECT_H5_RE.match(p.name)]),
        "n_h5_files_total": len(h5_files),
        "bad_subject_filename": bad_subject_files,
        "n_unique_conditions": len(unique_conditions),
        "n_unique_trial_leaf_names": len(unique_trial_leaves),
        "n_unique_subject_condition_trial_triples": len(triples),
        "unique_conditions_sorted": sorted(unique_conditions),
        "unique_trial_leaves_sorted": sorted(unique_trial_leaves, key=lambda s: (len(s), s)),
        "n_issues": len(issues),
        "issues": issues,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Jinwoo_Final (flat S*.h5) trial naming.")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/media/metamobility3/Samsung_T51/Processed/Jinwoo_Final"),
        help="Directory containing S###.h5 files",
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write full report JSON to this path",
    )
    ap.add_argument(
        "--print-conditions",
        action="store_true",
        help="Print all unique condition names (can be long)",
    )
    ap.add_argument(
        "--print-trial-leaves",
        action="store_true",
        help="Print all unique trial_* leaf names",
    )
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    report = audit_root(root)

    print(f"Root: {report['root']}")
    print(f"Subject H5 files (S###.h5): {report['n_subject_h5_files']}  (all glob S*.h5: {report['n_h5_files_total']})")
    if report["bad_subject_filename"]:
        print(f"WARN: non-standard filenames: {report['bad_subject_filename'][:10]}{'...' if len(report['bad_subject_filename']) > 10 else ''}")
    print(f"Unique condition names: {report['n_unique_conditions']}")
    print(f"Unique trial leaf names (e.g. trial_01): {report['n_unique_trial_leaf_names']}")
    print(f"Unique (subject, condition, trial) triples: {report['n_unique_subject_condition_trial_triples']}")
    print(f"Issues found: {report['n_issues']}")

    if args.print_conditions:
        for c in report["unique_conditions_sorted"]:
            print(f"  cond: {c}")
    if args.print_trial_leaves:
        for t in report["unique_trial_leaves_sorted"]:
            print(f"  trial: {t}")

    if report["issues"]:
        print("\n--- issues (first 50) ---")
        for row in report["issues"][:50]:
            print(json.dumps(row, ensure_ascii=False))
        if len(report["issues"]) > 50:
            print(f"... ({len(report['issues']) - 50} more)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON: {args.json}")

    if report["n_issues"]:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()

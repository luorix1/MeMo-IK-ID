#!/usr/bin/env python3
"""
Scan MeMo subject H5 files and sum IK trial durations for task families:

  - incline_*   (condition name starts with ``incline_``)
  - stair_*
  - levelground_*
  - treadmill_*

Plots a bar chart of total time per family (aggregated across all subjects).

Example:

  python memo_task_duration_composition.py \\
    --memo-root /media/metamobility3/Samsung_T51/Processed/MeMo \\
    --output runs/memo_task_composition.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    import h5py
except ImportError as e:
    raise SystemExit("Install h5py: pip install h5py") from e

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("Install matplotlib: pip install matplotlib") from e

from dataset import _read_h5_opensim_table  # noqa: E402


# Ordered display labels (must match classify_condition keys)
FAMILY_LABELS = ["incline_*", "stair_*", "levelground_*", "treadmill_*"]


def classify_condition(condition_name: str) -> Optional[str]:
    """Map H5 condition group name to a task family, or None if not one of the four."""
    n = (condition_name or "").lower()
    if n.startswith("incline_"):
        return "incline_*"
    if n.startswith("stair_"):
        return "stair_*"
    if n.startswith("levelground_"):
        return "levelground_*"
    if n.startswith("treadmill_"):
        return "treadmill_*"
    return None


def trial_duration_sec_from_ik(trial_group) -> Optional[float]:
    """Duration from first IK table: last(time) - first(time)."""
    if "ik" not in trial_group:
        return None
    ik_group = trial_group["ik"]
    keys = sorted(list(ik_group.keys()))
    if not keys:
        return None
    ik_cols, ik_data = _read_h5_opensim_table(ik_group[keys[0]])
    if "time" not in ik_cols:
        return None
    t = ik_data[:, ik_cols.index("time")].astype(np.float64)
    if len(t) < 2:
        return None
    return float(t[-1] - t[0])


def scan_memo_root(memo_root: Path) -> Tuple[Dict[str, float], Dict[str, int], List[str]]:
    """
    Returns:
      seconds_per_family: family label -> total seconds
      trial_counts: family -> number of trials contributing
      errors: list of error strings (missing ik, etc.)
    """
    seconds: Dict[str, float] = {k: 0.0 for k in FAMILY_LABELS}
    trial_counts: Dict[str, int] = {k: 0 for k in FAMILY_LABELS}
    errors: List[str] = []

    h5_files = sorted(memo_root.glob("S*.h5"))
    for h5_path in h5_files:
        subj = h5_path.stem.upper()
        try:
            with h5py.File(h5_path, "r") as h5f:
                for cond in sorted(h5f.keys()):
                    fam = classify_condition(cond)
                    if fam is None:
                        continue
                    for trial_name in sorted(h5f[cond].keys()):
                        trial_g = h5f[cond][trial_name]
                        try:
                            dur = trial_duration_sec_from_ik(trial_g)
                        except Exception as e:
                            errors.append(f"{subj}/{cond}/{trial_name}: {e}")
                            continue
                        if dur is None or not np.isfinite(dur) or dur < 0:
                            errors.append(f"{subj}/{cond}/{trial_name}: bad duration {dur!r}")
                            continue
                        seconds[fam] += dur
                        trial_counts[fam] += 1
        except Exception as e:
            errors.append(f"{h5_path.name}: {e}")

    return seconds, trial_counts, errors


def plot_bar_chart(
    seconds_per_family: Dict[str, float],
    trial_counts: Dict[str, int],
    out_path: Path,
    title: str,
) -> None:
    labels = FAMILY_LABELS
    hours = [seconds_per_family[k] / 3600.0 for k in labels]
    counts = [trial_counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, hours, color=["#2E86AB", "#A23B72", "#6A994E", "#F4A261"], edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Total time (hours)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.35)

    for i, (b, h, c) in enumerate(zip(bars, hours, counts)):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{h:.2f} h\n({c} trials)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="MeMo task-family duration composition (bar plot)")
    ap.add_argument(
        "--memo-root",
        type=str,
        default="/media/metamobility3/Samsung_T51/Processed/MeMo",
        help="Processed MeMo root (contains S###.h5)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path (default: runs/memo_task_composition.png under repo)",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write JSON summary (seconds, trial counts, errors)",
    )
    args = ap.parse_args()

    memo_root = Path(args.memo_root).expanduser().resolve()
    if not memo_root.is_dir():
        raise SystemExit(f"Not a directory: {memo_root}")

    out = args.output.strip()
    if not out:
        out = str(_REPO / "runs" / "memo_task_composition.png")
    out_path = Path(out).expanduser().resolve()

    print(f"Scanning {memo_root} ...")
    seconds_per_family, trial_counts, errors = scan_memo_root(memo_root)

    total_sec = sum(seconds_per_family.values())
    total_trials = sum(trial_counts.values())
    print(f"\nSubjects (H5 files): {len(list(memo_root.glob('S*.h5')))}")
    print(f"Trials counted (four families): {total_trials}")
    print(f"Total time (four families): {total_sec/3600:.3f} hours\n")

    for fam in FAMILY_LABELS:
        s = seconds_per_family[fam]
        n = trial_counts[fam]
        pct = 100.0 * s / total_sec if total_sec > 0 else 0.0
        print(f"  {fam:<18}  {s/3600:8.3f} h  ({n:5d} trials)  {pct:5.1f}%")

    if errors:
        print(f"\nWarnings/errors: {len(errors)} (see JSON if saved)")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    plot_bar_chart(
        seconds_per_family,
        trial_counts,
        out_path,
        title=f"MeMo composition — incline / stair / levelground / treadmill",
    )
    print(f"\nWrote {out_path}")

    json_path = args.json_out.strip()
    if json_path:
        p = Path(json_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(
                {
                    "memo_root": str(memo_root),
                    "seconds_per_family": {k: seconds_per_family[k] for k in FAMILY_LABELS},
                    "trial_counts_per_family": {k: trial_counts[k] for k in FAMILY_LABELS},
                    "total_seconds_four_families": total_sec,
                    "total_trials_four_families": total_trials,
                    "errors": errors,
                },
                f,
                indent=2,
            )
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()

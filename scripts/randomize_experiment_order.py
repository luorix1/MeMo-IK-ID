#!/usr/bin/env python3
"""
Generate per-subject randomized experiment schedules.

For each subject:
1. Shuffle the order of main conditions (hip exo, knee exo, no exo / Awinda).
2. Pick sub-condition orders that minimize LG ↔ RA/RD treadmill reconfigs overall:
   RA and RD share one treadmill configuration; counts include transitions across
   block boundaries (last trial of block N → first trial of block N+1).

Use --seed for a reproducible cohort; omit --seed for a fresh random draw each run.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple


MAIN_CONDITIONS: Sequence[str] = ("hip_exo", "knee_exo", "no_exo_awinda")

# Display labels for CSV / console
MAIN_LABELS: Dict[str, str] = {
    "hip_exo": "Hip exo",
    "knee_exo": "Knee exo",
    "no_exo_awinda": "No exo (Awinda)",
}

SUBCONDITIONS: Dict[str, List[str]] = {
    "hip_exo": ["LG", "RA"],
    "knee_exo": ["RA", "RD"],
    "no_exo_awinda": ["LG", "RA", "RD"],
}


@dataclass(frozen=True)
class BlockSchedule:
    """One main condition with a sub-condition order (globally minimized LG↔RA/RD switches)."""

    main_key: str
    main_label: str
    sub_order: tuple[str, ...]


@dataclass(frozen=True)
class SubjectSchedule:
    subject_id: int
    blocks: tuple[BlockSchedule, ...]


def _is_ra_rd_cluster(cond: str) -> bool:
    return cond in ("RA", "RD")


def treadmill_cluster_transitions(sequence: Tuple[str, ...]) -> int:
    """Count adjacent LG ↔ RA/RD boundaries along a flat trial sequence."""
    n = 0
    for a, b in zip(sequence, sequence[1:]):
        if _is_ra_rd_cluster(a) != _is_ra_rd_cluster(b):
            n += 1
    return n


def valid_sub_orders(main_key: str) -> List[tuple[str, ...]]:
    """All sub-orders that keep LG adjacent to the RA/RD pair when three trials exist."""
    if main_key == "hip_exo":
        return [("LG", "RA"), ("RA", "LG")]
    if main_key == "knee_exo":
        return [("RA", "RD"), ("RD", "RA")]
    if main_key == "no_exo_awinda":
        out: list[tuple[str, ...]] = []
        for p0, p1 in ("RA", "RD"), ("RD", "RA"):
            out.append(("LG", p0, p1))
            out.append((p0, p1, "LG"))
        return out
    raise ValueError(f"unknown main_key: {main_key!r}")


def schedule_subject(rng: random.Random, subject_id: int) -> SubjectSchedule:
    main_order = list(MAIN_CONDITIONS)
    rng.shuffle(main_order)
    option_lists = [valid_sub_orders(k) for k in main_order]
    best_cost: int | None = None
    best_combos: list[tuple[tuple[str, ...], ...]] = []
    for combo in product(*option_lists):
        flat = tuple(cond for block in combo for cond in block)
        cost = treadmill_cluster_transitions(flat)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_combos = [combo]
        elif cost == best_cost:
            best_combos.append(combo)
    assert best_cost is not None
    chosen = rng.choice(best_combos)
    blocks = tuple(
        BlockSchedule(
            main_key=k,
            main_label=MAIN_LABELS[k],
            sub_order=sub,
        )
        for k, sub in zip(main_order, chosen)
    )
    return SubjectSchedule(subject_id=subject_id, blocks=blocks)


def schedule_cohort(n_subjects: int, seed: int | None) -> List[SubjectSchedule]:
    rng = random.Random(seed)
    return [schedule_subject(rng, sid) for sid in range(1, n_subjects + 1)]


def _schedule_to_jsonable(s: SubjectSchedule) -> dict:
    return {
        "subject_id": s.subject_id,
        "blocks": [
            {
                "main_key": b.main_key,
                "main_label": b.main_label,
                "sub_order": list(b.sub_order),
            }
            for b in s.blocks
        ],
    }


def print_table(schedules: Sequence[SubjectSchedule]) -> None:
    """Human-readable table to stdout."""
    for s in schedules:
        print(f"\nSubject {s.subject_id}")
        for i, b in enumerate(s.blocks, start=1):
            subs = ", ".join(b.sub_order)
            print(f"  Block {i}: {b.main_label} → order: {subs}")


def write_csv(schedules: Sequence[SubjectSchedule], path: str) -> None:
    """
    One row per subject; columns block1_main, block1_subs, block2_..., block3_...
    Sub-order is pipe-separated (e.g. LG|RA).
    """
    fieldnames = ["subject_id"]
    for k in range(len(MAIN_CONDITIONS)):
        fieldnames.extend([f"block{k + 1}_main", f"block{k + 1}_sub_order"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in schedules:
            row: dict[str, str | int] = {"subject_id": s.subject_id}
            for k, b in enumerate(s.blocks):
                row[f"block{k + 1}_main"] = b.main_label
                row[f"block{k + 1}_sub_order"] = "|".join(b.sub_order)
            w.writerow(row)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "n_subjects",
        type=int,
        help="Number of subjects (each schedule draws from the same seeded RNG stream).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible schedules across runs.",
    )
    p.add_argument(
        "--csv",
        metavar="PATH",
        help="Write schedules to CSV at PATH.",
    )
    p.add_argument(
        "--json",
        metavar="PATH",
        help="Write schedules to JSON at PATH.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the table (use with --csv/--json).",
    )
    args = p.parse_args(argv)

    if args.n_subjects < 1:
        print("n_subjects must be >= 1", file=sys.stderr)
        return 2

    schedules = schedule_cohort(args.n_subjects, args.seed)

    if not args.quiet:
        print_table(schedules)

    if args.csv:
        write_csv(schedules, args.csv)

    if args.json:
        payload = [_schedule_to_jsonable(s) for s in schedules]
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

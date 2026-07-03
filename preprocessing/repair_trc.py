"""Repair malformed Vicon TRC files for OpenSim IK.

Some exports bundle multiple marker sets (no_exo / hip_exo / knee_exo) into one
file with broken timing (inf DataRate, all-zero timestamps).  This script keeps
the marker set matching the trial type, strips subject prefixes, forward-fills
missing coordinates, and writes a clean 24-marker 100 Hz TRC.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FS_HZ = 100.0

FOLDER_PREFIX = {
    "awinda": "no_exo",
    "hip-exo": "hip_exo",
    "knee-exo": "knee_exo",
}


def _subject_key(output_dir: Path) -> str:
    # AB01_Jinwoo -> ab01_jinwoo
    return output_dir.name.split("_", 1)[0].lower() + "_" + output_dir.name.split("_", 1)[1].lower()


def _parse_trc(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text().splitlines()
    marker_line = lines[3].split("\t")
    markers: list[str] = []
    idx = 2
    while idx < len(marker_line):
        name = marker_line[idx].strip()
        if name:
            markers.append(name)
        idx += 3
    data = [ln.split("\t") for ln in lines[5:] if ln.strip()]
    return markers, data


def _needs_repair(path: Path) -> bool:
    lines = path.read_text().splitlines()
    if len(lines) < 4:
        return True
    meta = lines[2].split("\t")
    rate = meta[0].strip().lower()
    nmarkers = meta[3].strip()
    return rate in {"inf.0", "inf", "nan"} or nmarkers != "24"


def _is_plain_trc(markers: list[str]) -> bool:
    return bool(markers) and all(":" not in m for m in markers)


def _plain_score(markers: list[str], data: list[list[str]]) -> int:
    score = 0
    for mi in range(len(markers)):
        for row in data:
            base = 2 + mi * 3
            if base + 2 < len(row) and all(row[base + j].strip() for j in range(3)):
                score += 1
                break
    return score


def _prefix_score(markers: list[str], data: list[list[str]], marker_prefix: str) -> int:
    want = f"{marker_prefix}:"
    sel = [i for i, m in enumerate(markers) if m.startswith(want)]
    if len(sel) != 24:
        return -1
    score = 0
    for mi in sel:
        for row in data:
            base = 2 + mi * 3
            if base + 2 < len(row) and all(row[base + j].strip() for j in range(3)):
                score += 1
                break
    return score


def _choose_prefix(subject_key: str, folder_type: str, markers: list[str], data: list[list[str]]) -> str:
    preferred = f"{subject_key}_{FOLDER_PREFIX[folder_type]}"
    candidates = [preferred]
    for suffix in ("hip_exo", "knee_exo", "no_exo"):
        alt = f"{subject_key}_{suffix}"
        if alt not in candidates:
            candidates.append(alt)
    best = max(candidates, key=lambda p: _prefix_score(markers, data, p))
    if _prefix_score(markers, data, best) <= 0:
        raise RuntimeError(f"No marker prefix with usable data among {candidates}")
    if best != preferred:
        print(f"  [note] using '{best}' markers (preferred '{preferred}' empty)")
    return best


def _has_missing_coords(markers: list[str], data: list[list[str]]) -> bool:
    for row in data:
        for mi in range(len(markers)):
            base = 2 + mi * 3
            if base + 2 >= len(row) or not all(row[base + j].strip() for j in range(3)):
                return True
    return False


def _fill_marker_coords(
    path: Path,
    marker_indices: list[int],
    data: list[list[str]],
) -> list[np.ndarray]:
    nframes = len(data)
    coords: list[np.ndarray] = []
    for mi in marker_indices:
        arr = np.full((nframes, 3), np.nan)
        for f, row in enumerate(data):
            base = 2 + mi * 3
            if base + 2 < len(row) and all(row[base + j].strip() for j in range(3)):
                arr[f] = [float(row[base + j]) for j in range(3)]
        for j in range(3):
            col = arr[:, j]
            bad = ~np.isfinite(col)
            if bad.any():
                good = np.where(~bad)[0]
                if len(good) == 0:
                    raise RuntimeError(f"{path.name}: marker index {mi} has no valid samples")
                first = good[0]
                col[:first] = col[first]
                for k in range(first + 1, len(col)):
                    if not np.isfinite(col[k]):
                        col[k] = col[k - 1]
            arr[:, j] = col
        coords.append(arr)
    return coords


def _write_trc(path: Path, marker_names: list[str], coords: list[np.ndarray], dry_run: bool) -> bool:
    nframes = coords[0].shape[0] if coords else 0
    header_markers = "\t\t\t".join(marker_names) + "\t\t\t"
    axis_labels = []
    for i in range(len(marker_names)):
        axis_labels.extend([f"X{i + 1}", f"Y{i + 1}", f"Z{i + 1}"])

    out = [
        f"PathFileType\t4\t(X/Y/Z)\t{path}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{FS_HZ}\t{FS_HZ}\t{nframes}\t{len(marker_names)}\tmm\t{FS_HZ}\t1\t{nframes}\t",
        "Frame#\tTime\t" + header_markers,
        "\t" + "\t".join(axis_labels) + "\t",
    ]
    for f in range(nframes):
        row = [str(f + 1), f"{f / FS_HZ:.5f}"]
        for m in range(len(marker_names)):
            row.extend(f"{coords[m][f, j]:.5f}" for j in range(3))
        out.append("\t".join(row))

    if dry_run:
        print(f"[dry] would repair {path}")
        return True

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(path.read_text())
    path.write_text("\n".join(out) + "\n")
    print(f"repaired {path}")
    return True


def repair_plain_trc(path: Path, dry_run: bool = False) -> bool:
    """Forward-fill missing coordinates in a plain (unprefixed) TRC export."""
    markers, data = _parse_trc(path)
    if not _is_plain_trc(markers) or not _has_missing_coords(markers, data):
        return False

    coords = _fill_marker_coords(path, list(range(len(markers))), data)
    return _write_trc(path, markers, coords, dry_run)


def repair_trc(path: Path, marker_prefix: str, dry_run: bool = False, force: bool = False) -> bool:
    """Return True if a repaired file was written."""
    if not force and not _needs_repair(path):
        return False

    markers, data = _parse_trc(path)
    want = f"{marker_prefix}:"
    sel = [(i, m.split(":", 1)[1]) for i, m in enumerate(markers) if m.startswith(want)]
    if len(sel) != 24:
        raise RuntimeError(f"{path.name}: expected 24 markers for prefix '{want}', found {len(sel)}")

    coords = _fill_marker_coords(path, [mi for mi, _short in sel], data)
    keep_names = [s[1] for s in sel]
    return _write_trc(path, keep_names, coords, dry_run)


def repair_output_dir(output_dir: Path, dry_run: bool = False, force: bool = False) -> int:
    subject_key = _subject_key(output_dir)
    count = 0
    for folder_type in FOLDER_PREFIX:
        marker_dir = output_dir / folder_type / "marker"
        if not marker_dir.is_dir():
            continue
        for trc in sorted(marker_dir.glob("*.trc")):
            if "static" in trc.stem.lower():
                continue
            markers, data = _parse_trc(trc)
            if _is_plain_trc(markers):
                if _has_missing_coords(markers, data):
                    try:
                        repair_plain_trc(trc, dry_run=dry_run)
                    except RuntimeError as exc:
                        quarantine = trc.with_suffix(trc.suffix + ".missing_data")
                        print(f"  [skip] {trc.name}: {exc}")
                        if not dry_run:
                            if quarantine.exists():
                                quarantine.unlink()
                            trc.rename(quarantine)
                            print(f"  [quarantine] {trc.name} → {quarantine.name}")
                        continue
                    count += 1
                    continue
                if not _needs_repair(trc) and _plain_score(markers, data) >= 20:
                    continue
                if not _needs_repair(trc):
                    continue
            preferred = f"{subject_key}_{FOLDER_PREFIX[folder_type]}"
            needs = force or _needs_repair(trc)
            if not needs and not _is_plain_trc(markers):
                needs = _prefix_score(markers, data, preferred) < 20
            if not needs:
                continue
            if _is_plain_trc(markers):
                continue
            try:
                prefix = _choose_prefix(subject_key, folder_type, markers, data)
                repair_trc(trc, prefix, dry_run=dry_run, force=True)
            except RuntimeError as exc:
                # Unrepairable (e.g. empty export). Quarantine so IK/ID skip it
                # instead of aborting the whole pipeline on one bad trial.
                quarantine = trc.with_suffix(trc.suffix + ".missing_data")
                print(f"  [skip] {trc.name}: {exc}")
                if not dry_run:
                    if quarantine.exists():
                        quarantine.unlink()
                    trc.rename(quarantine)
                    print(f"  [quarantine] {trc.name} → {quarantine.name}")
                continue
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-repair even if already 24 markers.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = repair_output_dir(args.output_dir, dry_run=args.dry_run, force=args.force)
    print(f"Done — repaired {n} TRC file(s) under {args.output_dir}")


if __name__ == "__main__":
    main()

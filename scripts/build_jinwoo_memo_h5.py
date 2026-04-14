#!/usr/bin/env python3
"""
Copy Processed/MeMo_Final subject H5 bundles to Processed/Jinwoo with cohort-specific sign fixes:

  - S001–S022 (Camargo):     flip ankle & hip-flexion *moments* (ID).
  - S023–S034 (Scherpereel): flip ankle *angles* (IK) and ankle *moments* (ID).
  - S035–S056 (Molinaro):    flip ankle *angles* (IK); flip knee & ankle *moments* (ID).
  - S057–S070:               flip hip-flexion & ankle *angles* (IK) only.
  - Other subjects:          verbatim copy.

Layout matches MeMo flat H5: one ``S###.h5`` per subject, ``<condition>/<trial>/ik|id|imu/...``.

Example:
    python build_jinwoo_memo_h5.py \\
        --source /media/metamobility3/Samsung_T51/Processed/MeMo_Final \\
        --output /media/metamobility3/Samsung_T51/Processed/Jinwoo
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable


SUBJECT_H5_RE = re.compile(r"^S(\d{3})\.h5$", re.IGNORECASE)


# Cohort boundaries (inclusive), matching MeMo subject ids.
CAMARGO = (1, 22)
SCHERPEREEL = (23, 34)
MOLINARO = (35, 56)
MEMO_S057_S070 = (57, 70)

IK_ANKLE = ("ankle_angle_r", "ankle_angle_l")
IK_HIP_FLEX = ("hip_flexion_r", "hip_flexion_l")
ID_ANKLE = ("ankle_angle_r_moment", "ankle_angle_l_moment")
ID_HIP_FLEX = ("hip_flexion_r_moment", "hip_flexion_l_moment")
ID_KNEE = ("knee_angle_r_moment", "knee_angle_l_moment")


def subject_index_from_stem(stem: str) -> int:
    """``S001`` -> 1."""
    if len(stem) >= 4 and stem[0].upper() == "S" and stem[1:].isdigit():
        return int(stem[1:], 10)
    raise ValueError(f"Bad subject stem: {stem!r}")


def ik_columns_to_flip(sdx: int) -> frozenset[str]:
    if SCHERPEREEL[0] <= sdx <= SCHERPEREEL[1]:
        return frozenset(IK_ANKLE)
    if MOLINARO[0] <= sdx <= MOLINARO[1]:
        return frozenset(IK_ANKLE)
    if MEMO_S057_S070[0] <= sdx <= MEMO_S057_S070[1]:
        return frozenset(IK_HIP_FLEX) | frozenset(IK_ANKLE)
    return frozenset()


def id_columns_to_flip(sdx: int) -> frozenset[str]:
    if CAMARGO[0] <= sdx <= CAMARGO[1]:
        return frozenset(ID_ANKLE) | frozenset(ID_HIP_FLEX)
    if SCHERPEREEL[0] <= sdx <= SCHERPEREEL[1]:
        return frozenset(ID_ANKLE)
    if MOLINARO[0] <= sdx <= MOLINARO[1]:
        return frozenset(ID_KNEE) | frozenset(ID_ANKLE)
    return frozenset()


def cohort_label(sdx: int) -> str:
    if CAMARGO[0] <= sdx <= CAMARGO[1]:
        return "camargo"
    if SCHERPEREEL[0] <= sdx <= SCHERPEREEL[1]:
        return "scherpereel"
    if MOLINARO[0] <= sdx <= MOLINARO[1]:
        return "molinaro"
    if MEMO_S057_S070[0] <= sdx <= MEMO_S057_S070[1]:
        return "memo_S057_S070"
    return "other"


def _copy_attrs(src: Any, dst: Any) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def _clone_dataset(
    dst_parent: h5py.Group,
    name: str,
    src_ds: h5py.Dataset,
    data: np.ndarray,
) -> None:
    kwargs: dict[str, Any] = {}
    if src_ds.compression:
        kwargs["compression"] = src_ds.compression
        if src_ds.compression_opts is not None:
            kwargs["compression_opts"] = src_ds.compression_opts
    if src_ds.chunks is not None:
        kwargs["chunks"] = src_ds.chunks
    d = dst_parent.create_dataset(name, data=data, **kwargs)
    _copy_attrs(src_ds, d)


def _apply_signed_flips(
    data: np.ndarray,
    columns: Sequence[str],
    flip_names: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    """Return (possibly new array, list of column names actually flipped)."""
    flip_set = {n for n in flip_names if n in columns}
    if not flip_set:
        return data, []
    out = np.array(data, dtype=np.float64, copy=True)
    applied: list[str] = []
    for cname in sorted(flip_set):
        j = columns.index(cname)
        out[:, j] *= -1.0
        applied.append(cname)
    if np.issubdtype(data.dtype, np.floating):
        out = out.astype(data.dtype, copy=False)
    return out, applied


def _decode_columns_attr(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return list(json.loads(raw))


def _process_ik_or_id_group(
    src_g: h5py.Group,
    dst_g: h5py.Group,
    flip_cols: frozenset[str],
) -> dict[str, dict[str, Any]]:
    """Copy datasets under ik/ or id/; flip given coordinate columns. Returns stats per dataset."""
    per_dset: dict[str, dict[str, Any]] = {}
    for dname in src_g.keys():
        src_ds = src_g[dname]
        if not isinstance(src_ds, h5py.Dataset):
            continue
        data = src_ds[()]
        cols = _decode_columns_attr(src_ds.attrs.get("columns"))
        applied: list[str] = []
        if cols is not None and flip_cols:
            data, applied = _apply_signed_flips(data, cols, flip_cols)
        _clone_dataset(dst_g, dname, src_ds, data)
        per_dset[dname] = {"flipped_columns": applied}
    return per_dset


def _copy_group_shallow_structure(
    src: h5py.Group,
    dst: h5py.Group,
    sdx: int,
    trial_path: str,
    stats: dict[str, Any],
) -> None:
    """Copy trial contents: ik/id with flips, everything else recursively."""
    ik_flip = ik_columns_to_flip(sdx)
    id_flip = id_columns_to_flip(sdx)

    for name, item in src.items():
        if isinstance(item, h5py.Group):
            dst_g = dst.create_group(name)
            _copy_attrs(item, dst_g)
            if name == "ik":
                dstats = _process_ik_or_id_group(item, dst_g, ik_flip)
                if any(d["flipped_columns"] for d in dstats.values()):
                    stats.setdefault("ik_edits", {})[trial_path] = dstats
                continue
            if name == "id":
                dstats = _process_ik_or_id_group(item, dst_g, id_flip)
                if any(d["flipped_columns"] for d in dstats.values()):
                    stats.setdefault("id_edits", {})[trial_path] = dstats
                continue
            for child_name, child in item.items():
                _copy_nested(child, dst_g, child_name)


def _copy_nested(item: h5py.Dataset | h5py.Group, dst_parent: h5py.Group, name: str) -> None:
    if isinstance(item, h5py.Dataset):
        _clone_dataset(dst_parent, name, item, item[()])
        return
    dst_g = dst_parent.create_group(name)
    _copy_attrs(item, dst_g)
    for child_name, child in item.items():
        _copy_nested(child, dst_g, child_name)


def process_subject_h5(
    src_path: Path,
    dst_path: Path,
    stats: dict[str, Any],
) -> None:
    stem = src_path.stem
    sdx = subject_index_from_stem(stem)
    stats["subject_index"] = sdx
    stats["cohort"] = cohort_label(sdx)
    stats["ik_flip_columns"] = sorted(ik_columns_to_flip(sdx))
    stats["id_flip_columns"] = sorted(id_columns_to_flip(sdx))

    with h5py.File(src_path, "r") as src_f, h5py.File(dst_path, "w") as dst_f:
        _copy_attrs(src_f, dst_f)
        for cond in src_f.keys():
            src_c = src_f[cond]
            if not isinstance(src_c, h5py.Group):
                continue
            dst_c = dst_f.create_group(cond)
            _copy_attrs(src_c, dst_c)
            for trial in src_c.keys():
                src_t = src_c[trial]
                if not isinstance(src_t, h5py.Group):
                    continue
                dst_t = dst_c.create_group(trial)
                _copy_attrs(src_t, dst_t)
                trial_path = f"{cond}/{trial}"
                _copy_group_shallow_structure(src_t, dst_t, sdx, trial_path, stats)


def main() -> None:
    p = argparse.ArgumentParser(description="MeMo_Final → Jinwoo H5 with cohort sign fixes")
    p.add_argument(
        "--source",
        type=Path,
        default=Path("/media/metamobility3/Samsung_T51/Processed/MeMo_Final"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/media/metamobility3/Samsung_T51/Processed/Jinwoo"),
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional path to copy dataset_metadata.json from (default: source/dataset_metadata.json if exists)",
    )
    p.add_argument(
        "--only-subjects",
        type=str,
        default="",
        help="Comma-separated stems, e.g. S001,S023 (default: all S*.h5 in source)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    args = p.parse_args()

    source: Path = args.source.expanduser().resolve()
    output: Path = args.output.expanduser().resolve()
    if not source.is_dir():
        sys.exit(f"Source not found: {source}")

    output.mkdir(parents=True, exist_ok=True)

    meta_src = (
        args.metadata.expanduser().resolve()
        if args.metadata
        else source / "dataset_metadata.json"
    )
    if meta_src.is_file():
        import shutil

        meta_dst = output / "dataset_metadata.json"
        shutil.copy2(meta_src, meta_dst)
        with open(meta_dst, encoding="utf-8") as f:
            meta_doc = json.load(f)
        meta_doc["jinwoo_sign_fix"] = {
            "derived_from": str(source),
            "script": "os_kinetics/build_jinwoo_memo_h5.py",
            "rules": {
                "camargo_S001_S022": {
                    "id_flip": sorted(ID_ANKLE | frozenset(ID_HIP_FLEX)),
                },
                "scherpereel_S023_S034": {
                    "ik_flip": list(IK_ANKLE),
                    "id_flip": list(ID_ANKLE),
                },
                "molinaro_S035_S056": {
                    "ik_flip": list(IK_ANKLE),
                    "id_flip": sorted(ID_KNEE | frozenset(ID_ANKLE)),
                },
                "memo_S057_S070": {
                    "ik_flip": sorted(frozenset(IK_HIP_FLEX) | frozenset(IK_ANKLE)),
                },
            },
        }
        with open(meta_dst, "w", encoding="utf-8") as f:
            json.dump(meta_doc, f, indent=2)
            f.write("\n")

    h5_files = sorted(source.glob("S*.h5"))
    only = {s.strip().upper() for s in args.only_subjects.split(",") if s.strip()}
    if only:
        h5_files = [p for p in h5_files if p.stem.upper() in only]

    h5_candidates = [p for p in h5_files if SUBJECT_H5_RE.match(p.name)]

    manifest: dict[str, Any] = {
        "source": str(source),
        "output": str(output),
        "subjects": [],
    }
    exit_code = 0

    for src_h5 in tqdm(
        h5_candidates,
        desc="Jinwoo H5",
        unit="subject",
        disable=bool(args.no_progress),
    ):
        subj_stats: dict[str, Any] = {"file": src_h5.name}
        dst_h5 = output / src_h5.name
        try:
            process_subject_h5(src_h5, dst_h5, subj_stats)
        except Exception as e:
            subj_stats["error"] = str(e)
            exit_code = 2
        for k in ("ik_edits", "id_edits"):
            if k in subj_stats and not subj_stats[k]:
                del subj_stats[k]
        manifest["subjects"].append(subj_stats)

    report_path = output / "jinwoo_h5_build_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    n = len(manifest["subjects"])
    err = sum(1 for s in manifest["subjects"] if "error" in s)
    print(f"Wrote {n} subject H5 file(s) to {output}")
    print(f"Report: {report_path}")
    if err:
        print(f"Errors: {err}", file=sys.stderr)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

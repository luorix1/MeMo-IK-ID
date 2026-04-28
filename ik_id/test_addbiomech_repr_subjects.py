#!/usr/bin/env python3
"""
Evaluate a trained IK→ID checkpoint on **one subject per source dataset** in
``Addbiomech_final``-style roots, using **only** HDF5 and JSON files under ``--data-dir``.

**Subject grouping (defaults, no ``--subjects``):**

1. **Jinwoo / Molinaro–Scherpereel harmonized blocks** (fixed ``S###`` ranges; not read from JSON):
   Camargo ``S001–S022``, Scherpereel ``S023–S034``, Molinaro ``S035–S056``.

2. **AddBiomechanics and other exports** described by JSON in the same folder, especially
   ``subject_id_map_<dataset>_<arm>.json`` (see ``os-biomechanics-preprocessing/b3d_to_h5.py``),
   plus optional ``dataset_metadata.json`` / generic manifests with ``subject_range`` or
   ``datasets`` blocks (see ``discover_dataset_slices_from_dir``).

One representative subject is chosen per slice (preferred ``S001`` / ``S023`` / ``S035`` for
the three Jinwoo blocks; otherwise the lowest ``S###`` in that slice that has an ``.h5`` file).

Mirrors ``ik_id/testV2.py`` for model loading and metrics; ignores ``subject_split.json``.

Run from ``os_kinetics/``::

    python -m ik_id.test_addbiomech_repr_subjects
    python ik_id/test_addbiomech_repr_subjects.py \\
        --checkpoint runs/0427_ik_id_all_addbiomech/best_model.pt \\
        --data-dir /media/metamobility3/Samsung_T5/Processed/Addbiomech_final
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from dataset import KineticsTCNDataset
from training_utils import set_global_seed

from ik_id.test import (
    load_model,
    load_run_config,
    plot_per_channel_rmse,
    plot_per_channel_r2,
    plot_scatter_gt_vs_pred,
    plot_time_series,
    resolve_unilateral_paired_for_eval,
)
from ik_id.testV2 import _predict_full_trial_from_dataset, run_inference_streaming_v2

DEFAULT_DATA_DIR = Path("/media/metamobility3/Samsung_T5/Processed/Addbiomech_final")
DEFAULT_CHECKPOINT = _ROOT / "runs" / "0427_ik_id_all_addbiomech" / "best_model.pt"

# Preferred repr. subject when that exact ``S*.h5`` exists (Jinwoo blocks only).
DEFAULT_REPR_BY_SLICE: Dict[str, str] = {
    "Camargo": "S001",
    "Scherpereel": "S023",
    "Molinaro_Scherpereel": "S035",
}

# Jinwoo harmonized ID bands (same numbering as MeMo / Processed/Jinwoo; see ``scripts/jinwoo_dataset_README.md``).
JINWOO_SUBJECT_SLICES: Tuple[Tuple[str, int, int], ...] = (
    ("Camargo", 1, 22),
    ("Scherpereel", 23, 34),
    ("Molinaro_Scherpereel", 35, 56),
)

SKIP_JSON_NAMES = frozenset(
    {
        "subject_split.json",
    }
)


def _sid_set(lo: int, hi: int) -> Set[str]:
    return {f"S{i:03d}" for i in range(lo, hi + 1)}


def _norm_subject_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().upper()
    return s if re.match(r"^S\d+$", s) else None


def _subject_sort_key(sid: str) -> int:
    m = re.search(r"\d+", sid)
    return int(m.group()) if m else 0


def _expand_subject_range(lo: str, hi: str) -> Set[str]:
    a = _norm_subject_id(lo)
    b = _norm_subject_id(hi)
    if not a or not b:
        return set()
    na, nb = _subject_sort_key(a), _subject_sort_key(b)
    if na > nb:
        na, nb = nb, na
    return {f"S{i:03d}" for i in range(na, nb + 1)}


def _merge_slice(slices: Dict[str, Set[str]], label: str, ids: Iterable[str]) -> None:
    clean = {x for x in (_norm_subject_id(v) for v in ids) if x}
    if not clean:
        return
    slices.setdefault(label, set()).update(clean)


def _parse_subject_id_map_file(path: Path) -> Tuple[str, Set[str]]:
    label = path.stem
    if label.startswith("subject_id_map_"):
        label = label[len("subject_id_map_") :]
    data = json.loads(path.read_text())
    ids: Set[str] = set()
    if isinstance(data, dict):
        for v in data.values():
            nv = _norm_subject_id(v)
            if nv:
                ids.add(nv)
    return label, ids


def _parse_dataset_metadata(meta: Dict[str, Any]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for row in meta.get("subjects") or []:
        if not isinstance(row, dict):
            continue
        sid = _norm_subject_id(row.get("subject_id") or row.get("id"))
        if not sid:
            continue
        src = (
            row.get("addbiomech_dataset")
            or row.get("source_dataset")
            or row.get("dataset")
            or row.get("cohort")
            or row.get("collection")
        )
        if not src:
            continue
        key = str(src).strip()
        if key:
            out.setdefault(key, set()).add(sid)
    for block in meta.get("datasets") or []:
        if not isinstance(block, dict):
            continue
        name = block.get("name") or block.get("dataset") or block.get("label")
        if not name:
            continue
        key = str(name).strip()
        if block.get("subject_ids"):
            for sid in block["subject_ids"]:
                nv = _norm_subject_id(sid)
                if nv:
                    out.setdefault(key, set()).add(nv)
        elif isinstance(block.get("subject_range"), (list, tuple)) and len(block["subject_range"]) >= 2:
            lo, hi = block["subject_range"][0], block["subject_range"][1]
            out.setdefault(key, set()).update(_expand_subject_range(str(lo), str(hi)))
    for block in meta.get("dataset_subject_slices") or meta.get("subject_slices") or []:
        if not isinstance(block, dict):
            continue
        name = block.get("name") or block.get("dataset") or block.get("label")
        if not name:
            continue
        key = str(name).strip()
        if isinstance(block.get("subject_range"), (list, tuple)) and len(block["subject_range"]) >= 2:
            lo, hi = block["subject_range"][0], block["subject_range"][1]
            out.setdefault(key, set()).update(_expand_subject_range(str(lo), str(hi)))
    return out


def _parse_generic_manifest(obj: Dict[str, Any]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    if not isinstance(obj, dict):
        return out
    if isinstance(obj.get("subject_range"), (list, tuple)) and len(obj["subject_range"]) >= 2:
        lo, hi = obj["subject_range"][0], obj["subject_range"][1]
        label = str(obj.get("dataset") or obj.get("name") or "manifest").strip()
        out.setdefault(label, set()).update(_expand_subject_range(str(lo), str(hi)))
    for key in ("dataset_slices", "subject_slices", "sources"):
        for block in obj.get(key) or []:
            if not isinstance(block, dict):
                continue
            name = block.get("name") or block.get("dataset") or block.get("label")
            if not name:
                continue
            k = str(name).strip()
            if isinstance(block.get("subject_range"), (list, tuple)) and len(block["subject_range"]) >= 2:
                lo, hi = block["subject_range"][0], block["subject_range"][1]
                out.setdefault(k, set()).update(_expand_subject_range(str(lo), str(hi)))
            if block.get("subject_ids"):
                for sid in block["subject_ids"]:
                    nv = _norm_subject_id(sid)
                    if nv:
                        out.setdefault(k, set()).add(nv)
    return out


def discover_dataset_slices_from_dir(data_root: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    """
    Build ``slice_label -> set of S###`` from Jinwoo bands plus JSON in ``data_root``.

    Returns ``(slices, debug_info)`` where ``debug_info`` lists JSON files consulted.
    """
    slices: Dict[str, Set[str]] = {}
    for label, lo, hi in JINWOO_SUBJECT_SLICES:
        slices[label] = _sid_set(lo, hi)

    debug: Dict[str, Any] = {
        "jinwoo_slices": [{"label": lab, "from": f"S{lo:03d}", "to": f"S{hi:03d}"} for lab, lo, hi in JINWOO_SUBJECT_SLICES],
        "json_files_used": [],
    }

    for path in sorted(data_root.glob("subject_id_map_*.json")):
        try:
            label, ids = _parse_subject_id_map_file(path)
        except Exception as e:
            debug.setdefault("json_warnings", []).append(f"{path.name}: {e}")
            continue
        if ids:
            _merge_slice(slices, label, ids)
            debug["json_files_used"].append({"file": path.name, "slice": label, "n_ids": len(ids)})

    meta_path = data_root / "dataset_metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            if isinstance(meta, dict):
                extra = _parse_dataset_metadata(meta)
                for k, s in extra.items():
                    _merge_slice(slices, k, s)
                debug["json_files_used"].append(
                    {"file": meta_path.name, "slices_from_metadata": sorted(extra.keys())}
                )
        except Exception as e:
            debug.setdefault("json_warnings", []).append(f"{meta_path.name}: {e}")

    for path in sorted(data_root.glob("*.json")):
        if path.name in SKIP_JSON_NAMES:
            continue
        if path.name.startswith("subject_id_map_"):
            continue
        if path.name == "dataset_metadata.json":
            continue
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        extra = _parse_generic_manifest(obj)
        if not extra:
            continue
        for k, s in extra.items():
            _merge_slice(slices, k, s)
        debug["json_files_used"].append({"file": path.name, "slices": sorted(extra.keys())})

    return slices, debug


def slice_label_for_subject(subject_id: str, slices: Dict[str, Set[str]]) -> str:
    """If a subject appears in multiple slice sets, prefer the smallest set (most specific)."""
    sid = _norm_subject_id(subject_id)
    if not sid:
        return "invalid_id"
    cand = [(lab, len(sset)) for lab, sset in slices.items() if sid in sset]
    if not cand:
        return "unmapped"
    cand.sort(key=lambda x: (x[1], x[0]))
    return cand[0][0]


def list_h5_subjects_under_root(data_root: Path) -> List[str]:
    return sorted(
        (p.stem.upper() for p in data_root.glob("S*.h5") if p.is_file()),
        key=_subject_sort_key,
    )


def _assert_paths_only_under_root(data_root: Path, subject_ids: List[str]) -> None:
    root = data_root.resolve()
    for sid in subject_ids:
        p = (root / f"{sid}.h5").resolve()
        try:
            p.relative_to(root)
        except ValueError as e:
            raise ValueError(f"Refusing to use path outside data root: {p}") from e
        if not p.is_file():
            raise FileNotFoundError(f"Missing H5 for subject {sid}: {p}")


def resolve_repr_subject_ids(
    data_root: Path,
    *,
    explicit: Optional[List[str]],
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """
    Returns (``(slice_label, subject_id)`` pairs, slice debug dict).

    Default: one subject per **dataset slice** — three Jinwoo bands (Camargo, Scherpereel,
    Molinaro) plus one per slice discovered from ``subject_id_map_*.json`` and other
    manifests under ``data_root`` (see ``discover_dataset_slices_from_dir``).
    """
    available = set(list_h5_subjects_under_root(data_root))
    if not available:
        raise FileNotFoundError(f"No S*.h5 files found under {data_root}")

    slices, slice_debug = discover_dataset_slices_from_dir(data_root)

    covered: Set[str] = set()
    for s in slices.values():
        covered.update(s)
    not_in_slice = available - covered
    if not_in_slice:
        slice_debug["h5_subjects_not_in_any_slice"] = sorted(not_in_slice, key=_subject_sort_key)

    if explicit:
        out: List[Tuple[str, str]] = []
        for raw in explicit:
            sid = raw.strip().upper()
            nv = _norm_subject_id(sid)
            if not nv or nv not in available:
                raise FileNotFoundError(
                    f"Requested subject {sid!r} not found under {data_root}"
                )
            out.append((slice_label_for_subject(nv, slices), nv))
        return out, slice_debug

    order: List[str] = []
    for lab in ("Camargo", "Scherpereel", "Molinaro_Scherpereel"):
        if lab in slices and (slices[lab] & available):
            order.append(lab)
    for lab in sorted(
        k for k in slices if k not in ("Camargo", "Scherpereel", "Molinaro_Scherpereel")
    ):
        if slices[lab] & available:
            order.append(lab)

    chosen: List[Tuple[str, str]] = []
    used_ids: Set[str] = set()
    for lab in order:
        pool = [s for s in sorted(slices[lab] & available, key=_subject_sort_key) if s not in used_ids]
        if not pool:
            continue
        pref = DEFAULT_REPR_BY_SLICE.get(lab, "")
        if pref and pref in pool:
            pick = pref
        else:
            pick = pool[0]
        used_ids.add(pick)
        chosen.append((lab, pick))
    if not chosen:
        raise RuntimeError(
            f"No subject selected: no S*.h5 overlap with any dataset slice. "
            f"Available: {sorted(available, key=_subject_sort_key)}. "
            f"Slice keys: {sorted(slices.keys())}."
        )

    return chosen, slice_debug


def build_h5_eval_kwargs(
    *,
    data_root: Path,
    model: torch.nn.Module,
    stats: Any,
    window_size: int,
    input_indices: Any,
    moment_indices: Any,
    input_mode: str,
    output_mode: str,
    laterality_ckpt: str,
    unilateral_paired_ckpt: Optional[bool],
    run_cfg: Optional[Dict[str, Any]],
    subject_ids: List[str],
    walking_only: bool,
    levelground_only_cli: bool,
    rollout_force: bool,
) -> Dict[str, Any]:
    """Same effective kwargs as ``testV2.py`` for H5-only layout; h5_dir == data_dir."""
    if stats is not None:
        for k, v in list(stats.items()):
            if isinstance(v, torch.Tensor):
                stats[k] = v.numpy()

    test_ds_kwargs: Dict[str, Any] = dict(
        data_dir=str(data_root),
        h5_dir=str(data_root),
        use_h5=True,
        subject_ids=subject_ids,
        window_size=window_size,
        stride=1,
        walking_only=walking_only,
        normalize=False,
        stats=stats,
    )

    _levelground_only = levelground_only_cli
    if run_cfg is not None and "levelground_only" in run_cfg:
        _levelground_only = bool(run_cfg["levelground_only"])
    test_ds_kwargs["levelground_only"] = _levelground_only

    if run_cfg is not None:
        _lat = str(run_cfg.get("laterality", laterality_ckpt))
        test_ds_kwargs.update(
            input_mode=run_cfg.get("input_mode", input_mode),
            output_mode=run_cfg.get("output_mode", output_mode),
            laterality=_lat,
        )
        _cfg_paired = run_cfg.get("unilateral_paired_side_windows", None)
        if _cfg_paired is not None:
            _cfg_paired = bool(_cfg_paired)
        _paired_flag = _cfg_paired if _cfg_paired is not None else unilateral_paired_ckpt
        test_ds_kwargs["unilateral_paired_side_windows"] = resolve_unilateral_paired_for_eval(
            laterality=_lat,
            ckpt_flag=_paired_flag,
            n_in_model=model.n_input_channels,
            input_indices=input_indices,
        )
    else:
        test_ds_kwargs.update(
            input_indices=input_indices,
            moment_indices=moment_indices,
            laterality=laterality_ckpt,
            unilateral_paired_side_windows=resolve_unilateral_paired_for_eval(
                laterality=laterality_ckpt,
                ckpt_flag=unilateral_paired_ckpt,
                n_in_model=model.n_input_channels,
                input_indices=input_indices,
            ),
        )

    test_ds_kwargs["apply_lowpass_filter"] = True
    if run_cfg is not None:
        if "lowpass_cutoff_hz" in run_cfg:
            test_ds_kwargs["lowpass_cutoff_hz"] = float(run_cfg["lowpass_cutoff_hz"])
        if "lowpass_order" in run_cfg:
            test_ds_kwargs["lowpass_order"] = int(run_cfg["lowpass_order"])

    vel_lpf_apply = False
    vel_lpf_cut = None
    vel_lpf_ord = None
    if run_cfg is not None:
        if run_cfg.get("velocity_lowpass_filter") is not None:
            vel_lpf_apply = bool(run_cfg.get("velocity_lowpass_filter"))
        vel_lpf_cut = run_cfg.get("velocity_lowpass_cutoff_hz")
        vel_lpf_ord = run_cfg.get("velocity_lowpass_order")
    test_ds_kwargs["apply_velocity_lowpass_filter"] = bool(vel_lpf_apply)
    test_ds_kwargs["velocity_lowpass_cutoff_hz"] = vel_lpf_cut
    test_ds_kwargs["velocity_lowpass_order"] = vel_lpf_ord

    if rollout_force:
        rollout_step = 2
    else:
        rollout_step = 1
        if run_cfg is not None:
            rollout_step = int(run_cfg.get("rollout_decimate_step", 1))
            if rollout_step == 1 and bool(run_cfg.get("rollout")):
                rollout_step = 2
        rollout_step = max(1, rollout_step)
    if rollout_step > 1:
        test_ds_kwargs["rollout_decimate_step"] = rollout_step

    return test_ds_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval checkpoint: one subject per dataset slice (Jinwoo + JSON-defined AddBiomechanics exports)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to best_model.pt (default: runs/0427_ik_id_all_addbiomech/best_model.pt under os_kinetics)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Folder containing ONLY S*.h5 files to use (default: Addbiomech_final on T5)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="",
        help='Comma-separated subject IDs. If empty: one subject per discovered slice — '
        "Jinwoo bands (Camargo, Scherpereel, Molinaro) plus each dataset from "
        "subject_id_map_*.json / dataset_metadata.json / manifests in --data-dir.",
    )
    parser.add_argument("--output-dir", type=str, default="results/addbiomech_repr_eval")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--walking-only", action="store_true", default=True)
    parser.add_argument("--no-walking-only", dest="walking_only", action="store_false")
    parser.add_argument("--levelground-only", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-plot-samples", type=int, default=2)
    parser.add_argument(
        "--rollout",
        action="store_true",
        default=False,
        help="Force stride-2 IK/ID decimation (same as testV2 --rollout).",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    data_root = Path(args.data_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"--data-dir is not a directory: {data_root}")

    explicit_list: Optional[List[str]] = None
    if args.subjects.strip():
        explicit_list = [s.strip() for s in args.subjects.split(",") if s.strip()]

    cohort_subject_pairs, slice_debug = resolve_repr_subject_ids(data_root, explicit=explicit_list)
    subject_ids = [sid for _c, sid in cohort_subject_pairs]
    _assert_paths_only_under_root(data_root, subject_ids)

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = (_ROOT / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        ckpt_try = (_ROOT / ckpt_path).resolve()
        if ckpt_try.is_file():
            ckpt_path = ckpt_try
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Data root (H5 only): {data_root}")
    print(f"Checkpoint:          {ckpt_path}")
    print("Representative subjects (dataset slice → S###):")
    for cohort, sid in cohort_subject_pairs:
        print(f"  {cohort:<32} {sid}")
    for msg in slice_debug.get("json_warnings", []) or []:
        print(f"  [json warning] {msg}")
    if "h5_subjects_not_in_any_slice" in slice_debug:
        print("  H5 with no JSON slice match:", ", ".join(slice_debug["h5_subjects_not_in_any_slice"]))

    print(f"\nLoading model from {ckpt_path}")
    (
        model,
        stats,
        dof_names,
        window_size,
        input_indices,
        moment_indices,
        input_mode,
        output_mode,
        laterality_ckpt,
        unilateral_paired_ckpt,
    ) = load_model(str(ckpt_path), args.device)

    run_cfg = load_run_config(str(ckpt_path))

    common_kw = dict(
        data_root=data_root,
        model=model,
        stats=stats,
        window_size=window_size,
        input_indices=input_indices,
        moment_indices=moment_indices,
        input_mode=input_mode,
        output_mode=output_mode,
        laterality_ckpt=laterality_ckpt,
        unilateral_paired_ckpt=unilateral_paired_ckpt,
        run_cfg=run_cfg,
        walking_only=args.walking_only,
        levelground_only_cli=args.levelground_only,
        rollout_force=args.rollout,
    )

    # Plot sample rate (mirror testV2 rollout_decimate_step resolution)
    if args.rollout:
        rollout_step = 2
    else:
        rollout_step = 1
        if run_cfg is not None:
            rollout_step = int(run_cfg.get("rollout_decimate_step", 1))
            if rollout_step == 1 and bool(run_cfg.get("rollout")):
                rollout_step = 2
        rollout_step = max(1, rollout_step)
    plot_sample_hz = 200.0 / float(rollout_step) if rollout_step > 1 else 200.0

    all_results: Dict[str, Any] = {
        "data_dir": str(data_root),
        "checkpoint": str(ckpt_path),
        "subject_slices": [{"dataset_slice": c, "subject_id": s} for c, s in cohort_subject_pairs],
        "slice_discovery": slice_debug,
    }

    for cohort, sid in cohort_subject_pairs:
        sub_out = out_root / sid
        sub_out.mkdir(parents=True, exist_ok=True)

        test_ds_kwargs = build_h5_eval_kwargs(**common_kw, subject_ids=[sid])
        print(f"\n{'='*60}\nSubject {sid}  [{cohort}]\n{'='*60}")
        print(f"  unilateral_paired_side_windows: {test_ds_kwargs.get('unilateral_paired_side_windows')}")
        if test_ds_kwargs.get("rollout_decimate_step", 1) > 1:
            print(f"  rollout_decimate_step: {test_ds_kwargs['rollout_decimate_step']}")

        test_ds = KineticsTCNDataset(**test_ds_kwargs)
        print(f"  Windows: {len(test_ds):,}")
        if len(test_ds) == 0:
            print("  [skip] No windows for this subject (filters / missing trials).")
            continue

        loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )
        metrics, _pred_plot, _true_plot, scatter_gt, scatter_pred_arr, smoothness = (
            run_inference_streaming_v2(
                model,
                loader,
                args.device,
                dof_names,
                n_plot_samples=args.n_plot_samples,
                scatter_max_points=50_000,
            )
        )
        metrics["smoothness_score"] = smoothness
        metrics["dataset_slice"] = cohort
        metrics["subject_id"] = sid

        print(f"  Overall RMSE: {metrics['overall']['rmse']:.6f}")
        print(f"  Overall MAE:  {metrics['overall']['mae']:.6f}")
        print(f"  Overall R²:   {metrics['overall']['r2']:.6f}")
        print(f"  Smoothness:   {smoothness:.6f}")

        with open(sub_out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        plot_per_channel_rmse(metrics, sub_out / "per_dof_rmse.png")
        plot_per_channel_r2(metrics, sub_out / "per_dof_r2.png")
        plot_scatter_gt_vs_pred(
            scatter_pred_arr,
            scatter_gt,
            dof_names,
            sub_out / "scatter_gt_vs_pred.png",
            overall_r2=metrics["overall"].get("r2"),
        )

        # One full-trial time series per subject (first trial index 0).
        pred_t, true_t = _predict_full_trial_from_dataset(model, test_ds, 0, args.device)
        ref = test_ds.h5_trial_refs[0]
        trial_tag = f"{ref[0]}_{ref[1]}_{ref[2]}".replace("/", "_")
        plot_time_series(
            pred_t[None, ...],
            true_t[None, ...],
            dof_names,
            sub_out / f"timeseries_first_trial_{trial_tag}.png",
            sample_idx=0,
            sample_rate_hz=plot_sample_hz,
        )

        all_results[sid] = {
            "dataset_slice": cohort,
            "metrics": metrics["overall"],
            "smoothness": smoothness,
        }

    with open(out_root / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results under {out_root}")


if __name__ == "__main__":
    main()

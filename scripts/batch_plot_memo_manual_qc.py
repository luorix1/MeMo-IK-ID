#!/usr/bin/env python3
"""
Run plot_memo_trial_inference for every (condition, trial) in MeMo H5 that has ik+id.

Loads the checkpoint once per process for speed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import torch

from plot_memo_trial_inference import load_checkpoint_model, run_single_trial_inference


def _safe_subdir_part(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", s)


def iter_trials_with_ik_id(h5_path: Path):
    with h5py.File(h5_path, "r") as h5f:
        for cond in sorted(h5f.keys()):
            cg = h5f[cond]
            if not isinstance(cg, h5py.Group):
                continue
            for trial in sorted(cg.keys()):
                tg = cg[trial]
                if not isinstance(tg, h5py.Group):
                    continue
                if "ik" in tg.keys() and "id" in tg.keys():
                    yield str(cond), str(trial)


def main() -> None:
    p = argparse.ArgumentParser(description="Batch MeMo inference plots for manual QC")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--memo-root", type=str, default="/media/metamobility3/Samsung_T51/Processed/MeMo")
    p.add_argument(
        "--subjects",
        type=str,
        required=True,
        help='Comma-separated subject ids, e.g. "S001,S023,S035"',
    )
    p.add_argument("--output-base", type=str, required=True, help="Directory; each subject gets a subfolder")
    p.add_argument("--write-combined-html", action="store_true")
    p.add_argument(
        "--no-lowpass",
        action="store_true",
        help="Disable LPF on loaded trials (forwarded to plot_memo_trial_inference).",
    )
    p.add_argument("--lowpass-cutoff-hz", type=float, default=4.0)
    p.add_argument("--lowpass-order", type=int, default=4)
    p.add_argument("--median-kernel-samples", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    memo_root = Path(args.memo_root)
    out_base = Path(args.output_base)
    out_base.mkdir(parents=True, exist_ok=True)

    model, input_indices, moment_indices, dof_names, window_size = load_checkpoint_model(
        Path(args.checkpoint), args.device
    )
    ckpt_str = str(Path(args.checkpoint).resolve())

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    summary = []

    for subject_id in subjects:
        h5_path = memo_root / f"{subject_id}.h5"
        if not h5_path.is_file():
            print(f"[skip] missing {h5_path}")
            continue

        trials = list(iter_trials_with_ik_id(h5_path))
        print(f"{subject_id}: {len(trials)} trials with ik+id")

        for condition, trial in trials:
            sub = (
                out_base
                / subject_id
                / f"{_safe_subdir_part(condition)}__{_safe_subdir_part(trial)}"
            )
            try:
                run_single_trial_inference(
                    model=model,
                    input_indices=input_indices,
                    moment_indices=moment_indices,
                    ckpt_dof_names=dof_names,
                    window_size=window_size,
                    memo_root=memo_root,
                    subject_id=subject_id,
                    condition=condition,
                    trial=trial,
                    out_dir=sub,
                    write_combined_html=bool(args.write_combined_html),
                    device=args.device,
                    checkpoint_path=ckpt_str,
                    apply_lowpass_filter=not bool(args.no_lowpass),
                    lowpass_cutoff_hz=float(args.lowpass_cutoff_hz),
                    lowpass_order=int(args.lowpass_order),
                    median_kernel_samples=int(args.median_kernel_samples),
                )
                summary.append((subject_id, condition, trial, "ok"))
            except ValueError as e:
                print(f"  [skip] {condition} / {trial}: {e}")
                summary.append((subject_id, condition, trial, f"skip:{e}"))
            except Exception as e:
                print(f"  [fail] {condition} / {trial}: {e}")
                summary.append((subject_id, condition, trial, f"fail:{e}"))

    ok = sum(1 for *_, st in summary if st == "ok")
    print(f"Done. ok={ok} / total_attempts={len(summary)}. Base: {out_base.resolve()}")


if __name__ == "__main__":
    main()

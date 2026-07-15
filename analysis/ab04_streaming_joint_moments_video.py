#!/usr/bin/env python3
"""Streaming GT vs estimated joint-moment video for AB04 Changseob.

Uses steady-state segments only (10 s trim from start and end), matching
compare_processed_hip_exo_id.ipynb. Renders a fixed-viewport scrolling waveform
with GT (blue) and estimate (red) plus legend.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "analysis" / "cache"
SUBJECT = "AB04_Changseob"

TRIM_START_SEC = 10.0
TRIM_END_SEC = 10.0

VIEW_WINDOW_SEC = 2.0
FIGSIZE_SINGLE = (7.5, 4.2)
FIGSIZE_PANEL_H = 3.6
FIG_MARGINS = {"left": 0.18, "right": 0.98, "bottom": 0.16, "top": 0.92}
FIG_MARGINS_PANELS = {"left": 0.18, "right": 0.98, "bottom": 0.10, "top": 0.97, "hspace": 0.32}
YLABEL_TEXT = "N·m/kg"
YLABEL_X = 0.085
FPS = 20
RENDER_DPI = 100
VIDEO_EXT = ".mov"
WRITER_EXTRA_ARGS = ["-vcodec", "mpeg4"]
PALETTE = {
    "gt": "#1e88e5",
    "est": "#e53935",
}

DATASETS = {
    "hip-exo": {
        "label": "Hip exo — hip flexion R",
        "trials": ["LG_0p8mps", "LG_1p2mps", "LG_1p6mps", "RA_0p8mps"],
        "source": "hip_npz",
    },
    "knee-exo": {
        "label": "Knee exo — knee angle R",
        "trials": ["RA_0p8mps", "RD_0p8mps"],
        "source": "knee_npz",
    },
    "awinda": {
        "label": "Awinda — hip flexion R",
        "trials": ["LG_0p8mps", "LG_1p2mps", "LG_1p6mps", "RA_0p8mps", "RD_0p8mps"],
        "source": "awinda_npz",
    },
}

HIP_STEM_BY_CONDITION = {
    "LG_0p8mps": "ab04_changseob_hip_0p8mps_lg_exo_on",
    "LG_1p2mps": "ab04_changseob_hip_1p2mps_lg_exo_on",
    "LG_1p6mps": "ab04_changseob_hip_1p6mps_lg_exo_on",
    "RA_0p8mps": "ab04_changseob_hip_0p8mps_ra_exo_on",
}
KNEE_STEM_BY_CONDITION = {
    "RA_0p8mps": "ab04_changseob_knee_0p8mps_ra_exo_on",
    "RD_0p8mps": "ab04_changseob_knee_0p8mps_rd_exo_on",
}


@dataclass
class TrialWaveform:
    dataset: str
    condition: str
    t: np.ndarray
    gt_nmpkg: np.ndarray
    est_nmpkg: np.ndarray


def analysis_trim_mask(t: np.ndarray) -> np.ndarray:
    t_rel = np.asarray(t, dtype=np.float64) - float(np.nanmin(t))
    t_end = float(np.nanmax(t_rel))
    return (t_rel >= TRIM_START_SEC) & (t_rel <= t_end - TRIM_END_SEC)


def _trim_waveform(
    dataset: str,
    condition: str,
    t: np.ndarray,
    gt: np.ndarray,
    est: np.ndarray,
) -> TrialWaveform:
    keep = analysis_trim_mask(t)
    if not np.any(keep):
        raise RuntimeError(f"No steady-state samples for {dataset} {condition}")
    t_trim = t[keep] - float(t[keep][0])
    return TrialWaveform(
        dataset=dataset,
        condition=condition,
        t=t_trim,
        gt_nmpkg=np.asarray(gt[keep], dtype=np.float64),
        est_nmpkg=np.asarray(est[keep], dtype=np.float64),
    )


def _load_hip_trial(condition: str) -> TrialWaveform:
    stem = HIP_STEM_BY_CONDITION[condition]
    data = np.load(CACHE_DIR / "compare_processed_hip_exo_id.npz", allow_pickle=True)
    t = data[f"{stem}__t"]
    gt = data[f"{stem}__gt_nmpkg"]
    est = data[f"{stem}__model_out_nmpkg"]
    return _trim_waveform("hip-exo", condition, t, gt, est)


def _load_knee_trial(condition: str) -> TrialWaveform:
    stem = KNEE_STEM_BY_CONDITION[condition]
    data = np.load(CACHE_DIR / "compare_processed_knee_exo_id.npz", allow_pickle=True)
    t = data[f"{stem}__t"]
    gt = data[f"{stem}__gt_nmpkg"]
    est = data[f"{stem}__model_out_nmpkg"]
    return _trim_waveform("knee-exo", condition, t, gt, est)


def _load_awinda_trial(condition: str) -> TrialWaveform:
    trial_key = f"{SUBJECT}::{condition}"
    prefix = trial_key.replace("::", "__")
    data = np.load(CACHE_DIR / "process_awinda.npz", allow_pickle=True)
    t = data[f"{prefix}__t"]
    gt = data[f"{prefix}__id_nmpkg"][:, 0]
    est = data[f"{prefix}__pred_nmpkg"][:, 0]
    return _trim_waveform("awinda", condition, t, gt, est)


def load_trial_waveform(dataset: str, condition: str) -> TrialWaveform:
    source = DATASETS[dataset]["source"]
    if source == "hip_npz":
        return _load_hip_trial(condition)
    if source == "knee_npz":
        return _load_knee_trial(condition)
    if source == "awinda_npz":
        return _load_awinda_trial(condition)
    raise ValueError(source)


def load_waveforms_for_trials(
    trial_specs: List[Tuple[str, str]],
) -> List[TrialWaveform]:
    return [load_trial_waveform(dataset, condition) for dataset, condition in trial_specs]


def load_all_waveforms() -> Dict[str, List[TrialWaveform]]:
    out: Dict[str, List[TrialWaveform]] = {}
    for dataset, cfg in DATASETS.items():
        out[dataset] = [load_trial_waveform(dataset, cond) for cond in cfg["trials"]]
    return out


def _panel_ylim(waves: List[TrialWaveform], pad_frac: float = 0.08) -> Tuple[float, float]:
    vals = np.concatenate([np.r_[w.gt_nmpkg, w.est_nmpkg] for w in waves])
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    span = max(hi - lo, 0.05)
    pad = span * pad_frac
    return lo - pad, hi + pad


def _window_bounds(t_cursor: float, view_window_sec: float) -> Tuple[float, float]:
    t1 = float(t_cursor)
    if t1 <= view_window_sec:
        return 0.0, view_window_sec
    return t1 - view_window_sec, t1


def _frame_xy(
    wave: TrialWaveform,
    t_cursor: float,
    view_window_sec: float,
    channel: str,
) -> Tuple[np.ndarray, np.ndarray]:
    t0, t1 = _window_bounds(t_cursor, view_window_sec)
    m = (wave.t >= t0) & (wave.t <= t1)
    x = wave.t[m]
    y = wave.gt_nmpkg[m] if channel == "gt" else wave.est_nmpkg[m]
    return x, y


def _style_axes(ax, ylo: float, yhi: float, show_xlabel: bool) -> None:
    ax.set_ylim(ylo, yhi)
    if show_xlabel:
        ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)


def _place_figure_ylabels(fig, axes) -> None:
    """Place y-labels in figure coords so FFMpegWriter does not clip them."""
    fig.canvas.draw()
    axes_list = list(axes) if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
    for ax in axes_list:
        pos = ax.get_position()
        fig.text(
            YLABEL_X,
            0.5 * (pos.y0 + pos.y1),
            YLABEL_TEXT,
            va="center",
            ha="center",
            rotation="vertical",
            fontsize=plt.rcParams["axes.labelsize"],
        )


def render_dataset_video(
    waves: List[TrialWaveform],
    out_path: Path,
    fps: int = FPS,
    view_window_sec: float = VIEW_WINDOW_SEC,
) -> None:
    ylo, yhi = _panel_ylim(waves)
    segments: List[Tuple[TrialWaveform, float, float]] = [
        (wave, 0.0, float(wave.t[-1])) for wave in waves
    ]
    total_duration = sum(seg[2] - seg[1] for seg in segments)
    n_frames = max(2, int(np.ceil(total_duration * fps)))

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    gt_line, = ax.plot([], [], color=PALETTE["gt"], lw=2.0, label="Ground Truth")
    est_line, = ax.plot([], [], color=PALETTE["est"], lw=1.6, label="Model Estimate")
    _style_axes(ax, ylo, yhi, show_xlabel=True)

    def _locate(t_global: float) -> Tuple[TrialWaveform, float]:
        elapsed = 0.0
        for wave, _, dur in segments:
            if t_global <= elapsed + dur:
                return wave, t_global - elapsed
            elapsed += dur
        wave, _, dur = segments[-1]
        return wave, dur

    def _update(frame_idx: int) -> None:
        t_global = frame_idx / fps
        wave, t_local = _locate(t_global)
        x_gt, y_gt = _frame_xy(wave, t_local, view_window_sec, "gt")
        x_est, y_est = _frame_xy(wave, t_local, view_window_sec, "est")
        gt_line.set_data(x_gt, y_gt)
        est_line.set_data(x_est, y_est)
        x0, x1 = _window_bounds(t_local, view_window_sec)
        ax.set_xlim(x0, x1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(**FIG_MARGINS)
    _place_figure_ylabels(fig, ax)
    writer = FFMpegWriter(fps=fps, bitrate=4000, extra_args=WRITER_EXTRA_ARGS)
    with writer.saving(fig, str(out_path), dpi=RENDER_DPI):
        for frame_idx in range(n_frames):
            _update(frame_idx)
            writer.grab_frame()
    plt.close(fig)
    print(f"Saved {out_path}")


def render_panels_video(
    waves: List[TrialWaveform],
    out_path: Path,
    fps: int = FPS,
    view_window_sec: float = VIEW_WINDOW_SEC,
) -> None:
    """One panel per trial, all panels sweep in sync."""
    ylims = [_panel_ylim([wave]) for wave in waves]
    segments = [(wave, 0.0, float(wave.t[-1])) for wave in waves]
    total_duration = max(seg[2] - seg[1] for seg in segments)
    n_frames = max(2, int(np.ceil(total_duration * fps)))

    fig, axes = plt.subplots(len(waves), 1, figsize=(7.5, FIGSIZE_PANEL_H * len(waves)))
    if len(waves) == 1:
        axes = [axes]

    artists = []
    for ax, wave, (ylo, yhi) in zip(axes, waves, ylims):
        gt_line, = ax.plot([], [], color=PALETTE["gt"], lw=2.0, label="Ground Truth")
        est_line, = ax.plot([], [], color=PALETTE["est"], lw=1.6, label="Model Estimate")
        _style_axes(ax, ylo, yhi, show_xlabel=False)
        artists.append((ax, gt_line, est_line, wave))

    axes[-1].set_xlabel("Time (s)")
    fig.subplots_adjust(**FIG_MARGINS_PANELS)
    _place_figure_ylabels(fig, axes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=5000, extra_args=WRITER_EXTRA_ARGS)
    with writer.saving(fig, str(out_path), dpi=RENDER_DPI):
        for frame_idx in range(n_frames):
            t_global = frame_idx / fps
            for ax, gt_line, est_line, wave in artists:
                t_local = min(t_global, float(wave.t[-1]))
                x_gt, y_gt = _frame_xy(wave, t_local, view_window_sec, "gt")
                x_est, y_est = _frame_xy(wave, t_local, view_window_sec, "est")
                gt_line.set_data(x_gt, y_gt)
                est_line.set_data(x_est, y_est)
                x0, x1 = _window_bounds(t_local, view_window_sec)
                ax.set_xlim(x0, x1)
            writer.grab_frame()
    plt.close(fig)
    print(f"Saved {out_path}")


def render_combined_video(
    waves_by_dataset: Dict[str, List[TrialWaveform]],
    out_path: Path,
    fps: int = FPS,
    view_window_sec: float = VIEW_WINDOW_SEC,
) -> None:
    datasets = list(DATASETS.keys())
    ylims = {ds: _panel_ylim(waves_by_dataset[ds]) for ds in datasets}
    segments: Dict[str, List[Tuple[TrialWaveform, float, float]]] = {
        ds: [(wave, 0.0, float(wave.t[-1])) for wave in waves_by_dataset[ds]]
        for ds in datasets
    }
    total_duration = max(sum(seg[2] - seg[1] for seg in segments[ds]) for ds in datasets)
    n_frames = max(2, int(np.ceil(total_duration * fps)))

    fig, axes = plt.subplots(len(datasets), 1, figsize=(7.5, FIGSIZE_PANEL_H * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]

    artists = []
    for ax, ds in zip(axes, datasets):
        gt_line, = ax.plot([], [], color=PALETTE["gt"], lw=2.0, label="Ground Truth")
        est_line, = ax.plot([], [], color=PALETTE["est"], lw=1.6, label="Model Estimate")
        ylo, yhi = ylims[ds]
        _style_axes(ax, ylo, yhi, show_xlabel=False)
        artists.append((ax, gt_line, est_line, segments[ds]))

    axes[-1].set_xlabel("Time (s)")
    fig.subplots_adjust(**FIG_MARGINS_PANELS)
    _place_figure_ylabels(fig, axes)

    def _locate(segs: List[Tuple[TrialWaveform, float, float]], t_global: float):
        elapsed = 0.0
        for wave, _, dur in segs:
            if t_global <= elapsed + dur:
                return wave, t_global - elapsed
            elapsed += dur
        wave, _, dur = segs[-1]
        return wave, dur

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=5000, extra_args=WRITER_EXTRA_ARGS)
    with writer.saving(fig, str(out_path), dpi=RENDER_DPI):
        for frame_idx in range(n_frames):
            t_global = frame_idx / fps
            for ax, gt_line, est_line, segs in artists:
                wave, t_local = _locate(segs, t_global)
                x_gt, y_gt = _frame_xy(wave, t_local, view_window_sec, "gt")
                x_est, y_est = _frame_xy(wave, t_local, view_window_sec, "est")
                gt_line.set_data(x_gt, y_gt)
                est_line.set_data(x_est, y_est)
                x0, x1 = _window_bounds(t_local, view_window_sec)
                ax.set_xlim(x0, x1)
            writer.grab_frame()
    plt.close(fig)
    print(f"Saved {out_path}")


def _parse_trial_spec(spec: str) -> Tuple[str, str]:
    dataset, condition = spec.split(":", 1)
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}' in trial spec '{spec}'")
    if condition not in DATASETS[dataset]["trials"]:
        raise ValueError(f"Unknown condition '{condition}' for {dataset}")
    return dataset, condition


AB04_SELECTED_TRIALS = [
    ("knee-exo", "RA_0p8mps"),
    ("hip-exo", "RA_0p8mps"),
    ("hip-exo", "LG_1p2mps"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "figures" / "ab04_streaming_moments",
        help="Output directory for MP4 files",
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--view-window", type=float, default=VIEW_WINDOW_SEC, help="Visible time window (s)")
    parser.add_argument("--combined-only", action="store_true", help="Only render the 3-panel combined video")
    parser.add_argument(
        "--trials",
        nargs="+",
        metavar="DATASET:CONDITION",
        help="Selected trials, e.g. knee-exo:RA_0p8mps hip-exo:LG_1p2mps",
    )
    parser.add_argument(
        "--selected",
        action="store_true",
        help="Render AB04 preset: knee RA 0.8, hip RA 0.8, hip LG 1.2",
    )
    args = parser.parse_args()

    view_window_sec = float(args.view_window)

    if args.selected or args.trials:
        trial_specs = (
            [_parse_trial_spec(spec) for spec in args.trials]
            if args.trials
            else AB04_SELECTED_TRIALS
        )
        waves = load_waveforms_for_trials(trial_specs)
        print(
            "Selected trials: "
            + ", ".join(f"{w.dataset} {w.condition} ({w.t[-1]:.1f}s)" for w in waves)
        )
        out_path = args.out_dir / f"ab04_changseob_selected_streaming{VIDEO_EXT}"
        render_panels_video(waves, out_path, fps=args.fps, view_window_sec=view_window_sec)
        if not args.combined_only:
            for wave in waves:
                slug = f"{wave.dataset.replace('-', '_')}_{wave.condition.lower()}"
                trial_path = args.out_dir / f"ab04_changseob_{slug}_streaming{VIDEO_EXT}"
                render_dataset_video([wave], trial_path, fps=args.fps, view_window_sec=view_window_sec)
        return

    waves_by_dataset = load_all_waveforms()
    for ds, waves in waves_by_dataset.items():
        print(
            f"{ds}: {len(waves)} trials, steady-state duration "
            f"{', '.join(f'{w.condition}={w.t[-1]:.1f}s' for w in waves)}"
        )

    combined_path = args.out_dir / f"ab04_changseob_streaming_moments_combined{VIDEO_EXT}"
    render_combined_video(waves_by_dataset, combined_path, fps=args.fps, view_window_sec=view_window_sec)

    if not args.combined_only:
        for ds, waves in waves_by_dataset.items():
            out_path = args.out_dir / f"ab04_changseob_{ds.replace('-', '_')}_streaming{VIDEO_EXT}"
            render_dataset_video(waves, out_path, fps=args.fps, view_window_sec=view_window_sec)


if __name__ == "__main__":
    main()

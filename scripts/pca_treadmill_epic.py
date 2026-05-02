#!/usr/bin/env python3
"""
Per-joint 2-D PCA of treadmill trials from Jinwoo_EPIC (S001–S056),
with pairwise distribution tests between the three cohort clusters.

Produces:
  pca_treadmill_per_joint.png     — 3×2 combined scatter (hip/knee/ankle × angle/moment)
  individual/pca_<joint>_<mod>.png — one PNG per panel
  distribution_tests_table.png    — summary table of all pairwise statistics
  distribution_tests.json / .csv  — raw numbers
  pca_per_joint_summary.json      — PCA variance explained

Cohort clusters:
  S001–S022  →  red   (Camargo)
  S023–S034  →  green (Scherpereel)
  S035–S056  →  blue  (Molinaro)

Statistical test:
  Hotelling's T² (two-sample, pooled covariance) on per-subject PC1/PC2 centroids.
  Statistical unit = subject (one mean PC1+PC2 vector per subject), so that
  per-frame rows do not inflate degrees of freedom.
  Effect size = Mahalanobis distance between group centroids (pooled covariance).

Usage
-----
  python scripts/pca_treadmill_epic.py [OPTIONS]

  --epic-root PATH      Jinwoo_EPIC folder
  --duration-sec FLOAT  Seconds of data per trial  [2.0]
  --output-dir PATH     Output directory  [results/pca_treadmill_epic]
  --alpha FLOAT         Scatter point alpha  [0.20]
  --pt-size FLOAT       Scatter point size   [2]
  --seed INT            [42]
  --show                Pop up an interactive window after saving
  --per-trial-centroid  One row per trial instead of per-frame
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError as e:
    raise SystemExit("pip install h5py") from e

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
except ImportError as e:
    raise SystemExit("pip install matplotlib") from e

try:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import f as f_dist
except ImportError as e:
    raise SystemExit("pip install scikit-learn scipy") from e

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dataset import _read_h5_opensim_table  # noqa: E402

# ---------------------------------------------------------------------------
# Joint definitions
# ---------------------------------------------------------------------------
JOINTS: List[Tuple[str, str, str, str, str]] = [
    # label           ik_col_r          ik_col_l          mom_base_r        mom_base_l
    ("Hip\nflexion", "hip_flexion_r",  "hip_flexion_l",  "hip_flexion_r",  "hip_flexion_l"),
    ("Knee",         "knee_angle_r",   "knee_angle_l",   "knee_angle_r",   "knee_angle_l"),
    ("Ankle",        "ankle_angle_r",  "ankle_angle_l",  "ankle_angle_r",  "ankle_angle_l"),
]

# ---------------------------------------------------------------------------
# Cohort clusters
# ---------------------------------------------------------------------------
CLUSTERS: Dict[str, Tuple[str, str]] = {
    "Camargo":              ("red",   "S001–S022  Camargo"),
    "Scherpereel":          ("green", "S023–S034  Scherpereel"),
    "Molinaro_Scherpereel": ("blue",  "S035–S056  Molinaro"),
}
CLUSTER_PAIRS = list(combinations(list(CLUSTERS.keys()), 2))


def cluster_for(subject_id: str) -> Optional[str]:
    m = re.match(r"^S(\d+)$", subject_id.strip().upper())
    if not m:
        return None
    n = int(m.group(1))
    if 1 <= n <= 22:
        return "Camargo"
    if 23 <= n <= 34:
        return "Scherpereel"
    if 35 <= n <= 56:
        return "Molinaro_Scherpereel"
    return None


def is_treadmill(name: str) -> bool:
    return "treadmill" in name.lower()


# ---------------------------------------------------------------------------
# HDF5 loading
# ---------------------------------------------------------------------------
def load_trial(
    h5_path: Path,
    condition: str,
    trial: str,
    duration_sec: float,
) -> Optional[Dict]:
    try:
        with h5py.File(h5_path, "r") as h5f:
            if condition not in h5f or trial not in h5f[condition]:
                return None
            tg = h5f[condition][trial]
            result: Dict = {}
            for modality in ("ik", "id"):
                if modality not in tg or len(tg[modality].keys()) == 0:
                    result[modality] = None
                    result[f"{modality}_cols"] = []
                    continue
                key = sorted(tg[modality].keys())[0]
                cols, data = _read_h5_opensim_table(tg[modality][key])
                result[modality] = data
                result[f"{modality}_cols"] = cols
    except Exception:
        return None

    for mod in ("ik", "id"):
        arr = result[mod]
        cols = result[f"{mod}_cols"]
        if arr is None or "time" not in cols:
            result[mod] = None
            continue
        t = arr[:, cols.index("time")].astype(np.float64)
        mask = (t >= t[0]) & (t <= t[0] + duration_sec)
        result[mod] = arr[mask] if np.any(mask) else None

    return result


def extract_ik_pair(td: Dict, col_r: str, col_l: str) -> Optional[np.ndarray]:
    arr, cols = td.get("ik"), td.get("ik_cols", [])
    if arr is None:
        return None
    out = np.full((arr.shape[0], 2), np.nan)
    if col_r in cols:
        out[:, 0] = np.deg2rad(arr[:, cols.index(col_r)])
    if col_l in cols:
        out[:, 1] = np.deg2rad(arr[:, cols.index(col_l)])
    return out


def extract_id_pair(td: Dict, base_r: str, base_l: str) -> Optional[np.ndarray]:
    arr, cols = td.get("id"), td.get("id_cols", [])
    if arr is None:
        return None
    out = np.full((arr.shape[0], 2), np.nan)
    for j, base in enumerate((base_r, base_l)):
        col = f"{base}_moment"
        if col in cols:
            out[:, j] = arr[:, cols.index(col)]
    return out


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def fit_pca_2d(X: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Impute → StandardScale → PCA(2).  Returns (Z, explained_variance_ratio)."""
    imp = SimpleImputer(strategy="mean")
    X_z = StandardScaler().fit_transform(imp.fit_transform(X.astype(np.float64)))
    n_comp = min(2, X_z.shape[1], X_z.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    return pca.fit_transform(X_z), pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def hotelling_t2(X1: np.ndarray, X2: np.ndarray) -> Tuple[float, float, float]:
    """
    Two-sample Hotelling's T² (pooled covariance).
    Returns (T², F, p_value).  Statistical unit = rows (use per-subject means).
    """
    n1, p = X1.shape
    n2 = X2.shape[0]
    mu1, mu2 = X1.mean(0), X2.mean(0)
    S1 = np.cov(X1, rowvar=False) if n1 > 1 else np.zeros((p, p))
    S2 = np.cov(X2, rowvar=False) if n2 > 1 else np.zeros((p, p))
    S_pool = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    S_inv = np.linalg.pinv(S_pool + np.eye(p) * 1e-10)
    diff = mu1 - mu2
    T2 = float((n1 * n2) / (n1 + n2) * diff @ S_inv @ diff)
    dof2 = n1 + n2 - p - 1
    if dof2 <= 0:
        return T2, np.nan, np.nan
    F = float((n1 + n2 - p - 1) / ((n1 + n2 - 2) * p) * T2)
    p_val = float(1.0 - f_dist.cdf(F, p, dof2))
    return T2, F, p_val


def mahalanobis_D(X1: np.ndarray, X2: np.ndarray) -> float:
    """Mahalanobis distance between group centroids (pooled covariance)."""
    n1, p = X1.shape
    n2 = X2.shape[0]
    mu1, mu2 = X1.mean(0), X2.mean(0)
    S1 = np.cov(X1, rowvar=False) if n1 > 1 else np.zeros((p, p))
    S2 = np.cov(X2, rowvar=False) if n2 > 1 else np.zeros((p, p))
    S_pool = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    S_inv = np.linalg.pinv(S_pool + np.eye(p) * 1e-10)
    diff = mu1 - mu2
    return float(np.sqrt(max(0.0, diff @ S_inv @ diff)))


def sig_stars(p: Optional[float]) -> str:
    if p is None or np.isnan(p):
        return "?"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def run_pairwise_stats(
    Z: np.ndarray,
    subj_arr: np.ndarray,
    cl_arr: np.ndarray,
) -> Dict:
    """
    Per-subject centroid in PC space → pairwise Hotelling's T² + Mahalanobis D.
    Returns dict keyed by '<cl1>_vs_<cl2>'.
    """
    # Aggregate to one vector per subject
    subjects = np.unique(subj_arr)
    centroids = np.array([Z[subj_arr == s].mean(0) for s in subjects])
    clusters  = np.array([cl_arr[subj_arr == s][0] for s in subjects])

    results: Dict = {}
    for cl1, cl2 in CLUSTER_PAIRS:
        X1 = centroids[clusters == cl1]
        X2 = centroids[clusters == cl2]
        key = f"{cl1}_vs_{cl2}"
        if X1.shape[0] < 2 or X2.shape[0] < 2:
            results[key] = None
            continue
        T2, F, p = hotelling_t2(X1, X2)
        D = mahalanobis_D(X1, X2)
        results[key] = {
            "group1": cl1,
            "group2": cl2,
            "n1": int(X1.shape[0]),
            "n2": int(X2.shape[0]),
            "T2":  round(float(T2), 3),
            "F":   round(float(F),  3) if not np.isnan(F) else None,
            "p_value":        round(float(p), 6) if not np.isnan(p) else None,
            "sig":            sig_stars(p),
            "mahalanobis_D":  round(float(D), 3),
        }
    return results


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _stats_annotation(stats: Dict) -> str:
    """Compact 3-line stats text for a panel."""
    short = {
        "Camargo":              "Cam",
        "Scherpereel":          "Sch",
        "Molinaro_Scherpereel": "Mol",
    }
    lines = []
    for cl1, cl2 in CLUSTER_PAIRS:
        key = f"{cl1}_vs_{cl2}"
        r = stats.get(key)
        if r is None:
            lines.append(f"{short[cl1]} vs {short[cl2]}: —")
        else:
            p_str = f"p={r['p_value']:.3f}" if r["p_value"] is not None else "p=?"
            lines.append(
                f"{short[cl1]} vs {short[cl2]}: {r['sig']}  {p_str}  D={r['mahalanobis_D']:.2f}"
            )
    return "\n".join(lines)


def _draw_panel(
    ax: plt.Axes,
    pd: Optional[Dict],
    args: argparse.Namespace,
    title_fontsize: int = 9,
) -> None:
    if pd is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    Z, ev, cl_arr = pd["Z"], pd["ev"], pd["cl_arr"]

    for cl_name, (color, _) in CLUSTERS.items():
        mask = cl_arr == cl_name
        if not np.any(mask):
            continue
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   c=color, s=args.pt_size, alpha=args.alpha,
                   linewidths=0, rasterized=True)

    # Overlay per-subject centroids as larger, opaque markers
    subj_arr = pd["subj_arr"]
    for s in np.unique(subj_arr):
        cl = cl_arr[subj_arr == s][0]
        color = CLUSTERS[cl][0]
        c = Z[subj_arr == s].mean(0)
        ax.scatter(c[0], c[1], c=color, s=28, marker="D",
                   edgecolors="white", linewidths=0.4, zorder=5, alpha=0.85)

    pc2_pct = pd["pc2_pct"]
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f} %)", fontsize=title_fontsize - 1)
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f} %)",   fontsize=title_fontsize - 1)
    ax.tick_params(labelsize=title_fontsize - 2)
    ax.grid(True, alpha=0.2)
    ax.set_title(f"{pd['joint_short']} — {pd['col_title']}",
                 fontsize=title_fontsize, pad=4)

    # Stats annotation box
    stats_text = _stats_annotation(pd["stats"])
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=max(5, title_fontsize - 3),
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.80, ec="0.6"),
        family="monospace",
    )


# ---------------------------------------------------------------------------
# Stats summary table figure
# ---------------------------------------------------------------------------
def save_stats_table_figure(
    all_stats: Dict[str, Dict],   # key = "joint_modality", value = pairwise stats dict
    out_path: Path,
    mode_str: str,
) -> None:
    """Render a concise table figure of all pairwise test results."""
    short_cl = {
        "Camargo":              "Cam",
        "Scherpereel":          "Sch",
        "Molinaro_Scherpereel": "Mol",
    }
    pair_labels = [f"{short_cl[a]} vs {short_cl[b]}" for a, b in CLUSTER_PAIRS]

    row_labels = list(all_stats.keys())  # e.g. "Hip flexion_angle"
    n_rows = len(row_labels)
    n_pairs = len(CLUSTER_PAIRS)

    # Build cell text arrays
    cell_p   = [[""] * n_pairs for _ in range(n_rows)]
    cell_D   = [[""] * n_pairs for _ in range(n_rows)]
    cell_sig = [[""] * n_pairs for _ in range(n_rows)]
    cell_col = [["white"] * n_pairs for _ in range(n_rows)]

    SIG_COLOR = {"***": "#d62728", "**": "#ff7f0e", "*": "#bcbd22", "ns": "#aec7e8", "?": "#999"}

    for ri, panel_key in enumerate(row_labels):
        stats = all_stats[panel_key]
        for pi, (cl1, cl2) in enumerate(CLUSTER_PAIRS):
            r = stats.get(f"{cl1}_vs_{cl2}")
            if r is None:
                cell_p[ri][pi] = "—"
                cell_D[ri][pi] = "—"
                cell_sig[ri][pi] = "—"
            else:
                p = r["p_value"]
                cell_p[ri][pi]   = f"{p:.4f}" if p is not None else "?"
                cell_D[ri][pi]   = f"{r['mahalanobis_D']:.2f}"
                cell_sig[ri][pi] = r["sig"]
                cell_col[ri][pi] = SIG_COLOR.get(r["sig"], "white")

    # Table: rows = panels, columns = pairs × {p, D, sig}
    col_labels_top = []
    for pl in pair_labels:
        col_labels_top += [pl, "", ""]

    # Flatten cells
    flat_cells = []
    flat_colors = []
    for ri in range(n_rows):
        row_vals, row_cols = [], []
        for pi in range(n_pairs):
            row_vals += [cell_sig[ri][pi], cell_p[ri][pi], cell_D[ri][pi]]
            row_cols += [cell_col[ri][pi], "white", "white"]
        flat_cells.append(row_vals)
        flat_colors.append(row_cols)

    sub_col_labels = ["sig", "p", "D(Mah)"] * n_pairs

    fig_h = max(4, 0.45 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=flat_cells,
        rowLabels=row_labels,
        colLabels=sub_col_labels,
        cellColours=flat_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.5)

    # Group header row above table
    y_top = 1.02
    pair_span = 3  # sig + p + D
    for pi, pl in enumerate(pair_labels):
        x_left  = (pi * pair_span + 1) / (n_pairs * pair_span + 1)
        x_right = ((pi + 1) * pair_span + 1) / (n_pairs * pair_span + 1)
        x_mid = (x_left + x_right) / 2
        ax.annotate(
            pl,
            xy=(x_mid, y_top), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_title(
        f"Pairwise Hotelling's T² — treadmill IK/ID PCA  ({mode_str})\n"
        f"Statistical unit: per-subject PC centroid  |  sig: * p<0.05  ** p<0.01  *** p<0.001  ns p≥0.05\n"
        f"D = Mahalanobis distance between group centroids (pooled covariance)",
        fontsize=9, pad=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-joint 2-D PCA + distribution tests on Jinwoo_EPIC treadmill trials"
    )
    p.add_argument("--epic-root",
                   default="/media/metamobility3/Samsung_T51/Processed/Jinwoo_EPIC")
    p.add_argument("--duration-sec", type=float, default=2.0)
    p.add_argument("--output-dir", default="results/pca_treadmill_epic")
    p.add_argument("--alpha",   type=float, default=0.20)
    p.add_argument("--pt-size", type=float, default=2.0)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--show",    action="store_true")
    p.add_argument("--per-trial-centroid", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    epic_root = Path(args.epic_root)
    if not epic_root.is_dir():
        raise SystemExit(f"EPIC root not found: {epic_root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = out_dir / "individual"
    individual_dir.mkdir(exist_ok=True)

    h5_files = sorted(
        p for p in epic_root.glob("S*.h5") if cluster_for(p.stem) is not None
    )
    if not h5_files:
        raise SystemExit(f"No S001–S056 .h5 files found in {epic_root}")

    print(f"Found {len(h5_files)} subject files")

    # ------------------------------------------------------------------
    # Data collection
    # rows_*[ji]   = list of (T, 2) arrays
    # cluster_*[ji] = parallel cluster label per row
    # subj_*[ji]    = parallel subject ID per row
    # ------------------------------------------------------------------
    n_joints = len(JOINTS)
    rows_ik:    List[List[np.ndarray]] = [[] for _ in range(n_joints)]
    rows_id:    List[List[np.ndarray]] = [[] for _ in range(n_joints)]
    cluster_ik: List[List[str]]        = [[] for _ in range(n_joints)]
    cluster_id: List[List[str]]        = [[] for _ in range(n_joints)]
    subj_ik:    List[List[str]]        = [[] for _ in range(n_joints)]
    subj_id:    List[List[str]]        = [[] for _ in range(n_joints)]

    for h5_path in h5_files:
        sid     = h5_path.stem.upper()
        cluster = cluster_for(sid)

        with h5py.File(h5_path, "r") as h5f:
            conditions = sorted(h5f.keys())

        treadmill_conds = [c for c in conditions if is_treadmill(c)]
        if not treadmill_conds:
            continue

        n_trials = 0
        for cond in treadmill_conds:
            with h5py.File(h5_path, "r") as h5f:
                if cond not in h5f:
                    continue
                trials = sorted(h5f[cond].keys())

            for trial in trials:
                td = load_trial(h5_path, cond, trial, args.duration_sec)
                if td is None:
                    continue

                for ji, (_, col_r, col_l, base_r, base_l) in enumerate(JOINTS):
                    for data, rows_acc, cl_acc, subj_acc, extractor, extra_args in [
                        (extract_ik_pair(td, col_r, col_l),
                         rows_ik[ji], cluster_ik[ji], subj_ik[ji], None, None),
                        (extract_id_pair(td, base_r, base_l),
                         rows_id[ji], cluster_id[ji], subj_id[ji], None, None),
                    ]:
                        if data is None:
                            continue
                        good = np.mean(np.isfinite(data), axis=1) > 0.5
                        data = data[good]
                        if data.shape[0] == 0:
                            continue
                        if args.per_trial_centroid:
                            rows_acc.append(np.nanmean(data, axis=0, keepdims=True))
                            cl_acc.append(cluster)
                            subj_acc.append(sid)
                        else:
                            rows_acc.append(data)
                            cl_acc.extend([cluster] * data.shape[0])
                            subj_acc.extend([sid] * data.shape[0])

                n_trials += 1
        print(f"  {sid} ({cluster}): {n_trials} treadmill trial(s)")

    # ------------------------------------------------------------------
    # PCA + stats per panel
    # ------------------------------------------------------------------
    col_titles = ["Joint Angle (rad)", "Joint Moment (N·m/kg)"]
    col_slugs  = ["angle",             "moment"]
    mode_str   = "per-trial centroid" if args.per_trial_centroid else "per-frame"

    panel_data: List[List[Optional[Dict]]] = []
    pca_summary: Dict = {}
    all_stats_flat: Dict[str, Dict] = {}   # for stats table figure

    print("\n--- PCA + Hotelling's T² (per-subject centroids) ---")
    for ji, (joint_label, col_r, col_l, base_r, base_l) in enumerate(JOINTS):
        joint_short = joint_label.replace("\n", " ")
        row_panels = []
        for mi, (data_rows, cl_list, subj_list, col_title) in enumerate([
            (rows_ik[ji], cluster_ik[ji], subj_ik[ji], col_titles[0]),
            (rows_id[ji], cluster_id[ji], subj_id[ji], col_titles[1]),
        ]):
            panel_key = f"{joint_short}_{col_slugs[mi]}"
            if not data_rows:
                row_panels.append(None)
                continue

            X        = np.vstack(data_rows).astype(np.float64)
            cl_arr   = np.array(cl_list)
            subj_arr = np.array(subj_list)
            Z, ev    = fit_pca_2d(X, seed=args.seed)
            pc2_pct  = float(ev[1] * 100) if len(ev) > 1 else 0.0

            stats = run_pairwise_stats(Z, subj_arr, cl_arr)

            row_panels.append(dict(
                X=X, Z=Z, ev=ev,
                cl_arr=cl_arr, subj_arr=subj_arr,
                joint_short=joint_short, col_title=col_title,
                pc2_pct=pc2_pct, stats=stats,
            ))

            pca_summary[panel_key] = {
                "n_rows": int(X.shape[0]),
                "n_subjects": int(len(np.unique(subj_arr))),
                "ev_pc1":    float(ev[0]),
                "ev_pc2":    float(ev[1]) if len(ev) > 1 else 0.0,
                "ev_pc1_pc2": float(ev[0] + (ev[1] if len(ev) > 1 else 0)),
            }
            all_stats_flat[panel_key] = stats

            for cl1, cl2 in CLUSTER_PAIRS:
                key = f"{cl1}_vs_{cl2}"
                r = stats.get(key)
                s = (f"  p={r['p_value']:.4f} {r['sig']:3s}  D={r['mahalanobis_D']:.2f}"
                     f"  (n={r['n1']},{r['n2']})")  if r else "  —"
                short = {"Camargo": "Cam", "Scherpereel": "Sch",
                         "Molinaro_Scherpereel": "Mol"}
                print(f"  [{joint_short}] {col_slugs[mi]:6s}  "
                      f"{short[cl1]} vs {short[cl2]}: {s}")

        panel_data.append(row_panels)

    # ------------------------------------------------------------------
    # Save JSON + CSV
    # ------------------------------------------------------------------
    with open(out_dir / "pca_per_joint_summary.json", "w") as f:
        json.dump(pca_summary, f, indent=2)

    flat_rows = []
    for panel_key, stats in all_stats_flat.items():
        for pair_key, r in stats.items():
            if r is None:
                continue
            flat_rows.append({
                "panel": panel_key,
                "comparison": pair_key,
                **{k: v for k, v in r.items() if k not in ("group1", "group2")},
            })

    with open(out_dir / "distribution_tests.json", "w") as f:
        json.dump(all_stats_flat, f, indent=2)

    csv_path = out_dir / "distribution_tests.csv"
    if flat_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            w.writeheader()
            w.writerows(flat_rows)
        print(f"\nStats CSV: {csv_path.resolve()}")

    # ------------------------------------------------------------------
    # Legend patches (shared)
    # ------------------------------------------------------------------
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for _, (color, label) in CLUSTERS.items()
    ]

    # ------------------------------------------------------------------
    # Individual PNGs (one per panel)
    # ------------------------------------------------------------------
    for ji, (joint_label, *_) in enumerate(JOINTS):
        joint_slug = joint_label.replace("\n", "_").replace(" ", "_").lower()
        for mi, slug in enumerate(col_slugs):
            pd = panel_data[ji][mi]
            fig_i, ax_i = plt.subplots(figsize=(7, 5.5))
            _draw_panel(ax_i, pd, args, title_fontsize=11)
            ax_i.legend(handles=legend_patches, fontsize=9,
                        loc="upper right", framealpha=0.85)
            fig_i.suptitle(f"Treadmill PCA — Jinwoo EPIC  ({mode_str})",
                           fontsize=10, y=1.01)
            fig_i.tight_layout()
            fname = individual_dir / f"pca_{joint_slug}_{slug}.png"
            fig_i.savefig(fname, dpi=180, bbox_inches="tight")
            plt.close(fig_i)
            print(f"  Saved: {fname.resolve()}")

    # ------------------------------------------------------------------
    # Combined 3 × 2 figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 16))
    gs = GridSpec(
        n_joints, 2, figure=fig,
        hspace=0.50, wspace=0.32,
        left=0.08, right=0.97, top=0.93, bottom=0.05,
    )
    for ji in range(n_joints):
        for mi in range(2):
            ax = fig.add_subplot(gs[ji, mi])
            _draw_panel(ax, panel_data[ji][mi], args)

    fig.legend(handles=legend_patches, loc="upper center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(
        f"Treadmill PCA — Jinwoo EPIC  ({mode_str})\n"
        f"◆ = per-subject centroid  |  box: Hotelling T² (per-subject)  sig: * p<.05  ** p<.01  *** p<.001",
        fontsize=11, y=1.00,
    )
    combined_path = out_dir / "pca_treadmill_per_joint.png"
    fig.savefig(combined_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCombined plot saved:    {combined_path.resolve()}")

    # ------------------------------------------------------------------
    # Stats table figure
    # ------------------------------------------------------------------
    table_path = out_dir / "distribution_tests_table.png"
    save_stats_table_figure(all_stats_flat, table_path, mode_str)
    print(f"Stats table saved:      {table_path.resolve()}")

    print(f"\nAll outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

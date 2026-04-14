# Jinwoo (MeMo harmonized) — processed H5 bundle

This README describes **`Processed/Jinwoo`**: **one HDF5 file per subject** (`S001.h5` … `S070.h5`) built from **`Processed/MeMo_Final`** with cohort-specific sign harmonization (`os_kinetics/build_jinwoo_memo_h5.py`). The on-disk layout matches the standard MeMo flat export: **no per-subject folders**, only subject H5 files at the dataset root.

**Install note:** Copy or symlink this file to `Processed/Jinwoo/README.md` next to the H5 files if you want it colocated on the data disk.

## Sagittal sign convention (angles and moments)

For this release, **sagittal** kinematics and kinetics are aligned so that **counterclockwise rotation about an axis that points to the subject’s right is positive** when viewed in the usual right-side (sagittal) diagram:

| Joint   | Angle / moment sense (sagittal) |
|--------|----------------------------------|
| **Hip**   | Hip **flexion** is **positive**. |
| **Knee**  | Knee **flexion** is **negative** (knee **extension** is positive). |
| **Ankle** | **Dorsiflexion** is **positive** (plantarflexion is negative). |

The same right-handed / “right-axis CCW positive” rule applies to the corresponding **joint moments** in the ID tables, after the harmonization passes described below.

Non-sagittal coordinates (e.g. pelvis list, hip adduction, hip rotation, lumbar, subtalar, MTP) keep the column naming of the underlying OpenSim pipeline; interpret them with the same global right-handed conventions as in the source models.

## Layout (inside each `S###.h5`)

```
/<condition_name>/<trial_NN>/
    imu/          # per-segment datasets (e.g. pelvis, right_thigh, right_shank, right_foot)
    ik/<stem>     # inverse kinematics table; columns attr = JSON list (time + joint angles in degrees)
    id/<stem>     # inverse dynamics table; columns attr = JSON list (time + moments in N·m/kg, etc.)
```

- **IMU:** one dataset per segment; `columns` attribute lists channel names (e.g. `time`, `*_acc_*`, `*_gyr_*`).
- **IK / ID:** OpenSim-style tables stored as 2D float arrays; **`columns`** is required to map columns.

## Snapshot (parsed from `/media/metamobility3/Samsung_T51/Processed/Jinwoo` on 2026-03-30)

| Quantity | Value |
|---------|--------|
| Subject H5 files | **70** (`S001`–`S070`) |
| Distinct condition names (union across files) | **218** |
| Total `trial_*` groups (sum over all subjects) | **6692** |
| Typical IMU segments | `pelvis`, `right_foot`, `right_shank`, `right_thigh` |
| Typical IK / ID width | **24** columns each (including `time`) |

Re-scan after updates:

```bash
python3 - << 'PY'
import h5py
from pathlib import Path
root = Path("/path/to/Processed/Jinwoo")
h5s = sorted(root.glob("S*.h5"))
conds, trials = set(), 0
for p in h5s:
    with h5py.File(p, "r") as f:
        for c in f:
            g = f[c]
            if not hasattr(g, "keys"):
                continue
            conds.add(c)
            for t in g:
                if str(t).startswith("trial_"):
                    trials += 1
print(len(h5s), "subjects,", len(conds), "conditions,", trials, "trial_* nodes")
PY
```

## Task family duration & trial counts (four families)

The following aggregates **only** trials whose condition name matches one of these prefixes: `incline_`, `stair_`, `levelground_`, `treadmill_`. Duration is the span of the `time` column from the **first IK table** in each `trial_*` group (see `os_kinetics/memo_task_duration_composition.py`). Trials outside those families (and any trial skipped by the script, e.g. missing IK) are **not** included—hence the trial count below is lower than the total `trial_*` count in the snapshot above.

Scan: **`/media/metamobility3/Samsung_T51/Processed/Jinwoo`** (same run that produced `os_kinetics/final_task_composition.png`).

| Metric | Value |
|--------|--------|
| Subjects (H5 files) | **70** |
| Trials counted (four families) | **5158** |
| Total time (four families) | **40.675 h** |

| Family (condition prefix) | Time | Trials | Share of four-family time |
|----------------------------|------|--------|---------------------------|
| `incline_*` | 14.053 h | 2307 | 34.5% |
| `stair_*` | 3.440 h | 1161 | 8.5% |
| `levelground_*` | 9.128 h | 857 | 22.4% |
| `treadmill_*` | 14.054 h | 833 | 34.6% |

Reproduce:

```bash
python memo_task_duration_composition.py \
  --memo-root /media/metamobility3/Samsung_T51/Processed/Jinwoo \
  --output final_task_composition.png
```

## Provenance and cohort harmonization

Source: **`Processed/MeMo_Final`**.

Processing script: **`os_kinetics/build_jinwoo_memo_h5.py`**.

Per-subject cohorts (inclusive indices). Values are **multiplied by −1** on the listed channels when copying into this dataset; everything else is copied unchanged.

| Subject IDs | Cohort label | IK columns negated | ID columns negated |
|-------------|--------------|-------------------|--------------------|
| S001–S022 | Camargo | — | `ankle_angle_{r,l}_moment`, `hip_flexion_{r,l}_moment` |
| S023–S034 | Scherpereel | `ankle_angle_r`, `ankle_angle_l` | `ankle_angle_{r,l}_moment` |
| S035–S056 | Molinaro | `ankle_angle_r`, `ankle_angle_l` | `knee_angle_{r,l}_moment`, `ankle_angle_{r,l}_moment` |
| S057–S070 | MeMo tail | `hip_flexion_r`, `hip_flexion_l`, `ankle_angle_r`, `ankle_angle_l` | — |

If present, **`dataset_metadata.json`** may include a **`jinwoo_sign_fix`** block echoing these rules. A per-build log may exist as **`jinwoo_h5_build_report.json`**.

## Files expected in `Processed/Jinwoo`

- **`S###.h5`** — subject bundle (see layout above).
- **`README.md`** — optional copy of this document.
- **`dataset_metadata.json`** / **`jinwoo_h5_build_report.json`** — optional.

## Citation

Use the original MeMo / contributing study citations from the upstream **`MeMo_Final`** metadata. This folder is a **re-packaged, sign-harmonized derivative** for joint modeling; cite the primary data papers and describe the harmonization step if you publish with these files.

# Ramp descent in open-source training data: imbalance quantification

## Purpose

This note responds to the reviewer comment that claims about ramp descent being underrepresented in open-source training data were not quantified. It documents (1) trial- vs window-level composition of the training corpus, (2) whether the deployed `checkpoints/` models were trained with locomotion-bucket balancing, and (3) how that affects interpretation of the original claim.

## Training corpus and labeling

- **Data root:** `AddBiomechanics_final` (Camargo / Scherpereel / Molinaro-style open-source walking; subject H5 bundles).
- **Typical filters used for the paper checkpoints:** `walking_only=true`, `exclude_stair_tasks=true` (level + ramp kept; stairs dropped).
- **Locomotion buckets:** `LG` (level / treadmill), `RA` (ramp ascent), `RD` (ramp descent), via `dataset.classify_loc_bucket` plus `jinwoo_addbiomechanics_final_ascent_descent_mapping.json` (odd/even trial map for ambiguous incline groups).
- **Split:** subject-level holdout (not condition-level). Example finetune inventory on S001–S056: 51 train / 2 val / 3 test.

Family-level duration stats previously published for this corpus (`incline_*` ≈ 34.5% of four-family time) **do not** separate ascent from descent and therefore cannot support an RD-specific scarcity claim.

## Quantification

### Trial counts (S001–S056, stairs excluded, ascent/descent map applied)

| Bucket | Trials | Share |
|--------|--------|-------|
| LG | 3,774 | 31.7% |
| RA | 3,899 | 32.7% |
| RD | 3,843 | 32.2% |

**RA ≈ RD at the trial level** (ratio ≈ 1.01). Underrepresentation of RD is **not** supported by trial counts alone.

### Sliding-window counts (effective training exposure)

Measured on the same cohort / filter style used for LG–RA–RD finetunes (`runs/0709_knee_finetune_stage4_lg_ra_rd_light_aug/finetune_balance.json`, `counts_before`):

| Bucket | Windows | Share of LG+RA+RD |
|--------|---------|-------------------|
| LG | 7,362,539 | **84.3%** |
| RA | 931,791 | **10.7%** |
| RD | 440,661 | **5.0%** |

Relative exposure:

- RD ≈ **0.47×** RA
- RD ≈ **0.06×** LG
- LG : RA : RD ≈ **16.7 : 2.1 : 1**

So RD **is underrepresented in window / duration mass**, even though trial labels are nearly balanced. Longer level/treadmill (and, secondarily, ascent) recordings drive this skew.

## Were `checkpoints/` models trained with balancing?

**Yes — all four published checkpoints enabled locomotion-bucket oversampling.**

| Checkpoint | Source run | `balance_loc_buckets_oversample` | Stairs excluded | Ascent/descent map |
|------------|------------|----------------------------------|-----------------|--------------------|
| `checkpoints/hip/` | `runs/0512_ik_id_hip_offline_zero_phase` | **true** | true | yes |
| `checkpoints/knee/` | `runs/0512_ik_id_knee_offline_zero_phase` | **true** | true | yes |
| `checkpoints/ankle/` | `runs/0512_ik_id_ankle_offline_zero_phase` | **true** | true | yes |
| `checkpoints/hip-knee-ankle/` | `runs/0512_ik_id_all_zero_in_zero_out` | **true** | true | yes |

Mechanism (`dataset.KineticsTCNDataset._apply_loc_bucket_oversample`): among buckets present in the loader (`LG`, `RA`, `RD`; stairs already excluded), windows are duplicated so each bucket reaches the **max** per-bucket count (typically LG). Raw RD scarcity is therefore **mitigated at train time** by oversampling, not left unaddressed.

## Implications for the paper claim

**Supported (quantified):** In the open-source corpus used for training, **raw sliding-window exposure** is heavily skewed toward level walking, and RD contributes roughly half as many windows as RA (~5% vs ~11% of LG+RA+RD).

**Not supported by trial inventory:** RD is not scarce as a number of trials relative to RA.

**Weakened as a sole explanation of residual RD error:** Because the deployed checkpoints were trained with `balance_loc_buckets_oversample=true`, remaining RD-specific errors (if any) cannot be attributed only to “few RD samples in the loss.” More plausible residual factors include domain shift (lab open-source ramps vs in-house ramp protocols), grade/speed differences, exo-on vs exo-off kinematics, or participant-specific strategies — which still require separate biomechanical analysis.

## Suggested manuscript wording

> In the open-source training corpus (walking trials, stairs excluded), sliding windows were heavily skewed toward level walking (84.3% LG vs 10.7% RA vs 5.0% RD). Trial counts for ramp ascent and descent were nearly balanced, so the underrepresentation is primarily in duration / window exposure rather than number of trials. Models were trained with locomotion-bucket oversampling to equalize LG/RA/RD window counts during optimization; therefore residual ramp-descent errors, if present, are unlikely to be explained by raw sample scarcity alone and may reflect domain or strategy differences that warrant further biomechanical analysis.

## Sources

- Window imbalance: `runs/0709_knee_finetune_stage4_lg_ra_rd_light_aug/finetune_balance.json` (and matching `counts_before` in 0707/0708 finetunes)
- Family duration (no RA/RD split): `scripts/jinwoo_dataset_README.md`
- Bucket labeling / oversample: `dataset.py` (`classify_loc_bucket`, `_apply_loc_bucket_oversample`)
- Checkpoint configs: `checkpoints/{hip,knee,ankle,hip-knee-ankle}/config.json`

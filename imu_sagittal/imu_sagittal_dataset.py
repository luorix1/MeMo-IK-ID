"""
Backward-compatible imports for IMU sagittal training/eval.

**24** IMU inputs per sample (pelvis + one-side thigh/shank/foot), **3** sagittal outputs for that side.
See ``imu_sagittal.imu_sagittal_leg_dataset``.
"""

from imu_sagittal.imu_sagittal_leg_dataset import (  # noqa: F401
    IMU_UNILATERAL_N_CHANNELS,
    ImuSagittalH5Dataset,
    TrialRef,
    discover_imu_schema_first_trial,
    discover_imu_schemas_paired_first_trial,
    imu_lower_limb_segment_order,
    imu_paired_chain_orders,
    imu_segment_order_for_laterality,
    imu_unilateral_24_segment_order,
    molinaro_subject_ids,
    sorted_imu_schema_for_leg,
)

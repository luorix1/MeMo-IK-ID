#!/usr/bin/env python3
"""
Evaluate a trained **IMU → sagittal joint angle** model (same role as ``ik_id.test`` for ``ik_id.train``).

Loads a checkpoint from an IMU sagittal training run, applies saved IMU normalization and schema,
evaluates on H5 subjects, and writes ``metrics.json``, ``eval_subjects.json``, and plots.

Example::

    python imu_sagittal/test_imu_sagittal_angles.py \\
        --checkpoint runs/imu_sagittal_angles/best_model.pt \\
        --test-dir /path/to/Processed/Jinwoo \\
        --output-dir results/imu_angle_eval
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from imu_sagittal.imu_sagittal_eval import run_main

if __name__ == "__main__":
    run_main("angle")

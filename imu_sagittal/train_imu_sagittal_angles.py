#!/usr/bin/env python3
"""Train IMU → sagittal joint angles (24 IMU ch → 3 right-leg DOFs). See imu_sagittal/train_imu_sagittal.py."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from imu_sagittal.train_imu_sagittal import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--target", "angle"] + sys.argv[1:]
    main()

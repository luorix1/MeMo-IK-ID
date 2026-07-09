"""Unit tests for RA/RD-only trial filtering."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import classify_loc_bucket, include_condition_for_dataset, is_ramp_loc_bucket


def test_is_ramp_loc_bucket():
    assert is_ramp_loc_bucket("RA")
    assert is_ramp_loc_bucket("RD")
    assert is_ramp_loc_bucket("RA/RD")
    assert not is_ramp_loc_bucket("LG")
    assert not is_ramp_loc_bucket("SA")


def test_ra_rd_only_includes_incline_treadmill():
    cond = "incline_treadmill_up10deg"
    assert classify_loc_bucket("S024", cond, "trial_01") == "RA"
    assert include_condition_for_dataset(
        cond,
        walking_only=True,
        levelground_only=False,
        subject_id="S024",
        exclude_stair_tasks=True,
        ra_rd_only=True,
        trial_name="trial_01",
    )


def test_ra_rd_only_excludes_levelground():
    cond = "levelground_walk_1"
    assert classify_loc_bucket("S001", cond, "trial_01") == "LG"
    assert not include_condition_for_dataset(
        cond,
        walking_only=True,
        levelground_only=False,
        subject_id="S001",
        ra_rd_only=True,
        trial_name="trial_01",
    )


def test_ra_rd_only_uses_loc_map():
    loc_map = {("S057", "ambiguous_ramp_block", "trial_01"): "RA"}
    assert classify_loc_bucket("S057", "ambiguous_ramp_block", "trial_01", loc_map) == "RA"
    assert classify_loc_bucket("S057", "ambiguous_ramp_block", "trial_01") == "OTHER"
    assert include_condition_for_dataset(
        "ambiguous_ramp_block",
        walking_only=True,
        levelground_only=False,
        subject_id="S057",
        ra_rd_only=True,
        loc_map=loc_map,
        trial_name="trial_01",
    )
    assert not include_condition_for_dataset(
        "ambiguous_ramp_block",
        walking_only=True,
        levelground_only=False,
        subject_id="S057",
        ra_rd_only=True,
        loc_map=loc_map,
        trial_name="trial_02",
    )


def test_ra_rd_only_requires_trial_name():
    assert not include_condition_for_dataset(
        "incline_treadmill_up10deg",
        walking_only=True,
        levelground_only=False,
        ra_rd_only=True,
        trial_name=None,
    )

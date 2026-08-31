#!/usr/bin/env python3
"""Verify Vicon IK paths match compare_processed_{hip,knee}_exo_id.ipynb rules."""
import io
import re
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = Path('/media/metamobility3/Samsung_T52/Results/processed')
SUBJECT_TOKEN_TO_DIR = {
    'ab01_jinwoo': 'AB01_Jinwoo', 'ab02_oscar': 'AB02_Oscar', 'ab03_ilseung': 'AB03_Ilseung',
    'ab04_changseob': 'AB04_Changseob', 'ab05_maria': 'AB05_Maria', 'ab06_jimin': 'AB06_Jimin',
    'ab07_amy': 'AB07_Amy', 'ab08_seokhyun': 'AB08_Seokhyun',
}
SUBJECT_DIR_OVERRIDE: dict = {}


def _subject_token(stem: str) -> str:
    return '_'.join(stem.lower().split('_')[:2])


def subject_dir_from_stem(stem: str) -> Path:
    token = _subject_token(stem)
    if token in SUBJECT_DIR_OVERRIDE:
        return Path(SUBJECT_DIR_OVERRIDE[token])
    return PROCESSED_ROOT / SUBJECT_TOKEN_TO_DIR[token]


def trial_cond_speed(stem: str):
    parts = stem.lstrip('_').lower().split('_')
    cond = re.sub(r'\d+$', '', parts[4]).upper()
    return cond, parts[3]


def ik_path(stem: str, exo_kind: str) -> Path:
    cond, speed = trial_cond_speed(stem)
    return subject_dir_from_stem(stem) / exo_kind / 'ik' / f'{cond}_{speed}_ik.mot'


def list_stems(pattern: str):
    return sorted(p.stem.lstrip('_') for p in PROJECT.glob(pattern))


def main():
    if not PROCESSED_ROOT.is_dir():
        raise SystemExit(f'Mount sda1: sudo mount -t exfat /dev/sda1 /media/metamobility3/Samsung_T52\n  Missing {PROCESSED_ROOT}')

    for exo_kind, pattern in [('hip-exo', '*_hip_*_exo_on.npz'), ('knee-exo', '*_knee_*_exo_on.npz')]:
        print(f'\n=== {exo_kind} ===')
        ok = miss = 0
        for stem in list_stems(pattern):
            p = ik_path(stem, exo_kind)
            if p.is_file():
                ok += 1
                print(f'  OK  {stem}\n       {p}')
            else:
                miss += 1
                print(f'  MISS {stem}\n       {p}')
        print(f'Summary: {ok} found, {miss} missing')


if __name__ == '__main__':
    main()

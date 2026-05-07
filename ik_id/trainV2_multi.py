#!/usr/bin/env python3
"""
Run multiple trainV2-style IK->ID experiments from one JSON file.

This script is meant for hyperparameter sweeps where the expensive dataset
construction is identical across runs. It loads each compatible train/val
dataset bundle once, then recreates DataLoaders, models, optimizers, W&B runs,
and checkpoints for each experiment config.

Example JSON:

{
  "base_config": "runs/0507_ik_id_hip/config.json",
  "configs": [
    {"name": "hip_rf57", "n_blocks": 3},
    {"name": "hip_rf121", "n_blocks": 4},
    {"name": "hip_window249", "window_size": 249, "n_blocks": 5}
  ]
}

Configs may also be a top-level list of full config objects. If a config has a
``name`` but no explicit ``output_dir``, the output directory is derived from
the base output directory's parent.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import KineticsTCNDataset, extract_subject_id, find_trial_dirs
from model import GaussianDiffusion1D, TCN, TransformerMoment
from training_utils import MomentLoss, evaluate, set_global_seed, train_one_epoch

try:
    from .trainV2 import (
        HAS_WANDB,
        _save_checkpoint,
        _write_run_metadata_json,
        plot_curves,
        plot_sample_prediction,
        train_one_epoch_diffusion,
    )
except ImportError:
    from trainV2 import (
        HAS_WANDB,
        _save_checkpoint,
        _write_run_metadata_json,
        plot_curves,
        plot_sample_prediction,
        train_one_epoch_diffusion,
    )

if HAS_WANDB:
    import wandb
else:
    wandb = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "train_dir": None,
    "output_dir": "runs/tcn_run_v2",
    "window_size": 200,
    "stride": 1,
    "batch_size": 256,
    "epochs": 50,
    "num_workers": 4,
    "max_train_files": None,
    "max_val_files": None,
    "n_val_subjects": 1,
    "n_test_subjects": 2,
    "val_subjects": None,
    "test_subjects": None,
    "seed": 42,
    "walking_only": True,
    "levelground_only": False,
    "balance_loc_buckets_oversample": False,
    "loc_bucket_balance_seed": None,
    "loc_ascent_descent_map": "./jinwoo_addbiomechanics_final_ascent_descent_mapping.json",
    "lowpass_cutoff_hz": 6.0,
    "lowpass_order": 4,
    "rollout": False,
    "rollout_decimate_step": None,
    "velocity_lowpass_filter": True,
    "velocity_lowpass_cutoff_hz": None,
    "velocity_lowpass_order": None,
    "model_type": "tcn",
    "hidden_channels": 80,
    "n_blocks": 5,
    "kernel_size": 5,
    "dropout": 0.1,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "n_diffusion_timesteps": 1000,
    "diffusion_schedule": "cosine",
    "diffusion_predict_epsilon": True,
    "n_inference_steps": 50,
    "input_mode": "lower_limb",
    "output_mode": "sagittal_hip_knee_ankle",
    "laterality": "unilateral",
    "legacy_unilateral_full_window": False,
    "lr": 5e-6,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "early_stopping_patience": 4,
    "input_noise_std": 0.0,
    "angle_jitter_std": 0.0,
    "correlated_vel_noise": False,
    "loss_type": "mse",
    "huber_delta": 0.5,
    "dof_loss_weights": None,
    "smoothness_lambda": 0.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_freq": 10,
    "use_wandb": True,
    "wandb_project": "os-kinetics-tcn",
    "wandb_entity": None,
    "wandb_run_name": None,
}

DATASET_KEY_FIELDS: Tuple[str, ...] = (
    "train_dir",
    "window_size",
    "stride",
    "max_train_files",
    "max_val_files",
    "n_val_subjects",
    "n_test_subjects",
    "val_subjects",
    "test_subjects",
    "seed",
    "walking_only",
    "levelground_only",
    "balance_loc_buckets_oversample",
    "loc_bucket_balance_seed",
    "loc_ascent_descent_map",
    "lowpass_cutoff_hz",
    "lowpass_order",
    "rollout_decimate_step",
    "velocity_lowpass_filter",
    "velocity_lowpass_cutoff_hz",
    "velocity_lowpass_order",
    "input_mode",
    "output_mode",
    "laterality",
    "legacy_unilateral_full_window",
)

JOINT_SAGITTAL_MODES = frozenset(
    {
        "sagittal_hip_knee",
        "sagittal_hip_ankle",
        "sagittal_knee_ankle",
        "sagittal_hip_flexion",
        "sagittal_knee",
        "sagittal_ankle",
    }
)


@dataclass
class DatasetBundle:
    train_ds: KineticsTCNDataset
    val_ds: Optional[KineticsTCNDataset]
    subjects: List[str]
    train_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str]
    train_files: List[Any]
    val_files: List[Any]
    test_files_info: List[Any]
    is_h5_only_layout: bool


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_config_fragment(fragment: Any, *, root: Path) -> Dict[str, Any]:
    if isinstance(fragment, str):
        path = Path(fragment)
        if not path.is_absolute():
            path = root / path
        loaded = _load_json(path)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file {path} must contain a JSON object.")
        return dict(loaded)
    if isinstance(fragment, Mapping):
        return dict(fragment)
    raise TypeError(f"Expected config object or config path, got {type(fragment).__name__}.")


def load_experiment_configs(path: Path) -> List[Dict[str, Any]]:
    root = path.resolve().parent
    payload = _load_json(path)

    if isinstance(payload, list):
        base: Dict[str, Any] = {}
        raw_configs = payload
    elif isinstance(payload, dict):
        base = {}
        if "base_config" in payload:
            base.update(_load_config_fragment(payload["base_config"], root=root))
        if "base" in payload:
            base.update(_load_config_fragment(payload["base"], root=root))
        raw_configs = payload.get("configs", payload.get("experiments"))
        if raw_configs is None:
            raw_configs = [payload]
            base = {}
    else:
        raise ValueError("Multi-config JSON must be a list or an object.")

    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError("Multi-config JSON must contain a non-empty 'configs' list.")

    configs: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_configs, start=1):
        fragment = _load_config_fragment(raw, root=root)
        merged = dict(DEFAULT_CONFIG)
        merged.update(base)
        merged.update(fragment)

        name = fragment.get("name", merged.get("name"))
        if name and "output_dir" not in fragment:
            base_out = Path(str(base.get("output_dir", DEFAULT_CONFIG["output_dir"])))
            merged["output_dir"] = str(base_out.parent / str(name))
        if name and merged.get("wandb_run_name") is None:
            merged["wandb_run_name"] = str(name)
        merged.setdefault("name", f"experiment_{idx:02d}")
        configs.append(merged)

    output_dirs = [str(c["output_dir"]) for c in configs]
    duplicates = sorted({d for d in output_dirs if output_dirs.count(d) > 1})
    if duplicates:
        raise ValueError(f"Each experiment needs a unique output_dir; duplicates: {duplicates}")

    return configs


def make_args(config: Mapping[str, Any]) -> SimpleNamespace:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)
    if cfg.get("train_dir") is None:
        raise ValueError("Each config must provide train_dir, either directly or via base_config/base.")

    rollout_step = cfg.get("rollout_decimate_step")
    if rollout_step is None:
        rollout_step = 2 if bool(cfg.get("rollout")) else 1
    cfg["rollout_decimate_step"] = int(rollout_step)
    cfg["rollout"] = cfg["rollout_decimate_step"] > 1

    if cfg["input_mode"] in JOINT_SAGITTAL_MODES or cfg["output_mode"] in JOINT_SAGITTAL_MODES:
        if cfg["input_mode"] != cfg["output_mode"]:
            raise ValueError(
                "For sagittal pair/single-joint modes, input_mode and output_mode must match "
                f"(got {cfg['input_mode']!r} vs {cfg['output_mode']!r})."
            )

    return SimpleNamespace(**cfg)


def dataset_cache_key(args: SimpleNamespace) -> Tuple[Tuple[str, Any], ...]:
    key_items: List[Tuple[str, Any]] = []
    for field in DATASET_KEY_FIELDS:
        value = getattr(args, field)
        if isinstance(value, list):
            value = tuple(value)
        key_items.append((field, value))
    return tuple(key_items)


def _resolve_subject_split(args: SimpleNamespace) -> Tuple[
    List[str], List[str], List[str], List[str], List[Any], List[Any], List[Any], bool
]:
    set_global_seed(args.seed)
    train_root = Path(args.train_dir)
    h5_subject_files = sorted([p for p in train_root.glob("S*.h5") if p.is_file()])
    is_h5_only_layout = len(h5_subject_files) > 0

    if is_h5_only_layout:
        subjects = sorted([p.stem.upper() for p in h5_subject_files])
        subject_to_trials = None
    else:
        all_trials = find_trial_dirs(args.train_dir)
        subject_to_trials: Dict[str, Any] = {}
        for td in all_trials:
            sid = extract_subject_id(td)
            subject_to_trials.setdefault(sid, []).append(td)
        subjects = sorted(subject_to_trials.keys())

    n_total = len(subjects)
    if n_total < 4:
        raise ValueError(f"Need at least 4 subjects, found {n_total}.")

    if args.test_subjects:
        test_subjects = sorted([s.upper() for s in args.test_subjects])
    else:
        shuffled = subjects.copy()
        random.shuffle(shuffled)
        test_subjects = sorted(shuffled[: args.n_test_subjects])

    remaining = [s for s in subjects if s not in set(test_subjects)]

    if args.val_subjects:
        val_subjects = sorted([s.upper() for s in args.val_subjects])
    else:
        random.shuffle(remaining)
        val_subjects = sorted(remaining[: args.n_val_subjects])

    train_subjects = sorted([s for s in remaining if s not in set(val_subjects)])
    if len(train_subjects) == 0:
        raise ValueError("Split consumed all subjects.")

    if is_h5_only_layout:
        train_files = train_subjects
        val_files = val_subjects
        test_files_info = test_subjects
    else:
        assert subject_to_trials is not None
        train_files = [td for s in train_subjects for td in subject_to_trials[s]]
        val_files = [td for s in val_subjects for td in subject_to_trials[s]]
        test_files_info = [str(td) for s in test_subjects for td in subject_to_trials[s]]

    if args.max_train_files is not None:
        train_files = train_files[: args.max_train_files]
    if args.max_val_files is not None:
        val_files = val_files[: args.max_val_files]

    return (
        subjects,
        train_subjects,
        val_subjects,
        test_subjects,
        train_files,
        val_files,
        test_files_info,
        is_h5_only_layout,
    )


def build_dataset_bundle(args: SimpleNamespace) -> DatasetBundle:
    (
        subjects,
        train_subjects,
        val_subjects,
        test_subjects,
        train_files,
        val_files,
        test_files_info,
        is_h5_only_layout,
    ) = _resolve_subject_split(args)

    print("=" * 70)
    print("SUBJECT SPLIT")
    print("=" * 70)
    print(f"All subjects ({len(subjects)}): {subjects}")
    print(f"Train  ({len(train_subjects):2d}): {train_subjects}")
    print(f"Val    ({len(val_subjects):2d}): {val_subjects}")
    print(f"Test   ({len(test_subjects):2d}): {test_subjects}  <- never used during training")
    print(f"Train files: {len(train_files)}  |  Val files: {len(val_files)}")

    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)
    ds_denoise_kw = dict(
        apply_lowpass_filter=True,
        lowpass_cutoff_hz=args.lowpass_cutoff_hz,
        lowpass_order=args.lowpass_order,
        rollout_decimate_step=args.rollout_decimate_step,
        apply_velocity_lowpass_filter=args.velocity_lowpass_filter,
        velocity_lowpass_cutoff_hz=args.velocity_lowpass_cutoff_hz,
        velocity_lowpass_order=args.velocity_lowpass_order,
    )
    vel_cut = args.velocity_lowpass_cutoff_hz or args.lowpass_cutoff_hz
    vel_ord = args.velocity_lowpass_order or args.lowpass_order
    print(f"  Dataset denoise: zero-phase LPF ({args.lowpass_cutoff_hz} Hz, order {args.lowpass_order})")
    print(
        f"  Velocity LPF: {'on' if args.velocity_lowpass_filter else 'off'}"
        + (f" ({vel_cut} Hz, order {vel_ord})" if args.velocity_lowpass_filter else "")
    )
    if args.rollout_decimate_step > 1:
        print(
            f"  Rollout decimation: stride={args.rollout_decimate_step} "
            f"(native ~200 Hz -> ~{200.0 / args.rollout_decimate_step:.0f} Hz)"
        )

    pair_kw: Dict[str, Any] = {}
    if args.legacy_unilateral_full_window:
        pair_kw["unilateral_paired_side_windows"] = False

    loc_bal_kw: Dict[str, Any] = {}
    if args.balance_loc_buckets_oversample:
        loc_bal_kw = {
            "balance_loc_buckets_oversample": True,
            "loc_bucket_balance_seed": (
                args.loc_bucket_balance_seed if args.loc_bucket_balance_seed is not None else args.seed
            ),
            "loc_ascent_descent_map": args.loc_ascent_descent_map,
        }

    if is_h5_only_layout:
        train_ds = KineticsTCNDataset(
            data_dir=args.train_dir,
            h5_dir=args.train_dir,
            use_h5=True,
            subject_ids=train_files,
            window_size=args.window_size,
            stride=args.stride,
            walking_only=args.walking_only,
            levelground_only=args.levelground_only,
            normalize=False,
            input_mode=args.input_mode,
            output_mode=args.output_mode,
            laterality=args.laterality,
            max_files=args.max_train_files,
            **pair_kw,
            **ds_denoise_kw,
            **loc_bal_kw,
        )
    else:
        train_ds = KineticsTCNDataset(
            data_dir=args.train_dir,
            b3d_files=train_files,
            window_size=args.window_size,
            stride=args.stride,
            walking_only=args.walking_only,
            levelground_only=args.levelground_only,
            normalize=False,
            input_mode=args.input_mode,
            output_mode=args.output_mode,
            laterality=args.laterality,
            **pair_kw,
            **ds_denoise_kw,
            **loc_bal_kw,
        )

    val_ds = None
    if len(val_files) > 0:
        print("\n" + "=" * 70)
        print("LOADING VALIDATION DATA")
        print("=" * 70)
        if is_h5_only_layout:
            val_ds = KineticsTCNDataset(
                data_dir=args.train_dir,
                h5_dir=args.train_dir,
                use_h5=True,
                subject_ids=val_subjects,
                window_size=args.window_size,
                stride=1,
                walking_only=args.walking_only,
                levelground_only=args.levelground_only,
                normalize=False,
                stats=train_ds.get_stats(),
                input_mode=args.input_mode,
                output_mode=args.output_mode,
                laterality=args.laterality,
                max_files=args.max_val_files,
                **pair_kw,
                **ds_denoise_kw,
            )
        else:
            val_ds = KineticsTCNDataset(
                data_dir=args.train_dir,
                b3d_files=val_files,
                window_size=args.window_size,
                stride=1,
                walking_only=args.walking_only,
                levelground_only=args.levelground_only,
                normalize=False,
                stats=train_ds.get_stats(),
                input_mode=args.input_mode,
                output_mode=args.output_mode,
                laterality=args.laterality,
                **pair_kw,
                **ds_denoise_kw,
            )

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        subjects=subjects,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        train_files=train_files,
        val_files=val_files,
        test_files_info=test_files_info,
        is_h5_only_layout=is_h5_only_layout,
    )


def make_loaders(bundle: DatasetBundle, args: SimpleNamespace) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_loader = DataLoader(
        bundle.train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = None
    if bundle.val_ds is not None:
        val_loader = DataLoader(
            bundle.val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )
    return train_loader, val_loader


def build_model(args: SimpleNamespace, n_in: int, n_out: int) -> torch.nn.Module:
    if args.model_type == "transformer":
        if args.d_model % args.n_heads != 0:
            raise ValueError(f"--d-model ({args.d_model}) must be divisible by --n-heads ({args.n_heads}).")
        model = TransformerMoment(
            n_input_channels=n_in,
            n_output_channels=n_out,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        ).to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: TransformerMoment  |  params: {n_params:,}  |  in={n_in}  out={n_out}")
        print(
            f"  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}  "
            f"d_ff={args.d_ff}  dropout={args.dropout}"
        )
        print("  [bidirectional - not suitable for real-time streaming]")
        return model

    if args.model_type == "diffusion":
        if args.d_model % args.n_heads != 0:
            raise ValueError(f"--d-model ({args.d_model}) must be divisible by --n-heads ({args.n_heads}).")
        model = GaussianDiffusion1D(
            n_input_channels=n_in,
            n_output_channels=n_out,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            n_timesteps=args.n_diffusion_timesteps,
            schedule=args.diffusion_schedule,
            predict_epsilon=args.diffusion_predict_epsilon,
            n_inference_steps=args.n_inference_steps,
        ).to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: GaussianDiffusion1D  |  params: {n_params:,}  |  in={n_in}  out={n_out}")
        print(
            f"  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}  "
            f"d_ff={args.d_ff}  dropout={args.dropout}"
        )
        print(
            f"  timesteps={args.n_diffusion_timesteps}  schedule={args.diffusion_schedule}  "
            f"predict_epsilon={args.diffusion_predict_epsilon}  ddim_steps={args.n_inference_steps}"
        )
        print("  [offline - DDIM inference is not real-time suitable]")
        return model

    model = TCN(
        n_input_channels=n_in,
        n_output_channels=n_out,
        hidden_channels=args.hidden_channels,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    rf = model.receptive_field
    print(f"\nModel: TCN  |  params: {n_params:,}  |  in={n_in}  out={n_out}")
    print(
        f"  hidden={args.hidden_channels}  blocks={args.n_blocks}  "
        f"kernel={args.kernel_size}  dropout={args.dropout}"
    )
    print(f"  receptive_field={rf} samples  (window={args.window_size})")
    if rf > args.window_size:
        print(
            f"  WARNING: RF ({rf}) > window_size ({args.window_size}). "
            f"Deep blocks see only zero-padding during training."
        )
    return model


def run_experiment(args: SimpleNamespace, bundle: DatasetBundle, *, index: int, total: int) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)

    print("\n" + "#" * 80)
    print(f"EXPERIMENT {index}/{total}: {getattr(args, 'name', out_dir.name)}")
    print(f"Output: {out_dir}")
    print("#" * 80)

    _write_run_metadata_json(
        out_dir,
        args,
        subjects=bundle.subjects,
        train_subjects=bundle.train_subjects,
        val_subjects=bundle.val_subjects,
        test_subjects=bundle.test_subjects,
        train_files=bundle.train_files,
        val_files=bundle.val_files,
        test_files_info=bundle.test_files_info,
        unilateral_paired_side_windows=bundle.train_ds.unilateral_paired,
    )

    train_loader, val_loader = make_loaders(bundle, args)
    train_ds = bundle.train_ds
    val_ds = bundle.val_ds
    n_in = train_ds.n_input_channels
    n_out = train_ds.n_output_channels

    model = build_model(args, n_in, n_out)
    print(f"  device={args.device}")
    print(f"  input_mode={args.input_mode}  ({n_in} channels)")
    print(f"  output_mode={args.output_mode}  ({n_out} moments)")
    print(f"  Input DOFs:  {train_ds.input_dof_names}")
    print(f"  Output DOFs: {train_ds.output_dof_names}")
    print(f"  unilateral_paired_side_windows: {train_ds.unilateral_paired}")

    out_dof_names = train_ds.output_dof_names
    dof_weights_tensor: Optional[torch.Tensor] = None
    if args.dof_loss_weights is not None:
        if len(args.dof_loss_weights) != n_out:
            raise ValueError(
                f"--dof-loss-weights has {len(args.dof_loss_weights)} values but "
                f"n_out={n_out} ({out_dof_names}). Must match."
            )
        dof_weights_tensor = torch.tensor(args.dof_loss_weights, dtype=torch.float32)

    criterion = MomentLoss(
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        dof_weights=dof_weights_tensor,
    )

    print(f"\n  Loss: {args.loss_type.upper()}" + (f"  delta={args.huber_delta}" if args.loss_type == "huber" else ""))
    if dof_weights_tensor is not None:
        w_str = "  ".join(f"{n}:{w:.2f}" for n, w in zip(out_dof_names, args.dof_loss_weights))
        print(f"  DOF weights: {w_str}")
    if args.smoothness_lambda > 0:
        print(f"  Smoothness reg: lambda={args.smoothness_lambda}")
    if args.angle_jitter_std > 0:
        flag = "(correlated vel)" if args.correlated_vel_noise else "(angle-only, vel unchanged)"
        print(f"  Angle jitter: std={args.angle_jitter_std} rad  {flag}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    sample_rate_hz = 200.0 / float(args.rollout_decimate_step) if args.rollout_decimate_step > 1 else 200.0

    wandb_run = None
    if args.use_wandb:
        if not HAS_WANDB:
            print("wandb is not installed. Install with: pip install wandb")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
                reinit=True,
            )
            wandb.config.update(
                {
                    "train_subjects": bundle.train_subjects,
                    "val_subjects": bundle.val_subjects,
                    "test_subjects": bundle.test_subjects,
                    "n_train_files": len(bundle.train_files),
                    "n_val_files": len(bundle.val_files),
                    "n_train_windows": len(train_ds),
                    "n_val_windows": len(val_ds) if val_ds is not None else 0,
                    "multi_config_runner": True,
                },
                allow_val_change=True,
            )

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_r2_globals: List[float] = []
    best_val_loss = float("inf")
    best_val_r2 = float("nan")
    epochs_no_improve = 0
    last_epoch_idx = -1
    t0 = time.time()

    print(f"\n{'=' * 70}")
    print(
        f"TRAINING  |  epochs={args.epochs}  batch={args.batch_size}  "
        f"lr={args.lr}  window={args.window_size}"
    )
    print(f"{'=' * 70}")

    is_diffusion = isinstance(model, GaussianDiffusion1D)
    for epoch in range(args.epochs):
        last_epoch_idx = epoch
        ep_start = time.time()
        if is_diffusion:
            train_loss = train_one_epoch_diffusion(
                model,
                train_loader,
                optimizer,
                args.device,
                grad_clip=args.grad_clip,
                input_noise_std=args.input_noise_std,
            )
        else:
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                args.device,
                epoch,
                grad_clip=args.grad_clip,
                input_noise_std=args.input_noise_std,
                angle_jitter_std=args.angle_jitter_std,
                n_position_channels=train_ds.n_input_channels // 2,
                correlated_vel_noise=args.correlated_vel_noise,
                sample_rate_hz=sample_rate_hz,
                smoothness_lambda=args.smoothness_lambda,
            )
        train_losses.append(train_loss)

        log_parts = [f"Epoch {epoch + 1:3d}/{args.epochs}  train_loss={train_loss:.6f}"]

        val_loss = None
        r2_global = float("nan")
        per_ch_rmse = np.asarray([])
        per_ch_r2 = np.asarray([])
        if val_loader is not None:
            mse_crit = nn.MSELoss()
            val_loss, per_ch_rmse, r2_global, per_ch_r2 = evaluate(model, val_loader, mse_crit, args.device)
            val_losses.append(val_loss)
            val_r2_globals.append(r2_global)
            log_parts.append(f"val_mse={val_loss:.6f}")
            r2_str = f"{r2_global:.4f}" if np.isfinite(r2_global) else "nan"
            log_parts.append(f"val_R2={r2_str}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_r2 = r2_global
                epochs_no_improve = 0
                _save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_ds, args, out_dir / "best_model.pt", out_dof_names)
                log_parts.append("*best*")
            else:
                epochs_no_improve += 1

            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {epochs_no_improve} epochs).")
                last_epoch_idx = epoch
                scheduler.step()
                break

        scheduler.step()
        ep_time = time.time() - ep_start
        lr_now = optimizer.param_groups[0]["lr"]
        log_parts.append(f"lr={lr_now:.2e}  time={ep_time:.1f}s")
        print("  ".join(log_parts))

        if wandb_run is not None:
            log_dict: Dict[str, Any] = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/lr": lr_now,
                "train/epoch_time_sec": ep_time,
            }
            if val_loader is not None and val_loss is not None:
                log_dict["val/mse"] = val_loss
                if np.isfinite(r2_global):
                    log_dict["val/r2"] = float(r2_global)
                for i, rmse in enumerate(per_ch_rmse):
                    if i < len(out_dof_names):
                        log_dict[f"val/rmse/{out_dof_names[i]}"] = float(rmse)
                for i, r2c in enumerate(per_ch_r2):
                    if i < len(out_dof_names) and np.isfinite(r2c):
                        log_dict[f"val/r2/{out_dof_names[i]}"] = float(r2c)
            wandb.log(log_dict)

        if (epoch + 1) % args.save_freq == 0:
            _save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_losses[-1] if val_losses else None,
                train_ds,
                args,
                out_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                out_dof_names,
            )

    total_time = time.time() - t0
    _save_checkpoint(
        model,
        optimizer,
        max(last_epoch_idx, 0),
        train_losses[-1],
        val_losses[-1] if val_losses else None,
        train_ds,
        args,
        out_dir / "final_model.pt",
        out_dof_names,
    )

    plot_curves(train_losses, val_losses, out_dir / "training_curves.png")
    plot_ds = val_ds if val_ds is not None else train_ds
    plot_sample_prediction(model, plot_ds, args.device, out_dir / "sample_prediction.png", out_dof_names, sample_rate_hz=sample_rate_hz)

    if wandb_run is not None:
        for img_name in ("training_curves.png", "sample_prediction.png"):
            img_path = out_dir / img_name
            if img_path.exists():
                wandb.log({f"plots/{img_path.stem}": wandb.Image(str(img_path))})

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"  Final val MSE:    {val_losses[-1]:.6f}")
        print(f"  Best val MSE:     {best_val_loss:.6f}")
        if val_r2_globals:
            fr2_s = f"{val_r2_globals[-1]:.4f}" if np.isfinite(val_r2_globals[-1]) else "nan"
            br2_s = f"{best_val_r2:.4f}" if np.isfinite(best_val_r2) else "nan"
            print(f"  Final val R2:     {fr2_s}")
            print(f"  Best val R2:      {br2_s}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 70}")

    if wandb_run is not None:
        wandb.finish()
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sequential trainV2 experiments from a JSON config list while caching datasets."
    )
    parser.add_argument("config_json", type=str, help="JSON file with base_config/base and configs list.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later experiments if one experiment fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse configs and show dataset cache groups without loading data or training.",
    )
    return parser


def main() -> None:
    cli_args = build_argparser().parse_args()
    configs = load_experiment_configs(Path(cli_args.config_json))
    args_list = [make_args(cfg) for cfg in configs]

    print(f"Loaded {len(args_list)} experiment config(s) from {cli_args.config_json}")
    keys = [dataset_cache_key(args) for args in args_list]
    unique_key_count = len({key for key in keys})
    print(f"Dataset cache groups: {unique_key_count}")
    if unique_key_count > 1:
        print("  Some configs change data-affecting settings; those groups will be loaded separately.")

    if cli_args.dry_run:
        for i, args in enumerate(args_list, start=1):
            print(f"[{i}] name={getattr(args, 'name', None)} output_dir={args.output_dir}")
        return

    dataset_cache: Dict[Tuple[Tuple[str, Any], ...], DatasetBundle] = {}
    failures: List[Tuple[str, BaseException]] = []
    total = len(args_list)

    for index, args in enumerate(args_list, start=1):
        key = dataset_cache_key(args)
        name = str(getattr(args, "name", Path(args.output_dir).name))
        try:
            if key not in dataset_cache:
                print("\n" + "=" * 80)
                print(f"BUILDING DATASET CACHE GROUP for experiment {index}: {name}")
                print("=" * 80)
                dataset_cache[key] = build_dataset_bundle(args)
            else:
                print("\n" + "=" * 80)
                print(f"REUSING CACHED DATASET for experiment {index}: {name}")
                print("=" * 80)
            run_experiment(args, dataset_cache[key], index=index, total=total)
        except BaseException as exc:
            failures.append((name, exc))
            print(f"\nExperiment {name!r} failed: {exc}")
            if not cli_args.continue_on_error:
                raise

    if failures:
        print("\nFailures:")
        for name, exc in failures:
            print(f"  {name}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

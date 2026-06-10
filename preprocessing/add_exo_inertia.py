"""Step 2 — Add exoskeleton mass and scale inertia for exo-condition models.

For each *_exo.osim found in output/opensim/ (i.e. any exo model that does not
already have _added_inertia in its name), this step produces a sibling file
named {stem}_added_inertia.osim with the exo mass added to the relevant bodies
and diagonal inertia moments scaled proportionally.

Already-processed models (name contains _added_inertia) are skipped so the
step is safe to re-run.

Usage (from project root):
    python code/add_exo_inertia.py
    python code/add_exo_inertia.py --output-dir data/AB01_Jinwoo
    python code/add_exo_inertia.py --input path/to/model.osim --output path/to/out.osim
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import opensim as osim

sys.path.insert(0, str(Path(__file__).parent))
from utils import ADDED_MASS_BY_EXO_TYPE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "AB01_Jinwoo"


# ---------------------------------------------------------------------------
# Exo-type detection
# ---------------------------------------------------------------------------

def exo_type_from_name(stem: str) -> str | None:
    """Return 'hip_exo' or 'knee_exo' by inspecting the model filename, or None."""
    name = stem.lower()
    if "_hip_exo" in name:
        return "hip_exo"
    if "_knee_exo" in name:
        return "knee_exo"
    return None


def canonical_output_stem(stem: str, exo_type: str) -> str:
    """Return the canonical *_added_inertia stem for any input naming convention.

    Strips everything after the exo-type token so the output name is always
    ``{prefix}_{exo_type}_added_inertia`` regardless of suffixes like
    ``_before_inertia`` in the source file.

    Examples
    --------
    ab01_jinwoo_hip_exo               → ab01_jinwoo_hip_exo_added_inertia
    ab02_oscar_hip_exo_before_inertia → ab02_oscar_hip_exo_added_inertia
    """
    token = f"_{exo_type}"
    idx = stem.lower().find(token)
    base = stem[:idx + len(token)] if idx != -1 else stem
    return f"{base}_added_inertia"


# ---------------------------------------------------------------------------
# Core update logic
# ---------------------------------------------------------------------------

def _vec3_to_tuple(vec: osim.Vec3) -> tuple[float, float, float]:
    return float(vec.get(0)), float(vec.get(1)), float(vec.get(2))


def _update_body(body: osim.Body, added_mass: float) -> dict[str, float | str]:
    original_mass = float(body.getMass())
    if original_mass <= 0:
        raise ValueError(f"Body '{body.getName()}' has non-positive mass: {original_mass}")

    scale = (original_mass + added_mass) / original_mass
    inertia = body.getInertia()
    ixx, iyy, izz = _vec3_to_tuple(inertia.getMoments())
    ixy, ixz, iyz = _vec3_to_tuple(inertia.getProducts())

    body.setMass(original_mass + added_mass)
    body.setInertia(osim.Inertia(ixx * scale, iyy * scale, izz * scale, ixy, ixz, iyz))

    return {
        "body":           body.getName(),
        "original_mass":  original_mass,
        "added_mass":     added_mass,
        "new_mass":       original_mass + added_mass,
        "inertia_scale":  scale,
    }


def add_exo_mass_and_inertia(
    input_model: Path,
    output_model: Path,
    added_mass_by_body: dict[str, float] | None = None,
) -> list[dict[str, float | str]]:
    """Load *input_model*, add exo mass/inertia, save to *output_model*.

    If *added_mass_by_body* is not provided the correct table is selected
    automatically from the model filename via :func:`exo_type_from_name`.
    """
    if added_mass_by_body is None:
        exo_type = exo_type_from_name(input_model.stem)
        if exo_type is None:
            raise ValueError(
                f"Cannot determine exo type from '{input_model.name}'. "
                "Expected '_hip_exo' or '_knee_exo' in the filename, "
                "or pass added_mass_by_body explicitly."
            )
        added_mass_by_body = ADDED_MASS_BY_EXO_TYPE[exo_type]

    model = osim.Model(str(input_model))
    body_set = model.getBodySet()

    results = []
    for body_name, added_mass in added_mass_by_body.items():
        if not body_set.contains(body_name):
            raise KeyError(f"Body '{body_name}' not found in {input_model.name}")
        results.append(_update_body(body_set.get(body_name), added_mass))

    model.finalizeConnections()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.printToXML(str(output_model))
    return results


# ---------------------------------------------------------------------------
# Per-subject runner (used by pipeline)
# ---------------------------------------------------------------------------

def add_exo_inertia_for_output(output_dir: Path) -> None:
    """Process all base exo models in output_dir/opensim/ that need inertia added."""
    opensim_dir = output_dir / "opensim"
    if not opensim_dir.is_dir():
        print("  opensim/ not found — skipping.")
        return

    processed = 0
    for model_file in sorted(opensim_dir.glob("*.osim")):
        name = model_file.stem
        # Skip no-exo models and models already processed
        if "_no_exo" in name or "_added_inertia" in name:
            continue
        if "_exo" not in name:
            continue

        exo_type = exo_type_from_name(name)
        if exo_type is None:
            print(f"  [skip] {model_file.name}: cannot determine exo type.")
            continue

        output_file = opensim_dir / f"{canonical_output_stem(name, exo_type)}.osim"
        if output_file.exists():
            print(f"  [skip] {output_file.name} already exists.")
            continue

        print(f"\n  {model_file.name}  →  {output_file.name}  [{exo_type}]", flush=True)
        results = add_exo_mass_and_inertia(
            model_file, output_file, ADDED_MASS_BY_EXO_TYPE[exo_type]
        )
        for r in results:
            print(
                "    {body}: {original_mass:.4f} + {added_mass:.4f} = {new_mass:.4f} kg"
                "  (inertia ×{inertia_scale:.4f})".format(**r)
            )
        processed += 1

    if processed == 0:
        print("  No base exo models found to process.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add exo mass/inertia to scaled OpenSim models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Subject output dir (opensim/ sub-folder is used). Default: {DEFAULT_OUTPUT}")
    p.add_argument("--input",  type=Path, default=None,
                   help="Single input .osim model (overrides --output-dir scan).")
    p.add_argument("--output", type=Path, default=None,
                   help="Single output .osim model (required when --input is given).")
    p.add_argument("--exo-type", choices=list(ADDED_MASS_BY_EXO_TYPE), default=None,
                   help="Override exo type for --input mode (inferred from filename by default).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.input is not None:
        if args.output is None:
            raise SystemExit("--output is required when --input is given.")
        override = ADDED_MASS_BY_EXO_TYPE[args.exo_type] if args.exo_type else None
        results = add_exo_mass_and_inertia(args.input, args.output, override)
        print(f"Saved: {args.output}")
        for r in results:
            print(
                "  {body}: {original_mass:.6f} + {added_mass:.6f} = {new_mass:.6f} kg"
                "  (inertia ×{inertia_scale:.6f})".format(**r)
            )
    else:
        add_exo_inertia_for_output(args.output_dir)


if __name__ == "__main__":
    main()

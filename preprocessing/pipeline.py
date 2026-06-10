"""Full OpenSim processing pipeline for a cascade subject.

Pipeline steps (run in sequence):
  1. organise        — copy & rename source files into the output folder structure
  2. add_exo_inertia — add exo mass/inertia to scaled exo models in opensim/
  3. create_grf_xml  — generate ExternalLoads XML from force-plate MOT files
  4. run_ik          — run Inverse Kinematics on all marker TRC files
  5. run_id          — run Inverse Dynamics on all non-static IK results

Output structure
----------------
data/AB01_Jinwoo/
  opensim/                   scaled models (.osim) and scale sets (.xml)
  awinda/
    marker/                  .trc marker trajectories
    grf/                     .mot force-plate data  +  *_grf.xml
    mocap/                   .csv Vicon exports
    ik/                      *_ik.mot  (IK results)
    id/                      *_id.sto  (ID results)
  hip-exo/                   (same sub-structure)
  knee-exo/                  (same sub-structure)

Usage (from project root)
--------------------------
    # Full pipeline
    python code/pipeline.py

    # Custom source / output
    python code/pipeline.py --subject-root cascade/AB01_Jinwoo --output-dir data/AB01_Jinwoo

    # Organise + GRF XML only (no OpenSim needed)
    python code/pipeline.py --skip-exo-inertia --skip-ik --skip-id

    # Dry run (print planned copies, no files written)
    python code/pipeline.py --skip-exo-inertia --skip-ik --skip-id --dry-run

Note — OpenSim on Mac Apple Silicon
-------------------------------------
OpenSim 4.5 Python bindings are compiled for x86_64.  Use:

    PYTHONPATH="/Applications/OpenSim 4.5/OpenSim 4.5.app/Contents/Resources/opensim/sdk/Python" \\
      arch -x86_64 /path/to/opensim-x86/bin/python3.12 \\
      code/pipeline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "cascade" / "AB01_Jinwoo"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "AB01_Jinwoo"

_SEP = "=" * 60


def _header(title: str) -> None:
    print(f"\n{_SEP}\n{title}\n{_SEP}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full OpenSim processing pipeline for a cascade subject.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--subject-root", type=Path, default=DEFAULT_SOURCE,
                   help=f"Source directory.  Default: {DEFAULT_SOURCE}")
    p.add_argument("--output-dir",   type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output directory. Default: {DEFAULT_OUTPUT}")
    p.add_argument("--skip-organise",    action="store_true", help="Skip step 1.")
    p.add_argument("--skip-exo-inertia", action="store_true", help="Skip step 2.")
    p.add_argument("--skip-grf-xml",    action="store_true", help="Skip step 3.")
    p.add_argument("--skip-ik",         action="store_true", help="Skip step 4.")
    p.add_argument("--skip-id",         action="store_true", help="Skip step 5.")
    p.add_argument("--dry-run",         action="store_true",
                   help="Step 1 only: print planned copies without writing files.")
    p.add_argument("--type",  dest="type_filter",
                   help="Restrict IK/ID to one trial-type folder, e.g. hip-exo.")
    p.add_argument("--trial", dest="trial_filter",
                   help="Restrict IK/ID to trials whose name contains this string.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.skip_organise:
        _header("STEP 1 — Organise source files")
        from organise import organise
        organise(args.subject_root, args.output_dir, args.dry_run)

    if not args.skip_exo_inertia:
        _header("STEP 2 — Add exoskeleton mass & inertia")
        from add_exo_inertia import add_exo_inertia_for_output
        add_exo_inertia_for_output(args.output_dir)

    if not args.skip_grf_xml:
        _header("STEP 3 — Create GRF XML files")
        from create_grf_xml import create_grf_xmls
        created = create_grf_xmls(args.output_dir)
        print(f"\nCreated {len(created)} GRF XML file(s).")

    if not args.skip_ik:
        _header("STEP 4 — Inverse Kinematics")
        from run_ik import run_ik_for_output
        run_ik_for_output(args.output_dir, args.type_filter, args.trial_filter)

    if not args.skip_id:
        _header("STEP 5 — Inverse Dynamics")
        from run_id import run_id_for_output
        run_id_for_output(args.output_dir, args.type_filter, args.trial_filter)

    _header(f"Pipeline complete  →  {args.output_dir}")


if __name__ == "__main__":
    main()

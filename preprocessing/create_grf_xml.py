"""Step 2 — Generate OpenSim ExternalLoads XML files from force-plate MOT files.

One XML is written per .mot file found in each trial-type's grf/ sub-folder.
Trials whose name contains the token 'rd' have their force plates swapped
(plate 2 → right foot, plate 1 → left foot), matching the example pipeline.

Usage (from project root):
    python code/create_grf_xml.py
    python code/create_grf_xml.py --output-dir data/AB01_Jinwoo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from utils import FOLDER_TYPE_TO_OUTPUT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "AB01_Jinwoo"


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def read_storage_name(mot_file: Path) -> str:
    """Return the storage name from the first line of a MOT file."""
    with mot_file.open("r", encoding="utf-8") as fh:
        return fh.readline().strip() or mot_file.stem


def should_swap_plates(trial_name: str) -> bool:
    """Return True for ramp-descending (rd) trials where plates are swapped."""
    tokens = trial_name.lower().replace("-", "_").split("_")
    return "rd" in tokens


def _make_external_force(
    name: str,
    applied_to_body: str,
    force_id: str,
    point_id: str,
    data_source: str,
) -> ET.Element:
    el = ET.Element("ExternalForce", {"name": name})
    ET.SubElement(el, "applied_to_body").text = applied_to_body
    ET.SubElement(el, "force_expressed_in_body").text = "ground"
    ET.SubElement(el, "point_expressed_in_body").text = "ground"
    ET.SubElement(el, "force_identifier").text = force_id
    ET.SubElement(el, "point_identifier").text = point_id
    ET.SubElement(el, "torque_identifier").text = ""
    ET.SubElement(el, "data_source_name").text = data_source
    return el


def build_grf_xml(mot_file: Path) -> ET.ElementTree:
    """Build an ExternalLoads XML tree for *mot_file*."""
    data_source = read_storage_name(mot_file)
    swap = should_swap_plates(mot_file.stem)

    right_plate = "ground_force2" if swap else "ground_force1"
    left_plate  = "ground_force1" if swap else "ground_force2"

    doc = ET.Element("OpenSimDocument", {"Version": "40500"})
    ext_loads = ET.SubElement(doc, "ExternalLoads", {"name": "externalloads"})
    objects = ET.SubElement(ext_loads, "objects")

    objects.append(_make_external_force(
        "grf_r", "calcn_r",
        f"{right_plate}_v", f"{right_plate}_p", data_source,
    ))
    objects.append(_make_external_force(
        "grf_l", "calcn_l",
        f"{left_plate}_v", f"{left_plate}_p", data_source,
    ))

    ET.SubElement(ext_loads, "groups")
    ET.SubElement(ext_loads, "datafile").text = str(mot_file.resolve())
    return ET.ElementTree(doc)


def _write_pretty(tree: ET.ElementTree, path: Path) -> None:
    raw = ET.tostring(tree.getroot(), encoding="utf-8")
    path.write_bytes(minidom.parseString(raw).toprettyxml(indent="\t", encoding="UTF-8"))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def create_grf_xmls(output_dir: Path) -> list[Path]:
    """Create one GRF XML per MOT file in each trial-type grf/ directory."""
    created: list[Path] = []

    for output_name in FOLDER_TYPE_TO_OUTPUT.values():
        grf_dir = output_dir / output_name / "grf"
        if not grf_dir.is_dir():
            continue

        for mot_file in sorted(grf_dir.glob("*.mot")):
            xml_path = grf_dir / f"{mot_file.stem}_grf.xml"
            _write_pretty(build_grf_xml(mot_file), xml_path)
            print(f"  created {xml_path.relative_to(output_dir)}")
            created.append(xml_path)

    return created


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create OpenSim ExternalLoads XML files from force-plate MOT files."
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output directory. Default: {DEFAULT_OUTPUT}")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    created = create_grf_xmls(args.output_dir)
    print(f"\nCreated {len(created)} GRF XML file(s).")


if __name__ == "__main__":
    main()

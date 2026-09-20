"""Command line entry point.

`build` validates a scenario, `schema` prints the JSON schema, `assets` lists the
meshes with the credit they carry.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

from pydantic import ValidationError

from seascape.assets import fetch, manifest
from seascape.config import Scenario, load


def _build(scenario_path: Path) -> None:
    scenario = load(scenario_path)
    cameras = scenario.rig.cameras
    kinds = ", ".join(sorted({camera.kind for camera in cameras}))
    print(f"{scenario_path}: valid")
    print(f"  rig     {len(cameras)} cameras ({kinds}) at {scenario.rig.height_m} m")
    print("  scene build needs Blender; not implemented yet")


def _assets(download: bool) -> None:
    for name, asset in manifest().items():
        print(f"{name}  {asset.licence}  {asset.attribution}  {asset.page}")
        if download:
            print(f"  {fetch(name)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seascape")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="validate a scenario and summarise it")
    build.add_argument("scenario", type=Path)

    commands.add_parser("schema", help="print the scenario JSON schema on stdout")

    listing = commands.add_parser("assets", help="list the meshes and their credit")
    listing.add_argument("--fetch", action="store_true", help="download any not cached")

    args = parser.parse_args(argv)
    if args.command == "schema":
        print(json.dumps(Scenario.model_json_schema(), indent=2))
        return 0
    try:
        if args.command == "assets":
            _assets(args.fetch)
        else:
            _build(args.scenario)
    except (ValidationError, OSError, ValueError, TypeError, tomllib.TOMLDecodeError):
        # A scenario or manifest mistake is the user's, not a crash; a traceback
        # buries the line that says which.
        print(sys.exception(), file=sys.stderr)
        return 1
    return 0

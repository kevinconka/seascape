"""Command line entry point.

`build` validates a scenario and summarises it; `schema` prints the JSON schema.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

from pydantic import ValidationError

from seascape.config import Scenario, load


def _build(scenario_path: Path) -> None:
    scenario = load(scenario_path)
    cameras = scenario.rig.cameras
    kinds = ", ".join(sorted({camera.kind for camera in cameras}))
    print(f"{scenario_path}: valid")
    print(f"  rig     {len(cameras)} cameras ({kinds}) at {scenario.rig.height_m} m")
    print("  scene build needs Blender; not implemented yet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seascape")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="validate a scenario and summarise it")
    build.add_argument("scenario", type=Path)

    commands.add_parser("schema", help="print the scenario JSON schema on stdout")

    args = parser.parse_args(argv)
    if args.command == "schema":
        print(json.dumps(Scenario.model_json_schema(), indent=2))
        return 0
    try:
        _build(args.scenario)
    except (ValidationError, OSError, ValueError, TypeError, tomllib.TOMLDecodeError):
        # A scenario mistake is the user's, not a crash; a traceback buries the line.
        print(sys.exception(), file=sys.stderr)
        return 1
    return 0

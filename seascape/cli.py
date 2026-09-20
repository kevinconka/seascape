"""Command line entry point.

`build` writes a .blend from a scenario; `schema` prints the JSON schema.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

from pydantic import ValidationError

from seascape.config import Scenario, load


def _build(scenario_path: Path, output: Path | None) -> None:
    scenario = load(scenario_path)
    # Imported here, not at module scope: bpy is a 400 MB library and `schema` and a
    # failed validation should not wait for it.
    import bpy

    from seascape import scene

    scene.build(scenario)
    path = output or scenario_path.with_suffix(".blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(path.resolve()))
    cameras = scenario.rig.cameras
    kinds = ", ".join(sorted({camera.kind for camera in cameras}))
    print(f"{path}: {len(cameras)} cameras ({kinds}) at {scenario.rig.height_m} m")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seascape")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="write a .blend from a scenario")
    build.add_argument("scenario", type=Path)
    build.add_argument("-o", "--output", type=Path, help="default: alongside the input")

    commands.add_parser("schema", help="print the scenario JSON schema on stdout")

    args = parser.parse_args(argv)
    if args.command == "schema":
        print(json.dumps(Scenario.model_json_schema(), indent=2))
        return 0
    try:
        _build(args.scenario, args.output)
    except (ValidationError, OSError, ValueError, TypeError, tomllib.TOMLDecodeError):
        # A scenario mistake is the user's, not a crash; a traceback buries the line.
        print(sys.exception(), file=sys.stderr)
        return 1
    return 0

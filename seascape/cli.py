"""Command line entry point.

`build` writes a .blend, `render` writes the images, `schema` prints the JSON schema.
"""

import argparse
import json
import sys
from pathlib import Path

from seascape.config import Band, Scenario, load


def _build(scenario_path: Path, output: Path | None, band: Band) -> None:
    scenario = load(scenario_path)
    # Imported here, not at module scope: bpy is a 400 MB library and `schema` and a
    # failed validation should not wait for it.
    import bpy

    from seascape import scene

    scene.build(scenario, band)
    path = output or scenario_path.with_suffix(f".{band}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(path.resolve()))
    cameras = scenario.rig.cameras
    kinds = ", ".join(sorted({camera.kind for camera in cameras}))
    print(
        f"{path}: {band}, {len(cameras)} cameras ({kinds}) at {scenario.rig.height_m} m"
    )


def _render(scenario_path: Path, output: Path | None) -> None:
    scenario = load(scenario_path)
    from seascape import render

    into = output or scenario_path.with_suffix("")
    written = render.render(scenario, into)
    for path in written:
        print(path)
    print(f"{len(written)} images in {into}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seascape")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="write a .blend from a scenario")
    build.add_argument("scenario", type=Path)
    build.add_argument("-o", "--output", type=Path, help="default: alongside the input")
    # A scene is one band or the other: EO and LWIR share no units.
    build.add_argument("--band", choices=("eo", "ir"), default="eo")

    shoot = commands.add_parser("render", help="write one image per camera")
    shoot.add_argument("scenario", type=Path)
    shoot.add_argument("-o", "--output", type=Path, help="default: alongside the input")

    commands.add_parser("schema", help="print the scenario JSON schema on stdout")

    args = parser.parse_args(argv)
    if args.command == "schema":
        print(json.dumps(Scenario.model_json_schema(), indent=2))
        return 0
    try:
        if args.command == "render":
            _render(args.scenario, args.output)
        else:
            _build(args.scenario, args.output, args.band)
    except (
        OSError,
        ValueError,
        TypeError,
    ):  # pydantic and tomllib both raise ValueError
        # A scenario mistake is the user's, not a crash; a traceback buries the line.
        print(sys.exception(), file=sys.stderr)
        return 1
    return 0

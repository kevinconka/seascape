"""Command line entry point.

`build` and `render` need Blender and land with the scene; until then `build`
validates the scenario and says what it would have built.
"""

import argparse
import json
from pathlib import Path

from seascape.config import Scenario, load


def _build(scenario_path: Path) -> None:
    scenario = load(scenario_path)
    cameras = scenario.rig.cameras
    kinds = ", ".join(sorted({c.kind for c in cameras}))
    print(f"{scenario_path}: valid")
    print(f"  rig     {len(cameras)} cameras ({kinds}) at {scenario.rig.height_m} m")
    print(f"  sea     {scenario.sea.t_sea_k} K, {scenario.sea.wind_speed_mps} m/s wind")
    print(f"  objects {len(scenario.objects)}")
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
    else:
        _build(args.scenario)
    return 0

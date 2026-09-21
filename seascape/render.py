"""Render the cameras a scenario asks for: one EXR each, per band.

Nothing here decides what a pixel looks like. The scene already sets the LWIR view
transform, because those pixels are radiance rather than a picture, and turning them
into something viewable needs the sensor's gain curve.

Beside the images go the ground truth, which `truth` computes without Blender, and a
provenance record. A render nobody can reproduce or attribute is not evidence.
"""

import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import bpy
from pydantic import BaseModel

from seascape import ground_truth, scene
from seascape.config import Engine, Scenario

# The scenario names engines in lower case because Blender's identifiers move between
# versions; `_engine` resolves one and checks it against this build.
_ENGINES: dict[Engine, str] = {"cycles": "CYCLES", "eevee": "BLENDER_EEVEE"}


def _set_engine(name: Engine) -> None:
    """Assign the engine and let Blender validate it.

    Cycles registers itself as an add-on and never appears in the engine enum this
    build reports, so checking that enum first rejects the default. Assignment is the
    real check: an identifier Blender does not know raises TypeError.
    """
    identifier = _ENGINES[name]
    try:
        bpy.context.scene.render.engine = identifier
    except TypeError as error:
        raise ValueError(
            f"Blender {bpy.app.version_string} rejected engine {identifier}"
        ) from error


def _settings(scenario: Scenario) -> None:
    outputs = scenario.outputs
    render = bpy.context.scene.render
    _set_engine(outputs.engine)
    if outputs.engine == "cycles":
        bpy.context.scene.cycles.samples = outputs.samples
    else:
        bpy.context.scene.eevee.taa_render_samples = outputs.samples
    render.image_settings.file_format = "OPEN_EXR"
    # Full float, not half: these are radiance in W m^-2 sr^-1 and a half's 11-bit
    # mantissa is a lossy step nobody would expect in a file meant to be defensible.
    render.image_settings.color_depth = "32"


class Provenance(BaseModel):
    """Enough to reproduce a render, or to know why it cannot be reproduced."""

    rendered_at: datetime
    seascape_version: str
    git_sha: str | None
    blender_version: str
    blender_build_hash: str
    engine: str
    device: str


def _git_sha() -> str | None:
    """The checkout this ran from, or None when it ran from an installed wheel."""
    try:
        finished = subprocess.run(
            ("git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return finished.stdout.strip() or None


def provenance(scenario: Scenario) -> Provenance:
    cycles = bpy.context.scene.cycles
    return Provenance(
        rendered_at=datetime.now(UTC),
        seascape_version=version("seascape"),
        git_sha=_git_sha(),
        blender_version=bpy.app.version_string,
        blender_build_hash=bpy.app.build_hash.decode(),
        engine=scenario.outputs.engine,
        # Cycles reports CPU until a compute device is configured; on a machine with
        # none this is the whole story, and on one with a GPU it says which was used.
        device=f"{cycles.device}:{bpy.context.preferences.addons['cycles'].preferences.compute_device_type}",
    )


def render(scenario: Scenario, into: Path) -> list[Path]:
    """Write one EXR per camera into `into`, building each band's scene once."""
    into.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for band in scenario.outputs.bands:
        scene.build(scenario, band)
        _settings(scenario)
        sc = bpy.context.scene
        for spec in (c for c in scenario.rig.cameras if c.kind == band):
            name = scene.camera_name(spec)
            sc.camera = bpy.data.objects[name]
            sc.render.resolution_x, sc.render.resolution_y = (
                spec.width_px,
                spec.height_px,
            )
            sc.render.filepath = str(into / name)
            bpy.ops.render.render(write_still=True)
            written.append(into / f"{name}.exr")

    for name, record in (
        ("ground_truth.json", ground_truth.ground_truth(scenario)),
        ("provenance.json", provenance(scenario)),
    ):
        path = into / name
        path.write_text(record.model_dump_json(indent=2))
        written.append(path)
    return written

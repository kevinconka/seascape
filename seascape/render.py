"""Render the cameras a scenario asks for: one image each, per band.

Nothing here decides what a pixel looks like. The scene already sets the LWIR view
transform, because those pixels are radiance rather than a picture, and turning them
into something viewable needs the sensor's gain curve.
"""

from pathlib import Path

import bpy

from seascape import scene
from seascape.config import Band, Engine, ImageFormat, Scenario

# The scenario names engines in lower case because Blender's identifiers move between
# versions; `_set_engine` resolves one and lets Blender reject what it does not know.
_ENGINES: dict[Engine, str] = {"cycles": "CYCLES", "eevee": "BLENDER_EEVEE"}

# Blender's format identifier and the bit depth that goes with it. Full float for EXR,
# not half: a half's 11-bit mantissa is a lossy step nobody would expect in a file
# meant to be defensible.
_FORMATS: dict[ImageFormat, tuple[str, str]] = {
    "exr": ("OPEN_EXR", "32"),
    "png": ("PNG", "8"),
}


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


def _settings(scenario: Scenario, band: Band) -> None:
    outputs = scenario.outputs
    render = bpy.context.scene.render
    _set_engine(outputs.engine)
    if band == "eo":
        # LWIR keeps the exposure `build` pinned: those pixels are radiance, and any
        # gain on them belongs to the sensor model.
        bpy.context.scene.view_settings.exposure = outputs.exposure_ev
    if outputs.engine == "cycles":
        bpy.context.scene.cycles.samples = outputs.samples
    else:
        bpy.context.scene.eevee.taa_render_samples = outputs.samples
    file_format, depth = _FORMATS[outputs.format]
    render.image_settings.file_format = file_format
    render.image_settings.color_depth = depth


def render(scenario: Scenario, into: Path) -> list[Path]:
    """Write one EXR per camera into `into`, building each band's scene once."""
    into.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for band in scenario.outputs.bands:
        scene.build(scenario, band)
        _settings(scenario, band)
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
            written.append(into / f"{name}.{scenario.outputs.format}")
    return written

"""Render the cameras a scenario asks for: one image each, per band.

Nothing here decides what a pixel looks like. The scene already sets the LWIR view
transform, because those pixels are radiance rather than a picture, and turning them
into something viewable needs the sensor's gain curve.
"""

from pathlib import Path

import bpy

from seascape import scene
from seascape.config import Band, ImageFormat, Scenario

# Blender's format identifier and the bit depth that goes with it. Full float for EXR,
# not half: a half's 11-bit mantissa is a lossy step nobody would expect in a file
# meant to be defensible.
_FORMATS: dict[ImageFormat, tuple[str, str]] = {
    "exr": ("OPEN_EXR", "32"),
    "png": ("PNG", "8"),
}


def _settings(scenario: Scenario, band: Band) -> None:
    outputs = scenario.outputs
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = outputs.samples
    if band == "eo":
        # LWIR keeps the exposure `build` pinned: those pixels are radiance, and any
        # gain on them belongs to the sensor model.
        sc.view_settings.exposure = outputs.exposure_ev
    file_format, depth = _FORMATS[outputs.format]
    sc.render.image_settings.file_format = file_format
    sc.render.image_settings.color_depth = depth


def render(scenario: Scenario, into: Path) -> list[Path]:
    """Write one image per camera into `into`, building each band's scene once."""
    into.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for band in scenario.outputs.bands:
        # An EO-only rig is legitimate, and the default bands ask for both. Selecting
        # first means such a rig renders its EO cameras instead of raising on IR.
        specs = [c for c in scenario.rig.cameras if c.kind == band]
        if not specs:
            continue
        scene.build(scenario, band)
        _settings(scenario, band)
        sc = bpy.context.scene
        for spec in specs:
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

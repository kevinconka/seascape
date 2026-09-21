"""Render the cameras a scenario asks for: one EXR each, per band.

Nothing here decides what a pixel looks like. An EXR is float, so an LWIR pixel is
the radiance the render produced; mapping either band onto 8 bits is a display
decision and does not belong in the renderer.
"""

from pathlib import Path

import bpy

from seascape import scene
from seascape.config import Band, Scenario


def _settings(scenario: Scenario, band: Band) -> None:
    """Engine, sampling and output format for one band's scene."""
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = scenario.outputs.samples
    # OIDN is an edge-aware image filter, not a radiometric one, and it is on by
    # default. On a world flat at 290.00 K it returns 282.43-293.00 K and breaks the
    # R=G=B an LWIR scene guarantees. EO is a picture and keeps it.
    sc.cycles.use_denoising = band == "eo"
    sc.render.image_settings.file_format = "OPEN_EXR"
    # 32-bit, not half: an 11-bit mantissa loses radiance.
    sc.render.image_settings.color_depth = "32"


def render(scenario: Scenario, into: Path) -> list[Path]:
    """Write one EXR per camera into `into`, building each band's scene once."""
    into.mkdir(parents=True, exist_ok=True)
    outputs = scenario.outputs
    written: list[Path] = []
    for band in outputs.bands:
        # The default bands ask for both, so an EO-only rig must skip ir, not fail.
        specs = [c for c in scenario.rig.cameras if c.kind == band]
        if not specs:
            continue
        scene.build(scenario, band)
        _settings(scenario, band)
        sc = bpy.context.scene
        for spec in specs:
            sc.camera = bpy.data.objects[spec.name]
            sc.render.resolution_x, sc.render.resolution_y = (
                spec.width_px,
                spec.height_px,
            )
            sc.render.filepath = str(into / spec.name)
            bpy.ops.render.render(write_still=True)
            written.append(into / f"{spec.name}.exr")
    if not written:
        # Skipping a band the default asked for is right; writing nothing at all
        # means the scenario names only bands its rig has no camera for.
        raise ValueError(
            f"the rig has no camera in any of {outputs.bands}: nothing to render"
        )
    return written

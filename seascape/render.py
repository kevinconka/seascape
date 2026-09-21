"""Render the cameras a scenario asks for: one image each, per band.

EO reaches 8 bits through the exposure and Blender's film curve. LWIR cannot: its
pixels are radiance in W m^-2 sr^-1, which Blender would clip to white, so an ir png
is rendered float and mapped here through the scenario's temperature window.
"""

from pathlib import Path

import bpy
import numpy as np

from seascape import lwir, scene
from seascape.config import Band, ImageFormat, Scenario

# Blender's format identifier and the bit depth that goes with it. Full float for EXR,
# not half: a half's 11-bit mantissa is a lossy step nobody would expect in a file
# meant to be defensible.
_FORMATS: dict[ImageFormat, tuple[str, str]] = {
    "exr": ("OPEN_EXR", "32"),
    "png": ("PNG", "8"),
}


def _settings(scenario: Scenario, band: Band, writing: ImageFormat) -> None:
    outputs = scenario.outputs
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = outputs.samples
    if band == "eo":
        # LWIR keeps the exposure `build` pinned: those pixels are radiance, and any
        # gain on them belongs to the display mapping, not to the render.
        sc.view_settings.exposure = outputs.exposure_ev
    file_format, depth = _FORMATS[writing]
    sc.render.image_settings.file_format = file_format
    sc.render.image_settings.color_depth = depth


def _thermal_png(exr: Path, png: Path, window_k: tuple[float, float]) -> None:
    """Rewrite a float LWIR render as 8-bit grey, linear in brightness temperature.

    Black is the low end of the window and white the high end, so a pixel value is a
    temperature and the same value means the same thing in every frame.
    """
    source = bpy.data.images.load(str(exr))
    width, height = source.size
    radiance = np.asarray(source.pixels[:], dtype=np.float32).reshape(-1, 4)[:, 0]
    low, high = window_k
    t_k = lwir.brightness_temperature(radiance)
    grey = np.clip((t_k - low) / (high - low), 0.0, 1.0)

    out = bpy.data.images.new(png.stem, width, height)
    # Non-Color, so the values written are the mapping above and not sRGB-encoded.
    out.colorspace_settings.name = "Non-Color"
    out.pixels = np.column_stack([grey, grey, grey, np.ones_like(grey)]).ravel()
    out.file_format = "PNG"
    out.filepath_raw = str(png)
    out.save()

    bpy.data.images.remove(source)
    bpy.data.images.remove(out)
    exr.unlink()


def render(scenario: Scenario, into: Path) -> list[Path]:
    """Write one image per camera into `into`, building each band's scene once."""
    into.mkdir(parents=True, exist_ok=True)
    outputs = scenario.outputs
    written: list[Path] = []
    for band in outputs.bands:
        # An EO-only rig is legitimate, and the default bands ask for both. Selecting
        # first means such a rig renders its EO cameras instead of raising on IR.
        specs = [c for c in scenario.rig.cameras if c.kind == band]
        if not specs:
            continue
        thermal_png = band == "ir" and outputs.format == "png"
        scene.build(scenario, band)
        _settings(scenario, band, "exr" if thermal_png else outputs.format)
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
            image = into / f"{name}.{outputs.format}"
            if thermal_png:
                _thermal_png(into / f"{name}.exr", image, outputs.ir_window_k)
            written.append(image)
    return written

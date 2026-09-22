"""Render the cameras a scenario asks for: one image each, per band.

EO reaches 8 bits through the exposure and Blender's film curve. LWIR cannot: its
pixels are radiance in W m^-2 sr^-1, which Blender would clip to white, so an ir png
is rendered float and mapped here through the scenario's temperature window.
"""

from pathlib import Path

import bpy
import numpy as np

from seascape import lwir, scene
from seascape.calibration import Calibration, CameraCalibration
from seascape.config import Scenario


def _thermal_png(exr: Path, png: Path) -> None:
    """Rewrite a float LWIR render as 8-bit grey, auto-contrasted over the frame.

    Black is the coldest pixel and white the hottest, so the frame uses the whole
    range whatever the scene. The cost is that the scale is the frame's own: two
    images are not comparable and a pixel is not a temperature. The exr beside it is
    where both of those live.
    """
    source = bpy.data.images.load(str(exr))
    width, height = source.size
    out = bpy.data.images.new(png.stem, width, height)
    try:
        buffer = np.empty(width * height * 4, dtype=np.float32)
        source.pixels.foreach_get(buffer)
        t_k = lwir.brightness_temperature(buffer.reshape(-1, 4)[:, 0])
        # Full span, not a percentile: a target is a small fraction of the frame and
        # trimming the tails is what flattens it to white. A render has no dead
        # pixels; a real sensor would need the tails trimmed here.
        low, high = float(t_k.min()), float(t_k.max())
        # A frame of one temperature has no contrast to stretch; mid-grey, not NaN.
        if high - low < 1e-6:
            low, high = low - 0.5, low + 0.5
        # float32: foreach_set takes the buffer's type literally and rejects a double.
        grey = np.clip((t_k - low) / (high - low), 0.0, 1.0).astype(np.float32)

        # Before the pixels, never after: assigning the colorspace second re-reads
        # what is already there and leaves the image black, with no error.
        out.colorspace_settings.name = "Non-Color"
        out.pixels.foreach_set(
            np.column_stack([grey, grey, grey, np.ones_like(grey)]).ravel()
        )
        out.file_format = "PNG"
        out.filepath_raw = str(png)
        out.save()
    finally:
        # Datablocks go whatever happened; the float render survives a failure, so a
        # conversion that raised can be retried without paying for the render again.
        bpy.data.images.remove(source)
        bpy.data.images.remove(out)
    exr.unlink()


def render(scenario: Scenario, into: Path) -> list[Path]:
    """Write one image per camera into `into`, and their calibration beside them."""
    into.mkdir(parents=True, exist_ok=True)
    outputs = scenario.outputs
    written: list[Path] = []
    cameras: list[CameraCalibration] = []
    for band in outputs.bands:
        # The default bands ask for both, so an EO-only rig must skip ir, not fail.
        mounts = [m for m in scenario.rig.mounts if m.camera.kind == band]
        if not mounts:
            continue
        thermal_png = band == "ir" and outputs.format == "png"
        built = scene.build(scenario, band)
        sc = bpy.context.scene
        for mount in mounts:
            sc.camera = built.cameras[mount.name]
            sc.render.resolution_x, sc.render.resolution_y = (
                mount.camera.width_px,
                mount.camera.height_px,
            )
            sc.render.filepath = str(into / mount.name)
            bpy.ops.render.render(write_still=True)
            image = into / f"{mount.name}.{outputs.format}"
            if thermal_png:
                _thermal_png(into / f"{mount.name}.exr", image)
            written.append(image)
            cameras.append(scene.calibrate(built, mount, image.name))
    if not written:
        # Skipping a band the default asked for is right; writing nothing at all
        # means the scenario names only bands its rig has no camera for.
        raise ValueError(
            f"the rig has no camera in any of {outputs.bands}: nothing to render"
        )
    written.append(Calibration(cameras=cameras).write(into))
    return written

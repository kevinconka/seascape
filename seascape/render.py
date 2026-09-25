"""Render the cameras a scenario asks for: one image each, per band.

EO reaches 8 bits through the exposure and Blender's film curve. LWIR cannot: its
pixels are radiance in W m^-2 sr^-1, which Blender would clip to white, so an ir png
is rendered float and mapped here through the scenario's temperature window.
"""

import subprocess
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import bpy
import numpy as np

from seascape import labels, lwir, scene
from seascape.calibration import Calibration, CameraCalibration
from seascape.config import Scenario


def _pixels(path: Path) -> np.ndarray:
    """An image's RGBA, (height, width, 4), bottom row first as Blender stores it."""
    image = bpy.data.images.load(str(path))
    try:
        width, height = image.size
        buffer = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(buffer)
    finally:
        bpy.data.images.remove(image)
    return buffer.reshape(height, width, 4)


def _thermal_png(exr: Path, png: Path) -> None:
    """Rewrite a float LWIR render as 8-bit grey, auto-contrasted over the frame.

    Black is the coldest pixel and white the hottest, so the frame uses the whole
    range whatever the scene. The cost is that the scale is the frame's own: two
    images are not comparable and a pixel is not a temperature. The exr beside it is
    where both of those live.
    """
    radiance = _pixels(exr)
    height, width, _ = radiance.shape
    out = bpy.data.images.new(png.stem, width, height)
    try:
        t_k = lwir.brightness_temperature(radiance[..., 0].ravel())
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
        # The datablock goes whatever happened; the float render survives a failure,
        # so a conversion that raised can be retried without paying for the render.
        bpy.data.images.remove(out)
    exr.unlink()


def _index_output(folder: Path) -> bpy.types.CompositorNodeOutputFile:
    """Write each render's object-index pass to `folder`, as float EXR named
    `<file_name>index.exr`.

    Cycles takes it on the ray through the pixel centre, where calibration.json puts
    the pixel, whatever the pixel filter.
    """
    bpy.context.view_layer.use_pass_object_index = True
    tree = bpy.data.node_groups.new("labels", "CompositorNodeTree")
    bpy.context.scene.compositing_node_group = tree
    layers = tree.nodes.new("CompositorNodeRLayers")
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.directory = str(folder)
    output.format.media_type = "IMAGE"
    output.format.file_format = "OPEN_EXR"
    output.format.color_depth = "32"
    output.file_output_items.new("FLOAT", "index")
    tree.links.new(layers.outputs["Object Index"], output.inputs["index"])
    return output


def _targets(built: scene.Built) -> list[labels.Target]:
    return [
        labels.Target(
            pass_index=anchor.pass_index,
            name=anchor.name,
            category=asset,
            centre_m=tuple(anchor.matrix_world.translation.xy),
            waterline_m=scene.waterline_m(anchor),
        )
        for asset, anchors in built.targets.items()
        for anchor in anchors
    ]


def _commit() -> str | None:
    """The source checkout's commit, or None for an installed package."""
    root = Path(__file__).parents[1]
    # A wheel installed into another project's .venv sits inside that project's work
    # tree, and git would answer with its commit.
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(root), "describe", "--always", "--dirty", "--abbrev=40"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _info(scenario: Scenario) -> dict[str, Any]:
    """COCO's info block, and what it takes to reproduce the render."""
    devices = bpy.context.preferences.addons["cycles"].preferences.devices
    gpu = bpy.context.scene.cycles.device == "GPU"
    return {
        "version": version("seascape"),
        "date_created": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _commit(),
        "blender": bpy.app.version_string,
        "blender_build": bpy.app.build_hash.decode(),
        "devices": [d.name for d in devices if d.use] if gpu else ["CPU"],
        "scenario": scenario.model_dump(mode="json"),
    }


def render(scenario: Scenario, into: Path) -> list[Path]:
    """Write one image per camera into `into`, their calibration and their labels."""
    # Blender resolves a relative render.filepath against the .blend, not the shell.
    into = into.resolve()
    into.mkdir(parents=True, exist_ok=True)
    outputs = scenario.outputs
    radius_m = scene.earth_radius_m(scenario.sea.refraction_k)
    written: list[Path] = []
    cameras: list[CameraCalibration] = []
    truth = labels.Labels()
    with tempfile.TemporaryDirectory() as tmp:
        passes = Path(tmp)
        for band in outputs.bands:
            # The default bands ask for both, so an EO-only rig must skip ir, not fail.
            mounts = [m for m in scenario.rig.mounts if m.camera.kind == band]
            if not mounts:
                continue
            thermal_png = band == "ir" and outputs.format == "png"
            built = scene.build(scenario, band)
            targets = _targets(built)
            index_output = _index_output(passes)
            sc = bpy.context.scene
            for mount in mounts:
                sc.camera = built.cameras[mount.name]
                sc.render.resolution_x, sc.render.resolution_y = (
                    mount.camera.width_px,
                    mount.camera.height_px,
                )
                sc.render.filepath = str(into / mount.name)
                index_output.file_name = f"{mount.name}."
                bpy.ops.render.render(write_still=True)
                image = into / f"{mount.name}.{outputs.format}"
                if thermal_png:
                    _thermal_png(into / f"{mount.name}.exr", image)
                written.append(image)
                camera = scene.calibrate(built, mount, image.name)
                cameras.append(camera)
                # Blender's rows run bottom up.
                index = _pixels(passes / f"{mount.name}.index.exr")[::-1, :, 0]
                truth.add(camera, np.rint(index).astype(int), targets, radius_m)
    if not written:
        # Skipping a band the default asked for is right; writing nothing at all
        # means the scenario names only bands its rig has no camera for.
        raise ValueError(
            f"the rig has no camera in any of {outputs.bands}: nothing to render"
        )
    written.append(Calibration(cameras=cameras).write(into))
    # After the builds: the first one picks the device.
    truth.info = _info(scenario)
    written.append(truth.write(into))
    return written

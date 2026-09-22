"""What a rendered LWIR frame has to look like, as numbers rather than an opinion.

A render is the only check on whether the sea and sky read correctly; the physics
assertions in test_lwir.py pass just as happily on a scene that renders black. These
are the properties a refactor must not shift, measured against the working reference
renders these modules were ported from.

Skipped unless `--render` is given. Each one renders in Cycles, which takes seconds and
which CI has no GPU for. Run them before touching the sea or sky shader chain.
"""

from pathlib import Path

import bpy
import numpy as np
import pytest

from seascape import lwir, scene
from seascape.config import Band, Scenario, load

pytestmark = pytest.mark.render

SCENARIO = load(Path(__file__).parent.parent / "scenarios" / "baseline.toml")
SAMPLES = 48


def shoot(size: tuple[int, int], name: str) -> np.ndarray:
    """Render the current scene and return its radiance, top row first."""
    sc = bpy.context.scene
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.filepath = str(Path(bpy.app.tempdir) / name)
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(sc.render.filepath + ".exr")
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(size[1], size[0], 4)[::-1, :, 0]


def radiance(band: Band, kind: str, size: tuple[int, int]) -> np.ndarray:
    """Build for a band, render the named camera, return its radiance."""
    scene.build(SCENARIO, band)
    sc = bpy.context.scene
    sc.camera = next(
        o for o in bpy.data.objects if o.type == "CAMERA" and f"_{kind}_" in o.name
    )
    # Build already set the band's engine and denoiser; only the sample count is the
    # test's own. ir keeps its build default of no OIDN, which is not radiometric.
    sc.cycles.samples = SAMPLES
    return shoot(size, f"drift_{band}")


def texture(rows: np.ndarray) -> float:
    """Spread within each row, robust to a target sitting in the band.

    Median absolute deviation, not standard deviation: a hot hull is a handful of very
    bright pixels and would otherwise swamp the sea it sits on.
    """
    return float(np.median(np.abs(rows - np.median(rows, axis=1, keepdims=True))))


@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    return radiance("ir", "ir", (640, 512))


def test_the_sky_runs_from_cold_overhead_to_ambient_at_the_horizon(frame) -> None:
    """The reference reads 0.90 of ambient at the top of an 18 degree frame.

    Sea and sky have to meet at the same radiance or the horizon reads as an edge
    rather than a boundary, and contrast stops collapsing where a target is hardest
    to see.

    Medians, not means: a target's superstructure stands above the horizon and lands
    in the band this samples, and a hot hull is far enough off ambient to drag it.
    """
    horizon = frame.shape[0] // 2
    ambient = lwir.band_radiance(SCENARIO.sky.t_air_k)
    just_above = float(np.median(frame[horizon - 6 : horizon - 1]))
    assert just_above == pytest.approx(ambient, rel=0.02)
    assert 0.80 <= float(np.median(frame[:5])) / ambient <= 0.95


def test_sea_texture_fades_with_range(frame) -> None:
    """Distant water has to be the smoothest thing in frame.

    Wave relief falls below a pixel with range, so it should average away. Displaced
    geometry does the opposite -- sub-pixel geometry aliases rather than averaging --
    and inverts the profile, which is the whole difference between a frame that reads
    as sea and one that reads as noise. Against the reference renders the grid left the
    far field thirteen times rougher than shader normals do.

    Not strict monotonicity across all four bands: that holds at the reference's 40 m
    eye height but not at this scenario's 12 m, where a foreground row spans less than
    one wavelength and so varies little. Rig height is not the property under test.
    """
    sea = frame[frame.shape[0] // 2 + 4 :]
    band = len(sea) // 4
    far_to_near = [texture(sea[i * band : (i + 1) * band]) for i in range(4)]
    assert far_to_near[0] == min(far_to_near), (
        f"the far field has to settle, not sparkle: {far_to_near}"
    )


def sea_of(scenario: Scenario, waves: bool) -> np.ndarray:
    """The near sea of an ir frame, optionally with the wave relief flattened."""
    scene.build(scenario, "ir")
    bump = next(
        n
        for n in bpy.data.materials["sea"].node_tree.nodes
        if n.bl_idname == "ShaderNodeBump"
    )
    if not waves:
        bump.inputs["Distance"].default_value = 0.0
    sc = bpy.context.scene
    sc.camera = next(
        o for o in bpy.data.objects if o.type == "CAMERA" and "_ir_" in o.name
    )
    # Flat sea is the control, so grain must sit well under the relief; 48 does not.
    sc.cycles.samples = 256
    frame = shoot((320, 256), "isothermal")
    horizon = frame.shape[0] // 2
    return frame[horizon + 30 : horizon + 80]


@pytest.mark.render
def test_waves_survive_a_sea_at_air_temperature() -> None:
    """Wave relief survives a sea exactly at air temperature.

    A tilted facet reflects a different sky elevation, cold overhead to ambient at
    the horizon, so relief shows without `t_sea_k - t_air_k`; 3 K buys 7-18% of it.
    """
    isothermal = SCENARIO.model_copy(
        update={
            "sea": SCENARIO.sea.model_copy(update={"t_sea_k": SCENARIO.sky.t_air_k})
        }
    )

    rippled, flat = sea_of(isothermal, True), sea_of(isothermal, False)

    # 3.2 measured; grain-limited flat control reads 1.0.
    assert texture(rippled) > 2.5 * texture(flat)


@pytest.mark.render
def test_the_noise_delivers_the_slope_it_is_asked_for() -> None:
    """`NOISE_SLOPE_PER_UNIT` against the node itself.

    Bump Distance is metres of relief per wavelength, which is only the slope the chain
    asked for if the noise's own gradient is known. Baked flat and differenced, so a
    Blender change shows up as a number rather than as a sea that looks slightly wrong.
    """
    span, px = 20.0, 1024  # 2 cm sampling; see the constant's comment
    bpy.ops.wm.read_factory_settings(use_empty=True)
    frame = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=span)
    material = bpy.data.materials.new("probe")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.0  # one noise unit is one metre
    noise.inputs["Detail"].default_value = scene.NOISE_DETAIL
    noise.inputs["Roughness"].default_value = scene.NOISE_ROUGHNESS
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    position = tree.nodes.new("ShaderNodeNewGeometry").outputs["Position"]
    tree.links.new(position, noise.inputs["Vector"])
    tree.links.new(noise.outputs["Fac"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    bpy.context.object.data.materials.append(material)

    lens = bpy.data.cameras.new("probe")
    lens.type, lens.ortho_scale = "ORTHO", span
    camera = bpy.data.objects.new("probe", lens)
    frame.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 10.0)
    frame.camera = camera
    frame.render.engine = "CYCLES"
    frame.cycles.samples = 1
    frame.cycles.use_denoising = False
    frame.render.resolution_x = frame.render.resolution_y = px
    frame.view_settings.view_transform = "Standard"

    fac = shoot((px, px), "noise_probe")
    gradient_y, gradient_x = np.gradient(fac.astype(np.float64), span / px)
    measured = float(np.sqrt(np.mean(gradient_x**2 + gradient_y**2)))
    assert measured == pytest.approx(scene.NOISE_SLOPE_PER_UNIT, abs=0.03)

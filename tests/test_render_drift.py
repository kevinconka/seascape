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
from seascape.config import Band, load

pytestmark = pytest.mark.render

SCENARIO = load(Path(__file__).parent.parent / "scenarios" / "baseline.toml")
SAMPLES = 48


def radiance(band: Band, kind: str, size: tuple[int, int]) -> np.ndarray:
    """Render one camera and return its radiance, top row first."""
    scene.build(SCENARIO, band)
    sc = bpy.context.scene
    sc.camera = next(
        o for o in bpy.data.objects if o.type == "CAMERA" and f"_{kind}_" in o.name
    )
    sc.render.engine = "CYCLES"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.filepath = str(Path(bpy.app.tempdir) / f"drift_{band}")
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(sc.render.filepath + ".exr")
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(size[1], size[0], 4)[::-1, :, 0]


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

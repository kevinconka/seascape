"""The parts of `render` that need no Cycles run.

`bpy.ops.render.render` is the only piece that needs a GPU, and `test_render_drift`
covers it behind `--render`. Everything here is settings and pixel arithmetic, which
is also where the silent failures live.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import bpy
import numpy as np
import pytest

from seascape import labels, render, scene
from seascape.calibration import Calibration
from seascape.config import Band, ImageFormat, load

BASELINE = Path(__file__).parent.parent / "scenarios" / "baseline.toml"
TWIN_POD = BASELINE.with_name("twin-pod.toml")


def _raise(*_: object) -> np.ndarray:
    raise RuntimeError("injected")


def exr_of(tmp_path: Path, radiance: list[float], width: int = 4) -> Path:
    """One row per value, bottom row first, as Blender orders pixels."""
    image = bpy.data.images.new("probe", width, len(radiance), float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    rows = np.repeat(np.array(radiance, dtype=np.float32), width)
    image.pixels.foreach_set(
        np.column_stack([rows, rows, rows, np.ones_like(rows)]).ravel()
    )
    image.file_format = "OPEN_EXR"
    path = tmp_path / "probe.exr"
    image.filepath_raw = str(path)
    image.save()
    bpy.data.images.remove(image)
    return path


def grey_of(png: Path) -> np.ndarray:
    """The rows of a written png as one value each, bottom row first."""
    image = bpy.data.images.load(str(png))
    image.colorspace_settings.name = "Non-Color"
    width, height = image.size
    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    bpy.data.images.remove(image)
    return buffer.reshape(height, width, 4)[:, 0, 0]


class TestThermalPng:
    """Coldest pixel black, hottest white, linear in brightness temperature between.

    An inverted, flipped or sRGB-encoded frame is still a plausible-looking picture,
    so only the numbers catch it.
    """

    def test_the_frame_is_stretched_to_the_full_range(self, tmp_path: Path) -> None:
        from seascape import lwir

        exr = exr_of(tmp_path, [lwir.band_radiance(t) for t in (272.0, 295.0)])
        png = tmp_path / "out.png"
        render._thermal_png(exr, png)
        low, high = grey_of(png)
        assert low == pytest.approx(0.0, abs=0.01)
        assert high == pytest.approx(1.0, abs=0.01)

    def test_the_middle_temperature_is_mid_grey(self, tmp_path: Path) -> None:
        """Catches an sRGB encode, which puts 0.5 at 0.74."""
        from seascape import lwir

        exr = exr_of(tmp_path, [lwir.band_radiance(t) for t in (270.0, 285.0, 300.0)])
        png = tmp_path / "out.png"
        render._thermal_png(exr, png)
        assert grey_of(png)[1] == pytest.approx(0.5, abs=0.01)

    def test_a_target_is_not_flattened_to_white(self, tmp_path: Path) -> None:
        """A percentile stretch would trim the few rows a distant hull occupies."""
        from seascape import lwir

        sea = [lwir.band_radiance(285.0)] * 40
        exr = exr_of(tmp_path, [*sea, lwir.band_radiance(300.0)])
        png = tmp_path / "out.png"
        render._thermal_png(exr, png)
        grey = grey_of(png)
        assert grey[-1] == pytest.approx(1.0, abs=0.01)  # the hull
        assert grey[0] == pytest.approx(0.0, abs=0.01)  # the sea

    def test_a_frame_of_one_temperature_does_not_divide_by_zero(
        self, tmp_path: Path
    ) -> None:
        from seascape import lwir

        exr = exr_of(tmp_path, [lwir.band_radiance(290.0)] * 4)
        png = tmp_path / "out.png"
        render._thermal_png(exr, png)
        assert np.isfinite(grey_of(png)).all()

    def test_the_float_render_is_removed_on_success(self, tmp_path: Path) -> None:
        from seascape import lwir

        exr = exr_of(tmp_path, [lwir.band_radiance(285.0), lwir.band_radiance(295.0)])
        render._thermal_png(exr, tmp_path / "out.png")
        assert not exr.exists()

    def test_a_failure_keeps_the_float_render_and_leaks_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A render costs minutes; a failed conversion must not throw it away."""
        from seascape import lwir

        exr = exr_of(tmp_path, [lwir.band_radiance(285.0)])
        before = len(bpy.data.images)
        monkeypatch.setattr(render.lwir, "brightness_temperature", _raise)
        with pytest.raises(RuntimeError):
            render._thermal_png(exr, tmp_path / "out.png")
        assert exr.exists()
        assert len(bpy.data.images) == before


def built(band: Band, **outputs: object) -> bpy.types.Scene:
    scenario = load(BASELINE)
    # No ship: these are render settings, and a hull is a second of FBX import.
    scenario = scenario.model_copy(
        update={
            "objects": [],
            "outputs": scenario.outputs.model_copy(update=outputs),
        }
    )
    scene.build(scenario, band)
    return bpy.context.scene


class TestSettings:
    def test_every_format_maps_to_one_blender_identifier(self) -> None:
        """A format added to the Literal alone renders as whatever was set last."""
        assert set(scene._FORMATS) == set(get_args(ImageFormat.__value__))

    @pytest.mark.parametrize("fmt", get_args(ImageFormat.__value__))
    def test_the_extension_matches_the_name_render_files_under(
        self, fmt: ImageFormat
    ) -> None:
        """`render` composes its return paths from the format, not from Blender."""
        sc = built("eo", format=fmt)

        assert sc.render.file_extension == f".{fmt}"

    def test_ir_renders_float_even_when_a_png_is_asked_for(self) -> None:
        """Radiance through 8 bits is no longer radiance."""
        sc = built("ir", format="png")

        assert sc.render.image_settings.color_depth == "32"

    @pytest.mark.parametrize(("band", "denoised"), [("eo", True), ("ir", False)])
    def test_only_eo_is_denoised(self, band: Band, denoised: bool) -> None:
        """OIDN invents 10 K of structure on a field that is flat by construction."""
        sc = built(band)

        assert sc.cycles.use_denoising is denoised

    @pytest.mark.parametrize("band", get_args(Band.__value__))
    def test_both_bands_render_in_cycles(self, band: Band) -> None:
        """EEVEE renders the sea at half its radiance."""
        sc = built(band)

        assert sc.render.engine == "CYCLES"

    def test_the_active_camera_sets_the_resolution(self) -> None:
        """Factory 1920x1080 otherwise, whatever the camera says it is."""
        eo = next(m.camera for m in load(BASELINE).rig.mounts if m.camera.kind == "eo")

        sc = built("eo")

        assert (sc.render.resolution_x, sc.render.resolution_y) == (
            eo.width_px,
            eo.height_px,
        )

    @pytest.mark.parametrize(
        ("band", "exposure_ev"),
        [("eo", load(BASELINE).outputs.exposure_ev), ("ir", 0.0)],
    )
    def test_only_eo_is_exposed(self, band: Band, exposure_ev: float) -> None:
        """Radiance through an exposure is no longer radiance."""
        sc = built(band)

        assert sc.view_settings.exposure == exposure_ev


def test_a_relative_output_reaches_blender_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative, Blender fails to save: 'cannot save: EO_PORT_P.png'."""
    handed: list[str] = []

    def capture(**_: object) -> None:
        handed.append(bpy.context.scene.render.filepath)
        raise RuntimeError("captured")

    # bpy.ops.render is rebuilt on every access, so patch the attribute that holds it.
    monkeypatch.setattr(bpy.ops, "render", SimpleNamespace(render=capture))
    monkeypatch.chdir(tmp_path)
    scenario = load(BASELINE)
    scenario = scenario.model_copy(
        update={
            "objects": [],
            "outputs": scenario.outputs.model_copy(update={"bands": ("eo",)}),
        }
    )

    with pytest.raises(RuntimeError, match="captured"):
        render.render(scenario, Path("out"))

    assert Path(handed[0]).is_absolute()
    assert Path(handed[0]).parent == tmp_path / "out"


@pytest.mark.render
@pytest.mark.parametrize(("edge_m", "columns"), [(3.45, 3), (3.55, 4)])
def test_the_object_index_pass_samples_the_pixel_centre(
    tmp_path: Path, edge_m: float, columns: int
) -> None:
    """A plane's edge 0.05 px either side of a column of centres, 1 m to the pixel.

    The boxes come from this pass. A sample anywhere else in the 1.5 px filter would
    put the edge in a different column from row to row.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x, sc.render.resolution_y = 8, 64
    mesh = bpy.data.meshes.new("plane")
    corners = [(-10, -10, 0), (edge_m, -10, 0), (edge_m, 74, 0), (-10, 74, 0)]
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    plane = bpy.data.objects.new("plane", mesh)
    plane.pass_index = 1
    lens = bpy.data.cameras.new("lens")
    lens.type, lens.sensor_fit, lens.ortho_scale = "ORTHO", "HORIZONTAL", 8.0
    camera = bpy.data.objects.new("camera", lens)
    camera.location = (4.0, 32.0, 10.0)  # looking down, frame over x 0-8, y 0-64
    for obj in (plane, camera):
        sc.collection.objects.link(obj)
    sc.camera = camera
    render._index_output(tmp_path).file_name = "probe."
    sc.render.filepath = str(tmp_path / "frame")

    bpy.ops.render.render(write_still=True)

    index = np.rint(render._pixels(tmp_path / "probe.index.exr")[..., 0])
    assert (index[:, :columns] == 1).all()
    assert (index[:, columns:] == 0).all()


@pytest.mark.render
def test_each_box_holds_its_hull_centre_through_the_calibration(
    tmp_path: Path,
) -> None:
    """Boxes from the pass and calibration from the camera agree on where a hull is.

    A box is sampled at pixel centres, so the hull can reach half a pixel past it.
    """
    scenario = load(TWIN_POD)
    scenario = scenario.model_copy(
        update={"outputs": scenario.outputs.model_copy(update={"bands": ("ir",)})}
    )

    render.render(scenario, tmp_path)

    truth = labels.Labels.model_validate_json((tmp_path / labels.FILENAME).read_text())
    cameras = {c.name: c for c in Calibration.read(tmp_path).cameras}
    frames = {image.id: cameras[image.camera] for image in truth.images}
    assert truth.annotations, "no target in any frame: nothing below ran"
    for found in truth.annotations:
        camera = frames[found.image_id]
        centre = bpy.data.objects[found.name].matrix_world.translation
        x, y, z, _ = np.linalg.inv(camera.extrinsics["world"]) @ (*centre, 1.0)
        u, v, _ = np.array(camera.K) @ (x, y, z) / z + 0.5  # COCO pixels
        left, top, width, height = found.bbox
        assert left - 0.5 <= u <= left + width + 0.5, found.name
        assert top - 0.5 <= v <= top + height + 0.5, found.name
        assert found.waterline_range_m < found.range_m

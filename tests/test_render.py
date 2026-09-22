"""The parts of `render` that need no Cycles run.

`bpy.ops.render.render` is the only piece that needs a GPU, and `test_render_drift`
covers it behind `--render`. Everything here is settings and pixel arithmetic, which
is also where the silent failures live.
"""

from pathlib import Path
from typing import get_args

import bpy
import numpy as np
import pytest

from seascape import render, scene
from seascape.config import Band, ImageFormat, load

BASELINE = Path(__file__).parent.parent / "scenarios" / "baseline.toml"


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
    scenario = scenario.model_copy(
        update={"outputs": scenario.outputs.model_copy(update=outputs)}
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
        assert built("eo", format=fmt).render.file_extension == f".{fmt}"

    def test_ir_renders_float_even_when_a_png_is_asked_for(self) -> None:
        """Radiance through 8 bits is no longer radiance."""
        assert built("ir", format="png").render.image_settings.color_depth == "32"

    @pytest.mark.parametrize(("band", "denoised"), [("eo", True), ("ir", False)])
    def test_only_eo_is_denoised(self, band: Band, denoised: bool) -> None:
        """OIDN invents 10 K of structure on a field that is flat by construction."""
        assert built(band).cycles.use_denoising is denoised

    @pytest.mark.parametrize("band", get_args(Band.__value__))
    def test_both_bands_render_in_cycles(self, band: Band) -> None:
        """EEVEE renders the sea at half its radiance."""
        assert built(band).render.engine == "CYCLES"

    def test_the_active_camera_sets_the_resolution(self) -> None:
        """Factory 1920x1080 otherwise, whatever the camera says it is."""
        eo = next(m.camera for m in load(BASELINE).rig.mounts if m.camera.kind == "eo")
        sc = built("eo")
        assert (sc.render.resolution_x, sc.render.resolution_y) == (
            eo.width_px,
            eo.height_px,
        )

    def test_eo_is_exposed_and_ir_is_not(self) -> None:
        """Radiance through an exposure is no longer radiance."""
        assert built("eo").view_settings.exposure == load(BASELINE).outputs.exposure_ev
        assert built("ir").view_settings.exposure == 0.0

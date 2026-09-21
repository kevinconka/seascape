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

from seascape import render
from seascape.config import Band, ImageFormat, Scenario, load

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
    """Black at the window's low end, white at its high end, linear between.

    An inverted, flipped or sRGB-encoded image is a plausible-looking picture, so
    only the numbers catch it.
    """

    WINDOW = (270.0, 300.0)

    def test_the_window_ends_map_to_black_and_white(self, tmp_path: Path) -> None:
        from seascape import lwir

        exr = exr_of(tmp_path, [lwir.band_radiance(t) for t in (270.0, 300.0)])
        png = tmp_path / "out.png"
        render._thermal_png(exr, png, self.WINDOW)
        low, high = grey_of(png)
        assert low == pytest.approx(0.0, abs=0.01)
        assert high == pytest.approx(1.0, abs=0.01)

    def test_the_middle_of_the_window_is_mid_grey(self, tmp_path: Path) -> None:
        """Catches an sRGB encode, which puts 0.5 at 0.74."""
        from seascape import lwir

        exr = exr_of(tmp_path, [lwir.band_radiance(285.0)])
        png = tmp_path / "out.png"
        render._thermal_png(exr, png, self.WINDOW)
        assert grey_of(png)[0] == pytest.approx(0.5, abs=0.01)

    def test_the_float_render_is_removed_on_success(self, tmp_path: Path) -> None:
        from seascape import lwir

        exr = exr_of(tmp_path, [lwir.band_radiance(285.0)])
        render._thermal_png(exr, tmp_path / "out.png", self.WINDOW)
        assert not exr.exists()

    def test_a_failure_keeps_the_float_render_and_leaks_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A render costs minutes; a failed conversion must not throw it away."""
        from seascape import lwir

        exr = exr_of(tmp_path, [lwir.band_radiance(285.0)])
        before = len(bpy.data.images)
        monkeypatch.setattr(render.lwir, "brightness_temperature", _raise, raising=True)
        with pytest.raises(RuntimeError):
            render._thermal_png(exr, tmp_path / "out.png", self.WINDOW)
        assert exr.exists()
        assert len(bpy.data.images) == before


class TestSettings:
    scenario: Scenario

    @classmethod
    def setup_class(cls) -> None:
        cls.scenario = load(BASELINE)

    def test_every_format_maps_to_one_blender_identifier(self) -> None:
        """A format added to the Literal alone renders as whatever was set last."""
        assert set(render._FORMATS) == set(get_args(ImageFormat.__value__))

    @pytest.mark.parametrize("fmt", get_args(ImageFormat.__value__))
    def test_the_extension_matches_the_name_render_files_under(
        self, fmt: ImageFormat
    ) -> None:
        """`render` composes its return paths from the format, not from Blender."""
        render._settings(self.scenario, "eo", fmt)
        assert bpy.context.scene.render.file_extension == f".{fmt}"

    @pytest.mark.parametrize(("band", "denoised"), [("eo", True), ("ir", False)])
    def test_only_eo_is_denoised(self, band: Band, denoised: bool) -> None:
        """OIDN invents 10 K of structure on a field that is flat by construction."""
        render._settings(self.scenario, band, "exr")
        assert bpy.context.scene.cycles.use_denoising is denoised

    def test_radiance_keeps_its_full_float(self) -> None:
        render._settings(self.scenario, "ir", "exr")
        assert bpy.context.scene.render.image_settings.color_depth == "32"

    def test_eo_is_exposed_and_ir_is_not(self) -> None:
        """Radiance through an exposure is no longer radiance."""
        render._settings(self.scenario, "eo", "png")
        assert bpy.context.scene.view_settings.exposure == (
            self.scenario.outputs.exposure_ev
        )
        bpy.context.scene.view_settings.exposure = 0.0
        render._settings(self.scenario, "ir", "exr")
        assert bpy.context.scene.view_settings.exposure == 0.0

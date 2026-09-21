"""The parts of `render` that need no Cycles run.

`bpy.ops.render.render` is the only piece that needs a GPU, and `test_render_drift`
covers it behind `--render`. Everything here is settings, which is where a wrong
render comes from something that never raised.
"""

from pathlib import Path

import bpy
import pytest

from seascape import render
from seascape.config import Band, Scenario, load

BASELINE = Path(__file__).parent.parent / "scenarios" / "baseline.toml"


class TestSettings:
    scenario: Scenario

    @classmethod
    def setup_class(cls) -> None:
        cls.scenario = load(BASELINE)

    def test_the_extension_matches_the_name_render_files_under(self) -> None:
        """`render` composes its return paths itself, not from what Blender wrote."""
        render._settings(self.scenario, "eo")
        assert bpy.context.scene.render.file_extension == ".exr"

    @pytest.mark.parametrize(("band", "denoised"), [("eo", True), ("ir", False)])
    def test_only_eo_is_denoised(self, band: Band, denoised: bool) -> None:
        """OIDN invents 10 K of structure on a field that is flat by construction."""
        render._settings(self.scenario, band)
        assert bpy.context.scene.cycles.use_denoising is denoised

    def test_radiance_keeps_its_full_float(self) -> None:
        render._settings(self.scenario, "ir")
        assert bpy.context.scene.render.image_settings.color_depth == "32"

"""Ground truth against the scene it describes.

`truth` never imports bpy, so these two are independent calculations of the same
thing. That is the point: if the rig convention drifts in one, the other disagrees.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from seascape import truth
from seascape.config import load

SCENARIO = load(Path(__file__).parent.parent / "scenarios" / "baseline.toml")
GROUND_TRUTH = truth.ground_truth(SCENARIO)
FANS = {f"{c.pod}_{c.kind}_{c.bearing_deg:+g}": c for c in SCENARIO.rig.cameras}


def test_intrinsics_follow_from_the_field_of_view() -> None:
    for spec, camera in zip(SCENARIO.rig.cameras, GROUND_TRUTH.cameras, strict=True):
        k = camera.intrinsics
        assert k.fx_px == k.fy_px, "square pixels"
        # Half the sensor subtends half the field of view.
        assert math.degrees(2 * math.atan(k.cx_px / k.fx_px)) == pytest.approx(
            spec.hfov_deg
        )


def test_a_rotation_matrix_is_orthonormal() -> None:
    for camera in GROUND_TRUTH.cameras:
        r = np.array(camera.pose.r_camera_from_world)
        assert r @ r.T == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0), "right handed, no reflection"


def test_a_target_is_in_frame_exactly_when_its_bearing_is() -> None:
    """The cheap check the rig has to agree with: is the target inside the fan?"""
    for target in GROUND_TRUTH.targets:
        for sighting in target.sightings:
            spec = FANS[sighting.camera]
            off_axis = abs(target.bearing_deg - spec.bearing_deg)
            assert sighting.in_frame == (off_axis < spec.hfov_deg / 2), sighting.camera


def test_the_timestamp_is_seconds_not_a_frame_index() -> None:
    assert isinstance(GROUND_TRUTH.t_s, float)
    assert truth.ground_truth(SCENARIO, t_s=1.5).t_s == 1.5


def test_it_carries_the_resolved_scenario() -> None:
    """A truth file that cannot say what it came from explains nothing later."""
    assert GROUND_TRUTH.scenario == SCENARIO
    assert GROUND_TRUTH.model_dump_json()


class TestAgainstTheBuiltScene:
    """Acceptance check 3: the recorded pose and K land on the rendered pixel."""

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        from seascape import scene

        scene.build(SCENARIO, "eo")

    def test_the_recorded_pose_matches_the_camera_in_the_scene(self) -> None:
        import bpy

        for camera in GROUND_TRUTH.cameras:
            built = bpy.data.objects[camera.name]
            # The scene holds world-from-camera; truth records its inverse.
            # 1e-6: Blender stores object matrices as float32.
            assert np.array(built.matrix_world.to_3x3()) == pytest.approx(
                np.array(camera.pose.r_camera_from_world).T, abs=1e-6
            ), camera.name
            assert tuple(built.location) == pytest.approx(camera.pose.position_m)

    def test_the_recorded_pixel_is_where_blender_projects_it(self) -> None:
        """Blender's own projection against this module's K and R."""
        import bpy
        from bpy_extras.object_utils import world_to_camera_view
        from mathutils import Vector

        by_name = {camera.name: camera for camera in GROUND_TRUTH.cameras}
        for target in GROUND_TRUTH.targets:
            bearing = math.radians(target.bearing_deg)
            at = Vector(
                (
                    target.range_m * math.sin(bearing),
                    target.range_m * math.cos(bearing),
                    0.0,
                )
            )
            for sighting in (s for s in target.sightings if s.in_frame):
                k = by_name[sighting.camera].intrinsics
                # world_to_camera_view normalises over the render frame, so the frame
                # has to be this camera's before it means anything.
                render = bpy.context.scene.render
                render.resolution_x, render.resolution_y = k.width_px, k.height_px
                x, y, _ = world_to_camera_view(
                    bpy.context.scene, bpy.data.objects[sighting.camera], at
                )
                blender_px = (x * k.width_px, (1.0 - y) * k.height_px)
                assert sighting.centre_px == pytest.approx(blender_px, abs=0.5), (
                    sighting.camera
                )


def test_only_the_rendered_bands_appear() -> None:
    """A camera in the truth file has an image beside it, or it should not be there."""
    ir_only = SCENARIO.model_copy(
        update={"outputs": SCENARIO.outputs.model_copy(update={"bands": ("ir",)})}
    )
    assert {camera.band for camera in truth.ground_truth(ir_only).cameras} == {"ir"}

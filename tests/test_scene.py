"""What the built scene measures, not what the scenario says.

Blender is one global session, so a scene built for one band replaces the other. Each
class rebuilds in its own band on entry, which keeps the file order-independent.
"""

import math
from itertools import pairwise
from pathlib import Path

import bpy
import numpy as np
import pytest
from mathutils import Vector

from seascape import lwir, scene
from seascape.assets import manifest
from seascape.config import Camera, load

SCENARIO = load(Path(__file__).parent.parent / "scenarios" / "baseline.toml")


def name_of(spec: Camera) -> str:
    return f"{spec.pod}_{spec.kind}_{spec.bearing_deg:+g}"


def counts() -> tuple[int, ...]:
    return tuple(
        len(block) for block in (bpy.data.objects, bpy.data.materials, bpy.data.images)
    )


def test_a_named_substream_is_reproducible_and_local_to_its_name() -> None:
    """Adding a component must not perturb one that already draws from the seed."""

    def draw(seed: int, name: str) -> int:
        return int(scene._substream(seed, name).integers(2**31))

    assert draw(7, "sea/surface") == draw(7, "sea/surface")
    assert draw(7, "sea/surface") != draw(8, "sea/surface")
    assert draw(7, "sea/surface") != draw(7, "sky/haze")


class TestGeometry:
    """Everything the band does not change: where things are and where cameras look."""

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(SCENARIO, "eo")

    def test_starboard_bearings_yaw_to_port(self) -> None:
        """The one negation. Two of them cancel and the whole rig mirrors unnoticed."""
        for spec in SCENARIO.rig.cameras:
            camera = bpy.data.objects[name_of(spec)]
            yaw = math.degrees(camera.rotation_euler.z)
            assert yaw == pytest.approx(-spec.bearing_deg)

    def test_cameras_carry_their_field_of_view_horizontally(self) -> None:
        """AUTO fits the angle to the longer image side, flipping a portrait sensor."""
        for spec in SCENARIO.rig.cameras:
            data = bpy.data.objects[name_of(spec)].data
            assert data.sensor_fit == "HORIZONTAL"
            assert math.degrees(data.angle_x) == pytest.approx(spec.hfov_deg)

    def test_pod_span_and_overlap_measured_from_the_scene(self) -> None:
        """The acceptance numbers, read off the built cameras rather than the config."""
        arcs: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for spec in SCENARIO.rig.cameras:
            camera = bpy.data.objects[name_of(spec)]
            half = math.degrees(camera.data.angle_x) / 2
            # Blender yaw is the bearing negated, so read the bearing back out.
            centre = -math.degrees(camera.rotation_euler.z)
            arcs.setdefault((spec.pod, spec.kind), []).append(
                (centre - half, centre + half)
            )

        for key, span, overlap in (
            (("port", "eo"), 125.0, 5.0),
            (("starboard", "eo"), 125.0, 5.0),
            (("bow", "ir"), 44.0, 4.0),
        ):
            sectors = sorted(arcs[key])
            assert sectors[-1][1] - sectors[0][0] == pytest.approx(span), key
            gaps = [a[1] - b[0] for a, b in pairwise(sectors)]
            assert gaps == pytest.approx([overlap] * len(gaps)), key

    def test_the_far_clip_clears_every_target(self) -> None:
        """Blender's default 1000 m renders a 2 km target as sky, reporting nothing."""
        furthest = max(spec.range_m for spec in SCENARIO.objects)
        for spec in SCENARIO.rig.cameras:
            assert bpy.data.objects[name_of(spec)].data.clip_end > furthest

    def test_a_target_lands_at_its_range_and_bearing(self) -> None:
        for spec in SCENARIO.objects:
            east, north, _ = bpy.data.objects[spec.asset].location
            assert math.hypot(east, north) == pytest.approx(spec.range_m)
            assert math.degrees(math.atan2(east, north)) == pytest.approx(
                spec.bearing_deg
            )

    def test_a_target_is_fitted_to_its_manifest_length(self) -> None:
        """The mesh arrives ~1 unit long; unfitted it is a speck at 2 km."""
        for spec in SCENARIO.objects:
            anchor = bpy.data.objects[spec.asset]
            into_hull = anchor.matrix_world.inverted()
            corners = [
                into_hull @ part.matrix_world @ Vector(corner)
                for part in anchor.children
                if part.type == "MESH"
                for corner in part.bound_box
            ]
            axes = list(zip(*corners, strict=True))
            length = manifest()[spec.asset].length_m
            assert max(axes[1]) - min(axes[1]) == pytest.approx(length)
            # 1 mm: the fit runs through float32 mesh coordinates.
            assert min(axes[2]) == pytest.approx(0.0, abs=1e-3), "keel on the waterline"

    def test_the_ocean_takes_its_wind_from_the_scenario(self) -> None:
        ocean = bpy.data.objects["sea"].modifiers["ocean"]
        assert ocean.wind_velocity == pytest.approx(SCENARIO.sea.wind_speed_mps)
        assert ocean.choppiness == pytest.approx(SCENARIO.sea.choppiness)
        assert ocean.spectrum == "PIERSON_MOSKOWITZ"

    def test_the_sea_reaches_past_the_furthest_target(self) -> None:
        mesh = (
            bpy.data.objects["sea"]
            .evaluated_get(bpy.context.evaluated_depsgraph_get())
            .to_mesh()
        )
        reach = max(abs(vertex.co.x) for vertex in mesh.vertices)
        assert reach > max(spec.range_m for spec in SCENARIO.objects)

    def test_building_twice_leaves_the_same_scene(self) -> None:
        """Node trees leak when a build appends to what is already there."""
        before = counts()
        scene.build(SCENARIO, "eo")
        assert counts() == before


class TestEoBand:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(SCENARIO, "eo")

    def test_the_sea_refracts_at_seawater_ior(self) -> None:
        bsdf = bpy.data.materials["sea"].node_tree.nodes["Principled BSDF"]
        assert bsdf.inputs["IOR"].default_value == pytest.approx(1.33)

    def test_the_sky_is_lit(self) -> None:
        """The EO sky is the scattering model, not the empty world the IR band gets."""
        assert bpy.data.worlds["sky"].node_tree.nodes["Sky Texture"]


class TestIrBand:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(SCENARIO, "ir")

    def test_the_sky_is_empty(self) -> None:
        """8-14 um downwelling needs an atmospheric model this project does not have."""
        assert tuple(bpy.data.worlds["sky"].color) == (0.0, 0.0, 0.0)

    def test_a_target_radiates_at_its_own_temperature(self) -> None:
        """`t_k` is in the scenario; a target rendering at its albedo ignores it."""
        for spec in SCENARIO.objects:
            hull = next(
                part
                for part in bpy.data.objects[spec.asset].children_recursive
                if part.type == "MESH"
            )
            emission = hull.data.materials[0].node_tree.nodes["Emission"]
            assert emission.inputs["Strength"].default_value == pytest.approx(
                lwir.band_radiance(spec.t_k), rel=1e-5
            )

    def test_the_baked_emissivity_matches_the_curve(self) -> None:
        """The shader reads this by cos(theta); the curve is sampled by theta."""
        image = bpy.data.images["sea_emissivity"]
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        baked = pixels.reshape(-1, 4)[:, 0]

        _, eps = lwir.emissivity_curve(t_sea_k=SCENARIO.sea.t_sea_k)
        assert baked[-1] == pytest.approx(eps[0], rel=1e-4), "cos(theta)=1 is normal"
        assert baked[0] == pytest.approx(eps[-1], abs=2e-3), "cos(theta)=0 is grazing"
        assert np.all(np.diff(baked) >= -1e-6), "emissivity rises towards normal"

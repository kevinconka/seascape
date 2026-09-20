"""What the built scene measures, not what the scenario says.

Building takes a few seconds, most of it the FBX import, so the module shares one scene.
Every test that changes it rebuilds the same scenario, so order stays irrelevant.
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
from seascape.config import Camera, Scenario, load

BASELINE = Path(__file__).parent.parent / "scenarios" / "baseline.toml"


def name_of(spec: Camera) -> str:
    return f"{spec.pod}_{spec.kind}_{spec.bearing_deg:+g}"


def counts() -> tuple[int, ...]:
    return tuple(
        len(block) for block in (bpy.data.objects, bpy.data.materials, bpy.data.images)
    )


@pytest.fixture(scope="module")
def built() -> Scenario:
    scenario = load(BASELINE)
    scene.build(scenario)
    return scenario


def test_a_named_substream_is_reproducible_and_local_to_its_name() -> None:
    """Adding a component must not perturb one that already draws from the seed."""
    draw = lambda seed, name: scene._substream(seed, name).integers(2**31)  # noqa: E731
    assert draw(7, "sea/surface") == draw(7, "sea/surface")
    assert draw(7, "sea/surface") != draw(8, "sea/surface")
    assert draw(7, "sea/surface") != draw(7, "sky/haze")


def test_starboard_bearings_yaw_to_port(built: Scenario) -> None:
    """The one negation. Two of them cancel and the whole rig mirrors unnoticed."""
    for spec in built.rig.cameras:
        camera = bpy.data.objects[name_of(spec)]
        assert math.degrees(camera.rotation_euler.z) == pytest.approx(-spec.bearing_deg)


def test_cameras_carry_their_field_of_view_horizontally(built: Scenario) -> None:
    """AUTO fits the angle to the longer image side, so a portrait sensor flips it."""
    for spec in built.rig.cameras:
        data = bpy.data.objects[name_of(spec)].data
        assert data.sensor_fit == "HORIZONTAL"
        assert math.degrees(data.angle_x) == pytest.approx(spec.hfov_deg)


def test_pod_span_and_overlap_measured_from_the_scene(built: Scenario) -> None:
    """The acceptance numbers, read off the built cameras rather than the config."""
    edges = {}
    for spec in built.rig.cameras:
        data = bpy.data.objects[name_of(spec)].data
        half = math.degrees(data.angle_x) / 2
        # Blender yaw is the bearing negated, so read the bearing back out of the scene.
        centre = -math.degrees(bpy.data.objects[name_of(spec)].rotation_euler.z)
        edges.setdefault((spec.pod, spec.kind), []).append(
            (centre - half, centre + half)
        )

    for key, expected_span, expected_overlap in (
        (("port", "eo"), 125.0, 5.0),
        (("starboard", "eo"), 125.0, 5.0),
        (("bow", "ir"), 44.0, 4.0),
    ):
        arcs = sorted(edges[key])
        assert arcs[-1][1] - arcs[0][0] == pytest.approx(expected_span), key
        overlaps = [a[1] - b[0] for a, b in pairwise(arcs)]
        assert overlaps == pytest.approx([expected_overlap] * len(overlaps)), key


def test_the_far_clip_clears_every_target(built: Scenario) -> None:
    """At Blender's default 1000 m a 2 km target renders as sky, reporting nothing."""
    furthest = max(spec.range_m for spec in built.objects)
    for spec in built.rig.cameras:
        assert bpy.data.objects[name_of(spec)].data.clip_end > furthest


def test_a_target_lands_at_its_range_and_bearing(built: Scenario) -> None:
    for spec in built.objects:
        east, north, _ = bpy.data.objects[spec.asset].location
        assert math.hypot(east, north) == pytest.approx(spec.range_m)
        assert math.degrees(math.atan2(east, north)) == pytest.approx(spec.bearing_deg)


def test_a_target_is_fitted_to_its_manifest_length(built: Scenario) -> None:
    """The mesh arrives ~1 unit long and off-origin; unfitted it is a speck at range."""
    for spec in built.objects:
        anchor = bpy.data.objects[spec.asset]
        into_hull = anchor.matrix_world.inverted()
        corners = [
            into_hull @ part.matrix_world @ Vector(corner)
            for part in anchor.children
            if part.type == "MESH"
            for corner in part.bound_box
        ]
        axes = list(zip(*corners, strict=True))
        assert max(axes[1]) - min(axes[1]) == pytest.approx(
            manifest()[spec.asset].length_m
        )
        # 1 mm: the fit runs through float32 mesh coordinates.
        assert min(axes[2]) == pytest.approx(0.0, abs=1e-3), "keel on the waterline"


def test_the_ocean_takes_its_wind_from_the_scenario(built: Scenario) -> None:
    ocean = bpy.data.objects["sea"].modifiers["ocean"]
    assert ocean.wind_velocity == pytest.approx(built.sea.wind_speed_mps)
    assert ocean.choppiness == pytest.approx(built.sea.choppiness)
    assert ocean.spectrum == "PIERSON_MOSKOWITZ"


def test_the_sea_reaches_past_the_furthest_target(built: Scenario) -> None:
    mesh = (
        bpy.data.objects["sea"]
        .evaluated_get(bpy.context.evaluated_depsgraph_get())
        .to_mesh()
    )
    reach = max(abs(vertex.co.x) for vertex in mesh.vertices)
    assert reach > max(spec.range_m for spec in built.objects)


def test_the_baked_emissivity_matches_the_curve(built: Scenario) -> None:
    """The shader reads this by cos(theta); the curve is sampled by theta."""
    image = bpy.data.images["sea_emissivity"]
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    baked = pixels.reshape(-1, 4)[:, 0]

    _, eps = lwir.emissivity_curve(t_sea_k=built.sea.t_sea_k)
    assert baked[-1] == pytest.approx(eps[0], rel=1e-4), "cos(theta)=1 is normal"
    assert baked[0] == pytest.approx(eps[-1], abs=2e-3), "cos(theta)=0 is grazing"
    assert np.all(np.diff(baked) >= -1e-6), "emissivity rises towards normal incidence"


def test_building_twice_leaves_the_same_scene(built: Scenario) -> None:
    """Node trees leak when a build appends to what is already there."""
    before = counts()
    scene.build(built)
    assert counts() == before

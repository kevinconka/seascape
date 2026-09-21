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
            asset = manifest()[spec.asset]
            assert max(axes[1]) - min(axes[1]) == pytest.approx(asset.length_m)
            # 1 mm: the fit runs through float32 mesh coordinates.
            assert min(axes[2]) == pytest.approx(-asset.draught_m, abs=1e-3), (
                "keel sits at the manifest draught, not on the surface"
            )
            assert max(axes[2]) > 0.0, "and the rest of it is above water"

    def test_the_noise_carries_the_slope_its_octaves_reach(self) -> None:
        """A ratio of octaves, so more wind means a longer dominant wave and less of it.

        Measured against the reference renders it runs about 12% under, which is the
        price of deriving it rather than fitting it.
        """
        calm, blowing = (
            scene.resolved_slope_fraction(2.0),
            scene.resolved_slope_fraction(18.0),
        )
        assert 0.0 < blowing < calm < 1.0
        # A dominant wave at the capillary scale would leave nothing unresolved.
        assert scene.resolved_slope_fraction(0.0) <= 1.0

    def test_the_sea_takes_its_wind_from_the_scenario(self) -> None:
        """Wind reaches the waves through wavelength and slope, or it is a dead knob.

        Both are derived, so this checks the shader carries what the derivation gives
        rather than restating the formulas.
        """
        tree = bpy.data.materials["sea"].node_tree
        wind = SCENARIO.sea.wind_speed_mps
        scaling = next(
            n
            for n in tree.nodes
            if n.bl_idname == "ShaderNodeVectorMath" and n.operation == "SCALE"
        )
        assert scaling.inputs["Scale"].default_value == pytest.approx(
            1.0 / scene.wave_length_m(wind)
        )
        bump = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeBump")
        assert bump.inputs["Distance"].default_value == pytest.approx(
            scene.bump_slope(wind) * scene.wave_length_m(wind)
        )

    def test_the_sea_edge_falls_inside_a_pixel(self) -> None:
        """A flat sea has no horizon of its own; it runs to a vanishing point.

        What makes that read as a horizon is the edge being finer than the sharpest
        camera can resolve. Checked as an angle, because that is the thing that has to
        be small -- a distance in metres says nothing without the optics.
        """
        corners = [
            bpy.data.objects["sea"].matrix_world @ Vector(c)
            for c in bpy.data.objects["sea"].bound_box
        ]
        reach = min(max(abs(v.x), abs(v.y)) for v in corners)
        edge_rad = math.atan(SCENARIO.rig.height_m / reach)
        sharpest = min(
            math.radians(c.hfov_deg) / c.width_px for c in SCENARIO.rig.cameras
        )
        assert edge_rad <= sharpest / 2
        assert reach > max(spec.range_m for spec in SCENARIO.objects)

    def test_the_sea_costs_no_geometry(self) -> None:
        """Waves are shading. Displacing them instead aliases past the range their own
        relief covers a pixel, which is most of the frame, and costs millions of
        vertices to do it."""
        mesh = (
            bpy.data.objects["sea"]
            .evaluated_get(bpy.context.evaluated_depsgraph_get())
            .to_mesh()
        )
        assert len(mesh.vertices) == 4

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

    def test_the_sky_carries_downwelling_radiance(self) -> None:
        """What renders is the Background node, not `World.color`.

        A new world already has `use_nodes` set, so assigning `World.color` changes
        nothing a camera sees. An earlier version of this test asserted that attribute
        and passed against a sky the renderer never read.
        """
        background = bpy.data.worlds["sky"].node_tree.nodes["Background"]
        assert background.inputs["Color"].is_linked

    def test_the_baked_sky_runs_cold_towards_the_zenith(self) -> None:
        """The shader reads this by sin(elevation), not by the angle itself."""
        image = bpy.data.images["sky_radiance"]
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        baked = pixels.reshape(-1, 4)[:, 0]

        ambient = lwir.band_radiance(SCENARIO.sky.t_air_k)
        assert baked[0] == pytest.approx(ambient, rel=1e-4), (
            "sin(elev)=0 is the horizon"
        )
        assert baked[-1] < 0.5 * ambient, "the zenith is much colder than ambient"
        assert np.all(np.diff(baked) <= 1e-6), "radiance falls towards the zenith"

    def test_the_sea_reflects_what_it_does_not_emit(self) -> None:
        """Emission alone falls to a fiftieth of ambient by 2 km: emissivity collapses
        at grazing incidence and nothing fills the gap.

        The factor is emissivity, so the mirror has to sit on the 0 input. That is the
        grazing end, where the sea stops emitting and starts reflecting.
        """
        mix = bpy.data.materials["sea"].node_tree.nodes["Mix Shader"]
        assert mix.inputs["Factor"].is_linked
        # ShaderNodeBsdfGlossy still reports its pre-4.0 bl_idname.
        assert mix.inputs[1].links[0].from_node.bl_idname == "ShaderNodeBsdfAnisotropic"
        assert mix.inputs[2].links[0].from_node.bl_idname == "ShaderNodeEmission"

    def test_the_reflection_lobe_matches_the_emissivity_curve(self) -> None:
        """Both come from the slope the bump does not carry, and must move together.

        Roughening one alone spreads the reflection into colder sky with nothing to pay
        it back: eps flat and the lobe rough put the sea at the horizon at 0.65 of
        ambient, against 0.985 in the reference.
        """
        mirror = next(
            n
            for n in bpy.data.materials["sea"].node_tree.nodes
            if n.bl_idname == "ShaderNodeBsdfAnisotropic"
        )
        assert mirror.inputs["Roughness"].default_value == pytest.approx(
            scene.specular_roughness(SCENARIO.sea.wind_speed_mps)
        )

    def test_emissivity_is_averaged_over_the_slopes_the_bump_misses(self) -> None:
        """Flat Fresnel reads 0.11 at 89 deg where a 7 m/s sea is nearer 0.63.

        That is the angle a target at 2 km sits at, so it sets the contrast the whole
        band is for.
        """
        image = bpy.data.images["sea_emissivity"]
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        baked = pixels.reshape(-1, 4)[:, 0]

        _, flat = lwir.emissivity_curve(t_sea_k=SCENARIO.sea.t_sea_k)
        assert baked[0] > 4 * flat[-1], (
            "grazing emissivity is lifted well clear of flat"
        )
        _, eps = lwir.emissivity_curve(
            t_sea_k=SCENARIO.sea.t_sea_k,
            slope_sigma=scene.unresolved_slope(SCENARIO.sea.wind_speed_mps),
        )
        assert baked[-1] == pytest.approx(eps[0], rel=1e-4), "nadir is unchanged"

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

        _, eps = lwir.emissivity_curve(
            t_sea_k=SCENARIO.sea.t_sea_k,
            slope_sigma=scene.unresolved_slope(SCENARIO.sea.wind_speed_mps),
        )
        assert baked[-1] == pytest.approx(eps[0], rel=1e-4), "cos(theta)=1 is normal"
        assert baked[0] == pytest.approx(eps[-1], abs=2e-3), "cos(theta)=0 is grazing"
        assert np.all(np.diff(baked) >= -1e-6), "emissivity rises towards normal"

    def test_radiance_is_not_sent_through_a_film_curve(self) -> None:
        """Pixels are W m^-2 sr^-1. Blender defaults to AgX, which is built to make
        photographs pleasant and would leave nothing measurable behind."""
        view = bpy.context.scene.view_settings
        assert (view.view_transform, view.look) == ("Standard", "None")
        assert (view.exposure, view.gamma) == (0.0, 1.0)

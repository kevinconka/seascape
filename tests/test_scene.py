"""The built scene, measured.

Blender is one global session, so each class rebuilds in its own band on entry.
"""

import math
from pathlib import Path

import bpy
import numpy as np
import pytest
from mathutils import Vector

from seascape import lwir, scene
from seascape.assets import Asset, manifest
from seascape.calibration import CameraCalibration
from seascape.config import Mount, load

BASELINE = Path(__file__).parent.parent / "scenarios" / "baseline.toml"
SCENARIO = load(BASELINE)
RIG_ONLY = f'extends = "{BASELINE}"\nobjects = []\n'


def camera_of(mount: Mount) -> bpy.types.Object:
    return bpy.data.objects[mount.name]


def _in_frame(camera: CameraCalibration, direction: Vector | np.ndarray) -> bool:
    """Whether a world direction from the camera lands on its sensor."""
    rotation = np.array(camera.extrinsics["world"])[:3, :3]
    x, y, z = np.array(camera.K) @ (rotation.T @ np.asarray(direction))
    # Pixel centres sit at integers, so the sensor spans -0.5 to size - 0.5.
    return bool(
        z > 0
        and -0.5 <= x / z <= camera.width_px - 0.5
        and -0.5 <= y / z <= camera.height_px - 0.5
    )


@pytest.mark.parametrize("name", ["baseline.toml", "twin-pod.toml"])
def test_the_sun_is_out_of_every_frame(name: str) -> None:
    scenario = load(BASELINE.parent / name)
    ownship = scenario.ownship.model_copy(update={"asset": None})
    bare = scenario.model_copy(
        update={"objects": [], "targets": None, "ownship": ownship}
    )
    built = scene.build(bare, "eo")
    sun = np.array(scene._sun_vector(scenario.sky))
    for mount in scenario.rig.mounts:
        camera = scene.calibrate(built, mount, "")
        assert not _in_frame(camera, sun), mount.name


def baked(name: str) -> np.ndarray:
    """The red channel of a 1-D lookup image, as the shader samples it."""
    image = bpy.data.images[name]
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(-1, 4)[:, 0]


def _blocked(origin: Vector, target: Vector) -> bool:
    ray = target - origin
    hit, *_ = bpy.context.scene.ray_cast(
        bpy.context.evaluated_depsgraph_get(),
        origin,
        ray.normalized(),
        distance=ray.length * 0.999,
    )
    return bool(hit)


def counts() -> tuple[int, ...]:
    return tuple(
        len(block) for block in (bpy.data.objects, bpy.data.materials, bpy.data.images)
    )


def test_a_named_substream_is_reproducible_and_local_to_its_name() -> None:
    def draw(seed: int, name: str) -> int:
        return int(scene._substream(seed, name).integers(2**31))

    assert draw(7, "sea/surface") == draw(7, "sea/surface")
    assert draw(7, "sea/surface") != draw(8, "sea/surface")
    assert draw(7, "sea/surface") != draw(7, "sky/haze")


def test_published_values_have_not_drifted() -> None:
    # Cox & Munk 1954: RMS slope of a clean sea at 7 m/s, off sun glitter photographs.
    assert scene.wave_slope(7.0) == pytest.approx(0.197, abs=5e-4)
    # Surveying's rule of thumb for the horizon, 3.86 sqrt(h_m) km at k = 0.13.
    rule_m = 3.86e3 * math.sqrt(51.8)
    assert scene.horizon_m(51.8, 0.13) == pytest.approx(rule_m, rel=0.01)
    # Pierson-Moskowitz: a fully developed sea at 7 m/s peaks near 41 m.
    assert scene.wave_length_m(7.0) == pytest.approx(40.8, abs=0.2)
    # Minimum phase speed of a surface wave, where surface tension takes over.
    assert abs(scene.CAPILLARY_WAVELENGTH_M - 0.0173) < 1e-4
    # Masuda 1988 at this wind speed: near nadir, and at 80 deg where roughness has
    # taken hold, lifting emissivity above flat Fresnel.
    theta, eps = lwir.emissivity_curve(
        t_sea_k=291.0, slope_sigma=scene.unresolved_slope(7.0)
    )
    assert float(np.interp(0.0, theta, eps)) == pytest.approx(0.985, abs=0.005)
    assert float(np.interp(math.radians(80.0), theta, eps)) == pytest.approx(
        0.76, abs=0.03
    )


class TestGeometry:
    """Everything the band does not change: where things are and where cameras look."""

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> scene.Built:
        return scene.build(SCENARIO, "eo")

    def test_a_flat_rig_points_where_the_scenario_asked(self) -> None:
        """The one negation. Two of them cancel and the whole rig mirrors unnoticed."""
        for mount in SCENARIO.rig.mounts:
            bearing, _ = scene.boresight_deg(camera_of(mount))
            assert bearing == pytest.approx(mount.nominal_bearing_deg, abs=1e-6)

    def test_cameras_carry_their_field_of_view_horizontally(self) -> None:
        """AUTO fits the angle to the longer image side, flipping a portrait sensor."""
        for mount in SCENARIO.rig.mounts:
            data = camera_of(mount).data
            assert data.sensor_fit == "HORIZONTAL"
            assert math.degrees(data.angle_x) == pytest.approx(mount.camera.hfov_deg)

    def test_the_near_clip_leaves_the_depth_buffer_usable_at_range(self) -> None:
        for mount in SCENARIO.rig.mounts:
            data = camera_of(mount).data
            assert data.clip_start == SCENARIO.rig.near_clip_m
            assert data.clip_start < data.clip_end

    def test_every_3d_view_clips_past_the_sea(self) -> None:
        corner_m = math.sqrt(2) * scene.sea_reach_m(SCENARIO.rig, SCENARIO.sea)
        views = [
            space
            for screen in bpy.data.screens
            for area in screen.areas
            if area.type == "VIEW_3D"
            for space in area.spaces
            if space.type == "VIEW_3D"
        ]

        assert views, "no 3D view to clip: the assertions below would pass on nothing"
        for space in views:
            assert space.clip_start == SCENARIO.rig.near_clip_m
            assert space.clip_end > corner_m

    def test_the_far_clip_clears_every_target(self) -> None:
        furthest = max(spec.range_m for spec in SCENARIO.objects)
        for mount in SCENARIO.rig.mounts:
            assert camera_of(mount).data.clip_end > furthest

    def test_a_target_lands_at_its_range_and_bearing(self) -> None:
        for spec in SCENARIO.objects:
            east, north, _ = bpy.data.objects[spec.asset].location
            assert math.hypot(east, north) == pytest.approx(spec.range_m)
            assert math.degrees(math.atan2(east, north)) == pytest.approx(
                spec.bearing_deg
            )

    def test_the_ship_is_in_every_frame(self, built: scene.Built) -> None:
        """Both bands' tests measure the one ship."""
        (spec,) = SCENARIO.objects
        ship = bpy.data.objects[spec.asset].matrix_world.translation
        for mount in SCENARIO.rig.mounts:
            eye = camera_of(mount).matrix_world.translation
            camera = scene.calibrate(built, mount, "")
            assert _in_frame(camera, ship - eye), mount.name

    def test_a_target_is_fitted_to_its_manifest_length(self) -> None:
        """The mesh arrives in its author's units; unfitted it is a speck at range."""
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
        """A ratio of octaves, so more wind means a longer wave and less of it."""
        calm, blowing = (
            scene.resolved_slope_fraction(2.0),
            scene.resolved_slope_fraction(18.0),
        )
        assert 0.0 < blowing < calm < 1.0

    def test_the_sea_takes_its_wind_from_the_scenario(self) -> None:
        """Wind reaches the waves through wavelength and slope, or it is a dead knob."""
        tree = bpy.data.materials["sea"].node_tree
        wind = SCENARIO.sea.wind_speed_mps
        scaling = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeVectorMath")
        assert tuple(scaling.inputs[1].default_value) == pytest.approx(
            (1.0 / scene.wave_length_m(wind), 1.0 / scene.wave_length_m(wind), 0.0)
        )
        bump = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeBump")
        assert bump.inputs["Distance"].default_value == pytest.approx(
            scene.bump_slope(wind)
            * scene.wave_length_m(wind)
            / scene.NOISE_SLOPE_PER_UNIT
        )
        noise = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeTexNoise")
        assert noise.inputs["Detail"].default_value == scene.NOISE_DETAIL, (
            "the transfer was measured at this Detail"
        )

    def test_the_sea_reaches_past_its_own_horizon(self) -> None:
        """The grid must contain the tangent point, or its edge becomes the horizon."""
        corners = [
            bpy.data.objects["sea"].matrix_world @ Vector(c)
            for c in bpy.data.objects["sea"].bound_box
        ]
        reach = min(max(abs(v.x), abs(v.y)) for v in corners)
        horizon = scene.horizon_m(SCENARIO.rig.height_m, SCENARIO.sea.refraction_k)

        assert reach > horizon
        assert reach > max(spec.range_m for spec in SCENARIO.objects)

    def test_a_hull_floats_on_the_sea_and_not_on_the_tangent_plane(self) -> None:
        """Hulls left at z = 0 fly with range and bearing still right, so nothing else
        catches it."""
        radius = scene.earth_radius_m(SCENARIO.sea.refraction_k)

        for spec in SCENARIO.objects:
            anchor = bpy.data.objects[spec.asset]
            east, north, up = anchor.matrix_world.translation

            assert up == pytest.approx(scene.sea_z_m(east, north, radius), abs=1e-3)

    def test_a_hull_beyond_the_horizon_is_cut_off(self) -> None:
        """A target past the horizon shows its waterline when it should be hull-down."""
        eye = Vector((0.0, 0.0, SCENARIO.rig.height_m))
        radius = scene.earth_radius_m(SCENARIO.sea.refraction_k)
        beyond = 18_000.0
        surface = beyond * beyond / (2.0 * radius)
        horizon = scene.horizon_m(SCENARIO.rig.height_m, SCENARIO.sea.refraction_k)
        assert horizon < beyond < scene.sea_reach_m(SCENARIO.rig, SCENARIO.sea)

        waterline = _blocked(eye, Vector((0.0, beyond, -surface)))
        mast = _blocked(eye, Vector((0.0, beyond, 30.0 - surface)))

        assert waterline, "the bulge has to hide the hull"
        assert not mast, "and leave what stands above it"

    def test_waves_cost_no_geometry(self) -> None:
        """A displaced sea would change its vertex count with the wind."""
        blowing = SCENARIO.model_copy(
            update={"sea": SCENARIO.sea.model_copy(update={"wind_speed_mps": 18.0})}
        )

        before = len(bpy.data.objects["sea"].data.vertices)
        scene.build(blowing, "eo")
        after = len(bpy.data.objects["sea"].data.vertices)
        scene.build(SCENARIO, "eo")  # the class shares one scene; put it back

        assert before == after == (scene.SEA_CELLS + 1) ** 2

    def test_building_twice_leaves_the_same_scene(self) -> None:
        """Node trees leak when a build appends to what is already there."""
        before = counts()
        scene.build(SCENARIO, "eo")
        assert counts() == before


@pytest.mark.parametrize(
    ("bow_deg", "bow_corner"),
    [(0.0, (0, 5, 0)), (90.0, (5, 0, 0)), (180.0, (0, -5, 0)), (270.0, (-5, 0, 0))],
)
def test_a_hull_is_fitted_along_its_own_bow_axis(bow_deg, bow_corner) -> None:
    """180 is its own inverse: the shipped hull passes with a sign error or the
    length measured along the beam. Any other bow catches both."""
    asset = Asset(
        url="x",
        sha256="0" * 64,
        length_m=200.0,
        draught_m=5.0,
        bow_deg=bow_deg,
        licence="x",
        attribution="x",
    )
    long, beam = Vector(bow_corner), Vector((-bow_corner[1], bow_corner[0], 0)) / 5
    corners = [
        s * long + b * beam + Vector((0, 0, z))
        for s in (-1, 1)
        for b in (-1, 1)
        for z in (0, 1)
    ]

    fit = scene._fit(corners, asset)
    fitted = [fit @ c for c in corners]

    ys = [c.y for c in fitted]
    assert max(ys) - min(ys) == pytest.approx(200.0), "scaled along the bow axis"
    assert max(c.x for c in fitted) - min(c.x for c in fitted) == pytest.approx(40.0)
    assert min(c.z for c in fitted) == pytest.approx(-5.0), "keel at the draught"
    assert (fit @ long).y == pytest.approx(100.0), "and the bow ends up at +Y"


@pytest.mark.parametrize("band", ["eo", "ir"])
def test_the_active_camera_belongs_to_the_band_built(band) -> None:
    """Opened on the rig's first camera, an IR build could render through EO optics
    against IR materials, with nothing to say so.
    """
    scene.build(SCENARIO.model_copy(update={"objects": []}), band)
    assert f"_{band}_" in bpy.context.scene.camera.name


@pytest.mark.parametrize("yaw_deg", [-40.0, 0.0, 40.0])
def test_a_pitched_pod_rolls_the_horizon_of_its_off_axis_cameras(
    tmp_path, yaw_deg
) -> None:
    """Horizon rolls by asin(sin(pitch) sin(yaw))."""
    pitch_deg = -5.0
    path = tmp_path / "pitched.toml"
    path.write_text(
        f"{RIG_ONLY}\n"
        f"[rig]\npitch_deg = {pitch_deg}\n\n"
        '[[rig.pods]]\nname = "bow"\nyaw_deg = 0.0\n\n'
        f'[[rig.pods.cameras]]\npreset = "eo_4k_49deg"\nyaw_deg = {yaw_deg}\n'
    )
    scenario = load(path)
    # Through `_yaw`: Blender's +Z turns to port, so the sign follows the scene's.
    expected = math.degrees(
        math.asin(math.sin(math.radians(pitch_deg)) * math.sin(scene._yaw(yaw_deg)))
    )

    scene.build(scenario, "eo")
    across = bpy.data.objects[
        scenario.rig.mounts[0].name
    ].matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))

    assert math.degrees(math.asin(across.normalized().z)) == pytest.approx(
        expected, abs=1e-6
    )


def _lens_pitched(
    tmp_path, yaw_deg: float, pitch_deg: float, rig_pitch_deg: float = 0.0
):
    path = tmp_path / "lens.toml"
    path.write_text(
        f"{RIG_ONLY}\n"
        f"[rig]\npitch_deg = {rig_pitch_deg}\n\n"
        '[[rig.pods]]\nname = "port"\nyaw_deg = -60.0\n\n'
        f'[[rig.pods.cameras]]\npreset = "eo_4k_49deg"\n'
        f"yaw_deg = {yaw_deg}\npitch_deg = {pitch_deg}\n"
    )
    scenario = load(path)
    scene.build(scenario, "eo")
    return scenario.rig.mounts[0]


@pytest.mark.parametrize("yaw_deg", [-40.0, 0.0, 40.0])
def test_a_pitched_lens_points_exactly_where_it_was_asked_to(tmp_path, yaw_deg) -> None:
    """Rx inside the camera's yaw: neither angle disturbs the other."""
    pitch_deg = -10.0

    mount = _lens_pitched(tmp_path, yaw_deg, pitch_deg)

    bearing, elevation = scene.boresight_deg(bpy.data.objects[mount.name])
    assert bearing == pytest.approx(mount.nominal_bearing_deg, abs=1e-4)
    assert elevation == pytest.approx(pitch_deg, abs=1e-4)


@pytest.mark.parametrize("yaw_deg", [-40.0, 40.0])
def test_a_pitched_pod_disturbs_a_pitched_lens(tmp_path, yaw_deg) -> None:
    """Rx(rig) still sits between the yaws: neither angle survives intact."""
    mount = _lens_pitched(tmp_path, yaw_deg, -10.0, rig_pitch_deg=-5.0)

    bearing, elevation = scene.boresight_deg(bpy.data.objects[mount.name])

    assert abs(bearing - mount.nominal_bearing_deg) > 0.1
    assert abs(elevation - (-10.0)) > 0.1


@pytest.mark.parametrize("yaw_deg", [-40.0, 0.0, 40.0])
def test_a_pitched_lens_keeps_its_horizon_level(tmp_path, yaw_deg) -> None:
    mount = _lens_pitched(tmp_path, yaw_deg, -10.0)

    across = bpy.data.objects[mount.name].matrix_world.to_3x3() @ Vector(
        (1.0, 0.0, 0.0)
    )

    assert across.normalized().z == pytest.approx(0.0, abs=1e-6)


def _pod_pitched(tmp_path, yaw_deg: float, pitch_deg: float = -5.0):
    """A one-pod rig yawed off the bow, so pitch sits between two non-zero yaws."""
    path = tmp_path / "pitched.toml"
    path.write_text(
        f"{RIG_ONLY}\n"
        f"[rig]\npitch_deg = {pitch_deg}\n\n"
        '[[rig.pods]]\nname = "port"\nyaw_deg = -60.0\n\n'
        f'[[rig.pods.cameras]]\npreset = "eo_4k_49deg"\nyaw_deg = {yaw_deg}\n'
    )
    scenario = load(path)
    scene.build(scenario, "eo")
    mount = scenario.rig.mounts[0]
    bearing, _ = scene.boresight_deg(bpy.data.objects[mount.name])
    return bearing, mount.nominal_bearing_deg


@pytest.mark.parametrize("yaw_deg", [-40.0, 40.0])
def test_pitch_moves_an_off_axis_camera_off_its_bearing(tmp_path, yaw_deg) -> None:
    bearing, nominal = _pod_pitched(tmp_path, yaw_deg)

    assert abs(bearing - nominal) > 0.1


def test_pitch_leaves_a_centre_camera_on_its_nominal_bearing(tmp_path) -> None:
    """Exact down the pod axis. Microdegrees, not zero: matrix_world is float32."""
    bearing, nominal = _pod_pitched(tmp_path, 0.0)

    assert bearing == pytest.approx(nominal, abs=1e-4)


def test_an_ownship_with_no_hull_still_carries_the_rig(tmp_path) -> None:
    path = tmp_path / "rolled.toml"
    path.write_text(f"{RIG_ONLY}\n[ownship]\nroll_deg = 5.0\n")
    scene.build(load(path), "eo")

    right = camera_of(SCENARIO.rig.mounts[0]).matrix_world.to_3x3() @ Vector(
        (1.0, 0.0, 0.0)
    )

    assert math.degrees(math.asin(-right.z)) == pytest.approx(5.0)


def test_a_near_clip_past_the_far_plane_is_an_error(tmp_path) -> None:
    path = tmp_path / "deep.toml"
    path.write_text(f'extends = "{BASELINE}"\n\n[rig]\nnear_clip_m = 500000.0\n')

    with pytest.raises(ValueError, match="near clip"):
        scene.build(load(path), "eo")


def test_a_band_the_rig_cannot_see_is_an_error(tmp_path) -> None:
    """Otherwise `next()` raises StopIteration, naming nothing."""
    path = tmp_path / "eo_only.toml"
    path.write_text(
        f'extends = "{BASELINE}"\n\n'
        '[[rig.pods]]\nname = "bow"\nyaw_deg = 0.0\n\n'
        '[[rig.pods.cameras]]\npreset = "eo_4k_49deg"\n'
    )
    with pytest.raises(ValueError, match="no ir camera"):
        scene.build(load(path), "ir")


class TestEoBand:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(SCENARIO, "eo")

    def test_the_sea_refracts_at_seawater_ior(self) -> None:
        bsdf = bpy.data.materials["sea"].node_tree.nodes["Principled BSDF"]
        assert bsdf.inputs["IOR"].default_value == pytest.approx(1.33)

    def test_the_sky_is_lit(self) -> None:
        assert bpy.data.worlds["sky"].node_tree.nodes["Sky Texture"]


class TestIrBand:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(SCENARIO, "ir")

    def test_the_sky_carries_downwelling_radiance(self) -> None:
        """What renders is the Background node, not `World.color`.

        A new world already has `use_nodes` set, so assigning `World.color` changes
        nothing a camera sees.
        """
        background = bpy.data.worlds["sky"].node_tree.nodes["Background"]
        assert background.inputs["Color"].is_linked

    def test_the_baked_sky_runs_cold_towards_the_zenith(self) -> None:
        """The shader reads this by sin(elevation), not by the angle itself."""
        curve = baked("sky_radiance")

        ambient = lwir.band_radiance(SCENARIO.sky.t_air_k)
        assert curve[0] == pytest.approx(ambient, rel=1e-4), (
            "sin(elev)=0 is the horizon"
        )
        assert curve[-1] < 0.5 * ambient, "the zenith is much colder than ambient"
        assert np.all(np.diff(curve) <= 1e-6), "radiance falls towards the zenith"

    def test_a_vessel_reflects_what_it_does_not_emit(self) -> None:
        """A pure emitter renders a hull one flat value whichever way it is turned."""
        skin = next(m for m in bpy.data.materials if m.name.endswith("_ir"))
        # The outermost mix, not the one grading emission from shaded to sunlit.
        output = next(
            n for n in skin.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"
        )
        mix = output.inputs["Surface"].links[0].from_node

        assert scene.PAINT_EMISSIVITY < 1.0, "a blackbody has no angular structure"
        assert mix.inputs["Factor"].default_value == pytest.approx(
            scene.PAINT_EMISSIVITY
        )
        assert mix.inputs[1].links[0].from_node.bl_idname == "ShaderNodeBsdfDiffuse"

    def test_a_vessel_is_hotter_on_the_side_the_sun_is_on(self) -> None:
        skin = next(m for m in bpy.data.materials if m.name.endswith("_ir"))
        output = next(
            n for n in skin.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"
        )
        grade = output.inputs["Surface"].links[0].from_node.inputs[2].links[0].from_node

        shaded, sunlit = (grade.inputs[i].links[0].from_node for i in (1, 2))
        assert (shaded.bl_idname, sunlit.bl_idname) == (
            "ShaderNodeEmission",
            "ShaderNodeEmission",
        )
        assert sunlit.inputs["Strength"].default_value == pytest.approx(
            lwir.band_radiance(SCENARIO.objects[0].t_k + SCENARIO.sky.solar_gain_k),
            rel=1e-5,
        )
        assert grade.inputs["Factor"].links[0].from_node.operation == "MAXIMUM"

    def test_the_sea_reflects_what_it_does_not_emit(self) -> None:
        """Emission alone goes dark toward grazing, where emissivity falls to zero."""
        mix = bpy.data.materials["sea"].node_tree.nodes["Mix Shader"]
        assert mix.inputs["Factor"].is_linked
        # ShaderNodeBsdfGlossy still reports its pre-4.0 bl_idname.
        assert mix.inputs[1].links[0].from_node.bl_idname == "ShaderNodeBsdfAnisotropic"
        assert mix.inputs[2].links[0].from_node.bl_idname == "ShaderNodeEmission"

    def test_the_reflection_lobe_matches_the_emissivity_curve(self) -> None:
        """Both come from the slope the bump does not carry, and must move together."""
        mirror = next(
            n
            for n in bpy.data.materials["sea"].node_tree.nodes
            if n.bl_idname == "ShaderNodeBsdfAnisotropic"
        )
        assert mirror.inputs["Roughness"].default_value == pytest.approx(
            scene.specular_roughness(SCENARIO.sea.wind_speed_mps)
        )

    def test_emissivity_is_averaged_over_the_slopes_the_bump_misses(self) -> None:
        """Flat Fresnel collapses toward grazing, which is where distant targets sit."""
        curve = baked("sea_emissivity")  # sampled over cos(theta), grazing first
        grazing = math.radians(89.0)
        mu = np.linspace(0.0, 1.0, len(curve))
        rough = float(np.interp(math.cos(grazing), mu, curve))

        theta, flat = lwir.emissivity_curve(t_sea_k=SCENARIO.sea.t_sea_k)
        assert rough > 4 * float(np.interp(grazing, theta, flat)), (
            "grazing emissivity is lifted well clear of flat"
        )

    def test_a_target_radiates_at_its_own_temperature(self) -> None:
        """`t_k` is in the scenario; a target rendering at its albedo ignores it."""
        for spec in SCENARIO.objects:
            hull = next(
                part
                for part in bpy.data.objects[spec.asset].children_recursive
                if part.type == "MESH"
            )
            emission = hull.material_slots[0].material.node_tree.nodes["Emission"]
            assert emission.inputs["Strength"].default_value == pytest.approx(
                lwir.band_radiance(spec.t_k), rel=1e-5
            )

    def test_the_baked_emissivity_matches_the_curve(self) -> None:
        """The shader reads this by cos(theta); the curve is sampled by theta."""
        curve = baked("sea_emissivity")

        _, eps = lwir.emissivity_curve(
            t_sea_k=SCENARIO.sea.t_sea_k,
            slope_sigma=scene.unresolved_slope(SCENARIO.sea.wind_speed_mps),
        )
        assert curve[-1] == pytest.approx(eps[0], rel=1e-4), "cos(theta)=1 is normal"
        assert curve[0] == pytest.approx(eps[-1], abs=2e-3), "cos(theta)=0 is grazing"
        assert np.all(np.diff(curve) >= -1e-6), "emissivity rises towards normal"

    def test_radiance_is_not_sent_through_a_film_curve(self) -> None:
        """Blender defaults to AgX."""
        view = bpy.context.scene.view_settings
        assert (view.view_transform, view.look) == ("Standard", "None")
        assert (view.exposure, view.gamma) == (0.0, 1.0)


class TestAnimate:
    @pytest.fixture
    def empty(self) -> bpy.types.Object:
        scene.build(load(BASELINE, ["outputs.duration_s = 0.3"]))
        obj = bpy.data.objects.new("probe", None)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    def test_every_frame_holds_the_value_at_its_time(self, empty) -> None:
        sc = bpy.context.scene
        scene._animate(empty, "location", [0.0, 0.1, 0.2], lambda t: 10 * t, index=0)
        assert (sc.frame_start, sc.frame_end, sc.render.fps, sc.render.fps_base) == (
            0,
            2,
            10,
            1.0,
        )
        for frame, x in enumerate([0.0, 1.0, 2.0]):
            sc.frame_set(frame)
            assert empty.matrix_world.translation.x == pytest.approx(x)

    def test_a_still_is_set_and_not_keyed(self, empty) -> None:
        scene._animate(empty, "location", [0.0], lambda t: (1.0, 2.0, 3.0))
        assert tuple(empty.location) == (1.0, 2.0, 3.0)
        assert empty.animation_data is None

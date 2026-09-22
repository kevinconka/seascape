"""The installed rig: pods on an ownship's bridge wings, traffic at 7 NM.

One global Blender session, so the scene is built once per module.
"""

import json
import math
from itertools import pairwise
from pathlib import Path

import bpy
import cv2
import numpy as np
import pytest
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

from seascape import panorama, scene
from seascape.calibration import Calibration
from seascape.config import Scenario, load

SCENARIO: Scenario = load(Path(__file__).parent.parent / "scenarios" / "twin-pod.toml")
MOUNTS = [pytest.param(mount, id=mount.name) for mount in SCENARIO.rig.mounts]
PODS = [pytest.param(pod, id=pod.name) for pod in SCENARIO.rig.pods]


@pytest.fixture(scope="module", autouse=True)
def built() -> None:
    scene.build(SCENARIO, "eo")


def targets() -> list[bpy.types.Object]:
    return [obj for obj in bpy.data.objects if obj.name.startswith("target_")]


@pytest.mark.parametrize("pod", PODS)
def test_a_pods_cameras_all_sit_at_its_mount_point(pod) -> None:
    """Cameras yawed on the centreline pass every bearing check with no baseline."""
    expected = (pod.offset_x_m, pod.offset_y_m, SCENARIO.rig.height_m)

    places = {
        tuple(_in_ship_frame(bpy.data.objects[mount.name]).translation)
        for mount in SCENARIO.rig.mounts
        if mount.pod.name == pod.name
    }

    assert len(places) == 1
    assert places.pop() == pytest.approx(expected)


def test_the_ownship_is_at_the_origin() -> None:
    """The rig's offsets are in the ownship's frame."""
    anchor = bpy.data.objects["ownship"]

    assert tuple(anchor.location) == pytest.approx((0.0, 0.0, 0.0))


def test_the_hull_takes_the_attitude_it_was_given() -> None:
    """Pins both signs and the order."""
    roll = math.radians(SCENARIO.ownship.roll_deg)
    pitch = math.radians(SCENARIO.ownship.pitch_deg)
    rotation = bpy.data.objects["ownship"].matrix_world.to_3x3()

    bow = rotation @ Vector((0.0, 1.0, 0.0))
    starboard = rotation @ Vector((1.0, 0.0, 0.0))

    assert bow.z == pytest.approx(math.sin(pitch))
    assert starboard.z == pytest.approx(-math.cos(pitch) * math.sin(roll))


def test_pod_span_and_overlap_measured_from_the_scene() -> None:
    """The acceptance numbers, read off the built cameras rather than the config."""
    arcs: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for mount in SCENARIO.rig.mounts:
        camera = bpy.data.objects[mount.name]
        half = math.degrees(camera.data.angle_x) / 2
        centre, _ = scene.boresight_deg(camera, bpy.data.objects["ownship"])
        # IR is one camera per pod; its span and overlap are a rig-level property.
        pod = "rig" if mount.camera.kind == "ir" else mount.pod.name
        arcs.setdefault((pod, mount.camera.kind), []).append(
            (centre - half, centre + half)
        )

    for key, span, overlap in (
        (("port", "eo"), 129.0, 9.0),
        (("starboard", "eo"), 129.0, 9.0),
        (("rig", "ir"), 44.0, 4.0),
    ):
        sectors = sorted(arcs[key])
        assert sectors[-1][1] - sectors[0][0] == pytest.approx(span), key
        gaps = [a[1] - b[0] for a, b in pairwise(sectors)]
        # Through the pod transform, matrix_world is a few microdegrees out.
        assert gaps == pytest.approx([overlap] * len(gaps), abs=1e-4), key


# A bracket stands on something. Further than this below a pod and it floats beside
# the ship.
MAX_BRACKET_M = 5.0


@pytest.mark.parametrize("pod", PODS)
def test_a_pod_stands_on_the_ship_rather_than_beside_it(pod) -> None:
    """A pod at the wrong height is outboard of the hull with nothing under it, and
    bearings, overlaps, lens clearance and target coverage all still pass."""
    pod_at = bpy.data.objects[f"pod_{pod.name}"].matrix_world.translation

    down = bpy.data.objects["ownship"].matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))

    drop_m = _distance_to_geometry(pod_at, pod_at + down)

    assert drop_m < MAX_BRACKET_M, (
        f"nothing within {MAX_BRACKET_M} m below pod_{pod.name}: it is not mounted"
    )


@pytest.mark.parametrize("mount", MOUNTS)
def test_no_camera_is_buried_in_the_structure_it_is_mounted_on(mount) -> None:
    """A lens flush with the plate it is bolted to fills the lower frame. Inside the
    near clip the plate is cut away instead, which reads as open sky."""
    camera = bpy.data.objects[mount.name]
    origin = camera.matrix_world.translation
    # The corners, not the axis: a deck the pod stands on is below the optical centre,
    # so a ray down the axis flies over it and the check passes on a buried camera.
    corners = [
        camera.matrix_world @ corner
        for corner in camera.data.view_frame(scene=bpy.context.scene)
    ]

    nearest_m = min(_distance_to_geometry(origin, corner) for corner in corners)

    assert nearest_m > SCENARIO.rig.near_clip_m


@pytest.mark.parametrize("mount", MOUNTS)
def test_every_camera_has_a_target_in_frame(mount) -> None:
    """Targets are world objects: coverage breaks silently if either geometry moves."""
    camera = bpy.data.objects[mount.name]

    framed = [
        target
        for target in targets()
        if _in_frame(world_to_camera_view(bpy.context.scene, camera, target.location))
    ]

    assert framed, f"{mount.name} sees none of {len(targets())} targets"


def test_every_target_sits_at_the_configured_range() -> None:
    assert SCENARIO.targets is not None

    ranges_m = [
        math.hypot(target.location.x, target.location.y) for target in targets()
    ]

    assert ranges_m == pytest.approx([SCENARIO.targets.range_m] * len(ranges_m))


def test_the_ring_is_one_mesh_however_many_targets() -> None:
    """Clones share datablocks; copied meshes show only when a build runs out of RAM."""
    assert SCENARIO.targets is not None
    hulls = [obj for obj in bpy.data.objects if obj.name.startswith("target_")]

    meshes = {
        part.data.name
        for hull in hulls
        for part in hull.children_recursive
        if part.type == "MESH"
    }
    per_target = len([p for p in hulls[0].children_recursive if p.type == "MESH"])

    assert len(hulls) == SCENARIO.targets.count
    assert len(meshes) == per_target


def test_the_calibration_projects_every_target_where_blender_draws_it(
    tmp_path, monkeypatch
) -> None:
    """Read back from disk, so a matrix that does not survive JSON fails here too."""
    written = Calibration(
        cameras=[scene.calibrate(m, f"{m.name}.png") for m in SCENARIO.rig.mounts]
    ).write(tmp_path)
    render = bpy.context.scene.render

    projected = 0
    for camera in Calibration.read(written.parent).cameras:
        w, h = camera.width_px, camera.height_px
        # world_to_camera_view takes its aspect from the scene's resolution.
        monkeypatch.setattr(render, "resolution_x", w)
        monkeypatch.setattr(render, "resolution_y", h)
        world_to_cam = np.linalg.inv(camera.extrinsics["world"])
        for target in targets():
            uv = world_to_camera_view(
                bpy.context.scene, bpy.data.objects[camera.name], target.location
            )
            if not _in_frame(uv):
                continue
            x, y, z, _ = world_to_cam @ (*target.location, 1.0)
            u, v, _ = np.array(camera.K) @ (x, y, z) / z
            # Blender's view runs 0-1 across pixel edges, bottom up.
            assert (u, v) == pytest.approx(
                (uv.x * w - 0.5, (1.0 - uv.y) * h - 0.5), abs=1e-3
            ), camera.name
            projected += 1

    assert projected, "no target in any frame: the assertions above ran on nothing"


def test_a_calibration_with_fields_it_does_not_know_still_reads(tmp_path) -> None:
    mount = SCENARIO.rig.mounts[0]
    record = scene.calibrate(mount, "").model_dump()
    (tmp_path / "calibration.json").write_text(
        json.dumps({"cameras": [{**record, "serial": "X"}], "rig": "Y"})
    )

    (camera,) = Calibration.read(tmp_path).cameras

    assert camera.name == mount.name


@pytest.mark.parametrize("mount", MOUNTS)
def test_in_its_pod_a_camera_points_exactly_as_asked(mount) -> None:
    axis = np.array(scene.calibrate(mount, "").extrinsics["pod"])[:3, 2]

    bearing = math.degrees(math.atan2(axis[0], axis[1]))
    elevation = math.degrees(math.asin(axis[2]))

    assert (bearing, elevation) == pytest.approx(
        (mount.camera.yaw_deg, mount.camera.pitch_deg), abs=1e-4
    )


@pytest.mark.parametrize("mount", MOUNTS)
def test_a_panorama_puts_each_principal_point_on_its_boresight(mount) -> None:
    """A slip in the stitcher's frame flips or mirrors the panorama."""
    k, r = panorama.pose(scene.calibrate(mount, ""), "world", 0.0)
    warper = cv2.PyRotationWarper("spherical", 1.0)

    # Spherical: u is the bearing, v the angle down from straight up.
    u, v = warper.warpPoint((float(k[0, 2]), float(k[1, 2])), k, r)

    assert (math.degrees(u), 90.0 - math.degrees(v)) == pytest.approx(
        scene.boresight_deg(bpy.data.objects[mount.name]), abs=1e-3
    )


def test_a_panorama_lays_its_cameras_out_in_yaw_order(tmp_path) -> None:
    """Each frame one colour, so the stitch shows which camera landed where."""
    first = SCENARIO.rig.mounts[0]
    mounts = [
        m
        for m in SCENARIO.rig.mounts
        if m.pod == first.pod and m.camera.kind == first.camera.kind
    ][:3]
    mounts.sort(key=lambda m: m.camera.yaw_deg)
    for channel, mount in enumerate(mounts):
        frame = np.zeros((mount.camera.height_px, mount.camera.width_px, 3), np.uint8)
        frame[..., channel] = 255
        cv2.imwrite(str(tmp_path / f"{mount.name}.png"), frame)
    Calibration(cameras=[scene.calibrate(m, f"{m.name}.png") for m in mounts]).write(
        tmp_path
    )

    paths = panorama.panoramas(tmp_path, "cylindrical", "pod", max_width=400)

    image = cv2.imread(str(paths[0]))
    assert image is not None
    row = image[image.shape[0] // 2]
    columns = [np.flatnonzero(row[:, c] > 127).mean() for c in range(len(mounts))]
    assert columns == sorted(columns)


def test_the_width_is_capped_and_never_raised(tmp_path) -> None:
    mount = SCENARIO.rig.mounts[0]
    frame = np.zeros((mount.camera.height_px, mount.camera.width_px, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "frame.png"), frame)
    camera = scene.calibrate(mount, "frame.png")

    native = panorama.stitch(tmp_path, [camera], "cylindrical", "pod")
    capped = panorama.stitch(tmp_path, [camera], "cylindrical", "pod", 101)
    uncapped = panorama.stitch(tmp_path, [camera], "cylindrical", "pod", 10**6)

    assert capped.shape[1] == 101
    assert uncapped.shape == native.shape


def test_rectilinear_refuses_a_camera_behind_its_plane() -> None:
    """Two cameras back to back: no plane faces both, whatever their field."""
    camera = scene.calibrate(SCENARIO.rig.mounts[0], "")
    turned = Matrix.Rotation(math.pi, 4, "Z") @ Matrix(camera.extrinsics["world"])
    behind = camera.model_copy(
        update={"extrinsics": {"world": tuple(map(tuple, turned))}}
    )

    with pytest.raises(ValueError, match="rectilinear cannot show"):
        panorama.stitch(Path(), [camera, behind], "rectilinear", "world")


def test_a_frame_the_calibration_lacks_is_named() -> None:
    camera = scene.calibrate(SCENARIO.rig.mounts[0], "")

    with pytest.raises(ValueError, match="'deck'"):
        panorama.stitch(Path(), [camera], "cylindrical", "deck")


def test_a_max_width_under_a_pixel_is_an_error() -> None:
    camera = scene.calibrate(SCENARIO.rig.mounts[0], "")

    with pytest.raises(ValueError, match="max width"):
        panorama.stitch(Path(), [camera], "cylindrical", "pod", 0)


def test_a_shrunk_frame_keeps_its_principal_point_at_its_centre() -> None:
    camera = scene.calibrate(SCENARIO.rig.mounts[0], "")
    frame = np.zeros((camera.height_px, camera.width_px, 3), np.uint8)
    k, _ = panorama.pose(camera, "pod", 0.0)

    small, k = panorama._shrink(frame, k, 0.25)

    h, w = small.shape[:2]
    assert (k[0, 2], k[1, 2]) == pytest.approx(((w - 1) / 2, (h - 1) / 2))
    assert k[0, 0] == pytest.approx(camera.K[0][0] * w / camera.width_px)


def _in_ship_frame(obj: bpy.types.Object) -> Matrix:
    """The rig is specified on the hull, so it is measured there."""
    return bpy.data.objects["ownship"].matrix_world.inverted() @ obj.matrix_world


def _distance_to_geometry(origin, through) -> float:
    """Metres to the first surface along a ray, or infinity if it reaches the sky."""
    hit, location, *_ = bpy.context.scene.ray_cast(
        bpy.context.evaluated_depsgraph_get(), origin, (through - origin).normalized()
    )
    return (location - origin).length if hit else math.inf


def _in_frame(uv) -> bool:
    """Behind the camera `world_to_camera_view` still lands x, y in range; z decides."""
    return 0.0 <= uv.x <= 1.0 and 0.0 <= uv.y <= 1.0 and uv.z > 0.0

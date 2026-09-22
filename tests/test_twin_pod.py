"""The installed rig: pods on an ownship's bridge wings, traffic at 7 NM.

One global Blender session, so the scene is built once per module.
"""

import math
from itertools import pairwise
from pathlib import Path

import bpy
import pytest
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

from seascape import scene
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
    assert SCENARIO.ownship is not None

    anchor = bpy.data.objects["ownship"]

    assert tuple(anchor.location) == pytest.approx((0.0, 0.0, 0.0))


def test_roll_takes_starboard_down_and_pitch_the_bow_up() -> None:
    assert SCENARIO.ownship is not None
    assert SCENARIO.ownship.roll_deg > 0.0
    assert SCENARIO.ownship.pitch_deg > 0.0
    ship = bpy.data.objects["ownship"].matrix_world
    port, starboard = (
        bpy.data.objects[f"pod_{side}"].matrix_world.translation.z
        for side in ("port", "starboard")
    )
    bow, stern = ((ship @ Vector((0.0, y, 0.0))).z for y in (1.0, -1.0))

    assert starboard < port
    assert bow > stern


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


def _in_ship_frame(obj: bpy.types.Object) -> Matrix:
    """The rig is specified on the hull, so it is measured there, not on the sea."""
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

"""The installed rig: pods on an ownship's bridge wings, traffic at 7 NM.

Blender is one global session, so the scene is built once for the module and every
test reads that one scene.
"""

import math
from pathlib import Path

import bpy
import pytest
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from seascape import scene
from seascape.config import Mount, Scenario, load

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
    """Four cameras, one enclosure. A pod that only yawed its cameras and left them on
    the centreline would pass every bearing check and still have no baseline."""
    expected = (pod.offset_x_m, pod.offset_y_m, SCENARIO.rig.height_m)

    places = {
        tuple(bpy.data.objects[Mount(pod, camera).name].matrix_world.translation)
        for camera in pod.cameras
    }

    assert len(places) == 1
    assert places.pop() == pytest.approx(expected)


def test_the_ownship_is_at_the_origin() -> None:
    """The rig's offsets are measured in the ownship's frame, so anywhere else and
    every camera is somewhere other than the deck it is bolted to."""
    assert SCENARIO.ownship is not None

    anchor = bpy.data.objects[SCENARIO.ownship.asset]

    assert tuple(anchor.location) == pytest.approx((0.0, 0.0, 0.0))


# A pod bolted flush to the deck it stands on sees that deck and nothing else. The
# bracket has to clear the structure by more than a rounding error, and less than a
# mast: anything in this range is a mount, anything closer is the mount itself.
# A bracket stands on something. Further than this below a pod and it is floating
# beside the ship rather than bolted to it.
MAX_BRACKET_M = 5.0


@pytest.mark.parametrize("pod", PODS)
def test_a_pod_stands_on_the_ship_rather_than_beside_it(pod) -> None:
    """The failure this exists for: `extends` carried baseline's deliberately low 12 m
    into this scenario, leaving both pods at hull level 40 m under their bridge wings
    and just outboard of the beam, so no ray hit anything. Bearings, overlaps, lens
    clearance and target coverage all still passed, and the thermal frames filled with
    the ship's own bow.
    """
    pod_at = bpy.data.objects[f"pod_{pod.name}"].matrix_world.translation

    drop_m = _distance_to_geometry(pod_at, pod_at + Vector((0.0, 0.0, -1.0)))

    assert drop_m < MAX_BRACKET_M, (
        f"nothing within {MAX_BRACKET_M} m below pod_{pod.name}: it is not mounted"
    )


MIN_CLEARANCE_M = 5.0


@pytest.mark.parametrize("mount", MOUNTS)
def test_no_camera_is_buried_in_the_structure_it_is_mounted_on(mount) -> None:
    """The failure this exists for: `height_m` set to the top face of the bridge wing
    put every lens 0.2 m from the plate, which then filled the lower frame. Bearings,
    overlaps and target coverage all still passed."""
    camera = bpy.data.objects[mount.name]
    origin = camera.matrix_world.translation
    # The corners, not the axis: a deck the pod stands on is below the optical centre,
    # so a ray down the axis flies over it and the check passes on a buried camera.
    corners = [
        camera.matrix_world @ corner
        for corner in camera.data.view_frame(scene=bpy.context.scene)
    ]

    nearest_m = min(_distance_to_geometry(origin, corner) for corner in corners)

    assert nearest_m > MIN_CLEARANCE_M


@pytest.mark.parametrize("mount", MOUNTS)
def test_every_camera_has_a_target_in_frame(mount) -> None:
    """The point of the ring, and the reason it is asserted rather than constructed:
    the targets are world objects on their own bearings, so coverage is a property of
    the two geometries and breaks silently if either moves."""
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
    """Clones share their datablocks, so the ring costs one import. Copying the meshes
    instead is invisible until a build runs out of memory."""
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


def _distance_to_geometry(origin, through) -> float:
    """Metres to the first surface along a ray, or infinity if it reaches the sky."""
    hit, location, *_ = bpy.context.scene.ray_cast(
        bpy.context.evaluated_depsgraph_get(), origin, (through - origin).normalized()
    )
    return (location - origin).length if hit else math.inf


def _in_frame(uv) -> bool:
    """`world_to_camera_view` returns normalised coordinates and a depth; behind the
    camera the coordinates still land in range, so the depth is what decides it."""
    return 0.0 <= uv.x <= 1.0 and 0.0 <= uv.y <= 1.0 and uv.z > 0.0

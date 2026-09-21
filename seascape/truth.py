"""Ground truth, derived from the scenario rather than measured off the render.

Nothing here imports bpy. The scenario states where everything is, so the truth is a
calculation, and keeping it independent of Blender is what lets it check the scene
instead of agreeing with it by construction. `tests/test_truth.py` holds the two in
step; if they drift, one of them is wrong and the test says which.

Frames, stated once because a rotation matrix with no frame is a decoration:

- World is east, north, up in metres, origin at the ownship's waterline.
- A camera looks down its own -Z with +X right and +Y up, which is Blender's and
  OpenGL's convention, not OpenCV's.
- `r_camera_from_world` maps a world direction into that camera frame.
"""

import math
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel, Field

from seascape.assets import manifest
from seascape.config import Band, Camera, Rig, Scenario


class Intrinsics(BaseModel):
    """A pinhole K. Square pixels, principal point at the centre, no distortion."""

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width_px: int
    height_px: int


class Pose(BaseModel):
    position_m: tuple[float, float, float]
    r_camera_from_world: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


class Sighting(BaseModel):
    """Where one target falls in one camera."""

    camera: str
    in_frame: bool
    centre_px: tuple[float, float]
    # The target's full length across the line of sight, which is its widest case.
    extent_px: float


class Target(BaseModel):
    asset: str
    range_m: float
    bearing_deg: float
    heading_deg: float
    t_k: float
    length_m: float
    sightings: list[Sighting]


class CameraTruth(BaseModel):
    name: str
    band: Band
    intrinsics: Intrinsics
    pose: Pose


class GroundTruth(BaseModel):
    # A timestamp, not a frame index: EO and IR run at different rates, so frame n is
    # not one instant across sensors.
    t_s: float
    rendered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    world_frame: str = "east-north-up metres, origin at the ownship waterline"
    camera_frame: str = "looks down -Z, +X right, +Y up"
    cameras: list[CameraTruth]
    targets: list[Target]
    scenario: Scenario


def intrinsics(spec: Camera) -> Intrinsics:
    """K from the field of view and the pixel count, which is all a pinhole needs."""
    focal = spec.width_px / (2.0 * math.tan(math.radians(spec.hfov_deg) / 2.0))
    return Intrinsics(
        fx_px=focal,
        fy_px=focal,
        cx_px=spec.width_px / 2.0,
        cy_px=spec.height_px / 2.0,
        width_px=spec.width_px,
        height_px=spec.height_px,
    )


def _rotation(rig: Rig, spec: Camera) -> np.ndarray:
    """World-from-camera, matching the euler `scene` gives the camera object.

    Blender composes an XYZ euler as Rz @ Ry @ Rx, and the scene sets
    (90 + tilt, 0, -bearing): pitch up from looking down, then yaw.
    """
    pitch = math.radians(90.0 + rig.tilt_deg)
    # The bearing is negated here for the same reason as in `scene`: Blender's +Z
    # rotation turns a forward-facing object to port.
    yaw = -math.radians(spec.bearing_deg)
    about_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ]
    )
    about_z = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return about_z @ about_x


def _sighting(
    name: str,
    k: Intrinsics,
    r_camera_from_world: np.ndarray,
    offset_m: np.ndarray,
    length_m: float,
    range_m: float,
) -> Sighting:
    x, y, z = r_camera_from_world @ offset_m
    # The camera looks down -Z, so anything it can see has a negative z.
    depth = -z
    if depth <= 0.0:
        return Sighting(
            camera=name, in_frame=False, centre_px=(0.0, 0.0), extent_px=0.0
        )
    # Pixel rows run down while camera +Y is up, so v is flipped.
    u = k.cx_px + k.fx_px * x / depth
    v = k.cy_px - k.fy_px * y / depth
    return Sighting(
        camera=name,
        in_frame=0.0 <= u < k.width_px and 0.0 <= v < k.height_px,
        centre_px=(u, v),
        # Small angle: exact to 0.1% out to 5 deg, which is 200 m of hull at 2 km.
        extent_px=k.fx_px * length_m / range_m,
    )


def ground_truth(scenario: Scenario, t_s: float = 0.0) -> GroundTruth:
    """What the scene contains at `t_s`, computed rather than measured."""
    rig = scenario.rig
    eye = np.array([0.0, 0.0, rig.height_m])
    lengths = manifest()

    cameras, rotations = [], {}
    # Only the bands that were rendered: a truth file naming a camera with no image
    # sends its reader looking for a file that was never written.
    for spec in (c for c in rig.cameras if c.kind in scenario.outputs.bands):
        name = f"{spec.pod}_{spec.kind}_{spec.bearing_deg:+g}"
        rotations[name] = _rotation(rig, spec).T
        cameras.append(
            CameraTruth(
                name=name,
                band=spec.kind,
                intrinsics=intrinsics(spec),
                pose=Pose(
                    position_m=(0.0, 0.0, rig.height_m),
                    r_camera_from_world=tuple(map(tuple, rotations[name])),
                ),
            )
        )

    targets = []
    for spec in scenario.objects:
        bearing = math.radians(spec.bearing_deg)
        at = np.array(
            [spec.range_m * math.sin(bearing), spec.range_m * math.cos(bearing), 0.0]
        )
        length_m = lengths[spec.asset].length_m
        targets.append(
            Target(
                asset=spec.asset,
                range_m=spec.range_m,
                bearing_deg=spec.bearing_deg,
                heading_deg=spec.heading_deg,
                t_k=spec.t_k,
                length_m=length_m,
                sightings=[
                    _sighting(
                        camera.name,
                        camera.intrinsics,
                        rotations[camera.name],
                        at - eye,
                        length_m,
                        spec.range_m,
                    )
                    for camera in cameras
                ],
            )
        )

    return GroundTruth(t_s=t_s, cameras=cameras, targets=targets, scenario=scenario)

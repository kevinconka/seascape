"""Stitch each pod's cameras into one panorama, from a render's calibration.json.

The poses are known, so this is OpenCV's stitching pipeline with its estimation
stages skipped: `cv2.PyRotationWarper` projects each camera and
`cv2.detail.MultiBandBlender` joins the overlaps. No Blender.

Every panorama is built about its pod's axis, so x = 0 is the pod's bearing. The
frame decides what is level: `world` levels the horizon, `vessel` the deck, `pod`
the enclosure.
"""

from pathlib import Path

import cv2
import numpy as np

from seascape.calibration import Calibration, CameraCalibration, Frame

# OpenCV's names for them.
PROJECTIONS = {
    "rectilinear": "plane",
    "cylindrical": "cylindrical",
    "equirectangular": "spherical",
}

# The stitcher's frame is +X right, +Y down, +Z ahead.
_TO_CV = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def rotation(camera: CameraCalibration, frame: Frame) -> np.ndarray:
    return np.array(getattr(camera, f"T_{frame}_cam"))[:3, :3]


def axis(cameras: list[CameraCalibration], frame: Frame) -> float:
    """Bearing of the summed optical axes, which cannot wrap as a mean of angles can."""
    x, y, _ = np.sum([rotation(camera, frame)[:, 2] for camera in cameras], axis=0)
    return float(np.arctan2(x, y))


def pose(
    camera: CameraCalibration, frame: Frame, bearing: float
) -> tuple[np.ndarray, np.ndarray]:
    """K and R as the stitcher takes them, turned so `bearing` is straight ahead."""
    c, s = np.cos(bearing), np.sin(bearing)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (
        np.array(camera.K, np.float32),
        np.array(_TO_CV @ turn @ rotation(camera, frame), np.float32),
    )


def stitch(
    folder: Path,
    cameras: list[CameraCalibration],
    projection: str,
    frame: Frame,
    width: int,
) -> np.ndarray:
    """A width of 0 is native: fx is pixels per radian on axis, where every
    projection here runs at one unit per radian."""
    kind = PROJECTIONS[projection]
    bearing = axis(cameras, frame)
    poses = [pose(camera, frame, bearing) for camera in cameras]
    sizes = [(camera.width_px, camera.height_px) for camera in cameras]

    scale = max(float(k[0, 0]) for k, _ in poses)
    if width:
        native = cv2.PyRotationWarper(kind, scale)
        rois = [
            native.warpRoi(size, k, r)
            for size, (k, r) in zip(sizes, poses, strict=True)
        ]
        _, _, span, _ = cv2.detail.resultRoi(
            corners=[roi[:2] for roi in rois], sizes=[roi[2:] for roi in rois]
        )
        scale *= width / span
    warper = cv2.PyRotationWarper(kind, scale)

    warped = []
    for camera, (w, h), (k, r) in zip(cameras, sizes, poses, strict=True):
        image = cv2.imread(str(folder / camera.image), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read {camera.image} for {camera.name}")
        corner, pixels = warper.warp(image, k, r, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = np.full((h, w), 255, np.uint8)
        _, mask = warper.warp(mask, k, r, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        warped.append((corner, pixels, mask))

    blender = cv2.detail.MultiBandBlender()
    blender.prepare(
        cv2.detail.resultRoi(
            corners=[c for c, _, _ in warped],
            sizes=[m.shape[1::-1] for _, _, m in warped],
        )
    )
    for corner, pixels, mask in warped:
        blender.feed(pixels.astype(np.int16), mask, corner)
    image, _ = blender.blend(np.empty(0, np.int16), np.empty(0, np.uint8))
    return cv2.convertScaleAbs(image)


def panoramas(folder: Path, projection: str, frame: Frame, width: int) -> list[Path]:
    """One panorama per pod and band, written beside the frames."""
    groups: dict[tuple[str, str], list[CameraCalibration]] = {}
    for camera in Calibration.read(folder).cameras:
        groups.setdefault((camera.pod, camera.band), []).append(camera)

    written = []
    for (pod, band), cameras in groups.items():
        path = folder / f"panorama_{pod}_{band}_{projection}_{frame}.png"
        cv2.imwrite(str(path), stitch(folder, cameras, projection, frame, width))
        written.append(path)
    return written

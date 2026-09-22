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
    if width < 0:
        raise ValueError(f"width is {width}: 0 for native, or pixels")
    kind = PROJECTIONS[projection]
    bearing = axis(cameras, frame)
    poses = [pose(camera, frame, bearing) for camera in cameras]
    sizes = [(camera.width_px, camera.height_px) for camera in cameras]
    if kind == "plane" and not all(
        _faces_ahead(k, r, size) for (k, r), size in zip(poses, sizes, strict=True)
    ):
        raise ValueError(
            "rectilinear cannot show a ray 90 deg or more off its axis: "
            "use cylindrical or equirectangular"
        )

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
    for camera, (k, r) in zip(cameras, poses, strict=True):
        path = folder / camera.image
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                f"cannot read {path}: panorama takes the 8-bit frames that "
                'outputs.format = "png" writes'
            )
        image, k = _shrink(image, k, scale / float(k[0, 0]))
        corner, pixels = warper.warp(image, k, r, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = np.full(image.shape[:2], 255, np.uint8)
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
    # The pyramid overshoots at edges; clip, where convertScaleAbs would fold it back.
    return np.clip(image, 0, 255).astype(np.uint8)


def _faces_ahead(k: np.ndarray, r: np.ndarray, size: tuple[int, int]) -> bool:
    """Whether a whole frame lies less than 90 deg off the stitcher's +Z. Depth is
    linear across the frame, so the corners decide."""
    w, h = size
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]])
    _, _, z = r @ np.linalg.inv(k) @ corners.T
    return bool((z > 0).all())


def _shrink(
    image: np.ndarray, k: np.ndarray, factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """Area-average a frame down before warping: the warper samples it bilinearly,
    which aliases anything finer than the output grid."""
    if factor >= 1.0:
        return image, k
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_AREA
    )
    sx, sy = small.shape[1] / w, small.shape[0] / h
    # Pixel centres sit at integers, so the principal point scales about -0.5.
    k = np.array(
        [
            [k[0, 0] * sx, 0.0, (k[0, 2] + 0.5) * sx - 0.5],
            [0.0, k[1, 1] * sy, (k[1, 2] + 0.5) * sy - 0.5],
            [0.0, 0.0, 1.0],
        ],
        np.float32,
    )
    return small, k


def panoramas(folder: Path, projection: str, frame: Frame, width: int) -> list[Path]:
    """One panorama per pod and band, written beside the frames."""
    groups: dict[tuple[str, str], list[CameraCalibration]] = {}
    for camera in Calibration.read(folder).cameras:
        groups.setdefault((camera.pod, camera.band), []).append(camera)

    written = []
    for (pod, band), cameras in groups.items():
        path = folder / f"panorama_{pod}_{band}_{projection}_{frame}.png"
        try:
            image = stitch(folder, cameras, projection, frame, width)
        except cv2.error as error:
            raise RuntimeError(f"{path.name}: {error.err}") from error
        cv2.imwrite(str(path), image)
        written.append(path)
    return written

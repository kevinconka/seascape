"""The geometry a render was made with, written beside its frames.

Imports no Blender, so a consumer reads it without the bpy wheel.
"""

from pathlib import Path

from pydantic import Field

from seascape.config import Band, Model

FILENAME = "calibration.json"

type Row3 = tuple[float, float, float]
type Row4 = tuple[float, float, float, float]
type Matrix3 = tuple[Row3, Row3, Row3]
type Matrix4 = tuple[Row4, Row4, Row4, Row4]


class CameraCalibration(Model):
    """An ideal pinhole with no distortion, in OpenCV's conventions.

    Camera frame: +X right, +Y down, +Z along the optical axis. Pixel centres sit at
    integer coordinates. Each `T_<frame>_cam` takes camera coordinates into `<frame>`.
    """

    name: str
    band: Band
    image: str = Field(description="relative to the calibration file")
    pod: str
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    K: Matrix3
    T_world_cam: Matrix4 = Field(
        description="+X east, +Y north, +Z up, at sea level under the vessel's origin"
    )
    T_vessel_cam: Matrix4 = Field(
        description="+X starboard, +Y bow, +Z up, moving with the hull"
    )
    T_pod_cam: Matrix4 = Field(description="the enclosure, +Y along its axis")


class Calibration(Model):
    cameras: list[CameraCalibration] = Field(min_length=1)

    def write(self, folder: Path) -> Path:
        path = folder / FILENAME
        path.write_text(self.model_dump_json(indent=2) + "\n")
        return path

    @classmethod
    def read(cls, folder: Path) -> "Calibration":
        return cls.model_validate_json((folder / FILENAME).read_text())
